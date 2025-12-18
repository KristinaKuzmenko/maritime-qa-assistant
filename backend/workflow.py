"""
Agentic LangGraph workflow with Neo4j as a tool.

Architecture (OPTIMIZED):
1. Analyze & Route (MERGED) - Single LLM call: detect intent + select tools
2. Context Builder - merge results, expand with neighbor chunks
3. LLM Reasoning - generate answer

⚡ OPTIMIZATION #1: Merged LLM calls (50% latency reduction)
   - Before: 2 sequential LLM calls (analyze_question → router_agent)
   - After: 1 unified LLM call (analyze_and_route)
   - Result: ~50% reduction in initial latency, fewer API calls

⚡ OPTIMIZATION #2: Neo4j UNION query (60-70% reduction in DB round-trips)
   - Before: 3-4 sequential Neo4j queries for entity discovery
   - After: 1 UNION query (exact codes + equipment codes + phrases)
   - Result: Faster entity search, reduced DB latency

⚡ OPTIMIZATION #3: Embedding cache (eliminates redundant API calls)
   - Before: Each Qdrant tool creates new embedding (3× API calls)
   - After: Cache embedding, reuse across all Qdrant searches
   - Result: Saves 300-600ms per query (2 API calls eliminated)

⚡ OPTIMIZATION #4: Neo4j precomputed BM25 index (replaces in-memory BM25)
   - Before: Build BM25Okapi index from scratch on every entity search
   - After: Use neo4j_fulltext_search with section_ids filter (precomputed BM25)
   - Result: Eliminates corpus tokenization, faster re-ranking, no duplicate code

⚡ OPTIMIZATION #5: Entity preload at startup (eliminates first-query blocking)
   - Before: Sync entity load on first query (200-500ms blocking)
   - After: Async preload during app startup in lifespan event
   - Result: First user query is as fast as subsequent queries

⚡ OPTIMIZATION #6: Few-shot prompting (reduces regeneration loop frequency)
   - Before: Intent mismatch → regeneration LLM call (+1-2 seconds)
   - After: Few-shot examples in system prompt teach correct format upfront
   - Result: Fewer regenerations, faster responses, better first-attempt accuracy

Key features:
- Neo4j is a tool, not hidden pipeline step
- Agent decides when to use graph vs vector search
- Automatic neighbor chunk retrieval for complete context
- Single-pass intent classification + tool routing
- Smart embedding caching across tool calls
"""

from typing import TypedDict, List, Any, Dict, Optional, Literal, Annotated
from langgraph.graph import StateGraph, END
import logging
import operator
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# Optional: Groq provider (only needed if LLM_PROVIDER=groq)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    ChatGroq = None
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from neo4j import Driver

from core.config import settings
from services.entity_extractor import get_entity_extractor

logger = logging.getLogger(__name__)



# HELPER FUNCTIONS

def get_tool_calls(message: AIMessage) -> List[Dict]:
    """
    Safely extract tool_calls from AIMessage.
    """
    if hasattr(message, 'tool_calls'):
        return message.tool_calls if message.tool_calls else []
    return []

def has_tool_calls(message: AIMessage) -> bool:
    """Check if message has any tool calls."""
    tool_calls = get_tool_calls(message)
    return len(tool_calls) > 0


# GRAPH SCHEMA PROMPT

GRAPH_SCHEMA_PROMPT = """
You work with a Neo4j graph that describes maritime technical documentation.

GRAPH SCHEMA:
- (d:Document {id, title})
- (c:Chapter {id, title, number})
- (s:Section {id, title, doc_id, page_start, page_end, content})
- (t:Table {id, doc_id, page_number, title, caption, rows, cols, file_path})
- (sc:Schema {id, doc_id, page_number, title, caption, file_path})

RELATIONSHIPS:
- (d)-[:HAS_CHAPTER]->(c)
- (c)-[:HAS_SECTION]->(s)
- (s)-[:CONTAINS_TABLE]->(t)
- (s)-[:CONTAINS_SCHEMA]->(sc)

RETURN FIELD NAMING (IMPORTANT!):
When returning results, use these EXACT aliases:
- Table: t.id AS table_id, t.title AS title, t.page_number AS page, t.file_path AS file_path, t.doc_id AS doc_id
- Schema: sc.id AS schema_id, sc.title AS title, sc.page_number AS page, sc.file_path AS file_path
- Section: s.id AS section_id, s.title AS title, s.page_start AS page_start, s.page_end AS page_end

RULES:
- ONLY read queries (MATCH, WHERE, RETURN, ORDER BY, LIMIT)
- ALWAYS use LIMIT <= 5
- NO write operations (CREATE, MERGE, DELETE, SET)
"""


# STATE DEFINITION

class GraphState(TypedDict):
    """Agentic workflow state"""
    user_id: str
    question: str
    chat_history: List[Dict[str, str]]
    
    # Filters
    owner: Optional[str]
    doc_ids: Optional[List[str]]
    
    # Query analysis
    query_intent: str  # "text" | "table" | "schema" | "mixed"
    
    # Agent loop
    messages: Annotated[List, operator.add]  # Agent messages (with tool calls)
    
    # Anchor sections (selected after Qdrant text search)
    anchor_sections: List[Dict[str, Any]]
    
    # Search results from all sources (Qdrant semantic + Neo4j entity/fulltext)
    search_results: Dict[str, List[Dict[str, Any]]]  # {text: [...], tables: [...], schemas: [...]}
    neo4j_results: List[Dict[str, Any]]  # Direct Cypher query results
    
    # Entity search results (separate to avoid context pollution)
    entity_results: Optional[Dict[str, Any]]  # {entities: [...], sections: [...], tables: [...], schemas: [...]}
    
    # Enriched context (after merging + neighbor expansion)
    enriched_context: List[Dict[str, Any]]
    
    # Adaptive retry tracking
    retrieval_attempt: int  # 0 = first attempt, 1 = retried once (max 1 retry)
    
    # Answer
    answer: Dict[str, Any]


# ============================================================================
# TOOLS
# ============================================================================

class ToolContext:
    """Shared context for tools"""
    qdrant_client: Optional[QdrantClient] = None
    neo4j_driver: Optional[Driver] = None
    neo4j_uri: Optional[str] = None  # Neo4j connection URI
    neo4j_auth: Optional[tuple] = None  # Neo4j auth tuple (user, password)
    vector_service: Optional[Any] = None  # VectorService instance
    owner: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    known_entities: List[str] = []  # Cache of entity names from Neo4j
    entities_loaded: bool = False  # Flag to track if entities have been loaded
    
    # ⚡ OPTIMIZATION: Cache embedding for current query to avoid multiple API calls
    _current_query: Optional[str] = None
    _query_embedding: Optional[List[float]] = None
    
    def clear_embedding_cache(self):
        """Clear cached embedding (call at start of new query)"""
        self._current_query = None
        self._query_embedding = None
    
    def get_cached_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding if query matches"""
        if self._current_query == query and self._query_embedding is not None:
            logger.debug(f"⚡ Using cached embedding for query: {query[:50]}...")
            return self._query_embedding
        return None
    
    def cache_embedding(self, query: str, embedding: List[float]):
        """Cache embedding for query"""
        self._current_query = query
        self._query_embedding = embedding
        logger.debug(f"💾 Cached embedding for query: {query[:50]}...")


def get_llm_instance(temperature: float = 0):
    """
    Get LLM instance based on configured provider.
    Supports OpenAI, Groq, and Cerebras.
    
    Anti-hallucination parameters:
    - temperature=0: Deterministic, focused responses
    - max_tokens: 4096 for complete troubleshooting procedures and full table extraction
    """
    if settings.llm_provider == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError(
                "langchain-groq is not installed. "
                "Install it with: pip install langchain-groq>=1.1.1"
            )
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when llm_provider=groq")
        return ChatGroq(
            model=settings.llm_model,
            temperature=temperature,
            api_key=settings.groq_api_key,
            max_tokens=4096,  # Sufficient for complete troubleshooting procedures
        )
    elif settings.llm_provider == "cerebras":
        if not settings.cerebras_api_key:
            raise ValueError("CEREBRAS_API_KEY is required when llm_provider=cerebras")
        # Cerebras uses OpenAI-compatible API
        return ChatOpenAI(
            model=settings.llm_model,  # e.g., "gpt-oss-120b"
            temperature=temperature,
            api_key=settings.cerebras_api_key,
            base_url=settings.cerebras_base_url,
            max_tokens=4096,  # Sufficient for complete multi-step procedures
        )
    else:  # default to openai
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temperature,
            api_key=settings.openai_api_key,
            max_tokens=4096,  # Sufficient for detailed troubleshooting and full table data
        )


# Create global context (will be set by graph builder)
tool_ctx = ToolContext()


# =============================================================================
# HELPER: Load Known Entities from Neo4j (for entity hints)
# =============================================================================

async def load_known_entities() -> List[str]:
    """
    Load all entity names AND codes from Neo4j graph for entity detection hints.
    Called once at workflow initialization.
    Returns list of entity identifiers (names and codes).
    """
    if not tool_ctx.neo4j_driver:
        logger.warning("⚠️ tool_ctx.neo4j_driver is None - cannot load entities")
        return []
    
    try:
        logger.info(f"🔍 Querying Neo4j for entities (driver: {tool_ctx.neo4j_driver})")
        async with tool_ctx.neo4j_driver.session() as session:
            query = """
            MATCH (e:Entity)
            RETURN DISTINCT 
                e.name AS entity_name,
                e.code AS entity_code
            """
            result = await session.run(query)
            records = [rec async for rec in result]
            
            logger.info(f"📊 Retrieved {len(records)} entity records from Neo4j")
            
            entities = set()
            
            for rec in records:
                name = rec.get("entity_name")
                code = rec.get("entity_code")
                
                # Add name if it exists and is not too generic
                if name and len(name) > 2:
                    # Normalize: keep original case but store lowercase for matching
                    entities.add(name.lower())
                
                # Add code if it exists (codes are usually specific)
                # Keep codes in original case (they may be case-sensitive)
                if code and code.strip():
                    entities.add(code)
            
            entities_list = sorted(list(entities))
            logger.info(f"✅ Loaded {len(entities_list)} known entities (names + codes) from Neo4j")
            return entities_list
            
    except Exception as e:
        logger.error(f"❌ Failed to load known entities: {e}", exc_info=True)
        return []


def ensure_entities_loaded():
    """
    ⚡ OPTIMIZED: Check if entities are preloaded (should be done at startup).
    No longer blocks - entities are loaded asynchronously during app initialization.
    
    This function is now just a safeguard check.
    """
    logger.debug(f"🔍 ensure_entities_loaded: entities_loaded={tool_ctx.entities_loaded}, entities count={len(tool_ctx.known_entities)}")
    if not tool_ctx.entities_loaded:
        logger.warning("⚠️ Entities not preloaded at startup! Using empty list as fallback.")
        logger.info("💡 Entities should be preloaded in lifespan startup event")
        tool_ctx.known_entities = []
        tool_ctx.entities_loaded = True
    else:
        logger.debug(f"✅ Entities already loaded: {len(tool_ctx.known_entities)} entities")


def find_entities_in_question(question: str, known_entities: List[str]) -> List[str]:
    """
    Find entity mentions in the question by:
    1. Exact match with known entities from graph
    2. Equipment code patterns (e.g., PU3, SV4, HGM-30, PT-6018)
    
    Returns list of matched entity names/codes, filtering out generic single-word terms.
    """
    q_lower = question.lower()
    found = []
    found_lower = set()  # To avoid duplicates
    
    # Generic terms to ignore (too broad)
    generic_terms = {
        'valve', 'pump', 'filter', 'cooler', 'pipe', 'tank', 'sensor',
        'motor', 'fan', 'alarm', 'switch', 'gauge', 'system', 'unit',
        'cylinder', 'piston', 'bearing', 'seal', 'gasket', 'bolt',
        'engine', 'turbocharger', 'compressor', 'generator', 'boiler'
    }
    
    # 1. Match against known entities from graph
    for entity in known_entities:
        entity_lower = entity.lower()
        
        # Skip too short entities (< 3 chars)
        if len(entity_lower) < 3:
            continue
        
        # Skip generic single-word terms
        if entity_lower in generic_terms:
            continue
        
        # Skip if already found
        if entity_lower in found_lower:
            continue
        
        # Check for exact match as whole word or phrase
        if entity_lower in q_lower:
            # For multi-word entities, always include
            if ' ' in entity or '-' in entity:
                found.append(entity)
                found_lower.add(entity_lower)
            # For single-word entities, only include if they look like codes (uppercase, numbers, etc.)
            elif entity.isupper() or any(c.isdigit() for c in entity) or len(entity) > 8:
                found.append(entity)
                found_lower.add(entity_lower)
    
    # 2. Detect equipment code patterns (even if not in graph)
    # Patterns: PU3, SV4, HGM-30, PT-6018, CR-302, INC-8130, etc.
    import re
    code_pattern = r'\b([A-Z]{1,4}[-]?[0-9]{1,5})\b'
    matches = re.findall(code_pattern, question)
    
    for code in matches:
        code_lower = code.lower()
        if code_lower not in found_lower:
            found.append(code)
            found_lower.add(code_lower)
            logger.debug(f"Detected equipment code pattern: {code}")
    
    # Sort by length descending (longer matches are more specific)
    found.sort(key=len, reverse=True)
    
    # Limit to top 5 to avoid overwhelming the prompt
    return found[:5]


# =============================================================================
# HELPER: Neo4j Fulltext Search (used by multiple tools)
# =============================================================================

async def neo4j_fulltext_search(
    search_term: str, 
    limit: int = 10, 
    min_score: float = 0.5,
    include_content: bool = False,
    section_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    ⚡ OPTIMIZED: Search/re-rank sections using Neo4j's precomputed fulltext index (BM25).
    Replaces in-memory BM25Okapi construction.
    
    Args:
        search_term: Query string (use quotes for exact match: '"PU3" OR "7M2"')
        limit: Max results
        min_score: Minimum Lucene score
        include_content: If True, return full section content (slower but needed for context)
        section_ids: Optional list to filter/re-rank only specific sections (for BM25 re-ranking)
    
    Returns:
        List of sections with section_id, doc_id, title, score, [content]
    """
    if not tool_ctx.neo4j_driver:
        return []
    
    try:
        async with tool_ctx.neo4j_driver.session() as session:
            # Build WHERE clause with optional section_ids filter
            where_clause = "score > $min_score"
            if section_ids:
                where_clause += " AND node.id IN $section_ids"
            
            # Section nodes have 'id' field, not 'section_id'
            if include_content:
                query = f"""
                CALL db.index.fulltext.queryNodes('sectionSearch', $search_term)
                YIELD node, score
                WHERE {where_clause}
                RETURN node.id AS section_id,
                       node.doc_id AS doc_id, 
                       node.title AS title,
                       node.content AS content,
                       node.page_start AS page_start,
                       node.page_end AS page_end,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """
            else:
                query = f"""
                CALL db.index.fulltext.queryNodes('sectionSearch', $search_term)
                YIELD node, score
                WHERE {where_clause}
                RETURN node.id AS section_id,
                       node.doc_id AS doc_id, 
                       node.title AS title, 
                       score
                ORDER BY score DESC
                LIMIT $limit
                """
            
            params = {
                "search_term": search_term,
                "min_score": min_score,
                "limit": limit
            }
            if section_ids:
                params["section_ids"] = section_ids
            
            result = await session.run(query, params)
            data = await result.data()
            
            if data:
                logger.debug(f"neo4j_fulltext_search: found {len(data)} results")
            return data
                
    except Exception as e:
        logger.error(f"neo4j_fulltext_search failed: {e}")
        return []


@tool
def qdrant_search_text(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for relevant text chunks using semantic similarity.
    
    Args:
        query: Search query
        limit: Maximum number of results (default 10)
    
    Returns:
        List of text chunks with section_id, doc_id, text_preview, page info
    """
    try:
        # ⚡ OPTIMIZATION: Try cache first to avoid duplicate embedding API calls
        query_vector = tool_ctx.get_cached_embedding(query)
        
        if query_vector is None:
            embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key
            )
            query_vector = embeddings.embed_query(query)
            tool_ctx.cache_embedding(query, query_vector)
        
        # Build filters
        filter_conditions = []
        if tool_ctx.owner:
            filter_conditions.append(
                FieldCondition(key="owner", match=MatchValue(value=tool_ctx.owner))
            )
        if tool_ctx.doc_ids:
            filter_conditions.append(
                FieldCondition(key="doc_id", match=MatchAny(any=tool_ctx.doc_ids))
            )
        
        qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        results = tool_ctx.qdrant_client.search(
            collection_name=settings.text_chunks_collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=0.3,  # Lowered for better recall
            query_filter=qdrant_filter,
        )
        
        hits = []
        for hit in results:
            hits.append({
                "type": "text_chunk",
                "score": float(hit.score),
                "section_id": hit.payload.get("section_id"),
                "doc_id": hit.payload.get("doc_id"),
                "doc_title": hit.payload.get("doc_title"),
                "section_title": hit.payload.get("section_title", ""),
                "page_start": hit.payload.get("page_start"),
                "page_end": hit.payload.get("page_end"),
                "text": hit.payload.get("text", ""),  # Full text from Qdrant
                "text_preview": hit.payload.get("text_preview", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "chunk_char_start": hit.payload.get("chunk_char_start", 0),
                "chunk_char_end": hit.payload.get("chunk_char_end", 0),
            })
        
        logger.info(f"qdrant_search_text: found {len(hits)} chunks")
        if hits:
            scores = [f"{h['score']:.3f}" for h in hits[:5]]
            logger.debug(f"Top chunk scores: {scores}")
            pages_info = [(h.get('page_start'), h.get('section_title', '')[:50]) for h in hits[:3]]
            logger.debug(f"Top chunk pages: {pages_info}")
        return hits
        
    except Exception as e:
        logger.error(f"qdrant_search_text failed: {e}")
        return []


@tool
def qdrant_search_tables(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for relevant tables using semantic similarity.
    
    Args:
        query: Search query
        limit: Maximum number of results (default 5)
    
    Returns:
        List of table chunks with table_id, doc_id, text_preview, page info
    """
    try:
        # ⚡ OPTIMIZATION: Try cache first to avoid duplicate embedding API calls
        query_vector = tool_ctx.get_cached_embedding(query)
        
        if query_vector is None:
            embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key
            )
            query_vector = embeddings.embed_query(query)
            tool_ctx.cache_embedding(query, query_vector)
        
        # Build filters
        filter_conditions = []
        if tool_ctx.owner:
            filter_conditions.append(
                FieldCondition(key="owner", match=MatchValue(value=tool_ctx.owner))
            )
        if tool_ctx.doc_ids:
            filter_conditions.append(
                FieldCondition(key="doc_id", match=MatchAny(any=tool_ctx.doc_ids))
            )
        
        qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        results = tool_ctx.qdrant_client.search(
            collection_name=settings.tables_text_collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=0.3,
            query_filter=qdrant_filter,
        )
        
        hits = []
        for hit in results:
            hits.append({
                "type": "table_chunk",
                "score": float(hit.score),
                "table_id": hit.payload.get("table_id"),
                "doc_id": hit.payload.get("doc_id"),
                "doc_title": hit.payload.get("doc_title"),
                "page": hit.payload.get("page"),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "total_chunks": hit.payload.get("total_chunks", 1),
                "table_title": hit.payload.get("table_title", ""),
                "table_caption": hit.payload.get("table_caption", ""),
                "text_preview": hit.payload.get("text_preview", ""),
                "rows": hit.payload.get("rows"),
                "cols": hit.payload.get("cols"),
                "file_path": hit.payload.get("file_path"),  # Image path for display
            })
        
        logger.info(f"qdrant_search_tables: found {len(hits)} tables")
        return hits
        
    except Exception as e:
        logger.error(f"qdrant_search_tables failed: {e}")
        return []


@tool
def qdrant_search_schemas(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for relevant diagrams/schemas using semantic similarity.
    
    Args:
        query: Search query
        limit: Maximum number of results (default 5)
    
    Returns:
        List of schemas with schema_id, doc_id, file_path, caption
    """
    try:
        # ⚡ OPTIMIZATION: Try cache first to avoid duplicate embedding API calls
        query_vector = tool_ctx.get_cached_embedding(query)
        
        if query_vector is None:
            embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key
            )
            query_vector = embeddings.embed_query(query)
            tool_ctx.cache_embedding(query, query_vector)
        
        # Build filters
        filter_conditions = []
        if tool_ctx.owner:
            filter_conditions.append(
                FieldCondition(key="owner", match=MatchValue(value=tool_ctx.owner))
            )
        if tool_ctx.doc_ids:
            filter_conditions.append(
                FieldCondition(key="doc_id", match=MatchAny(any=tool_ctx.doc_ids))
            )
        
        qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        results = tool_ctx.qdrant_client.search(
            collection_name=settings.schemas_collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=0.3,
            query_filter=qdrant_filter,
        )
        
        hits = []
        for hit in results:
            hits.append({
                "type": "schema",
                "score": float(hit.score),
                "schema_id": hit.payload.get("schema_id"),
                "doc_id": hit.payload.get("doc_id"),
                "doc_title": hit.payload.get("doc_title"),
                "page": hit.payload.get("page"),
                "title": hit.payload.get("title", ""),
                "caption": hit.payload.get("caption", ""),
                "file_path": hit.payload.get("file_path"),
                "thumbnail_path": hit.payload.get("thumbnail_path"),
                "section_id": hit.payload.get("section_id"),
            })
        
        logger.info(f"qdrant_search_schemas: found {len(hits)} schemas")
        return hits
        
    except Exception as e:
        logger.error(f"qdrant_search_schemas failed: {e}")
        return []


@tool
async def neo4j_query(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Execute read-only Cypher query against Neo4j graph database.
    Use this for structural queries (finding tables/schemas by page, section relationships, etc).
    
    Args:
        cypher: READ-ONLY Cypher query (MATCH, OPTIONAL MATCH, RETURN, WHERE, ORDER BY, LIMIT)
        params: Query parameters (optional)
    
    Returns:
        List of result records as dictionaries
    
    IMPORTANT: 
    - Use ONLY read queries (no CREATE, MERGE, DELETE, SET, REMOVE, DROP)
    - See GRAPH_SCHEMA_PROMPT for schema details
    """
    if params is None:
        params = {}
    
    # Safety check - block write operations
    cypher_upper = cypher.upper()
    forbidden_keywords = ["CREATE", "MERGE", "DELETE", "REMOVE", "SET", "DROP", "DETACH"]
    
    for keyword in forbidden_keywords:
        if keyword in cypher_upper:
            error_msg = f"Write operation '{keyword}' is not allowed. Use only read queries."
            logger.error(error_msg)
            return [{"error": error_msg}]
    
    try:
        async with tool_ctx.neo4j_driver.session() as session:
            result = await session.run(cypher, params)
            records = await result.data()
            
            logger.info(f"neo4j_query: executed successfully, returned {len(records)} records")
            return records
            
    except Exception as e:
        error_msg = f"Neo4j query failed: {str(e)}"
        logger.error(error_msg)
        return [{"error": error_msg}]


@tool
async def neo4j_entity_search(query: str, include_tables: bool = True, include_schemas: bool = True) -> Dict[str, Any]:
    """
    Search for content related to maritime entities (systems, components) via Neo4j graph.
    
    ⚡ OPTIMIZED: Entity discovery uses single UNION query (was 3+ sequential queries)
    
    HOW IT WORKS:
    1. Extracts entity mentions from query (pumps, valves, FO, P-101, etc.)
    2. Single UNION query finds entities by: exact codes, equipment codes in names, multi-word phrases
    3. Queries Neo4j graph for nodes linked to these entities via:
       - Section -[:DESCRIBES]-> Entity (returns full section content!)
       - Table -[:MENTIONS]-> Entity (returns metadata)
       - Schema -[:DEPICTS]-> Entity (returns metadata + file path)
    4. Returns content directly from Neo4j (no Qdrant round-trip needed)
    
    WHAT GETS ADDED TO CONTEXT:
    - Full text content of sections that DESCRIBE the entity
    - Table metadata (title, caption, page) - content fetched separately if needed
    - Schema metadata (title, caption, file_path) for display
    
    USE ONLY WHEN question asks about:
    - Specific components: "fuel oil pump", "cooling water valve"
    - Equipment codes: "P-101", "V-205"
    - Component relationships: "what pumps are in system X"
    
    Args:
        query: Natural language query (entities will be auto-extracted)
        include_tables: Whether to include related tables (default True)
        include_schemas: Whether to include related schemas/diagrams (default True)
    
    Returns:
        Dict with extracted entities and related content
    """
    try:
        # Extract entities from query using dictionary-based extractor
        extractor = get_entity_extractor()
        extraction = extractor.extract_from_question(query)
        
        entity_ids = extraction.get("entity_ids", [])
        
        # Extract potential equipment codes from query (e.g., PU3, 7M2, P-101, V-205)
        # Pattern: alphanumeric codes like PU3, 7M2, P-101, TK-102
        equipment_codes = re.findall(r'\b([A-Z]{1,3}[-]?\d{1,4}[A-Z]?|\d{1,2}[A-Z]{1,3}\d?)\b', query, re.IGNORECASE)
        equipment_codes = list(dict.fromkeys([code.upper() for code in equipment_codes]))  # Dedupe, preserve order
        
        if equipment_codes:
            logger.info(f"neo4j_entity_search: detected equipment codes in query: {equipment_codes}")
        
        # Also search Neo4j for entities matching words in the query
        # This finds dynamically created entities not in the dictionary
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'what', 'how', 'when', 'where', 'why', 
                      'does', 'this', 'that', 'with', 'from', 'have', 'has', 'been', 'will',
                      'can', 'could', 'would', 'should', 'about', 'which', 'their', 'there'}
        query_words -= stop_words
        
        # Track if we found entities by equipment code specifically
        found_by_equipment_code = False
        
        # Extract 2-3 word phrases from query for phrase search
        phrases = []
        if len(query_words) >= 2:
            query_lower = query.lower()
            words = query_lower.split()
            
            # Build 2-word and 3-word phrases
            for i in range(len(words) - 1):
                two_word = f"{words[i]} {words[i+1]}"
                if len(two_word) >= 6 and not any(sw in words[i:i+2] for sw in stop_words):
                    phrases.append(two_word)
                
                if i < len(words) - 2:
                    three_word = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if len(three_word) >= 8 and not any(sw in words[i:i+3] for sw in stop_words):
                        phrases.append(three_word)
        
        if query_words or equipment_codes or entity_ids:
            async with tool_ctx.neo4j_driver.session() as session:
                # ⚡ OPTIMIZED: Single UNION query combines 3 searches (was 3 sequential queries)
                # 1. Search by entity_ids from dictionary (exact codes)
                # 2. Search by equipment codes in entity names (PU3, P-101, etc.)
                # 3. Search by multi-word phrases ("cut cock", "fuel oil pump")
                
                unified_entity_query = """
                // Priority 1: Exact match by entity codes from dictionary
                CALL {
                    WITH $entity_codes AS codes
                    UNWIND codes AS code
                    MATCH (e:Entity) WHERE e.code = code
                    RETURN e.code AS code, e.name AS name, e.entity_type AS entity_type, 1 AS priority, 'exact_code' AS source
                }
                RETURN code, name, entity_type, priority, source
                
                UNION
                
                // Priority 2: Equipment codes in entity names (PU3 in "Pump PU3")
                CALL {
                    WITH $equipment_codes AS codes
                    MATCH (e:Entity)
                    WHERE ANY(eq_code IN codes WHERE toUpper(e.name) CONTAINS eq_code)
                    RETURN e.code AS code, e.name AS name, e.entity_type AS entity_type, 2 AS priority, 'equipment_code' AS source
                    LIMIT 10
                }
                RETURN code, name, entity_type, priority, source
                
                UNION
                
                // Priority 3: Multi-word phrase match ("cut cock" in "Cut Cock Valve")
                CALL {
                    WITH $phrases AS phrase_list
                    MATCH (e:Entity)
                    WHERE SIZE(phrase_list) > 0 
                      AND ANY(phrase IN phrase_list WHERE toLower(e.name) CONTAINS phrase)
                    RETURN e.code AS code, e.name AS name, e.entity_type AS entity_type, 3 AS priority, 'phrase' AS source
                    ORDER BY size(e.name) ASC
                    LIMIT 10
                }
                RETURN code, name, entity_type, priority, source
                
                ORDER BY priority, name
                LIMIT 30
                """
                
                result = await session.run(unified_entity_query, {
                    "entity_codes": entity_ids if entity_ids else [],
                    "equipment_codes": equipment_codes if equipment_codes else [],
                    "phrases": phrases if phrases else []
                })
                unified_entities = await result.data()
                
                logger.info(f"⚡ Unified entity search returned {len(unified_entities)} entities (was 3 separate queries)")
                
                # Process results and update entity_ids
                for ent in unified_entities:
                    ent_code = ent.get("code")
                    ent_name = ent.get("name")
                    ent_type = ent.get("entity_type", "unknown")
                    source = ent.get("source")
                    
                    if ent_code and ent_code not in entity_ids:
                        entity_ids.append(ent_code)
                        found_by_equipment_code = True
                    
                    # Log with source indicator
                    logger.info(f"✓ Found entity [{source}]: {ent_name} ({ent_code}) [{ent_type}]")
                
                # Fourth: If equipment codes detected but NOT found as entities, do fulltext search
                if equipment_codes and not found_by_equipment_code:
                    logger.info(f"neo4j_entity_search: equipment codes {equipment_codes} not in Entity graph, using fulltext")
                    
                    # Try multiple search variations for equipment codes
                    # 1. Exact match with quotes: "CP-1"
                    # 2. Without hyphen: CP1
                    # 3. Fuzzy: CP~1
                    search_variations = []
                    for code in equipment_codes:
                        search_variations.append(f'"{code}"')  # Exact
                        # Also try without hyphen if present
                        if '-' in code:
                            search_variations.append(f'"{code.replace("-", "")}"')
                    
                    search_term = " OR ".join(search_variations)
                    logger.info(f"neo4j_entity_search: fulltext search term: {search_term}")
                    # Use higher min_score and lower limit to reduce pollution
                    sections_data = await neo4j_fulltext_search(search_term, limit=5, min_score=0.5, include_content=True)
                    
                    if sections_data:
                        logger.info(f"neo4j_entity_search: fulltext found {len(sections_data)} sections for codes {equipment_codes}")
                        
                        # Extract section IDs to find related tables/schemas
                        section_ids = [s["section_id"] for s in sections_data if s.get("section_id")]
                        doc_ids = list(set(s["doc_id"] for s in sections_data if s.get("doc_id")))
                        
                        logger.info(f"neo4j_entity_search: section_ids={section_ids}, doc_ids={doc_ids}")
                        logger.info(f"neo4j_entity_search: include_tables={include_tables}, include_schemas={include_schemas}")
                        
                        fulltext_tables = []
                        fulltext_schemas = []
                        
                        # Query for tables and schemas in these sections OR matching equipment code in llm_summary
                        if section_ids or equipment_codes:
                            # Tables: in section OR mentioned in llm_summary/text_preview
                            if include_tables:
                                table_query = """
                                    MATCH (t:Table)
                                    WHERE t.doc_id IN $doc_ids
                                      AND (
                                        EXISTS { MATCH (s:Section)-[:CONTAINS_TABLE]->(t) WHERE s.id IN $section_ids }
                                        OR ANY(code IN $codes WHERE 
                                            toUpper(t.text_preview) CONTAINS code 
                                            OR toUpper(t.caption) CONTAINS code
                                            OR toUpper(t.title) CONTAINS code
                                        )
                                      )
                                    RETURN DISTINCT
                                        t.id AS table_id,
                                        t.title AS table_title,
                                        t.caption AS caption,
                                        t.text_preview AS text_preview,
                                        t.page_number AS page,
                                        t.file_path AS file_path,
                                        t.doc_id AS doc_id
                                    LIMIT 5
                                """
                                result = await session.run(table_query, {
                                    "section_ids": section_ids,
                                    "doc_ids": doc_ids,
                                    "codes": equipment_codes
                                })
                                fulltext_tables = await result.data()
                                logger.info(f"neo4j_entity_search: found {len(fulltext_tables)} tables related to sections/codes")
                            
                            # Schemas: in section OR matching equipment code in llm_summary
                            schema_query = """
                                MATCH (sc:Schema)
                                WHERE sc.doc_id IN $doc_ids
                                  AND (
                                    EXISTS { MATCH (s:Section)-[:CONTAINS_SCHEMA]->(sc) WHERE s.id IN $section_ids }
                                    OR ANY(code IN $codes WHERE 
                                        toUpper(sc.llm_summary) CONTAINS code
                                        OR toUpper(sc.caption) CONTAINS code
                                        OR toUpper(sc.title) CONTAINS code
                                    )
                                  )
                                RETURN DISTINCT
                                    sc.id AS schema_id,
                                    sc.title AS title,
                                    sc.caption AS caption,
                                    sc.llm_summary AS llm_summary,
                                    sc.page_number AS page,
                                    sc.file_path AS file_path,
                                    sc.thumbnail_path AS thumbnail_path,
                                    sc.doc_id AS doc_id
                                LIMIT 5
                            """
                            result = await session.run(schema_query, {
                                "section_ids": section_ids,
                                "doc_ids": doc_ids,
                                "codes": equipment_codes
                            })
                            fulltext_schemas = await result.data()
                            logger.info(f"neo4j_entity_search: found {len(fulltext_schemas)} schemas related to sections/codes")
                        
                        return {
                            "entities": equipment_codes,
                            "entity_names": [f"Equipment {code}" for code in equipment_codes],
                            "sections": [
                                {
                                    "section_id": s["section_id"],
                                    "doc_id": s["doc_id"],
                                    "section_title": s.get("title", ""),
                                    "content": s.get("content", ""),
                                    "page_start": s.get("page_start"),
                                    "page_end": s.get("page_end"),
                                    "score": s["score"],
                                    "matched_entity": equipment_codes[0] if equipment_codes else None,
                                }
                                for s in sections_data
                            ],
                            "tables": [
                                {
                                    "table_id": t["table_id"],
                                    "table_title": t.get("table_title", ""),
                                    "caption": t.get("caption", ""),
                                    "text_preview": t.get("text_preview", ""),
                                    "page": t.get("page"),
                                    "file_path": t.get("file_path"),
                                    "doc_id": t.get("doc_id"),
                                    "matched_entity": equipment_codes[0] if equipment_codes else None,
                                }
                                for t in fulltext_tables
                            ],
                            "schemas": [
                                {
                                    "schema_id": sc["schema_id"],
                                    "title": sc.get("title", ""),
                                    "caption": sc.get("caption", ""),
                                    "llm_summary": sc.get("llm_summary", ""),
                                    "page": sc.get("page"),
                                    "file_path": sc.get("file_path"),
                                    "thumbnail_path": sc.get("thumbnail_path"),
                                    "doc_id": sc.get("doc_id"),
                                    "matched_entity": equipment_codes[0] if equipment_codes else None,
                                }
                                for sc in fulltext_schemas
                            ],
                            "message": f"Found via fulltext search for equipment codes: {equipment_codes}"
                        }
                    else:
                        # Fulltext also failed - return special message to trigger Qdrant fallback
                        logger.info(f"neo4j_entity_search: fulltext found nothing for {equipment_codes}, suggesting semantic search")
                        return {
                            "entities": equipment_codes,
                            "entity_names": [f"Equipment {code}" for code in equipment_codes],
                            "sections": [],
                            "tables": [],
                            "schemas": [],
                            "message": f"Equipment codes {equipment_codes} not found in graph or fulltext. Try qdrant_search_text or qdrant_search_tables for semantic search.",
                            "suggest_semantic_search": True
                        }
                
                # Second: Search by name words (fuzzy matching for component names like "Sf Valve")
                # Extract potential component names (capitalized sequences of 2+ words)
                component_name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
                component_names = re.findall(component_name_pattern, query)
                
                # Also extract equipment codes with prefixes like PT-8, PU-3, etc.
                equipment_prefix_pattern = r'\b([A-Z]{1,3})[-\s]?(\d+|[A-Z]+)\b'
                equipment_with_prefix = re.findall(equipment_prefix_pattern, query)
                
                # Also search by query words if no equipment codes
                search_by_name = (query_words and not equipment_codes) or component_names or equipment_with_prefix
                
                if search_by_name:
                    # Priority 1: Exact match with full component name (e.g., "Sf Valve")
                    exact_match_entities = []
                    if component_names:
                        logger.info(f"neo4j_entity_search: detected component names: {component_names}")
                        
                        for comp_name in component_names:
                            # Try exact match first (e.g., "Sf Valve" → entity name contains "sf valve")
                            exact_query = """
                            MATCH (e:Entity)
                            WHERE toLower(e.name) CONTAINS $full_name
                            RETURN e.code AS code, e.name AS name, 10 AS match_count
                            ORDER BY size(e.name) ASC
                            LIMIT 5
                            """
                            result = await session.run(exact_query, {"full_name": comp_name.lower()})
                            exact_matches = await result.data()
                            exact_match_entities.extend(exact_matches)
                            
                            if exact_matches:
                                logger.info(f"neo4j_entity_search: exact match for '{comp_name}': {len(exact_matches)} entities")
                    
                    # Priority 2: Equipment codes with prefixes (PT-8, PU-3)
                    if equipment_with_prefix:
                        logger.info(f"neo4j_entity_search: detected equipment prefixes: {equipment_with_prefix}")
                        for prefix, number in equipment_with_prefix:
                            # Search for entities starting with prefix (e.g., PT → pressure transmitter)
                            prefix_query = """
                            MATCH (e:Entity)
                            WHERE toLower(e.name) STARTS WITH $prefix
                            RETURN e.code AS code, e.name AS name, 8 AS match_count
                            LIMIT 5
                            """
                            result = await session.run(prefix_query, {"prefix": prefix.lower()})
                            prefix_matches = await result.data()
                            exact_match_entities.extend(prefix_matches)
                    
                    # Priority 3: Fallback to fuzzy matching by words (only if no exact matches)
                    # ⚠️ DISABLED - fuzzy matching causes context pollution
                    # Single generic words like "pump", "valve" match too many entities
                    fuzzy_entities = []
                    if not exact_match_entities:
                        logger.info(f"neo4j_entity_search: no exact matches found, fuzzy matching DISABLED to avoid pollution")
                        # OLD CODE (causes pollution):
                        # search_terms = list(query_words) if query_words else []
                        # MATCH (e:Entity) WHERE ANY(word IN $words WHERE toLower(e.name) CONTAINS word)
                        # Problem: "incinerator" matches 50+ entities → 50+ sections → context overload
                    
                    # Combine results (exact matches first, then fuzzy)
                    neo4j_entities = exact_match_entities + fuzzy_entities
                    
                    if neo4j_entities:
                        logger.info(f"neo4j_entity_search: total found {len(neo4j_entities)} entities (exact + fuzzy)")
                    
                    for ent in neo4j_entities:
                        if ent["code"] and ent["code"] not in entity_ids:
                            entity_ids.append(ent["code"])
                            logger.debug(f"Found entity from Neo4j by name: {ent['name']} ({ent['code']})")
        
        if not entity_ids:
            logger.info(f"neo4j_entity_search: no entities found in query '{query}'")
            return {
                "entities": [],
                "entity_names": [],
                "sections": [],
                "tables": [],
                "schemas": [],
                "message": "No maritime entities detected in query"
            }
        
        # NO EXPANSION - use exact entity codes only
        # Expansion causes pollution: "valve" → matches 50+ generic valve entities
        # Keep only specific multi-word entities (e.g., "cut_cock" not "valve")
        strict_ids = []
        for eid in entity_ids:
            parts = eid.split('_')
            # comp_valve_cock - generic (1 word after type)
            # comp_valve_cut_cock - specific (2+ words after type)
            if eid.startswith('comp_') and len(parts) >= 4:
                strict_ids.append(eid)  # Multi-word component - keep
            elif eid.startswith('sys_'):
                strict_ids.append(eid)  # System - always keep
            else:
                logger.debug(f"Filtered out generic entity: {eid} (single-word component)")
        
        if not strict_ids:
            # If all filtered out, keep original (better than nothing)
            logger.warning(f"All entities filtered as generic, keeping originals: {entity_ids}")
            strict_ids = entity_ids
        
        logger.info(
            f"neo4j_entity_search: extracted {len(entity_ids)} entities, "
            f"filtered to {len(strict_ids)} strict: {strict_ids[:5]}..."
        )
        
        # Get doc_ids filter for Neo4j queries
        doc_ids_filter = tool_ctx.doc_ids if tool_ctx.doc_ids else None
        
        results = {
            "entities": strict_ids,
            "entity_names": extraction.get("system_names", []) + extraction.get("component_names", []),
            "sections": [],
            "tables": [],
            "schemas": [],
        }
        
        # Query Neo4j for sections describing these entities
        # Note: Section.content stores full text, no need to fetch from Qdrant
        async with tool_ctx.neo4j_driver.session() as session:
            # Find sections via DESCRIBES relationship with EXACT MATCH PRIORITY
            # Boost sections where entity name appears in title or multiple times in content
            section_query = """
            UNWIND $entity_ids AS eid
            MATCH (s:Section)-[:DESCRIBES]->(e:Entity {code: eid})
            WHERE $doc_ids IS NULL OR s.doc_id IN $doc_ids
            OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s)
            
            // Calculate relevance score based on entity mentions
            WITH s, e, d,
                 // Title boost: 100x if entity name in section title
                 CASE 
                   WHEN toLower(s.title) CONTAINS toLower(e.name) THEN 100
                   ELSE 1 
                 END as title_boost,
                 // Mention count: how many times entity name appears in content
                 size(split(toLower(s.content), toLower(e.name))) - 1 as mention_count
            
            // Filter: section must actually mention the entity
            WHERE mention_count > 0
            
            RETURN DISTINCT 
                s.id AS section_id,
                s.title AS section_title,
                s.content AS content,
                s.page_start AS page_start,
                s.page_end AS page_end,
                s.doc_id AS doc_id,
                d.title AS doc_title,
                e.code AS entity_code,
                e.name AS entity_name,
                s.importance_score AS importance,
                title_boost * (1 + mention_count) as relevance_score,
                mention_count
            ORDER BY relevance_score DESC, s.importance_score DESC
            LIMIT 10
            """
            result = await session.run(section_query, {"entity_ids": strict_ids, "doc_ids": doc_ids_filter})
            section_records = await result.data()
            
            results["sections"] = [
                {
                    "section_id": r["section_id"],
                    "section_title": r["section_title"],
                    "content": r["content"],  # Full text from Neo4j!
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "doc_id": r["doc_id"],
                    "doc_title": r["doc_title"],
                    "matched_entity": r["entity_name"],
                    "score": r.get("importance", 0.5),  # Use importance as initial score for re-ranking
                    "relevance_score": r.get("relevance_score", 1.0),  # Exact match score
                    "mention_count": r.get("mention_count", 0),  # How many times entity mentioned
                }
                for r in section_records
            ]
            
            # FALLBACK: If no sections via DESCRIBES, search via Table-[:MENTIONS]->Entity
            # Some entities only appear in tables, not in section text
            # Run REGARDLESS of include_tables flag - we need sections, tables are just the link
            if len(results["sections"]) == 0:
                logger.info(f"neo4j_entity_search: no sections via DESCRIBES, trying Table-[:MENTIONS]->Entity path")
                
                table_to_section_query = """
                UNWIND $entity_ids AS eid
                MATCH (t:Table)-[:MENTIONS]->(e:Entity {code: eid})
                WHERE $doc_ids IS NULL OR t.doc_id IN $doc_ids
                
                // Get the section containing this table
                OPTIONAL MATCH (s:Section)-[:CONTAINS_TABLE]->(t)
                OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s)
                
                WHERE s IS NOT NULL  // Must have a section
                
                RETURN DISTINCT
                    s.id AS section_id,
                    s.title AS section_title,
                    s.content AS content,
                    s.page_start AS page_start,
                    s.page_end AS page_end,
                    s.doc_id AS doc_id,
                    d.title AS doc_title,
                    t.id AS table_id,
                    t.title AS table_title,
                    e.code AS entity_code,
                    e.name AS entity_name,
                    s.importance_score AS importance
                ORDER BY s.importance_score DESC
                LIMIT 10
                """
                
                result = await session.run(table_to_section_query, {"entity_ids": strict_ids, "doc_ids": doc_ids_filter})
                sections_via_table = await result.data()
                
                if sections_via_table:
                    logger.info(
                        f"✓ Found {len(sections_via_table)} sections via Table-[:MENTIONS]->Entity path "
                        f"(entity appears in table, not section text)"
                    )
                    
                    results["sections"] = [
                        {
                            "section_id": r["section_id"],
                            "section_title": r["section_title"],
                            "content": r["content"],
                            "page_start": r["page_start"],
                            "page_end": r["page_end"],
                            "doc_id": r["doc_id"],
                            "doc_title": r["doc_title"],
                            "matched_entity": r["entity_name"],
                            "score": r.get("importance", 0.5),
                            "relevance_score": 1.0,  # No mention count available (entity in table)
                            "mention_count": 0,  # Entity not in section text
                            "found_via": "table_mentions",  # Debug marker
                            "related_table_id": r.get("table_id"),
                        }
                        for r in sections_via_table
                    ]
                    
                    # ALSO fetch the actual tables that were found via MENTIONS
                    # This ensures tables with entities appear in results even if include_tables=False
                    table_ids_from_mentions = [r["table_id"] for r in sections_via_table if r.get("table_id")]
                    logger.info(f"   🔍 DEBUG: sections_via_table={len(sections_via_table)}, table_ids extracted={table_ids_from_mentions}")
                    if table_ids_from_mentions:
                        logger.info(f"   📊 Fetching {len(table_ids_from_mentions)} tables found via MENTIONS relationship")
                        
                        fetch_tables_query = """
                        UNWIND $table_ids AS tid
                        MATCH (t:Table {id: tid})
                        OPTIONAL MATCH (d:Document)-[:HAS_TABLE]->(t)
                        OPTIONAL MATCH (tc:TableChunk)-[:PART_OF]->(t)
                        OPTIONAL MATCH (t)-[:MENTIONS]->(e:Entity)
                        WHERE e.code IN $entity_ids
                        WITH t, d, collect(DISTINCT tc.text_preview) AS chunk_previews, collect(DISTINCT e.name) AS entity_names
                        RETURN DISTINCT 
                            t.id AS table_id,
                            t.title AS table_title,
                            t.caption AS caption,
                            t.text_preview AS text_preview,
                            t.page_number AS page,
                            t.rows AS rows,
                            t.cols AS cols,
                            t.csv_path AS csv_path,
                            t.file_path AS file_path,
                            t.doc_id AS doc_id,
                            d.title AS doc_title,
                            chunk_previews,
                            entity_names
                        """
                        
                        result = await session.run(fetch_tables_query, {
                            "table_ids": table_ids_from_mentions,
                            "entity_ids": strict_ids
                        })
                        tables_from_mentions = await result.data()
                        
                        results["tables"] = [
                            {
                                "table_id": r["table_id"],
                                "table_title": r["table_title"],
                                "caption": r["caption"],
                                "text_preview": r["text_preview"] or "",
                                "chunk_previews": r["chunk_previews"] or [],
                                "page": r["page"],
                                "rows": r["rows"],
                                "cols": r["cols"],
                                "csv_path": r["csv_path"],
                                "file_path": r["file_path"],
                                "doc_id": r["doc_id"],
                                "doc_title": r["doc_title"],
                                "matched_entity": ", ".join(r.get("entity_names", [])),
                                "found_via": "table_mentions",
                            }
                            for r in tables_from_mentions
                        ]
                        logger.info(f"   ✓ Fetched {len(results['tables'])} tables with entity mentions")
                else:
                    logger.warning(f"neo4j_entity_search: no sections found via Table-[:MENTIONS]->Entity either")
            
            # Find tables via MENTIONS relationship
            if include_tables:
                table_query = """
                UNWIND $entity_ids AS eid
                MATCH (t:Table)-[:MENTIONS]->(e:Entity {code: eid})
                WHERE $doc_ids IS NULL OR t.doc_id IN $doc_ids
                OPTIONAL MATCH (d:Document)-[:HAS_TABLE]->(t)
                OPTIONAL MATCH (tc:TableChunk)-[:PART_OF]->(t)
                WITH e, t, d, collect(tc.text_preview) AS chunk_previews
                RETURN DISTINCT 
                    t.id AS table_id,
                    t.title AS table_title,
                    t.caption AS caption,
                    t.text_preview AS text_preview,
                    t.page_number AS page,
                    t.rows AS rows,
                    t.cols AS cols,
                    t.csv_path AS csv_path,
                    t.file_path AS file_path,
                    t.doc_id AS doc_id,
                    d.title AS doc_title,
                    e.code AS entity_code,
                    e.name AS entity_name,
                    chunk_previews
                LIMIT 3
                """
                result = await session.run(table_query, {"entity_ids": strict_ids, "doc_ids": doc_ids_filter})
                table_records = await result.data()
                
                results["tables"] = [
                    {
                        "table_id": r["table_id"],
                        "table_title": r["table_title"],
                        "caption": r["caption"],
                        "text_preview": r["text_preview"] or "",
                        "chunk_previews": r["chunk_previews"] or [],  # All chunk previews
                        "page": r["page"],
                        "rows": r["rows"],
                        "cols": r["cols"],
                        "csv_path": r["csv_path"],
                        "file_path": r["file_path"],
                        "doc_id": r["doc_id"],
                        "doc_title": r["doc_title"],
                        "matched_entity": r["entity_name"],
                    }
                    for r in table_records
                ]
            
            # Find schemas via DEPICTS relationship
            if include_schemas:
                schema_query = """
                UNWIND $entity_ids AS eid
                MATCH (sc:Schema)-[:DEPICTS]->(e:Entity {code: eid})
                WHERE $doc_ids IS NULL OR sc.doc_id IN $doc_ids
                OPTIONAL MATCH (d:Document)-[:HAS_SCHEMA]->(sc)
                RETURN DISTINCT 
                    sc.id AS schema_id,
                    sc.title AS title,
                    sc.caption AS caption,
                    sc.text_context AS text_context,
                    sc.llm_summary AS llm_summary,
                    sc.page_number AS page,
                    sc.file_path AS file_path,
                    sc.thumbnail_path AS thumbnail_path,
                    sc.doc_id AS doc_id,
                    d.title AS doc_title,
                    e.code AS entity_code,
                    e.name AS entity_name
                LIMIT 3
                """
                result = await session.run(schema_query, {"entity_ids": strict_ids, "doc_ids": doc_ids_filter})
                schema_records = await result.data()
                
                results["schemas"] = [
                    {
                        "schema_id": r["schema_id"],
                        "title": r["title"],
                        "caption": r["caption"],
                        "text_context": r["text_context"] or "",  # Text near schema
                        "llm_summary": r["llm_summary"] or "",    # LLM-generated description
                        "page": r["page"],
                        "file_path": r["file_path"],
                        "thumbnail_path": r["thumbnail_path"],
                        "doc_id": r["doc_id"],
                        "doc_title": r["doc_title"],
                        "matched_entity": r["entity_name"],
                    }
                    for r in schema_records
                ]
        
        logger.info(
            f"neo4j_entity_search: found {len(results['sections'])} sections, "
            f"{len(results['tables'])} tables, {len(results['schemas'])} schemas"
        )
        
        # Log relevance scores for debugging
        if results['sections']:
            top_3 = results['sections'][:3]
            for i, sec in enumerate(top_3, 1):
                logger.info(
                    f"  [{i}] {sec['section_title'][:50]}... | "
                    f"entity: {sec['matched_entity']} | "
                    f"mentions: {sec['mention_count']} | "
                    f"relevance: {sec['relevance_score']:.1f}"
                )
        
        return results
        
    except Exception as e:
        error_msg = f"Entity search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "entities": [], "sections": [], "tables": [], "schemas": []}


# Collect tools
TOOLS = [qdrant_search_text, qdrant_search_tables, qdrant_search_schemas, neo4j_query, neo4j_entity_search]


# ============================================================================
# NODE 1: UNIFIED QUERY ANALYZER & ROUTER (⚡ OPTIMIZED - SINGLE LLM CALL)
# ============================================================================
# 
# OPTIMIZATION: This node merges two previously separate LLM calls:
#   - OLD: node_analyze_question (intent classification) → node_router_agent (tool selection)
#   - NEW: node_analyze_and_route (both in one call)
#
# BENEFITS:
#   - Eliminates 1 LLM round-trip (~1-2 seconds saved)
#   - Reduces API costs by 50% for initial query processing
#   - Intent and tool selection are logically related - LLM can optimize both together
#
# HOW IT WORKS:
#   - Prompt instructs LLM to: (1) classify intent, (2) select tools accordingly
#   - LLM returns tool_calls which implicitly encode the intent
#   - We infer intent from selected tools (schemas→schema, tables→table, etc.)
#
# ============================================================================

def node_analyze_and_route(state: GraphState) -> GraphState:
    """
    UNIFIED NODE: Analyze question intent AND select tools in a single LLM call.
    This eliminates the bottleneck of sequential LLM calls (was the main performance issue).
    """
    question = state["question"]
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"📝 NEW QUESTION: {question}")
    logger.info(f"{'#'*60}\n")
    
    # ⚡ Clear embedding cache for new query
    tool_ctx.clear_embedding_cache()
    
    # Ensure entities are loaded (lazy initialization on first call)
    ensure_entities_loaded()
    
    anchors = state.get("anchor_sections", [])
    
    # Build anchor info for agent
    anchor_info = ""
    if anchors:
        anchor_info = "\n\nANCHOR SECTIONS (focus on these):\n"
        for a in anchors:
            anchor_info += f"- Section {a['section_id']} (doc: {a['doc_id']}, score: {a['score']:.2f})\n"
    
    # Entity detection hint (directive for equipment codes, informative for named components)
    # ONLY add hint if entities are VALIDATED against graph knowledge base
    found_entities = find_entities_in_question(question, tool_ctx.known_entities)
    
    # Filter: keep only entities that exist in graph knowledge base
    validated_entities = [e for e in found_entities if e in tool_ctx.known_entities]
    
    if validated_entities:
        logger.info(f"🔍 Detected & validated entities: {validated_entities}")
        # Check if any entities are equipment codes (uppercase + numbers pattern)
        import re
        has_equipment_codes = any(re.match(r'^[A-Z]{1,4}[-]?[0-9]{1,5}$', e) for e in validated_entities)
        
        if has_equipment_codes:
            # Directive hint for equipment codes
            entity_hint = f"""

⚠️ EQUIPMENT CODES DETECTED: {', '.join(validated_entities[:5])}

IMPORTANT: These are specific equipment identifiers. You SHOULD use neo4j_entity_search to find:
- Cross-references to this equipment across document sections
- Related tables, diagrams, and technical data
- Contextual information about this specific component

RECOMMENDED APPROACH:
1. Call neo4j_entity_search with the equipment codes
2. ALSO call qdrant_search_text/tables/schemas for semantic context
3. Combine graph cross-references + semantic search for complete answer

Equipment codes are the PRIMARY identifiers - graph search is essential here."""
        else:
            # Informative hint for named components
            entity_hint = f"""

📍 DETECTED ENTITIES: {', '.join(validated_entities[:5])}

These named components exist in documentation. Consider:
- WHERE/WHICH DIAGRAM/LOCATION queries → neo4j_entity_search
- HOW/WHY/EXPLAIN procedures → qdrant_search_text (semantic)
- SPECS/PARAMETERS → neo4j_entity_search + qdrant_search_tables

This is guidance - use your judgment based on question type."""
    elif found_entities:
        # Entities detected but NOT in graph - log warning
        logger.warning(f"⚠️ Detected entities NOT in graph: {found_entities[:5]}")
        entity_hint = """

📍 No specific equipment entities detected.
→ Use semantic search (qdrant_search_*) for best results."""
    else:
        entity_hint = """

📍 No specific equipment entities detected.
→ Use semantic search (qdrant_search_*) for best results."""
    
    # Build unified system prompt that includes BOTH intent classification AND tool selection
    system_prompt = f"""{GRAPH_SCHEMA_PROMPT}

You are a routing agent for maritime technical documentation Q&A system.

YOUR TASK (TWO STEPS IN ONE):
1. First, classify the question intent
2. Then, select the best tools to answer it

STEP 1: INTENT CLASSIFICATION (what format does user expect?)

Classify into ONE of these categories:

🔍 **"text"** - User wants TEXTUAL EXPLANATION
   - Questions: "How does X work", "What is Y", "Explain Z", troubleshooting ("no suction", "failure")
   - Even if answer comes from table, user expects TEXT output
   - Tools: qdrant_search_text (primary), optionally qdrant_search_tables for data support

📊 **"table"** - User wants to SEE THE TABLE (display request)
   - Keywords: "show table", "display table", "specifications", "specs", "parameters"
   - Examples: "Show me specs table", "What are the parameters" (wants to see table)
   - Tools: qdrant_search_tables (MANDATORY), optionally qdrant_search_text for context

📐 **"schema"** - User wants to SEE DIAGRAM/DRAWING
   - Keywords: "diagram", "schematic", "drawing", "show drawing", "display diagram"
   - Examples: "Show me the diagram", "Give me the schematic"
   - Tools: qdrant_search_schemas (MANDATORY), optionally qdrant_search_text for explanation

🔧 **"mixed"** - User wants to SEE BOTH TABLE AND DIAGRAM
   - Keywords: "show specifications and diagram", "display table and schematic", "specs with drawing"
   - Examples: "Show me the specs table and system diagram for pump"
   - User expects BOTH table AND diagram in answer
   - Tools: qdrant_search_tables + qdrant_search_schemas (BOTH MANDATORY)

STEP 2: TOOL SELECTION (based on intent)

Available tools:
- qdrant_search_text: Semantic search in text sections
- qdrant_search_tables: Semantic search in tables (specs, troubleshooting)
- qdrant_search_schemas: Semantic search in diagrams/drawings
- neo4j_entity_search: Graph search for specific equipment codes

Selection rules by intent:

✅ intent="text" → ["qdrant_search_text"]
   May optionally add qdrant_search_tables if answer needs data from tables
   Exception: if equipment codes detected → add "neo4j_entity_search"

✅ intent="table" → ["qdrant_search_tables"]
   Must include tables tool for table display
   May optionally add qdrant_search_text for context

✅ intent="schema" → ["qdrant_search_schemas"]
   Must include schemas tool for diagram display
   May optionally add qdrant_search_text for explanation

✅ intent="mixed" → ["qdrant_search_tables", "qdrant_search_schemas"]
   BOTH tables AND schemas MANDATORY - user expects to see both
   May optionally add qdrant_search_text for additional context
   Add neo4j_entity_search if equipment codes detected

INTENT-BASED PARAMETER SELECTION:
When using neo4j_entity_search, set include_tables and include_schemas based on intent:
- intent="text" → include_tables=False, include_schemas=False (only sections)
- intent="table" → include_tables=True, include_schemas=False
- intent="schema" → include_tables=False, include_schemas=True
- intent="mixed" → include_tables=True, include_schemas=True

🎯 CRITICAL TOOL SELECTION RULES (READ CAREFULLY!):

TOOL SELECTION GUIDE:

1. **qdrant_search_text** - For text questions:
   For: explanations, descriptions, procedures, "how/what/why" questions
   Examples: "How does X work", "What is Y", "explain Z"
   ✅ Finds relevant DESCRIPTIONS and ANSWERS
   ✅ Best F1 score (0.90) - most reliable tool

2. **qdrant_search_tables** - MANDATORY for troubleshooting/specs/parameters:
   
   ⚠️ ALWAYS USE when question contains these keywords:
   - Troubleshooting: "no suction", "not working", "failure", "alarm", "fault", "error", "problem", "cause"
   - Specifications: "specs", "specifications", "parameters", "capacity", "temperature", "pressure", "power", "range"
   - Technical values: "values", "ratings", "limits", "dimensions"
   
   🎯 TWO USAGE SCENARIOS:
   
   A) Troubleshooting/fault diagnosis (MOST IMPORTANT):
      Question: "Pump has no suction. What causes it?"
      → MUST call qdrant_search_tables (troubleshooting tables contain causes/solutions)
      → Extract info from table, return as TEXT explanation
      
   B) Specifications request:
      Question: "Show specifications table for pump"
      → Call qdrant_search_tables
      → Return [TABLE1] reference to show table
   
   📊 Tables contain: troubleshooting guides, specs, parameters, technical data
   ⚠️ CRITICAL: For ANY troubleshooting question, you MUST call both:
      1. qdrant_search_tables (to find troubleshooting table)
      2. qdrant_search_text (for additional context)
   
   DO NOT skip qdrant_search_tables for troubleshooting - it's the PRIMARY source!

3. **qdrant_search_schemas** - For diagrams/drawings/schematics/figures:
   Any question about visual content: "drawings", "diagram", "schema", "figure", "layout"
   Examples: "give me drawings of X", "show diagram of Y", "all schematics of Z"
   Works with semantic search - finds relevant images by caption/context
   IMPORTANT: When question asks to EXPLAIN diagram → ALSO call qdrant_search_text!

4. **neo4j_entity_search** - For SPECIFIC equipment (codes OR named components):
   
   ⚠️ USE CAREFULLY - Can pollute context if used for generic terms!
   
   ✅ USE WHEN question mentions:
   
   A) EQUIPMENT CODES (always use):
      - HGM-30, CR-302, PU3, SV4, PT-6018, INC-8130, P-101, etc.
      - Examples: "specs of PU3", "where is HGM-30", "troubleshoot valve SV4"
   
   B) NAMED COMPONENTS (specific names with qualifiers):
      - "Isolation Valve" (not just "valve")
      - "Fuel Oil Pump" (not just "pump")
      - "HFO Cooler" (specific equipment name)
      - Examples: "how does Isolation Valve work", "Fuel Oil Pump maintenance"
   
   C) LOCATION/REFERENCE queries:
      - "WHERE is X located/shown"
      - "find all references to X"
      - "which sections mention X"
   
   ❌ DO NOT USE FOR GENERIC TERMS (single word, no qualifier):
   - "incinerator" → too generic, use qdrant_search_text
   - "pump" → too generic, use qdrant_search_text
   - "valve" → too generic, use qdrant_search_text
   - "burner" → too generic, use qdrant_search_text
   - "chamber" → too generic, use qdrant_search_text
   
   WHY this distinction matters:
   - "pump" = 100+ mentions → context pollution
   - "Fuel Oil Pump" = 5-10 specific mentions → useful cross-references
   - Named components have clearer scope in entity graph
   
   📊 PERFORMANCE:
   - Entity search for codes/named components: F1 = 0.85
   - Entity search for generic terms: F1 = 0.11 (context pollution)
   - Semantic search alone: F1 = 0.90 (best for generic terms)

5. **neo4j_query** - ONLY for STRUCTURAL queries by section NUMBER:
   When question references SPECIFIC section/chapter NUMBER like "4.4", "3.2"
   Examples: "tables from section 4.4", "content of chapter 3.2"
   NOT for keyword search - Neo4j is for structure, not semantic search!

MULTI-TOOL STRATEGY (call multiple tools when needed):
- "explain diagram/table" → schemas/tables + text (need context!)
- "show drawings + describe" → schemas + text
- "specs and how it works" → tables + text
- Equipment code + diagrams → neo4j_entity_search + schemas
- Equipment + question (how/when/why/conditions) → neo4j_entity_search + qdrant_search_text
- Equipment + specs/parameters → neo4j_entity_search + qdrant_search_tables
- Section number + specific content → neo4j_query + text/tables/schemas

SINGLE TOOL CASES:
- "give me drawings" (no explanation) → schemas only
- "show me specs" (just data) → tables only
- "how does X work" (pure text) → text only

🚨 MANDATORY TOOL SELECTION RULES (FOLLOW STRICTLY!):

1. If intent="table" → YOU MUST CALL qdrant_search_tables
   - NO EXCEPTIONS - table intent = search tables tool
   - This is NOT optional - it's REQUIRED
   
2. If intent="schema" → YOU MUST CALL qdrant_search_schemas
   - NO EXCEPTIONS - schema intent = search schemas tool
   - This is NOT optional - it's REQUIRED

3. If intent="text" → Use qdrant_search_text
   - Unless troubleshooting → ALSO call qdrant_search_tables

⚡ FEW-SHOT EXAMPLES - MEMORIZE THE TOOL PATTERNS:

Example 1 - Specifications (intent=table):
Q: "What are specifications of the main engine?"
Intent: table
Tools: qdrant_search_tables ← MANDATORY for intent=table
Reason: Intent is table → MUST search tables

Example 2 - Troubleshooting (intent=text but needs tables):
Q: "The pump has no suction. What can be the cause?"
Intent: text
Tools: qdrant_search_tables AND qdrant_search_text
Reason: Troubleshooting keyword → search tables for causes

Example 3 - Diagram request (intent=schema):
Q: "Show me the cooling system diagram"
Intent: schema
Tools: qdrant_search_schemas ← MANDATORY for intent=schema
Reason: Intent is schema → MUST search schemas

Example 4 - Procedure (intent=text only):
Q: "How does the fuel injection system work?"
Intent: text
Tools: qdrant_search_text
Reason: Pure explanation → text search only

🔴 CRITICAL REMINDER:
- intent="table" → qdrant_search_tables is NOT OPTIONAL
- intent="schema" → qdrant_search_schemas is NOT OPTIONAL
- Failing to call the correct tool = WRONG ANSWER

QUESTION: "{question}"{anchor_info}

RESPONSE FORMAT (2-step structured output):
Step 1 - INTENT (single word on first line): text | table | schema | mixed
Step 2 - TOOLS (JSON array on second line): ["tool1", "tool2", ...]

Example response:
mixed
["qdrant_search_text", "qdrant_search_tables"]

NOW: Analyze and respond."""
    
    # Create LLM without tools (get text response first, then map to tool calls)
    llm = get_llm_instance(temperature=0)
    
    # Build messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # Get structured response from LLM
    response = llm.invoke(messages)
    
    # Parse response content
    response_text = response.content.strip()
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    # Extract intent and tools
    intent = None
    selected_tools = []
    
    try:
        # First non-empty line should be intent
        if lines:
            intent_candidate = lines[0].lower()
            if intent_candidate in ["text", "table", "schema", "mixed"]:
                intent = intent_candidate
                logger.info(f"🎯 LLM classified intent: {intent.upper()}")
        
        # Second line should be JSON array of tools
        if len(lines) > 1:
            import json
            tools_json = lines[1]
            # Handle case where tools might be in markdown code block
            if tools_json.startswith('[') and tools_json.endswith(']'):
                selected_tools = json.loads(tools_json)
                logger.info(f"🔧 LLM selected tools: {selected_tools}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to parse LLM response: {e}")
        logger.warning(f"Raw response: {response_text[:200]}")
    
    # Fallback: if parsing failed, infer from selected tools
    if not intent or not selected_tools:
        logger.warning("⚠️ Intent parsing failed, inferring from tools")
        
        # Infer intent from tools: mixed = BOTH tables AND schemas
        tool_set = set(selected_tools) if selected_tools else set()
        has_tables = "qdrant_search_tables" in tool_set
        has_schemas = "qdrant_search_schemas" in tool_set
        
        # mixed = именно tables + schemas одновременно
        if has_tables and has_schemas:
            intent = "mixed"
        elif has_schemas:
            intent = "schema"
        elif has_tables:
            intent = "table"
        else:
            intent = "text"
        
        logger.info(f"🎯 Inferred intent from tools: {intent.upper()}")
    
    # Convert selected tools to tool_calls format
    tool_calls = []
    for tool_name in selected_tools:
        if tool_name in ["qdrant_search_text", "qdrant_search_tables", "qdrant_search_schemas"]:
            tool_calls.append({
                "name": tool_name,
                "args": {"query": question, "limit": 5},
                "id": f"llm_{tool_name}_{len(tool_calls)}"
            })
        elif tool_name == "neo4j_entity_search":
            # ⚠️ STRICT VALIDATION: Only add neo4j_entity_search if we have VALIDATED entities
            # Use validated_entities from above (already filtered against graph knowledge base)
            if validated_entities:
                logger.info(f"✅ neo4j_entity_search validated with entities: {validated_entities[:3]}")
                tool_calls.append({
                    "name": tool_name,
                    "args": {
                        "query": question,
                        "entity_names": validated_entities[:3],  # Use only validated entities
                        "include_tables": intent in ["table", "mixed"],
                        "include_schemas": intent in ["schema", "mixed"]
                    },
                    "id": f"llm_neo4j_{len(tool_calls)}"
                })
            else:
                if found_entities:
                    logger.warning(f"❌ neo4j_entity_search SKIPPED: Found entities {found_entities[:3]} not in graph")
                else:
                    logger.warning(f"❌ neo4j_entity_search SKIPPED: No entities detected")
    
    # Create mock response object with tool_calls
    from langchain_core.messages import AIMessage
    response = AIMessage(content=response_text, tool_calls=tool_calls)
    tool_names = {tc.get('name') for tc in tool_calls}
    
    state["query_intent"] = intent
    logger.info(f"✅ Final intent: {intent.upper()}, tools: {list(tool_names)}")
    
    # STEP 3: Validate intent-tool consistency and force missing tools
    validation_failed = False
    
    # Check 1: intent=table MUST have qdrant_search_tables
    if intent == "table" and "qdrant_search_tables" not in tool_names:
        logger.error(f"❌ VALIDATION: intent=table but qdrant_search_tables NOT called!")
        logger.warning(f"🔧 Forcing qdrant_search_tables...")
        validation_failed = True
        forced_tool_call = {
            "name": "qdrant_search_tables",
            "args": {"query": question, "limit": 5},
            "id": "forced_table_search"
        }
        existing_calls = tool_calls if tool_calls else []
        response.tool_calls = existing_calls + [forced_tool_call]
        tool_calls = response.tool_calls
    
    # Check 2: intent=schema MUST have qdrant_search_schemas
    if intent == "schema" and "qdrant_search_schemas" not in tool_names:
        logger.error(f"❌ VALIDATION: intent=schema but qdrant_search_schemas NOT called!")
        logger.warning(f"🔧 Forcing qdrant_search_schemas...")
        validation_failed = True
        forced_tool_call = {
            "name": "qdrant_search_schemas",
            "args": {"query": question, "limit": 5},
            "id": "forced_schema_search"
        }
        existing_calls = tool_calls if tool_calls else []
        response.tool_calls = existing_calls + [forced_tool_call]
        tool_calls = response.tool_calls
    
    # Check 3: intent=mixed MUST have BOTH tables AND schemas
    if intent == "mixed":
        if "qdrant_search_tables" not in tool_names:
            logger.warning(f"🔧 Mixed intent: forcing qdrant_search_tables (required for mixed)")
            forced_tool_call = {
                "name": "qdrant_search_tables",
                "args": {"query": question, "limit": 5},
                "id": "forced_mixed_tables"
            }
            existing_calls = tool_calls if tool_calls else []
            response.tool_calls = existing_calls + [forced_tool_call]
            tool_calls = response.tool_calls
            validation_failed = True
        
        if "qdrant_search_schemas" not in tool_names:
            logger.warning(f"🔧 Mixed intent: forcing qdrant_search_schemas (required for mixed)")
            forced_tool_call = {
                "name": "qdrant_search_schemas",
                "args": {"query": question, "limit": 5},
                "id": "forced_mixed_schemas"
            }
            response.tool_calls = tool_calls + [forced_tool_call]
            tool_calls = response.tool_calls
            validation_failed = True
    
    # Check 4: AGGRESSIVE heuristic - force schemas for visual keywords
    # even if LLM didn't classify as schema intent
    question_lower = question.lower()
    visual_keywords = ["diagram", "drawing", "schematic", "figure", "layout", "schema", "image", "picture", "visual"]
    has_visual_keyword = any(kw in question_lower for kw in visual_keywords)
    
    if has_visual_keyword and "qdrant_search_schemas" not in tool_names:
        logger.warning(f"🔧 Visual keywords detected: forcing qdrant_search_schemas")
        forced_tool_call = {
            "name": "qdrant_search_schemas",
            "args": {"query": question, "limit": 5},
            "id": "forced_visual_schemas"
        }
        response.tool_calls = tool_calls + [forced_tool_call]
        tool_calls = response.tool_calls
        validation_failed = True
    
    # Check 5: Force tables for specification/parameter keywords
    spec_keywords = ["specification", "parameter", "spec", "value", "rating", "dimension", "capacity"]
    has_spec_keyword = any(kw in question_lower for kw in spec_keywords)
    
    if has_spec_keyword and "qdrant_search_tables" not in tool_names:
        logger.warning(f"🔧 Specification keywords detected: forcing qdrant_search_tables")
        forced_tool_call = {
            "name": "qdrant_search_tables",
            "args": {"query": question, "limit": 5},
            "id": "forced_spec_tables"
        }
        response.tool_calls = tool_calls + [forced_tool_call]
        tool_calls = response.tool_calls
        validation_failed = True
    
    # Check 6: Force entity search for validated equipment codes
    if validated_entities and "neo4j_entity_search" not in tool_names:
        # Check if there's at least one equipment code pattern
        import re
        has_equipment = any(re.match(r'^[A-Z]{1,4}[-]?[0-9]{1,5}$', e) for e in validated_entities)
        
        if has_equipment:
            logger.warning(f"🔧 Equipment codes detected but entity search not called: forcing neo4j_entity_search")
            forced_tool_call = {
                "name": "neo4j_entity_search",
                "args": {
                    "query": question,
                    "entity_names": validated_entities[:3],
                    "include_tables": intent in ["table", "mixed"],
                    "include_schemas": intent in ["schema", "mixed"]
                },
                "id": "forced_entity_search"
            }
            response.tool_calls = tool_calls + [forced_tool_call]
            tool_calls = response.tool_calls
            validation_failed = True
    
    if validation_failed:
        tool_names = {tc.get('name') for tc in tool_calls}
        logger.info(f"✅ Tools after validation: {', '.join(tool_names)}")
    else:
        logger.info(f"✅ Tool selection validated - no corrections needed")
    
    # Update tool_names after all validations (may have been modified by forced tools)
    tool_names = {tc.get('name') for tc in tool_calls}
    
    # Store tool names in state for later use in context building
    state["tool_names"] = list(tool_names)
    
    # Store messages for next iteration - APPEND to existing messages, don't overwrite!
    state["messages"] = state["messages"] + [*messages, response]
    
    # Log detailed tool call information
    if tool_calls:
        logger.info(f"\n{'='*60}")
        logger.info(f"🤖 UNIFIED NODE - TOOL CALLS ({len(tool_calls)} tools):")
        for i, tc in enumerate(tool_calls, 1):
            tool_name = tc.get('name', 'unknown')
            tool_args = tc.get('args', {})
            logger.info(f"  [{i}] 🔧 {tool_name}")
            for arg_name, arg_value in tool_args.items():
                # Truncate long values for readability
                display_value = str(arg_value)
                if len(display_value) > 100:
                    display_value = display_value[:100] + "..."
                logger.info(f"      └─ {arg_name}: {display_value}")
        logger.info(f"{'='*60}\n")
    else:
        logger.info("🤖 Agent responded WITHOUT tool calls (direct answer)")
    
    return state



# NODE 3: TOOL EXECUTOR

async def node_execute_tools(state: GraphState) -> GraphState:
    """
    Execute tools called by agent.
    Collects results from Qdrant and Neo4j.
    """
    # Set filters from state into tool context (for Qdrant/Neo4j queries)
    tool_ctx.owner = state.get("owner")
    tool_ctx.doc_ids = state.get("doc_ids")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    # Safely get tool calls
    tool_calls = get_tool_calls(last_message)
    
    if not tool_calls:
        logger.warning("No tool calls from agent")
        state["search_results"] = {"text": [], "tables": [], "schemas": []}
        state["neo4j_results"] = []
        state["anchor_sections"] = []
        return state
    
    # Execute each tool call
    search_results = {"text": [], "tables": [], "schemas": []}
    neo4j_results = []
    
    tool_messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        logger.info(f"\n⚙️  EXECUTING: {tool_name}")
        logger.info(f"   Args: {tool_args}")
        
        try:
            if tool_name == "qdrant_search_text":
                result = qdrant_search_text.invoke(tool_args)
                search_results["text"].extend(result)
                logger.info(f"   ✅ qdrant_search_text: found {len(result)} text chunks")
                tool_messages.append(
                    ToolMessage(
                        content=f"Found {len(result)} text chunks",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            
            elif tool_name == "qdrant_search_tables":
                result = qdrant_search_tables.invoke(tool_args)
                search_results["tables"].extend(result)
                logger.info(f"   ✅ qdrant_search_tables: found {len(result)} tables")
                tool_messages.append(
                    ToolMessage(
                        content=f"Found {len(result)} tables",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            
            elif tool_name == "qdrant_search_schemas":
                result = qdrant_search_schemas.invoke(tool_args)
                search_results["schemas"].extend(result)
                logger.info(f"   ✅ qdrant_search_schemas: found {len(result)} schemas")
                tool_messages.append(
                    ToolMessage(
                        content=f"Found {len(result)} schemas",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            
            elif tool_name == "neo4j_query":
                result = await neo4j_query.ainvoke(tool_args)
                neo4j_results.extend(result)
                logger.info(f"   ✅ neo4j_query: returned {len(result)} records")
                if result:
                    # Log sample of keys returned
                    sample_keys = list(result[0].keys()) if result[0] else []
                    logger.info(f"      Sample keys: {sample_keys[:5]}")
                tool_messages.append(
                    ToolMessage(
                        content=f"Neo4j returned {len(result)} records",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            
            elif tool_name == "neo4j_entity_search":
                result = await neo4j_entity_search.ainvoke(tool_args)
                
                # Process entity search results
                # Section content comes directly from Neo4j (no Qdrant fetch needed!)
                entity_sections = result.get("sections", [])
                entity_tables = result.get("tables", [])
                entity_schemas = result.get("schemas", [])
                
                # RE-RANK entity sections by semantic relevance to ENTITY NAMES (not full question)
                # Problem: entity_search finds ALL mentions, but we need sections about THESE specific entities
                # EXCEPTION: Skip BM25 for sections found via table_mentions (entity in table, not text)
                if entity_sections:
                    # Check if sections found via table_mentions
                    table_mention_sections = [sec for sec in entity_sections if sec.get("found_via") == "table_mentions"]
                    
                    if table_mention_sections:
                        # Skip BM25 - entity is in table, not section text
                        logger.info(f"   ⏭️  Skipping BM25 re-ranking: {len(table_mention_sections)} sections found via Table-[:MENTIONS] (entity in table, not text)")
                        
                        # Add all table_mention sections directly (no filtering)
                        for sec in table_mention_sections:
                            content = sec.get("content", "")
                            search_results["text"].append({
                                "type": "text_chunk",
                                "score": 0.4,  # Fixed moderate score
                                "section_id": sec.get("section_id"),
                                "doc_id": sec.get("doc_id"),
                                "doc_title": sec.get("doc_title", ""),
                                "section_title": sec.get("section_title", ""),
                                "page_start": sec.get("page_start"),
                                "page_end": sec.get("page_end"),
                                "text": content,
                                "text_preview": content[:500] if content else "",
                                "chunk_index": 0,
                                "source": "entity_search",
                                "matched_entity": sec.get("matched_entity"),
                                "entity_confidence": sec.get("score", 0.5),
                                "found_via": "table_mentions",
                                "related_table_id": sec.get("related_table_id"),
                            })
                        
                        # Store in neo4j_results for potential use
                        neo4j_results.extend(table_mention_sections)
                    
                    # ⚡ OPTIMIZED: Use Neo4j precomputed BM25 index instead of building BM25Okapi
                    regular_sections = [sec for sec in entity_sections if sec.get("found_via") != "table_mentions"]
                    
                    if regular_sections:
                        try:
                            import re
                            
                            # EXTRACT UNIQUE ENTITY NAMES from sections (e.g., "Fuel Oil Pump", "Oil Cooler")
                            entity_names = list(set(
                                sec.get("matched_entity", "")
                                for sec in regular_sections
                                if sec.get("matched_entity")
                            ))
                            
                            if not entity_names:
                                raise ValueError("No matched entities in sections")
                            
                            logger.info(f"   🎯 Entities to re-rank: {entity_names}")
                            
                            # Extract tokens from entity names for Lucene query
                            entity_tokens = []
                            for entity_name in entity_names:
                                tokens = [token for token in re.findall(r'\b\w+\b', entity_name) if len(token) > 2]
                                entity_tokens.extend(tokens)
                            entity_tokens = list(dict.fromkeys(entity_tokens))  # Dedupe
                            
                            if not entity_tokens:
                                raise ValueError("No valid tokens in entity names")
                            
                            # Build Lucene query with boosting
                            lucene_query = " OR ".join(f'"{term}"^2.0' for term in entity_tokens)
                            
                            # Get section IDs to re-rank
                            section_ids = [sec.get("section_id") for sec in regular_sections]
                            
                            # ⚡ Use Neo4j fulltext index with section_ids filter for BM25 re-ranking
                            bm25_results = await neo4j_fulltext_search(
                                search_term=lucene_query,
                                section_ids=section_ids,  # Filter to only these sections
                                limit=5,
                                min_score=0.5,
                                include_content=False
                            )
                            
                            if not bm25_results:
                                # Fallback: if Neo4j BM25 returned nothing, keep top 3 by relevance_score
                                logger.warning(f"   ⚠️ Neo4j BM25 returned no results, using fallback (top 3 by relevance)")
                                regular_sections.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                                top_sections = regular_sections[:3]
                            else:
                                # Build score map from results
                                bm25_score_map = {r["section_id"]: float(r["score"]) for r in bm25_results}
                                max_bm25 = max(bm25_score_map.values()) if bm25_score_map else 1.0
                                
                                # Filter and score sections based on Neo4j BM25 results
                                scored_sections = []
                                for sec in regular_sections:
                                    sec_id = sec.get("section_id")
                                    if sec_id in bm25_score_map:
                                        # Normalize BM25 score (Lucene scores are typically 0-10 range)
                                        raw_score = bm25_score_map[sec_id]
                                        normalized_score = raw_score / max_bm25 if max_bm25 > 0 else 0
                                        # Cap at 0.4 to not overpower Qdrant semantic scores
                                        final_score = min(0.4, normalized_score * 0.4)
                                        scored_sections.append((sec, final_score, raw_score))
                                
                                # Sort by BM25 score
                                scored_sections.sort(key=lambda x: x[2], reverse=True)
                                top_sections = scored_sections
                                
                                logger.info(f"   🔄 BM25 re-ranked: {len(regular_sections)} → {len(top_sections)} sections")
                                if top_sections:
                                    top_scores = [x[2] for x in top_sections[:3]]
                                    logger.info(f"   📊 Top BM25 scores: {[f'{s:.3f}' for s in top_scores]}")
                            
                            # Add BM25 re-ranked sections
                            for item in top_sections:
                                if isinstance(item, tuple):
                                    sec, final_score, bm25_score = item
                                else:
                                    # Fallback case (plain section dict)
                                    sec = item
                                    final_score = 0.3
                                    bm25_score = 0
                                content = sec.get("content", "")
                                search_results["text"].append({
                                    "type": "text_chunk",
                                    "score": final_score,  # Semantic-adjusted score
                                    "section_id": sec.get("section_id"),
                                    "doc_id": sec.get("doc_id"),
                                    "doc_title": sec.get("doc_title", ""),
                                    "section_title": sec.get("section_title", ""),
                                    "page_start": sec.get("page_start"),
                                    "page_end": sec.get("page_end"),
                                    "text": content,  # Full text from Neo4j!
                                    "text_preview": content[:500] if content else "",
                                    "chunk_index": 0,
                                    "source": "entity_search",
                                    "matched_entity": sec.get("matched_entity"),
                                    "entity_confidence": sec.get("score", 0.5),  # Original entity match score
                                    "bm25_score": bm25_score,  # BM25 relevance to question
                                })
                    
                        except Exception as e:
                            logger.warning(f"   ⚠️ BM25 re-ranking failed: {e}, using default scores")
                            # Fallback: use original logic with VERY LOW scores and strict limit
                            for sec in entity_sections[:5]:  # Limit to 5 only
                                content = sec.get("content", "")
                                if content:
                                    entity_score = 0.3  # Very low score - let qdrant win
                                    search_results["text"].append({
                                        "type": "text_chunk",
                                        "score": entity_score,
                                        "section_id": sec.get("section_id"),
                                        "doc_id": sec.get("doc_id"),
                                        "doc_title": sec.get("doc_title", ""),
                                        "section_title": sec.get("section_title", ""),
                                        "page_start": sec.get("page_start"),
                                        "page_end": sec.get("page_end"),
                                        "text": content,
                                        "text_preview": content[:500] if content else "",
                                        "chunk_index": 0,
                                        "source": "entity_search",
                                        "matched_entity": sec.get("matched_entity"),
                                        "entity_confidence": sec.get("score", 0.5),
                                    })
                
                # Add entity-matched tables with text preview + chunk previews
                for tbl in entity_tables:
                    # Build combined text preview from table + chunks
                    text_preview = tbl.get("text_preview", "")
                    chunk_previews = tbl.get("chunk_previews", [])
                    
                    # Combine table preview with first chunk previews (up to 3)
                    combined_preview = text_preview
                    if chunk_previews:
                        chunks_text = "\n---\n".join(chunk_previews[:3])
                        combined_preview = f"{text_preview}\n\nTable content:\n{chunks_text}"
                    
                    search_results["tables"].append({
                        "type": "table_chunk",
                        "score": 0.6,  # Medium score for entity tables
                        "table_id": tbl.get("table_id"),
                        "doc_id": tbl.get("doc_id"),
                        "doc_title": tbl.get("doc_title", ""),
                        "page": tbl.get("page"),
                        "rows": tbl.get("rows"),
                        "cols": tbl.get("cols"),
                        "table_title": tbl.get("table_title", ""),
                        "table_caption": tbl.get("caption", ""),
                        "text_preview": combined_preview,  # Now includes actual content!
                        "csv_path": tbl.get("csv_path"),
                        "file_path": tbl.get("file_path"),  # Image path for display!
                        "source": "entity_search",
                        "matched_entity": tbl.get("matched_entity"),
                    })
                
                # Add entity-matched schemas with text_context and llm_summary
                for sch in entity_schemas:
                    search_results["schemas"].append({
                        "type": "schema",
                        "score": 0.7,  # Good score for entity schemas (visual)
                        "schema_id": sch.get("schema_id"),
                        "doc_id": sch.get("doc_id"),
                        "doc_title": sch.get("doc_title", ""),
                        "page": sch.get("page"),
                        "title": sch.get("title", ""),
                        "caption": sch.get("caption", ""),
                        "text_context": sch.get("text_context", ""),  # Text near schema
                        "llm_summary": sch.get("llm_summary", ""),    # LLM-generated description
                        "file_path": sch.get("file_path"),
                        "thumbnail_path": sch.get("thumbnail_path"),
                        "source": "entity_search",
                        "matched_entity": sch.get("matched_entity"),
                    })
                
                # Store entity sections in neo4j_results for re-ranking
                # IMPORTANT: Only store sections that passed BM25 filtering (from search_results["text"])
                # Extract section_ids that were added to search_results (after BM25 filtering)
                added_section_ids = {
                    item.get("section_id") 
                    for item in search_results["text"] 
                    if item.get("source") == "entity_search"
                }
                
                # Store only BM25-filtered sections in neo4j_results
                for sec in entity_sections:
                    if sec.get("section_id") in added_section_ids:
                        neo4j_results.append({
                            "type": "text_chunk",
                            "section_id": sec.get("section_id"),
                            "doc_id": sec.get("doc_id"),
                            "doc_title": sec.get("doc_title", "Unknown"),
                            "section_title": sec.get("section_title", ""),
                            "content": sec.get("content", ""),  # Full text!
                            "page_start": sec.get("page_start"),
                            "page_end": sec.get("page_end"),
                            "page": sec.get("page_start"),  # Alias for compatibility
                            "importance": sec.get("score", 0.5),  # Entity match score for sorting
                            "source": "entity_search",
                        })
                
                # Store entity tables for re-ranking
                for tbl in entity_tables:
                    neo4j_results.append({
                        "type": "table_chunk",
                        "section_id": tbl.get("section_id"),  # Tables may have section_id
                        "table_id": tbl.get("table_id"),
                        "doc_id": tbl.get("doc_id"),
                        "doc_title": tbl.get("doc_title", "Unknown"),
                        "page": tbl.get("page"),
                        "source": "entity_search",
                    })
                
                # Store entity schemas for context building
                for sch in entity_schemas:
                    neo4j_results.append({
                        "type": "schema",
                        "schema_id": sch.get("schema_id"),
                        "doc_id": sch.get("doc_id"),
                        "doc_title": sch.get("doc_title", "Unknown"),
                        "title": sch.get("title", ""),
                        "caption": sch.get("caption", ""),
                        "page": sch.get("page"),
                        "file_path": sch.get("file_path"),
                        "source": "entity_search",
                    })
                
                entities_found = result.get("entity_names", [])
                entities_ids = result.get("entities", [])
                logger.info(f"   ✅ neo4j_entity_search results:")
                logger.info(f"      Extracted entities: {entities_ids[:5]}{'...' if len(entities_ids) > 5 else ''}")
                logger.info(f"      Entity names: {entities_found}")
                logger.info(f"      Found: {len(entity_sections)} sections, {len(entity_tables)} tables, {len(entity_schemas)} schemas")
                logger.info(f"      Stored {len(neo4j_results)} items in neo4j_results for re-ranking")
                
                # If entity search suggests semantic search (equipment code not found anywhere)
                # Automatically trigger Qdrant fallback
                if result.get("suggest_semantic_search") and not entity_sections and not entity_tables:
                    logger.info(f"   🔄 Entity search suggests semantic fallback, running qdrant_search_text + qdrant_search_tables")
                    question = state["question"]
                    
                    # Run Qdrant text search
                    text_fallback = qdrant_search_text.invoke({"query": question, "limit": 10})
                    search_results["text"].extend(text_fallback)
                    logger.info(f"      Qdrant text fallback: {len(text_fallback)} chunks")
                    
                    # Run Qdrant table search (since intent was TABLE)
                    table_fallback = qdrant_search_tables.invoke({"query": question, "limit": 5})
                    search_results["tables"].extend(table_fallback)
                    logger.info(f"      Qdrant table fallback: {len(table_fallback)} tables")
                
                # INTENT CORRECTION: If entity found in tables/schemas, intent should be "mixed"
                # This avoids hacky filtering later - just fix the root cause (wrong intent)
                current_intent = state.get("query_intent")
                
                # Check if entity found via table_mentions (entity in table, not text)
                table_mention_sections = [sec for sec in entity_sections if sec.get("found_via") == "table_mentions"]
                needs_tables = table_mention_sections and entity_tables and current_intent == "text"
                
                # Check if entity found in schemas
                needs_schemas = len(entity_schemas) > 0 and current_intent in ["text", "table"]
                
                if needs_tables or needs_schemas:
                    correction_reason = []
                    if needs_tables:
                        correction_reason.append(f"{len(entity_tables)} tables")
                    if needs_schemas:
                        correction_reason.append(f"{len(entity_schemas)} schemas")
                    
                    logger.info(f"   🔄 Intent correction: entity found in {' and '.join(correction_reason)} → changing intent from '{current_intent}' to 'mixed'")
                    state["query_intent"] = "mixed"
                
                tool_messages.append(
                    ToolMessage(
                        content=f"Entity search found: {len(entity_sections)} sections, {len(entity_tables)} tables, {len(entity_schemas)} schemas for entities: {entities_found}",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}: {e}")
            tool_messages.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_id,
                    name=tool_name
                )
            )
    
    state["search_results"] = search_results
    state["neo4j_results"] = neo4j_results
    state["messages"].extend(tool_messages)
    
    # RE-RANKING: Boost Qdrant results that match Neo4j entity sections
    # This implements "semantic search within entity-filtered sections"
    if neo4j_results and (search_results["text"] or search_results["tables"]):
        neo4j_section_ids = {item.get("section_id") for item in neo4j_results if item.get("section_id")}
        
        if neo4j_section_ids:
            # Boost text chunks that appear in Neo4j entity sections
            for chunk in search_results["text"]:
                if chunk.get("section_id") in neo4j_section_ids:
                    original_score = chunk.get("score", 0)
                    chunk["score"] = min(1.0, original_score * 1.5)  # 50% boost, cap at 1.0
                    chunk["boosted"] = True
                    logger.debug(f"Boosted text chunk score: {original_score:.3f} → {chunk['score']:.3f}")
            
            # Boost tables that appear in Neo4j entity sections
            for table in search_results["tables"]:
                if table.get("section_id") in neo4j_section_ids:
                    original_score = table.get("score", 0)
                    table["score"] = min(1.0, original_score * 1.5)
                    table["boosted"] = True
                    logger.debug(f"Boosted table score: {original_score:.3f} → {table['score']:.3f}")
            
            boosted_text = sum(1 for c in search_results["text"] if c.get("boosted"))
            boosted_tables = sum(1 for t in search_results["tables"] if t.get("boosted"))
            logger.info(f"🎯 Re-ranking: boosted {boosted_text} text chunks, {boosted_tables} tables matching entity sections")
    
    # Log summary of collected context
    logger.info(f"\n📊 TOOL EXECUTION SUMMARY:")
    logger.info(f"   Text chunks: {len(search_results['text'])}")
    logger.info(f"   Tables: {len(search_results['tables'])}")
    logger.info(f"   Schemas: {len(search_results['schemas'])}")
    logger.info(f"   Neo4j records: {len(neo4j_results)}")
    
    # FALLBACK: Neo4j fulltext search if Qdrant results are poor
    text_results = search_results.get("text", [])
    high_quality_results = [r for r in text_results if r.get("score", 0) > 0.3]
    
    if len(high_quality_results) < 2 and state["query_intent"] in ["text", "mixed"]:
        logger.info(f"⚠️  Poor results ({len(high_quality_results)} with score > 0.3), trying Neo4j fulltext fallback")
        
        try:
            # Use shared fulltext search function
            query = state["question"]
            fulltext_results = await neo4j_fulltext_search(query, limit=3, min_score=0.5)
            
            if fulltext_results:
                logger.info(f"✅ Neo4j fulltext found {len(fulltext_results)} sections")
                
                # Fetch chunks from Qdrant for these sections
                for ft_result in fulltext_results:
                    section_id = ft_result.get("section_id")
                    if not section_id:
                        continue
                    
                    # Get all chunks for this section from Qdrant
                    search_filter = Filter(
                        must=[
                            FieldCondition(key="section_id", match=MatchValue(value=section_id))
                        ]
                    )
                    
                    chunks = tool_ctx.qdrant_client.scroll(
                        collection_name=settings.text_chunks_collection,
                        scroll_filter=search_filter,
                        limit=10,
                        with_payload=True,
                        with_vectors=False,
                    )
                    
                    # Add chunks to text_results
                    for point in chunks[0]:
                        search_results["text"].append({
                            "type": "text_chunk",
                            "score": float(ft_result["score"]),  # Use Neo4j fulltext score
                            "section_id": point.payload.get("section_id"),
                            "doc_id": point.payload.get("doc_id"),
                            "section_title": point.payload.get("section_title", ""),
                            "page_start": point.payload.get("page_start"),
                            "page_end": point.payload.get("page_end"),
                            "text": point.payload.get("text", ""),
                            "text_preview": point.payload.get("text_preview", ""),
                            "chunk_index": point.payload.get("chunk_index", 0),
                            "source": "neo4j_fulltext_fallback"
                        })
                
                logger.info(f"Added {len(search_results['text']) - len(text_results)} chunks from fulltext sections")
            else:
                logger.info("Neo4j fulltext: no results")
                
        except Exception as e:
            logger.error(f"Neo4j fulltext fallback failed: {e}")
    
    # SELECT ANCHOR SECTIONS with importance scores
    # First, collect unique section_ids
    text_hits = search_results.get("text", [])
    section_ids = list({h.get("section_id") for h in text_hits if h.get("section_id")})
    
    # Fetch importance scores from Neo4j
    importance_scores = await _fetch_importance_scores(section_ids)
    
    anchor_sections = _select_anchor_sections(
        text_hits,
        max_sections=5,
        importance_scores=importance_scores
    )
    
    # FALLBACK: If no text anchors but have tables/schemas, create virtual anchors
    # This ensures primary_doc is set for table/schema-only queries
    if not anchor_sections:
        tables = search_results.get("tables", [])
        schemas = search_results.get("schemas", [])
        
        if tables or schemas:
            # Create virtual anchors from top tables/schemas
            from collections import Counter
            doc_scores = Counter()
            
            # Count doc appearances in tables/schemas (weighted by score)
            for hit in tables[:5]:
                doc_id = hit.get("doc_id")
                score = hit.get("score", 0.5)
                if doc_id:
                    doc_scores[doc_id] += score
            
            for hit in schemas[:5]:
                doc_id = hit.get("doc_id")
                score = hit.get("score", 0.5)
                if doc_id:
                    doc_scores[doc_id] += score
            
            # Create virtual anchor for top doc
            if doc_scores:
                top_doc = doc_scores.most_common(1)[0][0]
                anchor_sections = [{
                    "doc_id": top_doc,
                    "section_id": "virtual_anchor",  # Placeholder
                    "score": 1.0,
                    "virtual": True  # Mark as virtual anchor
                }]
                logger.info(f"Created virtual anchor for table/schema-only query: doc={top_doc}")
    
    state["anchor_sections"] = anchor_sections
    
    logger.info(
        f"Tools executed: "
        f"text={len(search_results['text'])}, "
        f"tables={len(search_results['tables'])}, "
        f"schemas={len(search_results['schemas'])}, "
        f"neo4j={len(neo4j_results)}, "
        f"anchors={len(anchor_sections)}"
    )
    
    return state


def _select_anchor_sections(text_hits: List[Dict[str, Any]], max_sections: int = 3, importance_scores: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Select top anchor sections based on combined score.
    
    Final score = similarity * 0.7 + importance * 0.2
    (remaining 0.1 reserved for future factors like recency)
    
    Groups text chunks by (doc_id, section_id) and picks top N sections.
    """
    from collections import defaultdict
    
    importance_scores = importance_scores or {}
    
    groups = defaultdict(list)
    for h in text_hits:
        key = (h.get("doc_id"), h.get("section_id"))
        groups[key].append(h)
    
    scored = []
    for (doc_id, section_id), hits in groups.items():
        # Take max similarity score among hits in this section
        raw_similarity = max(h.get("score", 0) for h in hits)
        
        # Check if this is from entity_search (graph relationships, not semantic similarity)
        source = hits[0].get("source", "qdrant")
        
        # Get importance score from Neo4j (default 0.5)
        importance = importance_scores.get(section_id, 0.5)
        
        # CRITICAL: entity_search scores are NOT semantic similarity!
        # They can be: BM25 normalized (0-1), Lucene scores (unbounded), or fallback (0.2-0.3)
        # Always use fixed scoring for entity sections to avoid score confusion
        if source == "entity_search":
            # Entity sections: moderate fixed score + importance boost
            # Ignore raw_similarity (unreliable - could be BM25, Lucene, or fallback)
            final_score = 0.5 + importance * 0.3  # Range: 0.5-0.65
            # Normalize similarity for display (clamp to 0-1 range)
            similarity = min(1.0, max(0.0, raw_similarity))
        else:
            # Qdrant semantic search: similarity dominant + importance boost
            # Qdrant scores are reliable cosine similarity (0-1 range)
            similarity = raw_similarity
            final_score = similarity * 0.7 + importance * 0.2
        
        scored.append({
            "doc_id": doc_id,
            "section_id": section_id,
            "score": final_score,
            "similarity": similarity,
            "importance": importance,
        })
    
    # Sort by final score and take top N
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    selected = scored[:max_sections]
    
    logger.info(f"Selected {len(selected)} anchor sections from {len(scored)} candidates")
    
    return selected


async def _fetch_importance_scores(section_ids: List[str]) -> Dict[str, float]:
    """Fetch importance_score for sections from Neo4j."""
    if not section_ids or not tool_ctx.neo4j_driver:
        return {}
    
    try:
        async with tool_ctx.neo4j_driver.session() as session:
            query = """
            MATCH (s:Section)
            WHERE s.id IN $section_ids
            RETURN s.id AS section_id, s.importance_score AS importance
            """
            result = await session.run(query, {"section_ids": section_ids})
            data = await result.data()
            
            return {
                r["section_id"]: r.get("importance") or 0.5 
                for r in data
            }
    except Exception as e:
        logger.warning(f"Failed to fetch importance scores: {e}")
        return {}


# NODE 4: CONTEXT BUILDER (WITH NEIGHBOR EXPANSION)

async def node_build_context(
    state: GraphState,
    driver: Driver,
) -> GraphState:
    """
    Build enriched context from tool results with anchor-based filtering.
    
    Steps:
    1. Get anchor sections (top 3 by relevance)
    2. Filter text/tables/schemas to only anchor docs/sections
    3. Fetch full content from Neo4j
    4. EXPAND with neighbor chunks from same section
    5. Deduplicate and sort by relevance
    6. Apply HARD LIMITS: max 3 sections, 3 tables, 3 schemas
    """
    search_results = state["search_results"]
    neo4j_results = state["neo4j_results"]
    anchors = state.get("anchor_sections", [])
    
    # Build anchor filter sets
    anchor_keys = {(a["doc_id"], a["section_id"]) for a in anchors if not a.get("virtual")}
    anchor_doc_ids = {a["doc_id"] for a in anchors}
    anchor_section_ids = {a["section_id"] for a in anchors if not a.get("virtual")}
    
    # Check if anchors are virtual (table/schema-only query)
    has_virtual_anchor = any(a.get("virtual") for a in anchors)
    
    # Determine PRIMARY document with intent-aware scoring
    # Tables/schemas should primarily come from this doc to avoid context pollution
    from collections import Counter
    doc_scores = Counter()
    query_intent = state.get("query_intent", "text")
    
    # Score anchors with intent-based weights
    for a in anchors:
        doc_id = a.get("doc_id")
        if doc_id:
            base_score = a.get("score", 0.5)
            # Boost score if anchor type matches query intent
            if query_intent == "table" and a.get("source") == "tables":
                base_score *= 1.5  # Strongly prefer doc with relevant tables
            elif query_intent == "schema" and a.get("source") == "schemas":
                base_score *= 1.5  # Strongly prefer doc with relevant schemas
            elif query_intent == "text" and a.get("source") == "text":
                base_score *= 1.2  # Prefer doc with relevant text
            doc_scores[doc_id] += base_score
    
    primary_doc_id = doc_scores.most_common(1)[0][0] if doc_scores else None
    
    logger.info(f"Anchor filtering: {len(anchor_keys)} sections, {len(anchor_doc_ids)} docs, primary_doc={primary_doc_id}, virtual={has_virtual_anchor}")
    
    enriched = []
    
    # Process text chunks with RELAXED anchor filtering
    # Strategy: anchor sections get priority, but allow top chunks from primary_doc to improve recall
    text_hits = search_results.get("text", [])
    text_chunks_added = 0
    MAX_TEXT_FROM_PRIMARY = 3  # Allow top 3 from primary_doc even if not in anchors
    
    for hit in text_hits:
        key = (hit.get("doc_id"), hit.get("section_id"))
        hit_doc_id = hit.get("doc_id")
        
        # Priority 1: Chunks from anchor sections (always include)
        if key in anchor_keys:
            item = await _fetch_and_expand_text_chunk(driver, hit, tool_ctx.vector_service)
            if item:
                enriched.append(item)
                text_chunks_added += 1
        # Priority 2: Top chunks from primary_doc (even if not in anchors, to improve recall)
        elif primary_doc_id and hit_doc_id == primary_doc_id and text_chunks_added < MAX_TEXT_FROM_PRIMARY:
            logger.debug(f"Including text chunk from primary_doc (not in anchors): {key}")
            item = await _fetch_and_expand_text_chunk(driver, hit, tool_ctx.vector_service)
            if item:
                enriched.append(item)
                text_chunks_added += 1
        else:
            logger.debug(f"Skipping text chunk - not in anchor sections or primary_doc: {key}")
            continue
    
    # Process tables (ONLY from PRIMARY doc to avoid context pollution)
    for hit in search_results.get("tables", []):
        hit_doc_id = hit.get("doc_id")
        
        # If virtual anchor (table-only query), allow tables from any doc (rely on semantic search)
        if has_virtual_anchor:
            # Virtual anchor means no text context - accept all relevant tables
            pass
        # Otherwise strict filtering: only from primary doc
        elif primary_doc_id and hit_doc_id != primary_doc_id:
            logger.debug(f"Skipping table - not from primary doc: {hit.get('table_id')} (doc={hit_doc_id})")
            continue
        elif not primary_doc_id and hit_doc_id not in anchor_doc_ids:
            logger.debug(f"Skipping table - not in anchor docs: {hit.get('table_id')}")
            continue
        
        item = await _fetch_table_full(driver, hit)
        if item:
            enriched.append(item)
    
    # Process schemas (ONLY from PRIMARY doc)
    for hit in search_results.get("schemas", []):
        hit_doc_id = hit.get("doc_id")
        
        # If virtual anchor (schema-only query), allow schemas from any doc (rely on semantic search)
        if has_virtual_anchor:
            # Virtual anchor means no text context - accept all relevant schemas
            pass
        # Otherwise strict filtering: only from primary doc
        elif primary_doc_id and hit_doc_id != primary_doc_id:
            logger.debug(f"Skipping schema - not from primary doc: {hit.get('schema_id')} (doc={hit_doc_id})")
            continue
        elif not primary_doc_id and anchor_doc_ids:
            if (hit.get("section_id") not in anchor_section_ids and 
                hit_doc_id not in anchor_doc_ids):
                logger.debug(f"Skipping schema - not in anchor docs/sections: {hit.get('schema_id')}")
                continue
        
        item = await _fetch_schema_full(driver, hit)
        if item:
            enriched.append(item)
    
    # Process Neo4j results (direct Cypher query results + entity search)
    # These are HIGH PRIORITY - user explicitly asked for this data via structured query
    # Neo4j entity search results are ALWAYS included (no anchor filtering)
    # but LIMITED to avoid context pollution
    
    # Separate Neo4j results by type and source
    neo4j_sections = []
    neo4j_tables = []
    neo4j_schemas = []
    
    for record in neo4j_results:
        if "error" in record:
            continue
        
        # Try to identify type from record
        if "table_id" in record or "id" in record and "page_number" in record:
            neo4j_tables.append(record)
        elif "section_id" in record or "content" in record:
            neo4j_sections.append(record)
        elif "schema_id" in record:
            neo4j_schemas.append(record)
    
    # LIMIT Neo4j entity sections: keep top 3 by importance
    # Sort by score (importance from entity search or default 0.5)
    neo4j_sections.sort(key=lambda r: r.get("importance", r.get("score", 0.5)), reverse=True)
    limited_sections = neo4j_sections[:3]  # Maximum 3 entity sections
    
    if len(neo4j_sections) > 3:
        logger.info(f"⚠️ Neo4j entity sections limited: {len(neo4j_sections)} → 3 (kept highest importance)")
    
    # Add limited sections
    for record in limited_sections:
        item = await _neo4j_record_to_text_chunk(driver, record)
        if item:
            enriched.append(item)
    
    # Add tables (already limited in entity search to 3)
    for record in neo4j_tables:
        item = await _neo4j_record_to_table(driver, record)
        if item:
            enriched.append(item)
            logger.info(f"Added table from neo4j: {item.get('table_id')} (p{item.get('page')})")
    
    # Add schemas (already limited in entity search to 3)
    for record in neo4j_schemas:
        item = await _neo4j_record_to_schema(driver, record)
        if item:
            enriched.append(item)
    
    # Deduplicate by ID
    seen_ids = set()
    deduplicated = []
    
    for item in enriched:
        item_id = item.get("section_id") or item.get("table_id") or item.get("schema_id")
        if item_id and item_id not in seen_ids:
            seen_ids.add(item_id)
            deduplicated.append(item)
    
    # FILTER: Remove junk sections (TABLE OF CONTENTS, INDEX, etc.)
    filtered = []
    junk_count = 0
    
    for item in deduplicated:
        if item["type"] == "text_chunk":
            section_title = item.get("section_title", "")
            if _is_junk_section(section_title):
                junk_count += 1
                logger.debug(f"Filtered junk section: {section_title}")
                continue
        filtered.append(item)
    
    if junk_count > 0:
        logger.info(f"🗑️  Filtered {junk_count} junk sections (TOC, INDEX, etc.)")
    
    # DOCUMENT-LEVEL RE-RANKING: Boost items from most relevant documents
    question = state["question"]
    
    # Calculate relevance score for each document
    doc_ids = list(set(item.get("doc_id") for item in filtered if item.get("doc_id")))
    doc_relevance = {}
    
    for doc_id in doc_ids:
        doc_relevance[doc_id] = _calculate_document_relevance(question, doc_id, filtered)
    
    # Apply document boost to item scores
    for item in filtered:
        doc_id = item.get("doc_id")
        if doc_id and doc_id in doc_relevance:
            original_score = item.get("score", 0.5)
            multiplier = doc_relevance[doc_id]
            item["score"] = min(1.0, original_score * multiplier)
            item["doc_boost"] = multiplier  # Store for debugging
    
    if doc_relevance:
        logger.info(f"📊 Document-level boost applied: {doc_relevance}")
    
    # Sort by boosted score
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # HARD LIMITS: adaptive based on content type AND query intent
    sections = [i for i in filtered if i["type"] == "text_chunk"]
    tables = [i for i in filtered if i["type"] == "table_chunk"]
    schemas = [i for i in filtered if i["type"] == "schema"]
    
    # Get query intent from state
    query_intent = state.get("query_intent", "text")
    
    # Check which tools were called (need this BEFORE setting limits)
    tools_called = state.get("tool_names", [])
    table_tool_called = "qdrant_search_tables" in tools_called
    schema_tool_called = "qdrant_search_schemas" in tools_called
    
    # Adaptive limits based on intent AND tool usage
    # KEY FIX: If tools called tables/schemas, don't limit them to 0
    if query_intent == "table":
        # Table-focused query: prioritize tables
        max_sections = 3
        max_tables = 5
        max_schemas = 1
    elif query_intent == "schema":
        # Schema-focused query: prioritize schemas
        max_sections = 2
        max_tables = 0 if not table_tool_called else 3  # Allow tables if tool called
        max_schemas = 5
    elif query_intent == "mixed":
        # Mixed query: allow both tables and schemas
        max_sections = 4
        max_tables = 3
        max_schemas = 2
    else:
        # Text query: Default no tables/schemas
        # BUT: if tools explicitly called them, allow limited number
        max_sections = 5
        max_tables = 3 if table_tool_called else 0     # Allow if tool called
        max_schemas = 2 if schema_tool_called else 0   # Allow if tool called
    
    # Log BEFORE trimming
    logger.info(
        f"📊 Before limits: sections={len(sections)}, tables={len(tables)}, schemas={len(schemas)}"
    )
    logger.info(
        f"📊 Applying limits: max_sections={max_sections}, max_tables={max_tables}, max_schemas={max_schemas}"
    )
    logger.info(
        f"📊 Tool calls: table_tool_called={table_tool_called}, schema_tool_called={schema_tool_called}"
    )
    
    sections = sections[:max_sections]
    tables = tables[:max_tables]
    schemas = schemas[:max_schemas]
    
    # Log AFTER trimming
    logger.info(
        f"📊 After limits: sections={len(sections)}, tables={len(tables)}, schemas={len(schemas)}"
    )
    
    # Log what we kept after limits
    if table_tool_called and len(tables) > 0:
        logger.info(f"Intent={query_intent} but qdrant_search_tables called → keeping {len(tables)} tables")
    elif table_tool_called and len(tables) == 0:
        logger.warning(f"⚠️ qdrant_search_tables called but no tables found in results")
    
    if schema_tool_called and len(schemas) > 0:
        logger.info(f"Intent={query_intent} but qdrant_search_schemas called → keeping {len(schemas)} schemas")
    elif schema_tool_called and len(schemas) == 0:
        logger.warning(f"⚠️ qdrant_search_schemas called but no schemas found in results")
    
    # Note: Intent-based filtering now handled by adaptive limits above
    # No need for additional stripping - limits already respect tool calls
    
    final_context = sections + tables + schemas
    
    state["enriched_context"] = final_context
    
    logger.info(
        f"Context built (intent={query_intent}): "
        f"{len(sections)}/{max_sections} sections, "
        f"{len(tables)}/{max_tables} tables, "
        f"{len(schemas)}/{max_schemas} schemas "
        f"(total: {len(final_context)})"
    )
    
    return state


async def _fetch_and_expand_text_chunk(
    driver: Driver,
    hit: Dict[str, Any],
    vector_service,
    expand_neighbors: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Fetch text chunk from Qdrant and optionally expand with neighbor chunks.
    
    :param driver: Neo4j driver (not used, kept for compatibility)
    :param hit: Qdrant search hit with chunk data
    :param vector_service: VectorService instance for fetching neighbors
    :param expand_neighbors: Whether to include ±1 neighbor chunks
    """
    chunk_text = hit.get("text")
    
    if not chunk_text:
        logger.error(f"Missing 'text' field in Qdrant payload for section {hit.get('section_id')}")
        return None
    
    # If expansion disabled, return just this chunk
    if not expand_neighbors:
        return {
            "type": "text_chunk",
            "section_id": hit["section_id"],
            "doc_id": hit.get("doc_id"),
            "doc_title": hit.get("doc_title", "Unknown"),
            "section_title": hit.get("section_title", ""),
            "page": hit.get("page_start"),
            "text": chunk_text,
            "score": hit.get("score", 0),
            "chunk_index": hit.get("chunk_index", 0),
            "expanded": False,
        }
    
    # Fetch neighbor chunks (±1) from Qdrant
    section_id = hit["section_id"]
    chunk_index = hit.get("chunk_index", 0)
    
    try:
        neighbor_chunks = await vector_service.get_neighbor_chunks(
            section_id=section_id,
            chunk_index=chunk_index,
            neighbor_range=1,  # ±1 chunk
        )
        
        if not neighbor_chunks:
            # No neighbors found, return just this chunk
            logger.debug(f"No neighbor chunks found for {section_id}[{chunk_index}]")
            return {
                "type": "text_chunk",
                "section_id": section_id,
                "doc_id": hit.get("doc_id"),
                "doc_title": hit.get("doc_title", "Unknown"),
                "section_title": hit.get("section_title", ""),
                "page": hit.get("page_start"),
                "text": chunk_text,
                "score": hit.get("score", 0),
                "chunk_index": chunk_index,
                "expanded": False,
            }
        
        # Combine neighbor chunks WITHOUT overlap duplication
        # Sort by chunk_char_start to ensure correct order
        neighbor_chunks.sort(key=lambda x: x["chunk_char_start"])
        
        combined_parts = []
        last_end = 0
        
        for chunk in neighbor_chunks:
            start = chunk["chunk_char_start"]
            end = chunk["chunk_char_end"]
            text = chunk["text"]
            
            # Skip overlap region (if this chunk starts before last one ended)
            if start < last_end:
                # Calculate how much to skip from beginning of this chunk
                overlap_size = last_end - start
                if overlap_size < len(text):
                    # Skip overlap, take rest of chunk
                    combined_parts.append(text[overlap_size:])
                # else: completely overlapped, skip this chunk
            else:
                # No overlap, add full chunk
                combined_parts.append(text)
            
            last_end = max(last_end, end)
        
        combined_text = "".join(combined_parts)  # No separator needed, already continuous
        chunk_indices = [c["chunk_index"] for c in neighbor_chunks]
        
        logger.debug(
            f"Expanded chunk {section_id}[{chunk_index}]: "
            f"combined {len(neighbor_chunks)} chunks (indices: {chunk_indices}), "
            f"removed overlap, total {len(combined_text)} chars"
        )
        
        return {
            "type": "text_chunk",
            "section_id": section_id,
            "doc_id": hit.get("doc_id"),
            "doc_title": hit.get("doc_title", "Unknown"),
            "section_title": hit.get("section_title", ""),
            "page": hit.get("page_start"),
            "text": combined_text,
            "score": hit.get("score", 0),
            "chunk_index": chunk_index,
            "expanded": True,
            "expansion_info": {
                "chunk_count": len(neighbor_chunks),
                "chunk_indices": chunk_indices,
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching neighbor chunks: {e}")
        # Fallback to just this chunk
        return {
            "type": "text_chunk",
            "section_id": section_id,
            "doc_id": hit.get("doc_id"),
            "doc_title": hit.get("doc_title", "Unknown"),
            "section_title": hit.get("section_title", ""),
            "page": hit.get("page_start"),
            "text": chunk_text,
            "score": hit.get("score", 0),
            "chunk_index": chunk_index,
            "expanded": False,
        }


async def _fetch_table_full(
    driver: Driver,
    hit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Fetch full table text from Neo4j.
    
    The Table node already stores the complete linearized table text in normalized_text field.
    No need to read CSV files or fetch chunks from Qdrant.
    """
    async with driver.session() as session:
        # Get table with full text
        query = """
        MATCH (t:Table {id: $table_id})
        OPTIONAL MATCH (doc:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)-[:CONTAINS_TABLE]->(t)
        RETURN 
            t.id AS table_id,
            t.title AS table_title,
            t.caption AS caption,
            t.page_number AS page,
            t.rows AS rows,
            t.cols AS cols,
            t.file_path AS file_path,
            t.normalized_text AS table_text,
            t.doc_id AS doc_id,
            doc.title AS doc_title,
            s.title AS section_title
        """
        
        result = await session.run(query, table_id=hit["table_id"])
        record = await result.single()
        
        if not record:
            logger.warning(f"Table not found in Neo4j: {hit.get('table_id')}")
            # Fallback - use hit data (may come from entity_search with file_path)
            return {
                "type": "table_chunk",
                "table_id": hit.get("table_id"),
                "doc_id": hit.get("doc_id"),
                "doc_title": hit.get("doc_title", "Unknown"),
                "title": hit.get("table_title", ""),
                "caption": hit.get("table_caption", ""),
                "page": hit.get("page"),
                "file_path": hit.get("file_path"),
                "text": hit.get("text_preview", ""),
                "score": hit.get("score", 0),
                "source": hit.get("source"),
            }
        
        # Get full table text from normalized_text field
        combined_text = record.get("table_text") or ""
        file_path = record.get("file_path") or hit.get("file_path")
        
        if not file_path:
            logger.warning(f"Table {hit.get('table_id')} has no file_path in Neo4j or hit")
        
        if not combined_text:
            logger.warning(f"⚠️ Table {hit['table_id']} has no normalized_text - table may be empty")
        else:
            # Log table text length and preview
            preview = combined_text[:200].replace('\n', ' ')
            logger.info(
                f"📊 Fetched table {hit['table_id']}: {len(combined_text)} chars, "
                f"{record['rows']}x{record['cols']}, preview: {preview}..."
            )
        
        return {
            "type": "table_chunk",
            "table_id": record["table_id"],
            "doc_id": record["doc_id"] or hit.get("doc_id"),
            "doc_title": record.get("doc_title") or hit.get("doc_title", "Unknown"),
            "section_title": record.get("section_title"),
            "title": record["table_title"] or hit.get("table_title", ""),
            "caption": record.get("caption") or hit.get("caption", ""),
            "page": record["page"],
            "rows": record["rows"],
            "cols": record["cols"],
            "file_path": file_path,
            "text": combined_text,
            "score": hit.get("score", 0),
            "source": hit.get("source"),
        }


async def _fetch_schema_full(
    driver: Driver,
    hit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Fetch full schema metadata including text_context and llm_summary"""
    async with driver.session() as session:
        query = """
        MATCH (sc:Schema {id: $schema_id})
        OPTIONAL MATCH (doc:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)-[:CONTAINS_SCHEMA]->(sc)
        RETURN 
            sc.id AS schema_id,
            sc.title AS title,
            sc.caption AS caption,
            sc.text_context AS text_context,
            sc.llm_summary AS llm_summary,
            sc.page_number AS page,
            sc.file_path AS file_path,
            sc.thumbnail_path AS thumbnail_path,
            sc.doc_id AS doc_id,
            doc.title AS doc_title,
            s.title AS section_title
        LIMIT 1
        """
        
        result = await session.run(query, schema_id=hit["schema_id"])
        records = [rec async for rec in result]
        record = records[0] if records else None
        
        if not record:
            return {
                "type": "schema",
                "schema_id": hit.get("schema_id"),
                "doc_id": hit.get("doc_id"),
                "doc_title": hit.get("doc_title", "Unknown"),
                "title": hit.get("title", ""),
                "caption": hit.get("caption", ""),
                "text_context": hit.get("text_context", ""),
                "llm_summary": hit.get("llm_summary", ""),
                "page": hit.get("page"),
                "file_path": hit.get("file_path"),
                "score": hit.get("score", 0),
            }
        
        doc_title = record.get("doc_title")
        if not doc_title and record.get("doc_id"):
            # Fallback: direct doc lookup
            doc_query = "MATCH (d:Document {id: $doc_id}) RETURN d.title AS title"
            doc_result = await session.run(doc_query, doc_id=record["doc_id"])
            doc_record = await doc_result.single()
            if doc_record:
                doc_title = doc_record["title"]
        
        return {
            "type": "schema",
            "schema_id": record["schema_id"],
            "doc_id": record["doc_id"],
            "doc_title": doc_title or "Unknown",
            "section_title": record.get("section_title"),
            "title": record["title"],
            "caption": record.get("caption", ""),
            "text_context": record.get("text_context", ""),
            "llm_summary": record.get("llm_summary", ""),
            "page": record["page"],
            "file_path": record["file_path"],
            "thumbnail_path": record.get("thumbnail_path"),
            "score": hit.get("score", 0),
        }


async def _neo4j_record_to_text_chunk(driver: Driver, record: Dict) -> Optional[Dict[str, Any]]:
    """Convert Neo4j record to text chunk format"""
    if "section_id" in record:
        return {
            "type": "text_chunk",
            "section_id": record.get("section_id"),
            "doc_id": record.get("doc_id"),
            "doc_title": record.get("doc_title", "Unknown"),
            "section_title": record.get("section_title", ""),
            "page": record.get("page_start") or record.get("page"),
            "text": record.get("content", ""),
            "score": 0.5,  # Medium score - allow qdrant to override if more relevant
        }
    return None


async def _neo4j_record_to_table(driver: Driver, record: Dict) -> Optional[Dict[str, Any]]:
    """Convert Neo4j record to table format"""
    # Support both "table_id" and "id" keys (from different queries)
    table_id = record.get("table_id") or record.get("id")
    if table_id:
        return {
            "type": "table_chunk",
            "table_id": table_id,
            "doc_id": record.get("doc_id"),
            "doc_title": record.get("doc_title", "Unknown"),
            "title": record.get("table_title") or record.get("title", ""),
            "caption": record.get("caption", ""),
            "page": record.get("page_number") or record.get("page"),
            "file_path": record.get("file_path"),
            "text": record.get("text_preview", ""),
            "score": 0.5,  # Medium score - allow qdrant to override if more relevant
        }
    return None


async def _neo4j_record_to_schema(driver: Driver, record: Dict) -> Optional[Dict[str, Any]]:
    """Convert Neo4j record to schema format"""
    if "schema_id" in record:
        return {
            "type": "schema",
            "schema_id": record.get("schema_id"),
            "doc_id": record.get("doc_id"),
            "doc_title": record.get("doc_title", "Unknown"),
            "title": record.get("title", ""),
            "caption": record.get("caption", ""),
            "page": record.get("page_number") or record.get("page"),
            "file_path": record.get("file_path"),
            "score": 0.5,  # Medium score - allow qdrant to override if more relevant
        }
    return None


# ============================================================================
# ADAPTIVE RETRY HELPERS: Answer quality check + fallback search
# ============================================================================

def check_answer_quality(
    answer_text: str,
    question: str,
    context_items: List[Dict[str, Any]]
) -> tuple[bool, str]:
    """
    Check if answer quality is acceptable or if retry needed.
    
    Returns:
        (needs_retry: bool, reason: str)
    
    Heuristics:
    1. Explicit "not found" markers → retry
    2. Answer too short (<100 chars) → retry
    3. Low context usage (few references) → retry
    """
    answer_lower = answer_text.lower()
    
    # Heuristic 1: Explicit "not found" markers
    not_found_markers = [
        "does not contain",
        "documentation does not",
        "not found in",
        "no information",
        "not available",
        "not specified",
        "not mentioned",
        "not documented",
        "information is missing",
        "insufficient context"
    ]
    
    for marker in not_found_markers:
        if marker in answer_lower:
            return (True, "explicit_not_found")
    
    # Heuristic 2: Answer too short (excluding greetings/short questions)
    # Skip if question is greeting or very short
    question_lower = question.lower()
    greeting_words = ["hello", "hi", "hey", "thanks", "thank you"]
    is_greeting = any(word in question_lower for word in greeting_words)
    
    if not is_greeting and len(answer_text.strip()) < 100:
        # Check if it's a generic "no info" response
        if any(marker in answer_lower for marker in ["no", "not", "unable"]):
            return (True, "answer_too_short")
    
    # Heuristic 3: Low context usage
    # Count how many context items are referenced (citations, [T1], [TABLE1], etc.)
    citation_pattern = r'\[T\d+\]|\[TABLE\d+\]|\[DIAGRAM\d+\]'
    import re
    references = re.findall(citation_pattern, answer_text)
    
    # If we have context but LLM didn't use much of it
    if len(context_items) >= 3 and len(references) <= 1:
        # Check if answer is substantive (not just "no info")
        if len(answer_text.strip()) < 200:
            return (True, "low_context_usage")
    
    # All checks passed
    return (False, "")


async def adaptive_fallback_search(
    question: str,
    retry_reason: str,
    existing_context: List[Dict[str, Any]],
    driver: Driver,
    qdrant_client: QdrantClient,
    query_intent: str
) -> List[Dict[str, Any]]:
    """
    Perform adaptive fallback search based on why first attempt failed.
    
    Strategies:
    - explicit_not_found → query expansion + Neo4j fulltext
    - answer_too_short → fetch neighbor sections from existing context
    - low_context_usage → semantic search with rephrased query
    
    Returns:
        List of additional context items
    """
    logger.info(f"🔍 Adaptive fallback: reason={retry_reason}, intent={query_intent}")
    
    additional_context = []
    
    try:
        if retry_reason == "explicit_not_found":
            # Strategy: Query expansion + Neo4j fulltext search
            # LLM said "not found" → maybe query was too specific or wrong keywords
            
            # Expand query with related terms
            expanded_queries = _expand_query(question)
            logger.info(f"   Query expansion: {expanded_queries[:3]}")
            
            # Try Neo4j fulltext with expanded terms
            for expanded_q in expanded_queries[:2]:  # Try top 2 variations
                fulltext_results = await neo4j_fulltext_search(
                    search_term=expanded_q,
                    limit=3,
                    min_score=0.4,  # Lower threshold
                    include_content=True
                )
                
                if fulltext_results:
                    logger.info(f"   Fulltext found {len(fulltext_results)} sections for: {expanded_q}")
                    
                    for result in fulltext_results:
                        section_id = result.get("section_id")
                        if not section_id:
                            continue
                        
                        # Check if not already in context
                        if any(item.get("section_id") == section_id for item in existing_context):
                            continue
                        
                        additional_context.append({
                            "type": "text_chunk",
                            "section_id": section_id,
                            "doc_id": result.get("doc_id"),
                            "doc_title": result.get("doc_title", "Unknown"),
                            "section_title": result.get("title", ""),
                            "page": result.get("page_start"),
                            "text": result.get("content", ""),
                            "score": result.get("score", 0.5),
                            "source": "adaptive_fulltext"
                        })
                    
                    if additional_context:
                        break  # Found something, stop expanding
            
            # Also try tables if intent=mixed (troubleshooting often needs tables)
            if query_intent == "mixed" and not any(item["type"] == "table_chunk" for item in existing_context):
                logger.info(f"   Mixed intent + not_found → searching tables")
                
                # Get embedding
                embeddings = OpenAIEmbeddings(
                    model=settings.openai_embedding_model,
                    api_key=settings.openai_api_key
                )
                query_vector = embeddings.embed_query(question)
                
                # Search tables with lower threshold
                table_results = qdrant_client.search(
                    collection_name=settings.tables_text_collection,
                    query_vector=query_vector,
                    limit=3,
                    with_payload=True,
                    score_threshold=0.25,  # Lower threshold for fallback
                )
                
                for hit in table_results:
                    additional_context.append({
                        "type": "table_chunk",
                        "table_id": hit.payload.get("table_id"),
                        "doc_id": hit.payload.get("doc_id"),
                        "doc_title": hit.payload.get("doc_title", "Unknown"),
                        "page": hit.payload.get("page"),
                        "title": hit.payload.get("table_title", ""),
                        "caption": hit.payload.get("table_caption", ""),
                        "text": hit.payload.get("text_preview", ""),
                        "score": float(hit.score),
                        "source": "adaptive_tables"
                    })
        
        elif retry_reason == "answer_too_short":
            # Strategy: Fetch neighbor sections from existing context
            # Maybe we had the right section but not enough surrounding context
            
            logger.info(f"   Fetching neighbor sections from existing context")
            
            # Get section_ids from existing context
            section_ids = [
                item.get("section_id")
                for item in existing_context
                if item.get("type") == "text_chunk" and item.get("section_id")
            ]
            
            if section_ids:
                # Query Neo4j for neighboring sections (same chapter)
                async with driver.session() as session:
                    query = """
                    UNWIND $section_ids AS sid
                    MATCH (s:Section {id: sid})
                    MATCH (c:Chapter)-[:HAS_SECTION]->(s)
                    MATCH (c)-[:HAS_SECTION]->(neighbor:Section)
                    WHERE neighbor.id <> sid
                      AND abs(neighbor.section_number - s.section_number) <= 1
                    OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(c)
                    RETURN DISTINCT
                        neighbor.id AS section_id,
                        neighbor.title AS title,
                        neighbor.content AS content,
                        neighbor.page_start AS page_start,
                        neighbor.doc_id AS doc_id,
                        d.title AS doc_title
                    LIMIT 3
                    """
                    
                    result = await session.run(query, {"section_ids": section_ids})
                    neighbors = await result.data()
                    
                    logger.info(f"   Found {len(neighbors)} neighbor sections")
                    
                    for neighbor in neighbors:
                        # Check not already in context
                        neighbor_id = neighbor.get("section_id")
                        if any(item.get("section_id") == neighbor_id for item in existing_context):
                            continue
                        
                        additional_context.append({
                            "type": "text_chunk",
                            "section_id": neighbor_id,
                            "doc_id": neighbor.get("doc_id"),
                            "doc_title": neighbor.get("doc_title", "Unknown"),
                            "section_title": neighbor.get("title", ""),
                            "page": neighbor.get("page_start"),
                            "text": neighbor.get("content", ""),
                            "score": 0.5,
                            "source": "adaptive_neighbors"
                        })
        
        elif retry_reason == "low_context_usage":
            # Strategy: Semantic search with rephrased query
            # Maybe original query phrasing didn't match well
            
            logger.info(f"   Rephrasing query for semantic search")
            
            # Simple rephrasing: extract key nouns/verbs
            rephrased = _rephrase_query(question)
            logger.info(f"   Rephrased: {rephrased}")
            
            # Get embedding
            embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key
            )
            query_vector = embeddings.embed_query(rephrased)
            
            # Search with lower threshold
            results = qdrant_client.search(
                collection_name=settings.text_chunks_collection,
                query_vector=query_vector,
                limit=5,
                with_payload=True,
                score_threshold=0.25,  # Lower for fallback
            )
            
            for hit in results:
                section_id = hit.payload.get("section_id")
                # Check not already in context
                if any(item.get("section_id") == section_id for item in existing_context):
                    continue
                
                additional_context.append({
                    "type": "text_chunk",
                    "section_id": section_id,
                    "doc_id": hit.payload.get("doc_id"),
                    "doc_title": hit.payload.get("doc_title", "Unknown"),
                    "section_title": hit.payload.get("section_title", ""),
                    "page": hit.payload.get("page_start"),
                    "text": hit.payload.get("text", ""),
                    "score": float(hit.score),
                    "source": "adaptive_rephrase"
                })
    
    except Exception as e:
        logger.error(f"Adaptive fallback failed: {e}", exc_info=True)
    
    return additional_context


def _expand_query(question: str) -> List[str]:
    """
    Expand query with synonyms and related terms.
    Simple rule-based expansion for maritime domain.
    
    Returns list of expanded query variations.
    """
    expansions = [question]  # Original query first
    
    # Maritime synonym mappings
    synonyms = {
        "pump": ["pump", "pumping unit", "circulation pump"],
        "valve": ["valve", "cut cock", "shut-off valve"],
        "failure": ["failure", "malfunction", "fault", "problem", "issue"],
        "cause": ["cause", "reason", "source", "origin"],
        "suction": ["suction", "intake", "inlet"],
        "pressure": ["pressure", "head", "discharge pressure"],
        "temperature": ["temperature", "heat", "thermal"],
        "fuel": ["fuel", "fuel oil", "FO", "diesel oil", "DO"],
        "cooling": ["cooling", "cooler", "heat exchanger"],
    }
    
    # Replace key terms with synonyms
    import re
    question_lower = question.lower()
    
    for term, variants in synonyms.items():
        if term in question_lower:
            # Create variant with first synonym
            if len(variants) > 1:
                expanded = re.sub(
                    r'\b' + term + r'\b',
                    variants[1],
                    question_lower,
                    flags=re.IGNORECASE
                )
                expansions.append(expanded)
                break  # One expansion enough
    
    return expansions


def _rephrase_query(question: str) -> str:
    """
    Rephrase query by extracting key terms.
    Simple implementation: remove question words, keep nouns/technical terms.
    """
    import re
    
    # Remove question words
    stop_words = ["what", "how", "why", "when", "where", "which", "who",
                  "is", "are", "does", "do", "can", "the", "a", "an"]
    
    words = question.lower().split()
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Keep technical terms (uppercase, numbers, hyphens)
    technical_terms = re.findall(r'\b[A-Z]{2,}[-]?\d*\b|\b\d+[A-Z]+\b', question)
    
    rephrased_words = filtered + [t.lower() for t in technical_terms]
    
    return " ".join(rephrased_words[:10])  # Limit to 10 words


def _is_junk_section(section_title: str) -> bool:
    """
    Filter out junk sections that don't contain useful content.
    
    Junk patterns:
    - TABLE OF CONTENTS, CONTENTS, INDEX
    - ABBREVIATIONS, GLOSSARY
    - COPYRIGHT, FOREWORD, PREFACE
    - Empty or very short titles
    """
    if not section_title:
        return True
    
    title_lower = section_title.lower().strip()
    
    # Too short (likely auto-generated)
    if len(title_lower) < 3:
        return True
    
    # Junk keywords
    junk_keywords = [
        "table of contents",
        "contents",
        "index",
        "abbreviations",
        "glossary",
        "copyright",
        "foreword",
        "preface",
        "list of figures",
        "list of tables",
        "acknowledgment",
        "revision history",
    ]
    
    for keyword in junk_keywords:
        if keyword in title_lower:
            return True
    
    return False


def _calculate_document_relevance(
    question: str,
    doc_id: str,
    context_items: List[Dict[str, Any]]
) -> float:
    """
    Calculate document relevance score based on:
    1. Number of relevant items from this document
    2. Average score of items from this document
    3. Keyword overlap between question and document content
    
    Returns score multiplier (0.8-1.2)
    """
    # Get items from this document
    doc_items = [item for item in context_items if item.get("doc_id") == doc_id]
    
    if not doc_items:
        return 1.0  # Neutral
    
    # Factor 1: Number of items (more items = more relevant)
    item_count = len(doc_items)
    count_score = min(1.0, item_count / 5.0)  # Cap at 5 items = 1.0
    
    # Factor 2: Average score of items
    scores = [item.get("score", 0.5) for item in doc_items]
    avg_score = sum(scores) / len(scores) if scores else 0.5
    
    # Factor 3: Keyword overlap
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    # Extract text from items
    doc_text = " ".join([
        item.get("text", "") or item.get("title", "") or item.get("caption", "")
        for item in doc_items
    ]).lower()
    
    doc_words = set(doc_text.split())
    
    # Calculate overlap (exclude stop words)
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
    question_words_filtered = question_words - stop_words
    
    if question_words_filtered:
        overlap = len(question_words_filtered & doc_words) / len(question_words_filtered)
    else:
        overlap = 0.5
    
    # Combined score: weighted average
    combined = (count_score * 0.3) + (avg_score * 0.4) + (overlap * 0.3)
    
    # Convert to multiplier: 0.8-1.2 range
    # combined 0.0-1.0 → multiplier 0.8-1.2
    multiplier = 0.8 + (combined * 0.4)
    
    return multiplier


# NODE 5: LLM REASONING

async def node_llm_reasoning(state: GraphState) -> GraphState:
    """
    Generate answer using enriched context.
    Context already includes expanded chunks from same sections.
    """
    llm = get_llm_instance(temperature=0)
    
    enriched_context = state["enriched_context"]
    chat_history = state.get("chat_history", [])
    
    # EMERGENCY FALLBACK: Only trigger if context is completely empty
    # NOTE: Scores are unreliable (entity_search = graph importance, not semantic similarity)
    # Trust anchor selection + entity search - they already found relevant content
    should_fallback = False
    fallback_reason = ""
    query_intent = state.get("query_intent", "text")
    
    if not enriched_context or len(enriched_context) == 0:
        should_fallback = True
        fallback_reason = "empty context"
        logger.warning(f"⚠️ Context is empty after filtering")
    else:
        # Context exists - trust the tools (Qdrant semantic + Neo4j graph relationships)
        # No score-based fallback - scores mix semantic similarity with graph importance
        logger.info(f"✅ Context ready: {len(enriched_context)} items (skipping score checks)")
    
    if should_fallback:
        logger.warning(f"⚠️ EMERGENCY FALLBACK triggered ({fallback_reason})! Running unfiltered semantic search...")
        
        question = state["question"]
        query_intent = state.get("query_intent", "text")
        
        # Run pure semantic search without anchor filtering
        try:
            emergency_results = []
            
            # Always get text
            text_hits = qdrant_search_text.invoke({"query": question, "limit": 5})
            logger.info(f"   Emergency text search: {len(text_hits)} results")
            
            # Fetch full chunks from Qdrant
            from services.vector_service import VectorService
            from services.embedding_service import EmbeddingService
            
            embedding_service = EmbeddingService(api_key=settings.openai_api_key)
            vector_service = VectorService(embedding_service=embedding_service)
            
            for hit in text_hits[:3]:
                section_id = hit.get("section_id")
                doc_id = hit.get("doc_id")
                chunk_index = hit.get("chunk_index", 0)
                
                # Fetch expanded content
                expanded = await vector_service.fetch_expanded_chunk(
                    section_id=section_id,
                    doc_id=doc_id, 
                    chunk_index=chunk_index,
                    expansion_window=2
                )
                
                if expanded:
                    expanded["score"] = hit.get("score", 0.5)
                    emergency_results.append(expanded)
            
            # Add tables/schemas if needed by intent
            if query_intent in ["table", "mixed"]:
                table_hits = qdrant_search_tables.invoke({"query": question, "limit": 3})
                logger.info(f"   Emergency table search: {len(table_hits)} results")
                emergency_results.extend(table_hits)
            
            if query_intent in ["schema", "mixed"]:
                schema_hits = qdrant_search_schemas.invoke({"query": question, "limit": 2})
                logger.info(f"   Emergency schema search: {len(schema_hits)} results")
                emergency_results.extend(schema_hits)
            
            enriched_context = emergency_results
            state["enriched_context"] = emergency_results
            
            # Calculate new scores for comparison
            new_scores = [item.get("score", 0) for item in emergency_results if item.get("type") == "text_chunk"]
            new_avg = sum(new_scores) / len(new_scores) if new_scores else 0
            new_max = max(new_scores) if new_scores else 0
            
            logger.info(
                f"✅ Emergency fallback: {len(emergency_results)} items retrieved "
                f"(avg={new_avg:.2f}, max={new_max:.2f})"
            )
        except Exception as e:
            logger.error(f"❌ Emergency fallback failed: {e}")
            # Continue with empty context - will return "not found"
    
    # Separate by type
    chunks = [c for c in enriched_context if c["type"] == "text_chunk"] if enriched_context else []
    tables = [c for c in enriched_context if c["type"] == "table_chunk"] if enriched_context else []
    schemas = [c for c in enriched_context if c["type"] == "schema"] if enriched_context else []
    
    # Build context text
    context_text = ""
    
    if chunks:
        context_text += "=== TEXT SECTIONS ===\n\n"
        for i, c in enumerate(chunks, 1):
            context_text += f"[T{i}] Document: {c.get('doc_title', 'Unknown')}\n"
            if c.get("chapter_title"):
                context_text += f"Chapter: {c['chapter_title']}\n"
            context_text += f"Section: {c.get('section_title', '')} (Page {c.get('page')})\n"
            if c.get("expanded"):
                context_text += f"[EXPANDED CONTEXT - includes neighboring content]\n"
            context_text += f"{c['text']}\n\n"
    

    
    schemas_text = ""
    schema_map = {}  # Map [DIAGRAM{i}] -> schema object
    if schemas:
        schemas_text += "=== AVAILABLE DIAGRAMS ===\n\n"
        for i, s in enumerate(schemas, 1):
            diagram_ref = f"[DIAGRAM{i}]"
            schema_map[diagram_ref] = s
            schemas_text += f"{diagram_ref} {s.get('title', 'Figure')}\n"
            schemas_text += f"Caption: {s.get('caption', '')}\n"
            if s.get('llm_summary'):
                schemas_text += f"Description: {s.get('llm_summary')}\n"
            schemas_text += f"Document: {s.get('doc_title', 'Unknown')} (Page {s.get('page')})\n\n"
    
    tables_text = ""
    table_map = {}  # Map [TABLE{i}] -> table object
    if tables:
        tables_text += "=== AVAILABLE TABLES ===\n\n"
        logger.info(f"🎯 Formatting {len(tables)} tables for LLM context...")
        for i, t in enumerate(tables, 1):
            table_ref = f"[TABLE{i}]"
            table_map[table_ref] = t
            table_text = t.get('text', '')
            
            # Log each table being added to context
            text_preview = table_text[:150].replace('\n', ' ') if table_text else 'EMPTY'
            logger.info(
                f"  {table_ref} '{t.get('title', 'Untitled')}': "
                f"{len(table_text)} chars, preview: {text_preview}..."
            )
            
            tables_text += f"{table_ref} {t.get('title', 'Untitled')}\n"
            tables_text += f"Caption: {t.get('caption', '')}\n"
            tables_text += f"Document: {t.get('doc_title', 'Unknown')} (Page {t.get('page')})\n"
            # Show FULL table data for complete extraction (not preview)
            tables_text += f"Complete table data:\n{table_text}\n\n"
    
    # System prompt with few-shot examples to reduce regeneration
    system_prompt = """You are a marine technical documentation answer generator.
Your only role is to produce factual answers strictly derived from supplied documentation.

⚠️ CRITICAL ANTI-HALLUCINATION RULES:

1. **QUOTE-FIRST APPROACH**: Before writing your answer, mentally identify the EXACT sentences/data from context that support each statement.

2. **FORBIDDEN ADDITIONS**:
   ❌ Do NOT add technical details not present in context
   ❌ Do NOT infer specifications or parameters
   ❌ Do NOT assume causes, reasons, or procedures
   ❌ Do NOT use general maritime knowledge - ONLY provided documents
   ❌ Do NOT elaborate beyond what documentation explicitly states

3. **VERIFICATION CHECK**: Before outputting, ask yourself:
   "Can I point to the EXACT location in the provided context for EVERY claim I make?"
   If answer is NO → remove that claim or rephrase as uncertainty.

4. **UNCERTAINTY HANDLING**:
   - If context is incomplete: State what IS documented, then add "Additional details not found in documentation."
   - If context is ambiguous: Quote the relevant part and acknowledge ambiguity
   - If context is missing: Output "The provided documentation does not contain this information."

5. **SPECIFICITY CONSTRAINT**:
   - Use EXACT values/numbers from context (don't round or approximate)
   - Use EXACT terminology from documents (don't paraphrase technical terms)
   - If document says "approximately", include "approximately" in answer

🔴 TABLE DATA EXTRACTION RULES (CRITICAL):

When extracting causes/solutions/specifications from tables:

1. **READ THE COMPLETE TABLE** - You are given FULL table data, not preview
   - Extract ALL relevant rows (all causes, all solutions, all parameters)
   - Maintain exact structure (symptom → cause → remedy)

2. **LIST FORMAT for multiple items**:
   ✅ CORRECT: "Possible causes: 1) Air leak in suction line, 2) Clogged strainer, 3) Worn impeller, 4) Insufficient NPSH, 5) Closed suction valve, 6) Wrong rotation direction..."
   ❌ WRONG: "Causes may include air leaks or clogged strainers." (incomplete - missing other causes!)
   ❌ WRONG: "The causes are not shown in the provided excerpt." (data IS in table!)

3. **CAUSE → SOLUTION pairing**: If table shows both causes AND remedies
   - List each cause with its corresponding action
   - Don't list only causes when solutions are available

4. **NEVER say "not shown" if data IS in table**:
   ❌ FORBIDDEN: "The specific causes are not shown in the provided excerpt."
   ❌ FORBIDDEN: "Additional details not found in documentation." (when they ARE in table)
   ✅ CORRECT: Extract and list ALL items from table

5. **COMPLETENESS CHECK**: 
   - If troubleshooting table has 15 rows → your answer must include ALL 15 causes/solutions
   - Don't truncate, don't summarize - list everything
   - Count items in table and verify your answer includes them all

LANGUAGE POLICY:
- Respond in exactly the same language as the user's question.

ANSWER BEHAVIOUR:
- Output must be factual, declarative, concise, and self-contained.
- Never ask user questions or offer cooperation, follow-ups, advice, or diagnostics.
- Every sentence must be traceable to provided context.

STRUCTURE AND RESOURCE SELECTION:
- You will be given AVAILABLE DIAGRAMS and AVAILABLE TABLES.
- Include a diagram/table only if it directly supports the answer.

📊 TABLE/DIAGRAM REFERENCE FORMAT:
When user wants to SEE table/diagram (intent=table/schema):
✅ NATURAL: "The specifications are shown in the table below [TABLE1]."
✅ NATURAL: "The system layout is shown below [DIAGRAM1]."
❌ AWKWARD: "According to [TABLE1], the specifications are..."
❌ AWKWARD: "As shown in [DIAGRAM1], the layout depicts..."

When extracting DATA from table (intent=text):
✅ NO REFERENCE: Just write the facts in text form
❌ DON'T: "According to [TABLE1], the causes are..." (intent is TEXT, not table display)

INTENT-BASED RESOURCE CONSTRAINTS (CRITICAL):

- query_intent="text" → Answer in TEXT format (even if data source is table)
  ✅ Extract information from tables and explain in sentences
  ✅ You MAY reference tables/diagrams for additional context, but not required
  ❌ Don't just say "see [TABLE1]" - user wants explanation in words

- query_intent="table" → User wants to SEE the table (display request)
  ✅ You MUST include at least one [TABLE] reference
  ✅ Brief explanation + [TABLE1] reference inline
  Example: "The specifications are shown below [TABLE1]."

- query_intent="schema" → User wants to SEE the diagram (display request)
  ✅ You MUST include at least one [DIAGRAM] reference
  ✅ Brief explanation + [DIAGRAM1] reference inline
  Example: "The system layout is shown below [DIAGRAM1]."

- query_intent="mixed" → Troubleshooting/complex query needing multiple sources
  ✅ Extract data from tables into text format
  ✅ Combine with procedural text from text chunks
  ✅ Provide complete answer with all causes/solutions/procedures
  ❌ Don't just reference [TABLE1] - extract the data!

⚡ FEW-SHOT EXAMPLES (learn from these to avoid regeneration):

Example 1 (intent=mixed - troubleshooting with complete table extraction):
Q: "The pump has no suction. What can be the cause?"
A: According to the troubleshooting table for the W.O. dosing pump, loss of suction can be caused by:
   1) Excessive adhesion between rotor and stator
   2) Wrong direction of rotation
   3) Leaks in the suction pipe or shaft sealing
   4) Suction head being too high
   5) Liquid viscosity being too high
   6) Incorrect pump speed
   7) Air inclusions in the conveyed liquid
   8) Discharge pressure head being too high
   9) Pump running partially or completely dry
   10) Misaligned or worn coupling components
   11) Excessive axial play in the linkage
   12) Foreign substances inside the pump
   13) Worn stator or rotor
   14) Partially or completely blocked suction pipework
   15) Pumping liquid temperature too high causing stator expansion
   
   Check each of these potential causes systematically and apply corresponding remedies from the troubleshooting guide.

Note: This extracts COMPLETE data from table (all 15+ causes), not "causes not shown"
⚠️ IMPORTANT: NO inline page references like "Page 101" - citations added automatically!

Example 2 (intent=table - user wants to SEE specs):
Q: "What are the specifications of pump PU3?"
A: The PU3 pump specifications are shown in the table below [TABLE1].

Example 3 (intent=schema - user wants to SEE diagram):
Q: "Show me the cooling system diagram"
A: The cooling system layout is shown below [DIAGRAM1].

Example 4 (intent=text - descriptive explanation):
Q: "How does the fuel injection system work?"
A: The fuel injection system operates by pressurizing fuel through a high-pressure pump, delivering it to individual injectors at precise timing controlled by the ECU.

Example 5 (intent=table - show troubleshooting table):
Q: "Show me the troubleshooting table for incinerator alarms"
A: The troubleshooting guide for incinerator alarms is shown below [TABLE2].

KEY PATTERNS:
- intent=text (troubleshooting/explanation) → Extract info, write as TEXT. NO need for [TABLE1] reference
- intent=table (wants to SEE table) → Write brief intro + [TABLE1] reference inline
- intent=schema (wants to SEE diagram) → Write brief intro + [DIAGRAM1] reference inline

🚨 CRITICAL RULE: NO INLINE PAGE NUMBERS IN YOUR ANSWER TEXT
❌ NEVER write: "see page 26", "pages 26-27", "on page 45", "(page 67)", "refer to page..."
❌ NEVER write: "as shown on page X", "described in pages Y-Z", "steps on page..."
✅ ALWAYS: Write ONLY factual content WITHOUT page numbers - citations added automatically

WHY: Inline page refs cause faithfulness failures. Context shows "page 26" but actual citation 
may be "page 95", creating direct contradiction between your text and sources.

📋 CITATION FORMAT (CRITICAL):
Format: [Document Name | Page X]

✅ CORRECT:
   [Incinerator Manual | Page 42]
   [Engine Manual | Page 67]

❌ WRONG (too verbose):
   [Incinerator Manual | Troubleshooting Guide | Page 42]
   [Engine Manual | Fuel System Section | Page 67]

CITATION RULES:
- Keep citations SHORT and CLEAN
- Document name + Page number ONLY (no section/chapter titles)
- Maximum TWO citations per answer
- Place citation at END of paragraph/answer
- If answer from ONE source → ONE citation only

STRICT BANS:
- No invented recommendations or operational guidance.
- No troubleshooting instructions unless explicitly stated in documentation.
- No conditional phrases such as “if needed,” “let me know,” “you can,” etc.
- No conversational filler, no rhetorical questions.

RESPONSE CONTRACT:
Your answer must always be:
✔ direct
✔ factual
✔ finished — no invitations, no continuation prompts

If insufficient context exists, state absence and stop.

You can answer general conversational greetings naturally without document context."""
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Build user message - include context only if available
    if context_text or schemas_text or tables_text:
        # Add explicit table reference requirement if tables present
        table_reminder = ""
        if tables:
            table_reminder = (
                f"\n\n⚠️ TABLE REFERENCE REQUIREMENT:\n"
                f"You have {len(tables)} table(s) available above ([TABLE1], [TABLE2], etc.).\n"
                f"IF your answer uses data from ANY table → YOU MUST include [TABLEi] reference inline.\n"
                f"Example: 'The causes are listed in [TABLE1].' or 'According to [TABLE2], the pressure is...'\n"
                f"FAILURE TO REFERENCE TABLES WHEN USED WILL CAUSE ANSWER REJECTION."
            )
        
        schema_reminder = ""
        if schemas:
            schema_reminder = (
                f"\n\n⚠️ DIAGRAM REFERENCE REQUIREMENT:\n"
                f"You have {len(schemas)} diagram(s) available above ([DIAGRAM1], [DIAGRAM2], etc.).\n"
                f"IF your answer references system layout/flow → YOU MUST include [DIAGRAMi] reference inline."
            )
        
        user_content = (
            f"Question: {state['question']}\n\n"
            f"=== DOCUMENTATION CONTEXT ===\n\n"
            f"{context_text}{schemas_text}{tables_text}"
            f"{table_reminder}{schema_reminder}\n\n"
            f"⚠️ REMINDER: Answer ONLY from the context above. "
            f"Do NOT add information from general knowledge. "
            f"Every claim must be directly supported by the provided documentation. "
            f"If information is missing, explicitly state that."
        )
    else:
        user_content = state['question']
    
    messages.append({"role": "user", "content": user_content})
    
    # Generate answer
    try:
        resp = llm.invoke(messages)
        answer_text = resp.content if resp.content else ""
        
        # Log initial answer length for debugging
        logger.info(f"Initial answer generated: {len(answer_text)} chars")
        if not answer_text or len(answer_text.strip()) < 10:
            logger.warning(f"⚠️ Initial answer is empty or very short ({len(answer_text)} chars)")
            logger.warning(f"   Response object: {type(resp)}, has content: {hasattr(resp, 'content')}")
            if hasattr(resp, 'content'):
                logger.warning(f"   Content value: {repr(resp.content)[:200]}")
    except Exception as e:
        logger.error(f"❌ LLM invocation failed: {e}")
        answer_text = ""
    
    # Parse which diagrams/tables the LLM referenced
    referenced_diagrams = set()
    referenced_tables = set()
    
    for ref in schema_map.keys():
        if ref in answer_text:
            referenced_diagrams.add(ref)
    
    for ref in table_map.keys():
        if ref in answer_text:
            referenced_tables.add(ref)
    
    logger.info(f"LLM referenced: {len(referenced_diagrams)} diagrams, {len(referenced_tables)} tables")
    
    # ============================================================================
    # ADAPTIVE RETRY: Quality check + context expansion fallback
    # ============================================================================
    # Check if this is the first retrieval attempt (retry only once)
    retrieval_attempt = state.get("retrieval_attempt", 0)
    
    if retrieval_attempt == 0 and enriched_context:
        # Quality check on first answer
        needs_retry, retry_reason = check_answer_quality(
            answer_text=answer_text,
            question=state["question"],
            context_items=enriched_context
        )
        
        if needs_retry:
            logger.warning(f"🔄 Answer quality check failed: {retry_reason}")
            logger.info(f"   Attempting adaptive fallback search...")
            
            # Get additional context based on failure reason
            additional_context = await adaptive_fallback_search(
                question=state["question"],
                retry_reason=retry_reason,
                existing_context=enriched_context,
                driver=tool_ctx.neo4j_driver,
                qdrant_client=tool_ctx.qdrant_client,
                query_intent=query_intent
            )
            
            if additional_context:
                logger.info(f"   ✅ Fallback found {len(additional_context)} additional items")
                
                # Extend context with new items (deduplicate)
                existing_ids = {
                    item.get("section_id") or item.get("table_id") or item.get("schema_id")
                    for item in enriched_context
                }
                
                new_items = []
                for item in additional_context:
                    item_id = item.get("section_id") or item.get("table_id") or item.get("schema_id")
                    if item_id and item_id not in existing_ids:
                        new_items.append(item)
                        existing_ids.add(item_id)
                
                if new_items:
                    # Update enriched context
                    enriched_context.extend(new_items)
                    state["enriched_context"] = enriched_context
                    
                    # Rebuild context text with new items
                    new_chunks = [c for c in new_items if c["type"] == "text_chunk"]
                    new_tables = [c for c in new_items if c["type"] == "table_chunk"]
                    new_schemas = [c for c in new_items if c["type"] == "schema"]
                    
                    # Append to existing context strings
                    if new_chunks:
                        context_text += "\n=== ADDITIONAL TEXT SECTIONS (fallback search) ===\n\n"
                        for i, c in enumerate(new_chunks, len(chunks) + 1):
                            context_text += f"[T{i}] Document: {c.get('doc_title', 'Unknown')}\n"
                            context_text += f"Section: {c.get('section_title', '')} (Page {c.get('page')})\n"
                            context_text += f"{c['text']}\n\n"
                    
                    if new_tables:
                        base_idx = len(tables)
                        for i, t in enumerate(new_tables, base_idx + 1):
                            table_ref = f"[TABLE{i}]"
                            table_map[table_ref] = t
                            tables_text += f"{table_ref} {t.get('title', 'Untitled')}\n"
                            tables_text += f"Caption: {t.get('caption', '')}\n"
                            tables_text += f"Document: {t.get('doc_title', 'Unknown')} (Page {t.get('page')})\n"
                            tables_text += f"Complete table data:\n{t['text']}\n\n"
                    
                    if new_schemas:
                        base_idx = len(schemas)
                        for i, s in enumerate(new_schemas, base_idx + 1):
                            diagram_ref = f"[DIAGRAM{i}]"
                            schema_map[diagram_ref] = s
                            schemas_text += f"{diagram_ref} {s.get('title', 'Figure')}\n"
                            schemas_text += f"Caption: {s.get('caption', '')}\n"
                            if s.get('llm_summary'):
                                schemas_text += f"Description: {s.get('llm_summary')}\n"
                            schemas_text += f"Document: {s.get('doc_title', 'Unknown')} (Page {s.get('page')})\n\n"
                    
                    # Rebuild user message with extended context
                    user_content = (
                        f"Question: {state['question']}\n\n"
                        f"=== DOCUMENTATION CONTEXT (EXPANDED) ===\n\n"
                        f"{context_text}{schemas_text}{tables_text}\n\n"
                        f"⚠️ REMINDER: Answer ONLY from the context above. "
                        f"Previous attempt had insufficient information. Use the expanded context carefully."
                    )
                    
                    messages[-1] = {"role": "user", "content": user_content}
                    
                    # Mark that we've retried
                    state["retrieval_attempt"] = 1
                    
                    # Save original answer before regeneration
                    original_answer = answer_text
                    
                    # Calculate context size
                    total_context_chars = len(user_content)
                    total_messages_chars = sum(len(str(m.get('content', ''))) for m in messages)
                    
                    # Regenerate answer with expanded context
                    logger.info(f"   🔄 Regenerating with expanded context ({total_context_chars} chars, {len(messages)} messages, total: {total_messages_chars} chars)")
                    
                    try:
                        resp = llm.invoke(messages)
                        new_answer = resp.content if resp.content else ""
                        
                        # Defensive check: empty answer after regeneration
                        if not new_answer or len(new_answer.strip()) < 10:
                            logger.error(f"   ❌ Regeneration returned empty/short answer ({len(new_answer)} chars) - keeping original")
                            logger.warning(f"      Context size: {total_context_chars} chars, may exceed token limit")
                            answer_text = original_answer  # Keep original answer
                        else:
                            answer_text = new_answer
                            logger.info(f"   ✅ Regenerated answer ({len(answer_text)} chars)")
                    except Exception as e:
                        logger.error(f"   ❌ Regeneration failed with error: {e}")
                        logger.warning(f"      Keeping original answer ({len(original_answer)} chars)")
                        answer_text = original_answer
                    
                    # Re-parse references
                    referenced_diagrams = set()
                    referenced_tables = set()
                    for ref in schema_map.keys():
                        if ref in answer_text:
                            referenced_diagrams.add(ref)
                    for ref in table_map.keys():
                        if ref in answer_text:
                            referenced_tables.add(ref)
                    
                    logger.info(f"      Explicit refs: {len(referenced_diagrams)} diagrams, {len(referenced_tables)} tables")
                else:
                    logger.warning(f"   ⚠️ Fallback returned only duplicates, no new context added")
            else:
                logger.warning(f"   ⚠️ Fallback search returned no additional context")
    
    # PHRASE EXTRACTION: Extract key phrases from answer for matching (used in citations & tables)
    import re
    answer_words = answer_text.lower().split() if answer_text else []
    answer_phrases = set()
    for i in range(len(answer_words) - 2):
        phrase = " ".join(answer_words[i:i+3])
        if len(phrase) > 10:  # Min 10 chars
            answer_phrases.add(phrase)
    
    # Get query intent for adaptive thresholds
    query_intent = state.get("query_intent", "text")
    
    # AUTO-DETECT TABLES: Check if answer contains tabular data patterns (before validator)
    # Patterns: numbered lists, parameter lists, troubleshooting steps
    if len(referenced_tables) == 0 and len(table_map) > 0 and answer_text:
        
        # Check if answer has tabular patterns
        has_numbered_list = bool(re.search(r'\n\s*\d+[\)\.]\s+', answer_text))
        has_bullet_list = bool(re.search(r'\n\s*[-•]\s+', answer_text))
        has_parameters = answer_text.count(':') >= 3  # Multiple key:value pairs
        has_multiple_items = answer_text.count('\n') >= 5  # Many lines
        
        if has_numbered_list or (has_bullet_list and has_parameters) or has_multiple_items:
            logger.info(f"   🔍 Auto-detected tabular data in answer (numbered={has_numbered_list}, params={has_parameters})")
            
            # Find table with best content match
            best_table = None
            best_table_ref = None
            best_score = 0
            
            for ref, t in table_map.items():
                table_text = t.get("text", "").lower()
                
                # Score by phrase overlap
                table_words = table_text.split()
                table_phrases = set()
                for i in range(len(table_words) - 2):
                    phrase = " ".join(table_words[i:i+3])
                    if len(phrase) > 10:
                        table_phrases.add(phrase)
                
                if answer_phrases and table_phrases:
                    overlap = len(answer_phrases & table_phrases)
                    overlap_ratio = overlap / len(answer_phrases) if answer_phrases else 0
                    
                    if overlap_ratio > best_score:
                        best_score = overlap_ratio
                        best_table = t
                        best_table_ref = ref
            
            # Adaptive threshold: stricter for intent=table (must include table even with low overlap)
            # For intent=table: if patterns detected + any table exists → add best match (min 1% to avoid noise)
            # For other intents: require 5% overlap to avoid false positives
            threshold = 0.01 if query_intent == "table" else 0.05
            
            if best_table:
                if best_score > threshold:
                    referenced_tables.add(best_table_ref)
                    logger.info(f"   ✅ Auto-added table {best_table_ref} (overlap: {best_score:.1%}, threshold={threshold:.0%}, intent={query_intent})")
                else:
                    logger.info(f"   ⚠️ Best table {best_table_ref} has low overlap ({best_score:.1%}), threshold={threshold:.0%}")
    
    # POST-GENERATION VALIDATOR: enforce intent-based constraints
    # ⚡ OPTIMIZATION: Few-shot examples in prompt should reduce regeneration frequency
    # Note: query_intent already fetched before auto-detection
    needs_regeneration = False
    regeneration_reason = ""
    
    # MUST include table if intent=table and tables available
    if query_intent == "table" and len(referenced_tables) == 0 and len(table_map) > 0:
        needs_regeneration = True
        regeneration_reason = "Intent=table but no table referenced. MUST include [TABLE1] or similar."
    # MUST include diagram if intent=schema and schemas available
    elif query_intent == "schema" and len(referenced_diagrams) == 0 and len(schema_map) > 0:
        needs_regeneration = True
        regeneration_reason = "Intent=schema but no diagram referenced. MUST include [DIAGRAM1] or similar."
    # MUST include BOTH table AND diagram if intent=mixed
    elif query_intent == "mixed":
        missing_parts = []
        if len(referenced_tables) == 0 and len(table_map) > 0:
            missing_parts.append("[TABLE1]")
        if len(referenced_diagrams) == 0 and len(schema_map) > 0:
            missing_parts.append("[DIAGRAM1]")
        if missing_parts:
            needs_regeneration = True
            regeneration_reason = f"Intent=mixed but missing: {', '.join(missing_parts)}. MUST include BOTH tables AND diagrams."
    # NOTE: text intent is NOT enforced - LLM can cite tables/diagrams if answer is there
    
    if needs_regeneration:
        logger.warning(f"⚠️ REGENERATION REQUIRED: {regeneration_reason}")
        
        # Build specific example for the intent
        if query_intent == "table":
            example = f"""
CORRECT FORMAT EXAMPLE:
"According to [TABLE1], the specifications are: [data from table].
 [Document | Table Title | Page X]"
 
AVAILABLE TABLES:
{tables_text}

YOU MUST use [TABLE1] or [TABLE2] etc. in your answer."""
        elif query_intent == "schema":
            example = f"""
CORRECT FORMAT EXAMPLE:
"The system layout is shown in [DIAGRAM1], which depicts [description].
 [Document | Diagram Title | Page X]"
 
AVAILABLE DIAGRAMS:
{schemas_text}

YOU MUST use [DIAGRAM1] or [DIAGRAM2] etc. in your answer."""
        elif query_intent == "mixed":
            example = f"""
CORRECT FORMAT EXAMPLE:
"According to [TABLE1], the parameters are: [data]. The system layout in [DIAGRAM1] shows [description].
 [Document | Table Title | Page X]
 [Document | Diagram Title | Page Y]"
 
AVAILABLE TABLES:
{tables_text}

AVAILABLE DIAGRAMS:
{schemas_text}

YOU MUST use BOTH [TABLE1] AND [DIAGRAM1] (or similar) in your answer."""
        else:
            example = ""
        
        # Add strict correction to messages
        correction_prompt = f"""
Your previous answer violated intent constraints:
{regeneration_reason}

REGENERATE your answer following this EXACT pattern:
- Intent={query_intent}
{example}

Start your answer with the reference inline (e.g., "According to [TABLE1]..." or "As shown in [DIAGRAM1]...").
Provide ONLY the corrected answer text."""
        
        messages.append({"role": "assistant", "content": answer_text})
        messages.append({"role": "user", "content": correction_prompt})
        
        # Regenerate
        resp_regenerated = llm.invoke(messages)
        new_answer = resp_regenerated.content
        
        # Defensive check: empty answer after intent regeneration
        if not new_answer or len(new_answer.strip()) < 10:
            logger.error(f"❌ Intent regeneration returned empty/short answer ({len(new_answer)} chars) - keeping original")
            # Keep previous answer_text
        else:
            answer_text = new_answer
            logger.info(f"✅ Regenerated answer with intent={query_intent} ({len(answer_text)} chars)")
        
        # Re-parse references
        referenced_diagrams = set()
        referenced_tables = set()
        for ref in schema_map.keys():
            if ref in answer_text:
                referenced_diagrams.add(ref)
        for ref in table_map.keys():
            if ref in answer_text:
                referenced_tables.add(ref)
        
        logger.info(f"   Final refs: {len(referenced_diagrams)} diagrams, {len(referenced_tables)} tables")
        if len(new_answer.strip()) < 10:
            logger.error(f"⚠️ REGENERATION FAILED - Intent={query_intent} produced empty answer")
        else:
            logger.warning(f"⚠️ REGENERATION OCCURRED - Consider improving few-shot examples for intent={query_intent}")
    
    # Build response - SMART FILTERING: only include citations actually used in answer
    # Strategy: Check which chunks have content overlap with answer_text
    # NOTE: answer_phrases already extracted earlier (reused here)
    citations = []
    seen_citations = set()
    
    # Score each chunk by phrase overlap
    chunk_scores = []
    for c in chunks:
        chunk_text = c.get("text", "").lower()
        chunk_words = chunk_text.split()
        
        # Build 3-word phrases from chunk
        chunk_phrases = set()
        for i in range(len(chunk_words) - 2):
            phrase = " ".join(chunk_words[i:i+3])
            if len(phrase) > 10:
                chunk_phrases.add(phrase)
        
        # Calculate overlap
        if answer_phrases and chunk_phrases:
            overlap = len(answer_phrases & chunk_phrases)
            overlap_ratio = overlap / len(answer_phrases) if answer_phrases else 0
        else:
            overlap_ratio = 0
        
        chunk_scores.append((c, overlap_ratio))
    
    # Sort by overlap, take top chunks
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Include chunks with meaningful overlap (>3%) or top 2 chunks
    # Relaxed threshold (5% → 3%) to avoid losing relevant citations
    for c, score in chunk_scores:
        if score > 0.03 or len(citations) < 2:  # At least 3% overlap OR top 2
            citation_key = (c.get("doc_id"), c.get("section_id"), c.get("page"))
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                citations.append({
                    "type": "text",
                    "doc_id": c.get("doc_id"),
                    "section_id": c.get("section_id"),
                    "page": c.get("page"),
                    "title": c.get("section_title", "Unknown"),
                    "doc_title": c.get("doc_title"),
                })
                
                if len(citations) >= 5:  # Max 5 citations (was 3)
                    break
    
    logger.info(f"📝 Citation filtering: {len(chunks)} chunks → {len(citations)} citations (by phrase overlap)")
    
    # Include schemas based on CONTENT OVERLAP (context/caption matching)
    # Strategy: Calculate overlap for ALL schemas, include those with significant overlap
    figures = []
    schema_overlaps = []  # Store (schema, overlap_score) for all schemas
    
    # Calculate overlap for ALL schemas
    for ref, s in schema_map.items():
        # Schema overlap: check caption + llm_summary (text_context already in answer via semantic search)
        schema_text = f"{s.get('caption', '')} {s.get('llm_summary', '')}".lower()
        
        # Score by phrase overlap
        schema_words = schema_text.split()
        schema_phrases = set()
        for i in range(len(schema_words) - 2):
            phrase = " ".join(schema_words[i:i+3])
            if len(phrase) > 10:
                schema_phrases.add(phrase)
        
        if answer_phrases and schema_phrases:
            overlap = len(answer_phrases & schema_phrases)
            overlap_ratio = overlap / len(answer_phrases) if answer_phrases else 0
        else:
            overlap_ratio = 0
        
        # Boost score if LLM explicitly referenced this schema
        was_referenced = ref in referenced_diagrams
        if was_referenced:
            overlap_ratio = max(overlap_ratio, 0.10)  # Minimum 10% if referenced
            logger.info(f"🖼️  Schema {ref} explicitly referenced by LLM + overlap={overlap_ratio:.1%}")
        
        schema_overlaps.append((ref, s, overlap_ratio, was_referenced))
    
    # Sort by overlap score
    schema_overlaps.sort(key=lambda x: x[2], reverse=True)
    
    # Add schemas with significant overlap (>2%) OR top 1 if intent requires schema
    # Lower threshold for schemas (2% vs 3% for tables) because captions are shorter
    overlap_threshold = 0.02
    
    for ref, s, overlap_score, was_referenced in schema_overlaps:
        # Include if: overlap > threshold OR explicitly referenced OR (intent=schema and top)
        should_include = (
            overlap_score >= overlap_threshold or
            was_referenced or
            (query_intent in ["schema", "mixed"] and len(figures) == 0 and overlap_score > 0)
        )
        
        if should_include:
            url = s.get("file_path")
            if url and not url.startswith('/'):
                url = '/' + url
            
            figures.append({
                "schema_id": s.get("schema_id"),
                "title": s.get("title", ""),
                "caption": s.get("caption", ""),
                "url": url,
                "page": s.get("page"),
                "doc_title": s.get("doc_title", "Unknown"),
            })
            
            logger.info(
                f"✅ Included schema {ref} '{s.get('title', 'Untitled')[:50]}': "
                f"overlap={overlap_score:.1%}, referenced={was_referenced}"
            )
            
            # Limit to 3 schemas
            if len(figures) >= 3:
                break
        else:
            logger.debug(f"⏭️  Skipped schema {ref}: overlap={overlap_score:.1%} < threshold")
    
    # Include tables based on CONTENT OVERLAP (phrase matching)
    # Strategy: Calculate overlap for ALL tables, include those with significant overlap
    table_refs = []
    table_overlaps = []  # Store (table, overlap_score) for all tables
    
    # Calculate overlap for ALL tables (not just referenced ones)
    for ref, t in table_map.items():
        table_text = t.get("text", "").lower()
        
        # Score by phrase overlap (same as citations)
        table_words = table_text.split()
        table_phrases = set()
        for i in range(len(table_words) - 2):
            phrase = " ".join(table_words[i:i+3])
            if len(phrase) > 10:
                table_phrases.add(phrase)
        
        if answer_phrases and table_phrases:
            overlap = len(answer_phrases & table_phrases)
            overlap_ratio = overlap / len(answer_phrases) if answer_phrases else 0
        else:
            overlap_ratio = 0
        
        # Boost score if LLM explicitly referenced this table
        was_referenced = ref in referenced_tables
        if was_referenced:
            overlap_ratio = max(overlap_ratio, 0.10)  # Minimum 10% if referenced
            logger.info(f"📊 Table {ref} explicitly referenced by LLM + overlap={overlap_ratio:.1%}")
        
        table_overlaps.append((ref, t, overlap_ratio, was_referenced))
    
    # Sort by overlap score
    table_overlaps.sort(key=lambda x: x[2], reverse=True)
    
    # Add tables with significant overlap (>3%) OR top 1 if intent requires table
    overlap_threshold = 0.03  # 3% phrase overlap
    
    for ref, t, overlap_score, was_referenced in table_overlaps:
        # Include if: overlap > threshold OR explicitly referenced OR (intent=table and top table)
        should_include = (
            overlap_score >= overlap_threshold or
            was_referenced or
            (query_intent in ["table", "mixed"] and len(table_refs) == 0 and overlap_score > 0)
        )
        
        if should_include:
            url = t.get("file_path")
            if url and not url.startswith('/'):
                url = '/' + url
            if not url:
                logger.warning(f"Table {t.get('table_id')} has no file_path/url - won't display in frontend")
            else:
                logger.debug(f"Table {t.get('table_id')} url: {url}")
            
            table_refs.append({
                "table_id": t.get("table_id"),
                "title": t.get("title", ""),
                "caption": t.get("caption", ""),
                "url": url,
                "page": t.get("page"),
                "doc_title": t.get("doc_title", "Unknown"),
                "rows": t.get("rows"),
                "cols": t.get("cols"),
            })
            
            logger.info(
                f"✅ Included table {ref} '{t.get('title', 'Untitled')[:50]}': "
                f"overlap={overlap_score:.1%}, referenced={was_referenced}"
            )
            
            # Limit to 3 tables
            if len(table_refs) >= 3:
                break
        else:
            logger.debug(f"⏭️  Skipped table {ref}: overlap={overlap_score:.1%} < threshold")
    
    # FALLBACK: AUTO-DETECT if answer contains tabular data but no tables selected
    if len(table_refs) == 0 and len(table_map) > 0:
        # Check if answer has tabular patterns
        has_numbered_list = bool(re.search(r'\n\s*\d+[\)\.]\s+', answer_text))
        has_bullet_list = bool(re.search(r'\n\s*[-•]\s+', answer_text))
        has_parameters = answer_text.count(':') >= 3  # Multiple key:value pairs
        has_multiple_items = answer_text.count('\n') >= 5  # Many lines
        
        if has_numbered_list or (has_bullet_list and has_parameters) or has_multiple_items:
            logger.info(f"🔍 Auto-detected tabular data in answer (numbered_list={has_numbered_list}, params={has_parameters})")
            
            # Find table with best content match
            best_table = None
            best_score = 0
            
            for ref, t in table_map.items():
                table_text = t.get("text", "").lower()
                
                # Score by phrase overlap (same as citations)
                table_words = table_text.split()
                table_phrases = set()
                for i in range(len(table_words) - 2):
                    phrase = " ".join(table_words[i:i+3])
                    if len(phrase) > 10:
                        table_phrases.add(phrase)
                
                if answer_phrases and table_phrases:
                    overlap = len(answer_phrases & table_phrases)
                    overlap_ratio = overlap / len(answer_phrases) if answer_phrases else 0
                    
                    if overlap_ratio > best_score:
                        best_score = overlap_ratio
                        best_table = t
            
            # If found good match (>5% overlap), add table
            if best_table and best_score > 0.05:
                url = best_table.get("file_path")
                if url and not url.startswith('/'):
                    url = '/' + url
                
                table_refs.append({
                    "table_id": best_table.get("table_id"),
                    "title": best_table.get("title", ""),
                    "caption": best_table.get("caption", ""),
                    "url": url,
                    "page": best_table.get("page"),
                    "doc_title": best_table.get("doc_title", "Unknown"),
                    "rows": best_table.get("rows"),
                    "cols": best_table.get("cols"),
                })
                logger.info(f"✅ Auto-added table {best_table.get('table_id')} (overlap={best_score:.2%})")
            elif best_table:
                logger.info(f"⚠️ Best table has low overlap ({best_score:.2%}), threshold=5%")
    
    # FINAL JSON VALIDATION: ensure answer matches intent constraints
    # Only validate MUST requirements, not prohibitions
    final_validation_failed = False
    final_validation_reason = ""
    
    if query_intent == "table" and len(table_refs) == 0:
        final_validation_failed = True
        final_validation_reason = "Intent=table requires at least one table in JSON, but tables=[]"
    elif query_intent == "schema" and len(figures) == 0:
        final_validation_failed = True
        final_validation_reason = "Intent=schema requires at least one diagram in JSON, but figures=[]"
    # NOTE: text intent has NO validation - LLM decides what to include
    
    if final_validation_failed:
        logger.error(f"❌ FINAL JSON VALIDATION FAILED: {final_validation_reason}")
        logger.warning("⚠️ Forcing compliance by stripping/adding resources...")
        
        # Force compliance (only for table/schema requirements, NOT for text prohibitions)
        if query_intent == "table" and len(table_refs) == 0 and len(table_map) > 0:
            # First attempt: try to regenerate answer with strict table requirement
            logger.warning("⚠️ Intent=table but no tables referenced. Attempting regeneration...")
            
            table_requirement_prompt = f"""Your previous answer did NOT include any table references, but the question requires tabular data.

MANDATORY REQUIREMENT:
You MUST reference at least one table from the AVAILABLE TABLES section below using [TABLE1], [TABLE2], etc.

Question: {state['question']}

{tables_text}

REGENERATE your answer and explicitly reference the most relevant table(s) using [TABLE1] format.
Provide ONLY the corrected answer text."""
            
            messages.append({"role": "assistant", "content": answer_text})
            messages.append({"role": "user", "content": table_requirement_prompt})
            
            # Try regeneration
            try:
                resp_table_regen = llm.invoke(messages)
                answer_text_regen = resp_table_regen.content
                
                # Re-parse table references
                referenced_tables_regen = set()
                for ref in table_map.keys():
                    if ref in answer_text_regen:
                        referenced_tables_regen.add(ref)
                
                if len(referenced_tables_regen) > 0:
                    # Success! Use regenerated answer
                    answer_text = answer_text_regen
                    referenced_tables = referenced_tables_regen
                    
                    # Rebuild table_refs
                    table_refs = []
                    for ref in referenced_tables:
                        t = table_map[ref]
                        url = t.get("file_path")
                        if url and not url.startswith('/'):
                            url = '/' + url
                        table_refs.append({
                            "table_id": t.get("table_id"),
                            "title": t.get("title", ""),
                            "caption": t.get("caption", ""),
                            "url": url,
                            "page": t.get("page"),
                            "doc_title": t.get("doc_title", "Unknown"),
                            "rows": t.get("rows"),
                            "cols": t.get("cols"),
                        })
                    logger.info(f"✅ Regeneration successful: {len(table_refs)} tables now included")
                else:
                    # Regeneration failed, force add
                    logger.error("❌ Regeneration failed to include tables. Force adding first table...")
                    first_table_ref = list(table_map.keys())[0]
                    t = table_map[first_table_ref]
                    url = t.get("file_path")
                    if url and not url.startswith('/'):
                        url = '/' + url
                    table_refs.append({
                        "table_id": t.get("table_id"),
                        "title": t.get("title", ""),
                        "caption": t.get("caption", ""),
                        "url": url,
                        "page": t.get("page"),
                        "doc_title": t.get("doc_title", "Unknown"),
                        "rows": t.get("rows"),
                        "cols": t.get("cols"),
                    })
                    logger.info(f"Forced compliance: added table {t.get('table_id')} for intent=table")
            except Exception as e:
                logger.error(f"Regeneration failed with error: {e}. Force adding first table...")
                first_table_ref = list(table_map.keys())[0]
                t = table_map[first_table_ref]
                url = t.get("file_path")
                if url and not url.startswith('/'):
                    url = '/' + url
                table_refs.append({
                    "table_id": t.get("table_id"),
                    "title": t.get("title", ""),
                    "caption": t.get("caption", ""),
                    "url": url,
                    "page": t.get("page"),
                    "doc_title": t.get("doc_title", "Unknown"),
                    "rows": t.get("rows"),
                    "cols": t.get("cols"),
                })
                logger.info(f"Forced compliance: added table {t.get('table_id')} for intent=table")
        elif query_intent == "schema" and len(figures) == 0 and len(schema_map) > 0:
            # Force add first available schema
            first_schema_ref = list(schema_map.keys())[0]
            s = schema_map[first_schema_ref]
            url = s.get("file_path")
            if url and not url.startswith('/'):
                url = '/' + url
            figures.append({
                "schema_id": s.get("schema_id"),
                "title": s.get("title", ""),
                "caption": s.get("caption", ""),
                "url": url,
                "page": s.get("page"),
                "doc_title": s.get("doc_title", "Unknown"),
            })
            logger.info(f"Forced compliance: added schema {s.get('schema_id')} for intent=schema")
    
    state["answer"] = {
        "answer_text": answer_text,
        "citations": citations,
        "figures": figures,
        "tables": table_refs,
    }
    
    # Log final answer stats
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ANSWER GENERATED")
    logger.info(f"   Intent: {query_intent}")
    logger.info(f"   Answer length: {len(answer_text)} chars")
    logger.info(f"   Citations: {len(citations)}")
    logger.info(f"   Figures: {len(figures)}")
    logger.info(f"   Tables: {len(table_refs)}")
    for tr in table_refs:
        logger.info(f"     - {tr.get('title')} (p{tr.get('page')}) url={tr.get('url')}")
    logger.info(f"{'='*60}\n")
    
    return state



# ROUTING LOGIC

def should_continue_to_tools(state: GraphState) -> str:
    """Decide if agent wants to call tools or is done"""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = get_tool_calls(last_message)
    if tool_calls:
        return "execute_tools"
    else:
        return "build_context"


# GRAPH BUILDER

async def preload_entities(neo4j_driver: Driver) -> List[str]:
    """
    ⚡ OPTIMIZATION: Preload entities at application startup (not on first query).
    Eliminates 200-500ms blocking call from first user request.
    
    Args:
        neo4j_driver: Async Neo4j driver
    
    Returns:
        List of entity names and codes
    """
    try:
        logger.info("⚡ Preloading known entities from Neo4j...")
        entities = await load_known_entities()
        logger.info(f"✅ Preloaded {len(entities)} entities at startup")
        return entities
    except Exception as e:
        logger.error(f"❌ Failed to preload entities: {e}", exc_info=True)
        return []


def build_qa_graph(
    qdrant_client: QdrantClient,
    neo4j_driver: Driver,
    vector_service: Optional[Any] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_auth: Optional[tuple] = None,
) -> Any:
    """
    Build agentic Q&A workflow with Neo4j as tool.
    
    Flow: analyze_and_route (MERGED) → execute_tools → build_context → llm_reasoning
    
    OPTIMIZATION: analyze_and_route merges intent classification + tool routing into ONE LLM call
    """
    # Set tool context
    tool_ctx.qdrant_client = qdrant_client
    tool_ctx.neo4j_driver = neo4j_driver
    tool_ctx.vector_service = vector_service
    tool_ctx.neo4j_uri = neo4j_uri
    tool_ctx.neo4j_auth = neo4j_auth
    
    # Log entity preload status
    logger.info(f"📊 build_qa_graph: entities_loaded={tool_ctx.entities_loaded}, entities count={len(tool_ctx.known_entities)}")
    
    graph = StateGraph(GraphState)
    
    # Wrap nodes with dependencies
    async def build_context_node(state: GraphState) -> GraphState:
        return await node_build_context(state, neo4j_driver)
    
    # Add nodes
    graph.add_node("analyze_and_route", node_analyze_and_route)  # MERGED NODE - single LLM call
    graph.add_node("execute_tools", node_execute_tools)
    graph.add_node("build_context", build_context_node)
    graph.add_node("llm_reasoning", node_llm_reasoning)
    
    # Build flow - simplified with merged node
    graph.set_entry_point("analyze_and_route")
    
    # Conditional edge from merged analyze_and_route node
    graph.add_conditional_edges(
        "analyze_and_route",
        should_continue_to_tools,
        {
            "execute_tools": "execute_tools",
            "build_context": "build_context",
        }
    )
    
    graph.add_edge("execute_tools", "build_context")
    graph.add_edge("build_context", "llm_reasoning")
    graph.add_edge("llm_reasoning", END)
    
    return graph.compile()
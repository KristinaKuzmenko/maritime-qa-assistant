"""
Agentic LangGraph workflow with Neo4j as a tool.

Architecture:
1. Query Analysis - detect intent
2. Router Agent - LLM decides which tools to use (Qdrant, Neo4j, or both)
3. Context Builder - merge results, expand with neighbor chunks
4. LLM Reasoning - generate answer

Key features:
- Neo4j is a tool, not hidden pipeline step
- Agent decides when to use graph vs vector search
- Automatic neighbor chunk retrieval for complete context
"""

from typing import TypedDict, List, Any, Dict, Optional, Literal, Annotated
from langgraph.graph import StateGraph, END
import logging
import operator
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
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
                    entities.add(name)
                
                # Add code if it exists (codes are usually specific)
                if code and code.strip():
                    entities.add(code)
            
            entities_list = sorted(list(entities))
            logger.info(f"✅ Loaded {len(entities_list)} known entities (names + codes) from Neo4j")
            return entities_list
            
    except Exception as e:
        logger.error(f"❌ Failed to load known entities: {e}", exc_info=True)
        return []


def load_known_entities_sync() -> List[str]:
    """
    Synchronous version of load_known_entities for use in sync context.
    Loads entity names and codes from Neo4j.
    """
    if not tool_ctx.neo4j_uri or not tool_ctx.neo4j_auth:
        logger.warning("⚠️ Neo4j connection info not available - cannot load entities")
        return []
    
    try:
        # Use synchronous Neo4j driver
        from neo4j import GraphDatabase
        
        # Create sync driver
        sync_driver = GraphDatabase.driver(
            tool_ctx.neo4j_uri,
            auth=tool_ctx.neo4j_auth
        )
        
        with sync_driver.session() as session:
            query = """
            MATCH (e:Entity)
            RETURN DISTINCT 
                e.name AS entity_name,
                e.code AS entity_code
            """
            result = session.run(query)
            records = list(result)
            
            logger.info(f"📊 Retrieved {len(records)} entity records from Neo4j")
            
            entities = set()
            
            for rec in records:
                name = rec.get("entity_name")
                code = rec.get("entity_code")
                
                # Add name if it exists and is not too generic
                if name and len(name) > 2:
                    entities.add(name)
                
                # Add code if it exists (codes are usually specific)
                if code and code.strip():
                    entities.add(code)
            
            entities_list = sorted(list(entities))
            logger.info(f"✅ Loaded {len(entities_list)} known entities (names + codes) from Neo4j")
            
            sync_driver.close()
            return entities_list
            
    except Exception as e:
        logger.error(f"❌ Failed to load known entities: {e}", exc_info=True)
        return []


def ensure_entities_loaded():
    """
    Ensure entities are loaded (lazy initialization).
    Called synchronously from node_router_agent.
    """
    if not tool_ctx.entities_loaded and tool_ctx.neo4j_driver:
        try:
            # Load entities using sync version
            tool_ctx.known_entities = load_known_entities_sync()
            tool_ctx.entities_loaded = True
            logger.info(f"✅ Entities loaded in router_agent: {len(tool_ctx.known_entities)} entities")
            
        except Exception as e:
            logger.error(f"Failed to load entities: {e}", exc_info=True)
            tool_ctx.entities_loaded = True  # Set to true to avoid repeated attempts


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
    include_content: bool = False
) -> List[Dict[str, Any]]:
    """
    Search sections using Neo4j fulltext index 'sectionSearch'.
    
    Args:
        search_term: Query string (use quotes for exact match: '"PU3" OR "7M2"')
        limit: Max results
        min_score: Minimum Lucene score
        include_content: If True, return full section content (slower but needed for context)
    
    Returns:
        List of sections with section_id, doc_id, title, score, [content]
    """
    if not tool_ctx.neo4j_driver:
        return []
    
    try:
        async with tool_ctx.neo4j_driver.session() as session:
            # Section nodes have 'id' field, not 'section_id'
            if include_content:
                query = """
                CALL db.index.fulltext.queryNodes('sectionSearch', $search_term)
                YIELD node, score
                WHERE score > $min_score
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
                query = """
                CALL db.index.fulltext.queryNodes('sectionSearch', $search_term)
                YIELD node, score
                WHERE score > $min_score
                RETURN node.id AS section_id,
                       node.doc_id AS doc_id, 
                       node.title AS title, 
                       score
                ORDER BY score DESC
                LIMIT $limit
                """
            
            result = await session.run(query, {
                "search_term": search_term,
                "min_score": min_score,
                "limit": limit
            })
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
        embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key
        )
        query_vector = embeddings.embed_query(query)
        
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
        embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key
        )
        query_vector = embeddings.embed_query(query)
        
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
        embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key
        )
        query_vector = embeddings.embed_query(query)
        
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
    
    HOW IT WORKS:
    1. Extracts entity mentions from query (pumps, valves, FO, P-101, etc.)
    2. Queries Neo4j graph for nodes linked to these entities via:
       - Section -[:DESCRIBES]-> Entity (returns full section content!)
       - Table -[:MENTIONS]-> Entity (returns metadata)
       - Schema -[:DEPICTS]-> Entity (returns metadata + file path)
    3. Returns content directly from Neo4j (no Qdrant round-trip needed)
    
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
        
        if query_words or equipment_codes or entity_ids:
            async with tool_ctx.neo4j_driver.session() as session:
                # First: Search by entity codes from EntityExtractor (comp_pump_fuel_oil_pump)
                # These are normalized codes generated during ingestion
                if entity_ids:
                    logger.info(f"neo4j_entity_search: searching by entity_ids from extractor: {entity_ids}")
                    entity_query = """
                    MATCH (e:Entity)
                    WHERE e.code IN $entity_codes
                    RETURN e.code AS code, e.name AS name, e.entity_type AS entity_type
                    LIMIT 20
                    """
                    result = await session.run(entity_query, {"entity_codes": entity_ids})
                    found_entities = await result.data()
                    logger.info(f"neo4j_entity_search: entity search returned {len(found_entities)} entities")
                    
                    if found_entities:
                        found_by_equipment_code = True
                        for ent in found_entities:
                            logger.info(f"✓ Found entity from dictionary: {ent['name']} ({ent['code']}) [{ent.get('entity_type', 'unknown')}]")
                
                # Second: Search by equipment codes (PU3, P-101, etc.) in entity names
                # Equipment codes are often embedded in entity names during ingestion
                if equipment_codes:
                    code_query = """
                    MATCH (e:Entity)
                    WHERE ANY(code IN $codes WHERE toUpper(e.name) CONTAINS code)
                    RETURN e.code AS code, e.name AS name, e.entity_type AS entity_type
                    LIMIT 10
                    """
                    result = await session.run(code_query, {"codes": equipment_codes})
                    code_entities = await result.data()
                    logger.info(f"neo4j_entity_search: equipment code search returned {len(code_entities)} entities")
                    
                    for ent in code_entities:
                        if ent["code"] and ent["code"] not in entity_ids:
                            entity_ids.append(ent["code"])
                            found_by_equipment_code = True
                            logger.info(f"✓ Found entity by equipment code in name: {ent['name']} ({ent['code']}) [{ent.get('entity_type', 'unknown')}]")
                
                # Third: ALWAYS try phrase search for multi-word entities (e.g., "cut cock")
                # EntityExtractor may miss multi-word entities if qualifier not in dictionary
                # Example: "cut cock" → extractor finds "cock", but phrase search finds "cut cock"
                # Run ALWAYS, not just as fallback - we want more specific entities
                if len(query_words) >= 2:
                    # Extract 2-3 word phrases from query
                    query_lower = query.lower()
                    words = query_lower.split()
                    phrases = []
                    
                    # Build 2-word and 3-word phrases
                    for i in range(len(words) - 1):
                        two_word = f"{words[i]} {words[i+1]}"
                        if len(two_word) >= 6 and not any(sw in words[i:i+2] for sw in stop_words):
                            phrases.append(two_word)
                        
                        if i < len(words) - 2:
                            three_word = f"{words[i]} {words[i+1]} {words[i+2]}"
                            if len(three_word) >= 8 and not any(sw in words[i:i+3] for sw in stop_words):
                                phrases.append(three_word)
                    
                    if phrases:
                        logger.info(f"neo4j_entity_search: trying phrase search for: {phrases}")
                        
                        phrase_query = """
                        MATCH (e:Entity)
                        WHERE ANY(phrase IN $phrases WHERE toLower(e.name) CONTAINS phrase)
                        RETURN e.code AS code, e.name AS name, e.entity_type AS entity_type
                        ORDER BY size(e.name) ASC
                        LIMIT 10
                        """
                        result = await session.run(phrase_query, {"phrases": phrases})
                        phrase_entities = await result.data()
                        
                        if phrase_entities:
                            logger.info(f"✓ Found {len(phrase_entities)} entities by phrase match")
                            for ent in phrase_entities:
                                if ent["code"] and ent["code"] not in entity_ids:
                                    entity_ids.append(ent["code"])
                                    found_by_equipment_code = True
                                    logger.info(f"  └─ {ent['name']} ({ent['code']}) [{ent.get('entity_type', 'unknown')}]")
                
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


# NODE 1: QUERY ANALYSIS

def node_analyze_question(state: GraphState) -> GraphState:
    """
    Analyze question to detect query intent using LLM classification.
    Intent guides agent's tool selection.
    """
    question = state["question"]
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"📝 NEW QUESTION: {question}")
    logger.info(f"{'#'*60}\n")
    
    # Use LLM for intent classification
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
    classification_prompt = f"""Classify the user's question into ONE of these categories:

Categories:
- "text" - seeking textual information, procedures, explanations, descriptions
- "table" - seeking tabular data, specifications, parameters, technical values, troubleshooting info in TABLE format
- "schema" - seeking diagrams, schematics, figures, DRAWINGS, visual representations, layout images
- "mixed" - needs both semantic search AND graph traversal (e.g., "find all X in section Y", structural queries)

CRITICAL CLASSIFICATION RULES:

1. TROUBLESHOOTING/FAULT KEYWORDS → "table"
   If question contains: "cause", "reason", "troubleshooting", "fault", "failure", "breakdown", "malfunction", "problem", "issue", "error", "no suction", "not working", "won't start"
   → Classify as "table" (troubleshooting tables contain causes/solutions)

2. VISUAL CONTENT KEYWORDS → "schema"
   If question mentions: "drawing", "drawings", "diagram", "scheme", "figure", "layout", "show me", "where is", "location"
   → Classify as "schema" (user wants visual representation)

3. SPECIFICATIONS/PARAMETERS → "table"
   If question asks for: "specifications", "specs", "parameters", "values", "temperature", "pressure", "capacity", "dimensions", "range", "calibration", "rating", "tolerance", "limits", "settings" (numeric/technical data)
   → Classify as "table"

4. DEFAULT → "text"
   Procedural questions, explanations, descriptions without specs/visuals

Examples:
- "How does the fuel system work?" → text
- "What are the engine specifications?" → table
- "Show me the water heater diagram" → schema
- "The pump has no suction. What can be a cause?" → table (troubleshooting)
- "Why does the incinerator fail to start?" → table (fault analysis)
- "What causes pump breakdown?" → table (failure reasons)
- "Temperature range for cooling water?" → table (specifications)
- "What is the calibration range for PT-8?" → table (technical parameters)
- "Give me the scheme of cooling system" → schema
- "Where are the fuel connections located?" → schema (location/layout)
- "List all tables in chapter 3" → mixed
- "Find sections about safety on page 5" → mixed

User's question: "{question}"

Respond with ONLY the category name (text/table/schema/mixed), nothing else."""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    intent = response.content.strip().lower()
    
    # Validate response
    if intent not in ["text", "table", "schema", "mixed"]:
        logger.warning(f"LLM returned invalid intent: {intent}, defaulting to 'text'")
        intent = "text"
    
    state["query_intent"] = intent
    
    logger.info(f"🎯 Query intent classified as: {intent.upper()}")
    
    return state



# NODE 2: ROUTER AGENT

def node_router_agent(state: GraphState) -> GraphState:
    """
    LLM Agent with tools decides which sources to query.
    Can call Qdrant tools and/or Neo4j tool based on question.
    """
    # Ensure entities are loaded (lazy initialization on first call)
    ensure_entities_loaded()
    
    intent = state["query_intent"]
    question = state["question"]
    anchors = state.get("anchor_sections", [])
    
    # Build anchor info for agent
    anchor_info = ""
    if anchors:
        anchor_info = "\n\nANCHOR SECTIONS (focus on these):\n"
        for a in anchors:
            anchor_info += f"- Section {a['section_id']} (doc: {a['doc_id']}, score: {a['score']:.2f})\n"
    
    # Entity detection hint (directive for equipment codes, informative for named components)
    found_entities = find_entities_in_question(question, tool_ctx.known_entities)
    
    if found_entities:
        logger.info(f"🔍 Detected entities in question: {found_entities}")
        # Check if any entities are equipment codes (uppercase + numbers pattern)
        import re
        has_equipment_codes = any(re.match(r'^[A-Z]{1,4}[-]?[0-9]{1,5}$', e) for e in found_entities)
        
        if has_equipment_codes:
            # Directive hint for equipment codes
            entity_hint = f"""

⚠️ EQUIPMENT CODES DETECTED: {', '.join(found_entities[:5])}

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

📍 DETECTED ENTITIES: {', '.join(found_entities[:5])}

These named components exist in documentation. Consider:
- WHERE/WHICH DIAGRAM/LOCATION queries → neo4j_entity_search
- HOW/WHY/EXPLAIN procedures → qdrant_search_text (semantic)
- SPECS/PARAMETERS → neo4j_entity_search + qdrant_search_tables

This is guidance - use your judgment based on question type."""
    else:
        entity_hint = """

📍 No specific equipment entities detected.
→ Use semantic search (qdrant_search_*) for best results."""
    
    # Build system prompt with intent guidance
    system_prompt = f"""{GRAPH_SCHEMA_PROMPT}

You are a routing agent for maritime technical documentation Q&A system.
Select the best tools to answer the user's question.

DETECTED INTENT: {intent}{entity_hint}

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

2. **qdrant_search_tables** - For specifications/parameters/data:
   Temperatures, pressures, specs, technical data, troubleshooting
   Examples: "specs of X", "temperature range", "power ratings", "what causes failure"
   IMPORTANT: When question asks to EXPLAIN table → ALSO call qdrant_search_text!

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

MANDATORY TOOL CALLS BY INTENT:
- If intent="table" → YOU MUST call qdrant_search_tables
- If intent="schema" → YOU MUST call qdrant_search_schemas

QUESTION: "{question}"
INTENT: {intent}{anchor_info}

Select tool(s). Schema is in GRAPH_SCHEMA_PROMPT above."""
    
    # Create LLM with tools
    llm = ChatOpenAI(
        model=settings.llm_model,  
        temperature=0,
        api_key=settings.openai_api_key,
    )
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Build messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # Invoke agent
    response = llm_with_tools.invoke(messages)
    
    # FORCE tool calls for specific intents and enforce hybrid strategy
    tool_calls = get_tool_calls(response)
    tool_names = {tc.get('name') for tc in tool_calls}
    
    # Force table search for table intent
    if intent == "table" and "qdrant_search_tables" not in tool_names:
        logger.warning(f"Intent=table but agent didn't call qdrant_search_tables, forcing it")
        from langchain_core.messages import AIMessage
        forced_tool_call = {
            "name": "qdrant_search_tables",
            "args": {"query": question, "limit": 5},
            "id": "forced_table_search"
        }
        existing_calls = tool_calls if tool_calls else []
        response.tool_calls = existing_calls + [forced_tool_call]
        tool_calls = response.tool_calls
        tool_names = {tc.get('name') for tc in tool_calls}
    
    # NOTE: Hybrid strategy DISABLED - entity_search pollutes context
    # Let LLM decide whether to use entity_search or not
    # If LLM chooses entity_search alone, that's its decision (usually wrong, but let it learn)
    
    # Store messages for next iteration
    state["messages"] = [*messages, response]
    
    # Safely check for tool calls
    tool_calls = get_tool_calls(response)
    
    # Log detailed tool call information
    if tool_calls:
        logger.info(f"\n{'='*60}")
        logger.info(f"🤖 AGENT TOOL CALLS ({len(tool_calls)} tools):")
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
                    
                    # Use BM25 for re-ranking ONLY for sections with entity in text (not tables)
                    regular_sections = [sec for sec in entity_sections if sec.get("found_via") != "table_mentions"]
                    
                    if regular_sections:
                        try:
                            from rank_bm25 import BM25Okapi
                            import re
                            
                            # EXTRACT UNIQUE ENTITY NAMES from sections (e.g., "Fuel Oil Pump", "Oil Cooler")
                            # These are the actual entities found in Neo4j, not the full question
                            entity_names = list(set(
                                sec.get("matched_entity", "")
                                for sec in regular_sections
                                if sec.get("matched_entity")
                            ))
                            
                            if not entity_names:
                                raise ValueError("No matched entities in sections")
                            
                            logger.info(f"   🎯 Entities to search for: {entity_names}")
                            
                            # Tokenize ALL entity names (combine tokens from all entities)
                            # Example: ["Fuel Oil Pump", "Oil Cooler"] → ["fuel", "oil", "pump", "cooler"]
                            entity_tokens = []
                            for entity_name in entity_names:
                                tokens = [
                                    token.lower() for token in re.findall(r'\b\w+\b', entity_name)
                                    if len(token) > 2  # Keep only words longer than 2 chars
                                ]
                                entity_tokens.extend(tokens)
                            
                            # Remove duplicates while preserving order
                            entity_tokens = list(dict.fromkeys(entity_tokens))
                            
                            if not entity_tokens:
                                raise ValueError(f"No valid tokens in entity names: {entity_names}")
                            
                            logger.info(f"   🔍 BM25 search tokens (entities only): {entity_tokens}")
                            
                            # Prepare corpus: full content of each section (keep all words for matching entity)
                            corpus = []
                            section_mapping = []
                            for sec in regular_sections:
                                content = sec.get("content", "")
                                if content:
                                    # Use FULL text, tokenize (keep all words - entity names are important!)
                                    # No stopwords removal in corpus - "Oil Cooler" must match exactly
                                    tokens = [
                                        token.lower() for token in re.findall(r'\b\w+\b', content)
                                        if len(token) > 2
                                    ]
                                    corpus.append(tokens)
                                    section_mapping.append(sec)
                            
                            if not corpus:
                                raise ValueError("No valid entity sections for BM25")
                            
                            # Build BM25 index
                            bm25 = BM25Okapi(corpus)
                            
                            # Score all sections using ENTITY tokens (not full question)
                            scores = bm25.get_scores(entity_tokens)
                            
                            # Normalize scores to 0-1 range (BM25 scores are unbounded)
                            max_score = max(scores) if scores.any() else 1.0
                            normalized_scores = scores / max_score if max_score > 0 else scores
                            
                            # Filter: only keep sections with normalized score > 0.3
                            # (more lenient than 0.5 for embeddings, since BM25 scores differently)
                            scored_sections = []
                            for sec, norm_score in zip(section_mapping, normalized_scores):
                                if norm_score > 0.3:
                                    # Score = normalized_bm25 * 0.4 (low cap to not overpower qdrant)
                                    final_score = min(0.4, norm_score * 0.4)
                                    scored_sections.append((sec, final_score, norm_score))
                            
                            # Sort by BM25 score and take top 5 ONLY
                            scored_sections.sort(key=lambda x: x[2], reverse=True)
                            top_sections = scored_sections[:5]
                            
                            logger.info(f"   🔄 BM25 re-ranked entity sections: {len(regular_sections)} → {len(top_sections)} (kept top 5 with score > 0.3)")
                            if top_sections:
                                logger.info(f"   📊 Top section scores: {[f'{x[2]:.3f}' for x in top_sections[:3]]}")
                            
                            # FALLBACK: If NO sections passed threshold, keep top 3 by ACTUAL BM25 score
                            if not top_sections and section_mapping:
                                logger.warning(f"   ⚠️ All entity sections filtered out by BM25 threshold, using fallback with actual scores")
                                # Use all BM25 scores and keep top 3
                                fallback_sections = []
                                for sec, norm_score in zip(section_mapping, normalized_scores):
                                    # Use ACTUAL BM25 score * 0.3 (scaled to not overpower Qdrant)
                                    final_score = norm_score * 0.3
                                    fallback_sections.append((sec, final_score, norm_score))
                                
                                # Sort by actual BM25 score and keep top 3
                                fallback_sections.sort(key=lambda x: x[2], reverse=True)
                                top_sections = fallback_sections[:3]
                                avg_score = sum(x[2] for x in top_sections) / len(top_sections) if top_sections else 0
                                logger.info(f"   📌 Fallback: kept top {len(top_sections)} entity sections (avg BM25: {avg_score:.2f})")
                            
                            # Add BM25 re-ranked sections
                            for sec, final_score, bm25_score in top_sections:
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
    
    # Determine PRIMARY document (most anchor sections or highest total score)
    # Tables/schemas should primarily come from this doc to avoid context pollution
    from collections import Counter
    doc_scores = Counter()
    for a in anchors:
        doc_id = a.get("doc_id")
        if doc_id:
            doc_scores[doc_id] += a.get("score", 0.5)  # Sum scores per doc
    
    primary_doc_id = doc_scores.most_common(1)[0][0] if doc_scores else None
    
    logger.info(f"Anchor filtering: {len(anchor_keys)} sections, {len(anchor_doc_ids)} docs, primary_doc={primary_doc_id}, virtual={has_virtual_anchor}")
    
    enriched = []
    
    # Process text chunks (ONLY from anchor sections)
    text_hits = search_results.get("text", [])
    for hit in text_hits:
        key = (hit.get("doc_id"), hit.get("section_id"))
        if key not in anchor_keys:
            logger.debug(f"Skipping text chunk - not in anchor sections: {key}")
            continue
        
        item = await _fetch_and_expand_text_chunk(driver, hit, tool_ctx.vector_service)
        if item:
            enriched.append(item)
    
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
    
    # Sort by score
    deduplicated.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # HARD LIMITS: adaptive based on content type AND query intent
    sections = [i for i in deduplicated if i["type"] == "text_chunk"]
    tables = [i for i in deduplicated if i["type"] == "table_chunk"]
    schemas = [i for i in deduplicated if i["type"] == "schema"]
    
    # Get query intent from state
    query_intent = state.get("query_intent", "text")
    
    # Adaptive limits based on intent
    if query_intent == "table":
        # Table-focused query: prioritize tables
        max_sections = 3
        max_tables = 5
        max_schemas = 1
    elif query_intent == "schema":
        # Schema-focused query: prioritize schemas
        max_sections = 2
        max_tables = 0
        max_schemas = 5
    elif query_intent == "mixed":
        # Mixed query: allow both tables and schemas
        max_sections = 4
        max_tables = 3
        max_schemas = 2
    else:
        # Text query: NO tables/schemas allowed (strict intent enforcement)
        max_sections = 5
        max_tables = 0
        max_schemas = 0
    
    sections = sections[:max_sections]
    tables = tables[:max_tables]
    schemas = schemas[:max_schemas]
    
    # INTENT-BASED STRIPPING: enforce strict context filtering
    if query_intent == "text":
        # Text queries: ONLY text chunks, no tables/schemas
        # NOTE: If entity found via table_mentions, intent is auto-corrected to "mixed" upstream
        tables = []
        schemas = []
        logger.info("Intent=text: stripped all tables and schemas")
    elif query_intent == "schema":
        # Schema queries: strip tables (keep diagrams + supporting text)
        tables = []
        logger.info("Intent=schema: stripped all tables")
    # Note: mixed intent keeps both tables and schemas (no stripping)
    
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
    """Fetch full table with all chunks"""
    async with driver.session() as session:
        query = """
        MATCH (t:Table {id: $table_id})
        OPTIONAL MATCH (tc:TableChunk)-[:PART_OF]->(t)
        OPTIONAL MATCH (doc:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)-[:CONTAINS_TABLE]->(t)
        WITH t, doc, s, tc
        ORDER BY tc.chunk_index
        RETURN 
            t.id AS table_id,
            t.title AS table_title,
            t.caption AS caption,
            t.page_number AS page,
            t.rows AS rows,
            t.cols AS cols,
            t.file_path AS file_path,
            t.doc_id AS doc_id,
            doc.title AS doc_title,
            s.title AS section_title,
            collect(tc.text_preview) AS chunk_texts
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
                "file_path": hit.get("file_path"),  # Use file_path from hit
                "text": hit.get("text_preview", ""),
                "score": hit.get("score", 0),
                "source": hit.get("source"),  # Preserve source
            }
        
        # Log file_path for debugging
        file_path = record.get("file_path") or hit.get("file_path")
        if not file_path:
            logger.warning(f"Table {hit.get('table_id')} has no file_path in Neo4j or hit")
        
        # Combine all chunks
        chunk_texts = [t for t in record["chunk_texts"] if t]
        combined_text = "\n\n".join(chunk_texts) if chunk_texts else ""
        
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
            "file_path": file_path,  # Use Neo4j first, fallback to hit
            "text": combined_text,
            "score": hit.get("score", 0),
            "source": hit.get("source"),  # Preserve source (e.g., "entity_search")
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



# NODE 5: LLM REASONING

async def node_llm_reasoning(state: GraphState) -> GraphState:
    """
    Generate answer using enriched context.
    Context already includes expanded chunks from same sections.
    """
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
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
        for i, t in enumerate(tables, 1):
            table_ref = f"[TABLE{i}]"
            table_map[table_ref] = t
            tables_text += f"{table_ref} {t.get('title', 'Untitled')}\n"
            tables_text += f"Caption: {t.get('caption', '')}\n"
            tables_text += f"Document: {t.get('doc_title', 'Unknown')} (Page {t.get('page')})\n"
            tables_text += f"Data preview:\n{t['text'][:500]}...\n\n"
    
    # System prompt
    system_prompt = """You are a marine technical documentation answer generator.
Your only role is to produce factual answers strictly derived from supplied documentation.

LANGUAGE POLICY:
- Respond in exactly the same language as the user's question.

CONTEXT USE:
- You must rely solely on retrieved documentation context.
- If the required information is absent, output:
  "The provided documentation does not contain this information."

ANSWER BEHAVIOUR:
- Output must be factual, declarative, concise, and self-contained.
- Never ask user questions or offer cooperation, follow-ups, advice, or diagnostics.
- Never extrapolate beyond documented statements.

STRUCTURE AND RESOURCE SELECTION:
- You will be given AVAILABLE DIAGRAMS and AVAILABLE TABLES.
- Include a diagram/table only if it directly supports the answer.
- Reference diagrams/tables inline like [DIAGRAM1] or [TABLE2].
- Only referenced tables/diagrams will be shown.

INTENT-BASED RESOURCE CONSTRAINTS (CRITICAL):
- query_intent="text" → Prefer text citations, but you MAY include tables/diagrams if the answer is there
- query_intent="table" → You MUST include at least one [TABLE] reference
- query_intent="schema" → You MUST include at least one [DIAGRAM] reference
- query_intent="mixed" → You may include both tables and diagrams

CITATION RULES:
- Cite facts using: [Document | Section/Table/Diagram Title | Page X]
- Use a maximum of TWO textual citations.
- If the question is table-driven, one table reference is mandatory.
- If the question is diagram-driven, a diagram reference is mandatory.
- Never cite Table of Contents, Contents, or unrelated sections.
- If the answer is fully supported by one section, include only one citation.
- If multiple citations appear in your draft - remove extras before output.

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
        user_content = (
            f"Question: {state['question']}\n\n"
            f"=== DOCUMENTATION CONTEXT ===\n\n"
            f"{context_text}{schemas_text}{tables_text}"
        )
    else:
        user_content = state['question']
    
    messages.append({"role": "user", "content": user_content})
    
    # Generate answer
    resp = llm.invoke(messages)
    answer_text = resp.content
    
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
    
    # POST-GENERATION VALIDATOR: enforce intent-based constraints
    # Only for table/schema intents - text intent is flexible
    query_intent = state.get("query_intent", "text")
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
    # NOTE: text intent is NOT enforced - LLM can cite tables/diagrams if answer is there
    
    if needs_regeneration:
        logger.warning(f"⚠️ REGENERATION REQUIRED: {regeneration_reason}")
        
        # Add strict correction to messages
        correction_prompt = f"""
Your previous answer violated intent constraints:
{regeneration_reason}

REGENERATE your answer following these STRICT rules:
- Intent={query_intent}
- If intent=table: MUST reference at least one [TABLE] from available tables
- If intent=text: CANNOT reference any [TABLE] or [DIAGRAM]
- If intent=schema: MUST reference at least one [DIAGRAM] from available diagrams

Available resources:
{schemas_text if schema_map else ''}
{tables_text if table_map else ''}

Provide ONLY the corrected answer text."""
        
        messages.append({"role": "assistant", "content": answer_text})
        messages.append({"role": "user", "content": correction_prompt})
        
        # Regenerate
        resp_regenerated = llm.invoke(messages)
        answer_text = resp_regenerated.content
        
        # Re-parse references
        referenced_diagrams = set()
        referenced_tables = set()
        for ref in schema_map.keys():
            if ref in answer_text:
                referenced_diagrams.add(ref)
        for ref in table_map.keys():
            if ref in answer_text:
                referenced_tables.add(ref)
        
        logger.info(f"✅ Regenerated answer: {len(referenced_diagrams)} diagrams, {len(referenced_tables)} tables")
    
    # Build response - deduplicate citations by (doc_id, section_id, page)
    citations = []
    seen_citations = set()
    for c in chunks:
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
    
    # Only include diagrams that LLM referenced
    figures = []
    for ref in referenced_diagrams:
        s = schema_map[ref]
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
    
    # Only include tables that LLM referenced
    table_refs = []
    for ref in referenced_tables:
        t = table_map[ref]
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

def build_qa_graph(
    qdrant_client: QdrantClient,
    neo4j_driver: Driver,
    vector_service: Optional[Any] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_auth: Optional[tuple] = None,
) -> Any:
    """
    Build agentic Q&A workflow with Neo4j as tool.
    
    Flow: analyze → router_agent → execute_tools → build_context → llm_reasoning
    """
    # Set tool context
    tool_ctx.qdrant_client = qdrant_client
    tool_ctx.neo4j_driver = neo4j_driver
    tool_ctx.vector_service = vector_service
    tool_ctx.neo4j_uri = neo4j_uri
    tool_ctx.neo4j_auth = neo4j_auth
    
    graph = StateGraph(GraphState)
    
    # Wrap nodes with dependencies
    async def build_context_node(state: GraphState) -> GraphState:
        return await node_build_context(state, neo4j_driver)
    
    # Add nodes
    graph.add_node("analyze_question", node_analyze_question)
    graph.add_node("router_agent", node_router_agent)
    graph.add_node("execute_tools", node_execute_tools)
    graph.add_node("build_context", build_context_node)
    graph.add_node("llm_reasoning", node_llm_reasoning)
    
    # Build flow
    graph.set_entry_point("analyze_question")
    graph.add_edge("analyze_question", "router_agent")
    
    # Conditional edge from router_agent
    graph.add_conditional_edges(
        "router_agent",
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
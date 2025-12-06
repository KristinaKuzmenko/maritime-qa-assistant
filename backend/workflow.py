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
    vector_service: Optional[Any] = None  # VectorService instance
    owner: Optional[str] = None
    doc_ids: Optional[List[str]] = None


# Create global context (will be set by graph builder)
tool_ctx = ToolContext()


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
        
        if query_words or equipment_codes:
            async with tool_ctx.neo4j_driver.session() as session:
                # First: Direct search by equipment code (PU3, 7M2, etc.)
                if equipment_codes:
                    code_query = """
                    MATCH (e:Entity)
                    WHERE e.code IN $codes 
                       OR ANY(code IN $codes WHERE toUpper(e.name) CONTAINS code)
                    RETURN e.code AS code, e.name AS name
                    LIMIT 10
                    """
                    result = await session.run(code_query, {"codes": equipment_codes})
                    code_entities = await result.data()
                    logger.info(f"neo4j_entity_search: code search returned {len(code_entities)} entities")
                    
                    for ent in code_entities:
                        if ent["code"] and ent["code"] not in entity_ids:
                            entity_ids.append(ent["code"])
                            found_by_equipment_code = True
                            logger.info(f"Found entity by code from Neo4j: {ent['name']} ({ent['code']})")
                    
                    # If equipment codes detected but NOT found as entities, do fulltext search
                    if not found_by_equipment_code:
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
                        sections_data = await neo4j_fulltext_search(search_term, limit=10, min_score=0.3, include_content=True)
                        
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
                
                # Second: Search by name words (only if no equipment codes or codes were found)
                if query_words and not equipment_codes:
                    neo4j_entity_query = """
                    MATCH (e:Entity)
                    WHERE ANY(word IN $words WHERE toLower(e.name) CONTAINS word)
                    RETURN e.code AS code, e.name AS name
                    LIMIT 10
                    """
                    result = await session.run(neo4j_entity_query, {"words": list(query_words)})
                    neo4j_entities = await result.data()
                    
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
        
        # Expand with hierarchy (e.g., FW cooling → cooling water system)
        expanded_ids = extractor.expand_entity_ids(entity_ids)
        
        logger.info(
            f"neo4j_entity_search: extracted {len(entity_ids)} entities, "
            f"expanded to {len(expanded_ids)}: {expanded_ids[:5]}..."
        )
        
        # Get doc_ids filter for Neo4j queries
        doc_ids_filter = tool_ctx.doc_ids if tool_ctx.doc_ids else None
        
        results = {
            "entities": expanded_ids,
            "entity_names": extraction.get("system_names", []) + extraction.get("component_names", []),
            "sections": [],
            "tables": [],
            "schemas": [],
        }
        
        # Query Neo4j for sections describing these entities
        # Note: Section.content stores full text, no need to fetch from Qdrant
        async with tool_ctx.neo4j_driver.session() as session:
            # Find sections via DESCRIBES relationship - include content!
            # Add doc_ids filter if specified
            section_query = """
            UNWIND $entity_ids AS eid
            MATCH (e:Entity {code: eid})<-[:DESCRIBES]-(s:Section)
            WHERE $doc_ids IS NULL OR s.doc_id IN $doc_ids
            OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s)
            RETURN DISTINCT 
                s.id AS section_id,
                s.title AS section_title,
                s.content AS content,
                s.page_start AS page_start,
                s.page_end AS page_end,
                s.doc_id AS doc_id,
                d.title AS doc_title,
                e.code AS entity_code,
                e.name AS entity_name
            LIMIT 10
            """
            result = await session.run(section_query, {"entity_ids": expanded_ids, "doc_ids": doc_ids_filter})
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
                }
                for r in section_records
            ]
            
            # Find tables via MENTIONS relationship
            if include_tables:
                table_query = """
                UNWIND $entity_ids AS eid
                MATCH (e:Entity {code: eid})<-[:MENTIONS]-(t:Table)
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
                LIMIT 10
                """
                result = await session.run(table_query, {"entity_ids": expanded_ids, "doc_ids": doc_ids_filter})
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
                MATCH (e:Entity {code: eid})<-[:DEPICTS]-(sc:Schema)
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
                LIMIT 5
                """
                result = await session.run(schema_query, {"entity_ids": expanded_ids, "doc_ids": doc_ids_filter})
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
- "table" - seeking tabular data, specifications, parameters, technical values in TABLE format
- "schema" - seeking diagrams, schematics, figures, DRAWINGS, visual representations, layout images
- "mixed" - needs both semantic search AND graph traversal (e.g., "find all X in section Y", structural queries)

IMPORTANT: If question mentions "drawing", "drawings", "diagram", "scheme", "figure", "layout" → classify as "schema"
These words indicate user wants VISUAL content, even if they ask about dimensions/configuration shown in drawings.

Examples:
- "How does the fuel system work?" → text
- "What are the engine specifications?" → table
- "Show me the water heater diagram" → schema
- "Give me the scheme of cooling system" → schema
- "What does the control panel look like according to drawings?" → schema
- "Dimensions of CP-1 from final drawings" → schema (user wants the DRAWING showing dimensions)
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
    intent = state["query_intent"]
    question = state["question"]
    anchors = state.get("anchor_sections", [])
    
    # Build anchor info for agent
    anchor_info = ""
    if anchors:
        anchor_info = "\n\nANCHOR SECTIONS (focus on these):\n"
        for a in anchors:
            anchor_info += f"- Section {a['section_id']} (doc: {a['doc_id']}, score: {a['score']:.2f})\n"
    
    # Build system prompt with intent guidance
    system_prompt = f"""{GRAPH_SCHEMA_PROMPT}

You are a routing agent for maritime technical documentation Q&A system.
Select the best tools to answer the user's question.

DETECTED INTENT: {intent}

INTENT-BASED PARAMETER SELECTION:
When using neo4j_entity_search, set include_tables and include_schemas based on intent:
- intent="text" → include_tables=False, include_schemas=False (only sections)
- intent="table" → include_tables=True, include_schemas=False
- intent="schema" → include_tables=False, include_schemas=True
- intent="mixed" → include_tables=True, include_schemas=True

TOOL SELECTION GUIDE:

1. **neo4j_query** - For STRUCTURAL queries by section/chapter NUMBER:
   Use when question references a SPECIFIC section NUMBER like "section 4.4", "chapter 3"
   Examples: "tables from section 4.4", "content of chapter 3.2", "all tables in section 5.1"
   → Query Section by title containing the number, then find related Tables/Schemas

2. **neo4j_entity_search** - When question has EQUIPMENT CODE (alphanumeric tag):
   Codes are SHORT like: 7M2, P-101, V-205, PU3, TK-102, CP-1
   Examples: "pump (7M2)", "valve V-205", "function of PU3", "CP-1 drawings"
   DO NOT use for equipment NAMES without codes: "fuel pump", "main engine"
   IMPORTANT: Set include_tables/include_schemas based on DETECTED INTENT above!

3. **qdrant_search_text** - DEFAULT for general questions:
   Descriptions, procedures, explanations, named equipment without codes
   Examples: "How does X work", "What is cooling system", "explain fuel pump"

4. **qdrant_search_tables** - For specs/parameters:
   Temperatures, pressures, specifications → combine with qdrant_search_text

5. **qdrant_search_schemas** - For diagrams:
   "Show diagram", "Where is X located"

DECISION:
- Section NUMBER (4.4, 3.2)? → neo4j_query
- Equipment CODE (7M2, P-101)? → neo4j_entity_search  
- Specs/temperatures? → qdrant_search_tables + text
- General? → qdrant_search_text

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
                
                # Add sections with full content from Neo4j
                for sec in entity_sections:
                    content = sec.get("content", "")
                    if content:
                        search_results["text"].append({
                            "type": "text_chunk",
                            "score": 0.85,  # High score for entity match
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
                        "score": 0.85,
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
                        "score": 0.85,
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
                
                entities_found = result.get("entity_names", [])
                entities_ids = result.get("entities", [])
                logger.info(f"   ✅ neo4j_entity_search results:")
                logger.info(f"      Extracted entities: {entities_ids[:5]}{'...' if len(entities_ids) > 5 else ''}")
                logger.info(f"      Entity names: {entities_found}")
                logger.info(f"      Found: {len(entity_sections)} sections, {len(entity_tables)} tables, {len(entity_schemas)} schemas")
                
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
        similarity = max(h.get("score", 0) for h in hits)
        
        # Get importance score from Neo4j (default 0.5)
        importance = importance_scores.get(section_id, 0.5)
        
        # Combined score: similarity * 0.7 + importance * 0.2
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
    anchor_keys = {(a["doc_id"], a["section_id"]) for a in anchors}
    anchor_doc_ids = {a["doc_id"] for a in anchors}
    anchor_section_ids = {a["section_id"] for a in anchors}
    
    # Determine PRIMARY document (most anchor sections or highest total score)
    # Tables/schemas should primarily come from this doc to avoid context pollution
    from collections import Counter
    doc_scores = Counter()
    for a in anchors:
        doc_id = a.get("doc_id")
        if doc_id:
            doc_scores[doc_id] += a.get("score", 0.5)  # Sum scores per doc
    
    primary_doc_id = doc_scores.most_common(1)[0][0] if doc_scores else None
    
    logger.info(f"Anchor filtering: {len(anchor_keys)} sections, {len(anchor_doc_ids)} docs, primary_doc={primary_doc_id}")
    
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
        # Strict: only from primary doc; fallback to any anchor doc if no primary
        if primary_doc_id and hit_doc_id != primary_doc_id:
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
        if primary_doc_id and hit_doc_id != primary_doc_id:
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
    
    # Process Neo4j results (direct Cypher query results)
    # These are HIGH PRIORITY - user explicitly asked for this data via structured query
    # Skip anchor filtering if anchors are empty (means no text search was done)
    skip_anchor_filter = len(anchor_doc_ids) == 0 and len(anchor_section_ids) == 0
    
    for record in neo4j_results:
        if "error" in record:
            continue
        
        # Check if record is from anchor docs/sections (unless anchors are empty)
        if not skip_anchor_filter:
            rec_doc_id = record.get("doc_id")
            rec_section_id = record.get("section_id")
            
            if rec_doc_id not in anchor_doc_ids and rec_section_id not in anchor_section_ids:
                logger.debug(f"Skipping Neo4j record - not in anchor scope")
                continue
        
        # Try to identify type from record
        if "table_id" in record or "id" in record and "page_number" in record:
            # This looks like a table record (from neo4j_query)
            item = await _neo4j_record_to_table(driver, record)
            if item:
                enriched.append(item)
                logger.info(f"Added table from neo4j_query: {item.get('table_id')} (p{item.get('page')})")
        
        elif "section_id" in record or "content" in record:
            item = await _neo4j_record_to_text_chunk(driver, record)
            if item:
                enriched.append(item)
        
        elif "schema_id" in record:
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
        max_tables = 1
        max_schemas = 5
    elif sections:
        # Text query with sections: allow reasonable tables/schemas
        max_sections = 5
        max_tables = 3   # Allow 3 tables - they provide specs
        max_schemas = 2  # Allow 2 schemas - diagrams help understanding
    else:
        # No sections found, allow more tables/schemas
        max_sections = 3
        max_tables = 3
        max_schemas = 3
    
    sections = sections[:max_sections]
    tables = tables[:max_tables]
    schemas = schemas[:max_schemas]
    
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
            "score": 0.9,  # High score for graph results
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
            "score": 0.9,
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
            "score": 0.9,
        }
    return None



# NODE 5: LLM REASONING

def node_llm_reasoning(state: GraphState) -> GraphState:
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
    
    if tables:
        context_text += "=== TABLES ===\n\n"
        for i, t in enumerate(tables, 1):
            context_text += f"[TABLE{i}] {t.get('title', 'Untitled')}\n"
            context_text += f"Caption: {t.get('caption', '')}\n"
            context_text += f"Document: {t.get('doc_title', 'Unknown')} (Page {t.get('page')})\n"
            context_text += f"{t['text']}\n\n"
    
    schemas_text = ""
    if schemas:
        schemas_text += "=== DIAGRAMS (will be displayed to user as images) ===\n\n"
        for i, s in enumerate(schemas, 1):
            schemas_text += f"[DIAGRAM{i}] {s.get('title', 'Figure')}\n"
            schemas_text += f"Caption: {s.get('caption', '')}\n"
            if s.get('llm_summary'):
                schemas_text += f"Description: {s.get('llm_summary')}\n"
            schemas_text += f"Document: {s.get('doc_title', 'Unknown')} (Page {s.get('page')})\n"
            schemas_text += f"NOTE: This diagram image will be shown to the user below your answer.\n\n"
    
    # System prompt
    system_prompt = """You are an expert marine technical documentation assistant.
Your primary role is to answer questions about maritime technical documentation.

CONVERSATION RULES:
- For greetings and casual conversation, respond naturally and friendly
- Remember context from the conversation (user's name, previous questions, etc.)
- For technical questions, use the provided documentation context

IMPORTANT - DIAGRAMS AND TABLES IN CONTEXT:
- When DIAGRAMS or TABLES are provided in context, they WILL BE DISPLAYED to the user as images
- These ARE the drawings/diagrams/figures the user is asking about
- DO NOT say "drawings are not provided" or "I don't have the drawings" if DIAGRAMS section exists
- Instead, REFERENCE these diagrams: "The drawings provided show..." or "See the diagrams below..."
- If a diagram shows what user asked for (e.g., control panel drawing), describe what it shows based on title/caption

CITATION RULES (for technical answers):
- Cite facts using: [Document Name | Section/Table/Diagram: Title | Page X]
- If documentation context is empty or insufficient for a technical question, say what information is missing
- Reference diagrams by their title/caption when answering diagram-related questions

You can answer general questions without documentation context."""
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Build user message - include context only if available
    if context_text or schemas_text:
        user_content = (
            f"Question: {state['question']}\n\n"
            f"=== DOCUMENTATION CONTEXT ===\n\n"
            f"{context_text}{schemas_text}"
        )
    else:
        user_content = state['question']
    
    messages.append({"role": "user", "content": user_content})
    
    # Generate answer
    resp = llm.invoke(messages)
    
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
    
    figures = []
    for s in schemas:
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
    
    table_refs = []
    for t in tables:
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
    
    state["answer"] = {
        "answer_text": resp.content,
        "citations": citations,
        "figures": figures,
        "tables": table_refs,
    }
    
    # Log final answer stats
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ANSWER GENERATED")
    logger.info(f"   Answer length: {len(resp.content)} chars")
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
) -> Any:
    """
    Build agentic Q&A workflow with Neo4j as tool.
    
    Flow: analyze → router_agent → execute_tools → build_context → llm_reasoning
    """
    # Set tool context
    tool_ctx.qdrant_client = qdrant_client
    tool_ctx.neo4j_driver = neo4j_driver
    tool_ctx.vector_service = vector_service
    
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
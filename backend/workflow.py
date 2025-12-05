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

GRAPH SCHEMA (IMPORTANT):
- (d:Document {id, title})
  - (d)-[:HAS_CHAPTER]->(c:Chapter {id, title, number})
  - (c)-[:HAS_SECTION]->(s:Section {id, title, doc_id, page_start, page_end, content})
  
- (s:Section)-[:CONTAINS_TABLE]->(t:Table {
      id, doc_id, page_number, title, caption, rows, cols, file_path, thumbnail_path, csv_path
  })
  
- (tc:TableChunk {id, parent_table_id, chunk_index, text_preview, text_length})-[:PART_OF]->(t:Table)

- (s:Section)-[:CONTAINS_SCHEMA]->(sc:Schema {
      id, doc_id, page_number, title, caption, file_path, thumbnail_path
  })

RELATIONSHIPS:
- Document → Chapter: [:HAS_CHAPTER]
- Chapter → Section: [:HAS_SECTION]
- Section → Table: [:CONTAINS_TABLE]
- Section → Schema: [:CONTAINS_SCHEMA]
- TableChunk → Table: [:PART_OF]

CRITICAL RULES:
- Use ONLY read-only Cypher: MATCH, OPTIONAL MATCH, RETURN, WHERE, ORDER BY, LIMIT
- DO NOT use: CREATE, MERGE, DELETE, REMOVE, SET, DROP, CALL (write procedures)
- ALWAYS use LIMIT <= 5 in your queries to prevent overload
- Prefer to:
  - Get sections by id (section_id) when provided
  - Get tables/schemas via CONTAINS_TABLE / CONTAINS_SCHEMA relationships
  - List items by page_number / doc_id when question is about specific page
- Return only needed fields: id, title, page_number, file_path, caption, doc_id, content
- If question doesn't need structural graph info, DO NOT call neo4j_query tool

EXAMPLE GOOD QUERIES:
```cypher
// Get section with its tables
MATCH (s:Section {id: $section_id})-[:CONTAINS_TABLE]->(t:Table)
RETURN s, t LIMIT 5

// Find tables on specific page
MATCH (t:Table {page_number: $page, doc_id: $doc_id})
RETURN t LIMIT 5
```

EXAMPLE BAD QUERIES:
```cypher
// ❌ NO LIMIT - will return too much data
MATCH (s:Section) RETURN s

// ❌ LIMIT too high
MATCH (t:Table) RETURN t LIMIT 100
```
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
    
    # Tool results (raw)
    qdrant_results: Dict[str, List[Dict[str, Any]]]  # {text: [...], tables: [...], schemas: [...]}
    neo4j_results: List[Dict[str, Any]]
    
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
        
        # Build filters for Qdrant/Neo4j
        filter_conditions = []
        if tool_ctx.owner:
            filter_conditions.append(
                FieldCondition(key="owner", match=MatchValue(value=tool_ctx.owner))
            )
        if tool_ctx.doc_ids:
            filter_conditions.append(
                FieldCondition(key="doc_id", match=MatchAny(any=tool_ctx.doc_ids))
            )
        
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
            section_query = """
            UNWIND $entity_ids AS eid
            MATCH (e:Entity {code: eid})<-[:DESCRIBES]-(s:Section)
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
            result = await session.run(section_query, {"entity_ids": expanded_ids})
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
                    t.doc_id AS doc_id,
                    d.title AS doc_title,
                    e.code AS entity_code,
                    e.name AS entity_name,
                    chunk_previews
                LIMIT 10
                """
                result = await session.run(table_query, {"entity_ids": expanded_ids})
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
                result = await session.run(schema_query, {"entity_ids": expanded_ids})
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
- "table" - seeking tabular data, specifications, parameters, technical values
- "schema" - seeking diagrams, schematics, figures, drawings, visual representations
- "mixed" - needs both semantic search AND graph traversal (e.g., "find all X in section Y", structural queries)

Examples:
- "How does the fuel system work?" → text
- "What are the engine specifications?" → table
- "Show me the water heater diagram" → schema
- "Give me the scheme of cooling system" → schema
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

You are a routing agent for maritime technical documentation system.
Your job: decide which tools to call to answer the user's question.

USER'S QUESTION INTENT: {intent}

TOOL SELECTION STRATEGY:

1. FOR SEMANTIC/CONTENT QUESTIONS (what, how, why, explain):
   - Use qdrant_search_text for general information, procedures, explanations
   - Use qdrant_search_tables for specifications, parameters, technical data
   - Use qdrant_search_schemas for diagrams, schematics, figures

2. FOR ENTITY-FOCUSED QUESTIONS (neo4j_entity_search):
   ✅ USE neo4j_entity_search ONLY when question asks about:
      - SPECIFIC COMPONENTS: "fuel oil pump", "cooling water valve", "ballast tank"
      - EQUIPMENT CODES: "P-101", "V-205", "TK-102"
      - RELATIONSHIPS between entities: "what pumps connect to tank X"
      - COMPONENT LISTS in a system: "list all valves in FO system"
   
   ❌ DO NOT use neo4j_entity_search for:
      - General questions: "What is SOLAS?", "Explain maintenance procedures"
      - System overviews: "How does fuel oil system work?" (use qdrant_search_text)
      - Abbreviation definitions: "What does FO mean?"
      - Procedures without specific components: "How to start the engine?"
   
   Examples:
   - "What pumps are in the fuel oil system?" → neo4j_entity_search ✅ (specific component type)
   - "Show P-101 specifications" → neo4j_entity_search ✅ (equipment code)
   - "How does the fuel oil system work?" → qdrant_search_text ✅ (general, no specific component)
   - "FO system diagram" → qdrant_search_schemas ✅ (diagram search, not entity relations)

3. FOR STRUCTURAL/NAVIGATIONAL QUESTIONS (where, which section, on page X):
   - Use neo4j_query to traverse document structure
   - Example: "What tables are in section 4.2?" → neo4j_query
   - Example: "Show me schemas on page 15" → neo4j_query
   - IMPORTANT: Always use LIMIT <= 5 in Neo4j queries

4. COMBINING TOOLS:
   - For complex questions, combine semantic + entity search
   - Example: "Find fuel oil pump maintenance schedule" 
     → neo4j_entity_search (find pump-related content) + qdrant_search_text (maintenance info)

CURRENT QUESTION: "{question}"{anchor_info}

Based on the question and intent, decide which tools to call.
Prefer qdrant_search_* for most questions. Use neo4j_entity_search only when specifically justified.
"""
    
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
    messages = state["messages"]
    last_message = messages[-1]
    
    # Safely get tool calls
    tool_calls = get_tool_calls(last_message)
    
    if not tool_calls:
        logger.warning("No tool calls from agent")
        state["qdrant_results"] = {"text": [], "tables": [], "schemas": []}
        state["neo4j_results"] = []
        state["anchor_sections"] = []
        return state
    
    # Execute each tool call
    qdrant_results = {"text": [], "tables": [], "schemas": []}
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
                qdrant_results["text"].extend(result)
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
                qdrant_results["tables"].extend(result)
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
                qdrant_results["schemas"].extend(result)
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
                        qdrant_results["text"].append({
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
                    
                    qdrant_results["tables"].append({
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
                        "source": "entity_search",
                        "matched_entity": tbl.get("matched_entity"),
                    })
                
                # Add entity-matched schemas with text_context and llm_summary
                for sch in entity_schemas:
                    qdrant_results["schemas"].append({
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
    
    state["qdrant_results"] = qdrant_results
    state["neo4j_results"] = neo4j_results
    state["messages"].extend(tool_messages)
    
    # Log summary of collected context
    logger.info(f"\n📊 TOOL EXECUTION SUMMARY:")
    logger.info(f"   Text chunks: {len(qdrant_results['text'])}")
    logger.info(f"   Tables: {len(qdrant_results['tables'])}")
    logger.info(f"   Schemas: {len(qdrant_results['schemas'])}")
    logger.info(f"   Neo4j records: {len(neo4j_results)}")
    
    # FALLBACK: Neo4j fulltext search if Qdrant results are poor
    text_results = qdrant_results.get("text", [])
    high_quality_results = [r for r in text_results if r.get("score", 0) > 0.3]
    
    if len(high_quality_results) < 2 and state["query_intent"] in ["text", "mixed"]:
        logger.info(f"⚠️  Poor Qdrant results ({len(high_quality_results)} with score > 0.3), trying Neo4j fulltext fallback")
        
        try:
            # Use Neo4j fulltext to find relevant section_ids
            query = state["question"]
            cypher = """
            CALL db.index.fulltext.queryNodes('sectionSearch', $query) 
            YIELD node, score
            WHERE node:Section AND score > 0.5
            RETURN node.id AS section_id,
                   node.title AS section_title,
                   node.page_start AS page,
                   node.doc_id AS doc_id,
                   score
            ORDER BY score DESC
            LIMIT 3
            """
            
            async with tool_ctx.neo4j_driver.session() as session:
                result = await session.run(cypher, {"query": query})
                fulltext_results = await result.data()
                
                if fulltext_results:
                    logger.info(f"✅ Neo4j fulltext found {len(fulltext_results)} sections")
                    
                    # Fetch chunks from Qdrant for these sections
                    for ft_result in fulltext_results:
                        section_id = ft_result["section_id"]
                        
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
                            qdrant_results["text"].append({
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
                    
                    logger.info(f"Added {len(qdrant_results['text']) - len(text_results)} chunks from fulltext sections")
                else:
                    logger.info("Neo4j fulltext: no results")
                    
        except Exception as e:
            logger.error(f"Neo4j fulltext fallback failed: {e}")
    
    # SELECT ANCHOR SECTIONS
    anchor_sections = _select_anchor_sections(
        qdrant_results.get("text", []),
        max_sections=5  # Increased to 5 for better coverage
    )
    state["anchor_sections"] = anchor_sections
    
    logger.info(
        f"Tools executed: "
        f"text={len(qdrant_results['text'])}, "
        f"tables={len(qdrant_results['tables'])}, "
        f"schemas={len(qdrant_results['schemas'])}, "
        f"neo4j={len(neo4j_results)}, "
        f"anchors={len(anchor_sections)}"
    )
    
    return state


def _select_anchor_sections(text_hits: List[Dict[str, Any]], max_sections: int = 3) -> List[Dict[str, Any]]:
    """
    Select top anchor sections based on best score per section.
    Groups text chunks by (doc_id, section_id) and picks top N sections.
    """
    from collections import defaultdict
    
    groups = defaultdict(list)
    for h in text_hits:
        key = (h.get("doc_id"), h.get("section_id"))
        groups[key].append(h)
    
    scored = []
    for (doc_id, section_id), hits in groups.items():
        # Take max score among hits in this section
        best_score = max(h.get("score", 0) for h in hits)
        scored.append({
            "doc_id": doc_id,
            "section_id": section_id,
            "score": best_score,
        })
    
    # Sort by score and take top N
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    selected = scored[:max_sections]
    
    logger.info(f"Selected {len(selected)} anchor sections from {len(scored)} candidates")
    
    return selected



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
    qdrant_results = state["qdrant_results"]
    neo4j_results = state["neo4j_results"]
    anchors = state.get("anchor_sections", [])
    
    # Build anchor filter sets
    anchor_keys = {(a["doc_id"], a["section_id"]) for a in anchors}
    anchor_doc_ids = {a["doc_id"] for a in anchors}
    anchor_section_ids = {a["section_id"] for a in anchors}
    
    logger.info(f"Anchor filtering: {len(anchor_keys)} sections, {len(anchor_doc_ids)} docs")
    
    enriched = []
    
    # Process Qdrant text chunks (ONLY from anchor sections)
    text_hits = qdrant_results.get("text", [])
    for hit in text_hits:
        key = (hit.get("doc_id"), hit.get("section_id"))
        if key not in anchor_keys:
            logger.debug(f"Skipping text chunk - not in anchor sections: {key}")
            continue
        
        item = await _fetch_and_expand_text_chunk(driver, hit, tool_ctx.vector_service)
        if item:
            enriched.append(item)
    
    # Process Qdrant tables (ONLY from anchor docs)
    for hit in qdrant_results.get("tables", []):
        if hit.get("doc_id") not in anchor_doc_ids:
            logger.debug(f"Skipping table - not in anchor docs: {hit.get('table_id')}")
            continue
        
        item = await _fetch_table_full(driver, hit)
        if item:
            enriched.append(item)
    
    # Process Qdrant schemas
    # If no anchor sections (schema-only query), process all schemas
    # Otherwise filter by anchor docs/sections
    for hit in qdrant_results.get("schemas", []):
        if anchor_doc_ids:  # Only filter if anchors exist
            if (hit.get("section_id") not in anchor_section_ids and 
                hit.get("doc_id") not in anchor_doc_ids):
                logger.debug(f"Skipping schema - not in anchor docs/sections: {hit.get('schema_id')}")
                continue
        
        item = await _fetch_schema_full(driver, hit)
        if item:
            enriched.append(item)
    
    # Process Neo4j results (filter by anchor docs/sections)
    for record in neo4j_results:
        if "error" in record:
            continue
        
        # Check if record is from anchor docs/sections
        rec_doc_id = record.get("doc_id")
        rec_section_id = record.get("section_id")
        
        if rec_doc_id not in anchor_doc_ids and rec_section_id not in anchor_section_ids:
            logger.debug(f"Skipping Neo4j record - not in anchor scope")
            continue
        
        # Try to identify type from record
        if "section_id" in record or "content" in record:
            item = await _neo4j_record_to_text_chunk(driver, record)
            if item:
                enriched.append(item)
        
        elif "table_id" in record:
            item = await _neo4j_record_to_table(driver, record)
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
    
    # HARD LIMITS: adaptive based on content type
    # If we have text chunks, prioritize them and limit schemas to prevent context pollution
    sections = [i for i in deduplicated if i["type"] == "text_chunk"]
    tables = [i for i in deduplicated if i["type"] == "table_chunk"]
    schemas = [i for i in deduplicated if i["type"] == "schema"]
    
    # Adaptive limits: if we have text, limit schemas; otherwise allow more schemas
    if sections:
        # Mixed/text query: prioritize text, limit schemas to 1
        max_sections = 5
        max_tables = 3
        max_schemas = 1  # Only 1 schema when we have text
    else:
        # Schema-only query: allow more schemas
        max_sections = 3
        max_tables = 3
        max_schemas = 3
    
    sections = sections[:max_sections]
    tables = tables[:max_tables]
    schemas = schemas[:max_schemas]
    
    final_context = sections + tables + schemas
    
    state["enriched_context"] = final_context
    
    logger.info(
        f"Context built with hard limits: "
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
        
        if not record or not record.get("doc_title"):
            # Fallback
            return {
                "type": "table_chunk",
                "table_id": hit.get("table_id"),
                "doc_id": hit.get("doc_id"),
                "doc_title": hit.get("doc_title", "Unknown"),
                "title": hit.get("table_title", ""),
                "caption": hit.get("table_caption", ""),
                "page": hit.get("page"),
                "file_path": None,
                "text": hit.get("text_preview", ""),
                "score": hit.get("score", 0),
            }
        
        # Combine all chunks
        chunk_texts = [t for t in record["chunk_texts"] if t]
        combined_text = "\n\n".join(chunk_texts) if chunk_texts else ""
        
        return {
            "type": "table_chunk",
            "table_id": record["table_id"],
            "doc_id": record["doc_id"],
            "doc_title": record["doc_title"],
            "section_title": record.get("section_title"),
            "title": record["table_title"],
            "caption": record.get("caption", ""),
            "page": record["page"],
            "rows": record["rows"],
            "cols": record["cols"],
            "file_path": record["file_path"],
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
    if "table_id" in record:
        return {
            "type": "table_chunk",
            "table_id": record.get("table_id"),
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
        schemas_text += "=== DIAGRAMS ===\n\n"
        for i, s in enumerate(schemas, 1):
            schemas_text += f"[DIAGRAM{i}] {s.get('title', 'Figure')}\n"
            schemas_text += f"Caption: {s.get('caption', '')}\n"
            schemas_text += f"Document: {s.get('doc_title', 'Unknown')} (Page {s.get('page')})\n\n"
    
    # System prompt
    system_prompt = """You are an expert marine technical documentation assistant.
Your primary role is to answer questions about maritime technical documentation.

CONVERSATION RULES:
- For greetings and casual conversation, respond naturally and friendly
- Remember context from the conversation (user's name, previous questions, etc.)
- For technical questions, use the provided documentation context

CITATION RULES (for technical answers):
- Cite facts using: [Document Name | Section/Table/Diagram: Title | Page X]
- If documentation context is empty or insufficient for a technical question, say what information is missing

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
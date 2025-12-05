# Q&A Workflow (Agentic RAG)

## Overview

The Maritime QA Assistant uses an **agentic LangGraph workflow** to answer questions. Unlike traditional RAG pipelines where retrieval is hardcoded, our agent **decides at runtime** which tools to use based on the question.

**Key Features:**
- Agent decides whether to retrieve context or answer directly
- For greetings and general questions → no retrieval needed
- For technical questions → selects appropriate tools (Qdrant, Neo4j, entity search)
- Supports entity-based graph traversal for specific components

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Q&A WORKFLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐     ┌──────────────┐                                │
│   │   Analyze    │────▶│    Router    │                                │
│   │   Question   │     │    Agent     │                                │
│   └──────────────┘     └──────┬───────┘                                │
│          │                    │                                         │
│          │           ┌───────┴────────┐                                │
│          ▼           │                │                                │
│   ┌──────────────┐   │ (has tools)    │ (no tools - direct answer)    │
│   │   Intent:    │   ▼                │                                │
│   │ text/table/  │ ┌─────────────┐    │                                │
│   │ schema/mixed │ │Execute Tools│    │                                │
│   └──────────────┘ └──────┬──────┘    │                                │
│                           │           │                                │
│                           └─────┬─────┘                                │
│                                 ▼                                       │
│                   ┌──────────────────────────────────────────┐         │
│                   │              Build Context               │         │
│                   │  - Merge Qdrant + Neo4j results          │         │
│                   │  - Expand with neighbor chunks           │         │
│                   │  - Deduplicate and rank                  │         │
│                   │  - (empty if no tools called)            │         │
│                   └──────────────────────────────────────────┘         │
│                                          │                              │
│                                          ▼                              │
│                   ┌──────────────────────────────────────────┐         │
│                   │           LLM Reasoning                  │         │
│                   │  - Generate answer with citations        │         │
│                   │  - Handle general conversation           │         │
│                   │  - Answer without context if appropriate │         │
│                   └──────────────────────────────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Decision: Context vs Direct Answer

The router agent can decide **not to call any tools** for questions that don't require document retrieval:

**Direct Answer (no retrieval):**
- Greetings: "Hello", "How are you?"
- General knowledge: "What is SOLAS?", "Explain IMO conventions"
- Conversational: "Thank you", "Can you help me?"
- Follow-ups that use chat history context

**Requires Retrieval:**
- Technical questions about specific documents
- Component/equipment queries
- Procedure lookups
- Specification requests

---

## Nodes

### 1. Analyze Question (`node_analyze_question`)

**Purpose:** Classify the question intent to guide tool selection.

**Classification Categories:**
- `text` - seeking textual information, procedures, explanations
- `table` - seeking tabular data, specifications, parameters
- `schema` - seeking diagrams, schematics, figures
- `mixed` - needs both semantic search AND graph traversal

**Process:**
1. Send question to LLM with classification prompt
2. LLM returns one of: `text`, `table`, `schema`, `mixed`
3. Store intent in state for router agent

**Example Classifications:**
| Question | Intent |
|----------|--------|
| "How does the fuel system work?" | `text` |
| "What are the engine specifications?" | `table` |
| "Show me the cooling water diagram" | `schema` |
| "List all tables in chapter 3" | `mixed` |

---

### 2. Router Agent (`node_router_agent`)

**Purpose:** LLM agent decides which tools to call based on question and intent.

**Available Tools:**

| Tool | Description | When to Use |
|------|-------------|-------------|
| `qdrant_search_text` | Semantic search in text chunks | General knowledge questions |
| `qdrant_search_tables` | Semantic search in tables | Specifications, parameters |
| `qdrant_search_schemas` | Semantic search in diagrams | Visual representations |
| `neo4j_query` | Execute Cypher query | Structural queries, page-specific |
| `neo4j_entity_search` | Entity-based graph traversal | Specific components, equipment codes |

**System Prompt Guidance:**

The router agent receives:
- Graph schema (nodes, relationships, properties)
- Query intent classification
- Examples of good/bad Cypher queries
- Rules for when to use each tool

**Tool Selection Logic:**

```
Intent = "text"
  → qdrant_search_text (required)
  → neo4j_entity_search (if specific component mentioned)

Intent = "table"  
  → qdrant_search_tables (required)
  → qdrant_search_text (optional, for context)

Intent = "schema"
  → qdrant_search_schemas (required)
  → qdrant_search_text (optional, for descriptions)

Intent = "mixed"
  → Multiple tools based on question structure
  → neo4j_query for structural/page-specific queries
```

---

### 3. Execute Tools (`node_execute_tools`)

**Purpose:** Execute tool calls made by the router agent.

**Process:**

1. Extract tool calls from agent's response
2. Execute each tool in sequence
3. Collect results into structured state:
   - `qdrant_results.text` - text chunks
   - `qdrant_results.tables` - table chunks
   - `qdrant_results.schemas` - schema metadata
   - `neo4j_results` - raw Cypher results
4. Create `ToolMessage` responses for agent conversation

**Fallback Logic:**

If Qdrant results are poor (< 2 results with score > 0.3):
- Trigger Neo4j fulltext search fallback
- Query `sectionSearch` fulltext index
- Retrieve section content directly from graph

**Logging:**

Each tool execution is logged with:
- Tool name and arguments
- Result count
- Sample scores/keys for debugging

---

### 4. Build Context (`node_build_context`)

**Purpose:** Merge and enrich results from all sources.

**Process:**

#### Step 4.1: Text Chunk Processing

For each text chunk from Qdrant:
1. **Neighbor Expansion:**
   - Fetch ±1 neighbor chunks from same section
   - Query Qdrant by section_id + chunk_index range
   - Sort by character position
   
2. **Overlap Removal:**
   ```python
   if chunk["char_start"] < last_end:
       overlap_size = last_end - chunk["char_start"]
       combined_text += chunk["text"][overlap_size:]
   else:
       combined_text += chunk["text"]
   ```

3. **Enrichment:**
   - Add chapter title from Neo4j
   - Mark as "expanded" if neighbors added
   - Calculate specific page from character position

#### Step 4.2: Table Processing

For each table from Qdrant or entity search:
1. Include CSV path for data access
2. Preserve table metadata (rows, cols, caption)
3. Include text_preview for LLM context

#### Step 4.3: Schema Processing

For each schema:
1. Include file path and thumbnail
2. Add text_context (surrounding text)
3. Add llm_summary if available

#### Step 4.4: Deduplication

- Remove duplicate chunks by (section_id, chunk_index)
- Remove duplicate tables by table_id
- Remove duplicate schemas by schema_id

**Output:** `enriched_context` list with all processed results

---

### 5. LLM Reasoning (`node_llm_reasoning`)

**Purpose:** Generate final answer using enriched context.

**System Prompt:**

```
You are an expert marine technical documentation assistant.
Your primary role is to answer questions about maritime technical documentation.

CONVERSATION RULES:
- For greetings and casual conversation, respond naturally and friendly
- Remember context from the conversation (user's name, previous questions)
- For technical questions, use the provided documentation context

CITATION RULES (for technical answers):
- Cite facts using: [Document Name | Section/Table/Diagram: Title | Page X]
- If documentation context is empty or insufficient, say what information is missing
```

**Message Construction:**

1. Add system prompt
2. Add last 10 messages from chat history
3. Add current question with documentation context (if available)

**Response Structure:**

```python
{
    "answer_text": str,           # LLM-generated answer
    "citations": [                # Text sources
        {
            "type": "text",
            "doc_id": str,
            "section_id": str,
            "page": int,
            "title": str,
            "doc_title": str
        }
    ],
    "figures": [                  # Diagram references
        {
            "schema_id": str,
            "title": str,
            "caption": str,
            "url": str,
            "page": int,
            "doc_title": str
        }
    ],
    "tables": [                   # Table references
        {
            "table_id": str,
            "title": str,
            "caption": str,
            "url": str,
            "page": int,
            "doc_title": str,
            "rows": int,
            "cols": int
        }
    ]
}
```

---

## Tools Detail

### `qdrant_search_text`

**Purpose:** Semantic search over text chunks.

**Parameters:**
- `query: str` - Search query
- `limit: int` - Max results (default 10)

**Process:**
1. Generate query embedding (OpenAI text-embedding-3-small)
2. Search `text_chunks` collection
3. Apply owner/doc_id filters
4. Return chunks with score > 0.3

**Returns:**
```python
{
    "type": "text_chunk",
    "score": float,
    "section_id": str,
    "doc_id": str,
    "doc_title": str,
    "section_title": str,
    "page_start": int,
    "page_end": int,
    "text": str,           # Full chunk text
    "text_preview": str,   # First 500 chars
    "chunk_index": int,
    "chunk_char_start": int,
    "chunk_char_end": int
}
```

---

### `qdrant_search_tables`

**Purpose:** Semantic search over table content.

**Parameters:**
- `query: str` - Search query
- `limit: int` - Max results (default 5)

**Process:**
1. Generate query embedding
2. Search `tables` collection
3. Return table chunks with metadata

**Returns:**
```python
{
    "type": "table_chunk",
    "score": float,
    "table_id": str,
    "doc_id": str,
    "doc_title": str,
    "page": int,
    "rows": int,
    "cols": int,
    "table_title": str,
    "table_caption": str,
    "text_preview": str,
    "csv_path": str
}
```

---

### `qdrant_search_schemas`

**Purpose:** Semantic search over diagram/schema descriptions.

**Parameters:**
- `query: str` - Search query
- `limit: int` - Max results (default 5)

**Process:**
1. Generate query embedding
2. Search `schemas` collection
3. Return schema metadata with file paths

**Returns:**
```python
{
    "type": "schema",
    "score": float,
    "schema_id": str,
    "doc_id": str,
    "doc_title": str,
    "page": int,
    "title": str,
    "caption": str,
    "file_path": str,
    "thumbnail_path": str,
    "section_id": str
}
```

---

### `neo4j_query`

**Purpose:** Execute read-only Cypher queries for structural information.

**Parameters:**
- `cypher: str` - Cypher query (read-only)
- `params: Dict` - Query parameters (optional)

**Safety Checks:**
- Blocks: CREATE, MERGE, DELETE, REMOVE, SET, DROP, DETACH
- Only allows: MATCH, OPTIONAL MATCH, RETURN, WHERE, ORDER BY, LIMIT

**Example Queries:**

```cypher
-- Get section with its tables
MATCH (s:Section {id: $section_id})-[:CONTAINS_TABLE]->(t:Table)
RETURN s, t LIMIT 5

-- Find tables on specific page
MATCH (t:Table {page_number: $page, doc_id: $doc_id})
RETURN t LIMIT 5

-- Get document structure
MATCH (d:Document {id: $doc_id})-[:HAS_CHAPTER]->(c:Chapter)
RETURN c.title, c.number ORDER BY c.number LIMIT 10
```

---

### `neo4j_entity_search`

**Purpose:** Find content related to specific maritime entities via graph relationships.

**Parameters:**
- `query: str` - Natural language query (entities auto-extracted)
- `include_tables: bool` - Include related tables (default true)
- `include_schemas: bool` - Include related schemas (default true)

**How It Works:**

1. **Entity Extraction:**
   - Use `EntityExtractor` to find entity mentions
   - Dictionary-based extraction (systems, components)
   - Fallback patterns for equipment codes (P-101, V-205)

2. **Graph Traversal:**
   ```cypher
   -- Find sections describing entities
   MATCH (e:Entity {code: $entity_id})<-[:DESCRIBES]-(s:Section)
   RETURN s.id, s.title, s.content
   
   -- Find tables mentioning entities
   MATCH (e:Entity {code: $entity_id})<-[:MENTIONS]-(t:Table)
   RETURN t.id, t.title, t.page_number
   
   -- Find schemas depicting entities
   MATCH (e:Entity {code: $entity_id})<-[:DEPICTS]-(sc:Schema)
   RETURN sc.id, sc.title, sc.file_path
   ```

3. **Content Retrieval:**
   - Section content returned directly from Neo4j (no Qdrant round-trip)
   - Table text_preview + chunk_previews included
   - Schema text_context included

**When to Use:**

✅ Use for:
- "Tell me about the fuel oil pump"
- "What is P-101?"
- "Cooling water system components"

❌ Don't use for:
- "How to start the engine" (procedure, use text search)
- "What's on page 15" (structural, use neo4j_query)
- General questions without specific entities

**Returns:**
```python
{
    "entities": ["fo_system", "fuel_oil_pump"],
    "entity_names": ["Fuel Oil System", "FO Pump"],
    "sections": [
        {
            "section_id": str,
            "section_title": str,
            "content": str,         # Full text from Neo4j!
            "page_start": int,
            "doc_id": str,
            "matched_entity": str
        }
    ],
    "tables": [...],
    "schemas": [...]
}
```

---

## State Schema

```python
class GraphState(TypedDict):
    # User context
    user_id: str
    question: str
    chat_history: List[Dict[str, str]]
    
    # Access control
    owner: Optional[str]
    doc_ids: Optional[List[str]]
    
    # Query analysis
    query_intent: str  # "text" | "table" | "schema" | "mixed"
    
    # Agent communication
    messages: List[Message]  # Agent messages with tool calls
    
    # Tool results (raw)
    qdrant_results: {
        "text": List[Dict],
        "tables": List[Dict],
        "schemas": List[Dict]
    }
    neo4j_results: List[Dict]
    
    # Processed context
    enriched_context: List[Dict]
    
    # Final answer
    answer: {
        "answer_text": str,
        "citations": List[Dict],
        "figures": List[Dict],
        "tables": List[Dict]
    }
```

---

## Graph Flow

```
                    ┌─────────────────────┐
                    │   analyze_question  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    router_agent     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
            (has tool calls)       (no tool calls)
                    │                     │
                    ▼                     │
         ┌─────────────────────┐          │
         │   execute_tools     │          │
         └──────────┬──────────┘          │
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   build_context     │
                    │  (empty if no tools)│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   llm_reasoning     │
                    │ (can answer without │
                    │  context for general│
                    │  questions)         │
                    └──────────┬──────────┘
                               │
                               ▼
                             [END]
```

### Conditional Edge: `should_continue_to_tools`

The router agent's decision to call tools determines the flow:

```python
def should_continue_to_tools(state: GraphState) -> str:
    """Decide if agent wants to call tools or answer directly"""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = get_tool_calls(last_message)
    if tool_calls:
        return "execute_tools"  # Agent requested tools
    else:
        return "build_context"  # Skip tools, answer directly
```

**When agent skips tools:**
- `build_context` receives empty results
- `llm_reasoning` generates answer without documentation context
- Appropriate for greetings, general questions, conversational responses

---

## Logging

The workflow produces detailed logs for debugging:

```
############################################################
📝 NEW QUESTION: How does the fuel oil system work?
############################################################

🎯 Query intent classified as: TEXT

============================================================
🤖 AGENT TOOL CALLS (2 tools):
  [1] 🔧 qdrant_search_text
      └─ query: fuel oil system operation
      └─ limit: 10
  [2] 🔧 neo4j_entity_search
      └─ query: fuel oil system
============================================================

⚙️  EXECUTING: qdrant_search_text
   Args: {'query': 'fuel oil system operation', 'limit': 10}
   ✅ qdrant_search_text: found 5 text chunks

⚙️  EXECUTING: neo4j_entity_search
   Args: {'query': 'fuel oil system'}
   ✅ neo4j_entity_search results:
      Extracted entities: ['fo_system', 'fuel_oil']
      Entity names: ['Fuel Oil System', 'FO']
      Found: 2 sections, 1 tables, 0 schemas

📊 TOOL EXECUTION SUMMARY:
   Text chunks: 7
   Tables: 1
   Schemas: 0
   Neo4j records: 0

============================================================
✅ ANSWER GENERATED
   Answer length: 1234 chars
   Citations: 3
   Figures: 0
   Tables: 1
============================================================
```

### Example: Direct Answer (No Tools)

```
############################################################
📝 NEW QUESTION: Hello, how are you?
############################################################

🎯 Query intent classified as: TEXT

============================================================
🤖 AGENT TOOL CALLS (0 tools):
   Agent decided to answer directly without retrieval
============================================================

📊 TOOL EXECUTION SUMMARY:
   Text chunks: 0
   Tables: 0
   Schemas: 0
   (No context - general conversation)

============================================================
✅ ANSWER GENERATED
   Answer length: 89 chars
   Citations: 0
   Figures: 0
   Tables: 0
============================================================
```

---

## Configuration

```python
# Tool defaults
TEXT_SEARCH_LIMIT = 10
TABLE_SEARCH_LIMIT = 5
SCHEMA_SEARCH_LIMIT = 5
SCORE_THRESHOLD = 0.3

# Context building
NEIGHBOR_CHUNK_RANGE = 1  # ±1 chunks
MAX_CHAT_HISTORY = 10     # Last 10 messages

# Neo4j safety
MAX_CYPHER_LIMIT = 5      # Enforce LIMIT clause
```

---

## Error Handling

### Tool Failures

- Each tool is wrapped in try/except
- Failed tools return empty results (not exceptions)
- Error logged and `ToolMessage` created with error content

### Fallback Mechanisms

1. **Poor Qdrant Results:**
   - If < 2 results with score > 0.3
   - Trigger Neo4j fulltext search
   - Retrieve sections directly from graph

2. **No Context Found:**
   - LLM still invoked (can answer general questions)
   - For technical questions, states "I don't have enough information"

3. **Entity Extraction Fails:**
   - Returns empty entity list
   - Tool message indicates "No maritime entities detected"

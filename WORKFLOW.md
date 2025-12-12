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
│   ┌──────────────┐      ┌────────────┐                                  │
│   │   Analyze    │────> │   Router   │                                  │
│   │   Question   │      │   Agent    │                                  │
│   └──────────────┘      └────┬───────┘                                  │
│          │                   │                                          │
│          │           ┌───────┴────────┐                                 │
│          ▼           │                │                                 │
│   ┌──────────────┐   │ (has tools)    │  (no tools - direct answer)     │
│   │   Intent:    │   ▼                │                                 │
│   │ text/table/  │ ┌─────────────┐    │                                 │
│   │ schema/mixed │ │Execute Tools│    │                                 │
│   └──────────────┘ └──────┬──────┘    │                                 │
│                           │           │                                 │
│                           └─────┬─────┘                                 │
│                                 ▼                                       │
│                   ┌──────────────────────────────────────────┐          │
│                   │              Build Context               │          │
│                   │  - Merge Qdrant + Neo4j results          │          │
│                   │  - Expand with neighbor chunks           │          │
│                   │  - Deduplicate and rank                  │          │
│                   │  - (empty if no tools called)            │          │
│                   └──────────────────────────────────────────┘          │
│                                          │                              │
│                                          ▼                              │
│                   ┌──────────────────────────────────────────┐          │
│                   │           LLM Reasoning                  │          │
│                   │  - Generate answer with citations        │          │
│                   │  - Handle general conversation           │          │
│                   │  - Answer without context if appropriate │          │
│                   └──────────────────────────────────────────┘          │
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
- `text` - seeking textual information, procedures, explanations, descriptions
- `table` - seeking tabular data, specifications, parameters, **troubleshooting info**
- `schema` - seeking diagrams, schematics, figures, drawings, visual representations
- `mixed` - needs both semantic search AND graph traversal (e.g., structural queries)

**Critical Classification Rules:**

1. **TROUBLESHOOTING/FAULT KEYWORDS → "table"**
   - Keywords: "cause", "reason", "troubleshooting", "fault", "failure", "breakdown", "malfunction", "problem", "issue", "error", "no suction", "not working", "won't start"
   - Why: Troubleshooting tables contain causes/solutions in structured format

2. **VISUAL CONTENT KEYWORDS → "schema"**
   - Keywords: "drawing", "drawings", "diagram", "scheme", "figure", "layout", "show me", "where is", "location"
   - Why: User wants visual representation

3. **SPECIFICATIONS/PARAMETERS → "table"**
   - Keywords: "specifications", "specs", "parameters", "values", "temperature", "pressure", "capacity", "dimensions", "range", "calibration", "rating", "tolerance", "limits", "settings"
   - Why: Technical data is typically in table format

4. **DEFAULT → "text"**
   - Procedural questions, explanations, descriptions without specs/visuals

**Process:**
1. Send question to LLM with classification prompt containing rules above
2. LLM returns one of: `text`, `table`, `schema`, `mixed`
3. Validate response (fallback to "text" if invalid)
4. Store intent in state for router agent

**Example Classifications:**
| Question | Intent | Reason |
|----------|--------|--------|
| "How does the fuel system work?" | `text` | Explanation |
| "What are the engine specifications?" | `table` | Specifications |
| "Show me the cooling water diagram" | `schema` | Visual content |
| "The pump has no suction. What can be a cause?" | `table` | Troubleshooting |
| "Why does the incinerator fail to start?" | `table` | Fault analysis |
| "Temperature range for cooling water?" | `table` | Parameters |
| "Where are the fuel connections located?" | `schema` | Location/layout |
| "List all tables in chapter 3" | `mixed` | Structural query |

---

### 2. Router Agent (`node_router_agent`)

**Purpose:** LLM agent decides which tools to call based on question and intent.

**Pre-Agent Processing:**

1. **Entity Detection** (before agent invocation):
   - Loads known entities from Neo4j (lazy initialization)
   - Scans question for entity mentions:
     - Exact match against known entities (min 3 chars, non-generic)
     - Equipment code patterns: `[A-Z]{1,4}[-]?[0-9]{1,5}` (PU3, SV4, HGM-30, PT-6018)
   - Returns top 5 entities by length (longer = more specific)

2. **Entity Hint Generation:**
   - **Equipment codes detected** → DIRECTIVE hint:
     ```
     ⚠️ EQUIPMENT CODES DETECTED: PU3, SV4
     
     IMPORTANT: These are specific equipment identifiers. You SHOULD use neo4j_entity_search to find:
     - Cross-references across document sections
     - Related tables, diagrams, technical data
     - Contextual information about this component
     
     RECOMMENDED: neo4j_entity_search + qdrant_search_text/tables/schemas
     ```
   
   - **Named components detected** → INFORMATIVE hint:
     ```
     📍 DETECTED ENTITIES: Fuel Oil Pump, Isolation Valve
     
     Consider:
     - WHERE/WHICH DIAGRAM/LOCATION → neo4j_entity_search
     - HOW/WHY/EXPLAIN procedures → qdrant_search_text
     - SPECS/PARAMETERS → neo4j_entity_search + qdrant_search_tables
     ```
   
   - **No entities** → Semantic search guidance:
     ```
     📍 No specific equipment entities detected.
     → Use semantic search (qdrant_search_*) for best results.
     ```

**Available Tools:**

| Tool | Description | When to Use | F1 Score |
|------|-------------|-------------|----------|
| `qdrant_search_text` | Semantic search in text chunks | Explanations, procedures | **0.90** ✅ |
| `qdrant_search_tables` | Semantic search in tables | Specifications, troubleshooting | 0.75 |
| `qdrant_search_schemas` | Semantic search in diagrams | Visual content | 0.68 |
| `neo4j_query` | Execute Cypher query | Section NUMBER queries (3.2, 4.4) | N/A |
| `neo4j_entity_search` | Entity graph traversal | Equipment codes/named components | 0.85* |

*Entity search F1 depends on usage:
- Equipment codes (P-101, PU3): **0.85**
- Named components (Fuel Oil Pump): **0.75**
- Generic terms (pump, valve): **0.11** ❌ (context pollution)

**System Prompt Structure:**

```
{GRAPH_SCHEMA_PROMPT}

You are a routing agent for maritime technical documentation Q&A.

DETECTED INTENT: {intent}
{entity_hint}  ← Dynamic hint based on detected entities

INTENT-BASED PARAMETER SELECTION:
When using neo4j_entity_search, set parameters based on intent:
- intent="text" → include_tables=False, include_schemas=False
- intent="table" → include_tables=True, include_schemas=False
- intent="schema" → include_tables=False, include_schemas=True
- intent="mixed" → include_tables=True, include_schemas=True

🎯 CRITICAL TOOL SELECTION RULES:

1. qdrant_search_text - For text questions
   ✅ Best F1 score (0.90) - most reliable
   Examples: "How does X work", "What is Y", "explain Z"

2. qdrant_search_tables - For specs/parameters/troubleshooting
   Examples: "specs of X", "temperature range", "what causes failure"
   IMPORTANT: When asked to EXPLAIN table → ALSO call qdrant_search_text!

3. qdrant_search_schemas - For diagrams/drawings
   Examples: "give me drawings of X", "show diagram of Y"
   IMPORTANT: When asked to EXPLAIN diagram → ALSO call qdrant_search_text!

4. neo4j_entity_search - For SPECIFIC equipment
   ⚠️ USE CAREFULLY - Can pollute context if used for generic terms!
   
   ✅ USE WHEN:
   A) EQUIPMENT CODES: HGM-30, PU3, SV4, P-101
   B) NAMED COMPONENTS: "Isolation Valve", "Fuel Oil Pump"
   C) LOCATION/REFERENCE: "WHERE is X", "find all references to X"
   
   ❌ DO NOT USE FOR GENERIC TERMS:
   - "incinerator", "pump", "valve", "burner" (single word)
   → Use qdrant_search_text instead
   
   WHY: "pump" = 100+ mentions → context pollution
        "Fuel Oil Pump" = 5-10 mentions → useful cross-references

5. neo4j_query - ONLY for section NUMBER queries
   Examples: "tables from section 4.4", "content of chapter 3.2"
   NOT for keyword search!

MULTI-TOOL STRATEGY:
- "explain diagram/table" → schemas/tables + text
- Equipment code + question → neo4j_entity_search + qdrant_search_text
- Equipment + specs → neo4j_entity_search + qdrant_search_tables

MANDATORY TOOL CALLS:
- intent="table" → YOU MUST call qdrant_search_tables
- intent="schema" → YOU MUST call qdrant_search_schemas
```

**Tool Selection Logic:**

```
Intent = "text"
  → qdrant_search_text (required)
  → neo4j_entity_search (if equipment code mentioned, include_schemas=False)

Intent = "table"  
  → qdrant_search_tables (required)
  → neo4j_entity_search (if equipment code, include_tables=True)

Intent = "schema"
  → qdrant_search_schemas (required)
  → neo4j_entity_search (if equipment code, include_schemas=True)

Intent = "mixed"
  → Multiple tools based on question structure
  → neo4j_query for structural/section NUMBER queries (4.4, 3.2)
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

**Purpose:** Merge and enrich results from all sources with anchor-based filtering.

**Anchor Section Selection:**

Before building context, top sections are selected as "anchors" to focus results:

```python
# Combined score = similarity * 0.7 + importance * 0.2
# importance_score is from Neo4j Section node
anchors = select_anchor_sections(text_hits, max_sections=5)

# Filter tables/schemas to PRIMARY document (most anchor sections)
primary_doc_id = most_common_doc_in_anchors
```

**Process:**

#### Step 4.1: Text Chunk Processing

For each text chunk from Qdrant:
1. **Anchor Filtering:** Skip if not in anchor sections
2. **Neighbor Expansion:**
   - Fetch ±1 neighbor chunks from same section
   - Query Qdrant by section_id + chunk_index range
   - Sort by character position
   
3. **Overlap Removal:**
   ```python
   if chunk["char_start"] < last_end:
       overlap_size = last_end - chunk["char_start"]
       combined_text += chunk["text"][overlap_size:]
   else:
       combined_text += chunk["text"]
   ```

4. **Enrichment:**
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

#### Step 4.5: Intent-Based Context Stripping (NEW)

**Purpose:** Enforce strict context filtering based on query intent.

**Logic:**
```python
if query_intent == "text":
    # Text queries: ONLY text chunks, no tables/schemas
    # NOTE: If entity found via table_mentions, intent auto-corrected to "mixed" upstream
    tables = []
    schemas = []
    
elif query_intent == "schema":
    # Schema queries: strip tables (keep diagrams + supporting text)
    tables = []
    
# Note: "table" and "mixed" intents keep their respective content types
```

**Intent Auto-Correction (in execute_tools):**

When entity found via Table-[:MENTIONS] relationship:
```python
# Entity search found entity in TABLES, not text
table_mention_sections = [sec for sec in entity_sections 
                         if sec.get("found_via") == "table_mentions"]

if table_mention_sections and entity_tables and current_intent == "text":
    logger.info("Intent correction: entity found in tables → 'text' to 'mixed'")
    state["query_intent"] = "mixed"
```

When entity found in SCHEMAS:
```python
if len(entity_schemas) > 0 and current_intent in ["text", "table"]:
    logger.info("Intent correction: entity found in schemas → changing to 'mixed'")
    state["query_intent"] = "mixed"
```

**Why This Matters:**
- Prevents table/schema context pollution in text-only queries
- Ensures focused, relevant context for LLM
- Auto-corrects intent when entity search reveals different content type

**Output:** `enriched_context` list with all processed and filtered results

---

### 5. LLM Reasoning (`node_llm_reasoning`)

**Purpose:** Generate final answer using enriched context.

**System Prompt (Updated):**

```
You are a marine technical documentation answer generator.
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
- query_intent="text" → Prefer text citations, but MAY include tables/diagrams if answer is there
- query_intent="table" → You MUST include at least one [TABLE] reference
- query_intent="schema" → You MUST include at least one [DIAGRAM] reference
- query_intent="mixed" → You may include both tables and diagrams

CITATION RULES:
- Cite facts using: [Document | Section/Table/Diagram Title | Page X]
- Use a maximum of TWO textual citations.
- If table/diagram-driven question → respective reference is mandatory.
- Never cite Table of Contents, Contents, or unrelated sections.
- If answer fully supported by one section → include only one citation.

STRICT BANS:
- No invented recommendations or operational guidance.
- No troubleshooting instructions unless explicitly stated in documentation.
- No conditional phrases: "if needed", "let me know", "you can", etc.
- No conversational filler, no rhetorical questions.

RESPONSE CONTRACT:
✔ direct
✔ factual  
✔ finished — no invitations, no continuation prompts

If insufficient context exists, state absence and stop.
You can answer general conversational greetings naturally without document context.
```

**POST-GENERATION VALIDATION (NEW):**

After LLM generates answer, validator checks intent constraints:

```python
# Check if mandatory references included
if query_intent == "table" and len(referenced_tables) == 0 and len(table_map) > 0:
    # REGENERATION REQUIRED
    correction_prompt = "Intent=table but no table referenced. MUST include [TABLE1]."
    
elif query_intent == "schema" and len(referenced_diagrams) == 0 and len(schema_map) > 0:
    # REGENERATION REQUIRED
    correction_prompt = "Intent=schema but no diagram referenced. MUST include [DIAGRAM1]."
```

If validation fails:
1. Add correction prompt to conversation
2. Regenerate answer with STRICT rules
3. Re-parse references
4. If still fails → emergency fallback

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

**⚠️ CRITICAL USAGE WARNING:**
- Use ONLY for equipment codes (P-101, PU3) or named components ("Fuel Oil Pump")
- DO NOT use for generic terms ("pump", "valve") → causes context pollution
- Performance: Equipment codes F1=0.85, Generic terms F1=0.11

**Parameters:**
- `query: str` - Natural language query (entities auto-extracted)
- `include_tables: bool` - Include related tables (default true)
- `include_schemas: bool` - Include related schemas (default true)

**How It Works:**

1. **Entity Extraction** (using EntityExtractor):
   - Equipment code patterns: `[A-Z]{1,4}[-]?[0-9]{1,5}` (PU3, SV4, HGM-30, P-101)
   - Named components with valid qualifiers: "main fuel pump", "isolation valve"
   - Generic terms filtered out: "pump", "valve", "incinerator" (too broad)
   - Returns: `{"systems": [...], "components": [...], "entity_ids": [...]}`

2. **Entity Search Flow:**
   ```
   Entity extracted?
     → Search Neo4j Entity nodes by code
       → Found Entity? → Use graph relationships:
           - Section -[:DESCRIBES]-> Entity (full text content)
           - Table -[:MENTIONS]-> Entity (metadata + text_preview)
           - Schema -[:DEPICTS]-> Entity (metadata + llm_summary)
       
       → NOT found in Entity graph? → Fulltext fallback:
           - Search sectionSearch index with entity terms
           - Build variations: "PU3" OR "PU-3" OR "pump PU3"
           - Find related tables/schemas via CONTAINS relationships
           - Also search table/schema captions for entity mentions
       
       → Still no results? → Return empty with suggestion:
           "Entity not found in graph. Try semantic search with qdrant_search_text."
   ```

3. **Graph Traversal (when Entity found):**
   ```cypher
   -- Find sections describing entities
   MATCH (e:Entity {code: $entity_id})<-[:DESCRIBES]-(s:Section)
   WHERE $doc_ids IS NULL OR s.doc_id IN $doc_ids
   RETURN s.id, s.title, s.content
   
   -- Find tables mentioning entities  
   MATCH (e:Entity {code: $entity_id})<-[:MENTIONS]-(t:Table)
   WHERE $doc_ids IS NULL OR t.doc_id IN $doc_ids
   RETURN t.id, t.title, t.page_number
   
   -- Find schemas depicting entities
   MATCH (e:Entity {code: $entity_id})<-[:DEPICTS]-(sc:Schema)
   WHERE $doc_ids IS NULL OR sc.doc_id IN $doc_ids
   RETURN sc.id, sc.title, sc.file_path
   ```

4. **Fulltext Fallback (when code not in Entity graph):**
   - Search sections via `sectionSearch` fulltext index
   - Also search tables/schemas by:
     - CONTAINS_TABLE/CONTAINS_SCHEMA relationship to found sections
     - OR equipment code in title/caption/llm_summary

5. **Content Retrieval:**
   - Section content returned directly from Neo4j (no Qdrant round-trip)
   - Table text_preview + chunk_previews included
   - Schema text_context + llm_summary included

**When to Use:**

✅ Use for:
- "Tell me about CP-1 control panel"
- "What is P-101?"
- "Function of pump 7M2"

❌ Don't use for:
- "How to start the engine" (procedure, use text search)
- "Tables in section 4.4" (structural, use neo4j_query)
- General questions without equipment codes

**Returns:**
```python
{
    "entities": ["CP-1"],
    "entity_names": ["Equipment CP-1"],
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
    "tables": [...],      # Only if include_tables=True
    "schemas": [...],     # Only if include_schemas=True
    "message": str,       # Status message
    "suggest_semantic_search": bool  # True if nothing found
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
    
    # Anchor sections (top relevant sections for filtering)
    anchor_sections: List[Dict]  # {doc_id, section_id, score, similarity, importance}
    
    # Tool results (raw)
    search_results: {          # Renamed from qdrant_results
        "text": List[Dict],
        "tables": List[Dict],
        "schemas": List[Dict]
    }
    neo4j_results: List[Dict]
    
    # Entity search (separate to avoid context pollution)
    entity_results: Optional[Dict]  # {entities, sections, tables, schemas}
    
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

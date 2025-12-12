# Document Ingestion Pipeline

## Overview

The Maritime QA Assistant uses a sophisticated multi-stage ingestion pipeline to process large technical maritime PDF documents. The pipeline combines layout analysis, content classification, **entity extraction**, and vector indexing to create a comprehensive knowledge graph suitable for intelligent question-answering.

## Architecture Components

### Core Services

1. **DocumentProcessor** - Main orchestrator for the ingestion workflow
2. **LayoutAnalyzer** - YOLO-based page segmentation (DocLayNet model)
3. **RegionClassifier** - Content-based region reclassification
4. **SchemaExtractor** - Technical diagram/schema extraction
5. **TableExtractor** - Structured table data extraction
6. **EntityExtractor** - Maritime domain entity recognition and normalization
7. **Neo4jClient** - Graph database operations (hierarchical document structure)
8. **VectorService** - Qdrant vector database operations (semantic search)
9. **StorageService** - File storage for extracted images and tables

### Data Stores

- **Neo4j**: Stores document hierarchy (Document → Chapter → Section), relationships between sections, tables, schemas, and entities
- **Qdrant**: Stores vector embeddings for text chunks, tables, and schemas with full metadata payloads
- **File System**: Stores extracted schema images, table images, CSV exports, and thumbnails

## Pipeline Stages

### Stage 1: Document Initialization

**Workflow:**
1. Calculate file hash for deduplication
2. Open PDF with PyMuPDF (fitz)
3. Extract document metadata (title, type, version, language, tags, owner)
4. Create Document node in Neo4j
5. **Extract and parse table of contents (TOC):**
   - Extract TOC structure using PyMuPDF
   - Parse hierarchical levels (chapters, sections, subsections)
   - Map TOC entries to page numbers
   - Create chapter boundaries from level-1 entries
   - **Improvement:** Better handling of multi-level TOC structures
6. Initialize tracking structures (page-to-section mapping, current chapter ID)

**Output:**
- Document node in Neo4j with metadata
- TOC structure for chapter boundaries
- Document title stored in `self._current_doc_title` for chunk metadata

---

### Stage 2: Page-by-Page Processing

Documents are processed in chunks (50 pages at a time) to handle large files efficiently. Each page goes through the following sub-stages:

#### Step 2.1: YOLO Layout Analysis

**Purpose:** Segment page into regions using a pre-trained YOLO DocLayNet model.

**Process:**
1. Convert PDF page to image
2. Run YOLOv10 inference to detect regions
3. Classify regions into: `TABLE`, `SCHEMA`, or `TEXT`
4. Extract bounding boxes and confidence scores

**Output:** List of `Region` objects with initial YOLO predictions

#### Step 2.2: LLM-Based Reclassification

**Purpose:** Use LLM to validate and refine YOLO predictions for ambiguous regions.

**Classification Process:**

For each `TABLE` or `SCHEMA` region with low confidence:

1. **Caption Detection (Enhanced)**
   - Search within 250 pixels above/below region for caption text
   - **Improved extraction:** Better handling of multi-line captions
   - Match patterns: "Figure X", "Table X", "Diagram X", "Schema X", "Drawing X"
   - Extract caption number and full description
   - **Fallback:** Use OCR-based text extraction if caption parsing fails

2. **LLM Classification (GPT-4o-mini)**
   - Triggered when YOLO confidence < 0.6
   - Analyze region image + caption + surrounding text context
   - Classification prompt asks LLM to determine:
     - Is this a TABLE (structured data in rows/columns)?
     - Is this a SCHEMA (diagram, flowchart, technical drawing)?
   - Returns classification with reasoning
   - **Benefit:** More accurate than rule-based heuristics, handles edge cases

**Decision Logic:**
```
YOLO confidence ≥ 0.8?
  → Keep YOLO prediction
  
YOLO confidence < 0.8?
  → LLM CLASSIFICATION (GPT-4o-mini)
     - Analyze visual + textual context
     - Classify as TABLE or SCHEMA
     - Override YOLO if needed
```

**Output:** Reclassified regions with updated `region_type` and confidence scores

#### Step 2.3: Smart Region Processing with Multi-Level Fallbacks

**Purpose:** Extract structured data from each region with intelligent multi-level fallback chain.

**For TABLE regions:**

1. **Primary Extraction (pdfplumber)**
   - Attempt structured table extraction using pdfplumber
   - Parse cell boundaries and text content
   - Generate CSV representation
   
2. **Quality Validation**
   - Check if extraction produced valid table structure
   - Verify column/row counts and cell data quality
   
3. **Fallback Level 1: LLM Table Extraction**
   - If pdfplumber fails, use GPT-4o-mini vision
   - Render region as image (expand bbox if no caption detected)
   - LLM extracts table as CSV format
   - Handles complex layouts pdfplumber misses
   
4. **Fallback Level 2: Content Type Verification**
   - If LLM extraction fails, verify actual content type
   - Use `region_classifier._llm_verify_type()` to re-classify
   - May determine region is actually TEXT or SCHEMA
   
5. **Fallback Level 3: Alternative Extraction**
   - If verified as TEXT: Extract text using LLM OCR
   - If verified as SCHEMA: Extract as image with context
   - If still TABLE: Mark as failed, skip region

**For SCHEMA regions:**

1. **Primary Schema Extraction**
   - Capture high-resolution image (original + thumbnail)
   - Extract caption from surrounding text (±250px)
   - Build context from nearby paragraphs
   - **NEW:** LLM generates rich description and detects schema type

2. **Embedded Table Detection**
   - Check if schema contains embedded table (legend, specs, parameter table)
   - Common in P&ID diagrams, circuit diagrams with component lists
   - Attempt pdfplumber extraction within schema bbox
   
3. **Dual Processing (if embedded table found)**
   - Extract schema as image with full context
   - Also extract embedded table as structured data
   - Create both schema chunk AND table chunk
   - Link both to same section
   - **Benefit:** Captures both visual diagram and structured data

**Fallback Chain Summary:**
```
TABLE region:
  pdfplumber → LLM CSV extraction → Content verification → Alternative extraction

SCHEMA region:
  Image + LLM description → Check for embedded table → Dual extraction if found
```

**Output:** 
- `page_schemas` dict: page_num → list of schema chunks
- `page_tables` dict: page_num → list of table chunks
- Each chunk includes: ID, content, metadata (bbox, caption, file paths)

---

### Stage 3: Text Extraction and Section Parsing

**Purpose:** Extract text content and parse hierarchical structure.

#### Step 3.1: Chapter Detection

**Process:**
1. Check TOC for chapter markers at current page
2. If chapter found (level 1 TOC entry):
   - Finalize any accumulated section
   - Create new Chapter node in Neo4j
   - Update `current_chapter_id`
   - Reset section accumulator

**Chapter Data:**
- ID (stable UUID from doc_id + title + page)
- Number (extracted from title)
- Title
- Start page
- Level

#### Step 3.2: Section Parsing

**Process:**
1. Extract full page text
2. Split into lines
3. For each line:
   - **Check if section header** (regex patterns):
     - Numbered format: `1.2.3 Title`
     - CHAPTER/PART/SECTION/APPENDIX format
   - If header found:
     - Finalize previous section
     - Start new section with number and title
   - If regular text:
     - Append to current section's content lines
     - Track page number for each line

**Section Accumulator:**
```python
{
    "number": "1.2.3",
    "title": "Safety Valve Operation",
    "content_lines": [
        {"text": "The safety valve...", "page": 14},
        {"text": "Operating pressure...", "page": 15},
    ],
    "start_page": 14,
    "end_page": 15,
    "chapter_id": "uuid-of-parent-chapter"
}
```

---

### Stage 4: Section Finalization and Merging

**Purpose:** Create sections in Neo4j and handle small sections intelligently.

#### Small Section Detection

**Threshold:** 200 characters

**Logic:**
1. If section < 200 chars:
   - Hold in `pending_small_section`
   - Wait for next section
2. If next section arrives:
   - Merge pending into current section
   - Combine content with separator
   - Track merged section numbers
3. If no next section (end of document):
   - Create standalone small section

#### Section Creation

**Process:**
1. Combine content lines into single text string
2. Generate stable section ID (UUID from doc_id + chapter_id + number + pages)
3. Create Section node in Neo4j with metadata:
   - ID, number, title
   - Full content text
   - Page range (start_page, end_page)
   - Section type (classification)
   - Importance score
   - Merged sections info (if applicable)
4. Link Section to Chapter
5. Store in page-to-section map for schema/table linking

---

### Stage 5: Text Chunking for Vector Search

**Purpose:** Split sections into overlapping chunks for semantic search.

**Parameters:**
- Chunk size: **400 tokens** (~1600 characters)
- Overlap: **100 tokens** (~400 characters)

**Process:**

1. **Tokenization** (tiktoken with cl100k_base encoding)
   - Convert section text to tokens
   - Calculate token boundaries

2. **Chunking**
   - Create sliding windows with overlap
   - Track character positions for each chunk
   - Calculate specific page for each chunk based on position:
     ```python
     chunk_mid_pos = (char_start + char_end) / 2
     page_progress = chunk_mid_pos / text_length
     chunk_page = start_page + int(page_progress * total_pages)
     ```

3. **Qdrant Indexing**
   For each chunk:
   - Generate embedding (OpenAI text-embedding-3-small, 1536 dimensions)
   - Create point with UUID (section_id + chunk_index)
   - Store **full chunk text** in payload (not just preview!)
   - Metadata payload:
     ```python
     {
         "type": "text_chunk",
         "section_id": str,
         "chunk_index": int,
         "chunk_char_start": int,
         "chunk_char_end": int,
         "doc_id": str,
         "doc_title": str,  # ← Document title for citations
         "section_number": str,
         "section_title": str,
         "page_start": int,  # Specific page for THIS chunk
         "page_end": int,
         "system_ids": List[str],
         "entity_ids": List[str],
         "owner": str,
         "text": str,  # Full chunk text
         "text_preview": str,  # First 500 chars
         "char_count": int
     }
     ```

**Output:** Text chunks indexed in Qdrant `text_chunks` collection

---

### Stage 6: Schema and Table Processing

**Purpose:** Create graph nodes and vector embeddings for extracted schemas and tables.

**Timing:** After sections are created (deferred from Stage 2.3)

#### Schema Processing

**For each schema chunk:**

1. **LLM-Enhanced Summary**
   - Generate comprehensive schema description using GPT-4o-mini
   - **Schema type detection:** Process flow, P&ID, electrical, hydraulic, etc.
   - Extract key components and connections
   - Identify system boundaries and interfaces
   - Build rich semantic description for better search

2. **Neo4j Node Creation**
   ```cypher
   CREATE (s:Schema {
       id: uuid,
       doc_id: str,
       page_number: int,
       title: str,
       caption: str,
       schema_type: str,      // flow/pid/electrical/hydraulic/etc.
       description: str,      // LLM-generated summary
       file_path: str,        // Original image
       thumbnail_path: str,   // Thumbnail
       bbox: [x1, y1, x2, y2],
       confidence: float,
       text_context: str      // Surrounding text
   })
   ```

2. **Section Linking (Enhanced)**
   - Find section on same page (using page-to-section map)
   - If not found, search previous pages
   - **Visual Section Creation**
     - If no text section found, create dedicated visual section:
       ```cypher
       CREATE (vs:Section {
           id: uuid,
           type: "visual",
           title: schema.caption,
           content: schema.description,
           page_start: schema.page,
           page_end: schema.page
       })
       ```
     - Ensures all schemas/tables are linked to a section
     - Improves context retrieval for orphan visuals
   - Create relationship: `(Section)-[:CONTAINS_SCHEMA]->(Schema)`

3. **Entity Extraction** (if enabled)
   - Extract maritime systems, components from caption + context
   - Create Entity nodes
   - Link: `(Schema)-[:REFERENCES]->(Entity)`

4. **Qdrant Indexing (Enhanced)**
   - Build embedding text: `caption + "\n" + schema_type + "\n" + llm_description + "\n" + text_context`
   - **Improvement:** LLM-generated description provides richer semantic content
   - Generate embedding (OpenAI text-embedding-3-small)
   - Store in `schemas` collection with enhanced metadata:
     ```python
     {
         "type": "schema_chunk",
         "schema_id": str,
         "doc_id": str,
         "page": int,
         "caption": str,
         "schema_type": str,      # NEW
         "description": str,      # NEW: LLM summary
         "file_path": str,
         "entity_ids": List[str],
         "system_ids": List[str],
         "text": str  # Full embedding text
     }
     ```

#### Table Processing

**For each table chunk:**

1. **Neo4j Node Creation**
   ```cypher
   CREATE (t:Table {
       id: uuid,
       doc_id: str,
       page_number: int,
       title: str,
       caption: str,
       rows: int,
       cols: int,
       file_path: str,        // Original image
       thumbnail_path: str,   // Thumbnail
       csv_path: str,         // CSV 
       bbox: [x1, y1, x2, y2]
   })
   ```

2. **Section Linking (Enhanced)**
   - Page-aware section detection
   - **NEW: Visual Section Creation** (same as schemas)
     - If no text section found on page, create visual section
     - Links orphan tables to dedicated section nodes
     - Ensures consistent graph structure
   - Create: `(Section)-[:CONTAINS_TABLE]->(Table)`

3. **Table Chunking**
   - Split table text into chunks (if large)
   - Create `TableChunk` nodes with chunk_index
   - Link: `(TableChunk)-[:PART_OF]->(Table)`

4. **Qdrant Indexing**
   - For each table chunk:
     - Build embedding text from cell values
     - Generate embedding
     - Store in `tables` collection with metadata
     - Include CSV path for data access

---

### Stage 7: Post-Processing

**Purpose:** Build additional relationships and enrich the graph.

#### Cross-Reference Detection

**Process:**
1. Scan section content for reference patterns:
   - "see Figure X"
   - "as shown in Table Y"
   - "refer to Section Z"
2. Parse referenced IDs
3. Create relationships:
   - `(Section)-[:REFERENCES_FIGURE]->(Schema)`
   - `(Section)-[:REFERENCES_TABLE]->(Table)`
   - `(Section)-[:REFERENCES_SECTION]->(Section)`

#### Schema-Table Linking

**Purpose:** Link schemas to tables that describe the same system.

**Process:**
1. Find schemas and tables on same/adjacent pages
2. Compare captions and titles for similarity
3. If match > threshold:
   - Create: `(Schema)-[:HAS_DATA_TABLE]->(Table)`

#### Similarity Calculation

**Purpose:** Find related sections for context expansion.

**Process:**
1. Compare section embeddings (cosine similarity)
2. For sections with similarity > 0.7:
   - Create: `(Section)-[:SIMILAR_TO {score: float}]->(Section)`

---

### Stage 8: Entity Extraction and Linking

**Purpose:** Extract maritime domain entities (systems, components, equipment codes) and create graph relationships for entity-based search.

**Note:** Entity extraction happens **during** Stages 5 and 6 (when creating text chunks, schemas, and tables), not as a separate stage.

#### EntityExtractor Service

The `EntityExtractor` uses a **dictionary-based approach** with strict qualifier validation:

**Key Features:**
- Dictionary-based system extraction (keywords, aliases, abbreviations)
- Component extraction with STRICT qualifier validation
- Equipment code detection (P-101, V-205, TK-102)
- Hierarchy inference (component → parent system)
- Singleton pattern for reuse

**Dictionary Structure (`entity_dictionary.json`):**
```json
{
  "systems": {
    "fuel_oil": {
      "code": "fo_system",
      "canonical": "Fuel Oil System",
      "aliases": ["fuel system", "FO system"],
      "keywords": ["fuel oil", "diesel oil", "heavy fuel"],
      "abbreviations": ["FO", "HFO", "MDO", "MGO"],
      "parent": null
    },
    "cooling_water": {
      "code": "cw_system",
      "canonical": "Cooling Water System",
      "aliases": ["cooling system", "CW system"],
      "keywords": ["cooling water", "jacket water"],
      "abbreviations": ["CW", "FW", "SW"]
    }
  },
  "component_types": {
    "pump": {
      "patterns": ["pump", "pumping unit"]
    },
    "valve": {
      "patterns": ["valve", "cock", "gate"]
    }
  }
}
```

#### Extraction Process

**1. System Extraction:**
- Match text against system keywords and aliases (longest match first)
- Resolve abbreviations (FO → fo_system)
- Return normalized system codes

**2. Component Extraction with STRICT Qualifier Validation:**

```python
# Valid qualifiers (whitelist)
valid_qualifiers = {
    'main', 'auxiliary', 'standby', 'emergency', 'primary', 'secondary',
    'fuel', 'oil', 'water', 'air', 'steam', 'cooling', 'heating',
    'inlet', 'outlet', 'suction', 'discharge', 'high', 'low', 'pressure',
    'safety', 'relief', 'control', 'isolation', 'check',
    # ... (50+ valid qualifiers)
}

# Stop words (blocklist) - NEVER part of component name
stop_words = {
    'the', 'a', 'an', 'of', 'to', 'in', 'on', 'is', 'are', 'this', 'that',
    'push', 'pull', 'check', 'ensure', 'verify', 'operate', 'damage',
    # ... (50+ stop words)
}
```

**Cleaning Process:**
```python
# Input: "ensure proper main fuel pump operation"
# 1. Split: ['ensure', 'proper', 'main', 'fuel', 'pump']
# 2. Component: 'pump' (last word)
# 3. Validate qualifiers: 
#    - 'ensure' → stop word → reject
#    - 'proper' → not valid qualifier → reject  
#    - 'main' → valid qualifier → keep
#    - 'fuel' → valid qualifier → keep
# 4. Output: "main fuel pump"
```

**3. Equipment Code Detection:**
```python
# Pattern: Letter(s) + hyphen + numbers
pattern = r'\b([A-Z]{1,4})[-–](\d{2,4}[A-Z]?)\b'

# Prefix → Type mapping
code_types = {
    'P': 'pump',
    'V': 'valve',
    'TK': 'tank',
    'HE': 'heat_exchanger',
    'FLT': 'filter',
    'SEP': 'separator',
    'ME': 'main_engine',
    'AE': 'auxiliary_engine',
}

# Example: "P-101" → {type: "pump", code: "eq_p_101"}
```

**4. Hierarchy Inference:**
```python
# Component "fuel oil pump" contains keyword "fuel oil"
# → Infer parent system: fo_system
# → Both codes added to entity_ids
```

**Code Example:**
```python
extractor = get_entity_extractor()  # Singleton

result = extractor.extract_from_text(
    "The main fuel oil pump P-101 supplies fuel to the engine..."
)
# Returns:
# {
#     "systems": ["fo_system"],
#     "components": [
#         {"name": "main fuel oil pump", "type": "pump", "code": "comp_pump_main_fuel_oil_pump"},
#         {"name": "P-101", "type": "pump", "code": "eq_p_101", "source": "equipment_code"}
#     ],
#     "entity_ids": ["fo_system", "comp_pump_main_fuel_oil_pump", "eq_p_101"]
# }
```

#### Query-Time Extraction

Same `EntityExtractor` used for question extraction with **entity search**:

```python
result = extractor.extract_from_question("What is the FO pump P-101?")
# Returns:
# {
#     "systems": ["fo_system"],  # FO abbreviation resolved
#     "system_names": ["Fuel Oil System"],
#     "components": [...],
#     "component_names": ["P-101"],
#     "entity_ids": ["fo_system", "eq_p_101"]
# }
```

**Entity Search Improvements:**
- **Neo4j graph traversal:** Find all sections/tables/schemas mentioning entity
- **Relationship types:**
  - `(Section)-[:DESCRIBES]->(Entity)` → text mentions
  - `(Table)-[:MENTIONS]->(Entity)` → table data
  - `(Schema)-[:DEPICTS]->(Entity)` → diagrams
- **Cross-reference expansion:** Follow entity relationships to find related content
- **Hybrid scoring:** Combine graph relationships + vector similarity
- **Benefit:** More complete entity coverage, better recall for technical queries

#### Graph Node Creation

**Entity Nodes:**
```cypher
CREATE (e:Entity {
    code: "fo_system",           // Normalized code
    type: "system",              // system | component | equipment
    canonical_name: "Fuel Oil System",
    source: "dictionary"         // dictionary | equipment_code | fallback
})
```

**Relationships:**
```cypher
// Section describes entity
(Section)-[:DESCRIBES]->(Entity)

// Table mentions entity (in content or caption)
(Table)-[:MENTIONS]->(Entity)

// Schema depicts entity (in diagram)
(Schema)-[:DEPICTS]->(Entity)

// Component is part of system
(Entity:Component)-[:PART_OF]->(Entity:System)
```

#### Entity Payload in Qdrant

Entities stored in Qdrant payloads for filtered search:
```python
{
    "type": "text_chunk",
    "text": "...",
    "entity_ids": ["fo_system", "comp_pump_main_fuel_oil_pump", "eq_p_101"],
    "system_ids": ["fo_system"],
    # ... other metadata
}
```

---

### Stage 9: Status Update and Completion

**Process:**
1. Update Document node status: `"completed"`
2. Set progress: 100%
3. Log ingestion summary:
   - Total chunks created
   - Schemas extracted
   - Tables extracted
   - Processing time

---

## Key Design Decisions


### 1. Neighbor Chunk Expansion

**Problem:** Single 400-token chunk lacks surrounding context.

**Solution:** Fetch ±1 neighbor chunks, remove overlaps, combine:
```python
chunks = get_neighbor_chunks(section_id, chunk_index, neighbor_range=1)
chunks.sort(key=lambda x: x["chunk_char_start"])

combined_parts = []
last_end = 0
for chunk in chunks:
    if chunk["char_start"] < last_end:
        # Skip overlap
        overlap_size = last_end - chunk["char_start"]
        combined_parts.append(chunk["text"][overlap_size:])
    else:
        combined_parts.append(chunk["text"])
    last_end = max(last_end, chunk["char_end"])

combined_text = "".join(combined_parts)
```

**Benefit:** Richer context for LLM without duplicate text.

---

### 2. Section Merging

**Problem:** Documents have many tiny sections (single paragraph).

**Solution:** Accumulate small sections (<200 chars) and merge with next section:
```python
if len(current_section) < 200:
    pending_small_section = current_section
    return

if pending_small_section:
    current_section["content"] = (
        pending_small_section["content"] + "\n---\n" + 
        current_section["content"]
    )
```

**Benefit:** Reduces fragmentation, better semantic coherence.

---

### 3. YOLO + LLM Reclassification

**Problem:** YOLO DocLayNet misclassifies complex technical diagrams and ambiguous regions.

**Solution:** Two-stage classification:
1. YOLO provides initial segmentation with confidence scores
2. LLM validation for low-confidence predictions (<0.6):
   - Analyzes region image + caption + context
   - GPT-4o-mini provides accurate classification
   - No brittle rule-based heuristics

**Benefit:** 
- Leverages YOLO's strong segmentation capabilities
- LLM handles ambiguous edge cases accurately
- More reliable than grid line/drawing coverage heuristics
- Better table vs schema distinction

---

### 4. Multi-Level Fallback Processing

**Problem:** Table extraction often fails on complex layouts; schemas may contain embedded tables.

**Solution:** Multi-level fallback chain:

**For TABLE regions:**
1. pdfplumber structured extraction
2. → LLM vision-based CSV extraction (if step 1 fails)
3. → Content type re-verification with LLM (if step 2 fails)
4. → Alternative extraction as TEXT or SCHEMA (if content type changed)

**For SCHEMA regions:**
1. Extract as image with LLM-generated description
2. → Check for embedded table within schema bbox
3. → Dual extraction: both schema image AND table data (if table found)

**Benefit:** 
- Robust handling of extraction failures
- No data loss - multiple recovery paths
- Hybrid extraction captures both visual and structured data
- LLM provides intelligent fallback when rule-based tools fail

---


## Performance Characteristics

### Scalability

- **Chunk Processing:** 50 pages per batch → handles 500+ page documents
- **Async Operations:** Neo4j and Qdrant operations run concurrently where possible
- **Memory Management:** Single PDF loaded, processed in chunks, closed after

### Throughput

Typical 100-page maritime manual:
- YOLO analysis: ~2-3 sec/page
- Text extraction: ~0.5 sec/page
- Embedding generation: ~0.2 sec/chunk
- Total time: ~10-15 minutes

### Storage

Per document (~100 pages):
- **Neo4j:** ~5-10 MB (nodes + relationships)
- **Qdrant:** ~20-30 MB (embeddings + payloads)
- **File System:** ~50-100 MB (images, CSVs)

---

## Error Handling

### Retry Logic

- **Embedding API failures:** Retry 3 times with exponential backoff
- **Neo4j connection errors:** Retry with fresh session
- **File I/O errors:** Log and continue (non-critical failures)

### Graceful Degradation

**Document Structure:**
- **No TOC found:** Create default chapter "Content"
- **No sections found:** Create fallback section per chapter
- **Empty sections:** Merge with adjacent sections if < 200 chars

**Region Processing:**
- **YOLO low confidence (<0.6):** Use LLM to verify and reclassify
- **pdfplumber table extraction fails:** Fall back to LLM CSV extraction
- **LLM table extraction fails:** Verify content type, extract as TEXT or SCHEMA
- **Schema with embedded table:** Extract both as separate chunks
- **Caption not found:** Expand search area, use OCR fallback
- **All extraction fails:** Log warning, skip region (non-critical)

**Entity Extraction:**
- **No entities found:** Continue without entity relationships
- **Ambiguous equipment codes:** Use fallback normalization
- **System inference fails:** Create standalone component entities

**Embedding & Storage:**
- **Embedding API failures:** Retry 3× with exponential backoff
- **Neo4j connection errors:** Retry with fresh session
- **Qdrant indexing fails:** Log error, continue (non-blocking)
- **File I/O errors:** Log warning, continue (images/CSVs are optional)


## Configuration

### Key Parameters

```python
# Chunking
chunk_size_tokens = 400
chunk_overlap_tokens = 100

# Section Merging
min_section_length = 200  # chars

# Region Classification
caption_search_distance = 250  # pixels
yolo_confidence_threshold = 0.8  # Use LLM if below this
llm_model = "gpt-4o-mini"  # For classification and schema analysis

# Search Thresholds
text_search_score_threshold = 0.2
table_search_score_threshold = 0.3
schema_search_score_threshold = 0.3

# Neighbor Expansion
neighbor_chunk_range = 1  # ±1 chunk
```

---

## Conclusion

This ingestion pipeline transforms unstructured maritime PDF documents into a richly structured, searchable knowledge graph. The combination of layout analysis, content classification, hierarchical parsing, and vector indexing enables accurate, context-aware question-answering over large technical manuals.

The system prioritizes **data quality** (accurate page numbers, complete text, proper citations) and **robustness** (fallback extraction, error handling, graceful degradation) while maintaining reasonable performance for production use.

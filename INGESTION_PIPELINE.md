# Document Ingestion Pipeline

## Overview

The Maritime QA Assistant uses a sophisticated multi-stage ingestion pipeline to process large technical maritime PDF documents. The pipeline combines layout analysis, content classification, entity extraction, and vector indexing to create a comprehensive knowledge graph suitable for intelligent question-answering.

## Architecture Components

### Core Services

1. **DocumentProcessor** - Main orchestrator for the ingestion workflow
2. **LayoutAnalyzer** - YOLO-based page segmentation (DocLayNet model)
3. **RegionClassifier** - Content-based region reclassification
4. **SchemaExtractor** - Technical diagram/schema extraction
5. **TableExtractor** - Structured table data extraction
6. **Neo4jClient** - Graph database operations (hierarchical document structure)
7. **VectorService** - Qdrant vector database operations (semantic search)
8. **StorageService** - File storage for extracted images and tables

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
5. Extract table of contents (TOC) for chapter detection
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

#### Step 2.2: Content-Based Reclassification

**Purpose:** Override YOLO predictions when content analysis provides stronger signals.

**Classification Logic:**

For each `TABLE` or `SCHEMA` region:

1. **Caption Detection**
   - Search within 250 pixels above/below region for caption text
   - Match patterns: "Figure X", "Table X", "Diagram X", "Schema X"
   - Caption type influences classification

2. **Grid Line Analysis**
   - Count horizontal/vertical lines in region
   - Threshold: ≥2 lines → likely table

3. **Drawing Coverage**
   - Calculate percentage of region covered by vector drawings/paths
   - Threshold: ≥30% → likely schema

4. **Numeric Density**
   - Calculate ratio of numeric characters to total text
   - Threshold: ≥15% → likely table

5. **Structural Patterns**
   - Detect cell-like text blocks in aligned grid
   - Detect header row patterns (bold, colored background)

**Decision Tree:**
```
Has "Table X" caption AND (has grid lines OR high numeric density OR cell alignment)?
  → TABLE
  
Has "Figure/Diagram/Schema X" caption AND high drawing coverage?
  → SCHEMA
  
Otherwise:
  → Keep YOLO prediction
```

**Output:** Reclassified regions with updated `region_type`

#### Step 2.3: Smart Region Processing

**Purpose:** Extract structured data from each region with intelligent fallback.

**For TABLE regions:**

1. **Primary Extraction** (pdfplumber)
   - Attempt structured table extraction
   - Parse cell boundaries and text content
   - Generate CSV representation
   
2. **Quality Validation**
   - Check if extraction produced valid table structure
   - Verify column/row counts
   
3. **Fallback to Schema** (if extraction fails)
   - Capture region as image
   - Extract surrounding text context
   - Store as schema instead

**For SCHEMA regions:**

1. **Hybrid Detection**
   - Check if schema contains embedded table
   - Look for table patterns within drawing region
   
2. **Extraction**
   - Capture high-resolution image (original + thumbnail)
   - Extract caption from surrounding text
   - Build context from nearby paragraphs
   
3. **Dual Processing** (if embedded table found)
   - Extract schema as image
   - Also extract embedded table as structured data
   - Return both chunks

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

1. **Neo4j Node Creation**
   ```cypher
   CREATE (s:Schema {
       id: uuid,
       doc_id: str,
       page_number: int,
       title: str,
       caption: str,
       file_path: str,        // Original image
       thumbnail_path: str,   // Thumbnail
       bbox: [x1, y1, x2, y2],
       confidence: float,
       text_context: str      // Surrounding text
   })
   ```

2. **Section Linking**
   - Find section on same page (using page-to-section map)
   - If not found, search previous pages
   - Fallback to last created section
   - Create relationship: `(Section)-[:CONTAINS_SCHEMA]->(Schema)`

3. **Entity Extraction** (if enabled)
   - Extract maritime systems, components from caption + context
   - Create Entity nodes
   - Link: `(Schema)-[:REFERENCES]->(Entity)`

4. **Qdrant Indexing**
   - Build embedding text: `caption + "\n" + text_context`
   - Generate embedding
   - Store in `schemas` collection with metadata

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

2. **Section Linking** (same logic as schemas)
   - Page-aware section detection
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

### Stage 8: Status Update and Completion

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

### 3. YOLO + Content Reclassification

**Problem:** YOLO DocLayNet misclassifies complex technical diagrams.

**Solution:** Two-stage classification:
1. YOLO provides initial segmentation
2. Content analysis overrides using:
   - Caption text patterns
   - Grid line detection
   - Drawing coverage percentage
   - Numeric density

**Benefit:** 
- Leverages YOLO's strong segmentation
- Fixes type errors with content heuristics
- Better table vs schema distinction

---

### 4. Fallback Processing

**Problem:** Some tables fail extraction, some schemas contain tables.

**Solution:** Bidirectional fallback:
- **TABLE region:** Extract → Validate → Fallback to schema if invalid
- **SCHEMA region:** Check for embedded table → Extract both if found

**Benefit:** No data loss, hybrid extraction when needed.

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

- **No TOC found:** Create default chapter
- **No sections found:** Create fallback section per chapter
- **Schema extraction fails:** Log warning, continue
- **Table extraction fails:** Fall back to schema


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
grid_line_threshold = 2
drawing_coverage_threshold = 0.3
numeric_density_threshold = 0.15

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

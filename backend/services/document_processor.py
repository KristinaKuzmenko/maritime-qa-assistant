"""
Document processor for large maritime technical PDFs.
Uses LayoutAnalyzer + RegionClassifier + SchemaExtractor + TableExtractor workflow.
Integrates Neo4j, Qdrant, Storage for comprehensive indexing.

"""

import fitz  # PyMuPDF
import pdfplumber
import hashlib
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable, Set
from pathlib import Path
import logging

from services.embedding_service import EmbeddingService
from services.schema_extractor import SchemaExtractor
from services.layout_analyzer import LayoutAnalyzer, RegionType
from services.table_extractor import TableExtractor
from services.storage_service import StorageService
from services.graph_service import Neo4jClient
from services.vector_service import VectorService
from services.region_classifier import RegionClassifier
from services.smart_region_processor import SmartRegionProcessor
from services.entity_extractor import get_entity_extractor, EntityExtractor

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process and index large technical maritime PDFs with layout-aware extraction.
    """

    def __init__(
        self,
        graph_client: Neo4jClient,
        layout_analyzer: LayoutAnalyzer,
        schema_extractor: SchemaExtractor,
        table_extractor: TableExtractor,
        embedding_service: EmbeddingService,
        storage_service: StorageService,
        vector_service: Optional[VectorService] = None,
    ) -> None:
        self.graph = graph_client
        self.layout_analyzer = layout_analyzer
        self.schema_extractor = schema_extractor
        self.table_extractor = table_extractor
        self.embeddings = embedding_service
        self.storage = storage_service
        self.vector = vector_service

        # Get LLM service from schema_extractor for smart detection
        llm_service = getattr(schema_extractor, 'llm_service', None)
        
        # Get vision detail settings from config
        from core.config import Settings
        settings = Settings()
        
        self.region_classifier = RegionClassifier(
            llm_service=llm_service,
            caption_search_distance=400,  # Increased from 250 for better caption detection
            yolo_confidence_threshold=0.8,  # Use LLM if YOLO confidence < 0.8
            enable_llm_verification=True,
        )
        
        self.smart_processor = SmartRegionProcessor(
            table_extractor=table_extractor,
            schema_extractor=schema_extractor,
            region_classifier=self.region_classifier,
            enable_llm_detection=True,  # Enable LLM-based type detection
            vision_detail=settings.vision_detail_tables,  # Cost optimization
        )

        # Entity extractor (singleton with dictionary)
        self.entity_extractor: EntityExtractor = get_entity_extractor()

        # Regex patterns for structure detection
        self.chapter_pattern = re.compile(
            r'^(?:CHAPTER|PART|SECTION|APPENDIX)\s+([A-Z0-9]+(?:\.[0-9]+)*)\s*[-–]\s*(.+)$',
            re.IGNORECASE | re.MULTILINE,
        )
        self.section_pattern = re.compile(
            r'^([0-9]+(?:\.[0-9]+)*)\s+(.+?)$',
            re.MULTILINE,
        )
        self.reference_pattern = re.compile(
            r'(?:see |refer to |as shown in |figure |fig\.|diagram |schema |table )\s*([0-9]+(?:\.[0-9]+)*)',
            re.IGNORECASE,
        )
        
        # TOC detection patterns
        self.toc_header_pattern = re.compile(
            r'^(?:TABLE\s+OF\s+CONTENTS?|CONTENTS?|INDEX)\s*$',
            re.IGNORECASE
        )
        self.toc_entry_pattern = re.compile(
            r'^(.+?)\s*[\.…·\-_]{3,}\s*(\d+)\s*$|^(\d+(?:\.\d+)*)\s+(.+?)\s+(\d+)\s*$',
            re.MULTILINE
        )
        
        # Track TOC pages to exclude from table extraction
        self._toc_pages: Set[int] = set()
        
        # Chunking parameters
        self.chunk_size_tokens = 400
        self.chunk_overlap_tokens = 100
        
        # Section merging threshold
        self.min_section_length = 200  # chars
        
        # Chapter tracking to avoid duplicates
        self._created_chapters: Dict[str, str] = {}  # title -> chapter_id

    # -------------------------------------------------------------------------
    # MAIN DOCUMENT PROCESS PIPELINE
    # -------------------------------------------------------------------------

    async def process_document(
        self,
        pdf_path: str,
        doc_id: str,
        metadata: Dict[str, Any],
        progress_callback: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> str:
        """
        Process PDF with layout-aware extraction.
        
        Workflow:
        1. Create Document node
        2. For each page:
           a. Run layout analysis (YOLO DocLayNet)
           b. Content reclassification 
           c. Extract tables and schemas 
           d. Extract text sections (continuous tracking)
        3. Generate embeddings (text + schema context + table chunks)
        4. Post-process (cross-refs, schema-table links, similarities)
        
        :param pdf_path: Path to PDF file
        :param doc_id: Document ID
        :param metadata: Document metadata
        :param progress_callback: Optional progress callback
        :return: Document ID
        """
        doc = None
        carry_pending = None
        self._current_chapter_id = None
        self._created_chapters = {}  # Reset chapter cache for new document

        try:
            file_hash = self._calculate_file_hash(pdf_path)
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            logger.info(f"Processing document: {pdf_path} ({total_pages} pages)")

            owner = metadata.get("owner", "global")

            doc_title = metadata.get("title", Path(pdf_path).stem)
            
            doc_data = {
                "id": doc_id,
                "title": doc_title,
                "doc_type": metadata.get("doc_type", "manual"),
                "version": metadata.get("version", "1.0"),
                "language": metadata.get("language", "en"),
                "file_path": pdf_path,
                "file_hash": file_hash,
                "total_pages": total_pages,
                "metadata": metadata,
                "tags": metadata.get("tags", []),
                "owner": owner,
            }
            
            # Store doc_title for use in chunking
            self._current_doc_title = doc_title

            # Create document node in Neo4j
            await self.graph.create_document(doc_data)
            logger.info(f"Created document node: {doc_id}")

            # Extract table of contents
            toc = self._extract_toc(doc)

            # Process document in chunks (for large documents)
            chunk_size = 50
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                carry_pending, last_chapter_id = await self._process_chunk(
                    doc=doc,
                    doc_id=doc_id,
                    start_page=start_page,
                    end_page=end_page,
                    toc=toc,
                    owner=owner,
                    pending_in=carry_pending,
                )

                if last_chapter_id:
                    self._current_chapter_id = last_chapter_id

                # Update progress
                progress = (end_page / total_pages) * 100.0
                await self.graph.update_document_status(
                    doc_id,
                    "processing",
                    {"progress": progress},
                )
                if progress_callback:
                    try:
                        await progress_callback(progress)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

            # Finalize any pending small section
            if carry_pending:
                logger.info(
                    f"Finalizing pending small section at end of document: "
                    f"'{carry_pending.get('title', 'Untitled')}' "
                    f"({len(carry_pending.get('content', ''))} chars)"
                )
                await self._create_section_with_chunking(
                    section_data=carry_pending,
                    chapter_id=self._current_chapter_id,
                    doc_id=doc_id,
                    doc_title=self._current_doc_title,
                    owner=owner,
                    merged_sections=[carry_pending["number"]] if carry_pending.get("number") else [],
                )
            else:
                logger.info("No pending small section to finalize at end of document")

            # Post-processing: similarities, cross-references, schema-table links
            await self._post_process_document(doc_id)

            # Collect statistics
            stats = await self.graph.get_document_stats(doc_id)
            
            # Count text chunks from Qdrant
            if self.vector:
                text_chunks_count = await self.vector.count_text_chunks(doc_id)
                stats["text_chunks"] = text_chunks_count
            else:
                stats["text_chunks"] = 0
            
            # Log detailed summary
            logger.info(
                f"✅ Document processing complete: {doc_id}\n"
                f"  Title: {self._current_doc_title}\n"
                f"  Pages: {total_pages}\n"
                f"  Chapters: {stats.get('chapters', 0)}\n"
                f"  Sections: {stats.get('sections', 0)}\n"
                f"  Text chunks: {stats.get('text_chunks', 0)}\n"
                f"  Schemas: {stats.get('schemas', 0)}\n"
                f"  Tables: {stats.get('tables', 0)}\n"
                f"  Table chunks: {stats.get('table_chunks', 0)}\n"
                f"  Entities: {stats.get('entities', 0)}"
            )

            # Mark document as complete with statistics
            await self.graph.update_document_status(
                doc_id, 
                "completed", 
                {
                    "progress": 100.0,
                    "stats": stats
                }
            )

            return doc_id

        except Exception as e:
            logger.exception(f"Error processing document: {e}")
            
            if doc_id:
                await self.graph.update_document_status(
                    doc_id,
                    "error",
                    {"error": str(e)},
                )
            
            raise

        finally:
            if doc:
                doc.close()
            
            # Log region classification statistics
            self.region_classifier.log_statistics()

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for change detection."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _generate_tags(self, content_type: str, title: str = "", caption: str = "", llm_summary: str = "", text_context: str = "", llm_tags: List[str] = None) -> List[str]:
        """
        Generate tags for schema or table based on content.
        
        :param content_type: 'schema' or 'table'
        :param title: Title/caption of the element (not used - not informative)
        :param caption: Caption text (not used - not informative)
        :param llm_summary: LLM-generated summary (for schemas)
        :param text_context: Table content/CSV text (for tables)
        :param llm_tags: Tags generated by LLM (optional)
        :return: List of tags
        """
        tags = []
        
        # Add LLM-generated tags if provided
        if llm_tags:
            tags.extend(llm_tags)
        
        # For schemas: analyze only LLM summary (caption/title not informative)
        # For tables: analyze table content (text_context = CSV content)
        if content_type == "schema":
            combined_text = llm_summary.lower()
        else:  # table
            combined_text = text_context.lower()
        
        # Base type tag
        if content_type == "schema":
            tags.append("diagram")
        elif content_type == "table":
            tags.append("table")
        
        # Schema-specific tags
        if content_type == "schema":
            # Diagram types
            if any(keyword in combined_text for keyword in ["p&id", "p & id", "piping and instrumentation"]):
                tags.append("P&ID")
            if any(keyword in combined_text for keyword in ["electrical", "circuit", "wiring", "schematic"]):
                tags.append("electrical")
            if any(keyword in combined_text for keyword in ["hydraulic", "pneumatic"]):
                tags.append("hydraulic-pneumatic")
            if any(keyword in combined_text for keyword in ["flowchart", "flow chart", "process flow"]):
                tags.append("flowchart")
            if any(keyword in combined_text for keyword in ["layout", "arrangement", "plan"]):
                tags.append("layout")
            if any(keyword in combined_text for keyword in ["assembly", "exploded view"]):
                tags.append("assembly")
        
        # Table-specific tags
        if content_type == "table":
            # Table types
            if any(keyword in combined_text for keyword in ["specification", "specs", "technical data"]):
                tags.append("specifications")
            if any(keyword in combined_text for keyword in ["parts list", "part list", "components", "spare parts"]):
                tags.append("parts-list")
            if any(keyword in combined_text for keyword in ["parameter", "setting", "configuration"]):
                tags.append("parameters")
            if any(keyword in combined_text for keyword in ["schedule", "maintenance", "inspection"]):
                tags.append("maintenance")
            if any(keyword in combined_text for keyword in ["performance", "rating", "capacity"]):
                tags.append("performance")
        
        # Common technical tags
        if any(keyword in combined_text for keyword in ["fuel", "diesel", "bunker"]):
            tags.append("fuel-system")
        if any(keyword in combined_text for keyword in ["cooling", "cooler", "heat exchanger"]):
            tags.append("cooling-system")
        if any(keyword in combined_text for keyword in ["lubrication", "lube oil", "lub oil"]):
            tags.append("lubrication-system")
        if any(keyword in combined_text for keyword in ["exhaust", "emission"]):
            tags.append("exhaust-system")
        if any(keyword in combined_text for keyword in ["starting", "air start"]):
            tags.append("starting-system")
        if any(keyword in combined_text for keyword in ["pump", "pumping"]):
            tags.append("pump")
        if any(keyword in combined_text for keyword in ["valve", "valves"]):
            tags.append("valve")
        if any(keyword in combined_text for keyword in ["motor", "engine", "generator"]):
            tags.append("motor-engine")
        if any(keyword in combined_text for keyword in ["sensor", "gauge", "meter", "monitor"]):
            tags.append("instrumentation")
        
        # Remove duplicates and return
        return list(set(tags))

    def _extract_toc(self, doc: fitz.Document) -> List[Dict]:
        """
        Extract table of contents from PDF metadata AND by scanning pages.
        Also identifies TOC pages to exclude from table extraction.
        """
        # Reset TOC pages tracking
        self._toc_pages = set()
        
        # Try PDF metadata TOC first
        toc = doc.get_toc()
        toc_entries = []
        
        for level, title, page in toc:
            # Apply filtering to remove technical codes/diagram labels
            # Skip entries like "TITLE-2 Model (1)", "100L-1 Model (1)", "FIG-3 Diagram"
            title_stripped = title.strip()
            
            # Filter Pattern 1: Code-Number format (TITLE-2, 100L-1, etc.)
            if re.match(r'^[A-Z0-9]{2,}-\d+', title_stripped):
                logger.debug(f"Filtered TOC entry (technical code): '{title}'")
                continue
            
            # Filter Pattern 2: Ends with (1), (2), etc. - usually figure/table references
            if re.search(r'\(\d+\)\s*$', title_stripped):
                logger.debug(f"Filtered TOC entry (ends with number): '{title}'")
                continue
            
            # Re-determine level based on content (don't trust PDF metadata level)
            adjusted_level = self._determine_toc_level(title)
            toc_entries.append({"level": adjusted_level, "title": title, "page": page})
        
        if toc_entries:
            logger.info(f"Found {len(toc_entries)} TOC entries from PDF metadata (after filtering)")
            return toc_entries
        
        # Scan ALL pages for TOC content (supports merged documents with TOC in middle)
        logger.info("No PDF TOC metadata, scanning all pages for table of contents...")
        
        # First pass: quick scan for TOC headers
        toc_start_pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Quick text extraction - just first 500 chars to find header
            text_start = page.get_text()[:500]
            
            if self.toc_header_pattern.search(text_start):
                toc_start_pages.append(page_num)
                logger.info(f"Found TOC header on page {page_num + 1}")
        
        # Process each TOC location found
        for toc_start in toc_start_pages:
            page = doc.load_page(toc_start)
            text = page.get_text()
            
            self._toc_pages.add(toc_start)
            
            # Parse TOC entries from this page
            entries = self._parse_toc_page(text, toc_start)
            toc_entries.extend(entries)
            
            # Check next few pages for continuation
            for next_page in range(toc_start + 1, min(toc_start + 5, len(doc))):
                next_text = doc.load_page(next_page).get_text()
                # If page has many dotted lines or page numbers, likely TOC continuation
                dot_matches = len(re.findall(r'[\.…·]{3,}\s*\d+', next_text))
                if dot_matches >= 3:
                    self._toc_pages.add(next_page)
                    entries = self._parse_toc_page(next_text, next_page)
                    toc_entries.extend(entries)
                    logger.info(f"TOC continues on page {next_page + 1} ({len(entries)} entries)")
                else:
                    break
        
        if toc_entries:
            # Filter: keep only level 1 entries for chapters (level 2+ are sections)
            chapter_entries = [e for e in toc_entries if e["level"] == 1]
            
            logger.info(
                f"Extracted {len(toc_entries)} TOC entries from pages {sorted(self._toc_pages)}, "
                f"{len(chapter_entries)} are chapter-level (level 1)"
            )
            
            # If multiple TOCs found (merged docs), log it
            if len(toc_start_pages) > 1:
                logger.info(f"Note: Found {len(toc_start_pages)} separate TOC sections (merged document?)")
            
            # Return only chapter-level entries for chapter creation
            # Sections will be detected from text parsing
            return chapter_entries
        else:
            logger.info("No TOC found in document")
            return []
    
    def _parse_toc_page(self, text: str, page_num: int) -> List[Dict]:
        """
        Parse TOC entries from page text.
        Handles formats like:
        - "Chapter 1 - Introduction .............. 5"
        - "1.2 System Overview .... 12"
        - "SAFETY PRECAUTIONS          3"
        - "3.0 INSTALLATION INSTRUCTIONS" (no page number)
        """
        entries = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Skip TOC header itself
            if self.toc_header_pattern.match(line):
                continue
            
            # Pattern 1: "Title .............. 5"
            match1 = re.match(r'^(.+?)\s*[\.…·\-_]{3,}\s*(\d+)\s*$', line)
            if match1:
                title = match1.group(1).strip()
                page = int(match1.group(2))
                level = self._determine_toc_level(title)
                entries.append({"level": level, "title": title, "page": page})
                continue
            
            # Pattern 2: "1.2.3 Title      12"
            match2 = re.match(r'^(\d+(?:\.\d+)*)\s+(.+?)\s{2,}(\d+)\s*$', line)
            if match2:
                number = match2.group(1)
                title = f"{number} {match2.group(2).strip()}"
                page = int(match2.group(3))
                level = number.count('.') + 1
                entries.append({"level": level, "title": title, "page": page})
                continue
            
            # Pattern 3: "CHAPTER 1 - Title    5" or just page number at end
            # BUT: be more strict to avoid matching schema captions like "TITLE-2 Model (1)"
            match3 = re.match(r'^(.+?)\s{2,}(\d+)\s*$', line)  # Require 2+ spaces before page number
            if match3 and len(match3.group(1)) > 5:  # Require longer titles
                title = match3.group(1).strip()
                
                # Skip if it looks like a technical code/diagram label
                # Examples to skip: "TITLE-2 Model (1)", "100L-1 Model", "FIG-3 Diagram"
                if re.match(r'^[A-Z0-9]{2,}-\d+\s', title):  # Skip codes like "TITLE-2", "100L-1"
                    continue
                if re.search(r'\(\d+\)\s*$', title):  # Skip titles ending with (1), (2)
                    continue
                
                # Avoid matching random lines with numbers at start
                if not re.match(r'^\d', title) or re.match(r'^\d+\.', title):
                    page = int(match3.group(2))
                    level = self._determine_toc_level(title)
                    entries.append({"level": level, "title": title, "page": page})
                    continue
            
            # Pattern 4: "X.0 TITLE" (no page number - TOC without page numbers)
            # Common in simple TOCs where page numbers are omitted
            match4 = re.match(r'^(\d+(?:\.\d+)*)\s+([A-Z\s&]+)$', line)
            if match4:
                number = match4.group(1)
                title_part = match4.group(2).strip()
                
                # Require substantial title (not just one word)
                if len(title_part) >= 10 and len(title_part.split()) >= 2:
                    title = f"{number} {title_part}"
                    level = number.count('.') + 1
                    
                    # Use page_num + offset as placeholder (will be corrected by text matching)
                    # This allows us to extract chapter structure even without page numbers
                    page = page_num + 1  # Placeholder page
                    
                    entries.append({"level": level, "title": title, "page": page})
                    continue
        
        return entries
    
    def _determine_toc_level(self, title: str) -> int:
        """
        Determine TOC entry level based on title format.
        Level 1 = Chapters (will create Chapter nodes)
        Level 2+ = Sections (will be detected from text, not TOC)
        """
        title_upper = title.upper().strip()
        title_stripped = title.strip()
        
        # Filter out technical codes/diagram labels (never level 1)
        # Examples: "TITLE-2 Model (1)", "100L-1 Model (1)", "FIG-3 Diagram"
        # Pattern 1: Code-Number format (TITLE-2, 100L-1, etc.)
        if re.match(r'^[A-Z0-9]{2,}-\d+', title_stripped):
            return 3  # Low level, will be filtered out
        
        # Pattern 2: Ends with (1), (2), etc. - usually figure/table references
        if re.search(r'\(\d+\)\s*$', title_stripped):
            return 3  # Low level, will be filtered out
        
        # Level 1: Explicit chapter/part/appendix markers
        if re.match(r'^(?:CHAPTER|PART|APPENDIX)\s+', title_upper):
            return 1
        
        # Level 1: ALL CAPS short title (likely major section)
        if title_upper == title_stripped and len(title_stripped.split()) <= 5:
            # But not if it starts with a number like "1.2 TITLE"
            if not re.match(r'^\d+\.', title_stripped):
                return 1
        
        # Level based on numbering depth
        num_match = re.match(r'^(\d+(?:\.\d+)*)\s+', title_stripped)
        if num_match:
            number = num_match.group(1)
            dot_count = number.count('.')
            
            # "1" or "2" = level 1 (chapter)
            # "1.1" = level 2 (section)
            # "1.1.1" = level 3 (subsection)
            return dot_count + 1
        
        # Roman numerals at start = level 1
        if re.match(r'^[IVX]+[\.\s]', title_upper):
            return 1
        
        return 2  # Default to level 2 (section)
    
    def is_toc_page(self, page_num: int) -> bool:
        """Check if a page is a TOC page (should not extract tables from it)."""
        return page_num in self._toc_pages

    def _clean_section_title(self, raw_title: str) -> str:
        """
        Clean and format section title for better readability.
        
        - Limits length to first sentence or 80 chars
        - Removes page numbers like "Page X of Y"
        - Removes trailing punctuation if truncated
        
        Examples:
            "Operation on max. 0.50% sulphur fuels. Page 4 of 7" 
            -> "Operation on max. 0.50% sulphur fuels"
            
            "This is a very long title that goes on and on with many words..."
            -> "This is a very long title that goes on and on with many words"
        """
        # Remove common page number patterns
        clean = re.sub(r'\s*[Pp]age\s+\d+\s+of\s+\d+\s*$', '', raw_title)
        clean = re.sub(r'\s*\(\s*\d+\s*/\s*\d+\s*\)\s*$', '', clean)
        
        # Get first sentence (stop at . ! ? followed by space or end)
        sentence_match = re.match(r'^(.+?[.!?])(?:\s|$)', clean)
        if sentence_match:
            clean = sentence_match.group(1)
        
        # Limit to 80 chars if still too long
        if len(clean) > 80:
            # Try to break at word boundary
            clean = clean[:77]
            last_space = clean.rfind(' ')
            if last_space > 40:  # Don't break too early
                clean = clean[:last_space]
            clean = clean.rstrip('.,;:') + '...'
        
        return clean.strip()

    # -------------------------------------------------------------------------
    # Chunk processing 
    # -------------------------------------------------------------------------

    async def _process_chunk(
        self,
        doc: fitz.Document,
        doc_id: str,
        start_page: int,
        end_page: int,
        toc: List[Dict],
        owner: str,
        pending_in: Optional[Dict] = None,
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Process a chunk of pages with layout-aware extraction.
        
        NEW: Uses layout analyzer to segment pages, then extracts schemas and tables
        in proper order to avoid conflicts.
        
        Returns (pending_small_section, last_chapter_id).
        """
        if not hasattr(self, '_page_to_section_map'):
            self._page_to_section_map = {}
        current_chapter_id = self._current_chapter_id
        current_section_accumulator = None
        pending_small_section = pending_in
        last_chapter_id = None

        logger.info(f"Processing pages {start_page+1} to {end_page}")
    
        # Create default chapter if needed
        if current_chapter_id is None and start_page == 0:
            logger.info("No chapter found in TOC, creating default chapter")
            default_title = doc.metadata.get("title") or "Document Content"
            current_chapter_id = await self._get_or_create_chapter(
                doc_id=doc_id,
                title=default_title,
                number="",
                start_page=1,
                level=1
            )
            last_chapter_id = current_chapter_id
            self._current_chapter_id = current_chapter_id
            logger.info(f"Set default chapter: {current_chapter_id}")

        # Open pdfplumber for table extraction
        pl_doc = pdfplumber.open(doc.name)

        # Track sections and content found in this chunk
        found_sections_in_chunk = False
        
        # Store schemas/tables per page for deferred processing
        # This ensures sections are created first, then linked to schemas/tables
        page_schemas = {}  # page_num -> list of schema dicts
        page_tables = {}   # page_num -> list of table dicts
        
        # Track which chapter each page belongs to
        page_to_chapter = {}  # page_num -> chapter_id

        # Process each page with NEW layout-aware workflow
        for page_num in range(start_page, end_page):
            page = doc.load_page(page_num)
            pl_page = pl_doc.pages[page_num]
            
            # Extract full page text for context (used by reclassifier and smart processor)
            page_text = page.get_text("text")

            # ===== STEP 1: YOLO Layout Analysis =====
            # Segment page into TABLE, SCHEMA, TEXT regions using YOLO
            yolo_regions = self.layout_analyzer.analyze_page(page, page_num)
            
            logger.debug(
                f"Page {page_num + 1}: YOLO detected {len(yolo_regions)} regions"
            )
            
            # ===== STEP 2: Content-based Reclassification =====
            # Analyze region content to override YOLO when content signals are stronger
            reclassified_regions = []
            
            for region in yolo_regions:
                if region.region_type in [RegionType.TABLE, RegionType.SCHEMA]:
                    # Reclassify based on content analysis (with LLM verification)
                    new_type = await self.region_classifier.reclassify_region(page, region)
                    
                    # Create new region with reclassified type (preserve caption_text and yolo_class_id)
                    from services.layout_analyzer import Region
                    reclassified_region = Region(
                        bbox=region.bbox,
                        region_type=new_type,
                        confidence=region.confidence,
                        page_number=page_num,
                        caption_text=region.caption_text,  # Preserve caption found by RegionClassifier
                        yolo_class_id=region.yolo_class_id,  # Preserve YOLO class ID
                    )
                    if region.caption_text:
                        logger.debug(f"✅ Preserved caption in reclassified region: {region.caption_text[:50]}")
                    reclassified_regions.append(reclassified_region)
                else:
                    # Keep TEXT regions as-is
                    reclassified_regions.append(region)
            
            # ===== STEP 2.5: Link YOLO Caption regions to schemas =====
            # Find Caption regions (YOLO class 0) and associate them with nearest schemas
            self._link_yolo_captions_to_schemas(page, reclassified_regions)
            
            # ===== STEP 3: Smart Region Processing with Fallback =====
            # Process each region with intelligent extraction and fallback logic
            page_schemas_list = []
            page_tables_list = []
            page_text_chunks_list = []  # For LLM-extracted text regions
            
            safe_doc_id = self._sanitize(doc_id)
            
            # Skip table extraction on TOC pages (pdfplumber often misdetects TOC as tables)
            is_toc_page = self.is_toc_page(page_num)
            if is_toc_page:
                logger.info(f"Page {page_num + 1}: Skipping table extraction (TOC page)")
            
            # Counters for unique ID generation (multiple schemas/tables per page)
            schema_counter = 0
            table_counter = 0
            
            for region in reclassified_regions:
                # Skip TEXT regions - they're handled by regular text extraction
                if region.region_type == RegionType.TEXT:
                    logger.debug(f"Page {page_num + 1}: Skipping TEXT region (handled by text extraction)")
                    continue
                
                if region.region_type == RegionType.TABLE:
                    # Skip tables on TOC pages
                    if is_toc_page:
                        continue
                    
                    # Try table extraction → fallback to LLM detection (TABLE/TEXT/DIAGRAM)
                    result = await self.smart_processor.process_table_region(
                        fitz_page=page,
                        pl_page=pl_page,
                        region=region,
                        doc_id=doc_id,
                        safe_doc_id=safe_doc_id,
                        page_num=page_num,
                        full_page_text=page_text,
                        section_id=None,  # Will be linked later
                    )
                    
                    # Collect based on result type
                    if result['type'] == 'table':
                        # Successfully extracted as table
                        page_tables_list.extend(result['chunks'])
                    elif result['type'] == 'text':
                        # LLM determined it's text content
                        page_text_chunks_list.extend(result['chunks'])
                    elif result['type'] in ('schema', 'schema_fallback'):
                        # Diagram/schema content
                        page_schemas_list.extend(result['chunks'])
                
                elif region.region_type == RegionType.SCHEMA:
                    # Try table inside → always extract schema
                    result = await self.smart_processor.process_schema_region(
                        fitz_page=page,
                        pl_page=pl_page,
                        region=region,
                        doc_id=doc_id,
                        safe_doc_id=safe_doc_id,
                        page_num=page_num,
                        full_page_text=page_text,
                        section_id=None,  # Will be linked later
                        schema_idx=schema_counter,  # Unique index for this page
                    )
                    schema_counter += 1  # Increment for next schema on this page
                    
                    # Collect based on result type
                    if result['type'] == 'hybrid':
                        # Both table and schema extracted (pdfplumber found embedded table)
                        # Track relationship for immediate linking
                        schema_chunk = None
                        table_chunks = []
                        
                        for chunk in result['chunks']:
                            if chunk['type'] == 'table':
                                page_tables_list.append(chunk)
                                table_chunks.append(chunk)
                            elif chunk['type'] == 'schema':
                                page_schemas_list.append(chunk)
                                schema_chunk = chunk
                        
                        # Mark schema with embedded table IDs (for immediate linking)
                        if schema_chunk and table_chunks:
                            if 'metadata' not in schema_chunk:
                                schema_chunk['metadata'] = {}
                            schema_chunk['metadata']['embedded_table_ids'] = [
                                tc['metadata']['table_id'] for tc in table_chunks
                            ]
                            logger.info(
                                f"Hybrid schema on page {page_num + 1}: "
                                f"schema {schema_chunk['id']} has {len(table_chunks)} embedded table(s) (pdfplumber)"
                            )
                    elif result['type'] == 'schema':
                        # Pure schema
                        page_schemas_list.extend(result['chunks'])
            
            # Store extracted regions for later processing
            if page_schemas_list:
                page_schemas[page_num] = page_schemas_list
                logger.info(
                    f"Page {page_num + 1}: extracted {len(page_schemas_list)} schema chunk(s)"
                )
            
            if page_tables_list:
                page_tables[page_num] = page_tables_list
                logger.info(
                    f"Page {page_num + 1}: extracted {len(page_tables_list)} table chunk(s)"
                )
            
            # Process LLM-extracted text chunks (add to page text for section processing)
            if page_text_chunks_list:
                for text_chunk in page_text_chunks_list:
                    # Append LLM-extracted text to page_text for normal text processing
                    llm_text = text_chunk.get('content', '')
                    if llm_text:
                        page_text = page_text + "\n\n" + llm_text
                logger.info(
                    f"Page {page_num + 1}: extracted {len(page_text_chunks_list)} text region(s) via LLM"
                )
            
            # # Get occupied bboxes for text extraction (from reclassified regions)
            # occupied_bboxes = self.layout_analyzer.get_occupied_bboxes(
            #     reclassified_regions,
            #     region_types=[RegionType.TABLE, RegionType.SCHEMA],
            # )
            
            # ===== STEP 4: Check for Chapter Headers =====
            # Try to match TOC entry by page number first
            toc_entry = next((t for t in toc if t["page"] == page_num + 1 and t["level"] == 1), None)
            
            # Also check if any TOC chapter title appears on this page (for merged docs with offset pages)
            if not toc_entry:
                page_text_lower = page_text.lower() if page_text else ""
                for t in toc:
                    if t["level"] == 1:
                        # Normalize title for comparison
                        toc_title_lower = t["title"].lower().strip()
                        # Check if TOC title appears at start of any line on this page
                        if toc_title_lower in page_text_lower:
                            # Verify it's actually a header (appears near top or as standalone line)
                            for line in page_text.split('\n')[:15]:  # Check first 15 lines
                                if toc_title_lower in line.lower().strip():
                                    toc_entry = t
                                    logger.debug(f"Matched TOC chapter '{t['title']}' by title on page {page_num + 1}")
                                    break
                            if toc_entry:
                                break
            
            if toc_entry and toc_entry["level"] == 1:
                # Finalize any accumulated section
                if current_section_accumulator:
                    pending_small_section = await self._finalize_section(
                        current_section_accumulator,
                        current_chapter_id,
                        doc_id,
                        owner,
                        pending_small_section
                    )
                
                # Get or create chapter (prevents duplicates)
                chapter_id = await self._get_or_create_chapter(
                    doc_id=doc_id,
                    title=toc_entry["title"],
                    number=self._extract_chapter_number(toc_entry["title"]),
                    start_page=page_num + 1,
                    level=toc_entry.get("level", 1)
                )
                
                current_chapter_id = chapter_id
                last_chapter_id = chapter_id
                current_section_accumulator = None
                
                logger.info(f"Set chapter from TOC: {toc_entry['title']} (page {page_num + 1})")
            
            # ===== STEP 5: Extract Text and Parse Sections =====
            
            # Track which chapter this page belongs to (MUST be after TOC check)
            if current_chapter_id:
                page_to_chapter[page_num] = current_chapter_id
                logger.debug(f"📖 Page {page_num + 1} → Chapter {current_chapter_id[:8]}")
            else:
                logger.warning(f"⚠️ Page {page_num + 1} has NO current_chapter_id!")
        
            text = page.get_text()
            lines = text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line_stripped = line.strip()
                
                if not line_stripped:
                    continue
                
                # Check for chapter header in text (if no TOC entry on this page)
                # ONLY check first 15 lines to avoid creating chapters from TOC listings in page content
                if (not toc_entry or toc_entry.get("level") != 1) and line_idx < 15:
                    chapter_match = self.chapter_pattern.match(line_stripped)
                    
                    # Also check for major numbered divisions (X.0 format) that should be chapters
                    major_division_match = None
                    if not chapter_match:
                        # Pattern: "3.0 TITLE" or "4.0 TITLE" (major division with .0)
                        # Must be: single digit, followed by .0, followed by text in CAPS or Title Case
                        major_division_match = re.match(
                            r'^([0-9]\.0)\s+(.+)$',
                            line_stripped
                        )
                        # Verify the title part is substantial (not just a word fragment)
                        if major_division_match:
                            title_part = major_division_match.group(2).strip()
                            # Require at least 3 words or 15 chars, and mostly uppercase for chapter-level
                            if len(title_part) < 15 or len(title_part.split()) < 2:
                                major_division_match = None
                    
                    if chapter_match or major_division_match:
                        if chapter_match:
                            chapter_number = chapter_match.group(1)
                            chapter_title = chapter_match.group(2).strip()
                            full_title = f"{line_stripped}"
                        else:
                            # Major division match
                            chapter_number = major_division_match.group(1)
                            chapter_title = major_division_match.group(2).strip()
                            full_title = f"{line_stripped}"
                        
                        # Finalize any accumulated section
                        if current_section_accumulator:
                            pending_small_section = await self._finalize_section(
                                current_section_accumulator,
                                current_chapter_id,
                                doc_id,
                                owner,
                                pending_small_section
                            )
                        
                        # Get or create chapter from text pattern
                        chapter_id = await self._get_or_create_chapter(
                            doc_id=doc_id,
                            title=full_title,
                            number=chapter_number,
                            start_page=page_num + 1,
                            level=1
                        )
                        
                        current_chapter_id = chapter_id
                        last_chapter_id = chapter_id
                        current_section_accumulator = None
                        
                        logger.info(f"Set chapter from text: {full_title} (page {page_num + 1})")
                        continue  # Don't process this line as section
                
                # Check for section header
                if self._is_section_header(line_stripped):
                    found_sections_in_chunk = True
                    
                    # Finalize accumulated section
                    if current_section_accumulator:
                        pending_small_section = await self._finalize_section(
                            current_section_accumulator,
                            current_chapter_id,
                            doc_id,
                            owner,
                            pending_small_section
                        )
                    
                    # Start new section
                    section_number = self._extract_section_number(line)
                    clean_title = self._clean_section_title(line_stripped)
                    current_section_accumulator = {
                        "number": section_number,
                        "title": clean_title,
                        "content_lines": [],
                        "start_page": page_num,
                        "end_page": page_num,
                        "chapter_id": current_chapter_id,
                    }
                    
                    logger.debug(f"Started section: {clean_title} (original: {line_stripped})")
                
                else:
                    # Add to current section
                    if current_section_accumulator:
                        # Store text with page information
                        current_section_accumulator["content_lines"].append({
                            "text": line,
                            "page": page_num
                        })
                        current_section_accumulator["end_page"] = page_num
                    else:
                        # No active section - start unnamed section
                        if len(line_stripped) > 20:
                            current_section_accumulator = {
                                "number": "",
                                "title": f"Content starting page {page_num + 1}",
                                "content_lines": [{"text": line, "page": page_num}],
                                "start_page": page_num,
                                "end_page": page_num,
                                "chapter_id": current_chapter_id,
                            }
        
        # Close pdfplumber doc
        pl_doc.close()
        
        # ===== STEP 6: Finalize Last Section =====
        if current_section_accumulator:
            pending_small_section = await self._finalize_section(
                current_section_accumulator,
                current_chapter_id,
                doc_id,
                owner,
                pending_small_section
            )
        
        # ===== STEP 7: Fallback Section Creation =====
        if not found_sections_in_chunk and current_chapter_id and end_page == len(doc):
            logger.warning(
                f"No section headers found in chapter {current_chapter_id}. "
                f"Creating fallback section for entire chapter."
            )
            
            chapter_query = """
            MATCH (c:Chapter {id: $chapter_id})
            RETURN c.start_page as start_page, c.title as chapter_title
            """
            chapter_info = await self.graph.run_query(
                chapter_query, 
                {"chapter_id": current_chapter_id}
            )
            
            if chapter_info:
                chapter_start = chapter_info[0]["start_page"] - 1
                chapter_title = chapter_info[0]["chapter_title"]
                
                chapter_text_lines = []
                for p_num in range(chapter_start, end_page):
                    p = doc.load_page(p_num)
                    chapter_text_lines.extend(p.get_text().split('\n'))
                
                chapter_content = '\n'.join(chapter_text_lines).strip()
                
                if len(chapter_content) > 50:
                    fallback_section = {
                        "number": "",
                        "title": f"{chapter_title} - Content",
                        "content": chapter_content,
                        "start_page": chapter_start,
                        "end_page": end_page - 1,
                    }
                    
                    await self._create_section_with_chunking(
                        section_data=fallback_section,
                        chapter_id=current_chapter_id,
                        doc_id=doc_id,
                        doc_title=self._current_doc_title,
                        owner=owner,
                        merged_sections=None
                    )
                    
                    logger.info(
                        f"✅ Created fallback section for chapter '{chapter_title}' "
                        f"(pages {chapter_start + 1}-{end_page})"
                    )
        
        # ===== STEP 8: Process Schemas and Tables =====
        # Now that sections are created, link schemas and tables to last section
        page_to_section = getattr(self, '_page_to_section_map', {})
        
        # Check if there are pages with tables/schemas that don't have sections
        # Create a section for those pages
        pages_with_content = set(page_schemas.keys()) | set(page_tables.keys())
        unmapped_pages = sorted([p for p in pages_with_content if p not in page_to_section])
        
        if unmapped_pages:
            logger.info(
                f"Found {len(unmapped_pages)} pages with tables/schemas without sections: "
                f"pages {[p+1 for p in unmapped_pages]}"
            )
            
            # Group consecutive unmapped pages BY CHAPTER
            logger.debug(f"📚 page_to_chapter mapping: {page_to_chapter}")
            
            page_groups = []
            current_group = [unmapped_pages[0]]
            current_group_chapter = page_to_chapter.get(unmapped_pages[0])
            logger.debug(f"🏁 Starting group with page {unmapped_pages[0] + 1}, chapter={current_group_chapter[:8] if current_group_chapter else 'None'}")
            
            for page in unmapped_pages[1:]:
                page_chapter = page_to_chapter.get(page)
                logger.debug(f"   Checking page {page + 1}: chapter={page_chapter[:8] if page_chapter else 'None'}, prev_chapter={current_group_chapter[:8] if current_group_chapter else 'None'}")
                # Group if consecutive AND same chapter
                if page == current_group[-1] + 1 and page_chapter == current_group_chapter:
                    current_group.append(page)
                    logger.debug(f"   ✅ Added to current group (now {len(current_group)} pages)")
                else:
                    logger.debug(f"   🔀 Starting NEW group (chapter change or gap)")
                    page_groups.append((current_group, current_group_chapter))
                    current_group = [page]
                    current_group_chapter = page_chapter
            page_groups.append((current_group, current_group_chapter))
            
            logger.info(f"📦 Created {len(page_groups)} Visual Content groups:")
            
            # Create section for each group
            for idx, (group, group_chapter_id) in enumerate(page_groups):
                logger.info(f"   Group {idx + 1}: pages {[p+1 for p in group]}, chapter={group_chapter_id[:8] if group_chapter_id else 'None'}")
                section_title = f"Visual Content (Pages {group[0] + 1}-{group[-1] + 1})"
                # Create substantial content to avoid buffering
                content = f"""This section contains visual content (schemas and tables) from pages {group[0] + 1} to {group[-1] + 1}.
                
These pages primarily contain technical diagrams, schematics, and tabular data.
The visual elements are processed separately and linked to this section for organizational purposes.
                """
                
                # Create section_accumulator in the format expected by _finalize_section
                section_accumulator = {
                    "number": "",  # No section number for visual content sections
                    "title": section_title,
                    "start_page": group[0],
                    "end_page": group[-1],
                    "content_lines": [
                        {
                            "text": content,
                            "page": group[0]
                        }
                    ]
                }
                
                # Use chapter from group (or fallback to current/default chapter)
                chapter_id = group_chapter_id
                if not chapter_id:
                    chapter_id = getattr(self, '_current_chapter_id', None)
                if not chapter_id:
                    chapter_id = getattr(self, '_default_chapter_id', None)
                
                if chapter_id:
                    # _finalize_section returns pending_small_section (None if section was created)
                    result = await self._finalize_section(
                        section_accumulator,
                        chapter_id,
                        doc_id,
                        owner,
                        None  # no pending_small_section
                    )
                    if result is None:
                        logger.info(f"✅ Created Visual Content section for pages {group[0] + 1}-{group[-1] + 1}")
                    else:
                        logger.warning(f"⚠️ Visual Content section buffered (too small): pages {group[0] + 1}-{group[-1] + 1}")
            
            # Reload page_to_section after creating new sections
            page_to_section = getattr(self, '_page_to_section_map', {})
            logger.debug(f"Updated page_to_section mapping. Now covers pages: {sorted(page_to_section.keys())}")
        
        # Process schemas with page-aware section linking
        schemas_processed = 0
        total_schema_chunks = sum(len(chunks) for chunks in page_schemas.values())
        
        logger.info(
            f"📊 Processing {len(page_schemas)} pages with schemas, "
            f"total {total_schema_chunks} schema chunks to process"
        )
        
        for page_num, schema_chunks in page_schemas.items():
            # Find section for this page
            section_id = page_to_section.get(page_num)
            
            logger.debug(
                f"Page {page_num + 1}: Looking for section for schema. "
                f"Direct match: {section_id}"
            )
            
            # If no section on exact page, try to find nearest section
            if not section_id:
                # Find section on previous pages (content continues)
                for p in range(page_num, start_page - 1, -1):
                    if p in page_to_section:
                        section_id = page_to_section[p]
                        logger.debug(f"Page {page_num + 1}: Found section from page {p + 1}")
                        break
            
            # If still no section, use last created section as fallback
            if not section_id:
                section_id = getattr(self, '_last_section_id', None)
                logger.warning(
                    f"Page {page_num + 1}: No section found for schema, using last section: {section_id}"
                )
            
            logger.info(
                f"Page {page_num + 1}: processing {len(schema_chunks)} schema chunk(s), "
                f"section_id={section_id}"
            )
            
            # Process each schema chunk
            for schema_chunk in schema_chunks:
                try:
                    await self._process_schema_chunk(
                        schema_chunk=schema_chunk,
                        section_id=section_id,
                        doc_id=doc_id,
                        owner=owner,
                    )
                    schemas_processed += 1
                except Exception as e:
                    logger.error(
                        f"Failed to process schema chunk on page {page_num + 1}: {e}",
                        exc_info=True
                    )

        # Process tables with page-aware section linking  
        tables_processed = 0
        total_table_chunks = sum(len(chunks) for chunks in page_tables.values())
        
        logger.info(
            f"Processing {len(page_tables)} pages with tables, "
            f"total {total_table_chunks} table chunks to process"
        )
        
        for page_num, table_chunks in page_tables.items():
            # Find section for this page (same logic as schemas)
            section_id = page_to_section.get(page_num)
            
            logger.debug(
                f"Page {page_num + 1}: Looking for section. "
                f"Direct match: {section_id}, page_to_section keys: {sorted(page_to_section.keys())}"
            )
            
            if not section_id:
                for p in range(page_num, start_page - 1, -1):
                    if p in page_to_section:
                        section_id = page_to_section[p]
                        logger.debug(f"Page {page_num + 1}: Found section from page {p + 1}")
                        break
            
            if not section_id:
                section_id = getattr(self, '_last_section_id', None)
                logger.warning(
                    f"Page {page_num + 1}: No section found in range, using last section: {section_id}"
                )
            
            logger.info(
                f"Page {page_num + 1}: processing {len(table_chunks)} table chunk(s), "
                f"section_id={section_id}"
            )
            
            # Group chunks by table_id to handle multi-chunk tables
            chunks_by_table = {}
            for table_chunk in table_chunks:
                table_id = table_chunk["metadata"]["table_id"]
                if table_id not in chunks_by_table:
                    chunks_by_table[table_id] = []
                chunks_by_table[table_id].append(table_chunk)
            
            # Sort chunks within each table by chunk_index
            for table_id in chunks_by_table:
                chunks_by_table[table_id].sort(key=lambda c: c["metadata"].get("chunk_index", 0))
            
            logger.debug(
                f"Page {page_num + 1}: grouped into {len(chunks_by_table)} table(s)"
            )
            
            # Process each table (all its chunks together)
            for table_id, chunks in chunks_by_table.items():
                try:
                    logger.debug(
                        f"Processing table {table_id} with {len(chunks)} chunk(s) on page {page_num + 1}"
                    )
                    await self._process_table_with_chunks(
                        table_chunks=chunks,
                        section_id=section_id,
                        doc_id=doc_id,
                        owner=owner,
                    )
                    tables_processed += 1
                except Exception as e:
                    logger.error(
                        f"❌ Failed to process table {table_id} with {len(chunks)} chunk(s) on page {page_num + 1}: {e}",
                        exc_info=True
                    )

        # Log summary
        if schemas_processed > 0 or tables_processed > 0:
            logger.info(
                f"Processed {schemas_processed} schemas and {tables_processed} tables "
                f"with page-aware section linking"
            )
        
        return pending_small_section, last_chapter_id

    # -------------------------------------------------------------------------
    # Schema processing 
    # -------------------------------------------------------------------------

    async def _process_schema_chunk(
        self,
        schema_chunk: Dict[str, Any],
        section_id: Optional[str],
        doc_id: str,
        owner: str,
    ) -> None:
        """
        Process schema chunk from SmartRegionProcessor.
        
        Schema chunks come with pre-built metadata and enhanced context.
        
        :param schema_chunk: Schema chunk dict from SmartRegionProcessor
        :param section_id: Parent section ID
        :param doc_id: Document ID
        :param owner: Owner identifier
        """
        schema_id = schema_chunk["id"]
        metadata = schema_chunk["metadata"]
        
        # Generate tags based on content (only LLM summary is informative for schemas)
        tags = self._generate_tags(
            content_type="schema",
            llm_summary=metadata.get("llm_summary", "")
        )
        
        # Build schema data for Neo4j node
        schema_data = {
            "id": schema_id,
            "doc_id": doc_id,
            "page_number": schema_chunk["page_number"],
            "title": metadata["title"],
            "caption": metadata["caption"],
            "file_path": metadata["file_path"],
            "thumbnail_path": metadata["thumbnail_path"],
            "bbox": metadata["bbox"],
            "confidence": metadata["confidence"],
            "text_context": schema_chunk["content"],  # Rich context (includes LLM summary if generated)
            "llm_summary": metadata.get("llm_summary", ""),  # Store LLM summary separately
            "tags": tags,
        }
        
        logger.debug(
            f"Creating Schema {schema_id} on page {schema_chunk['page_number']} "
            f"with section_id={section_id}, tags={tags}"
        )
        
        # Create Schema node in Neo4j
        await self.graph.create_schema(schema_data, section_id=section_id)
        
        logger.info(
            f"Created Schema node: {schema_id} "
            f"(page {schema_chunk['page_number']})" +
            (f" with tags: {', '.join(tags)}" if tags else "")
        )
        
        # Link embedded tables if detected by pdfplumber (hybrid case)
        # This handles small tables inside schema bbox that YOLO missed
        embedded_table_ids = metadata.get('embedded_table_ids', [])
        if embedded_table_ids:
            for table_id in embedded_table_ids:
                try:
                    await self.graph.link_schema_to_table(schema_id, table_id)
                    logger.info(
                        f"Linked Schema {schema_id} to embedded Table {table_id} "
                        f"(pdfplumber detected table inside schema bbox)"
                    )
                except Exception as e:
                    logger.error(f"Failed to link schema {schema_id} to table {table_id}: {e}")
        
        # NOTE: _link_schemas_to_tables() in post-processing also runs as fallback
        # for adjacent tables (BOM below schema) that YOLO detected separately

        # Extract entities from schema context using EntityExtractor
        caption = metadata["caption"]
        text_context = schema_chunk["content"]
        
        # Extract entities using new method
        schema_entity_ids, schema_system_ids = await self._extract_entities_for_schema(
            text=f"{caption} {text_context}",
            schema_id=schema_id,
            doc_id=doc_id,
        )
        
        # Index schema in Qdrant with enhanced context
        if self.vector is not None:
            embedding_text = schema_chunk["content"]  
            
            if embedding_text.strip():
                await self.vector.add_schema_embedding(
                    schema_id=schema_id,
                    text=embedding_text,
                    doc_id=doc_id,
                    page=schema_chunk["page_number"],
                    caption=caption,
                    system_ids=schema_system_ids,
                    entity_ids=schema_entity_ids,
                    owner=owner,
                )
                
                logger.info(f"✅ Indexed schema {schema_id} in Qdrant (page {schema_chunk['page_number']})")
            else:
                logger.warning(f"⚠️ Schema {schema_id} has empty content, skipping Qdrant indexing")
        else:
            logger.warning(f"⚠️ VectorService is None, skipping Qdrant indexing for schema {schema_id}")


    # -------------------------------------------------------------------------
    # Table processing
    # -------------------------------------------------------------------------

    async def _process_table_with_chunks(
        self,
        table_chunks: List[Dict[str, Any]],
        section_id: Optional[str],
        doc_id: str,
        owner: str,
    ) -> None:
        """
        Process all chunks of a single table together.
        
        This method handles multi-chunk tables correctly by:
        1. Combining all chunk contents into normalized_text
        2. Creating the parent Table node with full text
        3. Creating individual TableChunk nodes
        4. Indexing each chunk in Qdrant
        
        :param table_chunks: List of all chunks for this table (sorted by chunk_index)
        :param section_id: Section ID to link table to
        :param doc_id: Document ID
        :param owner: Owner identifier
        """
        if not table_chunks:
            logger.warning("_process_table_with_chunks called with empty table_chunks list")
            return
        
        # Get metadata from first chunk (same for all chunks)
        first_chunk = table_chunks[0]
        metadata = first_chunk["metadata"]
        table_id = metadata["table_id"]
        
        # Combine all chunk contents into full normalized_text
        combined_text = "\n\n".join(chunk["content"] for chunk in table_chunks)
        
        logger.debug(
            f"Processing table {table_id}: {len(table_chunks)} chunk(s), "
            f"combined_length={len(combined_text)}, "
            f"section_id={section_id}"
        )
        
        # Generate tags based on combined table content + LLM tags
        tags = self._generate_tags(
            content_type="table",
            text_context=combined_text,  # Use full text for tag generation
            llm_tags=metadata.get("llm_tags", [])
        )
        
        # Build table data for parent Table node
        table_data = {
            "id": table_id,
            "doc_id": doc_id,
            "page_number": first_chunk["page_number"],
            "title": metadata["title"],
            "caption": metadata["caption"],
            "rows": metadata["rows"],
            "cols": metadata["cols"],
            "file_path": metadata["file_path"],
            "thumbnail_path": metadata["thumbnail_path"],
            "csv_path": metadata["csv_path"],
            "bbox": metadata["bbox"],
            "text_preview": combined_text[:500],  # Preview from start of full text
            "normalized_text": combined_text,     # FULL text from ALL chunks
            "tags": tags,
        }
        
        llm_tags = metadata.get("llm_tags", [])
        tags_info = f"tags: {', '.join(tags)}" if tags else "no tags"
        if llm_tags:
            tags_info += f" (LLM suggested: {', '.join(llm_tags)})"
        
        logger.debug(f"Creating Table node: {table_id} with {tags_info}")
        
        # Create parent Table node in Neo4j
        await self.graph.create_table(table_data, section_id=section_id)
        
        logger.info(
            f"Created Table node: {table_id} "
            f"(page {first_chunk['page_number']}, "
            f"{metadata['rows']}x{metadata['cols']}, "
            f"{len(table_chunks)} chunk(s), "
            f"total_text={len(combined_text)} chars) with {tags_info}"
        )
        
        # Extract entities from FULL table text for graph linking
        table_entity_ids, table_system_ids = await self._extract_entities_for_table(
            text=combined_text,
            table_id=table_id,
            doc_id=doc_id,
        )
        
        # Now create TableChunk nodes and index in Qdrant
        for chunk_idx, table_chunk in enumerate(table_chunks):
            chunk_text = table_chunk["content"]
            chunk_index = metadata.get("chunk_index", 0) if chunk_idx == 0 else chunk_idx
            
            # Extract entities from this specific chunk for Qdrant metadata
            extraction = self.entity_extractor.extract_from_text(
                text=chunk_text,
                extract_systems=True,
                extract_components=True,
                link_hierarchy=True,
            )
            chunk_entity_ids = extraction["entity_ids"]
            chunk_system_ids = extraction["systems"]
            
            # Create TableChunk node in Neo4j
            logger.debug(
                f"Creating TableChunk: parent_table_id={table_id}, "
                f"chunk_index={chunk_index}, "
                f"content_length={len(chunk_text)}"
            )
            
            chunk_id = await self.graph.create_table_chunk(
                parent_table_id=table_id,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                total_chunks=len(table_chunks),
                doc_id=doc_id,
                page_number=table_chunk["page_number"],
            )
            
            logger.info(
                f"✅ Created TableChunk {chunk_id} "
                f"({chunk_index + 1}/{len(table_chunks)}) "
                f"for table {table_id}"
            )
            
            # Index chunk in Qdrant
            if self.vector is not None:
                await self.vector.add_table_chunk(
                    chunk_id=chunk_id,
                    table_id=table_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    doc_id=doc_id,
                    page=table_chunk["page_number"],
                    table_title=metadata["title"],
                    table_caption=metadata.get("caption", ""),
                    rows=metadata["rows"],
                    cols=metadata["cols"],
                    total_chunks=len(table_chunks),
                    system_ids=chunk_system_ids,
                    entity_ids=chunk_entity_ids,
                    owner=owner,
                )
                
                logger.debug(
                    f"Indexed TableChunk {chunk_id} in Qdrant "
                    f"({chunk_index + 1}/{len(table_chunks)})"
                )
            else:
                logger.warning(
                    f"⚠️ VectorService is None, skipping Qdrant indexing for chunk {chunk_id}"
                )

    async def _process_table_chunk_from_smart_processor(
        self,
        table_chunk: Dict[str, Any],
        section_id: Optional[str],
        doc_id: str,
        owner: str,
    ) -> None:
        """
        Process table chunk from SmartRegionProcessor.
        
        Table chunks come with pre-built metadata. Multiple chunks may share
        the same table_id (from chunking large tables).
        
        :param table_chunk: Table chunk dict from SmartRegionProcessor
        :param section_id: Parent section ID
        :param doc_id: Document ID
        :param owner: Owner identifier
        """
        metadata = table_chunk["metadata"]
        table_id = metadata["table_id"]
        
        logger.debug(
            f"Processing table chunk: table_id={table_id}, "
            f"chunk_index={metadata.get('chunk_index', 0)}, "
            f"content_length={len(table_chunk['content'])}, "
            f"section_id={section_id}"
        )
        
        # Check if parent table node already exists (for multi-chunk tables)
        # If this is the first chunk (chunk_index=0), create parent Table node
        if metadata.get("chunk_index", 0) == 0:
            # Generate tags based on table content + LLM tags if extracted by LLM
            tags = self._generate_tags(
                content_type="table",
                text_context=table_chunk["content"],  # CSV content
                llm_tags=metadata.get("llm_tags", [])  # Tags from LLM if table was processed by LLM
            )
            
            # Build table data for Neo4j node
            table_data = {
                "id": table_id,
                "doc_id": doc_id,
                "page_number": table_chunk["page_number"],
                "title": metadata["title"],
                "caption": metadata["caption"],
                "rows": metadata["rows"],
                "cols": metadata["cols"],
                "file_path": metadata["file_path"],
                "thumbnail_path": metadata["thumbnail_path"],
                "csv_path": metadata["csv_path"],
                "bbox": metadata["bbox"],
                "text_preview": table_chunk["content"][:500],
                "normalized_text": table_chunk["content"],
                "tags": tags,
            }
            
            llm_tags = metadata.get("llm_tags", [])
            tags_info = f"tags: {', '.join(tags)}" if tags else "no tags"
            if llm_tags:
                tags_info += f" (LLM suggested: {', '.join(llm_tags)})"
            
            logger.debug(f"Creating Table node: {table_id} with {tags_info}")
            
            # Create parent Table node in Neo4j
            await self.graph.create_table(table_data, section_id=section_id)
            
            logger.info(
                f"Created Table node: {table_id} "
                f"(page {table_chunk['page_number']}, "
                f"{metadata['rows']}x{metadata['cols']}) with {tags_info}"
            )
            
            # Extract and link entities from table content using EntityExtractor
            table_entity_ids, table_system_ids = await self._extract_entities_for_table(
                text=table_chunk["content"],
                table_id=table_id,
                doc_id=doc_id,
            )
        else:
            # Subsequent chunks - extract entities for vector metadata (no graph linking)
            extraction = self.entity_extractor.extract_from_text(
                text=table_chunk["content"],
                extract_systems=True,
                extract_components=True,
                link_hierarchy=True,
            )
            table_entity_ids = extraction["entity_ids"]
            table_system_ids = extraction["systems"]
        
        #  Create TableChunk node in Neo4j
        logger.debug(
            f"Attempting to create TableChunk: parent_table_id={table_id}, "
            f"chunk_index={metadata.get('chunk_index', 0)}, "
            f"content_length={len(table_chunk['content'])}"
        )
        
        chunk_id = await self.graph.create_table_chunk(
            parent_table_id=table_id,                      
            chunk_index=metadata.get("chunk_index", 0),
            chunk_text=table_chunk["content"],
            total_chunks=metadata.get("total_chunks", 1),  
            doc_id=doc_id,                                 
            page_number=table_chunk["page_number"],        
        )
        
        logger.info(
            f"✅ Created TableChunk {chunk_id} "
            f"({metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}) "
            f"for table {table_id}"
        )
        
        # Index chunk in Qdrant 
        if self.vector is not None:
            await self.vector.add_table_chunk(  
                chunk_id=chunk_id,
                table_id=table_id,
                chunk_index=metadata.get("chunk_index", 0),
                text=table_chunk["content"],
                doc_id=doc_id,
                page=table_chunk["page_number"],
                table_title=metadata["title"],
                table_caption=metadata.get("caption", ""),
                rows=metadata["rows"],
                cols=metadata["cols"],
                total_chunks=metadata.get("total_chunks", 1),
                system_ids=table_system_ids,
                entity_ids=table_entity_ids,
                owner=owner,
            )
            
            logger.debug(f"Indexed table chunk {chunk_id} in Qdrant")

    # -------------------------------------------------------------------------
    # Section finalization
    # -------------------------------------------------------------------------

    async def _finalize_section(
        self,
        section_accumulator: Dict,
        chapter_id: str,
        doc_id: str,
        owner: str,
        pending_small_section: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Finalize accumulated section: check size, merge if needed, create in Neo4j and chunk for Qdrant.
        Returns pending_small_section if current section is too small, otherwise None.
        
        :param section_accumulator: Accumulated section data
        :param chapter_id: Parent chapter ID
        :param doc_id: Document ID
        :param owner: Owner identifier
        :param pending_small_section: Previously buffered small section (to prepend)
        :return: New pending_small_section if current is too small, otherwise None
        """
        # Assemble full content from lines with page info
        content_lines_with_pages = section_accumulator["content_lines"]
        content = '\n'.join(line["text"] for line in content_lines_with_pages)
        content = content.strip()
        
        if not content:
            logger.debug("Skipping empty section")
            return pending_small_section
        
        # Check if section is too small
        if len(content) < self.min_section_length:
            logger.info(
                f"Section '{section_accumulator['title']}' is small "
                f"({len(content)} chars < {self.min_section_length}), "
                f"buffering to merge with next section"
            )
            
            # If there's already a pending small section, merge them
            if pending_small_section:
                logger.info(
                    f"Merging with previously buffered section "
                    f"'{pending_small_section['title']}'"
                )
                merged_content = pending_small_section["content"] + "\n\n" + content
                
                return {
                    "number": section_accumulator.get("number", ""),
                    "title": section_accumulator["title"],
                    "content": merged_content,
                    "start_page": pending_small_section["start_page"],
                    "end_page": section_accumulator["end_page"],
                    "merged_sections": [
                        pending_small_section.get("number", ""),
                        section_accumulator.get("number", "")
                    ]
                }
            else:
                # Buffer this small section
                return {
                    "number": section_accumulator.get("number", ""),
                    "title": section_accumulator["title"],
                    "content": content,
                    "start_page": section_accumulator["start_page"],
                    "end_page": section_accumulator["end_page"],
                    "merged_sections": []
                }
        
        # Section is large enough - check if we should prepend pending
        merged_sections = []
        
        if pending_small_section:
            logger.info(
                f"Prepending buffered small section '{pending_small_section['title']}' "
                f"({len(pending_small_section['content'])} chars) to current section "
                f"'{section_accumulator['title']}'"
            )
            
            content = pending_small_section["content"] + "\n\n" + content
            section_accumulator["start_page"] = pending_small_section["start_page"]
            
            if pending_small_section.get("merged_sections"):
                merged_sections.extend(pending_small_section["merged_sections"])
            elif pending_small_section.get("number"):
                merged_sections.append(pending_small_section["number"])
        
        if section_accumulator.get("number"):
            merged_sections.append(section_accumulator["number"])
        
        logger.info(
            f"Finalizing section '{section_accumulator['title']}' "
            f"(pages {section_accumulator['start_page'] + 1}-{section_accumulator['end_page'] + 1}, "
            f"{len(content)} chars)"
        )
        
        if len(merged_sections) > 1:
            logger.info(f"Merged sections: {merged_sections}")
        
        # Create section in Neo4j and chunk for Qdrant
        await self._create_section_with_chunking(
            section_data={
                "number": section_accumulator.get("number", ""),
                "title": section_accumulator["title"],
                "content": content,
                "start_page": section_accumulator["start_page"],
                "end_page": section_accumulator["end_page"],
            },
            chapter_id=chapter_id,
            doc_id=doc_id,
            doc_title=self._current_doc_title,
            owner=owner,
            merged_sections=merged_sections if len(merged_sections) > 1 else None
        )
        
        return None
    
    def _stable_section_id(self, doc_id: str, chapter_id: str, number: str, start_page: int, end_page: int) -> str:
        """Generate stable section ID from components."""
        basis = f"{doc_id}:{chapter_id}:{number or 'NA'}:{start_page}-{end_page}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, basis))
    
    def _stable_chapter_id(self, doc_id: str, title: str, page: int = 0) -> str:
        """Generate stable chapter ID from components. Page is ignored to avoid duplicates."""
        # Normalize title: strip, collapse multiple spaces, uppercase for consistent ID
        normalized_title = ' '.join(title.strip().split()).upper()
        basis = f"{doc_id}:{normalized_title}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, basis))
    
    async def _get_or_create_chapter(
        self,
        doc_id: str,
        title: str,
        number: str,
        start_page: int,
        level: int = 1
    ) -> str:
        """
        Get existing chapter or create new one. Prevents duplicate chapters.
        
        :param doc_id: Document ID
        :param title: Chapter title
        :param number: Chapter number
        :param start_page: Starting page (1-indexed)
        :param level: Chapter level
        :return: chapter_id
        """
        # Normalize title for consistent matching
        normalized_title = ' '.join(title.strip().split())
        
        # Check cache first
        if normalized_title in self._created_chapters:
            chapter_id = self._created_chapters[normalized_title]
            logger.debug(f"Using existing chapter from cache: {normalized_title} -> {chapter_id}")
            
            # Update end_page if needed
            query = """
            MATCH (c:Chapter {id: $chapter_id})
            SET c.end_page = CASE 
                WHEN c.end_page IS NULL OR c.end_page < $page 
                THEN $page 
                ELSE c.end_page 
            END
            RETURN c.id as chapter_id
            """
            await self.graph.run_query(query, {"chapter_id": chapter_id, "page": start_page})
            return chapter_id
        
        # Check if chapter exists in database
        chapter_id = self._stable_chapter_id(doc_id, normalized_title)
        check_query = """
        MATCH (c:Chapter {id: $chapter_id})
        RETURN c.id as chapter_id, c.start_page as start_page
        """
        existing = await self.graph.run_query(check_query, {"chapter_id": chapter_id})
        
        if existing:
            # Chapter exists - update cache and end_page
            self._created_chapters[normalized_title] = chapter_id
            logger.info(f"Found existing chapter in DB: {normalized_title} (page {existing[0]['start_page']})")
            
            # Update end_page
            update_query = """
            MATCH (c:Chapter {id: $chapter_id})
            SET c.end_page = CASE 
                WHEN c.end_page IS NULL OR c.end_page < $page 
                THEN $page 
                ELSE c.end_page 
            END
            RETURN c.id as chapter_id
            """
            await self.graph.run_query(update_query, {"chapter_id": chapter_id, "page": start_page})
            return chapter_id
        
        # Create new chapter
        # Normalize title to prevent duplicates from spacing variations
        normalized_title = ' '.join(title.strip().split())
        
        chapter_data = {
            "id": chapter_id,
            "number": number,
            "title": normalized_title,  # Store normalized title
            "start_page": start_page,
            "end_page": start_page,  # Will be updated as we process more pages
            "level": level,
        }
        
        await self.graph.create_chapter(chapter_data, doc_id)
        self._created_chapters[normalized_title] = chapter_id
        logger.info(f"✅ Created NEW chapter: {normalized_title} (page {start_page})")
        
        return chapter_id
    
    async def _create_section_with_chunking(
        self,
        section_data: Dict,
        chapter_id: str,
        doc_id: str,
        doc_title: str,
        owner: str,
        merged_sections: Optional[List[str]] = None
    ):
        """
        Create section in Neo4j and generate chunks for Qdrant.
        
        :param section_data: Section data with content, pages, etc.
        :param chapter_id: Parent chapter ID
        :param doc_id: Document ID
        :param doc_title: Document title
        :param owner: Owner identifier
        :param merged_sections: List of section numbers that were merged (if any)
        """
        content = section_data["content"]
        section_id = self._stable_section_id(
            doc_id=doc_id,
            chapter_id=chapter_id,
            number=section_data["number"],
            start_page=section_data["start_page"],
            end_page=section_data["end_page"],
        )
        
        # Prepare section metadata for Neo4j
        neo4j_section_data = {
            "id": section_id,
            "number": section_data["number"],
            "title": section_data["title"],
            "content": content,
            "page_start": section_data["start_page"],
            "page_end": section_data["end_page"],
            "section_type": self._classify_section_type(content),
            "importance_score": self._calculate_importance_score(content),
        }
        
        # Add merged sections info if applicable
        if merged_sections:
            neo4j_section_data["merged_sections"] = merged_sections
            neo4j_section_data["is_merged"] = True
            neo4j_section_data["original_count"] = len(merged_sections)
        
        # Create Section node in Neo4j
        section_id = await self.graph.create_section(neo4j_section_data, chapter_id)
        
        # Track ALL pages this section covers (not just start_page)
        for page in range(section_data["start_page"], section_data["end_page"] + 1):
            self._page_to_section_map[page] = section_id
        
        logger.debug(
            f"Mapped pages {section_data['start_page'] + 1}-{section_data['end_page'] + 1} "
            f"to section {section_id}"
        )
        
        # Store for schema/table linking
        self._last_section_id = section_id
        
        # Entity extraction using new EntityExtractor
        extraction_result = self.entity_extractor.extract_from_text(
            text=content,
            extract_systems=True,
            extract_components=True,
            link_hierarchy=True,
        )
        
        entity_ids = extraction_result["entity_ids"]
        system_ids = extraction_result["systems"]
        
        # Create/link entities in graph
        for sys_code in extraction_result["systems"]:
            await self._create_and_link_system_entity(
                system_code=sys_code,
                section_id=section_id,
                doc_id=doc_id,
            )
        
        for comp in extraction_result["components"]:
            await self._create_and_link_component_entity(
                component=comp,
                section_id=section_id,
                doc_id=doc_id,
            )
        
        # Create chunks for Qdrant
        if self.vector is not None:
            chunks = self._create_text_chunks(
                text=content,
                chunk_size=self.chunk_size_tokens,
                overlap=self.chunk_overlap_tokens,
            )
            
            # Calculate page distribution based on chunk position
            start_page = section_data["start_page"]
            end_page = section_data["end_page"]
            total_pages = end_page - start_page + 1
            text_length = len(content)
            
            # Add each chunk to Qdrant
            chunk_pages = []
            for idx, chunk_data in enumerate(chunks):
                # Determine chunk's page based on its position in text
                chunk_mid_pos = (chunk_data["char_start"] + chunk_data["char_end"]) / 2
                page_progress = chunk_mid_pos / text_length if text_length > 0 else 0
                chunk_page = start_page + int(page_progress * total_pages)
                
                # Ensure chunk_page is within bounds
                chunk_page = max(start_page, min(end_page, chunk_page))
                chunk_pages.append(chunk_page + 1)  # Store 1-based for logging
                
                await self.vector.add_text_chunk(
                    section_id=section_id,
                    chunk_index=idx,
                    text=chunk_data["text"],
                    doc_id=doc_id,
                    doc_title=doc_title,
                    page_start=chunk_page + 1,  # Convert to 1-based
                    page_end=chunk_page + 1,    # Same page for specific chunk
                    chunk_char_start=chunk_data["char_start"],
                    chunk_char_end=chunk_data["char_end"],
                    section_number=section_data["number"],
                    section_title=section_data["title"],
                    system_ids=system_ids,
                    entity_ids=entity_ids,
                    owner=owner,
                )
            
            logger.info(
                f"Created {len(chunks)} chunks for section {section_id} "
                f"(pages {section_data['start_page'] + 1}-{section_data['end_page'] + 1}, "
                f"chunk pages: {chunk_pages})"
            )

    # -------------------------------------------------------------------------
    # Chunking with overlap
    # -------------------------------------------------------------------------

    def _create_text_chunks(
        self,
        text: str,
        chunk_size: int = 400,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with sentence boundary awareness.
        
        :param text: Full text to chunk
        :param chunk_size: Target size in tokens
        :param overlap: Overlap size in tokens
        :return: List of chunk dicts with text and char positions
        """
        chars_per_token = 4
        chunk_chars = chunk_size * chars_per_token
        overlap_chars = overlap * chars_per_token
        
        chunks = []
        start = 0
        n = len(text)
        
        while start < n:
            end = min(start + chunk_chars, n)
            
            # Try to find sentence boundary
            if end < n:
                search_start = max(end - 200, start)
                search_end = min(end + 200, n)
                
                sentence_ends = [i+1 for i in range(search_start, search_end) if text[i] in '.!?\n']
                
                if sentence_ends:
                    target = start + chunk_chars
                    end = min(sentence_ends, key=lambda x: abs(x - target))
            else:
                end = len(text)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                token_estimate = len(chunk_text) // chars_per_token
                
                if token_estimate > 500:
                    logger.warning(
                        f"Chunk exceeds target: {token_estimate} tokens "
                        f"(target {chunk_size})"
                    )
                
                chunks.append({
                    "text": chunk_text,
                    "char_start": start,
                    "char_end": end,
                    "token_estimate": token_estimate,
                })
            
            if end == n:
                break
            
            start = max(0, end - overlap_chars)
            
        return chunks

    # -------------------------------------------------------------------------
    # Section parsing helpers
    # -------------------------------------------------------------------------

    def _is_section_header(self, line: str) -> bool:
        """
        Detect whether a line is a section header.
        Enhanced to recognize multiple header patterns in technical manuals.
        """
        line = line.strip()
        
        if len(line) < 3 or len(line) > 150:
            return False
    
        # Numbered sections: "3.2.1 Title"
        if re.match(r"^[0-9]+(?:\.[0-9]+)*\s+\S", line, re.IGNORECASE):
            return True
    
        # Chapter/Part headers: "CHAPTER 5 - Title"
        if re.match(r"^(?:CHAPTER|PART|SECTION|APPENDIX)\s+[A-Z0-9]", line, re.IGNORECASE):
            return True
    
        # All-caps headers (minimum 3 words)
        if re.match(r"^[A-Z][A-Z\s]{10,}$", line):
            words = line.split()
            if len(words) >= 3:
                return True
    
        # Headers with colon: "Description:"
        if re.match(r"^[A-Z][a-zA-Z\s]{3,}:$", line):
            return True
    
        # Common header keywords
        header_keywords = [
            r"^INTRODUCTION\b", r"^DESCRIPTION\b", r"^OPERATION\b",
            r"^MAINTENANCE\b", r"^INSTALLATION\b", r"^SPECIFICATIONS?\b",
            r"^TECHNICAL DATA\b", r"^SAFETY\b", r"^PROCEDURES?\b",
            r"^COMPONENTS?\b", r"^SYSTEMS?\b", r"^OVERVIEW\b",
            r"^GENERAL\b", r"^PURPOSE\b", r"^SCOPE\b",
        ]
    
        for pattern in header_keywords:
            if re.match(pattern, line, re.IGNORECASE):
                return True
    
        return False

    def _extract_section_number(self, line: str) -> str:
        """Extract section number from header line."""
        match = re.match(r"^([0-9]+(?:\.[0-9]+)*)", line.strip())
        return match.group(1) if match else ""

    def _extract_chapter_number(self, title: str) -> str:
        """Extract chapter number from title."""
        match = re.match(r"^(?:CHAPTER|PART|SECTION|APPENDIX)\s+([A-Z0-9]+)", title, re.IGNORECASE)
        return match.group(1) if match else ""

    def _classify_section_type(self, content: str) -> str:
        """Classify section content type."""
        content_lower = content.lower()
        
        if any(k in content_lower for k in ["warning", "danger", "caution"]):
            return "warning"
        if "note:" in content_lower or "notice:" in content_lower:
            return "note"
        if re.search(r"\|.*\|.*\|", content):
            return "table"
        if re.search(r"^\s*[\d•·]\s+", content, re.MULTILINE):
            return "list"
        
        return "text"

    # -------------------------------------------------------------------------
    # Entity extraction and linking (using EntityExtractor)
    # -------------------------------------------------------------------------

    async def _create_and_link_system_entity(
        self,
        system_code: str,
        section_id: str,
        doc_id: str,
    ) -> str:
        """
        Create system entity in graph and link to section.
        
        :param system_code: Normalized system code (e.g., 'sys_fuel_oil')
        :param section_id: Section ID to link to
        :param doc_id: Document ID
        :return: Entity ID
        """
        # Get system info from dictionary
        systems = self.entity_extractor.dictionary.get("systems", {})
        system_info = None
        
        for sys_key, sys_data in systems.items():
            if sys_data.get("code") == system_code:
                system_info = sys_data
                break
        
        entity_data = {
            "code": system_code,
            "name": system_info.get("canonical", system_code) if system_info else system_code,
            "entity_type": "System",
            "system": system_info.get("canonical", "") if system_info else "",
            "tags": ["from-section"],
            "metadata": {"doc_id": doc_id, "section_id": section_id},
        }
        
        entity_id = await self.graph.create_entity(entity_data)
        await self.graph.link_section_to_entity(section_id, entity_id)
        
        # Create PART_OF relationships if parent exists
        if system_info and system_info.get("parent"):
            parent_key = system_info["parent"]
            if parent_key in systems:
                parent_code = systems[parent_key].get("code", f"sys_{parent_key}")
                parent_data = {
                    "code": parent_code,
                    "name": systems[parent_key].get("canonical", parent_key),
                    "entity_type": "System",
                    "system": systems[parent_key].get("canonical", ""),
                    "tags": ["from-section"],
                    "metadata": {"doc_id": doc_id},
                }
                parent_id = await self.graph.create_entity(parent_data)
                await self.graph.create_entity_relation(entity_id, parent_id, "PART_OF")
        
        return entity_id

    async def _create_and_link_component_entity(
        self,
        component: Dict[str, Any],
        section_id: str,
        doc_id: str,
    ) -> str:
        """
        Create component entity in graph and link to section.
        
        :param component: Component dict with 'name', 'type', 'code'
        :param section_id: Section ID to link to
        :param doc_id: Document ID
        :return: Entity ID
        """
        entity_data = {
            "code": component["code"],
            "name": component["name"].title(),
            "entity_type": "Component",
            "system": None,  # Will be linked via PART_OF if inferable
            "tags": ["from-section", component["type"]],
            "metadata": {
                "doc_id": doc_id,
                "section_id": section_id,
                "component_type": component["type"],
            },
        }
        
        entity_id = await self.graph.create_entity(entity_data)
        await self.graph.link_section_to_entity(section_id, entity_id)
        
        return entity_id

    async def _extract_entities_for_schema(
        self,
        text: str,
        schema_id: str,
        doc_id: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Extract entities from schema caption/context using EntityExtractor.
        
        :param text: Combined caption + context text
        :param schema_id: Schema ID
        :param doc_id: Document ID
        :return: Tuple of (entity_ids, system_ids)
        """
        extraction = self.entity_extractor.extract_from_text(
            text=text,
            extract_systems=True,
            extract_components=True,
            link_hierarchy=True,
        )
        
        entity_ids = extraction["entity_ids"]
        system_ids = extraction["systems"]
        
        # Create entities and link to schema
        for sys_code in extraction["systems"]:
            await self._create_system_entity_for_schema(sys_code, schema_id, doc_id)
        
        for comp in extraction["components"]:
            await self._create_component_entity_for_schema(comp, schema_id, doc_id)
        
        return entity_ids, system_ids

    async def _create_system_entity_for_schema(
        self,
        system_code: str,
        schema_id: str,
        doc_id: str,
    ) -> str:
        """Create system entity and link to schema via DEPICTS."""
        systems = self.entity_extractor.dictionary.get("systems", {})
        system_info = None
        
        for sys_key, sys_data in systems.items():
            if sys_data.get("code") == system_code:
                system_info = sys_data
                break
        
        entity_data = {
            "code": system_code,
            "name": system_info.get("canonical", system_code) if system_info else system_code,
            "entity_type": "System",
            "system": system_info.get("canonical", "") if system_info else "",
            "tags": ["from-schema"],
            "metadata": {"doc_id": doc_id, "schema_id": schema_id},
        }
        
        entity_id = await self.graph.create_entity(entity_data)
        await self.graph.link_schema_to_entity(schema_id, entity_id)
        
        return entity_id

    async def _create_component_entity_for_schema(
        self,
        component: Dict[str, Any],
        schema_id: str,
        doc_id: str,
    ) -> str:
        """Create component entity and link to schema via DEPICTS."""
        entity_data = {
            "code": component["code"],
            "name": component["name"].title(),
            "entity_type": "Component",
            "system": None,
            "tags": ["from-schema", component["type"]],
            "metadata": {
                "doc_id": doc_id,
                "schema_id": schema_id,
                "component_type": component["type"],
            },
        }
        
        entity_id = await self.graph.create_entity(entity_data)
        await self.graph.link_schema_to_entity(schema_id, entity_id)
        
        return entity_id

    async def _extract_entities_for_table(
        self,
        text: str,
        table_id: str,
        doc_id: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Extract entities from table caption/content using EntityExtractor.
        
        :param text: Combined caption + content text
        :param table_id: Table ID
        :param doc_id: Document ID
        :return: Tuple of (entity_ids, system_ids)
        """
        extraction = self.entity_extractor.extract_from_text(
            text=text,
            extract_systems=True,
            extract_components=True,
            link_hierarchy=True,
        )
        
        entity_ids = extraction["entity_ids"]
        system_ids = extraction["systems"]
        
        # Create entities and link to table
        for sys_code in extraction["systems"]:
            await self._create_system_entity_for_table(sys_code, table_id, doc_id)
        
        for comp in extraction["components"]:
            await self._create_component_entity_for_table(comp, table_id, doc_id)
        
        return entity_ids, system_ids

    async def _create_system_entity_for_table(
        self,
        system_code: str,
        table_id: str,
        doc_id: str,
    ) -> str:
        """Create system entity and link to table via MENTIONS."""
        systems = self.entity_extractor.dictionary.get("systems", {})
        system_info = None
        
        for sys_key, sys_data in systems.items():
            if sys_data.get("code") == system_code:
                system_info = sys_data
                break
        
        entity_data = {
            "code": system_code,
            "name": system_info.get("canonical", system_code) if system_info else system_code,
            "entity_type": "System",
            "system": system_info.get("canonical", "") if system_info else "",
            "tags": ["from-table"],
            "metadata": {"doc_id": doc_id, "table_id": table_id},
        }
        
        entity_id = await self.graph.create_entity(entity_data)
        await self.graph.link_table_to_entity(table_id, entity_id)
        
        return entity_id

    async def _create_component_entity_for_table(
        self,
        component: Dict[str, Any],
        table_id: str,
        doc_id: str,
    ) -> str:
        """Create component entity and link to table via MENTIONS."""
        entity_data = {
            "code": component["code"],
            "name": component["name"].title(),
            "entity_type": "Component",
            "system": None,
            "tags": ["from-table", component["type"]],
            "metadata": {
                "doc_id": doc_id,
                "table_id": table_id,
                "component_type": component["type"],
            },
        }
        
        entity_id = await self.graph.create_entity(entity_data)
        await self.graph.link_table_to_entity(table_id, entity_id)
        
        return entity_id

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def _calculate_importance_score(self, content: str) -> float:
        """Compute importance score based on critical keywords."""
        score = 0.5
        content_lower = content.lower()
        
        important_keywords = [
            "must", "shall", "critical", "essential", "mandatory",
            "required", "safety", "warning", "danger", "emergency", "caution",
        ]
        
        for keyword in important_keywords:
            if keyword in content_lower:
                score += 0.1
        
        return min(score, 1.0)

    # -------------------------------------------------------------------------
    # Post-processing 
    # -------------------------------------------------------------------------

    async def _post_process_document(self, doc_id: str) -> None:
        """
        Perform document-level post-processing.
        
        NEW: Includes schema-table linking for BOM relationships.
        """
        logger.info(f"Post-processing document: {doc_id}")
        
        # Create cross-references (section <-> schema)
        await self._create_cross_references(doc_id)
        
        # Link schemas to tables (BOM detection)
        await self._link_schemas_to_tables(doc_id)
        
        # Compute section similarities
        await self._compute_section_similarities(doc_id)
        
        # Build entity relationships
        await self._build_entity_relationships(doc_id)
        
        logger.info(f"Post-processing complete for document: {doc_id}")

    async def _create_cross_references(self, doc_id: str) -> None:
        """Link sections that reference schemas."""
        logger.info(f"Creating cross-references for document {doc_id}")
        
        query = """
        MATCH (s:Section {document_id: $doc_id})
        WHERE s.content CONTAINS 'Figure' OR s.content CONTAINS 'Fig.'
        RETURN s.id as section_id, s.content as content
        """
        
        sections = await self.graph.run_query(query, {"doc_id": doc_id})
        
        for section in sections:
            refs = self.reference_pattern.findall(section["content"])
            
            for ref_num in refs:
                schema_query = """
                MATCH (sc:Schema)
                WHERE (sc.title CONTAINS $ref_num OR sc.caption CONTAINS $ref_num)
                RETURN sc.id as schema_id
                LIMIT 1
                """
                
                result = await self.graph.run_query(schema_query, {"ref_num": ref_num})
                
                if result:
                    await self.graph.link_section_to_schema(
                        section["section_id"],
                        result[0]["schema_id"],
                        f"Reference: {ref_num}"
                    )
        
        logger.info(f"Cross-references created for document {doc_id}")

    async def _link_schemas_to_tables(self, doc_id: str) -> None:
        """
        Link schemas to nearby tables (BOM relationships).
        Supports two patterns:
        1. Tables embedded inside schema bbox (e.g., BOM on technical drawing)
        2. Tables positioned below schema (e.g., separate BOM table under diagram)
        
        FIXED: Parses bbox as JSON string and checks both spatial patterns.
        """
        logger.info(f"Linking schemas to tables for document {doc_id}")
        
        # Get all schemas and tables on same pages
        query = """
        MATCH (sc:Schema {doc_id: $doc_id})
        MATCH (t:Table {doc_id: $doc_id, page_number: sc.page_number})
        RETURN sc.id as schema_id, sc.bbox as schema_bbox, 
            sc.page_number as page, sc.title as schema_title,
            t.id as table_id, t.bbox as table_bbox, t.title as table_title
        """
        
        try:
            results = await self.graph.run_query(query, {"doc_id": doc_id})
        except Exception as e:
            logger.error(f"Error querying schemas and tables: {e}")
            return
        
        if not results:
            logger.info("No schemas or tables found on same pages")
            return
        
        # Parse bboxes and check spatial relationships
        import json
        linked_count = 0
        
        for row in results:
            try:
                # Parse JSON strings to dicts
                schema_bbox = json.loads(row["schema_bbox"]) if isinstance(row["schema_bbox"], str) else row["schema_bbox"]
                table_bbox = json.loads(row["table_bbox"]) if isinstance(row["table_bbox"], str) else row["table_bbox"]
                
                # Extract coordinates
                schema_x0 = float(schema_bbox.get("x0", 0))
                schema_y0 = float(schema_bbox.get("y0", 0))
                schema_x1 = float(schema_bbox.get("x1", 0))
                schema_y1 = float(schema_bbox.get("y1", 0))
                
                table_x0 = float(table_bbox.get("x0", 0))
                table_y0 = float(table_bbox.get("y0", 0))
                table_x1 = float(table_bbox.get("x1", 0))
                table_y1 = float(table_bbox.get("y1", 0))
                
                # Check spatial relationships
                should_link = False
                link_reason = ""
                
                # Pattern 1: Table INSIDE schema (embedded BOM on drawing)
                # Allow 5% tolerance for YOLO detection imprecision
                tolerance = 0.05
                schema_width = schema_x1 - schema_x0
                schema_height = schema_y1 - schema_y0
                
                x_margin = schema_width * tolerance
                y_margin = schema_height * tolerance
                
                is_inside = (
                    table_x0 >= (schema_x0 - x_margin) and
                    table_x1 <= (schema_x1 + x_margin) and
                    table_y0 >= (schema_y0 - y_margin) and
                    table_y1 <= (schema_y1 + y_margin)
                )
                
                if is_inside:
                    should_link = True
                    link_reason = "embedded (table inside schema)"
                    logger.debug(
                        f"Table inside schema on page {row['page']}: "
                        f"table bbox [{table_x0:.1f}, {table_y0:.1f}, {table_x1:.1f}, {table_y1:.1f}] "
                        f"inside schema [{schema_x0:.1f}, {schema_y0:.1f}, {schema_x1:.1f}, {schema_y1:.1f}]"
                    )
                
                # Pattern 2: Table BELOW schema (classic BOM layout)
                else:
                    distance = table_y0 - schema_y1
                    
                    # Table starts below schema bottom, with reasonable gap
                    if schema_y1 < table_y0 and 0 < distance < 150:
                        should_link = True
                        link_reason = f"below (distance: {distance:.1f}px)"
                        logger.debug(
                            f"Table below schema on page {row['page']}: "
                            f"schema bottom y={schema_y1:.1f}, table top y={table_y0:.1f}, "
                            f"distance={distance:.1f}px"
                        )
                
                # Create link if criteria met
                if should_link:
                    await self.graph.link_schema_to_table(
                        row["schema_id"],
                        row["table_id"],
                    )
                    linked_count += 1
                    
                    logger.info(
                        f"✅ Linked '{row['schema_title']}' ← '{row['table_title']}' "
                        f"on page {row['page']} ({link_reason})"
                    )
                else:
                    logger.debug(
                        f"❌ Not linking on page {row['page']}: "
                        f"not inside (is_inside={is_inside}) and not below "
                        f"(distance={table_y0 - schema_y1:.1f}px, need 0-150px)"
                    )
            
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"Could not parse bbox or link schema {row.get('schema_id', '?')[:8]} "
                    f"to table {row.get('table_id', '?')[:8]}: {e}"
                )
                continue
        
        logger.info(
            f"Linked {linked_count} schema-table pairs out of {len(results)} candidates "
            f"(BOM relationships)"
        )

    async def _compute_section_similarities(
        self,
        doc_id: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> None:
        """Compute and store SIMILAR_TO relationships between sections."""
        if self.vector is None:
            logger.warning("VectorService not available, skipping similarity computation")
            return
        
        logger.info(f"Computing section similarities for document {doc_id}")
        
        try:
            # Delete existing similarities
            deleted = await self.graph.delete_section_similarities(doc_id)
            if deleted > 0:
                logger.info(f"Deleted {deleted} existing similarity relationships")
            
            # Compute similarities using Qdrant
            similarities = await self.vector.compute_section_similarities(
                doc_id=doc_id,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            
            if not similarities:
                logger.warning(f"No similarities found for document {doc_id}")
                return
            
            # Store in Neo4j
            created = await self.graph.create_section_similarities(similarities)
            
            logger.info(f"Created {created} SIMILAR_TO relationships")
            
            # Log statistics
            stats = await self.graph.get_similarity_stats(doc_id)
            logger.info(
                f"Similarity stats: {stats.get('total_sections')} sections, "
                f"avg {stats.get('avg_similarities_per_section', 0):.2f} "
                f"similar sections per section"
            )
            
        except Exception as e:
            logger.exception(f"Error computing section similarities: {e}")
            raise

    async def _build_entity_relationships(self, doc_id: str) -> None:
        """Build PART_OF relationships (component -> system)."""
        logger.info(f"Building entity relationships for document {doc_id}")
        
        query = """
        MATCH (s:Section {document_id: $doc_id})-[:DESCRIBES]->(e:Entity)
        RETURN DISTINCT e.id as entity_id, e.name as name, 
               e.entity_type as entity_type, e.system as system
        """
        
        entities = await self.graph.run_query(query, {"doc_id": doc_id})
        
        systems = [e for e in entities if e["entity_type"] == "System"]
        components = [e for e in entities if e["entity_type"] == "Component"]
        
        # Create PART_OF relationships
        for component in components:
            comp_name = component["name"].lower()
            
            for system in systems:
                system_name = system["system"] or system["name"]
                system_name_lower = system_name.lower()
                
                if system_name_lower in comp_name:
                    try:
                        await self.graph.create_entity_relation(
                            from_entity_id=component["entity_id"],
                            to_entity_id=system["entity_id"],
                            rel_type="PART_OF",
                            properties={"inferred": True, "doc_id": doc_id}
                        )
                        logger.debug(
                            f"Created PART_OF: {component['name']} -> {system['name']}"
                        )
                    except Exception as e:
                        logger.debug(f"Could not create relationship: {e}")
        
        logger.info(f"Entity relationships built for document {doc_id}")

    def _link_yolo_captions_to_schemas(
        self, 
        page: fitz.Page, 
        regions: List
    ) -> None:
        """
        Find YOLO Caption regions (class_id=0) and link them to nearest schema regions.
        Updates schema regions' caption_text field in-place.
        
        :param page: PyMuPDF page object
        :param regions: List of Region objects (modified in-place)
        """
        from services.layout_analyzer import RegionType
        
        # Separate Caption regions (YOLO class 0) and Schema regions
        caption_regions = [r for r in regions if r.yolo_class_id == 0 and r.region_type == RegionType.TEXT]
        schema_regions = [r for r in regions if r.region_type == RegionType.SCHEMA]
        
        if not caption_regions or not schema_regions:
            return
        
        logger.debug(f"Found {len(caption_regions)} YOLO Caption regions and {len(schema_regions)} schema regions")
        
        # For each schema, find the nearest caption (typically above or below)
        for schema in schema_regions:
            if schema.caption_text:  # Already has caption from RegionClassifier
                continue
            
            nearest_caption = None
            min_distance = float('inf')
            
            for caption in caption_regions:
                # Calculate vertical distance (captions usually directly above/below schemas)
                # Prefer captions above schema
                if caption.bbox.y1 <= schema.bbox.y0:  # Caption above schema
                    distance = schema.bbox.y0 - caption.bbox.y1
                elif caption.bbox.y0 >= schema.bbox.y1:  # Caption below schema
                    distance = caption.bbox.y0 - schema.bbox.y1
                else:  # Overlapping (unlikely for caption)
                    distance = 0
                
                # Also check horizontal alignment (caption should be roughly aligned with schema)
                horizontal_overlap = min(schema.bbox.x1, caption.bbox.x1) - max(schema.bbox.x0, caption.bbox.x0)
                if horizontal_overlap < 0:  # No horizontal overlap
                    distance += abs(horizontal_overlap) * 2  # Penalize horizontal misalignment
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_caption = caption
            
            # Link caption to schema if close enough (within 100px vertically)
            if nearest_caption and min_distance < 100:
                # Extract caption text
                caption_text = page.get_text(
                    "text", 
                    clip=fitz.Rect(
                        nearest_caption.bbox.x0, 
                        nearest_caption.bbox.y0, 
                        nearest_caption.bbox.x1, 
                        nearest_caption.bbox.y1
                    )
                ).strip()
                
                if caption_text and len(caption_text) > 3:  # Valid caption
                    schema.caption_text = caption_text
                    logger.info(
                        f"🔗 Linked YOLO Caption to schema: '{caption_text[:60]}' "
                        f"(distance={min_distance:.0f}px)"
                    )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    async def reindex_document(self, doc_id: str) -> None:
        """Reindex a document (delete and recreate all data)."""
        logger.info(f"Reindexing document {doc_id}")
        
        docs = await self.graph.get_all_documents()
        doc_info = next((d for d in docs if d["id"] == doc_id), None)
        
        if not doc_info:
            raise ValueError(f"Document {doc_id} not found")
        
        # Delete from Neo4j
        await self.graph.delete_document(doc_id)
        
        # Delete from Qdrant
        if self.vector:
            self.vector.delete_document_vectors(doc_id)
        
        logger.info(f"Deleted existing data for document {doc_id}")
        logger.info(f"Reindexing completed for document {doc_id}")


    def _sanitize(self, s: str) -> str:
        """Sanitize string for use in filenames and paths."""
        return re.sub(r"[^\w\-.]", "_", s)[:100]
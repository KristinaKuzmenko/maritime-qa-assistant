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
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable
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

        self.region_classifier = RegionClassifier(
            caption_search_distance=250,  # Increase if captions far from figures
            grid_line_threshold=2,          # Min lines for table detection
            drawing_coverage_threshold=0.3,  # Min coverage for schema
            numeric_density_threshold=0.15,  # Min numeric ratio for table
        )
        
        # Get LLM service from schema_extractor for smart detection
        llm_service = getattr(schema_extractor, 'llm_service', None)
        
        self.smart_processor = SmartRegionProcessor(
            table_extractor=table_extractor,
            schema_extractor=schema_extractor,
            enable_llm_detection=True,  # Enable LLM-based type detection
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
        
        # Chunking parameters
        self.chunk_size_tokens = 400
        self.chunk_overlap_tokens = 100
        
        # Section merging threshold
        self.min_section_length = 200  # chars

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

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for change detection."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_toc(self, doc: fitz.Document) -> List[Dict]:
        """Extract table of contents from PDF metadata."""
        toc = doc.get_toc()
        return [{"level": level, "title": title, "page": page} for level, title, page in toc]

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
            default_chapter_id = self._stable_chapter_id(
                doc_id=doc_id,
                title=doc.metadata.get("title", "Document Content"),
                page=0
            )
            chapter_data = {
                "id": default_chapter_id,
                "number": "",
                "title": doc.metadata.get("title") or "Document Content",
                "start_page": 1,
                "level": 1,
            }
            await self.graph.create_chapter(chapter_data, doc_id)
            current_chapter_id = default_chapter_id
            last_chapter_id = default_chapter_id
            self._current_chapter_id = default_chapter_id
            logger.info(f"Created default chapter: {default_chapter_id}")

        # Open pdfplumber for table extraction
        pl_doc = pdfplumber.open(doc.name)

        # Track sections and content found in this chunk
        found_sections_in_chunk = False
        
        # Store schemas/tables per page for deferred processing
        # This ensures sections are created first, then linked to schemas/tables
        page_schemas = {}  # page_num -> list of schema dicts
        page_tables = {}   # page_num -> list of table dicts

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
                    # Reclassify based on content analysis
                    new_type = self.region_classifier.reclassify_region(page, region)
                    
                    # Create new region with reclassified type
                    from services.layout_analyzer import Region
                    reclassified_region = Region(
                        bbox=region.bbox,
                        region_type=new_type,
                        confidence=region.confidence,
                        page_number=page_num,
                    )
                    reclassified_regions.append(reclassified_region)
                else:
                    # Keep TEXT regions as-is
                    reclassified_regions.append(region)
            
            # ===== STEP 3: Smart Region Processing with Fallback =====
            # Process each region with intelligent extraction and fallback logic
            page_schemas_list = []
            page_tables_list = []
            page_text_chunks_list = []  # For LLM-extracted text regions
            
            safe_doc_id = self._sanitize(doc_id)
            
            for region in reclassified_regions:
                if region.region_type == RegionType.TABLE:
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
                    )
                    
                    # Collect based on result type
                    if result['type'] == 'hybrid':
                        # Both table and schema extracted
                        for chunk in result['chunks']:
                            if chunk['type'] == 'table':
                                page_tables_list.append(chunk)
                            elif chunk['type'] == 'schema':
                                page_schemas_list.append(chunk)
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
            toc_entry = next((t for t in toc if t["page"] == page_num + 1), None)
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
                
                # Create new chapter
                chapter_id = self._stable_chapter_id(doc_id, toc_entry["title"], page_num)
                chapter_data = {
                    "id": chapter_id,
                    "number": self._extract_chapter_number(toc_entry["title"]),
                    "title": toc_entry["title"],
                    "start_page": page_num + 1,
                    "level": toc_entry.get("level", 1),
                }
                
                await self.graph.create_chapter(chapter_data, doc_id)
                current_chapter_id = chapter_id
                last_chapter_id = chapter_id
                current_section_accumulator = None
                
                logger.info(f"Created chapter: {toc_entry['title']}")
            
            # ===== STEP 5: Extract Text and Parse Sections =====
        
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line_stripped = line.strip()
                
                if not line_stripped:
                    continue
                
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
                    current_section_accumulator = {
                        "number": section_number,
                        "title": line_stripped,
                        "content_lines": [],
                        "start_page": page_num,
                        "end_page": page_num,
                        "chapter_id": current_chapter_id,
                    }
                    
                    logger.debug(f"Started section: {line_stripped}")
                
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
        
        # Process schemas with page-aware section linking
        schemas_processed = 0
        for page_num, schema_chunks in page_schemas.items():
            # Find section for this page
            section_id = page_to_section.get(page_num)
            
            # If no section on exact page, try to find nearest section
            if not section_id:
                # Find section on previous pages (content continues)
                for p in range(page_num, start_page - 1, -1):
                    if p in page_to_section:
                        section_id = page_to_section[p]
                        break
            
            # If still no section, use last created section as fallback
            if not section_id:
                section_id = getattr(self, '_last_section_id', None)
            
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
        for page_num, table_chunks in page_tables.items():
            # Find section for this page (same logic as schemas)
            section_id = page_to_section.get(page_num)
            
            if not section_id:
                for p in range(page_num, start_page - 1, -1):
                    if p in page_to_section:
                        section_id = page_to_section[p]
                        break
            
            if not section_id:
                section_id = getattr(self, '_last_section_id', None)
            
            # Process each table chunk
            for table_chunk in table_chunks:
                try:
                    await self._process_table_chunk_from_smart_processor(
                        table_chunk=table_chunk,
                        section_id=section_id,
                        doc_id=doc_id,
                        owner=owner,
                    )
                    tables_processed += 1
                except Exception as e:
                    logger.error(
                        f"Failed to process table chunk on page {page_num + 1}: {e}",
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
        }
        
        # Create Schema node in Neo4j
        await self.graph.create_schema(schema_data, section_id=section_id)
        
        logger.info(
            f"Created Schema node: {schema_id} "
            f"(page {schema_chunk['page_number']})"
        )

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
        
        # Check if parent table node already exists (for multi-chunk tables)
        # If this is the first chunk (chunk_index=0), create parent Table node
        if metadata.get("chunk_index", 0) == 0:
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
            }
            
            # Create parent Table node in Neo4j
            await self.graph.create_table(table_data, section_id=section_id)
            
            logger.info(
                f"Created Table node: {table_id} "
                f"(page {table_chunk['page_number']}, "
                f"{metadata['rows']}x{metadata['cols']})"
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
        chunk_id = await self.graph.create_table_chunk(
            parent_table_id=table_id,                      
            chunk_index=metadata.get("chunk_index", 0),
            chunk_text=table_chunk["content"],
            total_chunks=metadata.get("total_chunks", 1),  
            doc_id=doc_id,                                 
            page_number=table_chunk["page_number"],        
        )
        
        logger.debug(
            f"Created TableChunk {metadata.get('chunk_index', 0) + 1}/"
            f"{metadata.get('total_chunks', 1)} "
            f"for table {table_id} ({len(table_chunk['content'])} chars)"
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
                    "number": section_accumulator["number"],
                    "title": section_accumulator["title"],
                    "content": merged_content,
                    "start_page": pending_small_section["start_page"],
                    "end_page": section_accumulator["end_page"],
                    "merged_sections": [
                        pending_small_section.get("number", ""),
                        section_accumulator["number"]
                    ]
                }
            else:
                # Buffer this small section
                return {
                    "number": section_accumulator["number"],
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
        
        if section_accumulator["number"]:
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
                "number": section_accumulator["number"],
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
    
    def _stable_chapter_id(self, doc_id: str, title: str, page: int) -> str:
        """Generate stable chapter ID from components."""
        basis = f"{doc_id}:{title}:{page}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, basis))
    
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
        # Track which page this section is on
        self._page_to_section_map[section_data["start_page"]] = section_id
        
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
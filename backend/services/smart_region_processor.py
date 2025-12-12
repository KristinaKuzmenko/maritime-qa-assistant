"""
Smart region processor with YOLO-guided extraction logic.

YOLO detects region type (TABLE or SCHEMA) with high confidence:
- TABLE regions: pdfplumber → LLM extraction fallback (if needed)
- SCHEMA regions: direct schema extraction (no table attempts)

LLM fallback (TABLE only):
- Triggered when pdfplumber fails to extract table structure
- LLM extracts table as CSV content directly
- No redundant type detection (YOLO confidence is trusted)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging
import base64
import hashlib
import re

import fitz  # PyMuPDF
import pdfplumber

from services.layout_analyzer import RegionType, Region, BBox
from services.table_extractor import TableExtractor
from services.schema_extractor import SchemaExtractor
from core.config import Settings

logger = logging.getLogger(__name__)


class SmartRegionProcessor:
    """
    Process regions with intelligent extraction logic.
    
    YOLO determines region type (TABLE or SCHEMA):
    - TABLE regions: pdfplumber → LLM table extraction (if pdfplumber fails)
    - SCHEMA regions: schema extraction + embedded table check (hybrid support)
    
    LLM fallback (TABLE only, when pdfplumber fails):
    - LLM extracts table as CSV, saves to tables/
    - No type re-detection (YOLO already determined it's a table)
    
    Hybrid extraction (SCHEMA with embedded tables):
    - Common case: P&ID with legend, circuit with specs table
    - Extracts both schema (visual) and table (structured data)
    """
    
    def __init__(
        self,
        table_extractor: TableExtractor,
        schema_extractor: SchemaExtractor,
        region_classifier,
        enable_llm_detection: bool = True,
        vision_detail: str = "high",  # Vision detail for table extraction: low/high/auto
    ) -> None:
        """
        Initialize smart region processor.
        
        :param table_extractor: Table extractor instance
        :param schema_extractor: Schema extractor instance (has LLM service)
        :param region_classifier: Region classifier for LLM verification fallback
        :param enable_llm_detection: Enable LLM content type detection
        :param vision_detail: Vision API detail level for table extraction (high for better accuracy)
        """
        self.table_extractor = table_extractor
        self.schema_extractor = schema_extractor
        self.region_classifier = region_classifier
        
        # Reuse LLM service from schema_extractor
        self.llm_service = getattr(schema_extractor, 'llm_service', None)
        self.enable_llm_detection = enable_llm_detection and self.llm_service is not None
        self.vision_detail = vision_detail
    
    async def process_table_region(
        self,
        fitz_page: fitz.Page,
        pl_page: pdfplumber.page.Page,
        region: Region,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
        full_page_text: str,
        section_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Process TABLE region with fallback.
        
        :param fitz_page: PyMuPDF page object
        :param pl_page: pdfplumber page object
        :param region: Detected region
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :param full_page_text: Full page text for context
        :param section_id: Associated section ID
        :return: Dict with 'type' and 'chunks'
        """
        chunks = []
        
        # Try table extraction within bbox
        table_data = await self._extract_table_from_bbox(
            fitz_page=fitz_page,
            pl_page=pl_page,
            bbox=region.bbox,
            doc_id=doc_id,
            safe_doc_id=safe_doc_id,
            page_num=page_num,
            provided_caption=region.caption_text,  # Pass caption from RegionClassifier or YOLO
        )
        
        if table_data and self._is_valid_table_result(table_data):
            # Success: create table chunks
            logger.debug(
                f"TABLE region on page {page_num + 1}: "
                f"table extraction succeeded"
            )
            chunks.extend(self._create_table_chunks(table_data, section_id))
            return {'type': 'table', 'chunks': chunks}
        else:
            # Fallback: Use LLM to analyze content type and extract appropriately
            logger.info(
                f"TABLE region on page {page_num + 1}: "
                f"pdfplumber failed, using LLM smart detection"
            )
            
            # Render image for LLM analysis
            # Expand bbox to include caption area if no caption was found by RegionClassifier
            expand_bbox = not region.caption_text  # Expand if caption not found
            image_bytes = self._render_region_as_png(
                fitz_page, 
                region.bbox, 
                expand_for_caption=expand_bbox
            )
            
            if self.enable_llm_detection:
                # YOLO already determined it's a TABLE, just extract CSV with LLM
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                logger.info(
                    f"TABLE region on page {page_num + 1}: "
                    f"using LLM to extract table content (YOLO confidence confirmed)"
                    + (f", expanded bbox for caption search" if expand_bbox else ", using provided caption")
                )
                
                table_data = await self._llm_extract_table_direct(
                    image_bytes=image_bytes,
                    image_base64=image_base64,
                    bbox=region.bbox,
                    doc_id=doc_id,
                    safe_doc_id=safe_doc_id,
                    page_num=page_num,
                    provided_caption=region.caption_text,  # Pass caption to LLM extraction
                )
                
                if table_data:
                    table_chunks = self._create_table_chunks(table_data, section_id)
                    chunks.extend(table_chunks)
                    return {'type': 'table', 'chunks': chunks}
                else:
                    # Final fallback: use region_classifier's LLM verification
                    logger.warning(
                        f"⚠️ TABLE region on page {page_num + 1}: "
                        f"LLM table extraction failed, verifying content type with region_classifier"
                    )
                    
                    from services.layout_analyzer import RegionType
                    
                    # Reuse region_classifier's _llm_verify_type (same logic as low YOLO confidence)
                    verified_type = await self.region_classifier._llm_verify_type(
                        page=fitz_page,
                        region=region,
                        page_num=page_num
                    )
                    
                    if verified_type == RegionType.TEXT:
                        # Extract text using LLM
                        logger.info(f"📝 Page {page_num + 1}: Verified as TEXT, extracting")
                        prompt = "Extract ALL visible text from this image. Output only the text."
                        response = await self.llm_service.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}",
                                        "detail": self.vision_detail
                                    }}
                                ]
                            }],
                            max_tokens=1000,
                            temperature=0.1,
                        )
                        text_content = response.choices[0].message.content.strip()
                        if text_content and len(text_content) > 10:
                            return {'type': 'text', 'chunks': [{'content': text_content, 'page': page_num + 1}]}
                    
                    elif verified_type == RegionType.SCHEMA:
                        # Extract as schema
                        logger.info(f"🖼️ Page {page_num + 1}: Verified as SCHEMA, extracting")
                        schema_data = await self._extract_schema_from_region(
                            fitz_page=fitz_page,
                            region=region,
                            doc_id=doc_id,
                            safe_doc_id=safe_doc_id,
                            page_num=page_num,
                            full_page_text=full_page_text,
                            section_id=section_id,
                            schema_idx=0,
                        )
                        if schema_data:
                            chunks.append(self._create_schema_chunk(schema_data, section_id))
                            return {'type': 'schema', 'chunks': chunks}
                    
                    # TABLE or extraction failed
                    return {'type': 'table_failed', 'chunks': []}
            
            # LLM detection disabled - skip region
            logger.warning(
                f"⚠️ TABLE region on page {page_num + 1}: "
                f"pdfplumber failed and LLM detection disabled - skipping region"
            )
            return {'type': 'table_failed', 'chunks': []}
    
    async def process_schema_region(
        self,
        fitz_page: fitz.Page,
        pl_page: pdfplumber.page.Page,
        region: Region,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
        full_page_text: str,
        section_id: Optional[str],
        schema_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Process SCHEMA region (YOLO determined it's a schema).
        Also checks for embedded tables (legends, specs) within schema.
        
        :param fitz_page: PyMuPDF page object
        :param pl_page: pdfplumber page object
        :param region: Detected region
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :param full_page_text: Full page text for context
        :param section_id: Associated section ID
        :param schema_idx: Schema index on page for unique ID generation
        :return: Dict with 'type' ('schema' or 'hybrid') and 'chunks'
        """
        chunks = []
        
        # YOLO already determined it's a SCHEMA - extract it directly
        logger.debug(
            f"SCHEMA region on page {page_num + 1}: "
            f"extracting schema (YOLO confidence confirmed)"
        )
        
        schema_data = await self._extract_schema_from_region(
            fitz_page=fitz_page,
            region=region,
            doc_id=doc_id,
            safe_doc_id=safe_doc_id,
            page_num=page_num,
            full_page_text=full_page_text,
            section_id=section_id,
            schema_idx=schema_idx,
        )
        
        # Check if schema contains embedded table (legend, specs, etc.)
        # Common in technical schematics: P&ID + legend, circuit + specs
        table_data = await self._extract_table_from_bbox(
            fitz_page=fitz_page,
            pl_page=pl_page,
            bbox=region.bbox,
            doc_id=doc_id,
            safe_doc_id=safe_doc_id,
            page_num=page_num,
        )
        
        if table_data and self._is_valid_table_result(table_data):
            # Hybrid: schema with embedded table
            logger.info(
                f"SCHEMA region on page {page_num + 1}: "
                f"contains embedded table (legend/specs), creating hybrid chunks"
            )
            
            # Add table chunks first (usually legend/reference)
            chunks.extend(self._create_table_chunks(table_data, section_id))
            
            # Add schema chunk
            if schema_data:
                chunks.append(self._create_schema_chunk(schema_data, section_id))
            
            return {'type': 'hybrid', 'chunks': chunks}
        else:
            # Pure schema
            if schema_data:
                chunks.append(self._create_schema_chunk(schema_data, section_id))
            
            return {'type': 'schema', 'chunks': chunks}
    
    async def _extract_table_from_bbox(
        self,
        fitz_page: fitz.Page,
        pl_page: pdfplumber.page.Page,
        bbox: BBox,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
        provided_caption: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract table from specific bbox using pdfplumber.
        
        :param fitz_page: PyMuPDF page object
        :param pl_page: pdfplumber page object
        :param bbox: Bounding box to extract from
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :param provided_caption: Caption already found by RegionClassifier or YOLO (optional)
        :return: Table data dict or None
        """
        try:
            # Crop pdfplumber page to bbox
            cropped = pl_page.crop((bbox.x0, bbox.y0, bbox.x1, bbox.y1))
            
            # Try lattice strategy first
            tables = cropped.find_tables(
                table_settings={
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "intersection_tolerance": 5,
                }
            )
            
            # Fallback to stream strategy
            if not tables:
                tables = cropped.find_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "text_tolerance": 3,
                        "intersection_tolerance": 5,
                    }
                )
            
            if not tables or len(tables) == 0:
                return None
            
            # Take first/largest table
            table = tables[0]
            matrix = table.extract()
            
            if not matrix:
                return None
            
            # Trim and validate
            matrix = self.table_extractor._trim_matrix(matrix)
            if not matrix or not self.table_extractor._is_valid_table(matrix):
                return None
            
            # Build table data (simplified - similar to table_extractor)
            n_rows = min(len(matrix), self.table_extractor.max_rows)
            n_cols = min(max(len(r) for r in matrix), self.table_extractor.max_cols)
            n_cells = n_rows * n_cols
            
            if n_cells < self.table_extractor.min_cells:
                return None
            
            # Generate CSV and text chunks
            csv_bytes = self.table_extractor._matrix_to_csv_bytes(matrix[:n_rows], n_cols)
            text_chunks = self.table_extractor._table_to_text_chunks(matrix[:n_rows], n_cols)
            
            logger.debug(
                f"Table extraction: {n_rows}x{n_cols} table, "
                f"generated {len(text_chunks)} text chunks"
            )
            
            # Render image crop
            crop_png = self.table_extractor._crop_fitz_bbox_as_png(
                fitz_page,
                (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                self.table_extractor.zoom,
            )
            
            # Thumbnail
            thumb_png = self.table_extractor._make_thumbnail_png(crop_png, (400, 400))
            
            # Storage paths (include bbox coords for uniqueness on same page)
            base_name = f"page_{page_num + 1}_region_{int(bbox.x0)}_{int(bbox.y0)}_tbl"
            img_rel = f"tables/original/{safe_doc_id}/{base_name}.png"
            csv_rel = f"tables/csv/{safe_doc_id}/{base_name}.csv"
            thumb_rel = f"tables/thumbnail/{safe_doc_id}/{base_name}.png"
            
            # Save files
            img_path = await self.table_extractor.storage.save_file(
                crop_png, img_rel, content_type="image/png"
            )
            await self.table_extractor.storage.save_file(
                csv_bytes, csv_rel, content_type="text/csv"
            )
            thumb_path = await self.table_extractor.storage.save_file(
                thumb_png, thumb_rel, content_type="image/png"
            )
            
            # Generate table ID
            import hashlib
            table_id = hashlib.sha256(
                f"{doc_id}:{page_num}:{bbox.x0}:{bbox.y0}".encode("utf-8")
            ).hexdigest()[:24]
            
            first_chunk = text_chunks[0] if text_chunks else ""
            
            return {
                "id": table_id,
                "doc_id": doc_id,
                "page_number": page_num + 1,
                "title": f"Table - Page {page_num + 1}",
                "caption": provided_caption or "",  # Use provided caption if available
                "rows": n_rows,
                "cols": n_cols,
                "file_path": img_path,
                "thumbnail_path": thumb_path,
                "csv_path": csv_rel,
                "bbox": {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
                "text_preview": first_chunk[:500] if first_chunk else "",
                "text_chunks": text_chunks,
                "normalized_text": first_chunk,
            }
            
        except Exception as e:
            logger.debug(f"Table extraction from bbox failed: {e}")
            return None
    
    async def _extract_schema_from_region(
        self,
        fitz_page: fitz.Page,
        region: Region,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
        full_page_text: str,
        section_id: Optional[str],
        schema_idx: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract schema from region using SchemaExtractor.
        
        :param fitz_page: PyMuPDF page object
        :param region: Detected region
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :param full_page_text: Full page text for context
        :param section_id: Associated section ID
        :param schema_idx: Schema index on page for unique naming
        :return: Schema data dict or None
        """
        try:
            # Use extract_schema_from_region to avoid duplicate analyze_page calls
            schema = await self.schema_extractor.extract_schema_from_region(
                page=fitz_page,
                region=region,
                doc_id=doc_id,
                page_num=page_num,
                idx=schema_idx,
                text_context=full_page_text,
                section_id=section_id,
                doc_id_sanitized=safe_doc_id,
            )
            
            return schema
            
        except Exception as e:
            logger.error(f"Schema extraction from region failed: {e}")
            return None
    
    def _is_valid_table_result(self, table_data: Optional[Dict]) -> bool:
        """
        Validate table extraction result.
        
        :param table_data: Table data dict
        :return: True if valid table
        """
        if not table_data:
            logger.debug("Table validation failed: table_data is None")
            return False
        
        # Check minimum requirements
        if table_data.get('rows', 0) < 2:
            logger.debug(
                f"Table validation failed: rows={table_data.get('rows')} < 2"
            )
            return False
        
        if table_data.get('cols', 0) < 2:
            logger.debug(
                f"Table validation failed: cols={table_data.get('cols')} < 2"
            )
            return False
        
        # Check if has meaningful content
        text_chunks = table_data.get('text_chunks', [])
        if not text_chunks or all(not chunk.strip() for chunk in text_chunks):
            logger.warning(
                f"⚠️ Table validation failed: text_chunks is empty or all blank "
                f"(table: {table_data.get('rows')}x{table_data.get('cols')}, "
                f"chunks: {len(text_chunks)})"
            )
            return False
        
        logger.debug(
            f"✅ Table validation passed: {table_data.get('rows')}x{table_data.get('cols')}, "
            f"{len(text_chunks)} chunks"
        )
        return True
    
    def _create_table_chunks(
        self,
        table_data: Dict[str, Any],
        section_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Create chunk objects for table.
        
        :param table_data: Table data dict
        :param section_id: Associated section ID
        :return: List of chunk dicts
        """
        chunks = []
        text_chunks = table_data.get('text_chunks', [])
        
        if not text_chunks:
            logger.warning(
                f"⚠️ Table {table_data['id']} has NO text chunks! "
                f"Rows: {table_data.get('rows')}, Cols: {table_data.get('cols')}"
            )
        
        for idx, text_chunk in enumerate(text_chunks):
            chunk = {
                "id": f"{table_data['id']}_chunk_{idx}",
                "doc_id": table_data['doc_id'],
                "type": "table",
                "page_number": table_data['page_number'],
                "content": text_chunk,
                "metadata": {
                    "table_id": table_data['id'],
                    "title": table_data['title'],
                    "caption": table_data.get('caption', ''),
                    "file_path": table_data['file_path'],
                    "thumbnail_path": table_data['thumbnail_path'],
                    "csv_path": table_data['csv_path'],
                    "bbox": table_data['bbox'],
                    "rows": table_data['rows'],
                    "cols": table_data['cols'],
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks),
                    "section_id": section_id,
                    "llm_tags": table_data.get('llm_tags', []),  # Pass LLM tags to metadata
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_schema_chunk(
        self,
        schema_data: Dict[str, Any],
        section_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Create chunk object for schema.
        
        :param schema_data: Schema data dict
        :param section_id: Associated section ID
        :return: Chunk dict
        """
        # Schema already has rich context from SchemaExtractor
        chunk = {
            "id": schema_data['id'],
            "doc_id": schema_data['doc_id'],
            "type": "schema",
            "page_number": schema_data['page_number'],
            "content": schema_data.get('text_context', ''),  # Enhanced context (includes LLM summary)
            "metadata": {
                "title": schema_data['title'],
                "caption": schema_data.get('caption', ''),
                "file_path": schema_data['file_path'],
                "thumbnail_path": schema_data['thumbnail_path'],
                "bbox": schema_data['bbox'],
                "confidence": schema_data.get('confidence', 0.0),
                "section_id": section_id or schema_data.get('section_id'),
                "llm_summary": schema_data.get('llm_summary', ''),  # Preserve LLM summary
                "llm_detected": schema_data.get('llm_detected', False),  # Flag if LLM determined type
            }
        }
        
        return chunk
    
    # =========================================================================
    # LLM-BASED SMART CONTENT DETECTION
    # =========================================================================
    
    def _render_region_as_png(
        self,
        page: fitz.Page,
        bbox: BBox,
        zoom: float = 2.0,
        expand_for_caption: bool = False,
    ) -> bytes:
        """
        Render page region as PNG for LLM analysis.
        
        :param page: PyMuPDF page object
        :param bbox: Bounding box to render
        :param zoom: Zoom factor for resolution
        :param expand_for_caption: If True, expand bbox by 60px up/down to include captions
        :return: PNG image bytes
        """
        mat = fitz.Matrix(zoom, zoom)
        
        if expand_for_caption:
            # Expand bbox to include caption area (60px above and below)
            # This allows LLM to see captions that are outside the region bbox
            caption_margin = 60
            clip_rect = fitz.Rect(
                max(0, bbox.x0 - 10),  # Small horizontal margin for clarity
                max(0, bbox.y0 - caption_margin),  # Above region
                min(page.rect.width, bbox.x1 + 10),
                min(page.rect.height, bbox.y1 + caption_margin),  # Below region
            )
        else:
            clip_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        png_bytes = pix.tobytes("png")
        pix = None
        return png_bytes
    
    async def _llm_extract_table_direct(
        self,
        image_bytes: bytes,
        image_base64: str,
        bbox: BBox,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
        provided_caption: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Direct LLM table extraction (no type detection - YOLO already determined it's a table).
        
        :param image_bytes: PNG image bytes
        :param image_base64: Base64 encoded image
        :param bbox: Bounding box
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :param provided_caption: Caption already found by RegionClassifier or YOLO (optional)
        :return: Table data dict or None
        """
        # First, verify this is actually a table (YOLO can make mistakes)
        # If no caption provided, ask LLM to extract it
        if not provided_caption:
            prompt = """Analyze this image and determine its content type, then extract data accordingly.

Step 1: Identify content type (respond with ONE of these):
- TABLE (structured data in rows/columns with clear headers)
- DIAGRAM (technical drawing, schema, flowchart, P&ID)
- TEXT (regular text, logo, header, irrelevant content)

Step 2: If TABLE, extract:
First line: "TABLE"
Second line: Caption text (or "NO_CAPTION" if not visible)
Third line: TAGS: [comma-separated tags describing table type/content]
  Examples: "specifications, motor-engine" or "parts-list, pump" or "parameters, fuel-system"
Remaining lines: CSV data

CSV format rules:
- Use semicolon (;) as delimiter
- First row should be headers if present
- Preserve all text exactly as shown
- Use empty string for empty cells

Example output:
TABLE
Table 5-1: Gear specifications
TAGS: specifications, motor-engine
Model;Type;Size
R1;Helical;Large"""
        else:
            prompt = """Analyze this image and determine if it contains a valid table.

Step 1: Verify content type:
- Is this a TABLE (structured data in rows/columns)?
- Or is it a DIAGRAM, logo, text, or irrelevant content?

Step 2: If TABLE, extract:
First line: "TABLE"
Second line: TAGS: [comma-separated tags describing table type/content]
  Examples: "specifications, motor-engine" or "parts-list, pump" or "parameters"
Remaining lines: CSV data

CSV format rules:
- Use semicolon (;) as delimiter
- First row should be headers if present
- Preserve all text exactly as shown
- Use empty string for empty cells

Output ONLY as specified above."""

        import asyncio
        from openai import RateLimitError
        
        # Retry configuration
        max_retries = 3
        retry_delay = 0.5
        response = None
        
        for attempt in range(max_retries):
            try:
                response = await self.llm_service.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}",
                                        "detail": self.vision_detail  
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.1,
                )
                break  # Success, exit retry loop
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"⏳ Rate limit hit during table extraction, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Rate limit exceeded after {max_retries} retries for table extraction")
                    return None
            except Exception as e:
                logger.error(f"LLM table extraction failed: {e}")
                return None
        
        # Process response (outside try/except, after retry loop)
        if not response:
            logger.error("LLM table extraction failed: no response after retries")
            return None
        
        response_text = response.choices[0].message.content.strip()
        lines = response_text.split('\n')
        
        if not lines:
            logger.warning("LLM returned empty response")
            return None
        
        # Always try to extract table data
        logger.debug(f"LLM response ({len(lines)} lines): First line: '{lines[0][:100] if lines else 'EMPTY'}'")
        
        extracted_caption = provided_caption
        extracted_tags = []
        csv_start_index = 0
        
        # Check first line - could be "TABLE" or caption
        first_line = lines[0].strip().upper() if lines else ""
        if first_line == "TABLE":
            csv_start_index = 1
            logger.debug(f"Detected TABLE marker, CSV starts at line {csv_start_index}")
        
        if not provided_caption and len(lines) > csv_start_index:
            # Check if next line is caption
            next_line = lines[csv_start_index].strip()
            if next_line and next_line != "NO_CAPTION":
                # Check if it looks like a caption (not CSV data)
                if ';' not in next_line or next_line.lower().startswith(('table', 'figure', 'tab')):
                    extracted_caption = next_line
                    logger.info(f"✅ LLM extracted caption: {extracted_caption[:50]}")
                    csv_start_index += 1
        
        # Check for TAGS line
        if len(lines) > csv_start_index:
            tags_line = lines[csv_start_index].strip()
            if tags_line.upper().startswith("TAGS:"):
                # Extract tags
                tags_text = tags_line[5:].strip()  # Remove "TAGS:" prefix
                extracted_tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
                logger.info(f"✅ LLM extracted tags: {extracted_tags}")
                csv_start_index += 1
        
        # Extract CSV content
        csv_lines = lines[csv_start_index:]
        logger.debug(f"CSV extraction: {len(csv_lines)} lines starting from index {csv_start_index}")
        csv_content = self._clean_csv_response('\n'.join(csv_lines))
        
        if not csv_content or len(csv_content) < 5:
            logger.warning(
                f"⚠️ LLM returned empty or invalid CSV data: "
                f"content_length={len(csv_content) if csv_content else 0}, "
                f"first_line='{first_line[:50] if first_line else 'EMPTY'}'"
            )
            return None
        
        # Build table data from CSV
        table_data = await self._build_table_data_from_csv(
            csv_content=csv_content,
            image_bytes=image_bytes,
            bbox=bbox,
            doc_id=doc_id,
            safe_doc_id=safe_doc_id,
            page_num=page_num,
            extracted_caption=extracted_caption,
            extracted_tags=extracted_tags,
        )
        
        return table_data
    
    def _clean_csv_response(self, response: str) -> str:
        """
        Clean LLM CSV response (remove markdown, etc.)
        
        :param response: Raw LLM response
        :return: Clean CSV string
        """
        logger.debug(f"Raw LLM CSV response ({len(response)} chars): {response[:500]}")
        
        # Remove markdown code blocks
        response = re.sub(r'^```(?:csv)?\s*\n?', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n?```\s*$', '', response, flags=re.MULTILINE)
        
        # Remove extra whitespace
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        
        cleaned = '\n'.join(lines)
        logger.debug(f"Cleaned CSV ({len(cleaned)} chars, {len(lines)} lines): {cleaned[:300]}")
        
        return cleaned
    
    async def _build_table_data_from_csv(
        self,
        csv_content: str,
        image_bytes: bytes,
        bbox: BBox,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
        extracted_caption: Optional[str] = None,
        extracted_tags: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build table data dict from CSV content.
        Helper for unified LLM extraction.
        
        :param csv_content: CSV string
        :param image_bytes: PNG image bytes
        :param bbox: Bounding box
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :param extracted_caption: Caption extracted by LLM (optional)
        :return: Table data dict or None
        """
        try:
            # Parse CSV to get dimensions
            lines = [l for l in csv_content.split('\n') if l.strip()]
            n_rows = len(lines)
            n_cols = max(len(l.split(';')) for l in lines) if lines else 0
            
            # Filter out too small tables (minimum 2x2 for valid table structure)
            if n_rows < 2 or n_cols < 2:
                logger.warning(f"❌ CSV too small: {n_rows}x{n_cols} (minimum 2x2 required)")
                return None
            
            # Create text chunks from CSV
            text_chunks = self._csv_to_text_chunks(csv_content)
            
            if not text_chunks or all(not chunk.strip() for chunk in text_chunks):
                logger.warning(f"No valid text chunks from CSV ({n_rows}x{n_cols})")
                return None
            
            # Create thumbnail
            thumb_png = self.table_extractor._make_thumbnail_png(image_bytes, (400, 400))
            
            # Storage paths (include bbox coords for uniqueness on same page)
            base_name = f"page_{page_num + 1}_llm_{int(bbox.x0)}_{int(bbox.y0)}_tbl"
            img_rel = f"tables/original/{safe_doc_id}/{base_name}.png"
            csv_rel = f"tables/csv/{safe_doc_id}/{base_name}.csv"
            thumb_rel = f"tables/thumbnail/{safe_doc_id}/{base_name}.png"
            
            # Save files
            img_path = await self.table_extractor.storage.save_file(
                image_bytes, img_rel, content_type="image/png"
            )
            await self.table_extractor.storage.save_file(
                csv_content.encode('utf-8'), csv_rel, content_type="text/csv"
            )
            thumb_path = await self.table_extractor.storage.save_file(
                thumb_png, thumb_rel, content_type="image/png"
            )
            
            # Generate table ID
            table_id = hashlib.sha256(
                f"{doc_id}:{page_num}:{bbox.x0}:{bbox.y0}:llm".encode("utf-8")
            ).hexdigest()[:24]
            
            logger.info(
                f"✅ LLM table on page {page_num + 1}: "
                f"{n_rows}x{n_cols}, {len(text_chunks)} chunks"
            )
            
            return {
                "id": table_id,
                "doc_id": doc_id,
                "page_number": page_num + 1,
                "title": f"Table - Page {page_num + 1} (LLM)",
                "caption": extracted_caption or "",  # Use extracted caption (from RegionClassifier, YOLO, or LLM)
                "rows": n_rows,
                "cols": n_cols,
                "file_path": img_path,
                "thumbnail_path": thumb_path,
                "csv_path": csv_rel,
                "bbox": {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
                "text_preview": text_chunks[0][:500] if text_chunks else "",
                "text_chunks": text_chunks,
                "normalized_text": text_chunks[0] if text_chunks else "",
                "llm_extracted": True,
                "llm_tags": extracted_tags or [],  # Tags generated by LLM
            }
            
        except Exception as e:
            logger.warning(f"Failed to build table data from CSV: {e}")
            return None
    
    def _clean_csv_response(self, response: str) -> str:
        """
        Clean LLM CSV response (remove markdown, etc.)
        
        :param response: Raw LLM response
        :return: Clean CSV string
        """
        logger.debug(f"Raw LLM CSV response ({len(response)} chars): {response[:500]}")
        
        # Remove markdown code blocks
        response = re.sub(r'^```(?:csv)?\s*\n?', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n?```\s*$', '', response, flags=re.MULTILINE)
        
        # Remove extra whitespace
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        
        cleaned = '\n'.join(lines)
        logger.debug(f"Cleaned CSV ({len(cleaned)} chars, {len(lines)} lines): {cleaned[:300]}")
        
        return cleaned
    
    def _csv_to_text_chunks(
        self,
        csv_content: str,
        chunk_size: int = 500,
    ) -> List[str]:
        """
        Convert CSV content to text chunks for embedding.
        
        :param csv_content: CSV string
        :param chunk_size: Max chunk size in characters
        :return: List of text chunks (always at least 1 chunk)
        """
        if not csv_content or not csv_content.strip():
            logger.warning("⚠️ Empty CSV content provided to _csv_to_text_chunks")
            return ["[Empty table]"]
        
        chunks = []
        lines = [l.strip() for l in csv_content.split('\n') if l.strip()]
        
        if not lines:
            logger.warning("⚠️ No valid lines in CSV content")
            return [csv_content[:chunk_size] if len(csv_content) > chunk_size else csv_content]
        
        # Get header
        header = lines[0]
        header_cols = header.split(';')
        
        # If only header exists, create a chunk with header info
        if len(lines) == 1:
            logger.debug(f"CSV has only header: {header}")
            header_text = " | ".join([f"{col.strip()}" for col in header_cols if col.strip()])
            return [f"Table columns: {header_text}"]
        
        current_chunk = []
        current_size = 0
        
        # Process data rows (skip header at index 0)
        for i in range(1, len(lines)):
            line = lines[i]
            cols = line.split(';')
            
            # Create row text with column names
            row_parts = []
            for j, col in enumerate(cols):
                col_name = header_cols[j].strip() if j < len(header_cols) else f"Col{j+1}"
                if col.strip():
                    row_parts.append(f"{col_name}: {col.strip()}")
            
            if not row_parts:
                continue  # Skip empty rows
            
            row_text = " | ".join(row_parts)
            row_size = len(row_text)
            
            if current_size + row_size > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(row_text)
            current_size += row_size + 1
        
        # Add remaining
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        # Final fallback: if still no chunks, use raw CSV
        if not chunks:
            logger.warning("⚠️ Failed to create structured chunks, using raw CSV")
            return [csv_content[:chunk_size] if len(csv_content) > chunk_size else csv_content]
        
        logger.debug(f"Created {len(chunks)} text chunks from CSV ({len(lines)} lines)")
        return chunks
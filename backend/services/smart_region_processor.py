"""
Smart region processor with fallback extraction logic.
Attempts table extraction first (even for SCHEMA regions), then falls back to schema extraction.
Supports hybrid extraction where both table and schema data are preserved.

LLM-based content type detection for complex regions:
- When pdfplumber fails to extract table structure
- LLM analyzes image and decides: TABLE or SCHEMA
- If TABLE → LLM extracts CSV content
- If SCHEMA → delegates to SchemaExtractor (reuses existing LLM summary logic)
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

logger = logging.getLogger(__name__)


class SmartRegionProcessor:
    """
    Process regions with intelligent fallback logic.
    - TABLE regions: try pdfplumber → fallback to LLM type detection
    - SCHEMA regions: try table extraction → fallback to schema
    - Hybrid: preserve both table and schema if both succeed
    
    LLM Smart Detection (when pdfplumber fails):
    - LLM determines: TABLE or SCHEMA
    - If TABLE: LLM extracts CSV, saves to tables/
    - If SCHEMA: delegates to SchemaExtractor (reuses LLM summary)
    """
    
    def __init__(
        self,
        table_extractor: TableExtractor,
        schema_extractor: SchemaExtractor,
        enable_llm_detection: bool = True,
    ) -> None:
        """
        Initialize smart region processor.
        
        :param table_extractor: Table extractor instance
        :param schema_extractor: Schema extractor instance (has LLM service)
        :param enable_llm_detection: Enable LLM content type detection
        """
        self.table_extractor = table_extractor
        self.schema_extractor = schema_extractor
        
        # Reuse LLM service from schema_extractor
        self.llm_service = getattr(schema_extractor, 'llm_service', None)
        self.enable_llm_detection = enable_llm_detection and self.llm_service is not None
    
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
            image_bytes = self._render_region_as_png(fitz_page, region.bbox)
            
            if self.enable_llm_detection:
                # LLM determines: TABLE, TEXT, or SCHEMA (default)
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                content_type = await self._llm_determine_type(image_base64, "table")
                
                logger.info(
                    f"LLM detection on page {page_num + 1}: "
                    f"YOLO said 'table', LLM says '{content_type}'"
                )
                
                if content_type == "table":
                    # LLM confirms it's a table - extract CSV with retry
                    table_data = await self._llm_extract_table_with_retry(
                        image_bytes=image_bytes,
                        image_base64=image_base64,
                        bbox=region.bbox,
                        doc_id=doc_id,
                        safe_doc_id=safe_doc_id,
                        page_num=page_num,
                        max_retries=2,
                    )
                    
                    if table_data:
                        chunks.extend(self._create_table_chunks(table_data, section_id))
                        return {'type': 'table', 'chunks': chunks}
                
                elif content_type == "text":
                    # LLM says it's text - extract text content
                    text_content = await self._llm_extract_text(
                        image_base64=image_base64,
                        page_num=page_num,
                    )
                    
                    if text_content:
                        text_chunk = self._create_text_chunk(
                            text_content=text_content,
                            doc_id=doc_id,
                            page_num=page_num,
                            bbox=region.bbox,
                            section_id=section_id,
                        )
                        chunks.append(text_chunk)
                        return {'type': 'text', 'chunks': chunks}
            
            # Default fallback: SCHEMA - use SchemaExtractor
            schema_data = await self._extract_schema_from_region(
                fitz_page=fitz_page,
                region=region,
                doc_id=doc_id,
                safe_doc_id=safe_doc_id,
                page_num=page_num,
                full_page_text=full_page_text,
                section_id=section_id,
            )
            
            if schema_data:
                chunks.append(self._create_schema_chunk(schema_data, section_id))
            
            return {'type': 'schema_fallback', 'chunks': chunks}
    
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
    ) -> Dict[str, Any]:
        """
        Process SCHEMA region with table extraction attempt.
        
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
        
        # First try: extract table from schema bbox
        table_data = await self._extract_table_from_bbox(
            fitz_page=fitz_page,
            pl_page=pl_page,
            bbox=region.bbox,
            doc_id=doc_id,
            safe_doc_id=safe_doc_id,
            page_num=page_num,
        )
        
        # Always extract schema (for visual representation)
        schema_data = await self._extract_schema_from_region(
            fitz_page=fitz_page,
            region=region,
            doc_id=doc_id,
            safe_doc_id=safe_doc_id,
            page_num=page_num,
            full_page_text=full_page_text,
            section_id=section_id,
        )
        
        # Decide: hybrid or pure schema
        if table_data and self._is_valid_table_result(table_data):
            # Hybrid: schema contains table
            logger.info(
                f"SCHEMA region on page {page_num + 1}: "
                f"contains table, creating hybrid chunks"
            )
            
            # Create both table and schema chunks
            chunks.extend(self._create_table_chunks(table_data, section_id))
            
            if schema_data:
                chunks.append(self._create_schema_chunk(schema_data, section_id))
            
            return {'type': 'hybrid', 'chunks': chunks}
        else:
            # Pure schema
            logger.debug(
                f"SCHEMA region on page {page_num + 1}: "
                f"pure schema (no table inside)"
            )
            
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
    ) -> Optional[Dict[str, Any]]:
        """
        Extract table from specific bbox using pdfplumber.
        
        :param fitz_page: PyMuPDF page object
        :param pl_page: pdfplumber page object
        :param bbox: Bounding box to extract from
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
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
            
            # Render image crop
            crop_png = self.table_extractor._crop_fitz_bbox_as_png(
                fitz_page,
                (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                self.table_extractor.zoom,
            )
            
            # Thumbnail
            thumb_png = self.table_extractor._make_thumbnail_png(crop_png, (400, 400))
            
            # Storage paths
            base_name = f"page_{page_num + 1}_region_tbl"
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
                "caption": "",
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
            return False
        
        # Check minimum requirements
        if table_data.get('rows', 0) < 2:
            return False
        
        if table_data.get('cols', 0) < 2:
            return False
        
        # Check if has meaningful content
        text_chunks = table_data.get('text_chunks', [])
        if not text_chunks or all(not chunk.strip() for chunk in text_chunks):
            return False
        
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
    ) -> bytes:
        """
        Render page region as PNG for LLM analysis.
        
        :param page: PyMuPDF page object
        :param bbox: Bounding box to render
        :param zoom: Zoom factor for resolution
        :return: PNG image bytes
        """
        mat = fitz.Matrix(zoom, zoom)
        clip_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        png_bytes = pix.tobytes("png")
        pix = None
        return png_bytes
    
    async def _llm_determine_type(
        self,
        image_base64: str,
        yolo_hint: str,
    ) -> str:
        """
        Ask LLM to determine if image is TABLE or SCHEMA.
        
        :param image_base64: Base64 encoded image
        :param yolo_hint: What YOLO detected
        :return: "table", "text", or "schema" (default)
        """
        prompt = """Analyze this image from a technical maritime document.

Determine the content type:
1. TABLE - structured data with rows and columns (grid layout with cells)
2. TEXT - regular paragraphs, lists, or text blocks without visual diagrams
3. SCHEMA - schematic, flowchart, P&I diagram, wiring diagram, equipment layout, or any visual illustration

Respond with ONLY one word: "TABLE", "TEXT", or "SCHEMA"

Guidelines:
- Grid structure with data cells → TABLE
- Paragraphs, bullet points, text content → TEXT  
- Visual connections, flows, components, symbols → SCHEMA
- Technical specifications in tabular form → TABLE
- Equipment layouts, piping systems, circuits → SCHEMA"""

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
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10,
                temperature=0.1,
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            # Parse response - default to schema if unrecognized
            if "TABLE" in result:
                return "table"
            elif "TEXT" in result:
                return "text"
            else:
                # SCHEMA is the default for any other response
                return "schema"
                
        except Exception as e:
            logger.warning(f"LLM type determination failed: {e}")
            # Fallback to schema (safest default)
            return "schema"
    
    async def _llm_extract_table(
        self,
        image_bytes: bytes,
        image_base64: str,
        bbox: BBox,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to extract table content as CSV.
        
        :param image_bytes: PNG image bytes
        :param image_base64: Base64 encoded image
        :param bbox: Bounding box
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :return: Table data dict or None
        """
        prompt = """Extract the table data from this image.

Output the table as CSV format with:
- Use semicolon (;) as delimiter
- First row should be headers if present
- Preserve all text exactly as shown
- Use empty string for empty cells
- Handle merged cells by repeating content

Output ONLY the CSV data, no explanations or markdown code blocks."""

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
                                    "detail": "high"  # High detail for table OCR
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1,
            )
            
            csv_content = response.choices[0].message.content.strip()
            
            # Clean up response (remove markdown if present)
            csv_content = self._clean_csv_response(csv_content)
            
            if not csv_content or len(csv_content) < 5:
                logger.warning("LLM returned empty or invalid CSV")
                return None
            
            # Parse CSV to get dimensions
            lines = [l for l in csv_content.split('\n') if l.strip()]
            n_rows = len(lines)
            n_cols = max(len(l.split(';')) for l in lines) if lines else 0
            
            if n_rows < 2 or n_cols < 2:
                logger.warning(f"LLM CSV too small: {n_rows}x{n_cols}")
                return None
            
            # Create text chunks from CSV
            text_chunks = self._csv_to_text_chunks(csv_content)
            
            # Create thumbnail using table_extractor's method
            thumb_png = self.table_extractor._make_thumbnail_png(image_bytes, (400, 400))
            
            # Storage paths
            base_name = f"page_{page_num + 1}_llm_tbl"
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
                f"LLM extracted table on page {page_num + 1}: "
                f"{n_rows} rows x {n_cols} cols"
            )
            
            return {
                "id": table_id,
                "doc_id": doc_id,
                "page_number": page_num + 1,
                "title": f"Table - Page {page_num + 1} (LLM)",
                "caption": "",
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
            }
            
        except Exception as e:
            logger.warning(f"LLM table extraction failed: {e}")
            return None
    
    async def _llm_extract_table_with_retry(
        self,
        image_bytes: bytes,
        image_base64: str,
        bbox: BBox,
        doc_id: str,
        safe_doc_id: str,
        page_num: int,
        max_retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract table with retry logic for better reliability.
        
        :param image_bytes: PNG image bytes
        :param image_base64: Base64 encoded image
        :param bbox: Bounding box
        :param doc_id: Document ID
        :param safe_doc_id: Sanitized document ID
        :param page_num: Page number (zero-based)
        :param max_retries: Maximum retry attempts
        :return: Table data dict or None
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await self._llm_extract_table(
                    image_bytes=image_bytes,
                    image_base64=image_base64,
                    bbox=bbox,
                    doc_id=doc_id,
                    safe_doc_id=safe_doc_id,
                    page_num=page_num,
                )
                
                if result:
                    return result
                    
                # If result is None but no exception, log and retry
                if attempt < max_retries - 1:
                    logger.info(
                        f"Table extraction attempt {attempt + 1} returned empty, retrying..."
                    )
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.info(
                        f"Table extraction attempt {attempt + 1} failed: {e}, retrying..."
                    )
        
        if last_error:
            logger.warning(f"All {max_retries} table extraction attempts failed: {last_error}")
        else:
            logger.warning(f"All {max_retries} table extraction attempts returned empty")
        
        return None
    
    async def _llm_extract_text(
        self,
        image_base64: str,
        page_num: int,
    ) -> Optional[str]:
        """
        Use LLM to extract text content from image.
        
        :param image_base64: Base64 encoded image
        :param page_num: Page number (zero-based)
        :return: Extracted text or None
        """
        prompt = """Extract all text content from this image.

Output the text exactly as it appears, preserving:
- Paragraph structure
- Bullet points and lists
- Headers and subheaders

Output ONLY the extracted text, no explanations."""

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
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1,
            )
            
            text_content = response.choices[0].message.content.strip()
            
            if not text_content or len(text_content) < 10:
                logger.warning("LLM returned empty or too short text")
                return None
            
            logger.info(
                f"LLM extracted text on page {page_num + 1}: "
                f"{len(text_content)} chars"
            )
            
            return text_content
            
        except Exception as e:
            logger.warning(f"LLM text extraction failed: {e}")
            return None
    
    def _create_text_chunk(
        self,
        text_content: str,
        doc_id: str,
        page_num: int,
        bbox: BBox,
        section_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Create a text chunk from LLM-extracted text.
        
        :param text_content: Extracted text content
        :param doc_id: Document ID
        :param page_num: Page number (zero-based)
        :param bbox: Bounding box
        :param section_id: Associated section ID
        :return: Text chunk dict
        """
        # Generate chunk ID
        chunk_id = hashlib.sha256(
            f"{doc_id}:{page_num}:{bbox.x0}:{bbox.y0}:text".encode("utf-8")
        ).hexdigest()[:24]
        
        return {
            "id": chunk_id,
            "doc_id": doc_id,
            "type": "text",
            "page_number": page_num + 1,
            "content": text_content,
            "metadata": {
                "bbox": {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
                "section_id": section_id,
                "llm_extracted": True,
                "source": "llm_ocr",
            }
        }
    
    def _clean_csv_response(self, response: str) -> str:
        """
        Clean LLM CSV response (remove markdown, etc.)
        
        :param response: Raw LLM response
        :return: Clean CSV string
        """
        # Remove markdown code blocks
        response = re.sub(r'^```(?:csv)?\s*\n?', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n?```\s*$', '', response, flags=re.MULTILINE)
        
        # Remove extra whitespace
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        
        return '\n'.join(lines)
    
    def _csv_to_text_chunks(
        self,
        csv_content: str,
        chunk_size: int = 500,
    ) -> List[str]:
        """
        Convert CSV content to text chunks for embedding.
        
        :param csv_content: CSV string
        :param chunk_size: Max chunk size in characters
        :return: List of text chunks
        """
        chunks = []
        lines = csv_content.split('\n')
        
        # Get header
        header = lines[0] if lines else ""
        header_cols = header.split(';')
        
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            if i == 0:
                continue  # Skip header in iteration, we'll add it to each chunk
            
            cols = line.split(';')
            
            # Create row text with column names
            row_parts = []
            for j, col in enumerate(cols):
                col_name = header_cols[j] if j < len(header_cols) else f"Col{j+1}"
                if col.strip():
                    row_parts.append(f"{col_name}: {col.strip()}")
            
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
        
        # Fallback: if no chunks, use raw CSV
        if not chunks:
            chunks = [csv_content[:chunk_size]]
        
        return chunks
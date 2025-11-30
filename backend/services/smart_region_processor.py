"""
Smart region processor with fallback extraction logic.
Attempts table extraction first (even for SCHEMA regions), then falls back to schema extraction.
Supports hybrid extraction where both table and schema data are preserved.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging

import fitz  # PyMuPDF
import pdfplumber

from services.layout_analyzer import RegionType, Region, BBox
from services.table_extractor import TableExtractor
from services.schema_extractor import SchemaExtractor

logger = logging.getLogger(__name__)


class SmartRegionProcessor:
    """
    Process regions with intelligent fallback logic.
    - TABLE regions: try table extraction → fallback to schema
    - SCHEMA regions: try table extraction → fallback to schema (pure)
    - Hybrid: preserve both table and schema if both succeed
    """
    
    def __init__(
        self,
        table_extractor: TableExtractor,
        schema_extractor: SchemaExtractor,
    ) -> None:
        """
        Initialize smart region processor.
        
        :param table_extractor: Table extractor instance
        :param schema_extractor: Schema extractor instance
        """
        self.table_extractor = table_extractor
        self.schema_extractor = schema_extractor
    
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
            # Fallback: treat as schema
            logger.debug(
                f"TABLE region on page {page_num + 1}: "
                f"table extraction failed, falling back to schema"
            )
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
        :return: Schema data dict or None
        """
        try:
            # Use enhanced context extraction from schema_extractor
            schemas = await self.schema_extractor.extract_from_page(
                page=fitz_page,
                doc_id=doc_id,
                page_num=page_num,
                text_context=full_page_text,
                section_id=section_id,
                doc_id_sanitized=safe_doc_id,
            )
            
            # Return first schema (should be only one since we're processing single region)
            if schemas and len(schemas) > 0:
                return schemas[0]
            
            return None
            
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
            "content": schema_data.get('text_context', ''),  # Enhanced context
            "metadata": {
                "title": schema_data['title'],
                "caption": schema_data.get('caption', ''),
                "file_path": schema_data['file_path'],
                "thumbnail_path": schema_data['thumbnail_path'],
                "bbox": schema_data['bbox'],
                "confidence": schema_data.get('confidence', 0.0),
                "section_id": section_id or schema_data.get('section_id'),
            }
        }
        
        return chunk
"""
Table extractor using pdfplumber.
Creates images  + CSV, and returns normalized text for embeddings.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import io
import csv
import hashlib
import logging

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class TableExtractor:
    def __init__(
        self,
        storage_service,
        zoom: float = 2.0,
        min_cells: int = 4,
        max_rows: int = 500,
        max_cols: int = 50,
        max_tokens_per_chunk: int = 4000,  
    ) -> None:
        self.storage = storage_service
        self.zoom = zoom
        self.min_cells = min_cells
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_tokens_per_chunk = max_tokens_per_chunk

    async def extract_from_page(
        self,
        pl_page: pdfplumber.page.Page,
        fitz_page: fitz.Page,
        doc_id: str,
        page_num: int,  # zero-based
        doc_id_sanitized: Optional[str] = None,
        yolo_table_regions: Optional[List] = None,  # List of Region objects as hints
    ) -> List[Dict[str, Any]]:
        """
        Extract tables with pdfplumber, render crops as PNG, save CSV, and build metadata dicts.
        Uses YOLO table regions as hints for where to look for tables (not as exclusions).
        
        :param pl_page: pdfplumber page object
        :param fitz_page: PyMuPDF page object
        :param doc_id: Document ID
        :param page_num: Page number (zero-based)
        :param doc_id_sanitized: Sanitized document ID
        :param yolo_table_regions: YOLO-detected table regions as hints (not exclusions)
        :return: List of table metadata dicts
        """
        yolo_table_regions = yolo_table_regions or []
        tables: List[Dict[str, Any]] = []
        
        try:
            # Strategy 1: Lattice (visible borders)
            found_lattice = pl_page.find_tables(
                table_settings={
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "intersection_tolerance": 5,
                }
            )

            # Strategy 2: Stream (text-based)
            found_stream = pl_page.find_tables(
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "text_tolerance": 3,
                    "intersection_tolerance": 5,
                }
            )

            # Combine and deduplicate
            all_found = (found_lattice or []) + (found_stream or [])
            found = self._deduplicate_tables(all_found)
            
            # If YOLO provided hints, try to extract from those regions too
            if yolo_table_regions:
                logger.debug(
                    f"Page {page_num + 1}: Using {len(yolo_table_regions)} YOLO table hints"
                )
                # YOLO hints are processed separately in SmartRegionProcessor
                # This method still does full-page table extraction

            safe_doc_id = (doc_id_sanitized or self._sanitize(doc_id))
            idx = 0

            for t in found:
                if not t.bbox:
                    continue

                # Extract cells as matrix
                matrix = t.extract()
                if not matrix:
                    continue

                # Trim empty trailing columns/rows
                matrix = self._trim_matrix(matrix)
                if not matrix:
                    continue

                # Validate table structure
                if not self._is_valid_table(matrix):
                    logger.debug(f"Skipping invalid table structure")
                    continue

                n_rows = min(len(matrix), self.max_rows)
                n_cols = min(max(len(r) for r in matrix), self.max_cols)
                n_cells = n_rows * n_cols
                if n_cells < self.min_cells:
                    continue

                # Build CSV bytes with UTF-8 BOM for Excel compatibility
                csv_bytes = self._matrix_to_csv_bytes(matrix[:n_rows], n_cols)

                # Render cropped image via fitz for high quality
                try:
                    crop_png = self._crop_fitz_bbox_as_png(fitz_page, t.bbox, self.zoom)
                except Exception as e:
                    logger.error(f"Failed to render table crop: {e}")
                    continue

                # Normalized text for embeddings - WITH CHUNKING for large tables
                text_chunks = self._table_to_text_chunks(matrix[:n_rows], n_cols)

                # Storage paths
                base_name = f"page_{page_num+1}_tbl_{idx}"
                img_rel = f"tables/original/{safe_doc_id}/{base_name}.png"
                csv_rel = f"tables/csv/{safe_doc_id}/{base_name}.csv"
                thumb_rel = f"tables/thumbnail/{safe_doc_id}/{base_name}.png"

                img_path = await self.storage.save_file(crop_png, img_rel, content_type="image/png")
                await self.storage.save_file(csv_bytes, csv_rel, content_type="text/csv")

                # Thumbnail
                thumb_png = self._make_thumbnail_png(crop_png, (400, 400))
                thumb_path = await self.storage.save_file(thumb_png, thumb_rel, content_type="image/png")

                # Build Table node payload
                table_id = self._stable_table_id(doc_id, page_num, t.bbox, idx)

                # Use first chunk for preview, store all chunks
                first_chunk = text_chunks[0] if text_chunks else ""

                tables.append({
                    "id": table_id,
                    "doc_id": doc_id,
                    "page_number": page_num + 1,
                    "title": f"Table - Page {page_num+1}.{idx}",
                    "caption": "",
                    "rows": n_rows,
                    "cols": n_cols,
                    "file_path": img_path,
                    "thumbnail_path": thumb_path,
                    "csv_path": csv_rel,
                    "bbox": {"x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2], "y1": t.bbox[3]},
                    "text_preview": self._safe_truncate(first_chunk, 500),
                    "text_chunks": text_chunks,  # List of embedding-safe chunks
                    "normalized_text": first_chunk,  # Keep for backward compatibility
                })
                idx += 1

        except Exception as e:
            logger.error(f"pdfplumber table extraction failed on page {page_num+1}: {e}", exc_info=True)

        return tables

    # ----------------------------- helpers -----------------------------

    def _sanitize(self, s: str) -> str:
        """Sanitize string for use in filenames."""
        import re
        return re.sub(r"[^\w\-.]", "_", s)[:100]

    def _trim_matrix(self, matrix: List[List[str]]) -> List[List[str]]:
        """Remove empty rows and columns from table matrix."""
        # Drop fully empty rows
        m = [list(r or []) for r in matrix if any((c or "").strip() for c in r or [])]
        if not m:
            return m

        # Normalize row lengths
        max_len = max(len(r) for r in m)
        for r in m:
            if len(r) < max_len:
                r += [""] * (max_len - len(r))

        # Drop fully empty trailing columns
        drop_tail = 0
        for col in range(max_len - 1, -1, -1):
            if all((r[col] or "").strip() == "" for r in m):
                drop_tail += 1
            else:
                break
        if drop_tail:
            m = [r[:max_len - drop_tail] for r in m]
        return m

    def _matrix_to_csv_bytes(self, matrix: List[List[str]], n_cols: int) -> bytes:
        """Convert table matrix to CSV bytes with UTF-8 BOM for Excel."""
        buf = io.StringIO()
        w = csv.writer(buf)
        for r in matrix:
            row = (r + [""] * max(0, n_cols - len(r)))[:n_cols]
            w.writerow([self._clean_cell(c) for c in row])
        # Add UTF-8 BOM for Excel compatibility
        return b'\xef\xbb\xbf' + buf.getvalue().encode("utf-8")

    def _clean_cell(self, s: str) -> str:
        """Clean cell content for CSV/text output."""
        s = (s or "").replace("\r", " ").replace("\n", " ")
        return " ".join(s.split())

    def _table_to_text_chunks(self, matrix: List[List[str]], n_cols: int) -> List[str]:
        """
        Convert table to text chunks that fit within token limits.
        Large tables are split into row groups to avoid exceeding embedding limits.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0

        for r in matrix:
            row = (r + [""] * max(0, n_cols - len(r)))[:n_cols]
            row_text = " | ".join(self._clean_cell(c) for c in row)
            # Rough token estimate: 1 token ≈ 4 characters
            row_tokens = len(row_text) // 4

            # If single row exceeds limit, truncate it (rare edge case)
            if row_tokens > self.max_tokens_per_chunk:
                logger.warning(f"Single row exceeds token limit ({row_tokens} tokens), truncating")
                row_text = row_text[:self.max_tokens_per_chunk * 4]
                row_tokens = self.max_tokens_per_chunk

            # Check if adding this row would exceed limit
            if current_tokens + row_tokens > self.max_tokens_per_chunk and current_chunk:
                # Save current chunk and start new one
                chunks.append("\n".join(current_chunk))
                current_chunk = [row_text]
                current_tokens = row_tokens
            else:
                current_chunk.append(row_text)
                current_tokens += row_tokens

        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else [""]

    def _safe_truncate(self, text: str, max_chars: int) -> str:
        """Safely truncate text without breaking words."""
        if len(text) <= max_chars:
            return text
        # Truncate at word boundary
        truncated = text[:max_chars].rsplit(' ', 1)[0]
        return truncated + "..."

    def _stable_table_id(
        self,
        doc_id: str,
        page_num: int,
        bbox: Tuple[float, float, float, float],
        idx: int
    ) -> str:
        """Generate stable hash-based ID for table."""
        basis = f"{doc_id}:{page_num}:{bbox}:{idx}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]

    def _crop_fitz_bbox_as_png(
        self,
        fitz_page: fitz.Page,
        bbox: Tuple[float, float, float, float],
        zoom: float
    ) -> bytes:
        """
        Render a cropped region of PDF page as high-quality PNG.

        :param fitz_page: PyMuPDF page object
        :param bbox: Bounding box in PDF points (x0, y0, x1, y1)
        :param zoom: Zoom factor for rendering resolution
        :return: PNG image bytes
        """
        # Create transformation matrix for zoom
        mat = fitz.Matrix(zoom, zoom)

        # Render the specific bbox region at high resolution
        # clip parameter restricts rendering to bbox area
        clip_rect = fitz.Rect(bbox)
        pix = fitz_page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)

        # Convert pixmap to PNG bytes
        png_bytes = pix.pil_tobytes(format="PNG")

        # Clean up
        pix = None

        return png_bytes

    def _make_thumbnail_png(self, png_bytes: bytes, size: Tuple[int, int]) -> bytes:
        """Create thumbnail from PNG bytes."""
        img = Image.open(io.BytesIO(png_bytes))
        # Use contain to preserve aspect ratio
        img = ImageOps.contain(img, size)
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()

    def _deduplicate_tables(
        self,
        tables: List
    ) -> List:
        """Remove duplicate table detections."""
        if not tables:
            return []

        unique = []

        for t in tables:
            # Check if significantly overlaps with existing
            is_duplicate = False

            for existing in unique:
                iou = self._calculate_iou(t.bbox, existing.bbox)
                if iou > 0.7:  # 70% overlap = duplicate
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(t)

        return unique

    def _calculate_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Calculate Intersection over Union for bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _is_valid_table(self, matrix: List[List[str]]) -> bool:
        """
        Validate that matrix represents a real table.
        Filters out false positives from aligned text and diagrams.
        Balanced for maritime technical tables with some merged cells.
        """
        if not matrix or len(matrix) < 2:
            return False  # Need at least 2 rows

        # Check 1: Column count consistency (moderate tolerance for merged cells)
        col_counts = [len(row) for row in matrix]
        max_cols = max(col_counts)
        min_cols = min(col_counts)
        
        # Allow up to 40% variation for merged cells
        if max_cols > 0 and (max_cols - min_cols) / max_cols > 0.4:
            return False  # Too irregular

        # Check 2: Must have at least 2 columns on average
        avg_cols = sum(col_counts) / len(col_counts)
        if avg_cols < 2:
            return False  # Single column - probably not a table

        # Check 3: Cell content - must have reasonable content
        all_cells = [cell for row in matrix for cell in row]
        non_empty = [c for c in all_cells if (c or "").strip()]

        if len(non_empty) < len(all_cells) * 0.25:
            return False  # Too sparse (less than 25%)

        # Check 4: Cell length - avoid paragraphs but allow descriptions
        if non_empty:
            avg_cell_length = sum(len(c or "") for c in non_empty) / len(non_empty)
            if avg_cell_length > 150:
                return False  # Cells too long - probably paragraphs

        # Check 5: Must have structured data pattern (numbers, short strings)
        # Schemas often have very few actual text cells
        if len(non_empty) < 4:
            return False  # Too few cells with content

        return True
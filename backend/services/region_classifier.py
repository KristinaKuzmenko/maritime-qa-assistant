"""
Region classifier for reclassifying YOLO detections based on content analysis.
Analyzes bbox content (text structure, graphics, captions) to determine if region
is actually a TABLE or SCHEMA, overriding potentially incorrect YOLO predictions.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

import fitz  # PyMuPDF

from services.layout_analyzer import RegionType, Region, BBox

logger = logging.getLogger(__name__)


class RegionClassifier:
    """
    Reclassifies YOLO-detected regions based on content analysis.
    Improves accuracy by analyzing text structure, graphics, and captions.
    """
    
    # Caption patterns for tables
    TABLE_CAPTION_PATTERNS = [
        re.compile(r'(?i)\btable\s+([0-9]+(?:[.-][0-9]+)*)', re.MULTILINE),
        re.compile(r'(?i)\btab\.?\s+([0-9]+(?:[.-][0-9]+)*)', re.MULTILINE),
    ]
    
    # Caption patterns for figures/schemas
    FIGURE_CAPTION_PATTERNS = [
        re.compile(r'(?i)\b(?:figure|fig\.?|diagram|drawing|dwg\.?|schema|schematic)\s+([0-9]+(?:[.-][0-9]+)*)', re.MULTILINE),
    ]
    
    def __init__(
        self,
        caption_search_distance: int = 250,
        grid_line_threshold: int = 2,
        drawing_coverage_threshold: float = 0.3,
        numeric_density_threshold: float = 0.15,
        min_table_rows: int = 3,
        min_table_cols: int = 2,
    ) -> None:
        """
        Initialize region classifier.
        
        :param caption_search_distance: Distance in pixels to search for captions
        :param grid_line_threshold: Minimum grid lines to consider table-like
        :param drawing_coverage_threshold: Minimum drawing coverage ratio for schema
        :param numeric_density_threshold: Minimum numeric content ratio for table
        :param min_table_rows: Minimum rows to detect table structure
        :param min_table_cols: Minimum columns to detect table structure
        """
        self.caption_search_distance = caption_search_distance
        self.grid_line_threshold = grid_line_threshold
        self.drawing_coverage_threshold = drawing_coverage_threshold
        self.numeric_density_threshold = numeric_density_threshold
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
    
    def reclassify_region(
        self,
        page: fitz.Page,
        region: Region,
    ) -> RegionType:
        """
        Reclassify region based on content analysis.
        
        :param page: PyMuPDF page object
        :param region: Detected region from YOLO
        :return: Reclassified region type (TABLE or SCHEMA)
        """
        # Only reclassify TABLE and SCHEMA regions
        if region.region_type not in [RegionType.TABLE, RegionType.SCHEMA]:
            return region.region_type
        
        # Analyze content
        content_analysis = self._analyze_bbox_content(page, region.bbox)
        
        # Detect caption type
        caption_type = self._detect_caption_type(page, region.bbox)
        
        # Scoring
        table_score = 0.0
        schema_score = 0.0
        
        # A. Table structure detection (STRONG signal - aligned rows/columns of text)
        if content_analysis['has_table_structure']:
            table_score += 0.6  # Strong signal: text arranged in grid
        
        # B. Grid lines detection (tables often have grid lines)
        if content_analysis['has_grid_lines']:
            table_score += 0.5  # Increased from 0.4
            # Grid lines WITH text structure = almost certainly a table
            if content_analysis['has_table_structure']:
                table_score += 0.2
        
        # C. Text content analysis
        # Tables have significant text, schemas are mostly graphics
        if content_analysis['word_count'] > 10:
            table_score += 0.3  # Increased from 0.2
        elif content_analysis['word_count'] > 5:
            table_score += 0.15
        
        # D. Numeric density (tables often have many numbers)
        if content_analysis['numeric_density'] > self.numeric_density_threshold:
            table_score += 0.2
        
        # E. Vector shapes - only count as schema if NO table structure AND no grid lines
        # Tables with borders also have vector shapes!
        if content_analysis['has_many_vector_shapes']:
            if not content_analysis['has_table_structure'] and not content_analysis['has_grid_lines']:
                schema_score += 0.3  # Reduced from 0.5
            else:
                # Vector shapes WITH table structure/grid = table with borders
                table_score += 0.2  # Increased from 0.1
        
        # F. Complex paths (curved lines, shapes) = schema
        # But only if there's no grid structure (tables can have decorative elements)
        if content_analysis.get('has_complex_paths', False):
            if not content_analysis['has_grid_lines']:
                schema_score += 0.4
            # If grid lines present, complex paths are less indicative
        
        # G. Text-to-graphics ratio
        # Low ratio (little text, many graphics) = likely schema, but only if no table structure
        if content_analysis['text_to_graphics_ratio'] < 0.3:
            if not content_analysis['has_table_structure'] and not content_analysis['has_grid_lines']:
                schema_score += 0.2  # Reduced from 0.3
        elif content_analysis['text_to_graphics_ratio'] > 1.0:
            table_score += 0.1
        
        # H. Caption-based scoring (VERY strong signal)
        if caption_type == 'table':
            table_score += 0.7
        elif caption_type == 'figure':
            schema_score += 0.7
        
        # I. YOLO confidence - trust YOLO for both TABLE and SCHEMA
        # YOLO is specifically trained to detect these, trust it
        if region.region_type == RegionType.TABLE:
            # YOLO TABLE is a strong signal
            table_score += 0.5 * region.confidence
            if region.confidence > 0.7:
                table_score += 0.3
            elif region.confidence > 0.5:
                table_score += 0.15
        elif region.region_type == RegionType.SCHEMA:
            # YOLO SCHEMA (Picture) is also a strong signal - diagrams, schematics, figures
            schema_score += 0.5 * region.confidence
            if region.confidence > 0.7:
                schema_score += 0.3
            elif region.confidence > 0.5:
                schema_score += 0.15
        
        # J. Final adjustment: require strong evidence to override YOLO
        # This prevents false reclassifications
        if region.region_type == RegionType.TABLE:
            if schema_score > table_score and schema_score < table_score + 0.4:
                logger.debug(
                    f"Keeping YOLO TABLE classification despite schema_score={schema_score:.2f} > table_score={table_score:.2f} "
                    f"(margin too small)"
                )
                table_score = schema_score + 0.01
        elif region.region_type == RegionType.SCHEMA:
            if table_score > schema_score and table_score < schema_score + 0.4:
                logger.debug(
                    f"Keeping YOLO SCHEMA classification despite table_score={table_score:.2f} > schema_score={schema_score:.2f} "
                    f"(margin too small)"
                )
                schema_score = table_score + 0.01
        
        # Decision
        reclassified_type = RegionType.TABLE if table_score > schema_score else RegionType.SCHEMA
        
        # Log reclassification with detailed scores
        if reclassified_type != region.region_type:
            logger.info(
                f"Reclassified region on page {region.page_number + 1}: "
                f"{region.region_type.value} → {reclassified_type.value} "
                f"(table_score={table_score:.2f}, schema_score={schema_score:.2f}, "
                f"YOLO_conf={region.confidence:.2f}, "
                f"table_struct={content_analysis['has_table_structure']}, "
                f"grid={content_analysis['has_grid_lines']}, "
                f"words={content_analysis['word_count']}, "
                f"complex_paths={content_analysis.get('has_complex_paths', False)})"
            )
        else:
            logger.debug(
                f"Kept region on page {region.page_number + 1} as {region.region_type.value} "
                f"(table_score={table_score:.2f}, schema_score={schema_score:.2f})"
            )
        
        return reclassified_type
    
    def _analyze_bbox_content(
        self,
        page: fitz.Page,
        bbox: BBox,
    ) -> Dict[str, Any]:
        """
        Analyze bbox content for classification features.
        
        :param page: PyMuPDF page object
        :param bbox: Bounding box to analyze
        :return: Analysis results dict
        """
        clip_rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        
        # Extract text
        text = page.get_text("text", clip=clip_rect)
        words = page.get_text("words", clip=clip_rect)
        
        # Extract drawings (vector graphics)
        drawings = page.get_drawings()
        bbox_drawings = [
            d for d in drawings
            if self._is_drawing_in_bbox(d, bbox)
        ]
        
        # Initialize metrics
        analysis = {
            'text_length': len(text),
            'word_count': len(words),
            'drawing_count': len(bbox_drawings),
            'has_table_structure': False,
            'has_grid_lines': False,
            'has_many_vector_shapes': False,
            'has_complex_paths': False,
            'text_to_graphics_ratio': 0.0,
            'numeric_density': 0.0,
        }
        
        # A. Table structure detection
        if len(words) > 0:
            analysis['has_table_structure'] = self._detect_table_structure(words, bbox)
        
        # B. Grid lines detection (horizontal and vertical straight lines)
        horizontal_lines = [d for d in bbox_drawings if self._is_horizontal_line(d)]
        vertical_lines = [d for d in bbox_drawings if self._is_vertical_line(d)]
        
        logger.debug(
            f"Grid analysis: {len(horizontal_lines)} horiz, {len(vertical_lines)} vert lines, "
            f"threshold={self.grid_line_threshold}"
        )
        
        # Grid detection: either both directions OR many lines in one direction
        if len(horizontal_lines) >= self.grid_line_threshold and len(vertical_lines) >= self.grid_line_threshold:
            analysis['has_grid_lines'] = True
        elif len(horizontal_lines) >= 4 or len(vertical_lines) >= 4:
            # Many lines in one direction also indicates table structure
            analysis['has_grid_lines'] = True
        
        # B2. Rectangle detection (table cells are often rectangles)
        rectangles = [d for d in bbox_drawings if self._is_rectangle(d)]
        if len(rectangles) >= 3:
            analysis['has_grid_lines'] = True  # Rectangles also indicate table structure
            logger.debug(f"Detected {len(rectangles)} rectangles → has_grid_lines=True")
        
        # C. Complex paths detection (curves, polygons - typical for diagrams)
        # BUT: if we already detected grid lines, be more strict about complex paths
        complex_paths = [d for d in bbox_drawings if self._is_complex_path(d)]
        
        # Only count complex paths if there's no strong table evidence
        if analysis['has_grid_lines'] or analysis['has_table_structure']:
            # For regions with table features, need MORE complex paths to be classified as schema
            if len(complex_paths) >= 6:  # Higher threshold when table features present
                analysis['has_complex_paths'] = True
                logger.debug(f"Complex paths ({len(complex_paths)}) detected despite grid lines")
        else:
            if len(complex_paths) >= 3:
                analysis['has_complex_paths'] = True
        
        logger.debug(
            f"Content analysis: table_struct={analysis['has_table_structure']}, "
            f"grid_lines={analysis['has_grid_lines']}, complex_paths={analysis['has_complex_paths']}"
        )
        
        # C. Vector shapes density
        bbox_area = bbox.area()
        if bbox_area > 0:
            drawing_area = sum(self._estimate_drawing_area(d) for d in bbox_drawings)
            drawing_coverage = drawing_area / bbox_area
            
            if drawing_coverage > self.drawing_coverage_threshold:
                analysis['has_many_vector_shapes'] = True
        
        # D. Text to graphics ratio
        if len(bbox_drawings) > 0:
            analysis['text_to_graphics_ratio'] = len(text) / len(bbox_drawings)
        else:
            analysis['text_to_graphics_ratio'] = float('inf')
        
        # E. Numeric density (tables often have many numbers)
        if len(text) > 0:
            numeric_chars = sum(c.isdigit() for c in text)
            analysis['numeric_density'] = numeric_chars / len(text)
        
        return analysis
    
    def _detect_table_structure(
        self,
        words: List[Tuple],
        bbox: BBox,
    ) -> bool:
        """
        Detect table-like structure by analyzing word alignment.
        
        :param words: List of word tuples (x0, y0, x1, y1, text, ...)
        :param bbox: Bounding box
        :return: True if table structure detected
        """
        if len(words) < 4:  # Reduced minimum - small tables exist
            return False
        
        # Group words by Y-coordinate (rows)
        rows = {}
        tolerance = 15  # Increased from 5 - words in same row may have slight Y offset
        
        for word in words:
            x0, y0, x1, y1, text, *_ = word
            
            # Find row with similar Y-coordinate
            found_row = False
            for row_y in rows.keys():
                if abs(y0 - row_y) < tolerance:
                    rows[row_y].append(word)
                    found_row = True
                    break
            
            if not found_row:
                rows[y0] = [word]
        
        # Table needs at least 2 rows (small tables exist)
        if len(rows) < 2:
            return False
        
        # Check alignment by columns
        row_word_counts = [len(words) for words in rows.values()]
        avg_words_per_row = sum(row_word_counts) / len(row_word_counts)
        
        # More lenient check: just need multiple words per row on average
        # Tables can have merged cells with varying word counts
        if avg_words_per_row >= 1.5:  # Reduced from 2
            # Check X-alignment (columns)
            x_positions = []
            for row_words in rows.values():
                for word in row_words:
                    x_positions.append(word[0])  # x0
            
            if not x_positions:
                return False
                
            # Cluster X positions (columns)
            x_positions.sort()
            columns = []
            current_col = [x_positions[0]]
            
            for x in x_positions[1:]:
                if x - current_col[-1] < 40:  # Increased from 20 - columns can be wider
                    current_col.append(x)
                else:
                    columns.append(current_col)
                    current_col = [x]
            columns.append(current_col)
            
            # Table needs at least 2 columns
            if len(columns) >= 2:
                return True
            
            # Alternative: many rows even with 1 column might indicate a list/table
            if len(rows) >= 4:
                return True
        
        return False
    
    def _detect_caption_type(
        self,
        page: fitz.Page,
        bbox: BBox,
    ) -> Optional[str]:
        """
        Detect caption type: 'table', 'figure', or None.
        
        :param page: PyMuPDF page object
        :param bbox: Bounding box
        :return: Caption type or None
        """
        # Search regions above and below bbox
        search_regions = [
            fitz.Rect(
                max(0, bbox.x0 - 50),
                max(0, bbox.y0 - self.caption_search_distance),
                min(page.rect.width, bbox.x1 + 50),
                bbox.y0,
            ),
            fitz.Rect(
                max(0, bbox.x0 - 50),
                bbox.y1,
                min(page.rect.width, bbox.x1 + 50),
                min(page.rect.height, bbox.y1 + self.caption_search_distance),
            ),
        ]
        
        for search_rect in search_regions:
            text = page.get_text("text", clip=search_rect)
            
            if not text:
                continue
            
            # Check table patterns first (higher priority)
            for pattern in self.TABLE_CAPTION_PATTERNS:
                if pattern.search(text):
                    return 'table'
            
            # Check figure patterns
            for pattern in self.FIGURE_CAPTION_PATTERNS:
                if pattern.search(text):
                    return 'figure'
        
        return None
    
    def _is_drawing_in_bbox(self, drawing: Dict, bbox: BBox) -> bool:
        """Check if drawing is inside bbox"""
        try:
            # Drawing has 'rect' key with coordinates
            rect = drawing.get('rect')
            if not rect:
                return False
            
            # Check if drawing center is inside bbox
            center_x = (rect.x0 + rect.x1) / 2
            center_y = (rect.y0 + rect.y1) / 2
            
            return (bbox.x0 <= center_x <= bbox.x1 and
                    bbox.y0 <= center_y <= bbox.y1)
        except (KeyError, AttributeError, TypeError):
            return False
    
    def _is_horizontal_line(self, drawing: Dict) -> bool:
        """Check if drawing is a horizontal line"""
        try:
            rect = drawing.get('rect')
            if not rect:
                return False
            
            height = abs(rect.y1 - rect.y0)
            width = abs(rect.x1 - rect.x0)
            
            # Horizontal line: width >> height
            # Relaxed: height < 10 (was 5), width > 30 for meaningful lines
            return width > 30 and height < 10 and (width / max(height, 0.1)) > 5
        except (KeyError, AttributeError, TypeError):
            return False
    
    def _is_vertical_line(self, drawing: Dict) -> bool:
        """Check if drawing is a vertical line"""
        try:
            rect = drawing.get('rect')
            if not rect:
                return False
            
            height = abs(rect.y1 - rect.y0)
            width = abs(rect.x1 - rect.x0)
            
            # Vertical line: height >> width
            # Relaxed: width < 10 (was 5), height > 30 for meaningful lines
            return height > 30 and width < 10 and (height / max(width, 0.1)) > 5
        except (KeyError, AttributeError, TypeError):
            return False
    
    def _is_complex_path(self, drawing: Dict) -> bool:
        """
        Check if drawing is a complex path (curves, polygons).
        Complex paths indicate diagrams/schemas, not tables.
        
        :param drawing: Drawing dict from PyMuPDF
        :return: True if complex path
        """
        try:
            # Check for curved paths or many points
            items = drawing.get('items', [])
            
            for item in items:
                # Bezier curves (c = curve)
                if item[0] == 'c':
                    return True
                # Quadratic curves (qu)
                if item[0] == 'qu':
                    return True
            
            # Check number of path elements - many elements = complex shape
            if len(items) > 6:
                # But filter out simple rectangles (4 lines)
                line_count = sum(1 for item in items if item[0] == 'l')
                if line_count != len(items):  # Has non-line elements
                    return True
            
            # Check fill - filled shapes are often diagram elements
            if drawing.get('fill') is not None:
                fill = drawing.get('fill')
                # Non-white, non-black fill = probably diagram
                if fill and fill != (0, 0, 0) and fill != (1, 1, 1):
                    return True
            
            return False
        except (KeyError, AttributeError, TypeError):
            return False
    
    def _is_rectangle(self, drawing: Dict) -> bool:
        """
        Check if drawing is a rectangle (typical table cell border).
        
        :param drawing: Drawing dict from PyMuPDF
        :return: True if rectangle
        """
        try:
            items = drawing.get('items', [])
            
            # Rectangle: 4 lines forming a closed shape
            # Or a 're' (rectangle) command
            for item in items:
                if item[0] == 're':  # Rectangle command
                    return True
            
            # Check for 4 lines (l commands) forming a rectangle
            line_count = sum(1 for item in items if item[0] == 'l')
            if line_count == 4:
                # Check if it's a closed path
                if drawing.get('closePath', False) or len(items) == 5:  # 4 lines + close
                    return True
            
            # Check rect dimensions - if it's a reasonable rectangle shape
            rect = drawing.get('rect')
            if rect:
                width = abs(rect.x1 - rect.x0)
                height = abs(rect.y1 - rect.y0)
                # Not a line (both dimensions significant) and rectangular-ish
                if width > 20 and height > 10:
                    aspect_ratio = max(width, height) / max(min(width, height), 0.1)
                    # Aspect ratio not too extreme (not a line)
                    if aspect_ratio < 20:
                        # Check if it's just lines (stroke only, no complex paths)
                        has_stroke = drawing.get('stroke') is not None
                        has_no_curves = not any(item[0] in ['c', 'qu'] for item in items)
                        if has_stroke and has_no_curves:
                            return True
            
            return False
        except (KeyError, AttributeError, TypeError):
            return False
    
    def _estimate_drawing_area(self, drawing: Dict) -> float:
        """Estimate area covered by drawing"""
        try:
            rect = drawing.get('rect')
            if not rect:
                return 0.0
            
            return abs(rect.x1 - rect.x0) * abs(rect.y1 - rect.y0)
        except (KeyError, AttributeError, TypeError):
            return 0.0






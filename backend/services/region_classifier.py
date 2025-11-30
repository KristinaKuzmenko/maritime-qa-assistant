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
    ) -> None:
        """
        Initialize region classifier.
        
        :param caption_search_distance: Distance in pixels to search for captions
        :param grid_line_threshold: Minimum grid lines to consider table-like
        :param drawing_coverage_threshold: Minimum drawing coverage ratio for schema
        :param numeric_density_threshold: Minimum numeric content ratio for table
        """
        self.caption_search_distance = caption_search_distance
        self.grid_line_threshold = grid_line_threshold
        self.drawing_coverage_threshold = drawing_coverage_threshold
        self.numeric_density_threshold = numeric_density_threshold
    
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
        
        # A. Content-based scoring
        if content_analysis['has_table_structure']:
            table_score += 0.4
        
        if content_analysis['text_to_graphics_ratio'] > 0.5:
            table_score += 0.3
        else:
            schema_score += 0.3
        
        if content_analysis['has_grid_lines']:
            table_score += 0.3
        
        if content_analysis['has_many_vector_shapes']:
            schema_score += 0.4
        
        if content_analysis['numeric_density'] > self.numeric_density_threshold:
            table_score += 0.2
        
        # B. Caption-based scoring (strong signal)
        if caption_type == 'table':
            table_score += 0.5
        elif caption_type == 'figure':
            schema_score += 0.5
        
        # C. YOLO confidence as weak signal
        if region.region_type == RegionType.TABLE:
            table_score += 0.1 * region.confidence
        elif region.region_type == RegionType.SCHEMA:
            schema_score += 0.1 * region.confidence
        
        # Decision
        reclassified_type = RegionType.TABLE if table_score > schema_score else RegionType.SCHEMA
        
        # Log reclassification
        if reclassified_type != region.region_type:
            logger.info(
                f"Reclassified region on page {region.page_number + 1}: "
                f"{region.region_type.value} → {reclassified_type.value} "
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
            'text_to_graphics_ratio': 0.0,
            'numeric_density': 0.0,
        }
        
        # A. Table structure detection
        if len(words) > 0:
            analysis['has_table_structure'] = self._detect_table_structure(words, bbox)
        
        # B. Grid lines detection
        horizontal_lines = [d for d in bbox_drawings if self._is_horizontal_line(d)]
        vertical_lines = [d for d in bbox_drawings if self._is_vertical_line(d)]
        
        if len(horizontal_lines) >= self.grid_line_threshold and len(vertical_lines) >= self.grid_line_threshold:
            analysis['has_grid_lines'] = True
        
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
        if len(words) < 6:  # Minimum for a table
            return False
        
        # Group words by Y-coordinate (rows)
        rows = {}
        tolerance = 5  # pixels
        
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
        
        # Table needs at least 3 rows
        if len(rows) < 3:
            return False
        
        # Check alignment by columns
        # If each row has similar word count and words are X-aligned → table
        row_word_counts = [len(words) for words in rows.values()]
        avg_words_per_row = sum(row_word_counts) / len(row_word_counts)
        
        # Variance should be small
        variance = sum((c - avg_words_per_row)**2 for c in row_word_counts) / len(row_word_counts)
        
        if avg_words_per_row >= 2 and variance < avg_words_per_row:
            # Check X-alignment (columns)
            x_positions = []
            for row_words in rows.values():
                for word in row_words:
                    x_positions.append(word[0])  # x0
            
            # Cluster X positions (columns)
            x_positions.sort()
            columns = []
            current_col = [x_positions[0]]
            
            for x in x_positions[1:]:
                if x - current_col[-1] < 20:  # tolerance
                    current_col.append(x)
                else:
                    columns.append(current_col)
                    current_col = [x]
            columns.append(current_col)
            
            # Table needs at least 2 columns
            if len(columns) >= 2:
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
            return width > 10 and height < 5
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
            return height > 10 and width < 5
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






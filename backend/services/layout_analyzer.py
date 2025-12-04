"""
Layout analyzer using DocLayNet YOLO model for document segmentation.
Classifies page regions into: TABLE, SCHEMA (figures/pictures), and TEXT.
Provides fallback detection for schema-heavy pages based on vector graphics count.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import io

import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """Document region types (superclasses)"""
    TABLE = "table"
    SCHEMA = "schema"  # figures, pictures, diagrams
    TEXT = "text"      # text, titles, headers, lists


@dataclass
class BBox:
    """Bounding box with coordinates"""
    x0: float
    y0: float
    x1: float
    y1: float
    
    def area(self) -> float:
        """Calculate bbox area"""
        return (self.x1 - self.x0) * (self.y1 - self.y0)
    
    def overlaps(self, other: 'BBox', threshold: float = 0.3) -> bool:
        """Check if bboxes overlap significantly"""
        iou = self._calculate_iou(other)
        return iou > threshold
    
    def _calculate_iou(self, other: 'BBox') -> float:
        """Calculate Intersection over Union"""
        x1 = max(self.x0, other.x0)
        y1 = max(self.y0, other.y0)
        x2 = min(self.x1, other.x1)
        y2 = min(self.y1, other.y1)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class Region:
    """Detected document region"""
    bbox: BBox
    region_type: RegionType
    confidence: float
    page_number: int  # zero-based


class LayoutAnalyzer:
    """
    Analyze document layout using DocLayNet YOLO model.
    Segments pages into tables, schemas, and text regions.
    """
    
    # DocLayNet class mapping to our superclasses
    DOCLAYNET_TO_SUPERCLASS = {
        8: RegionType.TABLE,      # Table
        6: RegionType.SCHEMA,     # Picture
        0: RegionType.TEXT,       # Caption
        1: RegionType.TEXT,       # Footnote
        2: RegionType.TEXT,       # Formula (treat as text)
        3: RegionType.TEXT,       # List-item
        4: RegionType.TEXT,       # Page-footer
        5: RegionType.TEXT,       # Page-header
        7: RegionType.TEXT,       # Section-header
        9: RegionType.TEXT,       # Text
        10: RegionType.TEXT,      # Title
    }
    
    def __init__(
        self,
        model_path: str = "../models/yolov10s-best.pt",
        confidence_threshold: float = 0.4,  # Lower default for better table detection
        vector_drawing_threshold: int = 100,  # Min drawings to consider page as schema
        iou_threshold: float = 0.4,
    ) -> None:
        """
        Initialize layout analyzer.
        
        :param model_path: Path to DocLayNet YOLO model
        :param confidence_threshold: Minimum confidence for detections
        :param vector_drawing_threshold: Min vector objects to treat page as full schema
        :param iou_threshold: IoU threshold for NMS deduplication
        """
        self.confidence_threshold = confidence_threshold
        self.vector_drawing_threshold = vector_drawing_threshold
        self.iou_threshold = iou_threshold
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"✅ Loaded DocLayNet YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def analyze_page(
        self,
        page: fitz.Page,
        page_num: int,
    ) -> List[Region]:
        """
        Analyze page layout and detect regions.
        
        :param page: PyMuPDF page object
        :param page_num: Page number (zero-based)
        :return: List of detected regions
        """
        regions = []
        
        # Try YOLO detection first
        if self.model is not None:
            regions = self._detect_with_yolo(page, page_num)
        
        # Fallback: check for schema-heavy page based on vector graphics
        # BUT: Don't add full-page schema if YOLO already found TABLE regions
        # Tables also have many vector drawings (grid lines)!
        has_table = any(r.region_type == RegionType.TABLE for r in regions)
        
        if not self._has_schema_region(regions) and not has_table:
            if self._is_schema_heavy_page(page):
                # Treat entire page as schema
                page_rect = page.rect
                full_page_bbox = BBox(
                    x0=page_rect.x0,
                    y0=page_rect.y0,
                    x1=page_rect.x1,
                    y1=page_rect.y1,
                )
                regions.append(Region(
                    bbox=full_page_bbox,
                    region_type=RegionType.SCHEMA,
                    confidence=1.0,  # High confidence for fallback
                    page_number=page_num,
                ))
                logger.info(
                    f"Page {page_num + 1}: YOLO missed schema, "
                    f"but page has many vector drawings - treating as full-page schema"
                )
        
        return regions
    
    def _detect_with_yolo(
        self,
        page: fitz.Page,
        page_num: int,
    ) -> List[Region]:
        """
        Run YOLO detection on page.
        
        :param page: PyMuPDF page object
        :param page_num: Page number (zero-based)
        :return: List of detected regions
        """
        regions = []
        
        try:
            # ===== FIX 1: Render page to PIL Image =====
            zoom = 2.0  # 2x zoom for better quality
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Store dimensions BEFORE converting
            img_width = pix.width
            img_height = pix.height
            
            # Convert to PIL Image
            img_bytes = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Clean up pixmap
            pix = None
            
            # Get page dimensions for coordinate conversion
            page_width = page.rect.width
            page_height = page.rect.height
            
            # ===== FIX 2: Calculate correct scale =====
            scale_x = page_width / img_width
            scale_y = page_height / img_height
            
            # ===== FIX 3: Run YOLO with PIL Image =====
            results = self.model.predict(
                source=pil_image,  # ← Pass PIL Image, not bytes!
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            
            if not results or len(results) == 0:
                logger.debug(f"Page {page_num + 1}: No regions detected by YOLO")
                return regions
            
            result = results[0]
            
            # Parse detections
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Map to superclass
                region_type = self.DOCLAYNET_TO_SUPERCLASS.get(
                    class_id,
                    RegionType.TEXT,  # default
                )
                
                # Log raw YOLO detection for debugging
                logger.debug(
                    f"Page {page_num + 1}: YOLO raw detection - "
                    f"class_id={class_id}, conf={confidence:.2f}, mapped_to={region_type.value}"
                )
                
                # Get bbox coordinates (in image space)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # ===== FIX 4: Convert to page coordinates =====
                bbox = BBox(
                    x0=x1 * scale_x,
                    y0=y1 * scale_y,
                    x1=x2 * scale_x,
                    y1=y2 * scale_y,
                )
                
                regions.append(Region(
                    bbox=bbox,
                    region_type=region_type,
                    confidence=confidence,
                    page_number=page_num,
                ))
            
            # Log summary
            type_counts = {}
            for r in regions:
                type_name = r.region_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            logger.info(
                f"Page {page_num + 1}: YOLO detected {len(regions)} regions: "
                f"{', '.join(f'{k}={v}' for k, v in type_counts.items())}"
            )
            
        except Exception as e:
            logger.error(
                f"YOLO detection failed on page {page_num + 1}: {e}",
                exc_info=True
            )
        
        return regions
    
    def _is_schema_heavy_page(self, page: fitz.Page) -> bool:
        """
        Check if page has many vector drawings (likely a schema/diagram).
        Fallback detection when YOLO misses figures.
        
        :param page: PyMuPDF page object
        :return: True if page should be treated as full schema
        """
        try:
            drawings = page.get_drawings()
            vector_count = len(drawings)
            
            # Threshold for considering page as schema
            if vector_count >= self.vector_drawing_threshold:
                logger.debug(
                    f"Page has {vector_count} vector objects "
                    f"(threshold: {self.vector_drawing_threshold}) - schema candidate"
                )
                return True
            
        except Exception as e:
            logger.debug(f"Could not count vector drawings: {e}")
        
        return False
    
    def _has_schema_region(self, regions: List[Region]) -> bool:
        """Check if regions list already contains a schema"""
        return any(r.region_type == RegionType.SCHEMA for r in regions)
    
    def filter_regions_by_type(
        self,
        regions: List[Region],
        region_type: RegionType,
    ) -> List[Region]:
        """Filter regions by type"""
        return [r for r in regions if r.region_type == region_type]
    
    def get_occupied_bboxes(
        self,
        regions: List[Region],
        region_types: Optional[List[RegionType]] = None,
    ) -> List[BBox]:
        """
        Get bounding boxes for specified region types.
        Used to mark areas as "occupied" and exclude from other processing.
        
        :param regions: List of regions
        :param region_types: Types to include (default: TABLE, SCHEMA)
        :return: List of bounding boxes
        """
        if region_types is None:
            region_types = [RegionType.TABLE, RegionType.SCHEMA]
        
        return [
            r.bbox for r in regions
            if r.region_type in region_types
        ]
    
    def is_bbox_occupied(
        self,
        bbox: BBox,
        occupied_bboxes: List[BBox],
        overlap_threshold: float = 0.3,
    ) -> bool:
        """
        Check if bbox overlaps with occupied regions.
        
        :param bbox: Bounding box to check
        :param occupied_bboxes: List of occupied bboxes
        :param overlap_threshold: Minimum overlap to consider occupied
        :return: True if bbox is occupied
        """
        for occupied in occupied_bboxes:
            if bbox.overlaps(occupied, threshold=overlap_threshold):
                return True
        return False
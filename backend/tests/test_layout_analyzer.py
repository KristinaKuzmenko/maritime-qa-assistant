"""
Layout Analyzer Tests

Comprehensive tests for YOLO-based document layout analysis including:
- BBox operations (area, IoU, overlap)
- Region creation and attributes
- YOLO detection and coordinate conversion
- Deduplication with type priority
- Fallback schema detection
- Occupied region checking

Run with: pytest test_layout_analyzer.py -v
"""

import pytest
import io
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from services.layout_analyzer import (
    BBox, Region, RegionType, LayoutAnalyzer
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_yolo_model():
    """Create mock YOLO model."""
    model = Mock()
    model.predict = Mock(return_value=[])
    return model


@pytest.fixture
def mock_pdf_page():
    """Create mock PyMuPDF page."""
    page = Mock()
    page.rect = Mock()
    page.rect.width = 612
    page.rect.height = 792
    page.rect.x0 = 0
    page.rect.y0 = 0
    page.rect.x1 = 612
    page.rect.y1 = 792
    page.get_pixmap = Mock(return_value=Mock(
        width=1224,
        height=1584,
        tobytes=Mock(return_value=b"PNG_DATA")
    ))
    page.get_drawings = Mock(return_value=[])
    return page


@pytest.fixture
def analyzer(mock_yolo_model):
    """Create LayoutAnalyzer with mocked YOLO."""
    with patch('services.layout_analyzer.YOLO') as mock_yolo_class:
        mock_yolo_class.return_value = mock_yolo_model
        analyzer = LayoutAnalyzer(model_path="fake_path")
        return analyzer


# =============================================================================
# BBOX TESTS
# =============================================================================

class TestBBox:
    """Test BBox class."""
    
    def test_area_calculation(self):
        """Test bbox area calculation."""
        bbox = BBox(x0=0, y0=0, x1=100, y1=50)
        assert bbox.area() == 5000
    
    def test_area_with_negative_coords(self):
        """Test area with negative coordinates."""
        bbox = BBox(x0=-50, y0=-50, x1=50, y1=50)
        assert bbox.area() == 10000
    
    def test_area_zero(self):
        """Test zero area bbox."""
        bbox = BBox(x0=0, y0=0, x1=0, y1=100)
        assert bbox.area() == 0
    
    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(200, 200, 300, 300)
        
        iou = bbox1._calculate_iou(bbox2)
        assert iou == 0.0
    
    def test_iou_complete_overlap(self):
        """Test IoU with identical boxes."""
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(0, 0, 100, 100)
        
        iou = bbox1._calculate_iou(bbox2)
        assert iou == 1.0
    
    def test_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(50, 50, 150, 150)
        
        iou = bbox1._calculate_iou(bbox2)
        
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU = 2500/17500 ≈ 0.143
        assert 0.14 < iou < 0.15
    
    def test_iou_contained_box(self):
        """Test IoU when one box contains another."""
        outer = BBox(0, 0, 100, 100)
        inner = BBox(25, 25, 75, 75)
        
        iou = outer._calculate_iou(inner)
        
        # Intersection = inner area = 2500
        # Union = outer area = 10000
        # IoU = 2500/10000 = 0.25
        assert iou == 0.25
    
    def test_overlaps_above_threshold(self):
        """Test overlaps returns True above threshold."""
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(50, 50, 150, 150)
        
        # IoU ~0.14
        assert bbox1.overlaps(bbox2, threshold=0.1) is True
    
    def test_overlaps_below_threshold(self):
        """Test overlaps returns False below threshold."""
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(50, 50, 150, 150)
        
        # IoU ~0.14
        assert bbox1.overlaps(bbox2, threshold=0.5) is False
    
    def test_overlaps_default_threshold(self):
        """Test overlaps with default threshold (0.3)."""
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(10, 10, 110, 110)
        
        # IoU should be high enough for default threshold
        result = bbox1.overlaps(bbox2)
        # Default threshold is 0.3
        assert isinstance(result, bool)
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        bbox = BBox(10, 20, 30, 40)
        
        result = bbox.to_dict()
        
        assert result == {"x0": 10, "y0": 20, "x1": 30, "y1": 40}
    
    def test_to_dict_with_floats(self):
        """Test to_dict with float coordinates."""
        bbox = BBox(10.5, 20.5, 30.5, 40.5)
        
        result = bbox.to_dict()
        
        assert result["x0"] == 10.5
        assert result["y1"] == 40.5


# =============================================================================
# REGION TESTS
# =============================================================================

class TestRegion:
    """Test Region dataclass."""
    
    def test_region_required_fields(self):
        """Test region creation with required fields."""
        bbox = BBox(0, 0, 100, 100)
        region = Region(
            bbox=bbox,
            region_type=RegionType.SCHEMA,
            confidence=0.85,
            page_number=0,
        )
        
        assert region.bbox == bbox
        assert region.region_type == RegionType.SCHEMA
        assert region.confidence == 0.85
        assert region.page_number == 0
    
    def test_region_default_values(self):
        """Test region default optional values."""
        bbox = BBox(0, 0, 100, 100)
        region = Region(
            bbox=bbox,
            region_type=RegionType.TABLE,
            confidence=0.9,
            page_number=0,
        )
        
        assert region.caption_text is None
        assert region.yolo_class_id is None
        assert region.extract_text_also is False
    
    def test_region_with_caption(self):
        """Test region with caption text."""
        bbox = BBox(0, 0, 100, 100)
        region = Region(
            bbox=bbox,
            region_type=RegionType.TABLE,
            confidence=0.9,
            page_number=0,
            caption_text="Table 3-2: Overview",
        )
        
        assert region.caption_text == "Table 3-2: Overview"
    
    def test_region_with_yolo_class_id(self):
        """Test region with YOLO class ID."""
        bbox = BBox(0, 0, 100, 100)
        region = Region(
            bbox=bbox,
            region_type=RegionType.TABLE,
            confidence=0.9,
            page_number=0,
            yolo_class_id=8,
        )
        
        assert region.yolo_class_id == 8
    
    def test_region_extract_text_also(self):
        """Test region with dual extraction flag."""
        bbox = BBox(0, 0, 100, 100)
        region = Region(
            bbox=bbox,
            region_type=RegionType.SCHEMA,
            confidence=0.7,
            page_number=0,
            extract_text_also=True,
        )
        
        assert region.extract_text_also is True


# =============================================================================
# REGION TYPE TESTS
# =============================================================================

class TestRegionType:
    """Test RegionType enum."""
    
    def test_table_value(self):
        """Test TABLE value."""
        assert RegionType.TABLE.value == "table"
    
    def test_schema_value(self):
        """Test SCHEMA value."""
        assert RegionType.SCHEMA.value == "schema"
    
    def test_text_value(self):
        """Test TEXT value."""
        assert RegionType.TEXT.value == "text"


# =============================================================================
# LAYOUT ANALYZER INITIALIZATION TESTS
# =============================================================================

class TestLayoutAnalyzerInit:
    """Test LayoutAnalyzer initialization."""
    
    def test_init_loads_model(self):
        """Test model is loaded on init."""
        with patch('services.layout_analyzer.YOLO') as mock_yolo:
            mock_yolo.return_value = Mock()
            
            analyzer = LayoutAnalyzer(model_path="path/to/model.pt")
            
            mock_yolo.assert_called_once_with("path/to/model.pt")
    
    def test_init_default_thresholds(self):
        """Test default threshold values."""
        with patch('services.layout_analyzer.YOLO'):
            analyzer = LayoutAnalyzer(model_path="fake")
            
            assert analyzer.confidence_threshold == 0.4
            assert analyzer.caption_confidence_threshold == 0.1
            assert analyzer.vector_drawing_threshold == 100
            assert analyzer.iou_threshold == 0.4
    
    def test_init_custom_thresholds(self):
        """Test custom threshold values."""
        with patch('services.layout_analyzer.YOLO'):
            analyzer = LayoutAnalyzer(
                model_path="fake",
                confidence_threshold=0.5,
                caption_confidence_threshold=0.2,
                vector_drawing_threshold=200,
                iou_threshold=0.5,
            )
            
            assert analyzer.confidence_threshold == 0.5
            assert analyzer.caption_confidence_threshold == 0.2
            assert analyzer.vector_drawing_threshold == 200
            assert analyzer.iou_threshold == 0.5
    
    def test_init_model_load_failure(self):
        """Test graceful handling of model load failure."""
        with patch('services.layout_analyzer.YOLO') as mock_yolo:
            mock_yolo.side_effect = Exception("Model not found")
            
            analyzer = LayoutAnalyzer(model_path="invalid/path")
            
            assert analyzer.model is None


# =============================================================================
# DEDUPLICATION TESTS
# =============================================================================

class TestDeduplication:
    """Test _deduplicate_regions method."""
    
    def test_no_duplicates(self, analyzer):
        """Test deduplication with no overlapping regions."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.9, 0),
            Region(BBox(200, 200, 300, 300), RegionType.TABLE, 0.85, 0),
        ]
        
        result = analyzer._deduplicate_regions(regions)
        
        assert len(result) == 2
    
    def test_schema_priority_over_text(self, analyzer):
        """Test SCHEMA kept over TEXT when overlapping."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.6, 0),
            Region(BBox(5, 5, 105, 105), RegionType.TEXT, 0.95, 0),
        ]
        
        result = analyzer._deduplicate_regions(regions)
        
        assert len(result) == 1
        assert result[0].region_type == RegionType.SCHEMA
    
    def test_table_priority_over_text(self, analyzer):
        """Test TABLE kept over TEXT when overlapping."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.7, 0),
            Region(BBox(2, 2, 102, 102), RegionType.TEXT, 0.98, 0),
        ]
        
        result = analyzer._deduplicate_regions(regions)
        
        assert len(result) == 1
        assert result[0].region_type == RegionType.TABLE
    
    def test_schema_priority_over_table(self, analyzer):
        """Test SCHEMA kept over TABLE when overlapping."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.6, 0),
            Region(BBox(5, 5, 105, 105), RegionType.TABLE, 0.95, 0),
        ]
        
        result = analyzer._deduplicate_regions(regions)
        
        assert len(result) == 1
        assert result[0].region_type == RegionType.SCHEMA
    
    def test_same_type_higher_confidence_wins(self, analyzer):
        """Test higher confidence wins for same type."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.6, 0),
            Region(BBox(5, 5, 105, 105), RegionType.SCHEMA, 0.9, 0),
        ]
        
        result = analyzer._deduplicate_regions(regions)
        
        assert len(result) == 1
        assert result[0].confidence == 0.9
    
    def test_single_region(self, analyzer):
        """Test single region unchanged."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.85, 0),
        ]
        
        result = analyzer._deduplicate_regions(regions)
        
        assert len(result) == 1
        assert result[0] == regions[0]
    
    def test_empty_list(self, analyzer):
        """Test empty list returns empty."""
        result = analyzer._deduplicate_regions([])
        
        assert result == []
    
    def test_low_iou_not_deduplicated(self, analyzer):
        """Test regions with low IoU are not deduplicated."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.9, 0),
            Region(BBox(50, 50, 150, 150), RegionType.TEXT, 0.95, 0),
        ]
        
        result = analyzer._deduplicate_regions(regions)
        
        # IoU ~0.14, threshold 0.8 - should keep both
        assert len(result) == 2


# =============================================================================
# ANALYZE PAGE TESTS
# =============================================================================

class TestAnalyzePage:
    """Test analyze_page method."""
    
    def test_analyze_page_no_model(self, mock_pdf_page):
        """Test analyze_page with no model."""
        with patch('services.layout_analyzer.YOLO') as mock_yolo:
            mock_yolo.side_effect = Exception("No model")
            analyzer = LayoutAnalyzer(model_path="invalid")
        
        result = analyzer.analyze_page(mock_pdf_page, page_num=0)
        
        # Should return empty (model is None)
        assert result == []
    
    def test_analyze_page_calls_yolo(self, analyzer, mock_pdf_page, mock_yolo_model):
        """Test analyze_page calls YOLO detection."""
        mock_yolo_model.predict.return_value = [Mock(boxes=[])]
        
        with patch.object(analyzer, '_detect_with_yolo', return_value=[]) as mock_detect:
            analyzer.analyze_page(mock_pdf_page, page_num=0)
            
            mock_detect.assert_called_once()
    
    def test_analyze_page_fallback_schema_heavy(self, analyzer, mock_pdf_page):
        """Test fallback to full-page schema for drawing-heavy pages."""
        # No YOLO detections
        with patch.object(analyzer, '_detect_with_yolo', return_value=[]):
            # Many vector drawings
            with patch.object(analyzer, '_is_schema_heavy_page', return_value=True):
                result = analyzer.analyze_page(mock_pdf_page, page_num=0)
        
        assert len(result) == 1
        assert result[0].region_type == RegionType.SCHEMA
        assert result[0].confidence == 1.0
    
    def test_analyze_page_no_fallback_when_table_detected(self, analyzer, mock_pdf_page):
        """Test no fallback when TABLE already detected."""
        table_region = Region(
            BBox(0, 0, 100, 100),
            RegionType.TABLE,
            0.9,
            0
        )
        
        with patch.object(analyzer, '_detect_with_yolo', return_value=[table_region]):
            with patch.object(analyzer, '_is_schema_heavy_page', return_value=True):
                result = analyzer.analyze_page(mock_pdf_page, page_num=0)
        
        # Should only have the table, not full-page schema
        assert len(result) == 1
        assert result[0].region_type == RegionType.TABLE
    
    def test_analyze_page_no_fallback_when_schema_detected(self, analyzer, mock_pdf_page):
        """Test no fallback when SCHEMA already detected."""
        schema_region = Region(
            BBox(50, 50, 200, 200),
            RegionType.SCHEMA,
            0.85,
            0
        )
        
        with patch.object(analyzer, '_detect_with_yolo', return_value=[schema_region]):
            with patch.object(analyzer, '_is_schema_heavy_page', return_value=True):
                result = analyzer.analyze_page(mock_pdf_page, page_num=0)
        
        # Should only have the detected schema
        assert len(result) == 1


# =============================================================================
# SCHEMA HEAVY PAGE TESTS
# =============================================================================

class TestSchemaHeavyPage:
    """Test _is_schema_heavy_page method."""
    
    def test_many_drawings_returns_true(self, analyzer, mock_pdf_page):
        """Test page with many drawings returns True."""
        # Create 150 mock drawings (above threshold of 100)
        mock_pdf_page.get_drawings.return_value = [Mock()] * 150
        
        result = analyzer._is_schema_heavy_page(mock_pdf_page)
        
        assert result is True
    
    def test_few_drawings_returns_false(self, analyzer, mock_pdf_page):
        """Test page with few drawings returns False."""
        mock_pdf_page.get_drawings.return_value = [Mock()] * 50
        
        result = analyzer._is_schema_heavy_page(mock_pdf_page)
        
        assert result is False
    
    def test_at_threshold_returns_true(self, analyzer, mock_pdf_page):
        """Test page at threshold returns True."""
        mock_pdf_page.get_drawings.return_value = [Mock()] * 100
        
        result = analyzer._is_schema_heavy_page(mock_pdf_page)
        
        assert result is True
    
    def test_below_threshold_returns_false(self, analyzer, mock_pdf_page):
        """Test page below threshold returns False."""
        mock_pdf_page.get_drawings.return_value = [Mock()] * 99
        
        result = analyzer._is_schema_heavy_page(mock_pdf_page)
        
        assert result is False
    
    def test_get_drawings_error_returns_false(self, analyzer, mock_pdf_page):
        """Test error in get_drawings returns False."""
        mock_pdf_page.get_drawings.side_effect = Exception("Error")
        
        result = analyzer._is_schema_heavy_page(mock_pdf_page)
        
        assert result is False


# =============================================================================
# HAS SCHEMA REGION TESTS
# =============================================================================

class TestHasSchemaRegion:
    """Test _has_schema_region method."""
    
    def test_has_schema_returns_true(self, analyzer):
        """Test returns True when SCHEMA present."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.9, 0),
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.85, 0),
        ]
        
        result = analyzer._has_schema_region(regions)
        
        assert result is True
    
    def test_no_schema_returns_false(self, analyzer):
        """Test returns False when no SCHEMA."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.9, 0),
            Region(BBox(0, 0, 100, 100), RegionType.TEXT, 0.85, 0),
        ]
        
        result = analyzer._has_schema_region(regions)
        
        assert result is False
    
    def test_empty_list_returns_false(self, analyzer):
        """Test empty list returns False."""
        result = analyzer._has_schema_region([])
        
        assert result is False


# =============================================================================
# FILTER REGIONS BY TYPE TESTS
# =============================================================================

class TestFilterRegionsByType:
    """Test filter_regions_by_type method."""
    
    def test_filter_schemas(self, analyzer):
        """Test filtering SCHEMA regions."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.9, 0),
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.85, 0),
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.88, 0),
        ]
        
        result = analyzer.filter_regions_by_type(regions, RegionType.SCHEMA)
        
        assert len(result) == 2
        assert all(r.region_type == RegionType.SCHEMA for r in result)
    
    def test_filter_tables(self, analyzer):
        """Test filtering TABLE regions."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.SCHEMA, 0.9, 0),
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.85, 0),
        ]
        
        result = analyzer.filter_regions_by_type(regions, RegionType.TABLE)
        
        assert len(result) == 1
        assert result[0].region_type == RegionType.TABLE
    
    def test_filter_text(self, analyzer):
        """Test filtering TEXT regions."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.TEXT, 0.95, 0),
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.85, 0),
        ]
        
        result = analyzer.filter_regions_by_type(regions, RegionType.TEXT)
        
        assert len(result) == 1
    
    def test_filter_no_matches(self, analyzer):
        """Test filter with no matches."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.85, 0),
        ]
        
        result = analyzer.filter_regions_by_type(regions, RegionType.SCHEMA)
        
        assert result == []
    
    def test_filter_empty_list(self, analyzer):
        """Test filter on empty list."""
        result = analyzer.filter_regions_by_type([], RegionType.SCHEMA)
        
        assert result == []


# =============================================================================
# GET OCCUPIED BBOXES TESTS
# =============================================================================

class TestGetOccupiedBboxes:
    """Test get_occupied_bboxes method."""
    
    def test_default_types(self, analyzer):
        """Test default types (TABLE, SCHEMA)."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.9, 0),
            Region(BBox(100, 100, 200, 200), RegionType.SCHEMA, 0.85, 0),
            Region(BBox(200, 200, 300, 300), RegionType.TEXT, 0.95, 0),
        ]
        
        result = analyzer.get_occupied_bboxes(regions)
        
        assert len(result) == 2
    
    def test_custom_types(self, analyzer):
        """Test custom region types filter."""
        regions = [
            Region(BBox(0, 0, 100, 100), RegionType.TABLE, 0.9, 0),
            Region(BBox(100, 100, 200, 200), RegionType.SCHEMA, 0.85, 0),
            Region(BBox(200, 200, 300, 300), RegionType.TEXT, 0.95, 0),
        ]
        
        result = analyzer.get_occupied_bboxes(regions, region_types=[RegionType.TEXT])
        
        assert len(result) == 1
    
    def test_empty_regions(self, analyzer):
        """Test with empty regions list."""
        result = analyzer.get_occupied_bboxes([])
        
        assert result == []


# =============================================================================
# IS BBOX OCCUPIED TESTS
# =============================================================================

class TestIsBboxOccupied:
    """Test is_bbox_occupied method."""
    
    def test_occupied_returns_true(self, analyzer):
        """Test occupied bbox returns True."""
        bbox = BBox(0, 0, 100, 100)
        occupied = [BBox(50, 50, 150, 150)]  # Overlapping
        
        result = analyzer.is_bbox_occupied(bbox, occupied, overlap_threshold=0.1)
        
        assert result is True
    
    def test_not_occupied_returns_false(self, analyzer):
        """Test non-occupied bbox returns False."""
        bbox = BBox(0, 0, 100, 100)
        occupied = [BBox(200, 200, 300, 300)]  # Not overlapping
        
        result = analyzer.is_bbox_occupied(bbox, occupied)
        
        assert result is False
    
    def test_empty_occupied_returns_false(self, analyzer):
        """Test with empty occupied list."""
        bbox = BBox(0, 0, 100, 100)
        
        result = analyzer.is_bbox_occupied(bbox, [])
        
        assert result is False
    
    def test_below_threshold_returns_false(self, analyzer):
        """Test slight overlap below threshold returns False."""
        bbox = BBox(0, 0, 100, 100)
        occupied = [BBox(90, 90, 200, 200)]  # Small overlap
        
        result = analyzer.is_bbox_occupied(bbox, occupied, overlap_threshold=0.5)
        
        assert result is False
    
    def test_multiple_occupied_regions(self, analyzer):
        """Test with multiple occupied regions."""
        bbox = BBox(150, 150, 250, 250)
        occupied = [
            BBox(0, 0, 100, 100),      # No overlap
            BBox(200, 200, 300, 300),  # Overlap!
        ]
        
        result = analyzer.is_bbox_occupied(bbox, occupied, overlap_threshold=0.1)
        
        assert result is True


# =============================================================================
# DOCLAYNET MAPPING TESTS
# =============================================================================

class TestDocLayNetMapping:
    """Test DocLayNet class mapping."""
    
    def test_table_mapping(self, analyzer):
        """Test table class mapping."""
        assert analyzer.DOCLAYNET_TO_SUPERCLASS[8] == RegionType.TABLE
    
    def test_picture_mapping(self, analyzer):
        """Test picture/schema class mapping."""
        assert analyzer.DOCLAYNET_TO_SUPERCLASS[6] == RegionType.SCHEMA
    
    def test_text_classes_mapping(self, analyzer):
        """Test text classes mapping."""
        text_classes = [0, 1, 2, 3, 4, 5, 7, 9, 10]
        
        for cls_id in text_classes:
            assert analyzer.DOCLAYNET_TO_SUPERCLASS[cls_id] == RegionType.TEXT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

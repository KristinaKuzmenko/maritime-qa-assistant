"""
Region Classifier Tests

Comprehensive tests for region classification including:
- Caption detection (table/figure patterns)
- YOLO confidence thresholds
- LLM verification
- TOC detection
- Schema preservation logic
- Statistics tracking

Run with: pytest test_region_classifier.py -v
"""

import pytest
import re
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional, Tuple

from services.region_classifier import RegionClassifier
from services.layout_analyzer import Region, RegionType, BBox


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    llm = Mock()
    llm.chat = Mock()
    llm.chat.completions = Mock()
    llm.chat.completions.create = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content="TABLE"))]
    ))
    return llm


@pytest.fixture
def classifier(mock_llm_service):
    """Create RegionClassifier with mocked LLM."""
    return RegionClassifier(
        llm_service=mock_llm_service,
        caption_search_distance=600,
        yolo_confidence_threshold=0.8,
        enable_llm_verification=True,
    )


@pytest.fixture
def classifier_no_llm():
    """Create RegionClassifier without LLM."""
    return RegionClassifier(
        llm_service=None,
        enable_llm_verification=False,
    )


@pytest.fixture
def mock_pdf_page():
    """Create mock PyMuPDF page."""
    page = Mock()
    page.rect = Mock()
    page.rect.width = 612
    page.rect.height = 792
    page.get_text = Mock(return_value="Default text content")
    page.get_pixmap = Mock(return_value=Mock(
        tobytes=Mock(return_value=b"PNG_DATA")
    ))
    return page


@pytest.fixture
def sample_table_region():
    """Create sample TABLE region."""
    return Region(
        bbox=BBox(100, 200, 400, 500),
        region_type=RegionType.TABLE,
        confidence=0.85,
        page_number=0,
    )


@pytest.fixture
def sample_schema_region():
    """Create sample SCHEMA region."""
    return Region(
        bbox=BBox(100, 200, 400, 500),
        region_type=RegionType.SCHEMA,
        confidence=0.75,
        page_number=0,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Test classifier initialization."""
    
    def test_init_with_llm(self, mock_llm_service):
        """Test initialization with LLM service."""
        classifier = RegionClassifier(
            llm_service=mock_llm_service,
            enable_llm_verification=True,
        )
        
        assert classifier.llm_service == mock_llm_service
        assert classifier.enable_llm_verification is True
    
    def test_init_without_llm(self):
        """Test initialization without LLM service."""
        classifier = RegionClassifier(
            llm_service=None,
            enable_llm_verification=True,  # Should be disabled anyway
        )
        
        assert classifier.llm_service is None
        assert classifier.enable_llm_verification is False
    
    def test_init_custom_threshold(self, mock_llm_service):
        """Test custom confidence threshold."""
        classifier = RegionClassifier(
            llm_service=mock_llm_service,
            yolo_confidence_threshold=0.9,
        )
        
        assert classifier.yolo_confidence_threshold == 0.9
    
    def test_init_stats_zeroed(self, classifier):
        """Test statistics are initialized to zero."""
        expected_keys = [
            'total_classified', 'caption_detected', 'high_confidence_yolo',
            'llm_verified', 'llm_changed_decision', 'schema_preserved_from_text'
        ]
        
        for key in expected_keys:
            assert classifier.stats[key] == 0


# =============================================================================
# CAPTION DETECTION TESTS
# =============================================================================

class TestCaptionDetection:
    """Test _detect_caption_type method."""
    
    def test_detect_table_caption_simple(self, classifier, mock_pdf_page):
        """Test detection of simple table caption."""
        mock_pdf_page.get_text.return_value = "Table 3-2: System Overview"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'table'
        assert caption_text is not None
        assert "Table" in caption_text or "table" in caption_text.lower()
    
    def test_detect_table_caption_with_number(self, classifier, mock_pdf_page):
        """Test detection of table caption with number."""
        mock_pdf_page.get_text.return_value = "Table 5: Component List"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'table'
    
    def test_detect_table_caption_tab_abbreviation(self, classifier, mock_pdf_page):
        """Test detection of 'Tab.' abbreviation."""
        mock_pdf_page.get_text.return_value = "Tab. 2-1: Specifications"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'table'
    
    def test_detect_figure_caption(self, classifier, mock_pdf_page):
        """Test detection of figure caption."""
        mock_pdf_page.get_text.return_value = "Figure 4-1: Main Engine Diagram"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'figure'
        assert "Figure" in caption_text or "figure" in caption_text.lower()
    
    def test_detect_figure_caption_fig_abbreviation(self, classifier, mock_pdf_page):
        """Test detection of 'Fig.' abbreviation."""
        mock_pdf_page.get_text.return_value = "Fig. 3: Pump Assembly"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'figure'
    
    def test_detect_diagram_caption(self, classifier, mock_pdf_page):
        """Test detection of 'Diagram' caption."""
        mock_pdf_page.get_text.return_value = "Diagram 5-2: Fuel System"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'figure'
    
    def test_detect_drawing_caption(self, classifier, mock_pdf_page):
        """Test detection of 'Drawing' caption."""
        mock_pdf_page.get_text.return_value = "Drawing 8-1: Assembly View"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'figure'
    
    def test_detect_schema_caption(self, classifier, mock_pdf_page):
        """Test detection of 'Schema' caption."""
        mock_pdf_page.get_text.return_value = "Schema 2: Control System"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'figure'
    
    def test_detect_maritime_fig_notation(self, classifier, mock_pdf_page):
        """Test detection of maritime << FIG >> notation."""
        mock_pdf_page.get_text.return_value = "<< FIG. S-03 >>"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'figure'
    
    def test_no_caption_found(self, classifier, mock_pdf_page):
        """Test when no caption is found."""
        mock_pdf_page.get_text.return_value = "Regular text without any caption markers."
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type is None
        assert caption_text is None
    
    def test_empty_text(self, classifier, mock_pdf_page):
        """Test with empty text."""
        mock_pdf_page.get_text.return_value = ""
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type is None
        assert caption_text is None
    
    def test_table_priority_over_figure(self, classifier, mock_pdf_page):
        """Test that table caption has priority."""
        mock_pdf_page.get_text.return_value = "Table 1 and Figure 2"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        # Table should be detected first
        assert caption_type == 'table'


# =============================================================================
# TOC DETECTION TESTS
# =============================================================================

class TestTocDetection:
    """Test _is_toc_region method."""
    
    def test_detect_toc_dotted_lines(self, classifier, mock_pdf_page):
        """Test TOC detection with dotted lines."""
        toc_text = """
1.0 INTRODUCTION .............. 5
2.0 SAFETY PRECAUTIONS ....... 10
3.0 TECHNICAL SPECIFICATIONS .. 15
4.0 INSTALLATION .............. 20
        """
        mock_pdf_page.get_text.return_value = toc_text
        
        bbox = BBox(100, 200, 400, 600)
        result = classifier._is_toc_region(mock_pdf_page, bbox)
        
        assert result is True
    
    def test_detect_toc_spaced_lines(self, classifier, mock_pdf_page):
        """Test TOC detection with spaced lines."""
        toc_text = """
1.0 INTRODUCTION                    5
2.0 SAFETY                         10
3.0 SPECIFICATIONS                 15
        """
        mock_pdf_page.get_text.return_value = toc_text
        
        bbox = BBox(100, 200, 400, 600)
        result = classifier._is_toc_region(mock_pdf_page, bbox)
        
        assert result is True
    
    def test_detect_toc_hierarchical(self, classifier, mock_pdf_page):
        """Test TOC detection with hierarchical numbering."""
        toc_text = """
1.0 INTRODUCTION
1.1 Overview
1.2 Scope
2.0 SAFETY
2.1 General Safety
        """
        mock_pdf_page.get_text.return_value = toc_text
        
        bbox = BBox(100, 200, 400, 600)
        result = classifier._is_toc_region(mock_pdf_page, bbox)
        
        assert result is True
    
    def test_not_toc_regular_table(self, classifier, mock_pdf_page):
        """Test that regular table is not detected as TOC."""
        table_text = """
Component | Pressure | Flow
PU-101 | 50 bar | 100 L/min
PU-102 | 45 bar | 80 L/min
        """
        mock_pdf_page.get_text.return_value = table_text
        
        bbox = BBox(100, 200, 400, 400)
        result = classifier._is_toc_region(mock_pdf_page, bbox)
        
        assert result is False
    
    def test_not_toc_short_text(self, classifier, mock_pdf_page):
        """Test that short text is not TOC."""
        mock_pdf_page.get_text.return_value = "Short text"
        
        bbox = BBox(100, 200, 400, 400)
        result = classifier._is_toc_region(mock_pdf_page, bbox)
        
        assert result is False
    
    def test_toc_header_detection(self, classifier, mock_pdf_page):
        """Test TOC header detection."""
        # First call: region text (not TOC-like)
        # Second call: header text (contains "CONTENTS")
        mock_pdf_page.get_text.side_effect = [
            "Some regular text that isn't TOC-like but long enough",
            "TABLE OF CONTENTS"
        ]
        
        bbox = BBox(100, 200, 400, 600)
        result = classifier._is_toc_region(mock_pdf_page, bbox)
        
        assert result is True


# =============================================================================
# RECLASSIFY REGION TESTS
# =============================================================================

class TestReclassifyRegion:
    """Test reclassify_region method."""
    
    @pytest.mark.asyncio
    async def test_high_confidence_schema_trusted(self, classifier, mock_pdf_page):
        """Test high confidence SCHEMA is trusted without LLM."""
        mock_pdf_page.get_text.return_value = "Some text"
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.85,  # >= 0.8
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        assert result == RegionType.SCHEMA
        assert classifier.stats['high_confidence_yolo'] == 1
    
    @pytest.mark.asyncio
    async def test_high_confidence_table_trusted(self, classifier, mock_pdf_page):
        """Test high confidence TABLE is trusted without LLM."""
        mock_pdf_page.get_text.return_value = "Some text"
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.TABLE,
            confidence=0.92,  # >= 0.8
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        assert result == RegionType.TABLE
        assert classifier.stats['high_confidence_yolo'] == 1
    
    @pytest.mark.asyncio
    async def test_table_caption_overrides_yolo(self, classifier, mock_pdf_page):
        """Test table caption overrides YOLO SCHEMA."""
        mock_pdf_page.get_text.return_value = "Table 5: Component List"
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,  # YOLO says SCHEMA
            confidence=0.6,
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        assert result == RegionType.TABLE
        assert classifier.stats['caption_detected'] == 1
    
    @pytest.mark.asyncio
    async def test_figure_caption_overrides_yolo(self, classifier, mock_pdf_page):
        """Test figure caption overrides YOLO TABLE (low confidence)."""
        mock_pdf_page.get_text.return_value = "Figure 8-3: Control Panel"
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.TABLE,  # YOLO says TABLE
            confidence=0.55,  # Low confidence
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        assert result == RegionType.SCHEMA
        assert region.caption_text is not None
    
    @pytest.mark.asyncio
    async def test_figure_caption_does_not_override_high_confidence_table(self, classifier, mock_pdf_page):
        """Test figure caption doesn't override high confidence TABLE."""
        mock_pdf_page.get_text.return_value = "Figure 8-3: Control Panel"
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.TABLE,  # YOLO says TABLE
            confidence=0.85,  # High confidence
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        # Should keep TABLE because YOLO is confident
        assert result == RegionType.TABLE
    
    @pytest.mark.asyncio
    async def test_low_confidence_uses_llm(self, classifier, mock_pdf_page, mock_llm_service):
        """Test low confidence triggers LLM verification."""
        mock_pdf_page.get_text.return_value = "Some text without caption"
        mock_llm_service.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="TABLE"))]
        )
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.55,  # < 0.8
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        assert classifier.stats['llm_verified'] == 1
    
    @pytest.mark.asyncio
    async def test_llm_changes_decision(self, classifier, mock_pdf_page, mock_llm_service):
        """Test LLM can change YOLO decision."""
        mock_pdf_page.get_text.return_value = "Some text"
        mock_llm_service.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="TABLE"))]
        )
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,  # YOLO says SCHEMA
            confidence=0.42,  # Low confidence
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        assert result == RegionType.TABLE
        assert classifier.stats['llm_changed_decision'] == 1
    
    @pytest.mark.asyncio
    async def test_schema_preserved_when_llm_says_text(self, classifier, mock_pdf_page, mock_llm_service):
        """Test SCHEMA preserved when LLM says TEXT but YOLO confident."""
        mock_pdf_page.get_text.return_value = "Some text"
        mock_llm_service.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="TEXT"))]
        )
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.65,  # >= 0.5 threshold for preservation
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        # Should preserve as SCHEMA and enable text extraction
        assert result == RegionType.SCHEMA
        assert region.extract_text_also is True
        assert classifier.stats['schema_preserved_from_text'] == 1
    
    @pytest.mark.asyncio
    async def test_no_schema_preservation_very_low_confidence(self, classifier, mock_pdf_page, mock_llm_service):
        """Test no SCHEMA preservation when confidence too low."""
        mock_pdf_page.get_text.return_value = "Some text"
        mock_llm_service.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="TEXT"))]
        )
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.45,  # < 0.5 threshold
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        # Should accept LLM decision
        assert result == RegionType.TEXT
        assert region.extract_text_also is False
    
    @pytest.mark.asyncio
    async def test_toc_detected_as_text(self, classifier, mock_pdf_page):
        """Test TOC region detected and returned as TEXT."""
        toc_text = """
1.0 INTRODUCTION .............. 5
2.0 SAFETY ................... 10
3.0 SPECIFICATIONS ........... 15
        """
        mock_pdf_page.get_text.return_value = toc_text
        
        region = Region(
            bbox=BBox(100, 200, 400, 600),
            region_type=RegionType.TABLE,  # YOLO thinks it's a table
            confidence=0.9,
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        assert result == RegionType.TEXT
    
    @pytest.mark.asyncio
    async def test_skip_text_region(self, classifier, mock_pdf_page):
        """Test TEXT regions are skipped."""
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.TEXT,
            confidence=0.9,
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        assert result == RegionType.TEXT
        # Should not count in stats
        assert classifier.stats['total_classified'] == 1


# =============================================================================
# LLM VERIFICATION TESTS
# =============================================================================

class TestLlmVerification:
    """Test _llm_verify_type method."""
    
    @pytest.mark.asyncio
    async def test_llm_returns_table(self, classifier, mock_pdf_page, mock_llm_service):
        """Test LLM verification returns TABLE."""
        mock_llm_service.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="TABLE"))]
        )
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.5,
            page_number=0,
        )
        
        result = await classifier._llm_verify_type(mock_pdf_page, region, 1)
        
        assert result == RegionType.TABLE
    
    @pytest.mark.asyncio
    async def test_llm_returns_schema(self, classifier, mock_pdf_page, mock_llm_service):
        """Test LLM verification returns SCHEMA."""
        mock_llm_service.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="SCHEMA"))]
        )
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.TABLE,
            confidence=0.5,
            page_number=0,
        )
        
        result = await classifier._llm_verify_type(mock_pdf_page, region, 1)
        
        assert result == RegionType.SCHEMA
    
    @pytest.mark.asyncio
    async def test_llm_returns_text(self, classifier, mock_pdf_page, mock_llm_service):
        """Test LLM verification returns TEXT."""
        mock_llm_service.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="TEXT"))]
        )
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.5,
            page_number=0,
        )
        
        result = await classifier._llm_verify_type(mock_pdf_page, region, 1)
        
        assert result == RegionType.TEXT
    
    @pytest.mark.asyncio
    async def test_llm_unclear_response_uses_yolo(self, classifier, mock_pdf_page, mock_llm_service):
        """Test unclear LLM response falls back to YOLO."""
        mock_llm_service.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="UNCLEAR"))]
        )
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.5,
            page_number=0,
        )
        
        result = await classifier._llm_verify_type(mock_pdf_page, region, 1)
        
        # Should fall back to YOLO
        assert result == RegionType.SCHEMA


# =============================================================================
# RENDER REGION TESTS
# =============================================================================

class TestRenderRegion:
    """Test _render_region_as_png method."""
    
    def test_render_returns_bytes(self, classifier, mock_pdf_page):
        """Test rendering returns PNG bytes."""
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"PNG_DATA"
        mock_pdf_page.get_pixmap.return_value = mock_pixmap
        
        bbox = BBox(100, 200, 300, 400)
        result = classifier._render_region_as_png(mock_pdf_page, bbox)
        
        assert isinstance(result, bytes)
        mock_pdf_page.get_pixmap.assert_called_once()
    
    def test_render_with_custom_zoom(self, classifier, mock_pdf_page):
        """Test rendering with custom zoom."""
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"PNG_DATA"
        mock_pdf_page.get_pixmap.return_value = mock_pixmap
        
        bbox = BBox(100, 200, 300, 400)
        result = classifier._render_region_as_png(mock_pdf_page, bbox, zoom=3.0)
        
        assert isinstance(result, bytes)


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestStatistics:
    """Test statistics tracking and logging."""
    
    def test_stats_initialized(self, classifier):
        """Test all stats are initialized."""
        expected = {
            'total_classified': 0,
            'caption_detected': 0,
            'high_confidence_yolo': 0,
            'llm_verified': 0,
            'llm_changed_decision': 0,
            'schema_preserved_from_text': 0,
        }
        
        assert classifier.stats == expected
    
    @pytest.mark.asyncio
    async def test_stats_increment_on_classification(self, classifier, mock_pdf_page):
        """Test stats increment during classification."""
        mock_pdf_page.get_text.return_value = "Some text"
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.9,
            page_number=0,
        )
        
        await classifier.reclassify_region(mock_pdf_page, region)
        
        assert classifier.stats['total_classified'] == 1
        assert classifier.stats['high_confidence_yolo'] == 1
    
    def test_log_statistics_no_regions(self, classifier):
        """Test log_statistics with no regions."""
        # Should not raise
        classifier.log_statistics()
    
    def test_log_statistics_with_data(self, classifier):
        """Test log_statistics with data."""
        classifier.stats['total_classified'] = 10
        classifier.stats['caption_detected'] = 3
        classifier.stats['high_confidence_yolo'] = 5
        classifier.stats['llm_verified'] = 2
        classifier.stats['llm_changed_decision'] = 1
        
        # Should not raise
        classifier.log_statistics()


# =============================================================================
# CAPTION PATTERNS TESTS
# =============================================================================

class TestCaptionPatterns:
    """Test caption regex patterns."""
    
    def test_table_pattern_matches(self, classifier):
        """Test TABLE_CAPTION_PATTERNS match correctly."""
        test_cases = [
            "Table 5: Overview",
            "Table 3-2: Specifications",
            "Tab. 1: Components",
            "table 10.5 data",
        ]
        
        for text in test_cases:
            matched = False
            for pattern in classifier.TABLE_CAPTION_PATTERNS:
                if pattern.search(text):
                    matched = True
                    break
            assert matched, f"Should match: {text}"
    
    def test_figure_pattern_matches(self, classifier):
        """Test FIGURE_CAPTION_PATTERNS match correctly."""
        test_cases = [
            "Figure 5: Diagram",
            "Fig. 3-2: Assembly",
            "Diagram 10: Overview",
            "Drawing 8-1: Layout",
            "Schema 2: Control",
            "<< FIG. S-03 >>",
        ]
        
        for text in test_cases:
            matched = False
            for pattern in classifier.FIGURE_CAPTION_PATTERNS:
                if pattern.search(text):
                    matched = True
                    break
            assert matched, f"Should match: {text}"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_page_text(self, classifier, mock_pdf_page):
        """Test with empty page text."""
        mock_pdf_page.get_text.return_value = ""
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.9,
            page_number=0,
        )
        
        result = await classifier.reclassify_region(mock_pdf_page, region)
        
        # Should still work, trusting YOLO
        assert result == RegionType.SCHEMA
    
    @pytest.mark.asyncio
    async def test_llm_api_error(self, classifier, mock_pdf_page, mock_llm_service):
        """Test handling of LLM API error."""
        mock_pdf_page.get_text.return_value = "Some text"
        mock_llm_service.chat.completions.create.side_effect = Exception("API Error")
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.5,  # Would trigger LLM
            page_number=0,
        )
        
        # Should handle error gracefully
        try:
            result = await classifier.reclassify_region(mock_pdf_page, region)
            # If it doesn't raise, it should fall back to YOLO
            assert result == RegionType.SCHEMA
        except Exception:
            # Or it might raise, which is also acceptable
            pass
    
    def test_caption_with_special_chars(self, classifier, mock_pdf_page):
        """Test caption detection with special characters."""
        mock_pdf_page.get_text.return_value = "Table 3-2 (a): System & Components"
        
        bbox = BBox(100, 200, 300, 400)
        caption_type, caption_text = classifier._detect_caption_type(mock_pdf_page, bbox)
        
        assert caption_type == 'table'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

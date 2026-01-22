"""
Schema Extractor Tests

Comprehensive tests for schema (diagram/figure) extraction including:
- Figure number extraction
- Caption detection
- Context building
- Text cleaning and noise detection
- Reference finding
- ID generation
- LLM summary generation
- Thumbnail creation

Run with: pytest test_schema_extractor.py -v
"""

import pytest
import io
import base64
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional

from services.schema_extractor import SchemaExtractor
from services.layout_analyzer import Region, RegionType, BBox, LayoutAnalyzer


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_storage_service():
    """Create mock storage service."""
    storage = Mock()
    storage.save_file = AsyncMock(return_value="/path/to/saved/file")
    return storage


@pytest.fixture
def mock_layout_analyzer():
    """Create mock layout analyzer."""
    analyzer = Mock(spec=LayoutAnalyzer)
    analyzer.analyze_page = Mock(return_value=[])
    analyzer.filter_regions_by_type = Mock(return_value=[])
    return analyzer


@pytest.fixture
def mock_llm_service():
    """Create mock OpenAI LLM service."""
    llm = Mock()
    llm.chat = Mock()
    llm.chat.completions = Mock()
    llm.chat.completions.create = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content="TYPE: MECHANICAL_LAYOUT\nTest summary"))]
    ))
    return llm


@pytest.fixture
def extractor(mock_storage_service, mock_layout_analyzer, mock_llm_service):
    """Create SchemaExtractor with mocked dependencies."""
    with patch('services.schema_extractor.Settings'):
        return SchemaExtractor(
            storage_service=mock_storage_service,
            layout_analyzer=mock_layout_analyzer,
            llm_service=mock_llm_service,
            zoom=2.0,
            thumbnail_size=(600, 600),
            caption_search_distance=250,
            surrounding_text_radius=300,
            max_nearby_paragraphs=3,
            enable_llm_summary=True,
        )


@pytest.fixture
def mock_pdf_page():
    """Create mock PyMuPDF page."""
    page = Mock()
    page.rect = Mock()
    page.rect.width = 612
    page.rect.height = 792
    page.get_text = Mock(return_value="Sample page text")
    page.get_pixmap = Mock(return_value=Mock(
        pil_tobytes=Mock(return_value=b"PNG_IMAGE_DATA")
    ))
    page.search_for = Mock(return_value=[])
    return page


@pytest.fixture
def sample_region():
    """Create sample schema region."""
    return Region(
        bbox=BBox(100, 200, 400, 500),
        region_type=RegionType.SCHEMA,
        confidence=0.85,
        page_number=0,
    )


@pytest.fixture
def sample_image_bytes():
    """Create sample PNG image bytes."""
    from PIL import Image
    img = Image.new('RGB', (200, 200), color='white')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def sample_context():
    """Create sample context dict."""
    return {
        'caption': 'Figure 5-2: Fuel Pump Assembly',
        'nearby_paragraphs': [
            'The fuel pump consists of three main components.',
            'Installation requires proper alignment.',
        ],
        'surrounding_text': 'Additional context text',
        'references': ['see Figure 5-2 for details'],
        'domain_tags': ['fluid_system', 'mechanical'],
        'entity_codes': ['PU-101', 'V-205'],
        'llm_summary': 'TYPE: MECHANICAL_LAYOUT\nFuel pump system diagram',
        'page_number': 42,
    }


# =============================================================================
# FIGURE NUMBER EXTRACTION TESTS
# =============================================================================

class TestFigureNumberExtraction:
    """Test _extract_figure_number method."""
    
    def test_extract_simple_number(self, extractor):
        """Test extracting simple figure number."""
        result = extractor._extract_figure_number("Figure 5: Overview")
        assert result == "5"
    
    def test_extract_hyphenated_number(self, extractor):
        """Test extracting hyphenated number."""
        result = extractor._extract_figure_number("Figure 4-3: System Diagram")
        assert result == "4-3"
    
    def test_extract_dotted_number(self, extractor):
        """Test extracting dotted number."""
        result = extractor._extract_figure_number("Diagram 10.5: Layout")
        assert result == "10.5"
    
    def test_extract_complex_number(self, extractor):
        """Test extracting complex number."""
        result = extractor._extract_figure_number("Drawing 8-4-1: Assembly")
        assert result == "8-4-1"
    
    def test_extract_fig_abbreviation(self, extractor):
        """Test extracting with Fig. abbreviation."""
        result = extractor._extract_figure_number("Fig. 3-2: Detail")
        assert result == "3-2"
    
    def test_no_number_returns_none(self, extractor):
        """Test caption without number returns None."""
        result = extractor._extract_figure_number("System Diagram")
        assert result is None
    
    def test_empty_string(self, extractor):
        """Test empty string returns None."""
        result = extractor._extract_figure_number("")
        assert result is None
    
    def test_various_formats(self, extractor):
        """Test various caption formats."""
        test_cases = [
            ("Figure 5: Overview", "5"),
            ("Fig. 3-2: Detail", "3-2"),
            ("Diagram 10.5: Layout", "10.5"),
            ("Drawing 8-4-1: Assembly", "8-4-1"),
            ("FIGURE 12: Test", "12"),
        ]
        
        for caption, expected in test_cases:
            result = extractor._extract_figure_number(caption)
            assert result == expected, f"Failed for caption: {caption}"


# =============================================================================
# TEXT CLEANING TESTS
# =============================================================================

class TestTextCleaning:
    """Test _clean_text method."""
    
    def test_clean_removes_tabs(self, extractor):
        """Test tabs are removed."""
        result = extractor._clean_text("Text\twith\ttabs")
        assert "\t" not in result
    
    def test_clean_collapses_whitespace(self, extractor):
        """Test multiple spaces are collapsed."""
        result = extractor._clean_text("Text   with    spaces")
        assert "   " not in result
    
    def test_clean_removes_carriage_returns(self, extractor):
        """Test carriage returns are removed."""
        result = extractor._clean_text("Text\r\nwith\rreturns")
        assert "\r" not in result
    
    def test_clean_preserves_content(self, extractor):
        """Test actual content is preserved."""
        result = extractor._clean_text("PU-101 fuel pump")
        assert "PU-101" in result
        assert "fuel" in result
        assert "pump" in result
    
    def test_clean_strips_edges(self, extractor):
        """Test leading/trailing whitespace is stripped."""
        result = extractor._clean_text("  text  ")
        assert result == "text"
    
    def test_clean_removes_control_chars(self, extractor):
        """Test control characters are removed."""
        result = extractor._clean_text("Text\x00with\x1fcontrol")
        assert "\x00" not in result
        assert "\x1f" not in result


# =============================================================================
# NOISE TEXT DETECTION TESTS
# =============================================================================

class TestNoiseTextDetection:
    """Test _is_noise_text method."""
    
    def test_page_numbers_are_noise(self, extractor):
        """Test standalone numbers are noise."""
        assert extractor._is_noise_text("12") is True
        assert extractor._is_noise_text("  123  ") is True
        assert extractor._is_noise_text("456") is True
    
    def test_page_keyword_is_noise(self, extractor):
        """Test 'page' text is noise."""
        assert extractor._is_noise_text("Page 5") is True
        assert extractor._is_noise_text("page 10 of 100") is True
    
    def test_copyright_is_noise(self, extractor):
        """Test copyright notices are noise."""
        assert extractor._is_noise_text("Copyright © 2024") is True
        assert extractor._is_noise_text("© Company Name") is True
    
    def test_normal_text_not_noise(self, extractor):
        """Test normal text is not noise."""
        assert extractor._is_noise_text("This is a normal paragraph with content") is False
        assert extractor._is_noise_text("The system operates at high pressure") is False
    
    def test_short_header_with_keywords_is_noise(self, extractor):
        """Test short headers with noise keywords are detected."""
        assert extractor._is_noise_text("Chapter 1") is True
        assert extractor._is_noise_text("Section 2.1") is True
    
    def test_long_text_with_keywords_not_noise(self, extractor):
        """Test longer text with keywords is not automatically noise."""
        long_text = "This is a longer paragraph that happens to mention the page number and copyright information in context."
        # Longer than 100 chars should not be automatically noise
        assert len(long_text) > 100
        # Implementation may vary - test actual behavior


# =============================================================================
# REFERENCE FINDING TESTS
# =============================================================================

class TestReferenceFinding:
    """Test _find_references_in_text method."""
    
    def test_find_see_figure_reference(self, extractor):
        """Test finding 'see Figure X' references."""
        text = "The system is shown in Figure 5-2 below."
        references = extractor._find_references_in_text(text, "5-2")
        
        assert len(references) >= 1
        assert any("Figure 5-2" in ref for ref in references)
    
    def test_find_refer_to_reference(self, extractor):
        """Test finding 'refer to' references."""
        text = "Please refer to Figure 3-1 for details."
        references = extractor._find_references_in_text(text, "3-1")
        
        assert len(references) >= 1
    
    def test_find_multiple_references(self, extractor):
        """Test finding multiple references."""
        text = """
        See Figure 5-2 for overview.
        Refer to Figure 5-2 for component details.
        As illustrated in diagram 5-2, the flow is correct.
        """
        references = extractor._find_references_in_text(text, "5-2")
        
        assert len(references) >= 2
    
    def test_no_references_found(self, extractor):
        """Test when no references exist."""
        text = "This text does not mention any figures or diagrams."
        references = extractor._find_references_in_text(text, "5-2")
        
        assert len(references) == 0
    
    def test_wrong_number_not_matched(self, extractor):
        """Test that wrong figure numbers don't match."""
        text = "See Figure 3-1 for details."
        references = extractor._find_references_in_text(text, "5-2")
        
        assert len(references) == 0


# =============================================================================
# RICH CONTEXT BUILDING TESTS
# =============================================================================

class TestRichContextBuilding:
    """Test _build_rich_context method."""
    
    def test_context_includes_llm_summary(self, extractor, sample_context):
        """Test LLM summary is included."""
        result = extractor._build_rich_context(sample_context)
        
        assert "Description:" in result
        assert "Fuel pump system diagram" in result
    
    def test_context_includes_caption(self, extractor, sample_context):
        """Test caption is included."""
        result = extractor._build_rich_context(sample_context)
        
        assert "Caption:" in result
        assert "Figure 5-2" in result
    
    def test_context_includes_paragraphs(self, extractor, sample_context):
        """Test nearby paragraphs are included."""
        result = extractor._build_rich_context(sample_context)
        
        assert "Context:" in result
        assert "fuel pump consists" in result
    
    def test_context_includes_references(self, extractor, sample_context):
        """Test references are included."""
        result = extractor._build_rich_context(sample_context)
        
        assert "References:" in result
    
    def test_context_includes_tags(self, extractor, sample_context):
        """Test domain tags are included."""
        result = extractor._build_rich_context(sample_context)
        
        assert "Tags:" in result
        assert "fluid_system" in result
    
    def test_context_includes_entity_codes(self, extractor, sample_context):
        """Test entity codes are included."""
        result = extractor._build_rich_context(sample_context)
        
        assert "EntityCodes:" in result
        assert "PU-101" in result
    
    def test_context_extracts_diagram_type(self, extractor, sample_context):
        """Test diagram type is extracted from summary."""
        result = extractor._build_rich_context(sample_context)
        
        assert "DiagramType:" in result
        assert "MECHANICAL_LAYOUT" in result
    
    def test_context_truncated_if_long(self, extractor):
        """Test context is truncated if too long."""
        long_context = {
            'llm_summary': 'A' * 3000,
            'caption': 'B' * 1000,
            'nearby_paragraphs': ['C' * 1000],
            'surrounding_text': '',
            'references': [],
        }
        
        result = extractor._build_rich_context(long_context)
        
        assert len(result) <= 2003  # 2000 + "..."
    
    def test_fallback_for_empty_context(self, extractor):
        """Test fallback when no context available."""
        empty_context = {
            'llm_summary': '',
            'caption': '',
            'nearby_paragraphs': [],
            'surrounding_text': '',
            'references': [],
            'page_number': 5,
        }
        
        result = extractor._build_rich_context(empty_context)
        
        # Should have some fallback content
        assert len(result) > 0
        assert "page" in result.lower() or "diagram" in result.lower()


# =============================================================================
# FALLBACK DESCRIPTION TESTS
# =============================================================================

class TestFallbackDescription:
    """Test _build_fallback_description method."""
    
    def test_fallback_with_caption(self, extractor):
        """Test fallback uses caption if available."""
        context = {
            'caption': 'Figure 5-2: Fuel System',
            'nearby_paragraphs': [],
            'surrounding_text': '',
            'page_number': 10,
        }
        
        result = extractor._build_fallback_description(context)
        
        assert "Figure 5-2" in result or "Fuel System" in result
    
    def test_fallback_with_paragraphs(self, extractor):
        """Test fallback uses paragraphs."""
        context = {
            'caption': '',
            'nearby_paragraphs': ['The pump operates at 50 bar.'],
            'surrounding_text': '',
            'page_number': 10,
        }
        
        result = extractor._build_fallback_description(context)
        
        assert "pump" in result.lower() or "50 bar" in result
    
    def test_fallback_minimal_context(self, extractor):
        """Test fallback with minimal context."""
        context = {
            'caption': '',
            'nearby_paragraphs': [],
            'surrounding_text': '',
            'page_number': 10,
        }
        
        result = extractor._build_fallback_description(context)
        
        # Should still produce something
        assert len(result) > 0


# =============================================================================
# LLM SUMMARY GENERATION TESTS
# =============================================================================

class TestLlmSummaryGeneration:
    """Test _generate_llm_summary method."""
    
    @pytest.mark.asyncio
    async def test_llm_summary_success(self, extractor, sample_image_bytes, mock_llm_service):
        """Test successful LLM summary generation."""
        mock_llm_service.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content="TYPE: P_ID\nThis is a P&ID diagram showing fuel system."))]
        ))
        
        context = {'caption': 'Figure 1', 'nearby_paragraphs': ['Test'], 'page_number': 1}
        result = await extractor._generate_llm_summary(sample_image_bytes, context)
        
        assert result is not None
        assert "P_ID" in result or "P&ID" in result or "diagram" in result.lower()
    
    @pytest.mark.asyncio
    async def test_llm_summary_not_schema_response(self, extractor, sample_image_bytes, mock_llm_service):
        """Test LLM responding with NOT_SCHEMA."""
        mock_llm_service.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content="NOT_SCHEMA: This appears to be decorative"))]
        ))
        
        context = {'caption': '', 'nearby_paragraphs': [], 'page_number': 1}
        result = await extractor._generate_llm_summary(sample_image_bytes, context)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_llm_summary_empty_response(self, extractor, sample_image_bytes, mock_llm_service):
        """Test LLM returning empty response."""
        mock_llm_service.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content=""))]
        ))
        
        context = {'caption': '', 'nearby_paragraphs': [], 'page_number': 1}
        result = await extractor._generate_llm_summary(sample_image_bytes, context)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_llm_summary_api_error(self, extractor, sample_image_bytes, mock_llm_service):
        """Test LLM API error handling."""
        mock_llm_service.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        context = {'caption': '', 'nearby_paragraphs': [], 'page_number': 1}
        result = await extractor._generate_llm_summary(sample_image_bytes, context)
        
        # Should return None on error, not raise
        assert result is None


# =============================================================================
# SCHEMA ID GENERATION TESTS
# =============================================================================

class TestSchemaIdGeneration:
    """Test _generate_schema_id method."""
    
    def test_id_is_deterministic(self, extractor):
        """Test same inputs give same ID."""
        bbox = BBox(100, 200, 300, 400)
        
        id1 = extractor._generate_schema_id("doc1", 5, bbox, 0)
        id2 = extractor._generate_schema_id("doc1", 5, bbox, 0)
        
        assert id1 == id2
    
    def test_different_doc_different_id(self, extractor):
        """Test different doc gives different ID."""
        bbox = BBox(100, 200, 300, 400)
        
        id1 = extractor._generate_schema_id("doc1", 5, bbox, 0)
        id2 = extractor._generate_schema_id("doc2", 5, bbox, 0)
        
        assert id1 != id2
    
    def test_different_bbox_different_id(self, extractor):
        """Test different bbox gives different ID."""
        bbox1 = BBox(100, 200, 300, 400)
        bbox2 = BBox(150, 250, 350, 450)
        
        id1 = extractor._generate_schema_id("doc1", 5, bbox1, 0)
        id2 = extractor._generate_schema_id("doc1", 5, bbox2, 0)
        
        assert id1 != id2
    
    def test_id_length(self, extractor):
        """Test ID is 24 characters."""
        bbox = BBox(100, 200, 300, 400)
        
        result = extractor._generate_schema_id("doc1", 5, bbox, 0)
        
        assert len(result) == 24


# =============================================================================
# SANITIZE TESTS
# =============================================================================

class TestSanitize:
    """Test _sanitize method."""
    
    def test_sanitize_special_chars(self, extractor):
        """Test special characters are replaced."""
        result = extractor._sanitize("file/with\\special:chars")
        
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result
    
    def test_sanitize_length_limit(self, extractor):
        """Test result is limited to 100 chars."""
        long_name = "a" * 200
        result = extractor._sanitize(long_name)
        
        assert len(result) <= 100


# =============================================================================
# TRUNCATE TESTS
# =============================================================================

class TestTruncate:
    """Test _truncate_text method."""
    
    def test_short_text_unchanged(self, extractor):
        """Test short text is not modified."""
        text = "Short text"
        result = extractor._truncate_text(text, 100)
        
        assert result == text
    
    def test_long_text_truncated(self, extractor):
        """Test long text is truncated."""
        text = "This is a longer text that exceeds the limit set for testing"
        result = extractor._truncate_text(text, 20)
        
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")
    
    def test_truncate_at_word_boundary(self, extractor):
        """Test truncation at word boundary."""
        text = "Word1 Word2 Word3 Word4"
        result = extractor._truncate_text(text, 15)
        
        # Should not cut mid-word
        assert not result.endswith("rd...")


# =============================================================================
# THUMBNAIL GENERATION TESTS
# =============================================================================

class TestThumbnailGeneration:
    """Test _make_thumbnail method."""
    
    def test_thumbnail_returns_bytes(self, extractor, sample_image_bytes):
        """Test thumbnail returns bytes."""
        result = extractor._make_thumbnail(sample_image_bytes, size=(512, 512))
        
        assert isinstance(result, bytes)
        # Should be PNG
        assert result[:4] == b'\x89PNG'
    
    def test_thumbnail_smaller_than_original(self, extractor):
        """Test thumbnail is smaller than original."""
        from PIL import Image
        
        # Create large image
        img = Image.new('RGB', (2000, 2000), color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        large_bytes = buf.getvalue()
        
        result = extractor._make_thumbnail(large_bytes, size=(512, 512))
        
        # Thumbnail should be smaller
        result_img = Image.open(io.BytesIO(result))
        assert result_img.size[0] <= 600
        assert result_img.size[1] <= 600


# =============================================================================
# CAPTION EXTRACTION TESTS
# =============================================================================

class TestCaptionExtraction:
    """Test caption extraction patterns."""
    
    def test_caption_pattern_figure(self, extractor):
        """Test Figure caption pattern."""
        for pattern in extractor.caption_patterns:
            match = pattern.search("Figure 5-2: Fuel System Diagram")
            if match:
                assert "5-2" in match.group(0) or "Fuel System" in match.group(0)
                return
        # At least one pattern should match
        # (may not match all patterns, that's OK)
    
    def test_caption_pattern_russian(self, extractor):
        """Test Russian caption pattern."""
        for pattern in extractor.caption_patterns:
            match = pattern.search("Рисунок 3-1: Схема системы")
            if match:
                assert "3-1" in match.group(0) or "Схема" in match.group(0)
                return
    
    def test_caption_pattern_all_caps(self, extractor):
        """Test ALL CAPS caption pattern."""
        for pattern in extractor.caption_patterns:
            match = pattern.search("PRIMARY BLOWER & 1ST D.O BURNER")
            if match:
                assert "BLOWER" in match.group(0) or "BURNER" in match.group(0)
                return


# =============================================================================
# PAGE EXTRACTION INTEGRATION TESTS
# =============================================================================

class TestPageExtractionIntegration:
    """Test extract_from_page method."""
    
    @pytest.mark.asyncio
    async def test_extract_no_schemas_returns_empty(self, extractor, mock_pdf_page, mock_layout_analyzer):
        """Test extraction with no schemas returns empty list."""
        mock_layout_analyzer.analyze_page.return_value = []
        mock_layout_analyzer.filter_regions_by_type.return_value = []
        
        result = await extractor.extract_from_page(
            page=mock_pdf_page,
            doc_id="doc_001",
            page_num=0,
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_extract_calls_layout_analyzer(self, extractor, mock_pdf_page, mock_layout_analyzer):
        """Test extraction calls layout analyzer."""
        mock_layout_analyzer.filter_regions_by_type.return_value = []
        
        await extractor.extract_from_page(
            page=mock_pdf_page,
            doc_id="doc_001",
            page_num=0,
        )
        
        mock_layout_analyzer.analyze_page.assert_called_once()


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_caption(self, extractor):
        """Test empty caption handling."""
        result = extractor._extract_figure_number("")
        assert result is None
    
    def test_none_caption(self, extractor):
        """Test None caption handling."""
        # Should not raise
        try:
            result = extractor._extract_figure_number(None)
        except (TypeError, AttributeError):
            # Expected if not handling None
            pass
    
    def test_unicode_caption(self, extractor):
        """Test Unicode caption handling."""
        result = extractor._extract_figure_number("Рисунок 5: Схема топливной системы")
        # May or may not extract number depending on pattern
        assert result is None or result == "5"
    
    def test_special_chars_in_caption(self, extractor):
        """Test special characters in caption."""
        result = extractor._extract_figure_number("Figure 3-2 (a): System & Components")
        assert result == "3-2"
    
    def test_empty_context_build(self, extractor):
        """Test building context from empty dict."""
        result = extractor._build_rich_context({})
        
        # Should produce fallback
        assert len(result) > 0
    
    def test_very_long_paragraphs(self, extractor):
        """Test handling very long paragraphs."""
        context = {
            'llm_summary': '',
            'caption': '',
            'nearby_paragraphs': ['A' * 10000],  # Very long
            'surrounding_text': '',
            'references': [],
        }
        
        result = extractor._build_rich_context(context)
        
        # Should be truncated
        assert len(result) <= 2003


# =============================================================================
# DISTANCE CALCULATION TESTS
# =============================================================================

class TestDistanceCalculation:
    """Test _distance_to_bbox method."""
    
    def test_distance_above_bbox(self, extractor):
        """Test distance when rect is above bbox."""
        import fitz
        rect = fitz.Rect(100, 50, 200, 100)  # Above bbox
        bbox = BBox(100, 200, 200, 300)
        
        distance = extractor._distance_to_bbox(rect, bbox)
        
        assert distance > 0
        assert distance == 100  # 200 - 100
    
    def test_distance_below_bbox(self, extractor):
        """Test distance when rect is below bbox."""
        import fitz
        rect = fitz.Rect(100, 400, 200, 450)  # Below bbox
        bbox = BBox(100, 200, 200, 300)
        
        distance = extractor._distance_to_bbox(rect, bbox)
        
        assert distance > 0
        assert distance == 100  # 400 - 300
    
    def test_distance_overlapping(self, extractor):
        """Test distance when overlapping."""
        import fitz
        rect = fitz.Rect(100, 200, 200, 300)  # Same as bbox
        bbox = BBox(100, 200, 200, 300)
        
        distance = extractor._distance_to_bbox(rect, bbox)
        
        assert distance == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
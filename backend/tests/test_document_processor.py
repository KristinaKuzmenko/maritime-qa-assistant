"""
Document Processor Tests - Comprehensive Coverage

Tests for document processing logic including:
- Tag generation
- TOC parsing and level detection
- Section header detection
- Text chunking
- Importance scoring
- File hashing
- Title cleaning
- Sanitization
- Section/Chapter number extraction
- YOLO caption linking

Run with: pytest test_document_processor.py -v
"""

import pytest
import hashlib
import tempfile
import os
import re
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict

from services.document_processor import DocumentProcessor


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_services():
    """Create mock services for DocumentProcessor."""
    return {
        "graph_client": Mock(),
        "layout_analyzer": Mock(),
        "schema_extractor": Mock(),
        "table_extractor": Mock(),
        "embedding_service": Mock(),
        "storage_service": Mock(),
        "vector_service": Mock(),
    }


@pytest.fixture
def processor(mock_services):
    """Create DocumentProcessor with mocked dependencies."""
    with patch('services.document_processor.get_entity_extractor') as mock_entity:
        mock_entity.return_value = Mock()
        
        with patch('core.config.Settings') as mock_settings:
            mock_settings.return_value.vision_detail_tables = "low"
            
            return DocumentProcessor(**mock_services)


@pytest.fixture
def mock_pdf_page():
    """Mock PyMuPDF page object."""
    page = Mock()
    page.rect = Mock()
    page.rect.width = 612
    page.rect.height = 792
    page.get_text = Mock(return_value="Default page text")
    return page


# =============================================================================
# FILE HASH TESTS
# =============================================================================

class TestFileHash:
    """Test _calculate_file_hash method."""
    
    def test_hash_returns_sha256(self, processor):
        """Test hash is SHA256 format."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            
            try:
                result = processor._calculate_file_hash(f.name)
                
                # SHA256 is 64 hex characters
                assert len(result) == 64
                assert all(c in '0123456789abcdef' for c in result)
            finally:
                os.unlink(f.name)
    
    def test_hash_deterministic(self, processor):
        """Test same content gives same hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content for hashing")
            f.flush()
            
            try:
                hash1 = processor._calculate_file_hash(f.name)
                hash2 = processor._calculate_file_hash(f.name)
                
                assert hash1 == hash2
            finally:
                os.unlink(f.name)
    
    def test_hash_different_content(self, processor):
        """Test different content gives different hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"content one")
            f1.flush()
            
            with tempfile.NamedTemporaryFile(delete=False) as f2:
                f2.write(b"content two")
                f2.flush()
                
                try:
                    hash1 = processor._calculate_file_hash(f1.name)
                    hash2 = processor._calculate_file_hash(f2.name)
                    
                    assert hash1 != hash2
                finally:
                    os.unlink(f1.name)
                    os.unlink(f2.name)
    
    def test_hash_empty_file(self, processor):
        """Test hashing empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.flush()  # Empty file
            
            try:
                result = processor._calculate_file_hash(f.name)
                
                # Should return valid hash even for empty file
                assert len(result) == 64
            finally:
                os.unlink(f.name)
    
    def test_hash_large_file(self, processor):
        """Test hashing larger file (chunked reading)."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write > 4096 bytes to test chunked reading
            f.write(b"x" * 10000)
            f.flush()
            
            try:
                result = processor._calculate_file_hash(f.name)
                
                assert len(result) == 64
            finally:
                os.unlink(f.name)


# =============================================================================
# SANITIZE TESTS
# =============================================================================

class TestSanitize:
    """Test _sanitize method."""
    
    def test_sanitize_removes_special_chars(self, processor):
        """Test special characters are replaced."""
        result = processor._sanitize("file/name:with*special?chars")
        
        assert "/" not in result
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result
    
    def test_sanitize_preserves_alphanumeric(self, processor):
        """Test alphanumeric characters are preserved."""
        result = processor._sanitize("Normal_File-Name.pdf")
        
        assert "Normal" in result
        assert "File" in result
        assert "Name" in result
        assert ".pdf" in result
    
    def test_sanitize_preserves_underscore_hyphen_dot(self, processor):
        """Test underscore, hyphen, and dot are preserved."""
        result = processor._sanitize("file_name-v1.0")
        
        assert "_" in result
        assert "-" in result
        assert "." in result
    
    def test_sanitize_truncates_to_100_chars(self, processor):
        """Test long strings are truncated."""
        long_name = "a" * 200
        result = processor._sanitize(long_name)
        
        assert len(result) <= 100
    
    def test_sanitize_replaces_spaces(self, processor):
        """Test spaces are replaced with underscore."""
        result = processor._sanitize("file name with spaces")
        
        assert " " not in result
        assert "_" in result
    
    def test_sanitize_unicode(self, processor):
        """Test Unicode characters are handled."""
        result = processor._sanitize("файл_文件_αρχείο")
        
        # Should not crash, result should be valid
        assert isinstance(result, str)


# =============================================================================
# TAG GENERATION TESTS
# =============================================================================

class TestTagGeneration:
    """Test _generate_tags method."""
    
    def test_schema_base_tag(self, processor):
        """Test schema gets 'diagram' base tag."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="A simple diagram"
        )
        assert "diagram" in tags
    
    def test_table_base_tag(self, processor):
        """Test table gets 'table' base tag."""
        tags = processor._generate_tags(
            content_type="table",
            text_context="Some table content"
        )
        assert "table" in tags
    
    def test_schema_pid_tag(self, processor):
        """Test P&ID detection in schema."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="This is a P&ID diagram showing piping and instrumentation"
        )
        assert "P&ID" in tags
    
    def test_schema_electrical_tag(self, processor):
        """Test electrical diagram detection."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Electrical wiring schematic for motor control circuit"
        )
        assert "electrical" in tags
    
    def test_schema_hydraulic_tag(self, processor):
        """Test hydraulic/pneumatic detection."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Hydraulic system diagram with pneumatic valves"
        )
        assert "hydraulic-pneumatic" in tags
    
    def test_schema_flowchart_tag(self, processor):
        """Test flowchart detection."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Process flow chart for fuel treatment"
        )
        assert "flowchart" in tags
    
    def test_schema_assembly_tag(self, processor):
        """Test assembly/exploded view detection."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Exploded view assembly of pump components"
        )
        assert "assembly" in tags
    
    def test_table_specifications_tag(self, processor):
        """Test specifications table detection."""
        tags = processor._generate_tags(
            content_type="table",
            text_context="Technical specifications: pressure 50 bar, flow 100 L/min"
        )
        assert "specifications" in tags
    
    def test_table_parts_list_tag(self, processor):
        """Test parts list detection."""
        tags = processor._generate_tags(
            content_type="table",
            text_context="Parts list: Item 1, Item 2, Spare parts for pump"
        )
        assert "parts-list" in tags
    
    def test_table_maintenance_tag(self, processor):
        """Test maintenance schedule detection."""
        tags = processor._generate_tags(
            content_type="table",
            text_context="Maintenance schedule: daily inspection, weekly cleaning"
        )
        assert "maintenance" in tags
    
    def test_fuel_system_tag(self, processor):
        """Test fuel system detection."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Fuel oil transfer system with diesel pump"
        )
        assert "fuel-system" in tags
    
    def test_cooling_system_tag(self, processor):
        """Test cooling system detection."""
        tags = processor._generate_tags(
            content_type="table",
            text_context="Cooling water pump specifications, heat exchanger data"
        )
        assert "cooling-system" in tags
    
    def test_lubrication_system_tag(self, processor):
        """Test lubrication system detection."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Lube oil system diagram with lub oil cooler"
        )
        assert "lubrication-system" in tags
    
    def test_pump_tag(self, processor):
        """Test pump detection."""
        tags = processor._generate_tags(
            content_type="table",
            text_context="Pump PU-101 specifications, pumping capacity"
        )
        assert "pump" in tags
    
    def test_valve_tag(self, processor):
        """Test valve detection."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Valve arrangement showing multiple valves"
        )
        assert "valve" in tags
    
    def test_llm_tags_included(self, processor):
        """Test that LLM-provided tags are included."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Some diagram",
            llm_tags=["custom-tag", "another-tag"]
        )
        assert "custom-tag" in tags
        assert "another-tag" in tags
    
    def test_multiple_tags_combined(self, processor):
        """Test multiple tags from same content."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary="Fuel oil pump hydraulic system diagram with valves"
        )
        assert "diagram" in tags
        assert "fuel-system" in tags
        assert "pump" in tags
        assert "valve" in tags


# =============================================================================
# TOC LEVEL DETECTION TESTS
# =============================================================================

class TestTocLevelDetection:
    """Test _determine_toc_level method."""
    
    def test_chapter_keyword_level_1(self, processor):
        """Test CHAPTER keyword is level 1."""
        result = processor._determine_toc_level("CHAPTER 5 - Engine Description")
        assert result == 1
    
    def test_part_keyword_level_1(self, processor):
        """Test PART keyword is level 1."""
        result = processor._determine_toc_level("PART A - General Information")
        assert result == 1
    
    def test_appendix_keyword_level_1(self, processor):
        """Test APPENDIX keyword is level 1."""
        result = processor._determine_toc_level("APPENDIX B - Technical Data")
        assert result == 1
    
    def test_all_caps_short_title_level_1(self, processor):
        """Test ALL CAPS short title is level 1."""
        result = processor._determine_toc_level("SAFETY PRECAUTIONS")
        assert result == 1
    
    def test_numbered_section_level_by_dots(self, processor):
        """Test numbered section level based on dot count."""
        # "1" = level 1
        assert processor._determine_toc_level("1 Introduction") == 1
        # "1.2" = level 2
        assert processor._determine_toc_level("1.2 System Overview") == 2
        # "1.2.3" = level 3
        assert processor._determine_toc_level("1.2.3 Detailed Specs") == 3
    
    def test_roman_numeral_level_1(self, processor):
        """Test Roman numeral at start is level 1."""
        result = processor._determine_toc_level("III. System Description")
        assert result == 1
    
    def test_technical_code_filtered(self, processor):
        """Test technical codes are assigned low level."""
        result = processor._determine_toc_level("FIG-3 Diagram Reference")
        assert result == 3
    
    def test_ends_with_number_filtered(self, processor):
        """Test entries ending with (1) are filtered."""
        result = processor._determine_toc_level("Some Reference (1)")
        assert result == 3


# =============================================================================
# PARSE TOC PAGE TESTS
# =============================================================================

class TestParseTocPage:
    """Test _parse_toc_page method."""
    
    def test_parse_dotted_lines(self, processor):
        """Test parsing dotted line format."""
        text = """
TABLE OF CONTENTS
Introduction .............. 5
Safety Precautions ....... 10
System Description ....... 15
        """
        
        entries = processor._parse_toc_page(text, page_num=0)
        
        assert len(entries) >= 3
        for entry in entries:
            assert "title" in entry
            assert "page" in entry
            assert "level" in entry
    
    def test_parse_numbered_format(self, processor):
        """Test parsing numbered format."""
        text = """
CONTENTS
1.0 Introduction                    5
2.0 Safety Precautions             10
3.0 Installation                   15
        """
        
        entries = processor._parse_toc_page(text, page_num=0)
        
        assert len(entries) >= 1
    
    def test_parse_skips_header(self, processor):
        """Test TOC header is skipped."""
        text = """
TABLE OF CONTENTS
Chapter 1 - Introduction .... 5
        """
        
        entries = processor._parse_toc_page(text, page_num=0)
        
        for entry in entries:
            assert "TABLE OF CONTENTS" not in entry.get("title", "")
    
    def test_parse_empty_page(self, processor):
        """Test parsing empty page."""
        entries = processor._parse_toc_page("", page_num=0)
        
        assert entries == []
    
    def test_parse_assigns_levels(self, processor):
        """Test levels are assigned to entries."""
        text = """
CHAPTER 1 - Introduction .... 5
1.1 Overview ................ 6
1.2 Scope ................... 7
        """
        
        entries = processor._parse_toc_page(text, page_num=0)
        
        levels = [e["level"] for e in entries]
        assert len(set(levels)) > 0


# =============================================================================
# SECTION NUMBER EXTRACTION TESTS
# =============================================================================

class TestSectionNumberExtraction:
    """Test _extract_section_number method."""
    
    def test_simple_number(self, processor):
        """Test simple section number."""
        result = processor._extract_section_number("5 Introduction")
        assert result == "5"
    
    def test_dotted_number(self, processor):
        """Test dotted section number."""
        result = processor._extract_section_number("3.2.1 System Overview")
        assert result == "3.2.1"
    
    def test_deep_nesting(self, processor):
        """Test deeply nested number."""
        result = processor._extract_section_number("1.2.3.4 Detailed Specs")
        assert result == "1.2.3.4"
    
    def test_no_number(self, processor):
        """Test line without number."""
        result = processor._extract_section_number("Introduction")
        assert result == ""
    
    def test_number_with_text_after(self, processor):
        """Test number followed by text."""
        result = processor._extract_section_number("10.5 Operation and Maintenance")
        assert result == "10.5"


# =============================================================================
# CHAPTER NUMBER EXTRACTION TESTS
# =============================================================================

class TestChapterNumberExtraction:
    """Test _extract_chapter_number method."""
    
    def test_chapter_number(self, processor):
        """Test CHAPTER number extraction."""
        result = processor._extract_chapter_number("CHAPTER 5 - Engine Description")
        assert result == "5"
    
    def test_part_letter(self, processor):
        """Test PART letter extraction."""
        result = processor._extract_chapter_number("PART A - General Information")
        assert result == "A"
    
    def test_section_number(self, processor):
        """Test SECTION number extraction."""
        result = processor._extract_chapter_number("SECTION 3 - Installation")
        assert result == "3"
    
    def test_appendix_letter(self, processor):
        """Test APPENDIX letter extraction."""
        result = processor._extract_chapter_number("APPENDIX B - Technical Data")
        assert result == "B"
    
    def test_no_chapter_keyword(self, processor):
        """Test title without chapter keyword."""
        result = processor._extract_chapter_number("Introduction to Systems")
        assert result == ""
    
    def test_case_insensitive(self, processor):
        """Test case insensitivity."""
        result = processor._extract_chapter_number("chapter 7 - safety")
        assert result == "7"


# =============================================================================
# TITLE CLEANING TESTS
# =============================================================================

class TestTitleCleaning:
    """Test _clean_section_title method."""
    
    def test_removes_page_numbers(self, processor):
        """Test page number removal."""
        result = processor._clean_section_title(
            "Operation on max. 0.50% sulphur fuels. Page 4 of 7"
        )
        assert "Page 4 of 7" not in result
        assert "Operation" in result
    
    def test_removes_fraction_format(self, processor):
        """Test (X/Y) format removal."""
        result = processor._clean_section_title(
            "System Description (5/10)"
        )
        assert "(5/10)" not in result
    
    def test_truncates_at_sentence(self, processor):
        """Test truncation at first sentence."""
        result = processor._clean_section_title(
            "Introduction. This is additional text that should be removed."
        )
        assert result == "Introduction."
    
    def test_truncates_long_titles(self, processor):
        """Test truncation of very long titles."""
        long_title = "This is a very long title " * 10
        result = processor._clean_section_title(long_title)
        
        assert len(result) <= 83
    
    def test_preserves_short_titles(self, processor):
        """Test short titles are preserved."""
        result = processor._clean_section_title("Short Title")
        assert result == "Short Title"


# =============================================================================
# SECTION HEADER DETECTION TESTS
# =============================================================================

class TestSectionHeaderDetection:
    """Test _is_section_header method."""
    
    def test_numbered_section(self, processor):
        """Test numbered section is detected."""
        assert processor._is_section_header("3.2.1 System Overview") is True
    
    def test_chapter_header(self, processor):
        """Test CHAPTER header is detected."""
        assert processor._is_section_header("CHAPTER 5 - Engine Description") is True
    
    def test_all_caps_header(self, processor):
        """Test ALL CAPS header is detected."""
        assert processor._is_section_header("SAFETY PRECAUTIONS AND WARNINGS") is True
    
    def test_colon_header(self, processor):
        """Test header with colon is detected."""
        assert processor._is_section_header("Description:") is True
    
    def test_keyword_header(self, processor):
        """Test keyword headers are detected."""
        assert processor._is_section_header("INTRODUCTION") is True
        assert processor._is_section_header("MAINTENANCE") is True
        assert processor._is_section_header("SPECIFICATIONS") is True
    
    def test_too_short_rejected(self, processor):
        """Test too short lines are rejected."""
        assert processor._is_section_header("Ab") is False
    
    def test_too_long_rejected(self, processor):
        """Test too long lines are rejected."""
        long_line = "A" * 200
        assert processor._is_section_header(long_line) is False
    
    def test_regular_text_rejected(self, processor):
        """Test regular text is rejected."""
        assert processor._is_section_header("This is regular paragraph text.") is False


# =============================================================================
# SECTION TYPE CLASSIFICATION TESTS
# =============================================================================

class TestSectionTypeClassification:
    """Test _classify_section_type method."""
    
    def test_warning_type(self, processor):
        """Test warning content detection."""
        result = processor._classify_section_type(
            "WARNING: Do not operate without proper training."
        )
        assert result == "warning"
    
    def test_danger_type(self, processor):
        """Test danger content detection."""
        result = processor._classify_section_type(
            "DANGER: High voltage equipment."
        )
        assert result == "warning"
    
    def test_caution_type(self, processor):
        """Test caution content detection."""
        result = processor._classify_section_type(
            "CAUTION: Handle with care."
        )
        assert result == "warning"
    
    def test_note_type(self, processor):
        """Test note content detection."""
        result = processor._classify_section_type(
            "Note: This applies to all models."
        )
        assert result == "note"
    
    def test_table_type(self, processor):
        """Test table content detection."""
        result = processor._classify_section_type(
            "| Column 1 | Column 2 | Column 3 |"
        )
        assert result == "table"
    
    def test_list_type_numbered(self, processor):
        """Test numbered list detection."""
        result = processor._classify_section_type(
            "1. First item\n2. Second item\n3. Third item"
        )
        assert result == "list"
    
    def test_list_type_bullet(self, processor):
        """Test bullet list detection."""
        result = processor._classify_section_type(
            "• First item\n• Second item"
        )
        assert result == "list"
    
    def test_text_type_default(self, processor):
        """Test default text type."""
        result = processor._classify_section_type(
            "This is regular paragraph text without special markers."
        )
        assert result == "text"


# =============================================================================
# IMPORTANCE SCORING TESTS
# =============================================================================

class TestImportanceScoring:
    """Test _calculate_importance_score method."""
    
    def test_warning_increases_score(self, processor):
        """Test warning keyword increases score."""
        score_with = processor._calculate_importance_score(
            "WARNING: Critical information here."
        )
        score_without = processor._calculate_importance_score(
            "Regular information here."
        )
        
        assert score_with > score_without
    
    def test_procedure_increases_score(self, processor):
        """Test procedure keyword increases score."""
        score = processor._calculate_importance_score(
            "Procedure: Step 1, Step 2, Step 3"
        )
        
        assert score >= 0.5
    
    def test_score_bounded(self, processor):
        """Test score is between 0 and 1."""
        score = processor._calculate_importance_score(
            "Random content without keywords"
        )
        
        assert 0.0 <= score <= 1.0
    
    def test_multiple_keywords_cumulative(self, processor):
        """Test multiple keywords have cumulative effect."""
        score_single = processor._calculate_importance_score(
            "WARNING: Be careful."
        )
        score_multiple = processor._calculate_importance_score(
            "WARNING: CAUTION: DANGER: Be very careful!"
        )
        
        assert score_multiple >= score_single


# =============================================================================
# TEXT CHUNKING TESTS
# =============================================================================

class TestTextChunking:
    """Test _create_text_chunks method."""
    
    def test_short_text_single_chunk(self, processor):
        """Test short text creates single chunk."""
        chunks = processor._create_text_chunks(
            text="Short text"
        )
        
        assert len(chunks) == 1
    
    def test_long_text_multiple_chunks(self, processor):
        """Test long text creates multiple chunks."""
        long_text = "Word " * 500
        
        chunks = processor._create_text_chunks(
            text=long_text
        )
        
        assert len(chunks) > 1
    
    def test_chunk_has_required_fields(self, processor):
        """Test chunk has all required fields."""
        chunks = processor._create_text_chunks(
            text="Test content"
        )
        
        chunk = chunks[0]
        assert "text" in chunk
        assert "char_start" in chunk
        assert "char_end" in chunk
    
    def test_chunks_have_indices(self, processor):
        """Test chunks have sequential indices."""
        long_text = "Word " * 500
        
        chunks = processor._create_text_chunks(
            text=long_text
        )
        
        # Verify chunks have char positions
        for chunk in chunks:
            assert "char_start" in chunk
            assert "char_end" in chunk
            assert chunk["char_start"] < chunk["char_end"]
    
    def test_empty_text(self, processor):
        """Test empty text handling."""
        chunks = processor._create_text_chunks(
            text=""
        )
        
        assert isinstance(chunks, list)


# =============================================================================
# STABLE ID GENERATION TESTS
# =============================================================================

class TestStableIdGeneration:
    """Test stable ID generation methods."""
    
    def test_stable_section_id_deterministic(self, processor):
        """Test section ID is deterministic."""
        id1 = processor._stable_section_id("doc1", "ch1", "1.2", 5, 10)
        id2 = processor._stable_section_id("doc1", "ch1", "1.2", 5, 10)
        
        assert id1 == id2
    
    def test_stable_section_id_different_inputs(self, processor):
        """Test different inputs give different IDs."""
        id1 = processor._stable_section_id("doc1", "ch1", "1.2", 5, 10)
        id2 = processor._stable_section_id("doc1", "ch1", "1.3", 5, 10)
        
        assert id1 != id2
    
    def test_stable_chapter_id_deterministic(self, processor):
        """Test chapter ID is deterministic."""
        id1 = processor._stable_chapter_id("doc1", "Introduction", 1)
        id2 = processor._stable_chapter_id("doc1", "Introduction", 1)
        
        assert id1 == id2
    
    def test_stable_chapter_id_different_titles(self, processor):
        """Test different titles give different IDs."""
        id1 = processor._stable_chapter_id("doc1", "Introduction", 1)
        id2 = processor._stable_chapter_id("doc1", "Safety", 1)
        
        assert id1 != id2


# =============================================================================
# TOC PAGE DETECTION TESTS
# =============================================================================

class TestTocPageDetection:
    """Test is_toc_page method."""
    
    def test_toc_page_in_set(self, processor):
        """Test page in TOC set returns True."""
        processor._toc_pages = {0, 1, 2}
        
        assert processor.is_toc_page(0) is True
        assert processor.is_toc_page(1) is True
        assert processor.is_toc_page(2) is True
    
    def test_non_toc_page(self, processor):
        """Test page not in TOC set returns False."""
        processor._toc_pages = {0, 1}
        
        assert processor.is_toc_page(5) is False
        assert processor.is_toc_page(10) is False


# =============================================================================
# YOLO CAPTION LINKING TESTS
# =============================================================================

class TestYoloCaptionLinking:
    """Test _link_yolo_captions_to_schemas method."""
    
    def test_link_caption_above_schema(self, processor, mock_pdf_page):
        """Test linking caption above schema."""
        from services.layout_analyzer import Region, RegionType, BBox
        
        caption_region = Region(
            bbox=BBox(100, 100, 300, 130),
            region_type=RegionType.TEXT,
            confidence=0.8,
            page_number=0,
            yolo_class_id=0,
        )
        
        schema_region = Region(
            bbox=BBox(100, 150, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.9,
            page_number=0,
            yolo_class_id=6,
        )
        
        mock_pdf_page.get_text.return_value = "Figure 5-2: System Overview"
        
        regions = [caption_region, schema_region]
        processor._link_yolo_captions_to_schemas(mock_pdf_page, regions)
        
        assert schema_region.caption_text is not None
        assert "Figure" in schema_region.caption_text
    
    def test_no_link_when_too_far(self, processor, mock_pdf_page):
        """Test no linking when caption too far."""
        from services.layout_analyzer import Region, RegionType, BBox
        
        caption_region = Region(
            bbox=BBox(100, 100, 300, 130),
            region_type=RegionType.TEXT,
            confidence=0.8,
            page_number=0,
            yolo_class_id=0,
        )
        
        schema_region = Region(
            bbox=BBox(100, 350, 300, 600),
            region_type=RegionType.SCHEMA,
            confidence=0.9,
            page_number=0,
            yolo_class_id=6,
        )
        
        mock_pdf_page.get_text.return_value = "Some caption"
        
        regions = [caption_region, schema_region]
        processor._link_yolo_captions_to_schemas(mock_pdf_page, regions)
        
        assert schema_region.caption_text is None
    
    def test_no_link_when_already_has_caption(self, processor, mock_pdf_page):
        """Test no linking when schema already has caption."""
        from services.layout_analyzer import Region, RegionType, BBox
        
        caption_region = Region(
            bbox=BBox(100, 100, 300, 130),
            region_type=RegionType.TEXT,
            confidence=0.8,
            page_number=0,
            yolo_class_id=0,
        )
        
        schema_region = Region(
            bbox=BBox(100, 150, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.9,
            page_number=0,
            yolo_class_id=6,
            caption_text="Existing caption",
        )
        
        mock_pdf_page.get_text.return_value = "New caption"
        
        regions = [caption_region, schema_region]
        processor._link_yolo_captions_to_schemas(mock_pdf_page, regions)
        
        assert schema_region.caption_text == "Existing caption"
    
    def test_no_captions_no_error(self, processor, mock_pdf_page):
        """Test no error when no caption regions."""
        from services.layout_analyzer import Region, RegionType, BBox
        
        schema_region = Region(
            bbox=BBox(100, 150, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.9,
            page_number=0,
            yolo_class_id=6,
        )
        
        regions = [schema_region]
        
        processor._link_yolo_captions_to_schemas(mock_pdf_page, regions)
    
    def test_no_schemas_no_error(self, processor, mock_pdf_page):
        """Test no error when no schema regions."""
        from services.layout_analyzer import Region, RegionType, BBox
        
        caption_region = Region(
            bbox=BBox(100, 100, 300, 130),
            region_type=RegionType.TEXT,
            confidence=0.8,
            page_number=0,
            yolo_class_id=0,
        )
        
        regions = [caption_region]
        
        processor._link_yolo_captions_to_schemas(mock_pdf_page, regions)


# =============================================================================
# REGEX PATTERN TESTS
# =============================================================================

class TestRegexPatterns:
    """Test regex patterns used for structure detection."""
    
    def test_chapter_pattern(self, processor):
        """Test chapter pattern matches correctly."""
        matches = [
            "CHAPTER 1 - Introduction",
            "Chapter 2 - Safety Precautions",
            "PART A - Overview",
            "SECTION 3 - Installation",
            "APPENDIX B - Specifications",
        ]
        
        for text in matches:
            assert processor.chapter_pattern.search(text), f"Should match: {text}"
    
    def test_section_pattern(self, processor):
        """Test section pattern matches numbered sections."""
        matches = [
            "1.2 System Overview",
            "3.4.5 Detailed Procedure",
            "10.1 Introduction",
        ]
        
        for text in matches:
            assert processor.section_pattern.search(text), f"Should match: {text}"
    
    def test_reference_pattern(self, processor):
        """Test reference pattern matches cross-references."""
        matches = [
            "see Figure 1.2",
            "refer to Table 3",
            "as shown in diagram 5",
            "see schema 2.1",
        ]
        
        for text in matches:
            assert processor.reference_pattern.search(text), f"Should match: {text}"
    
    def test_toc_header_pattern(self, processor):
        """Test TOC header pattern."""
        matches = [
            "TABLE OF CONTENTS",
            "CONTENTS",
            "Table of Contents",
            "INDEX",
        ]
        
        for text in matches:
            assert processor.toc_header_pattern.search(text), f"Should match: {text}"
    
    def test_toc_entry_pattern(self, processor):
        """Test TOC entry pattern."""
        matches = [
            "Introduction .............. 5",
            "Safety ....... 10",
        ]
        
        for text in matches:
            assert processor.toc_entry_pattern.search(text), f"Should match: {text}"


# =============================================================================
# DUAL EXTRACTION LOGIC TESTS
# =============================================================================

class TestDualExtraction:
    """Test dual extraction logic (extract schema AND text)."""
    
    def test_dual_extraction_flag(self):
        """Test that extract_text_also flag works."""
        from services.layout_analyzer import Region, RegionType, BBox
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.65,
            page_number=0,
            extract_text_also=True,
        )
        
        assert region.extract_text_also is True
    
    def test_dual_extraction_default_false(self):
        """Test extract_text_also defaults to False."""
        from services.layout_analyzer import Region, RegionType, BBox
        
        region = Region(
            bbox=BBox(100, 200, 300, 400),
            region_type=RegionType.SCHEMA,
            confidence=0.65,
            page_number=0,
        )
        
        assert region.extract_text_also is False


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_content_handling(self, processor):
        """Test handling of empty content."""
        tags = processor._generate_tags(
            content_type="schema",
            llm_summary=""
        )
        assert "diagram" in tags
    
    def test_none_content_handling(self, processor):
        """Test handling of None values."""
        tags = processor._generate_tags(
            content_type="table",
            text_context=None,
            llm_tags=None
        )
        assert "table" in tags
    
    def test_unicode_content(self, processor):
        """Test handling of Unicode content."""
        result = processor._classify_section_type(
            "Процедура: Шаг 1, Шаг 2, Шаг 3"
        )
        assert result == "text"
    
    def test_very_long_content(self, processor):
        """Test handling of very long content."""
        long_content = "word " * 10000
        score = processor._calculate_importance_score(long_content)
        assert 0.0 <= score <= 1.0
    
    def test_special_characters_in_title(self, processor):
        """Test special characters in title."""
        result = processor._clean_section_title(
            "Operation & Maintenance (Rev. 2.0)"
        )
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_newlines_in_content(self, processor):
        """Test content with newlines."""
        result = processor._classify_section_type(
            "Line 1\nLine 2\nLine 3"
        )
        assert result == "text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
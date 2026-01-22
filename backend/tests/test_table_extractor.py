"""
Table Extractor Tests

Comprehensive tests for table detection, extraction, and CSV conversion including:
- Table validation logic
- Matrix manipulation (trimming, cleaning)
- CSV generation
- Text chunking for embeddings
- IoU calculation and deduplication
- Stable ID generation
- Image rendering

Run with: pytest test_table_extractor.py -v
"""

import pytest
import io
import csv
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Tuple

from services.table_extractor import TableExtractor


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
def extractor(mock_storage_service):
    """Create TableExtractor with mocked dependencies."""
    return TableExtractor(
        storage_service=mock_storage_service,
        zoom=2.0,
        min_cells=4,
        max_rows=500,
        max_cols=50,
        max_tokens_per_chunk=4000,
    )


@pytest.fixture
def sample_table_matrix():
    """Sample valid table matrix."""
    return [
        ["Component", "Pressure", "Flow Rate", "Notes"],
        ["PU-101", "50 bar", "100 L/min", "Main pump"],
        ["PU-102", "45 bar", "80 L/min", "Backup pump"],
        ["V-201", "N/A", "N/A", "Control valve"],
    ]


@pytest.fixture
def mock_pdf_page():
    """Mock PyMuPDF page."""
    page = Mock()
    page.rect = Mock()
    page.rect.width = 612
    page.rect.height = 792
    page.get_pixmap = Mock(return_value=Mock(
        pil_tobytes=Mock(return_value=b"PNG_IMAGE_DATA")
    ))
    return page


@pytest.fixture
def mock_pdfplumber_page():
    """Mock pdfplumber page."""
    page = Mock()
    page.find_tables = Mock(return_value=[])
    return page


# =============================================================================
# TABLE VALIDATION TESTS
# =============================================================================

class TestTableValidation:
    """Test _is_valid_table method."""
    
    def test_valid_table_accepted(self, extractor, sample_table_matrix):
        """Test valid table is accepted."""
        assert extractor._is_valid_table(sample_table_matrix) is True
    
    def test_empty_matrix_rejected(self, extractor):
        """Test empty matrix is rejected."""
        assert extractor._is_valid_table([]) is False
        assert extractor._is_valid_table(None) is False
    
    def test_single_row_rejected(self, extractor):
        """Test single row is rejected (need at least 2 rows)."""
        matrix = [["Col1", "Col2", "Col3"]]
        assert extractor._is_valid_table(matrix) is False
    
    def test_single_column_rejected(self, extractor):
        """Test single column is rejected."""
        matrix = [
            ["Row1"],
            ["Row2"],
            ["Row3"],
        ]
        assert extractor._is_valid_table(matrix) is False
    
    def test_sparse_table_rejected(self, extractor):
        """Test mostly empty table is rejected (<25% filled)."""
        matrix = [
            ["", "", "", ""],
            ["", "A", "", ""],
            ["", "", "", ""],
            ["", "", "", ""],
        ]
        assert extractor._is_valid_table(matrix) is False
    
    def test_long_cells_rejected(self, extractor):
        """Test table with paragraph-length cells is rejected."""
        long_text = "This is a very long paragraph that would not typically appear in a table cell. " * 5
        matrix = [
            ["Header", "Content"],
            ["Item", long_text],
            ["Item2", long_text],
        ]
        assert extractor._is_valid_table(matrix) is False
    
    def test_few_content_cells_rejected(self, extractor):
        """Test table with too few content cells is rejected."""
        matrix = [
            ["", ""],
            ["A", ""],
            ["", "B"],
        ]
        # Only 2 non-empty cells, need at least 4
        assert extractor._is_valid_table(matrix) is False
    
    def test_irregular_columns_rejected(self, extractor):
        """Test table with highly irregular column counts is rejected."""
        matrix = [
            ["A", "B", "C", "D", "E"],  # 5 cols
            ["X"],  # 1 col - too different
            ["Y", "Z"],  # 2 cols
        ]
        assert extractor._is_valid_table(matrix) is False
    
    def test_moderate_column_variation_accepted(self, extractor):
        """Test table with moderate column variation (merged cells) is accepted."""
        matrix = [
            ["Header 1", "Header 2", "Header 3"],
            ["Merged Cell", "", "Value"],  # Some merged cells
            ["A", "B", "C"],
        ]
        # Less than 40% variation should be accepted
        assert extractor._is_valid_table(matrix) is True
    
    def test_minimum_valid_table(self, extractor):
        """Test minimum valid table (2x2)."""
        matrix = [
            ["A", "B"],
            ["C", "D"],
        ]
        assert extractor._is_valid_table(matrix) is True


# =============================================================================
# MATRIX MANIPULATION TESTS
# =============================================================================

class TestMatrixManipulation:
    """Test matrix trimming and normalization."""
    
    def test_trim_empty_rows(self, extractor):
        """Test empty rows are removed."""
        matrix = [
            ["A", "B"],
            ["", ""],  # Empty row
            ["C", "D"],
            ["", ""],  # Empty row
        ]
        trimmed = extractor._trim_matrix(matrix)
        
        assert len(trimmed) == 2
        assert trimmed[0] == ["A", "B"]
        assert trimmed[1] == ["C", "D"]
    
    def test_trim_empty_trailing_columns(self, extractor):
        """Test empty trailing columns are removed."""
        matrix = [
            ["A", "B", "", ""],
            ["C", "D", "", ""],
        ]
        trimmed = extractor._trim_matrix(matrix)
        
        assert len(trimmed[0]) == 2
        assert trimmed[0] == ["A", "B"]
    
    def test_normalize_row_lengths(self, extractor):
        """Test rows are padded to same length."""
        matrix = [
            ["A", "B", "C"],
            ["D"],  # Short row
            ["E", "F"],
        ]
        trimmed = extractor._trim_matrix(matrix)
        
        # All rows should have same length
        lengths = [len(row) for row in trimmed]
        assert len(set(lengths)) == 1
    
    def test_trim_preserves_data(self, extractor, sample_table_matrix):
        """Test trimming preserves actual data."""
        trimmed = extractor._trim_matrix(sample_table_matrix)
        
        assert trimmed == sample_table_matrix
    
    def test_trim_completely_empty_returns_empty(self, extractor):
        """Test completely empty matrix returns empty."""
        matrix = [
            ["", ""],
            ["", ""],
        ]
        trimmed = extractor._trim_matrix(matrix)
        
        assert trimmed == []


# =============================================================================
# CELL CLEANING TESTS
# =============================================================================

class TestCellCleaning:
    """Test cell content cleaning."""
    
    def test_clean_cell_removes_newlines(self, extractor):
        """Test newlines are replaced with spaces."""
        result = extractor._clean_cell("Line1\nLine2")
        assert "\n" not in result
        assert "Line1" in result and "Line2" in result
    
    def test_clean_cell_removes_carriage_returns(self, extractor):
        """Test carriage returns are removed."""
        result = extractor._clean_cell("Text\r\nMore")
        assert "\r" not in result
    
    def test_clean_cell_collapses_whitespace(self, extractor):
        """Test multiple spaces are collapsed."""
        result = extractor._clean_cell("Word1    Word2")
        assert "    " not in result
    
    def test_clean_cell_handles_none(self, extractor):
        """Test None is handled gracefully."""
        result = extractor._clean_cell(None)
        assert result == ""
    
    def test_clean_cell_preserves_content(self, extractor):
        """Test actual content is preserved."""
        result = extractor._clean_cell("PU-101 50 bar")
        assert result == "PU-101 50 bar"


# =============================================================================
# CSV GENERATION TESTS
# =============================================================================

class TestCsvGeneration:
    """Test CSV generation from matrix."""
    
    def test_csv_has_utf8_bom(self, extractor, sample_table_matrix):
        """Test CSV has UTF-8 BOM for Excel compatibility."""
        csv_bytes = extractor._matrix_to_csv_bytes(sample_table_matrix, 4)
        
        assert csv_bytes.startswith(b'\xef\xbb\xbf')
    
    def test_csv_contains_data(self, extractor, sample_table_matrix):
        """Test CSV contains table data."""
        csv_bytes = extractor._matrix_to_csv_bytes(sample_table_matrix, 4)
        
        csv_content = csv_bytes.decode('utf-8-sig')
        assert "PU-101" in csv_content
        assert "50 bar" in csv_content
    
    def test_csv_row_count_matches(self, extractor, sample_table_matrix):
        """Test CSV has correct number of rows."""
        csv_bytes = extractor._matrix_to_csv_bytes(sample_table_matrix, 4)
        
        csv_content = csv_bytes.decode('utf-8-sig')
        rows = list(csv.reader(io.StringIO(csv_content)))
        
        assert len(rows) == len(sample_table_matrix)
    
    def test_csv_pads_short_rows(self, extractor):
        """Test short rows are padded in CSV."""
        matrix = [
            ["A", "B", "C"],
            ["D"],  # Short row
        ]
        csv_bytes = extractor._matrix_to_csv_bytes(matrix, 3)
        
        csv_content = csv_bytes.decode('utf-8-sig')
        rows = list(csv.reader(io.StringIO(csv_content)))
        
        # All rows should have 3 columns
        for row in rows:
            assert len(row) == 3


# =============================================================================
# TEXT CHUNKING TESTS
# =============================================================================

class TestTextChunking:
    """Test text chunking for embeddings."""
    
    def test_small_table_single_chunk(self, extractor, sample_table_matrix):
        """Test small table creates single chunk."""
        chunks = extractor._table_to_text_chunks(sample_table_matrix, 4)
        
        assert len(chunks) == 1
    
    def test_large_table_multiple_chunks(self, extractor):
        """Test large table creates multiple chunks."""
        # Create large table that exceeds token limit
        large_matrix = [["Cell" * 10] * 10 for _ in range(100)]
        
        # Use smaller chunk limit for test
        extractor.max_tokens_per_chunk = 100
        chunks = extractor._table_to_text_chunks(large_matrix, 10)
        
        assert len(chunks) > 1
    
    def test_chunks_use_pipe_delimiter(self, extractor, sample_table_matrix):
        """Test chunks use pipe delimiter between cells."""
        chunks = extractor._table_to_text_chunks(sample_table_matrix, 4)
        
        assert " | " in chunks[0]
    
    def test_chunks_preserve_content(self, extractor, sample_table_matrix):
        """Test all content is preserved across chunks."""
        chunks = extractor._table_to_text_chunks(sample_table_matrix, 4)
        
        combined = "\n".join(chunks)
        assert "PU-101" in combined
        assert "50 bar" in combined
    
    def test_empty_table_returns_single_empty_chunk(self, extractor):
        """Test empty table returns single empty chunk."""
        chunks = extractor._table_to_text_chunks([], 0)
        
        assert chunks == [""]


# =============================================================================
# IOU CALCULATION TESTS
# =============================================================================

class TestIouCalculation:
    """Test IoU (Intersection over Union) calculation."""
    
    def test_identical_boxes_iou_1(self, extractor):
        """Test identical boxes have IoU = 1.0."""
        bbox = (0, 0, 100, 100)
        iou = extractor._calculate_iou(bbox, bbox)
        
        assert iou == 1.0
    
    def test_no_overlap_iou_0(self, extractor):
        """Test non-overlapping boxes have IoU = 0.0."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (20, 20, 30, 30)
        iou = extractor._calculate_iou(bbox1, bbox2)
        
        assert iou == 0.0
    
    def test_partial_overlap(self, extractor):
        """Test partial overlap calculation."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 150, 150)
        iou = extractor._calculate_iou(bbox1, bbox2)
        
        # Overlap: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU ≈ 0.143
        assert 0.1 < iou < 0.2
    
    def test_contained_box(self, extractor):
        """Test when one box contains another."""
        outer = (0, 0, 100, 100)
        inner = (25, 25, 75, 75)
        iou = extractor._calculate_iou(outer, inner)
        
        # Inner area: 50x50 = 2500
        # Outer area: 100x100 = 10000
        # IoU = 2500 / 10000 = 0.25
        assert 0.2 < iou < 0.3
    
    def test_adjacent_boxes_no_overlap(self, extractor):
        """Test adjacent (touching) boxes have no overlap."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (10, 0, 20, 10)  # Adjacent, not overlapping
        iou = extractor._calculate_iou(bbox1, bbox2)
        
        assert iou == 0.0


# =============================================================================
# TABLE DEDUPLICATION TESTS
# =============================================================================

class TestTableDeduplication:
    """Test table deduplication logic."""
    
    def test_deduplicate_removes_high_overlap(self, extractor):
        """Test tables with >70% overlap are deduplicated."""
        # Create two nearly identical table objects
        table1 = Mock()
        table1.bbox = (0, 0, 100, 100)
        
        table2 = Mock()
        table2.bbox = (5, 5, 105, 105)  # High overlap with table1
        
        unique = extractor._deduplicate_tables([table1, table2])
        
        assert len(unique) == 1
    
    def test_deduplicate_keeps_distinct_tables(self, extractor):
        """Test distinct tables are kept."""
        table1 = Mock()
        table1.bbox = (0, 0, 100, 100)
        
        table2 = Mock()
        table2.bbox = (200, 200, 300, 300)  # No overlap
        
        unique = extractor._deduplicate_tables([table1, table2])
        
        assert len(unique) == 2
    
    def test_deduplicate_empty_list(self, extractor):
        """Test empty list returns empty."""
        unique = extractor._deduplicate_tables([])
        
        assert unique == []
    
    def test_deduplicate_single_table(self, extractor):
        """Test single table is preserved."""
        table = Mock()
        table.bbox = (0, 0, 100, 100)
        
        unique = extractor._deduplicate_tables([table])
        
        assert len(unique) == 1


# =============================================================================
# STABLE ID GENERATION TESTS
# =============================================================================

class TestStableIdGeneration:
    """Test stable table ID generation."""
    
    def test_id_is_deterministic(self, extractor):
        """Test same inputs give same ID."""
        id1 = extractor._stable_table_id("doc1", 5, (10, 20, 100, 200), 0)
        id2 = extractor._stable_table_id("doc1", 5, (10, 20, 100, 200), 0)
        
        assert id1 == id2
    
    def test_different_doc_different_id(self, extractor):
        """Test different doc gives different ID."""
        id1 = extractor._stable_table_id("doc1", 5, (10, 20, 100, 200), 0)
        id2 = extractor._stable_table_id("doc2", 5, (10, 20, 100, 200), 0)
        
        assert id1 != id2
    
    def test_different_page_different_id(self, extractor):
        """Test different page gives different ID."""
        id1 = extractor._stable_table_id("doc1", 5, (10, 20, 100, 200), 0)
        id2 = extractor._stable_table_id("doc1", 6, (10, 20, 100, 200), 0)
        
        assert id1 != id2
    
    def test_different_bbox_different_id(self, extractor):
        """Test different bbox gives different ID."""
        id1 = extractor._stable_table_id("doc1", 5, (10, 20, 100, 200), 0)
        id2 = extractor._stable_table_id("doc1", 5, (15, 25, 105, 205), 0)
        
        assert id1 != id2
    
    def test_id_length(self, extractor):
        """Test ID is 24 characters (hash truncated)."""
        id1 = extractor._stable_table_id("doc1", 5, (10, 20, 100, 200), 0)
        
        assert len(id1) == 24


# =============================================================================
# SAFE TRUNCATE TESTS
# =============================================================================

class TestSafeTruncate:
    """Test safe text truncation."""
    
    def test_short_text_unchanged(self, extractor):
        """Test short text is not modified."""
        text = "Short text"
        result = extractor._safe_truncate(text, 100)
        
        assert result == text
    
    def test_long_text_truncated(self, extractor):
        """Test long text is truncated."""
        text = "This is a longer text that exceeds the limit"
        result = extractor._safe_truncate(text, 20)
        
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")
    
    def test_truncate_at_word_boundary(self, extractor):
        """Test truncation happens at word boundary."""
        text = "Word1 Word2 Word3 Word4"
        result = extractor._safe_truncate(text, 15)
        
        # Should not cut in middle of word
        assert not result.endswith("d...")  # Not "Word..." but "Word1 Word2..."


# =============================================================================
# SANITIZE TESTS
# =============================================================================

class TestSanitize:
    """Test string sanitization for filenames."""
    
    def test_sanitize_special_chars(self, extractor):
        """Test special characters are replaced."""
        result = extractor._sanitize("file/with\\special:chars")
        
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result
    
    def test_sanitize_preserves_safe_chars(self, extractor):
        """Test safe characters are preserved."""
        result = extractor._sanitize("safe_file-name.txt")
        
        assert "safe" in result
        assert "-" in result
        assert "." in result
    
    def test_sanitize_length_limit(self, extractor):
        """Test result is limited to 100 chars."""
        long_name = "a" * 200
        result = extractor._sanitize(long_name)
        
        assert len(result) <= 100


# =============================================================================
# THUMBNAIL GENERATION TESTS
# =============================================================================

class TestThumbnailGeneration:
    """Test thumbnail generation."""
    
    def test_thumbnail_returns_bytes(self, extractor):
        """Test thumbnail returns PNG bytes."""
        # Create a simple test image
        from PIL import Image
        img = Image.new('RGB', (800, 600), color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        
        result = extractor._make_thumbnail_png(png_bytes, (400, 400))
        
        assert isinstance(result, bytes)
        # Should start with PNG signature
        assert result[:4] == b'\x89PNG'
    
    def test_thumbnail_preserves_aspect_ratio(self, extractor):
        """Test thumbnail preserves aspect ratio."""
        from PIL import Image
        
        # Create wide image
        img = Image.new('RGB', (1000, 500), color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        
        result = extractor._make_thumbnail_png(png_bytes, (400, 400))
        
        # Load result and check dimensions
        result_img = Image.open(io.BytesIO(result))
        width, height = result_img.size
        
        # Aspect ratio should be preserved (2:1)
        assert width == 400
        assert height == 200  # Scaled proportionally


# =============================================================================
# EXTRACTION INTEGRATION TESTS
# =============================================================================

class TestExtractionIntegration:
    """Test extract_from_page method (with mocks)."""
    
    @pytest.mark.asyncio
    async def test_extract_no_tables_returns_empty(self, extractor, mock_pdfplumber_page, mock_pdf_page):
        """Test extraction with no tables returns empty list."""
        mock_pdfplumber_page.find_tables.return_value = []
        
        result = await extractor.extract_from_page(
            pl_page=mock_pdfplumber_page,
            fitz_page=mock_pdf_page,
            doc_id="doc_001",
            page_num=0,
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_extract_saves_files(self, extractor, mock_pdfplumber_page, mock_pdf_page, mock_storage_service):
        """Test extraction saves PNG and CSV files."""
        # Create mock table
        mock_table = Mock()
        mock_table.bbox = (10, 20, 200, 300)
        mock_table.extract.return_value = [
            ["A", "B"],
            ["C", "D"],
            ["E", "F"],
        ]
        
        mock_pdfplumber_page.find_tables.return_value = [mock_table]
        
        # Mock the crop rendering
        with patch.object(extractor, '_crop_fitz_bbox_as_png', return_value=b'PNG_DATA'):
            result = await extractor.extract_from_page(
                pl_page=mock_pdfplumber_page,
                fitz_page=mock_pdf_page,
                doc_id="doc_001",
                page_num=0,
            )
        
        # Should have called save_file multiple times (PNG, CSV, thumbnail)
        assert mock_storage_service.save_file.call_count >= 2


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_none_matrix(self, extractor):
        """Test None matrix is handled."""
        # _trim_matrix handles None via list comprehension
        # _is_valid_table should return False
        assert extractor._is_valid_table(None) is False
    
    def test_matrix_with_none_rows(self, extractor):
        """Test matrix with None rows."""
        matrix = [
            ["A", "B"],
            None,  # None row
            ["C", "D"],
        ]
        # Should handle gracefully
        trimmed = extractor._trim_matrix(matrix)
        assert len(trimmed) == 2
    
    def test_matrix_with_none_cells(self, extractor):
        """Test matrix with None cells."""
        matrix = [
            ["A", None, "C"],
            [None, "B", None],
        ]
        result = extractor._is_valid_table(matrix)
        # Should handle None cells
        assert isinstance(result, bool)
    
    def test_very_wide_table(self, extractor):
        """Test table with many columns."""
        matrix = [["Col" + str(i) for i in range(100)]] * 5
        
        # Should truncate to max_cols
        chunks = extractor._table_to_text_chunks(matrix, 50)
        assert isinstance(chunks, list)
    
    def test_unicode_content(self, extractor):
        """Test table with Unicode content."""
        matrix = [
            ["Компонент", "Давление", "Расход"],
            ["ПУ-101", "50 бар", "100 л/мин"],
        ]
        
        assert extractor._is_valid_table(matrix) is True
        
        csv_bytes = extractor._matrix_to_csv_bytes(matrix, 3)
        csv_content = csv_bytes.decode('utf-8-sig')
        assert "Компонент" in csv_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

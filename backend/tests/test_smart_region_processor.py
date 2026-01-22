"""
Smart Region Processor Tests

Comprehensive tests for YOLO-guided region processing including:
- Table region processing with pdfplumber → LLM fallback
- Schema region processing with hybrid extraction
- CSV cleaning and text chunking
- Image rendering
- Table validation
- LLM extraction

Run with: pytest test_smart_region_processor.py -v
"""

import pytest
import base64
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional

from services.smart_region_processor import SmartRegionProcessor
from services.layout_analyzer import Region, RegionType, BBox


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_table_extractor():
    """Create mock table extractor."""
    extractor = Mock()
    extractor.storage = Mock()
    extractor.storage.save_file = AsyncMock(return_value="/path/to/file")
    extractor._make_thumbnail_png = Mock(return_value=b"THUMB_PNG")
    extractor._table_to_text_chunks = Mock(return_value=["chunk1", "chunk2"])
    return extractor


@pytest.fixture
def mock_schema_extractor():
    """Create mock schema extractor."""
    extractor = Mock()
    extractor.llm_service = Mock()
    extractor.llm_service.chat = Mock()
    extractor.llm_service.chat.completions = Mock()
    extractor.llm_service.chat.completions.create = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content="Col1;Col2\nVal1;Val2"))]
    ))
    extractor.extract_schema_from_region = AsyncMock(return_value={
        "id": "schema_001",
        "doc_id": "doc_001",
        "page_number": 1,
    })
    return extractor


@pytest.fixture
def mock_region_classifier():
    """Create mock region classifier."""
    classifier = Mock()
    classifier._llm_verify_type = AsyncMock(return_value=RegionType.TABLE)
    return classifier


@pytest.fixture
def processor(mock_table_extractor, mock_schema_extractor, mock_region_classifier):
    """Create SmartRegionProcessor with mocked dependencies."""
    return SmartRegionProcessor(
        table_extractor=mock_table_extractor,
        schema_extractor=mock_schema_extractor,
        region_classifier=mock_region_classifier,
        enable_llm_detection=True,
        vision_detail="high",
    )


@pytest.fixture
def mock_fitz_page():
    """Create mock PyMuPDF page."""
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
    """Create mock pdfplumber page."""
    page = Mock()
    page.crop = Mock(return_value=Mock(
        find_tables=Mock(return_value=[])
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
    """Test processor initialization."""
    
    def test_init_with_all_dependencies(self, mock_table_extractor, mock_schema_extractor, mock_region_classifier):
        """Test initialization with all dependencies."""
        processor = SmartRegionProcessor(
            table_extractor=mock_table_extractor,
            schema_extractor=mock_schema_extractor,
            region_classifier=mock_region_classifier,
        )
        
        assert processor.table_extractor == mock_table_extractor
        assert processor.schema_extractor == mock_schema_extractor
        assert processor.region_classifier == mock_region_classifier
    
    def test_init_extracts_llm_from_schema_extractor(self, mock_table_extractor, mock_schema_extractor, mock_region_classifier):
        """Test LLM service is extracted from schema extractor."""
        processor = SmartRegionProcessor(
            table_extractor=mock_table_extractor,
            schema_extractor=mock_schema_extractor,
            region_classifier=mock_region_classifier,
        )
        
        assert processor.llm_service == mock_schema_extractor.llm_service
    
    def test_init_llm_detection_disabled_without_llm(self, mock_table_extractor, mock_region_classifier):
        """Test LLM detection disabled when no LLM service."""
        schema_extractor = Mock()
        schema_extractor.llm_service = None
        
        processor = SmartRegionProcessor(
            table_extractor=mock_table_extractor,
            schema_extractor=schema_extractor,
            region_classifier=mock_region_classifier,
            enable_llm_detection=True,  # Requested but should be disabled
        )
        
        assert processor.enable_llm_detection is False
    
    def test_init_default_vision_detail(self, mock_table_extractor, mock_schema_extractor, mock_region_classifier):
        """Test default vision detail is 'high'."""
        processor = SmartRegionProcessor(
            table_extractor=mock_table_extractor,
            schema_extractor=mock_schema_extractor,
            region_classifier=mock_region_classifier,
        )
        
        assert processor.vision_detail == "high"
    
    def test_init_custom_vision_detail(self, mock_table_extractor, mock_schema_extractor, mock_region_classifier):
        """Test custom vision detail."""
        processor = SmartRegionProcessor(
            table_extractor=mock_table_extractor,
            schema_extractor=mock_schema_extractor,
            region_classifier=mock_region_classifier,
            vision_detail="low",
        )
        
        assert processor.vision_detail == "low"


# =============================================================================
# CSV CLEANING TESTS
# =============================================================================

class TestCsvCleaning:
    """Test _clean_csv_response method."""
    
    def test_clean_removes_markdown_code_block(self, processor):
        """Test markdown code blocks are removed."""
        raw = "```csv\nCol1;Col2\nVal1;Val2\n```"
        result = processor._clean_csv_response(raw)
        
        assert "```" not in result
        assert "Col1;Col2" in result
    
    def test_clean_removes_csv_language_tag(self, processor):
        """Test csv language tag is removed."""
        raw = "```csv\nHeader1;Header2\nData1;Data2```"
        result = processor._clean_csv_response(raw)
        
        assert "```csv" not in result
        assert "Header1;Header2" in result
    
    def test_clean_strips_whitespace(self, processor):
        """Test whitespace is stripped from lines."""
        raw = "  Col1;Col2  \n  Val1;Val2  "
        result = processor._clean_csv_response(raw)
        
        lines = result.split('\n')
        for line in lines:
            assert line == line.strip()
    
    def test_clean_removes_empty_lines(self, processor):
        """Test empty lines are removed."""
        raw = "Col1;Col2\n\n\nVal1;Val2\n\n"
        result = processor._clean_csv_response(raw)
        
        lines = result.split('\n')
        assert all(line.strip() for line in lines)
    
    def test_clean_preserves_data(self, processor):
        """Test actual data is preserved."""
        raw = "Header1;Header2\nData1;Data2"
        result = processor._clean_csv_response(raw)
        
        assert "Header1" in result
        assert "Data2" in result


# =============================================================================
# CSV TO TEXT CHUNKS TESTS
# =============================================================================

class TestCsvToTextChunks:
    """Test _csv_to_text_chunks method."""
    
    def test_basic_csv_to_chunks(self, processor):
        """Test basic CSV to chunks conversion."""
        csv_content = "Name;Value;Unit\nPressure;50;bar\nFlow;100;L/min"
        chunks = processor._csv_to_text_chunks(csv_content)
        
        assert len(chunks) >= 1
        assert "Pressure" in chunks[0] or "Name" in chunks[0]
    
    def test_empty_csv_returns_placeholder(self, processor):
        """Test empty CSV returns placeholder."""
        chunks = processor._csv_to_text_chunks("")
        
        assert len(chunks) == 1
        assert "[Empty table]" in chunks[0] or chunks[0] != ""
    
    def test_header_only_csv(self, processor):
        """Test CSV with only header."""
        csv_content = "Col1;Col2;Col3"
        chunks = processor._csv_to_text_chunks(csv_content)
        
        assert len(chunks) >= 1
        # Should have column info
        assert "Col1" in chunks[0] or "columns" in chunks[0].lower()
    
    def test_chunks_include_column_names(self, processor):
        """Test chunks include column names with values."""
        csv_content = "Component;Pressure\nPU-101;50 bar"
        chunks = processor._csv_to_text_chunks(csv_content)
        
        # Should format as "Column: Value"
        combined = " ".join(chunks)
        assert "Component" in combined
        assert "PU-101" in combined
    
    def test_large_csv_creates_multiple_chunks(self, processor):
        """Test large CSV creates multiple chunks."""
        # Create large CSV
        rows = ["Header1;Header2"]
        for i in range(100):
            rows.append(f"LongValue{i}WithMoreText;AnotherLongValue{i}WithEvenMoreText")
        csv_content = "\n".join(rows)
        
        chunks = processor._csv_to_text_chunks(csv_content, chunk_size=200)
        
        assert len(chunks) > 1
    
    def test_chunks_respect_size_limit(self, processor):
        """Test chunks respect size limit."""
        csv_content = "A;B\n" + "\n".join([f"Val{i};Data{i}" for i in range(50)])
        chunks = processor._csv_to_text_chunks(csv_content, chunk_size=100)
        
        # Most chunks should be around the size limit
        for chunk in chunks[:-1]:  # Last chunk may be smaller
            assert len(chunk) <= 150  # Some tolerance


# =============================================================================
# TABLE VALIDATION TESTS
# =============================================================================

class TestTableValidation:
    """Test _is_valid_table_result method."""
    
    def test_valid_table_result(self, processor):
        """Test valid table result passes."""
        table_data = {
            "id": "table_001",
            "rows": 5,
            "cols": 3,
            "text_chunks": ["chunk1", "chunk2"],
        }
        
        result = processor._is_valid_table_result(table_data)
        
        assert result is True
    
    def test_none_table_result(self, processor):
        """Test None is invalid."""
        result = processor._is_valid_table_result(None)
        
        assert result is False
    
    def test_empty_table_result(self, processor):
        """Test empty dict is invalid."""
        result = processor._is_valid_table_result({})
        
        assert result is False
    
    def test_table_without_chunks(self, processor):
        """Test table without chunks is invalid."""
        table_data = {
            "id": "table_001",
            "rows": 5,
            "cols": 3,
            "text_chunks": [],  # Empty chunks
        }
        
        result = processor._is_valid_table_result(table_data)
        
        assert result is False
    
    def test_table_with_only_empty_chunks(self, processor):
        """Test table with only empty chunks is invalid."""
        table_data = {
            "id": "table_001",
            "rows": 5,
            "cols": 3,
            "text_chunks": ["", "   "],  # Only whitespace
        }
        
        result = processor._is_valid_table_result(table_data)
        
        assert result is False


# =============================================================================
# BUILD TABLE FROM CSV TESTS
# =============================================================================

class TestBuildTableFromCsv:
    """Test _build_table_data_from_csv method."""
    
    @pytest.mark.asyncio
    async def test_build_table_success(self, processor):
        """Test successful table building from CSV."""
        csv_content = "Col1;Col2\nVal1;Val2\nVal3;Val4"
        image_bytes = b"PNG_DATA"
        bbox = BBox(100, 200, 300, 400)
        
        result = await processor._build_table_data_from_csv(
            csv_content=csv_content,
            image_bytes=image_bytes,
            bbox=bbox,
            doc_id="doc_001",
            safe_doc_id="doc_001",
            page_num=0,
        )
        
        assert result is not None
        assert result["rows"] == 3
        assert result["cols"] == 2
        assert result["llm_extracted"] is True
    
    @pytest.mark.asyncio
    async def test_build_table_too_small(self, processor):
        """Test table building rejects too small CSV."""
        csv_content = "Col1\nVal1"  # 1 column only
        image_bytes = b"PNG_DATA"
        bbox = BBox(100, 200, 300, 400)
        
        result = await processor._build_table_data_from_csv(
            csv_content=csv_content,
            image_bytes=image_bytes,
            bbox=bbox,
            doc_id="doc_001",
            safe_doc_id="doc_001",
            page_num=0,
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_build_table_includes_caption(self, processor):
        """Test table building includes extracted caption."""
        csv_content = "Col1;Col2\nVal1;Val2\nVal3;Val4"
        image_bytes = b"PNG_DATA"
        bbox = BBox(100, 200, 300, 400)
        
        result = await processor._build_table_data_from_csv(
            csv_content=csv_content,
            image_bytes=image_bytes,
            bbox=bbox,
            doc_id="doc_001",
            safe_doc_id="doc_001",
            page_num=0,
            extracted_caption="Table 5: Component List",
        )
        
        assert result is not None
        assert result["caption"] == "Table 5: Component List"
    
    @pytest.mark.asyncio
    async def test_build_table_includes_tags(self, processor):
        """Test table building includes LLM tags."""
        csv_content = "Col1;Col2\nVal1;Val2\nVal3;Val4"
        image_bytes = b"PNG_DATA"
        bbox = BBox(100, 200, 300, 400)
        
        result = await processor._build_table_data_from_csv(
            csv_content=csv_content,
            image_bytes=image_bytes,
            bbox=bbox,
            doc_id="doc_001",
            safe_doc_id="doc_001",
            page_num=0,
            extracted_tags=["specifications", "pump"],
        )
        
        assert result is not None
        assert result["llm_tags"] == ["specifications", "pump"]
    
    @pytest.mark.asyncio
    async def test_build_table_generates_id(self, processor):
        """Test table building generates unique ID."""
        csv_content = "Col1;Col2\nVal1;Val2\nVal3;Val4"
        image_bytes = b"PNG_DATA"
        bbox = BBox(100, 200, 300, 400)
        
        result = await processor._build_table_data_from_csv(
            csv_content=csv_content,
            image_bytes=image_bytes,
            bbox=bbox,
            doc_id="doc_001",
            safe_doc_id="doc_001",
            page_num=0,
        )
        
        assert result is not None
        assert "id" in result
        assert len(result["id"]) == 24  # SHA256 truncated


# =============================================================================
# RENDER REGION TESTS
# =============================================================================

class TestRenderRegion:
    """Test _render_region_as_png method."""
    
    def test_render_returns_bytes(self, processor, mock_fitz_page):
        """Test rendering returns PNG bytes."""
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"PNG_DATA"
        mock_fitz_page.get_pixmap.return_value = mock_pixmap
        
        bbox = BBox(100, 200, 300, 400)
        result = processor._render_region_as_png(mock_fitz_page, bbox)
        
        assert isinstance(result, bytes)
    
    def test_render_with_caption_expansion(self, processor, mock_fitz_page):
        """Test rendering with caption area expansion."""
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"PNG_DATA"
        mock_fitz_page.get_pixmap.return_value = mock_pixmap
        
        bbox = BBox(100, 200, 300, 400)
        result = processor._render_region_as_png(
            mock_fitz_page, bbox, expand_for_caption=True
        )
        
        assert isinstance(result, bytes)
        # Verify get_pixmap was called (expanded clip)
        mock_fitz_page.get_pixmap.assert_called_once()


# =============================================================================
# PROCESS TABLE REGION TESTS
# =============================================================================

class TestProcessTableRegion:
    """Test process_table_region method."""
    
    @pytest.mark.asyncio
    async def test_process_table_pdfplumber_success(self, processor, mock_fitz_page, mock_pdfplumber_page, sample_table_region):
        """Test successful table extraction with pdfplumber."""
        # Mock successful pdfplumber extraction
        with patch.object(processor, '_extract_table_from_bbox', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = {
                "id": "table_001",
                "rows": 5,
                "cols": 3,
                "text_chunks": ["chunk1", "chunk2"],
            }
            
            with patch.object(processor, '_is_valid_table_result', return_value=True):
                with patch.object(processor, '_create_table_chunks', return_value=[{"type": "table"}]):
                    result = await processor.process_table_region(
                        fitz_page=mock_fitz_page,
                        pl_page=mock_pdfplumber_page,
                        region=sample_table_region,
                        doc_id="doc_001",
                        safe_doc_id="doc_001",
                        page_num=0,
                        full_page_text="Sample text",
                        section_id="sec_001",
                    )
        
        assert result["type"] == "table"
        assert "chunks" in result
    
    @pytest.mark.asyncio
    async def test_process_table_llm_fallback(self, processor, mock_fitz_page, mock_pdfplumber_page, sample_table_region):
        """Test LLM fallback when pdfplumber fails."""
        # Mock pdfplumber failure
        with patch.object(processor, '_extract_table_from_bbox', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = None  # pdfplumber failed
            
            with patch.object(processor, '_render_region_as_png', return_value=b"PNG_DATA"):
                with patch.object(processor, '_llm_extract_table_direct', new_callable=AsyncMock) as mock_llm:
                    mock_llm.return_value = {
                        "id": "table_001",
                        "rows": 3,
                        "cols": 2,
                        "text_chunks": ["llm_chunk"],
                        "llm_extracted": True,
                    }
                    
                    with patch.object(processor, '_create_table_chunks', return_value=[{"type": "table"}]):
                        result = await processor.process_table_region(
                            fitz_page=mock_fitz_page,
                            pl_page=mock_pdfplumber_page,
                            region=sample_table_region,
                            doc_id="doc_001",
                            safe_doc_id="doc_001",
                            page_num=0,
                            full_page_text="Sample text",
                            section_id="sec_001",
                        )
        
        assert result["type"] == "table"
    
    @pytest.mark.asyncio
    async def test_process_table_uses_provided_caption(self, processor, mock_fitz_page, mock_pdfplumber_page):
        """Test table processing uses caption from region."""
        region = Region(
            bbox=BBox(100, 200, 400, 500),
            region_type=RegionType.TABLE,
            confidence=0.85,
            page_number=0,
            caption_text="Table 5: Specifications",  # Pre-detected caption
        )
        
        with patch.object(processor, '_extract_table_from_bbox', new_callable=AsyncMock) as mock_extract:
            # Verify caption is passed
            mock_extract.return_value = {
                "id": "table_001",
                "rows": 5,
                "cols": 3,
                "text_chunks": ["chunk1"],
            }
            
            with patch.object(processor, '_is_valid_table_result', return_value=True):
                with patch.object(processor, '_create_table_chunks', return_value=[]):
                    await processor.process_table_region(
                        fitz_page=mock_fitz_page,
                        pl_page=mock_pdfplumber_page,
                        region=region,
                        doc_id="doc_001",
                        safe_doc_id="doc_001",
                        page_num=0,
                        full_page_text="",
                        section_id=None,
                    )
            
            # Check caption was passed to extraction
            call_kwargs = mock_extract.call_args[1]
            assert call_kwargs.get("provided_caption") == "Table 5: Specifications"


# =============================================================================
# PROCESS SCHEMA REGION TESTS
# =============================================================================

class TestProcessSchemaRegion:
    """Test process_schema_region method."""
    
    @pytest.mark.asyncio
    async def test_process_schema_basic(self, processor, mock_fitz_page, sample_schema_region, mock_schema_extractor):
        """Test basic schema extraction."""
        mock_pl_page = Mock()

        result = await processor.process_schema_region(
            fitz_page=mock_fitz_page,
            pl_page=mock_pl_page,
            region=sample_schema_region,
            doc_id="doc_001",
            safe_doc_id="doc_001",
            page_num=0,
            full_page_text="Sample page text",
            section_id="sec_001",
        )
        
        assert result["type"] == "schema"
        mock_schema_extractor.extract_schema_from_region.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_schema_with_text_extraction(self, processor, mock_fitz_page, mock_schema_extractor):
        """Test schema with text extraction enabled."""
        mock_pl_page = Mock()
        
        region = Region(
            bbox=BBox(100, 200, 400, 500),
            region_type=RegionType.SCHEMA,
            confidence=0.75,
            page_number=0,
            extract_text_also=True,  # Enable dual extraction
        )
        
        result = await processor.process_schema_region(
            fitz_page=mock_fitz_page,
            pl_page=mock_pl_page,
            region=region,
            doc_id="doc_001",
            safe_doc_id="doc_001",
            page_num=0,
            full_page_text="Sample text",
            section_id="sec_001",
        )
        
        assert result["type"] == "schema"


# =============================================================================
# LLM TABLE EXTRACTION TESTS
# =============================================================================

class TestLlmTableExtraction:
    """Test _llm_extract_table_direct method."""
    
    @pytest.mark.asyncio
    async def test_llm_extract_success(self, processor):
        """Test successful LLM table extraction."""
        # Mock LLM response
        processor.llm_service.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content="Col1;Col2\nVal1;Val2\nVal3;Val4"))]
        ))
        
        with patch.object(processor, '_build_table_data_from_csv', new_callable=AsyncMock) as mock_build:
            mock_build.return_value = {
                "id": "table_001",
                "rows": 3,
                "cols": 2,
                "llm_extracted": True,
            }
            
            result = await processor._llm_extract_table_direct(
                image_bytes=b"PNG_DATA",
                image_base64=base64.b64encode(b"PNG_DATA").decode(),
                bbox=BBox(100, 200, 300, 400),
                doc_id="doc_001",
                safe_doc_id="doc_001",
                page_num=0,
            )
        
        assert result is not None
        assert result["llm_extracted"] is True
    
    @pytest.mark.asyncio
    async def test_llm_extract_with_caption(self, processor):
        """Test LLM extraction with provided caption."""
        processor.llm_service.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content="Col1;Col2\nVal1;Val2"))]
        ))
        
        with patch.object(processor, '_build_table_data_from_csv', new_callable=AsyncMock) as mock_build:
            mock_build.return_value = {"id": "t1", "caption": "Table 5"}
            
            result = await processor._llm_extract_table_direct(
                image_bytes=b"PNG_DATA",
                image_base64=base64.b64encode(b"PNG_DATA").decode(),
                bbox=BBox(100, 200, 300, 400),
                doc_id="doc_001",
                safe_doc_id="doc_001",
                page_num=0,
                provided_caption="Table 5: Specifications",
            )
        
        # Caption should be passed through
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs.get("extracted_caption") == "Table 5: Specifications"
    
    @pytest.mark.asyncio
    async def test_llm_extract_api_error(self, processor):
        """Test LLM extraction handles API errors."""
        processor.llm_service.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await processor._llm_extract_table_direct(
            image_bytes=b"PNG_DATA",
            image_base64=base64.b64encode(b"PNG_DATA").decode(),
            bbox=BBox(100, 200, 300, 400),
            doc_id="doc_001",
            safe_doc_id="doc_001",
            page_num=0,
        )
        
        assert result is None


# =============================================================================
# CREATE TABLE CHUNKS TESTS
# =============================================================================

class TestCreateTableChunks:
    """Test _create_table_chunks method."""
    
    def test_create_chunks_from_table_data(self, processor):
        """Test chunk creation from table data."""
        table_data = {
            "id": "table_001",
            "doc_id": "doc_001",
            "page_number": 5,
            "title": "Table 5",
            "caption": "Component List",
            "rows": 10,
            "cols": 4,
            "file_path": "/path/to/img.png",
            "csv_path": "/path/to/data.csv",
            "text_chunks": ["chunk1 content", "chunk2 content"],
            "normalized_text": "full text",
        }
        
        chunks = processor._create_table_chunks(table_data, section_id="sec_001")
        
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk["metadata"]["table_id"] == "table_001"
            assert chunk["doc_id"] == "doc_001"
    
    def test_create_chunks_includes_metadata(self, processor):
        """Test chunks include all metadata."""
        table_data = {
            "id": "table_001",
            "doc_id": "doc_001",
            "page_number": 5,
            "title": "Table 5",
            "caption": "Specifications",
            "rows": 5,
            "cols": 3,
            "file_path": "/img.png",
            "csv_path": "/data.csv",
            "text_chunks": ["chunk content"],
            "normalized_text": "full",
        }
        
        chunks = processor._create_table_chunks(table_data, section_id="sec_001")
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk["page_number"] == 5
        assert chunk["metadata"]["title"] == "Table 5"
        assert chunk["metadata"]["caption"] == "Specifications"
        assert chunk["metadata"]["section_id"] == "sec_001"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_csv_with_special_characters(self, processor):
        """Test CSV cleaning with special characters."""
        raw = "Col1;Col2\nValue with © and ™;Normal"
        result = processor._clean_csv_response(raw)
        
        assert "©" in result or "Value" in result
    
    def test_csv_with_unicode(self, processor):
        """Test CSV with Unicode content."""
        csv_content = "Компонент;Давление\nНасос;50 бар"
        chunks = processor._csv_to_text_chunks(csv_content)
        
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "Компонент" in combined or "Насос" in combined
    
    @pytest.mark.asyncio
    async def test_build_table_empty_csv(self, processor):
        """Test building table from empty CSV."""
        result = await processor._build_table_data_from_csv(
            csv_content="",
            image_bytes=b"PNG",
            bbox=BBox(0, 0, 100, 100),
            doc_id="d",
            safe_doc_id="d",
            page_num=0,
        )
        
        assert result is None
    
    def test_csv_chunks_with_missing_columns(self, processor):
        """Test CSV with inconsistent column counts."""
        csv_content = "A;B;C\nVal1;Val2\nX;Y;Z;Extra"
        chunks = processor._csv_to_text_chunks(csv_content)
        
        # Should handle gracefully
        assert len(chunks) >= 1


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================

class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_table_processing_flow(self, processor, mock_fitz_page, mock_pdfplumber_page):
        """Test complete table processing flow."""
        region = Region(
            bbox=BBox(100, 200, 400, 500),
            region_type=RegionType.TABLE,
            confidence=0.9,
            page_number=0,
        )
        
        with patch.object(processor, '_extract_table_from_bbox', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = {
                "id": "table_001",
                "doc_id": "doc_001",
                "page_number": 1,
                "title": "Table 1",
                "caption": "",
                "rows": 5,
                "cols": 3,
                "file_path": "/img.png",
                "csv_path": "/data.csv",
                "text_chunks": ["Row data"],
                "normalized_text": "Row data",
            }
            
            with patch.object(processor, '_is_valid_table_result', return_value=True):
                result = await processor.process_table_region(
                    fitz_page=mock_fitz_page,
                    pl_page=mock_pdfplumber_page,
                    region=region,
                    doc_id="doc_001",
                    safe_doc_id="doc_001",
                    page_num=0,
                    full_page_text="Page text",
                    section_id="sec_001",
                )
        
        assert result["type"] == "table"
        assert len(result["chunks"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

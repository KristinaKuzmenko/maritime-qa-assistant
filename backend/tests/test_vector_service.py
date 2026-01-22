"""
Vector Service Tests

Comprehensive tests for Qdrant vector database operations including:
- Collection management
- Embedding and upsert operations
- Search operations (text, schemas, tables)
- Filter construction
- Delete operations
- Neighbor chunk retrieval
- Statistics and info

Run with: pytest test_vector_service.py -v
"""

import pytest
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Mock qdrant_client models
from qdrant_client.models import ScoredPoint, PointStruct, Filter, FieldCondition, MatchValue


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    service = Mock()
    service.create_embedding = AsyncMock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client."""
    client = Mock()
    
    # Mock collection operations
    client.get_collections = Mock(return_value=Mock(collections=[]))
    client.create_collection = Mock()
    client.upsert = Mock()
    client.search = Mock(return_value=[])
    client.scroll = Mock(return_value=([], None))
    client.delete = Mock()
    client.count = Mock(return_value=Mock(count=0))
    
    return client


@pytest.fixture
def vector_service(mock_embedding_service, mock_qdrant_client):
    """Create VectorService with mocked dependencies."""
    with patch('services.vector_service.QdrantClient'):
        with patch('services.vector_service.settings') as mock_settings:
            mock_settings.qdrant_host = "localhost"
            mock_settings.qdrant_port = 6333
            mock_settings.qdrant_api_key = None
            mock_settings.text_chunks_collection = "text_chunks"
            mock_settings.schemas_collection = "schemas"
            mock_settings.tables_text_collection = "tables_text"
            mock_settings.vector_dimension = 1536
            
            from services.vector_service import VectorService
            
            service = VectorService(
                embedding_service=mock_embedding_service,
                client=mock_qdrant_client
            )
            
            return service


@pytest.fixture
def sample_text_chunk():
    """Sample text chunk data."""
    return {
        "section_id": "sec_001",
        "chunk_index": 0,
        "text": "This is sample text about fuel pump maintenance procedures.",
        "doc_id": "doc_001",
        "doc_title": "Engine Manual",
        "page_start": 10,
        "page_end": 12,
        "chunk_char_start": 0,
        "chunk_char_end": 100,
        "section_title": "Maintenance Procedures",
        "owner": "test_user",
    }


@pytest.fixture
def sample_schema():
    """Sample schema data."""
    return {
        "schema_id": "schema_001",
        "text": "Fuel system diagram showing pump PU-101 and valves",
        "doc_id": "doc_001",
        "doc_title": "Engine Manual",
        "page": 15,
        "caption": "Figure 1: Fuel System Overview",
        "owner": "test_user",
    }


@pytest.fixture
def sample_table_chunk():
    """Sample table chunk data."""
    return {
        "chunk_id": "table_chunk_001",
        "table_id": "table_001",
        "chunk_index": 0,
        "text": "Component | Pressure | Flow\nPU-101 | 50 bar | 100 L/min",
        "doc_id": "doc_001",
        "doc_title": "Engine Manual",
        "page": 20,
        "table_title": "Pump Specifications",
        "total_chunks": 1,
        "rows": 5,
        "cols": 3,
        "owner": "test_user",
    }


@pytest.fixture
def mock_search_result():
    """Create mock search result."""
    point = Mock(spec=ScoredPoint)
    point.id = "point_123"
    point.score = 0.85
    point.payload = {
        "section_id": "sec_001",
        "chunk_index": 0,
        "text": "Full text about fuel system maintenance",
        "doc_id": "doc_001",
        "section_title": "Maintenance Procedures",
        "page_start": 10,
        "page_end": 12,
        "text_preview": "Full text about fuel...",
        "chunk_char_start": 0,
        "chunk_char_end": 100,
        "char_count": 100,
        "system_ids": ["fuel_system"],
        "entity_ids": ["PU-101"],
    }
    return point


# =============================================================================
# COLLECTION MANAGEMENT TESTS
# =============================================================================

class TestCollectionManagement:
    """Test collection initialization and management."""
    
    def test_initialize_creates_missing_collections(self, vector_service, mock_qdrant_client):
        """Test that missing collections are created."""
        # No existing collections
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        
        vector_service.initialize_collections()
        
        # Should create all three collections
        assert mock_qdrant_client.create_collection.call_count == 3
    
    def test_initialize_skips_existing_collections(self, vector_service, mock_qdrant_client):
        """Test that existing collections are not recreated."""
        # All collections exist - need to set .name attribute, not Mock(name=...)
        existing = [
            Mock(name="text_chunks"),
            Mock(name="schemas"),
            Mock(name="tables_text"),
        ]
        existing[0].name = "text_chunks"
        existing[1].name = "schemas"
        existing[2].name = "tables_text"
        mock_qdrant_client.get_collections.return_value = Mock(collections=existing)
        
        vector_service.initialize_collections()
        
        # Should not create any collections
        mock_qdrant_client.create_collection.assert_not_called()
    
    def test_initialize_creates_only_missing(self, vector_service, mock_qdrant_client):
        """Test that only missing collections are created."""
        # Only text_chunks exists
        existing = [Mock(name="text_chunks")]
        existing[0].name = "text_chunks"
        mock_qdrant_client.get_collections.return_value = Mock(collections=existing)
        
        vector_service.initialize_collections()
        
        # Should create 2 collections (schemas, tables_text)
        assert mock_qdrant_client.create_collection.call_count == 2


# =============================================================================
# ADD TEXT CHUNK TESTS
# =============================================================================

class TestAddTextChunk:
    """Test adding text chunks to vector database."""
    
    @pytest.mark.asyncio
    async def test_add_text_chunk_creates_embedding(self, vector_service, mock_embedding_service, sample_text_chunk):
        """Test that embedding is created for text chunk."""
        await vector_service.add_text_chunk(**sample_text_chunk)
        
        mock_embedding_service.create_embedding.assert_called_once_with(
            sample_text_chunk["text"]
        )
    
    @pytest.mark.asyncio
    async def test_add_text_chunk_upserts_to_qdrant(self, vector_service, mock_qdrant_client, sample_text_chunk):
        """Test that chunk is upserted to Qdrant."""
        await vector_service.add_text_chunk(**sample_text_chunk)
        
        mock_qdrant_client.upsert.assert_called_once()
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "text_chunks"
    
    @pytest.mark.asyncio
    async def test_add_text_chunk_deterministic_id(self, vector_service, sample_text_chunk):
        """Test that chunk ID is deterministic based on section_id and chunk_index."""
        # Add same chunk twice
        await vector_service.add_text_chunk(**sample_text_chunk)
        await vector_service.add_text_chunk(**sample_text_chunk)
        
        # Both should use same point ID (deterministic)
        calls = vector_service.client.upsert.call_args_list
        point1 = calls[0][1]["points"][0]
        point2 = calls[1][1]["points"][0]
        
        assert point1.id == point2.id
    
    @pytest.mark.asyncio
    async def test_add_text_chunk_truncates_long_text(self, vector_service, mock_embedding_service):
        """Test that very long text is truncated."""
        long_text = "word " * 50000  # ~250k chars, ~60k tokens
        
        await vector_service.add_text_chunk(
            section_id="sec_001",
            chunk_index=0,
            text=long_text,
            doc_id="doc_001",
            doc_title="Test",
            page_start=1,
            page_end=1,
            chunk_char_start=0,
            chunk_char_end=len(long_text),
        )
        
        # Embedding should be called with truncated text
        call_text = mock_embedding_service.create_embedding.call_args[0][0]
        assert len(call_text) <= 32000
    
    @pytest.mark.asyncio
    async def test_add_text_chunk_payload_fields(self, vector_service, mock_qdrant_client, sample_text_chunk):
        """Test that payload contains all required fields."""
        await vector_service.add_text_chunk(**sample_text_chunk)
        
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        point = call_kwargs["points"][0]
        payload = point.payload
        
        assert payload["type"] == "text_chunk"
        assert payload["section_id"] == sample_text_chunk["section_id"]
        assert payload["chunk_index"] == sample_text_chunk["chunk_index"]
        assert payload["doc_id"] == sample_text_chunk["doc_id"]
        assert payload["text"] == sample_text_chunk["text"]
        assert "text_preview" in payload
        assert "char_count" in payload


# =============================================================================
# ADD SCHEMA EMBEDDING TESTS
# =============================================================================

class TestAddSchemaEmbedding:
    """Test adding schema embeddings."""
    
    @pytest.mark.asyncio
    async def test_add_schema_creates_embedding(self, vector_service, mock_embedding_service, sample_schema):
        """Test that embedding is created for schema."""
        await vector_service.add_schema_embedding(**sample_schema)
        
        mock_embedding_service.create_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_schema_upserts_to_correct_collection(self, vector_service, mock_qdrant_client, sample_schema):
        """Test that schema is upserted to schemas collection."""
        await vector_service.add_schema_embedding(**sample_schema)
        
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "schemas"
    
    @pytest.mark.asyncio
    async def test_add_schema_deterministic_id(self, vector_service, sample_schema):
        """Test that schema ID is deterministic."""
        await vector_service.add_schema_embedding(**sample_schema)
        await vector_service.add_schema_embedding(**sample_schema)
        
        calls = vector_service.client.upsert.call_args_list
        point1 = calls[0][1]["points"][0]
        point2 = calls[1][1]["points"][0]
        
        assert point1.id == point2.id


# =============================================================================
# ADD TABLE CHUNK TESTS
# =============================================================================

class TestAddTableChunk:
    """Test adding table chunks."""
    
    @pytest.mark.asyncio
    async def test_add_table_chunk_creates_embedding(self, vector_service, mock_embedding_service, sample_table_chunk):
        """Test that embedding is created for table chunk."""
        await vector_service.add_table_chunk(**sample_table_chunk)
        
        mock_embedding_service.create_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_table_chunk_upserts_to_correct_collection(self, vector_service, mock_qdrant_client, sample_table_chunk):
        """Test that table chunk is upserted to tables collection."""
        await vector_service.add_table_chunk(**sample_table_chunk)
        
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "tables_text"
    
    @pytest.mark.asyncio
    async def test_add_table_chunk_payload_fields(self, vector_service, mock_qdrant_client, sample_table_chunk):
        """Test that table chunk payload has required fields."""
        await vector_service.add_table_chunk(**sample_table_chunk)
        
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        payload = call_kwargs["points"][0].payload
        
        assert payload["type"] == "table_chunk"
        assert payload["table_id"] == sample_table_chunk["table_id"]
        assert payload["chunk_index"] == sample_table_chunk["chunk_index"]
        assert payload["total_chunks"] == sample_table_chunk["total_chunks"]


# =============================================================================
# SEARCH TEXT TESTS
# =============================================================================

class TestSearchText:
    """Test text search operations."""
    
    @pytest.mark.asyncio
    async def test_search_text_calls_embedding_service(self, vector_service, mock_embedding_service, mock_qdrant_client):
        """Test that query embedding is created."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_text("fuel pump maintenance")
        
        mock_embedding_service.create_embedding.assert_called_once_with("fuel pump maintenance")
    
    @pytest.mark.asyncio
    async def test_search_text_returns_formatted_results(self, vector_service, mock_qdrant_client, mock_search_result):
        """Test that search results are properly formatted."""
        mock_qdrant_client.search.return_value = [mock_search_result]
        
        results = await vector_service.search_text("fuel pump", limit=10)
        
        assert len(results) == 1
        result = results[0]
        
        assert "section_id" in result
        assert "text" in result
        assert "score" in result
        assert result["score"] == 0.85
        assert result["section_id"] == "sec_001"
    
    @pytest.mark.asyncio
    async def test_search_text_with_doc_filter(self, vector_service, mock_qdrant_client):
        """Test search with document filter."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_text("query", doc_id="doc_001", limit=5)
        
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["query_filter"] is not None
    
    @pytest.mark.asyncio
    async def test_search_text_with_owner_filter(self, vector_service, mock_qdrant_client):
        """Test search with owner filter."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_text("query", owner="user_001", limit=5)
        
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["query_filter"] is not None
    
    @pytest.mark.asyncio
    async def test_search_text_with_score_threshold(self, vector_service, mock_qdrant_client):
        """Test search with score threshold."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_text("query", score_threshold=0.7, limit=10)
        
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["score_threshold"] == 0.7
    
    @pytest.mark.asyncio
    async def test_search_text_respects_limit(self, vector_service, mock_qdrant_client):
        """Test search respects limit parameter."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_text("query", limit=5)
        
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["limit"] == 5
    
    @pytest.mark.asyncio
    async def test_search_text_empty_results(self, vector_service, mock_qdrant_client):
        """Test search with no results."""
        mock_qdrant_client.search.return_value = []
        
        results = await vector_service.search_text("nonexistent query")
        
        assert results == []


# =============================================================================
# SEARCH SCHEMAS TESTS
# =============================================================================

class TestSearchSchemas:
    """Test schema search operations."""
    
    @pytest.mark.asyncio
    async def test_search_schemas_returns_formatted_results(self, vector_service, mock_qdrant_client):
        """Test schema search results formatting."""
        mock_point = Mock(spec=ScoredPoint)
        mock_point.id = "schema_point"
        mock_point.score = 0.9
        mock_point.payload = {
            "schema_id": "schema_001",
            "doc_id": "doc_001",
            "page": 15,
            "caption": "Figure 1",
            "text_preview": "Fuel system diagram...",
            "char_count": 200,
            "system_ids": [],
            "entity_ids": [],
        }
        mock_qdrant_client.search.return_value = [mock_point]
        
        results = await vector_service.search_schemas("fuel diagram")
        
        assert len(results) == 1
        result = results[0]
        assert result["schema_id"] == "schema_001"
        assert result["score"] == 0.9
        assert result["page"] == 15
    
    @pytest.mark.asyncio
    async def test_search_schemas_uses_correct_collection(self, vector_service, mock_qdrant_client):
        """Test that schema search uses schemas collection."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_schemas("diagram")
        
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["collection_name"] == "schemas"


# =============================================================================
# SEARCH TABLES TESTS
# =============================================================================

class TestSearchTables:
    """Test table search operations."""
    
    @pytest.mark.asyncio
    async def test_search_tables_returns_formatted_results(self, vector_service, mock_qdrant_client):
        """Test table search results formatting."""
        mock_point = Mock(spec=ScoredPoint)
        mock_point.id = "table_point"
        mock_point.score = 0.88
        mock_point.payload = {
            "table_id": "table_001",
            "chunk_id": "chunk_001",
            "chunk_index": 0,
            "doc_id": "doc_001",
            "page": 20,
            "table_title": "Specifications",
            "text_preview": "Component | Value...",
            "rows": 10,
            "cols": 5,
            "system_ids": [],
            "entity_ids": [],
        }
        mock_qdrant_client.search.return_value = [mock_point]
        
        results = await vector_service.search_tables("pump specifications")
        
        assert len(results) == 1
        result = results[0]
        assert result["table_id"] == "table_001"
        assert result["score"] == 0.88
    
    @pytest.mark.asyncio
    async def test_search_tables_uses_correct_collection(self, vector_service, mock_qdrant_client):
        """Test that table search uses tables collection."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_tables("specifications")
        
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["collection_name"] == "tables_text"


# =============================================================================
# DELETE OPERATIONS TESTS
# =============================================================================

class TestDeleteOperations:
    """Test delete operations."""
    
    def test_delete_document_vectors_all_collections(self, vector_service, mock_qdrant_client):
        """Test that document deletion removes from all collections."""
        vector_service.delete_document_vectors("doc_001")
        
        # Should delete from 3 collections
        assert mock_qdrant_client.delete.call_count == 3
    
    def test_delete_section_vectors(self, vector_service, mock_qdrant_client):
        """Test deleting vectors for a specific section."""
        vector_service.delete_section_vectors("sec_001")
        
        mock_qdrant_client.delete.assert_called_once()
        call_kwargs = mock_qdrant_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "text_chunks"
    
    def test_delete_schema_vector(self, vector_service, mock_qdrant_client):
        """Test deleting a specific schema vector."""
        vector_service.delete_schema_vector("schema_001")
        
        mock_qdrant_client.delete.assert_called_once()
        call_kwargs = mock_qdrant_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "schemas"
    
    def test_delete_table_vectors(self, vector_service, mock_qdrant_client):
        """Test deleting table chunk vectors."""
        vector_service.delete_table_vectors("table_001")
        
        mock_qdrant_client.delete.assert_called_once()
        call_kwargs = mock_qdrant_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "tables_text"
    
    def test_delete_figure_vector_deprecated(self, vector_service, mock_qdrant_client):
        """Test that deprecated method still works."""
        with patch('services.vector_service.logger') as mock_logger:
            vector_service.delete_figure_vector("fig_001")
            
            # Should log deprecation warning
            mock_logger.warning.assert_called()
            # Should still delete
            mock_qdrant_client.delete.assert_called_once()


# =============================================================================
# NEIGHBOR CHUNKS TESTS
# =============================================================================

class TestNeighborChunks:
    """Test neighbor chunk retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_neighbor_chunks_returns_adjacent(self, vector_service, mock_qdrant_client):
        """Test getting adjacent chunks."""
        # Mock scroll results with multiple chunks
        points = []
        for i in range(5):
            point = Mock()
            point.payload = {
                "section_id": "sec_001",
                "chunk_index": i,
                "doc_id": "doc_001",
                "section_title": "Test Section",
                "page_start": 10,
                "page_end": 12,
                "text": f"Chunk {i} content",
                "chunk_char_start": i * 100,
                "chunk_char_end": (i + 1) * 100,
            }
            points.append(point)
        
        mock_qdrant_client.scroll.return_value = (points, None)
        
        # Get neighbors of chunk 2 with range 1
        results = await vector_service.get_neighbor_chunks(
            section_id="sec_001",
            chunk_index=2,
            neighbor_range=1
        )
        
        # Should return chunks 1, 2, 3
        assert len(results) == 3
        indices = [r["chunk_index"] for r in results]
        assert 1 in indices
        assert 2 in indices
        assert 3 in indices
    
    @pytest.mark.asyncio
    async def test_get_neighbor_chunks_sorted_by_index(self, vector_service, mock_qdrant_client):
        """Test that results are sorted by chunk_index."""
        points = []
        for i in [3, 1, 2]:  # Out of order
            point = Mock()
            point.payload = {
                "section_id": "sec_001",
                "chunk_index": i,
                "doc_id": "doc_001",
                "section_title": "Test",
                "page_start": 10,
                "page_end": 12,
                "text": f"Chunk {i}",
                "chunk_char_start": 0,
                "chunk_char_end": 100,
            }
            points.append(point)
        
        mock_qdrant_client.scroll.return_value = (points, None)
        
        results = await vector_service.get_neighbor_chunks(
            section_id="sec_001",
            chunk_index=2,
            neighbor_range=2
        )
        
        # Results should be sorted
        indices = [r["chunk_index"] for r in results]
        assert indices == sorted(indices)
    
    @pytest.mark.asyncio
    async def test_get_neighbor_chunks_respects_min_index(self, vector_service, mock_qdrant_client):
        """Test that min_index doesn't go below 0."""
        points = []
        for i in range(3):
            point = Mock()
            point.payload = {
                "section_id": "sec_001",
                "chunk_index": i,
                "doc_id": "doc_001",
                "section_title": "Test",
                "page_start": 10,
                "page_end": 12,
                "text": f"Chunk {i}",
                "chunk_char_start": 0,
                "chunk_char_end": 100,
            }
            points.append(point)
        
        mock_qdrant_client.scroll.return_value = (points, None)
        
        # Get neighbors of chunk 0 with range 2
        results = await vector_service.get_neighbor_chunks(
            section_id="sec_001",
            chunk_index=0,
            neighbor_range=2
        )
        
        # Should include chunks 0, 1, 2 (not -2, -1)
        indices = [r["chunk_index"] for r in results]
        assert all(i >= 0 for i in indices)


# =============================================================================
# COUNT AND INFO TESTS
# =============================================================================

class TestCountAndInfo:
    """Test count and info operations."""
    
    @pytest.mark.asyncio
    async def test_count_text_chunks(self, vector_service, mock_qdrant_client):
        """Test counting text chunks for a document."""
        mock_qdrant_client.count.return_value = Mock(count=42)
        
        count = await vector_service.count_text_chunks("doc_001")
        
        assert count == 42
        mock_qdrant_client.count.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_count_text_chunks_error_handling(self, vector_service, mock_qdrant_client):
        """Test count returns 0 on error."""
        mock_qdrant_client.count.side_effect = Exception("Connection error")
        
        count = await vector_service.count_text_chunks("doc_001")
        
        assert count == 0
    
    def test_get_collection_info(self, vector_service, mock_qdrant_client):
        """Test getting collection info."""
        mock_qdrant_client.count.return_value = Mock(count=100)
        
        info = vector_service.get_collection_info()
        
        assert "text_chunks" in info
        assert "schemas" in info
        assert "tables" in info
        assert "summary" in info
    
    def test_get_collection_info_with_errors(self, vector_service, mock_qdrant_client):
        """Test collection info handles errors gracefully."""
        mock_qdrant_client.count.side_effect = Exception("Error")
        
        info = vector_service.get_collection_info()
        
        # Should still return info with error status
        assert info["text_chunks"]["status"] == "error"


# =============================================================================
# FILTER CONSTRUCTION TESTS
# =============================================================================

class TestFilterConstruction:
    """Test filter construction for searches."""
    
    @pytest.mark.asyncio
    async def test_no_filter_when_no_params(self, vector_service, mock_qdrant_client):
        """Test that no filter is applied when no params given."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_text("query", limit=10)
        
        call_kwargs = mock_qdrant_client.search.call_args[1]
        # query_filter should be None when no filters
        assert call_kwargs["query_filter"] is None
    
    @pytest.mark.asyncio
    async def test_filter_with_multiple_conditions(self, vector_service, mock_qdrant_client):
        """Test filter with both doc_id and owner."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_text(
            "query",
            doc_id="doc_001",
            owner="user_001",
            limit=10
        )
        
        call_kwargs = mock_qdrant_client.search.call_args[1]
        assert call_kwargs["query_filter"] is not None


# =============================================================================
# EMBEDDING CACHING TESTS
# =============================================================================

class TestEmbeddingCaching:
    """Test embedding service usage."""
    
    @pytest.mark.asyncio
    async def test_embedding_called_for_search(self, vector_service, mock_embedding_service, mock_qdrant_client):
        """Test that embedding service is called for search queries."""
        mock_qdrant_client.search.return_value = []
        
        await vector_service.search_text("test query")
        
        mock_embedding_service.create_embedding.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_embedding_called_for_add(self, vector_service, mock_embedding_service, sample_text_chunk):
        """Test that embedding service is called when adding chunks."""
        await vector_service.add_text_chunk(**sample_text_chunk)
        
        mock_embedding_service.create_embedding.assert_called_once_with(
            sample_text_chunk["text"]
        )


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, vector_service, mock_embedding_service, mock_qdrant_client):
        """Test search with empty query."""
        mock_qdrant_client.search.return_value = []
        
        results = await vector_service.search_text("")
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_add_chunk_with_empty_text(self, vector_service, mock_embedding_service):
        """Test adding chunk with empty text."""
        await vector_service.add_text_chunk(
            section_id="sec_001",
            chunk_index=0,
            text="",  # Empty
            doc_id="doc_001",
            doc_title="Test",
            page_start=1,
            page_end=1,
            chunk_char_start=0,
            chunk_char_end=0,
        )
        
        # Should still call embedding (even if empty)
        mock_embedding_service.create_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_with_special_characters(self, vector_service, mock_qdrant_client):
        """Test search with special characters in query."""
        mock_qdrant_client.search.return_value = []
        
        # Should not raise
        results = await vector_service.search_text("PU-101 & valve (test)")
        
        assert results == []
    
    def test_delete_nonexistent_document(self, vector_service, mock_qdrant_client):
        """Test deleting non-existent document doesn't raise."""
        mock_qdrant_client.delete.return_value = Mock(status="ok")
        
        # Should not raise
        vector_service.delete_document_vectors("nonexistent_doc")
    
    def test_delete_handles_errors_gracefully(self, vector_service, mock_qdrant_client):
        """Test that delete operations handle errors."""
        mock_qdrant_client.delete.side_effect = Exception("Connection error")
        
        # Should not raise, just log error
        vector_service.delete_document_vectors("doc_001")


# =============================================================================
# PAYLOAD FORMATTING TESTS
# =============================================================================

class TestPayloadFormatting:
    """Test result payload formatting consistency."""
    
    @pytest.mark.asyncio
    async def test_text_search_result_fields(self, vector_service, mock_qdrant_client, mock_search_result):
        """Test all expected fields in text search results."""
        mock_qdrant_client.search.return_value = [mock_search_result]
        
        results = await vector_service.search_text("query")
        result = results[0]
        
        expected_fields = [
            "section_id", "chunk_index", "doc_id", "section_number",
            "section_title", "page_start", "page_end", "score",
            "text", "text_preview", "chunk_char_start", "chunk_char_end",
            "char_count", "system_ids", "entity_ids"
        ]
        
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"
    
    @pytest.mark.asyncio
    async def test_score_is_float(self, vector_service, mock_qdrant_client, mock_search_result):
        """Test that score is a float."""
        mock_qdrant_client.search.return_value = [mock_search_result]
        
        results = await vector_service.search_text("query")
        
        assert isinstance(results[0]["score"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
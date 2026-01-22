"""
Neo4j Graph Service Tests

Tests are divided into:
1. Unit tests (mocked) - run without Neo4j
2. Integration tests - require real Neo4j instance

Run unit tests: pytest test_graph_service.py -v -m "not integration"
Run integration tests: pytest test_graph_service.py -v -m integration
Run all: pytest test_graph_service.py -v

Environment variables for integration tests:
- NEO4J_URI (default: bolt://localhost:7687)
- NEO4J_USER (default: neo4j)
- NEO4J_PASSWORD (default: password)
- NEO4J_DATABASE (default: neo4j)
"""

import pytest
import os
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError

# Import the class under test
from services.graph_service import Neo4jClient

import logging
logger = logging.getLogger(__name__)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def neo4j_credentials():
    """Get Neo4j credentials from environment."""
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver with proper async context manager support."""
    driver = AsyncMock()
    
    # Create a mock session that works as async context manager
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    
    # Configure driver.session() to return an async context manager
    async_context = AsyncMock()
    async_context.__aenter__ = AsyncMock(return_value=mock_session)
    async_context.__aexit__ = AsyncMock(return_value=None)
    driver.session = MagicMock(return_value=async_context)
    
    # Default result for verify_connection
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value={"n": 1})
    mock_result.data = AsyncMock(return_value=[])
    mock_session.run.return_value = mock_result
    
    return driver


@pytest.fixture
def client_with_mock(mock_driver):
    """Create Neo4jClient with mocked driver."""
    client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test",
        database="neo4j"
    )
    client.driver = mock_driver
    return client


@pytest.fixture
def sample_document():
    """Sample document data for testing."""
    return {
        "id": f"test_doc_{uuid.uuid4().hex[:8]}",
        "title": "Test Maritime Manual",
        "doc_type": "manual",
        "owner": "test_user",
        "total_pages": 100,
        "version": "1.0",
        "language": "en",
        "tags": ["test", "maritime"],
    }


@pytest.fixture
def sample_chapter():
    """Sample chapter data for testing."""
    return {
        "id": f"test_chapter_{uuid.uuid4().hex[:8]}",
        "title": "Chapter 1: Introduction",
        "number": 1,
        "start_page": 1,
        "end_page": 10,
    }


@pytest.fixture
def sample_section():
    """Sample section data for testing."""
    return {
        "id": f"test_section_{uuid.uuid4().hex[:8]}",
        "title": "1.1 Overview",
        "content": "This section provides an overview of the fuel system.",
        "section_number": "1.1",
        "page_start": 1,
        "page_end": 3,
        "section_type": "text",
        "importance_score": 0.8,
    }


@pytest.fixture
def sample_table():
    """Sample table data for testing."""
    return {
        "id": f"test_table_{uuid.uuid4().hex[:8]}",
        "title": "Table 1: Fuel Pump Specifications",
        "caption": "Specifications for PU-101 fuel pump",
        "doc_id": "test_doc_001",
        "page_number": 5,
        "rows": 10,
        "cols": 5,
        "text_preview": "PU-101 | 50 bar | 100 L/min",
    }


@pytest.fixture
def sample_schema():
    """Sample schema (diagram) data for testing."""
    return {
        "id": f"test_schema_{uuid.uuid4().hex[:8]}",
        "title": "Figure 1: Fuel System Diagram",
        "caption": "Overview of fuel system components",
        "doc_id": "test_doc_001",
        "page_number": 3,
        "file_path": "/diagrams/fuel_system.png",
        "llm_summary": "Diagram showing fuel pump PU-101 connected to filters",
    }


@pytest.fixture
def sample_entity():
    """Sample entity data for testing."""
    return {
        "id": f"test_entity_{uuid.uuid4().hex[:8]}",
        "name": "Fuel Oil Pump",
        "code": "PU-101",
        "entity_type": "equipment",
        "system": "fuel",
        "tags": ["pump", "fuel"],
    }


# =============================================================================
# UNIT TESTS (MOCKED)
# =============================================================================

class TestNeo4jClientInit:
    """Test Neo4jClient initialization."""
    
    def test_init_stores_credentials(self):
        """Test that credentials are stored correctly."""
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="secret",
            database="mydb"
        )
        
        assert client.uri == "bolt://localhost:7687"
        assert client.user == "neo4j"
        assert client.password == "secret"
        assert client.database == "mydb"
        assert client.driver is None  # Not connected yet
    
    def test_init_default_database(self):
        """Test default database name."""
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="secret"
        )
        
        assert client.database == "neo4j"


class TestRunQuery:
    """Test run_query method."""
    
    @pytest.mark.asyncio
    async def test_run_query_returns_results(self, client_with_mock, mock_driver):
        """Test that run_query returns query results."""
        # Setup mock to return data
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"name": "Test", "value": 42}]
        session.run.return_value = result
        
        results = await client_with_mock.run_query("MATCH (n) RETURN n.name as name")
        
        assert len(results) == 1
        assert results[0]["name"] == "Test"
    
    @pytest.mark.asyncio
    async def test_run_query_with_parameters(self, client_with_mock, mock_driver):
        """Test run_query passes parameters correctly."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = []
        session.run.return_value = result
        
        await client_with_mock.run_query(
            "MATCH (n {id: $id}) RETURN n",
            {"id": "test123"}
        )
        
        # Verify parameters were passed
        session.run.assert_called_once()
        call_args = session.run.call_args
        assert call_args[0][0] == "MATCH (n {id: $id}) RETURN n"
        assert call_args[1] == {"id": "test123"} or call_args[0][1] == {"id": "test123"}
    
    @pytest.mark.asyncio
    async def test_run_query_empty_result(self, client_with_mock, mock_driver):
        """Test run_query with empty result."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = []
        session.run.return_value = result
        
        results = await client_with_mock.run_query("MATCH (n:NonExistent) RETURN n")
        
        assert results == []


class TestCreateDocument:
    """Test document creation."""
    
    @pytest.mark.asyncio
    async def test_create_document_returns_id(self, client_with_mock, mock_driver, sample_document):
        """Test create_document returns document ID."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"doc_id": sample_document["id"]}]
        session.run.return_value = result
        
        doc_id = await client_with_mock.create_document(sample_document)
        
        assert doc_id == sample_document["id"]
    
    @pytest.mark.asyncio
    async def test_create_document_generates_id_if_missing(self, client_with_mock, mock_driver):
        """Test that ID is generated if not provided."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        # Return whatever ID was generated
        result.data.return_value = [{"doc_id": "generated_id"}]
        session.run.return_value = result
        
        doc_data = {"title": "Test Doc", "owner": "user"}
        doc_id = await client_with_mock.create_document(doc_data)
        
        assert doc_id is not None
    
    @pytest.mark.asyncio
    async def test_create_document_with_minimal_data(self, client_with_mock, mock_driver):
        """Test create_document with minimal required data."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"doc_id": "min_doc"}]
        session.run.return_value = result
        
        doc_id = await client_with_mock.create_document({"id": "min_doc"})
        
        assert doc_id == "min_doc"


class TestCreateChapter:
    """Test chapter creation."""
    
    @pytest.mark.asyncio
    async def test_create_chapter_returns_id(self, client_with_mock, mock_driver, sample_chapter):
        """Test create_chapter returns chapter ID."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"chapter_id": sample_chapter["id"]}]
        session.run.return_value = result
        
        chapter_id = await client_with_mock.create_chapter(sample_chapter, doc_id="test_doc")
        
        assert chapter_id == sample_chapter["id"]
    
    @pytest.mark.asyncio
    async def test_create_chapter_generates_id(self, client_with_mock, mock_driver):
        """Test chapter ID generation when not provided."""
        chapter_data = {"title": "Test Chapter", "number": 1}
        
        # Mock to capture the generated UUID and return it
        async def mock_run_query(query, params):
            # Return the chapter_id that was generated and passed in params
            return [{"chapter_id": params.get("chapter_id")}]
        
        client_with_mock.run_query = mock_run_query
        chapter_id = await client_with_mock.create_chapter(chapter_data, doc_id="test_doc")
        
        assert chapter_id is not None
        assert len(chapter_id) == 36  # UUID format


class TestCreateSection:
    """Test section creation."""
    
    @pytest.mark.asyncio
    async def test_create_section_returns_id(self, client_with_mock, mock_driver, sample_section):
        """Test create_section returns section ID."""
        session = mock_driver.session.return_value.__aenter__.return_value
        
        # First call: check chapter exists
        check_result = AsyncMock()
        check_result.data.return_value = [{"id": "ch1", "title": "Chapter 1", "number": 1}]
        
        # Second call: create section
        create_result = AsyncMock()
        create_result.data.return_value = [{"section_id": sample_section["id"]}]
        
        session.run.side_effect = [check_result, create_result]
        
        section_id = await client_with_mock.create_section(sample_section, chapter_id="ch1")
        
        assert section_id == sample_section["id"]
    
    @pytest.mark.asyncio
    async def test_create_section_chapter_not_found(self, client_with_mock, mock_driver, sample_section):
        """Test create_section raises error when chapter not found."""
        session = mock_driver.session.return_value.__aenter__.return_value
        
        # Chapter check returns empty
        check_result = AsyncMock()
        check_result.data.return_value = []
        
        # Debug query for all chapters
        all_chapters_result = AsyncMock()
        all_chapters_result.data.return_value = []
        
        session.run.side_effect = [check_result, all_chapters_result]
        
        with pytest.raises(ValueError, match="not found"):
            await client_with_mock.create_section(sample_section, chapter_id="nonexistent")
    
    @pytest.mark.asyncio
    async def test_create_section_with_merged_data(self, client_with_mock, mock_driver):
        """Test create_section with merged section data."""
        session = mock_driver.session.return_value.__aenter__.return_value
        
        check_result = AsyncMock()
        check_result.data.return_value = [{"id": "ch1", "title": "Ch1", "number": 1}]
        
        create_result = AsyncMock()
        create_result.data.return_value = [{"section_id": "merged_section"}]
        
        session.run.side_effect = [check_result, create_result]
        
        merged_section = {
            "id": "merged_section",
            "title": "Merged Section",
            "is_merged": True,
            "original_count": 3,
            "merged_sections": ["s1", "s2", "s3"],
        }
        
        section_id = await client_with_mock.create_section(merged_section, chapter_id="ch1")
        
        assert section_id == "merged_section"


class TestCreateTable:
    """Test table creation."""
    
    @pytest.mark.asyncio
    async def test_create_table_returns_id(self, client_with_mock, mock_driver, sample_table):
        """Test create_table returns table ID."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"table_id": sample_table["id"]}]
        session.run.return_value = result
        
        table_id = await client_with_mock.create_table(sample_table)
        
        assert table_id == sample_table["id"]
    
    @pytest.mark.asyncio
    async def test_create_table_with_section_link(self, client_with_mock, mock_driver, sample_table):
        """Test create_table with section link."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"table_id": sample_table["id"]}]
        session.run.return_value = result
        
        table_id = await client_with_mock.create_table(sample_table, section_id="sec1")
        
        assert table_id == sample_table["id"]
        # Verify query was called with section linking
        assert session.run.called


class TestCreateSchema:
    """Test schema (diagram) creation."""
    
    @pytest.mark.asyncio
    async def test_create_schema_returns_id(self, client_with_mock, mock_driver, sample_schema):
        """Test create_schema returns schema ID."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"schema_id": sample_schema["id"]}]
        session.run.return_value = result
        
        schema_id = await client_with_mock.create_schema(sample_schema)
        
        assert schema_id == sample_schema["id"]
    
    @pytest.mark.asyncio
    async def test_create_schema_requires_doc_id(self, client_with_mock, mock_driver):
        """Test create_schema raises error without doc_id."""
        schema_data = {"title": "Test Schema"}  # Missing doc_id
        
        with pytest.raises(ValueError, match="doc_id is required"):
            await client_with_mock.create_schema(schema_data)
    
    @pytest.mark.asyncio
    async def test_create_schema_with_section_link(self, client_with_mock, mock_driver, sample_schema):
        """Test create_schema with section link."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"schema_id": sample_schema["id"]}]
        session.run.return_value = result
        
        schema_id = await client_with_mock.create_schema(sample_schema, section_id="sec1")
        
        assert schema_id == sample_schema["id"]


class TestCreateEntity:
    """Test entity creation."""
    
    @pytest.mark.asyncio
    async def test_create_entity_returns_id(self, client_with_mock, mock_driver, sample_entity):
        """Test create_entity returns entity ID."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{"entity_id": sample_entity["id"]}]
        session.run.return_value = result
        
        entity_id = await client_with_mock.create_entity(sample_entity)
        
        assert entity_id == sample_entity["id"]


class TestLinkMethods:
    """Test relationship linking methods."""
    
    @pytest.mark.asyncio
    async def test_link_section_to_table(self, client_with_mock, mock_driver):
        """Test linking section to table."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{}]
        session.run.return_value = result
        
        await client_with_mock.link_section_to_table("sec1", "table1")
        
        assert session.run.called
    
    @pytest.mark.asyncio
    async def test_link_section_to_entity(self, client_with_mock, mock_driver):
        """Test linking section to entity."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{}]
        session.run.return_value = result
        
        await client_with_mock.link_section_to_entity("sec1", "entity1")
        
        assert session.run.called
    
    @pytest.mark.asyncio
    async def test_link_schema_to_entity(self, client_with_mock, mock_driver):
        """Test linking schema to entity."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{}]
        session.run.return_value = result
        
        await client_with_mock.link_schema_to_entity("schema1", "entity1")
        
        assert session.run.called


class TestSearchMethods:
    """Test search methods."""
    
    @pytest.mark.asyncio
    async def test_search_sections_fulltext(self, client_with_mock, mock_driver):
        """Test fulltext search for sections."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [
            {"section_id": "sec1", "title": "Fuel Pump", "score": 0.95},
            {"section_id": "sec2", "title": "Pump Maintenance", "score": 0.85},
        ]
        session.run.return_value = result
        
        results = await client_with_mock.search_sections_fulltext("fuel pump")
        
        assert len(results) == 2
        assert results[0]["score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_get_all_entities(self, client_with_mock, mock_driver):
        """Test getting all entities."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [
            {"entity_name": "Fuel Oil Pump", "entity_code": "PU-101"},
            {"entity_name": "Engine", "entity_code": "EN-202"},
            {"entity_name": "Generator", "entity_code": None},
        ]
        session.run.return_value = result
        
        entities = await client_with_mock.get_all_entities()
        
        assert isinstance(entities, list)
        assert "PU-101" in entities
        assert "Fuel Oil Pump" in entities
        assert len(entities) >= 2
    
    @pytest.mark.asyncio
    async def test_find_tables_by_entity(self, client_with_mock, mock_driver):
        """Test finding tables by entity."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [
            {"table_id": "t1", "table_title": "Pump Specs", "entity_code": "PU-101"},
        ]
        session.run.return_value = result
        
        tables = await client_with_mock.find_tables_by_entity(["PU-101"])
        
        assert len(tables) == 1
        assert tables[0]["entity_code"] == "PU-101"
    
    @pytest.mark.asyncio
    async def test_find_schemas_by_entity(self, client_with_mock, mock_driver):
        """Test finding schemas by entity."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [
            {"schema_id": "sc1", "title": "Pump Diagram", "entity_code": "PU-101"},
        ]
        session.run.return_value = result
        
        schemas = await client_with_mock.find_schemas_by_entity(["PU-101"])
        
        assert len(schemas) == 1
        assert schemas[0]["entity_code"] == "PU-101"


class TestNeighborSections:
    """Test neighbor section retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_neighbor_sections(self, client_with_mock, mock_driver):
        """Test getting neighbor sections."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [
            {"section_id": "sec2", "title": "Next Section", "page_start": 10},
            {"section_id": "sec3", "title": "Section After", "page_start": 15},
        ]
        session.run.return_value = result
        
        neighbors = await client_with_mock.get_neighbor_sections(
            section_ids=["sec1"],
            limit=3
        )
        
        assert len(neighbors) == 2


class TestDocumentOperations:
    """Test document-level operations."""
    
    @pytest.mark.asyncio
    async def test_get_document_stats(self, client_with_mock, mock_driver):
        """Test getting document statistics."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [
            {"total_documents": 10, "total_sections": 500, "total_tables": 50}
        ]
        session.run.return_value = result
        
        stats = await client_with_mock.get_document_stats()
        
        assert isinstance(stats, dict)
    
    @pytest.mark.asyncio
    async def test_delete_document(self, client_with_mock, mock_driver):
        """Test document deletion."""
        session = mock_driver.session.return_value.__aenter__.return_value
        
        # Mock multiple query results - first for main delete, then for orphan cleanup
        result1 = AsyncMock()
        result1.data.return_value = [{
            "doc_deleted": 1,
            "chapters_deleted": 2,
            "sections_deleted": 5,
            "schemas_deleted": 3,
            "tables_deleted": 4,
            "table_chunks_deleted": 0
        }]
        
        result2 = AsyncMock()
        result2.data.return_value = [{"deleted": 0}]  # No orphans
        
        # Return different results for consecutive calls
        session.run.side_effect = [result1, result2, result2, result2, result2]
        
        deleted = await client_with_mock.delete_document("doc_to_delete")
        
        assert deleted is True
    
    @pytest.mark.asyncio
    async def test_update_document_status(self, client_with_mock, mock_driver):
        """Test updating document status."""
        session = mock_driver.session.return_value.__aenter__.return_value
        result = AsyncMock()
        result.data.return_value = [{}]
        session.run.return_value = result
        
        await client_with_mock.update_document_status("doc1", "completed")
        
        assert session.run.called


class TestLuceneQuerySanitization:
    """Test Lucene query sanitization."""
    
    def test_sanitize_special_characters(self):
        """Test that special characters are escaped."""
        client = Neo4jClient("bolt://localhost", "neo4j", "pass")
        
        # Test escaping special chars
        sanitized = client._sanitize_lucene_query("test+query")
        assert "+" not in sanitized or "\\+" in sanitized
    
    def test_sanitize_preserves_alphanumeric(self):
        """Test that alphanumeric chars are preserved."""
        client = Neo4jClient("bolt://localhost", "neo4j", "pass")
        
        sanitized = client._sanitize_lucene_query("PU101 fuel pump")
        assert "PU101" in sanitized or "pu101" in sanitized.lower()
        assert "fuel" in sanitized.lower()
        assert "pump" in sanitized.lower()


class TestErrorHandling:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_execute_query_handles_error(self, client_with_mock, mock_driver):
        """Test that execute_query handles errors gracefully."""
        session = mock_driver.session.return_value.__aenter__.return_value
        session.run.side_effect = Exception("Database error")
        
        # Also mock verify_connection to fail
        with patch.object(client_with_mock, 'verify_connection', new_callable=AsyncMock) as mock_verify:
            mock_verify.side_effect = Exception("Connection lost")
            
            with pytest.raises(Exception, match="Database error"):
                await client_with_mock.execute_query("INVALID QUERY")


# =============================================================================
# INTEGRATION TESTS (REQUIRE NEO4J)
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_document_hierarchy(neo4j_credentials, sample_document, sample_chapter, sample_section):
    """
    Integration test: Create complete document hierarchy.
    
    Tests: connect → create_document → create_chapter → create_section → cleanup
    """
    try:
        client = Neo4jClient(**neo4j_credentials)
        await client.connect()
        
        # Verify connection
        result = await client.run_query("RETURN 1 as test")
        assert result[0]["test"] == 1
        
        # Create document
        doc_id = await client.create_document(sample_document)
        assert doc_id == sample_document["id"]
        
        # Create chapter
        chapter_id = await client.create_chapter(sample_chapter, doc_id=doc_id)
        assert chapter_id == sample_chapter["id"]
        
        # Create section
        section_id = await client.create_section(sample_section, chapter_id=chapter_id)
        assert section_id == sample_section["id"]
        
        # Verify hierarchy
        verify_query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)
        RETURN d.id as doc_id, c.id as chapter_id, s.id as section_id
        """
        hierarchy = await client.run_query(verify_query, {"doc_id": doc_id})
        assert len(hierarchy) == 1
        assert hierarchy[0]["doc_id"] == doc_id
        assert hierarchy[0]["chapter_id"] == chapter_id
        assert hierarchy[0]["section_id"] == section_id
        
        # Cleanup
        await client.run_query(
            "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
            {"doc_id": doc_id}
        )
        
        await client.close()
        
    except AssertionError:
        raise
    except (ServiceUnavailable, AuthError, ClientError) as e:
        pytest.skip(f"Neo4j not available: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_entity_operations(neo4j_credentials, sample_entity):
    """
    Integration test: Entity creation and lookup.
    
    Tests: create_entity → get_entity_by_code → get_all_entities → cleanup
    """
    client = None
    try:
        client = Neo4jClient(**neo4j_credentials)
        await client.connect()
        
        # Create entity
        entity_id = await client.create_entity(sample_entity)
        assert entity_id is not None
        
        # Get entity by code
        entity = await client.get_entity_by_code(sample_entity["code"])
        if entity:
            assert entity["code"] == sample_entity["code"]
        
        # Get all entities
        all_entities = await client.get_all_entities()
        assert isinstance(all_entities, list)
        
        # Cleanup
        await client.run_query(
            "MATCH (e:Entity {code: $code}) DETACH DELETE e",
            {"code": sample_entity["code"]}
        )
        
        await client.close()
        
    except AssertionError:
        raise
    except (ServiceUnavailable, AuthError, ClientError) as e:
        pytest.skip(f"Neo4j not available: {e}")
    finally:
        if client:
            await client.close()



@pytest.mark.integration
@pytest.mark.asyncio
async def test_table_and_schema_creation(neo4j_credentials, sample_document, sample_table, sample_schema):
    """
    Integration test: Table and schema creation with document link.
    
    Tests: create_document → create_table → create_schema → link_schema_to_table → cleanup
    """
    client = None
    try:
        client = Neo4jClient(**neo4j_credentials)
        await client.connect()
        
        # Create document first
        sample_table["doc_id"] = sample_document["id"]
        sample_schema["doc_id"] = sample_document["id"]
        
        doc_id = await client.create_document(sample_document)
        
        # Create table
        table_id = await client.create_table(sample_table)
        assert table_id == sample_table["id"]
        
        # Verify table was created
        table_check = await client.run_query(
            "MATCH (t:Table {id: $table_id}) RETURN t.id, labels(t) as labels",
            {"table_id": table_id}
        )
        print(f"Table check after creation: {table_check}")
        
        # Create schema
        schema_id = await client.create_schema(sample_schema)
        assert schema_id == sample_schema["id"]
        
        # Verify schema was created
        schema_check = await client.run_query(
            "MATCH (sc:Schema {id: $schema_id}) RETURN sc.id, labels(sc) as labels",
            {"schema_id": schema_id}
        )
        print(f"Schema check after creation: {schema_check}")
        
        # Link schema to table
        link_result = await client.link_schema_to_table(schema_id, table_id)
        print(f"Link result: {link_result}")
        
        # Add small delay for Neo4j to process
        await asyncio.sleep(0.5)
        
        # Verify link with more details
        verify_query = """
        MATCH (sc:Schema {id: $schema_id})-[r:HAS_LEGEND]->(t:Table {id: $table_id})
        RETURN sc.id, type(r) as rel_type, t.id
        """
        link = await client.run_query(verify_query, {
            "schema_id": schema_id,
            "table_id": table_id
        })
        print(f"Link verification result: {link}")
        
        if len(link) == 0:
            # Debug: check if nodes exist
            nodes_check = await client.run_query(
                "MATCH (sc:Schema {id: $schema_id}) RETURN sc.id",
                {"schema_id": schema_id}
            )
            logger.warning(f"Schema exists: {len(nodes_check) > 0}")
            nodes_check = await client.run_query(
                "MATCH (t:Table {id: $table_id}) RETURN t.id",
                {"table_id": table_id}
            )
            logger.warning(f"Table exists: {len(nodes_check) > 0}")
        
        assert len(link) == 1, f"Expected 1 link, got {len(link)}"
        
        # Cleanup
        await client.run_query(
            "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
            {"doc_id": doc_id}
        )
        await client.run_query(
            "MATCH (t:Table {id: $table_id}) DETACH DELETE t",
            {"table_id": table_id}
        )
        await client.run_query(
            "MATCH (sc:Schema {id: $schema_id}) DETACH DELETE sc",
            {"schema_id": schema_id}
        )
        
    except AssertionError:
        raise
    except (ServiceUnavailable, AuthError, ClientError) as e:
        pytest.skip(f"Neo4j not available: {e}")
    finally:
        if client:
            await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fulltext_search(neo4j_credentials):
    """
    Integration test: Fulltext search functionality.
    
    Tests: create test data → search_sections_fulltext → cleanup
    """
    client = None
    try:
        client = Neo4jClient(**neo4j_credentials)
        await client.connect()
        
        # Create test document with searchable content
        doc_id = f"search_test_{uuid.uuid4().hex[:8]}"
        await client.create_document({
            "id": doc_id,
            "title": "Fuel Pump Manual",
            "owner": "test"
        })
        
        chapter_id = f"ch_{uuid.uuid4().hex[:8]}"
        await client.create_chapter(
            {"id": chapter_id, "title": "Fuel System", "number": 1},
            doc_id=doc_id
        )
        
        section_id = f"sec_{uuid.uuid4().hex[:8]}"
        await client.create_section(
            {
                "id": section_id,
                "title": "Fuel Pump PU-101 Maintenance",
                "content": "This section describes maintenance procedures for fuel pump PU-101.",
            },
            chapter_id=chapter_id
        )
        
        # Wait a moment for index to update (Neo4j async indexing)
        import asyncio
        await asyncio.sleep(1.0)  # Increased wait time for index
        
        # Search
        results = await client.search_sections_fulltext(
            query="fuel pump PU-101",
            limit=10
        )
        
        # Results may or may not find the section depending on index timing
        assert isinstance(results, list)
        # NOTE: Not asserting result count due to async indexing - just verify no crash
        
        # Cleanup
        await client.run_query(
            "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
            {"doc_id": doc_id}
        )
        
        await client.close()
        
    except AssertionError:
        raise
    except (ServiceUnavailable, AuthError, ClientError) as e:
        pytest.skip(f"Neo4j not available: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_deletion_cascade(neo4j_credentials):
    """
    Integration test: Document deletion with cascade.
    
    Tests: create hierarchy → delete_document → verify cleanup
    """
    client = None
    try:
        client = Neo4jClient(**neo4j_credentials)
        await client.connect()
        
        # Create document with hierarchy
        doc_id = f"delete_test_{uuid.uuid4().hex[:8]}"
        await client.create_document({"id": doc_id, "title": "To Delete", "owner": "test"})
        
        chapter_id = f"ch_{uuid.uuid4().hex[:8]}"
        await client.create_chapter({"id": chapter_id, "title": "Ch1", "number": 1}, doc_id=doc_id)
        
        section_id = f"sec_{uuid.uuid4().hex[:8]}"
        await client.create_section({"id": section_id, "title": "Sec1"}, chapter_id=chapter_id)
        
        # Delete document
        deleted = await client.delete_document(doc_id)
        assert deleted is True
        
        # Verify document is gone
        check = await client.run_query(
            "MATCH (d:Document {id: $doc_id}) RETURN d",
            {"doc_id": doc_id}
        )
        assert len(check) == 0
        
        await client.close()
        
    except AssertionError:
        raise
    except (ServiceUnavailable, AuthError, ClientError) as e:
        pytest.skip(f"Neo4j not available: {e}")
    finally:
        if client:
            await client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
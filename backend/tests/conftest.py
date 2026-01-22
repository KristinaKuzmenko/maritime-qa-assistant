"""
Pytest configuration and fixtures for Maritime QA Assistant tests.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, MagicMock
import fitz  # PyMuPDF
from PIL import Image
import io
import sys
import os
from pathlib import Path

from dotenv import load_dotenv

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Load .env from project root (do not override already-set env vars)
project_root = backend_dir.parent
load_dotenv(project_root / ".env", override=False)

from services.layout_analyzer import BBox, Region, RegionType


@pytest.fixture(autouse=True)
async def cancel_pending_tasks():
    """Cancel all pending async tasks after each test to prevent event loop errors."""
    yield
    try:
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task(loop)
        # Exclude current task to avoid recursion
        pending = [t for t in asyncio.all_tasks(loop) if not t.done() and t is not current_task]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    except RuntimeError:
        # No running loop, skip cleanup
        pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    # Give pending tasks time to complete
    try:
        loop.run_until_complete(asyncio.sleep(0.1))
    except:
        pass
    loop.close()


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box for testing."""
    return BBox(x0=100, y0=200, x1=300, y1=400)


@pytest.fixture
def sample_region():
    """Create a sample region for testing."""
    return Region(
        bbox=BBox(x0=100, y0=200, x1=300, y1=400),
        region_type=RegionType.SCHEMA,
        confidence=0.85,
        page_number=0,
    )


@pytest.fixture
def mock_pdf_page():
    """Create a mock PyMuPDF page object."""
    page = Mock(spec=fitz.Page)
    page.rect = fitz.Rect(0, 0, 595, 842)  # A4 size
    page.number = 0
    
    # Mock get_text
    page.get_text.return_value = "Sample page text"
    
    # Mock search_for
    page.search_for.return_value = [fitz.Rect(100, 100, 200, 120)]
    
    # Mock get_pixmap
    mock_pix = Mock()
    mock_pix.tobytes.return_value = b"fake_image_data"
    page.get_pixmap.return_value = mock_pix
    
    return page


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    model = Mock()
    
    # Mock prediction results
    mock_result = Mock()
    mock_box = Mock()
    mock_box.cls = [6]  # Picture class
    mock_box.conf = [0.85]
    mock_box.xyxy = [[100, 200, 300, 400]]
    
    mock_result.boxes = [mock_box]
    model.predict.return_value = [mock_result]
    
    return model


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock()
    
    # Mock chat completion
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "SCHEMA"
    client.chat.completions.create.return_value = mock_response
    
    return client


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    img = Image.new('RGB', (100, 100), color='white')
    return img


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    img = Image.new('RGB', (100, 100), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest_asyncio.fixture(scope="session")
async def setup_qdrant_test_collections():
    """
    Create Qdrant collections with test data for LOCAL Qdrant in benchmark_load.py.
    
    ⚠️ ONLY for benchmark_load.py infrastructure tests!
    
    benchmark_load.py  → Uses this fixture (local Qdrant + test data)
    benchmark_real.py  → Does NOT use this (production Qdrant Cloud)
    Other unit tests   → Does NOT use this (production Qdrant Cloud)
    
    Usage: Explicitly add to vector_service fixture in benchmark_load.py only.
    """
    qdrant_host = os.getenv("QDRANT_HOST")
    if not qdrant_host:
        # Skip if Qdrant not configured
        yield
        return
    
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    use_https = os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true"
    api_key = os.getenv("QDRANT_API_KEY") or None
    
    print(f"\n🔧 Setting up Qdrant test collections at {qdrant_host}:{qdrant_port}")
    
    client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        api_key=api_key,
        https=use_https,
        timeout=30
    )
    
    # Collection configurations
    collections = {
        "text_chunks": 1536,
        "tables": 1536,
        "schemas": 1536
    }
    
    for collection_name, vector_size in collections.items():
        # Recreate collection (clean slate)
        try:
            client.delete_collection(collection_name)
            print(f"  ♻️  Deleted existing collection: {collection_name}")
        except Exception:
            pass
        
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"  ✅ Created collection: {collection_name} (vector_size={vector_size})")
        
        # Insert 100 test vectors with proper payload structure
        points = []
        for i in range(100):
            # Create a simple but varied vector (not all zeros)
            vector = [(i + j) * 0.001 for j in range(vector_size)]
            
            # Create collection-specific payloads
            if collection_name == "text_chunks":
                payload = {
                    "section_id": f"test_section_{i % 20}",
                    "chunk_index": i,
                    "doc_id": f"test_doc_{i % 10}",
                    "section_number": f"{(i % 5) + 1}",
                    "section_title": f"Test Section {i % 20}",
                    "page_start": (i % 50) + 1,
                    "page_end": (i % 50) + 1,
                    "text": f"Test text chunk content {i}",
                    "text_preview": f"Test text chunk content {i}",
                    "chunk_char_start": 0,
                    "chunk_char_end": 100,
                    "char_count": 100,
                    "system_ids": [],
                    "entity_ids": []
                }
            elif collection_name == "tables":
                payload = {
                    "chunk_id": f"table_chunk_{i}",
                    "table_id": f"test_table_{i % 30}",
                    "chunk_index": i % 3,
                    "total_chunks": 3,
                    "doc_id": f"test_doc_{i % 10}",
                    "page": (i % 50) + 1,
                    "table_title": f"Test Table {i % 30}",
                    "table_caption": f"Caption for table {i % 30}",
                    "rows": 10,
                    "cols": 5,
                    "text_preview": f"Test table content {i}",
                    "char_count": 150,
                    "system_ids": [],
                    "entity_ids": []
                }
            else:  # schemas
                payload = {
                    "schema_id": f"test_schema_{i}",
                    "doc_id": f"test_doc_{i % 10}",
                    "page": (i % 50) + 1,
                    "caption": f"Test Schema {i}",
                    "text_preview": f"Test schema content {i}",
                    "char_count": 120,
                    "system_ids": [],
                    "entity_ids": []
                }
            
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload=payload
            ))
        
        client.upsert(collection_name=collection_name, points=points)
        print(f"  📊 Inserted 100 test vectors into {collection_name}")
    
    print("✅ All Qdrant test collections ready!\n")
    
    yield client
    
    # Cleanup after all tests (optional)
    print("\n🧹 Cleaning up test collections...")
    for collection_name in collections.keys():
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    
    client.close()

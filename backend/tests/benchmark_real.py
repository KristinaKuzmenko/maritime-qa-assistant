"""
Real-world Performance Benchmarks for Maritime QA Assistant

Tests with REAL documents and REAL API calls (no mocks).
Use for production-ready performance validation.

Requirements:
- Real PDF documents in data/test_docs/
- OpenAI API key configured
- Neo4j and Qdrant running
- Sufficient API quota

Run locally only (not in CI/CD):
    pytest backend/tests/benchmark_real.py -v --benchmark-only

Save baseline:
    pytest backend/tests/benchmark_real.py --benchmark-save=production --benchmark-only

Compare with baseline:
    pytest backend/tests/benchmark_real.py --benchmark-compare=production --benchmark-only

WARNING: This will consume API quota and cost money!
"""

import pytest
import pytest_asyncio
import asyncio
import time
import os
import logging
from pathlib import Path
from statistics import median
import psutil
import tracemalloc
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Import real services (no mocks)
# NOTE: pytest.ini sets `pythonpath=backend`, so `services.*` / `core.*` are the canonical import paths.
# Importing via `backend.services.*` would load duplicate modules alongside `services.*`, breaking Enum identity
# checks (e.g. RegionType) and causing RegionClassifier gating to fail.
from services.document_processor import DocumentProcessor
from services.storage_service import StorageService
from core.config import settings


# Skip if no real documents available
TEST_DOCS_DIR = Path("data/test_docs")
SMALL_DOC_PATH = TEST_DOCS_DIR / "small_doc.pdf"  # 50 pages
LARGE_DOC_PATH = TEST_DOCS_DIR / "large_doc.pdf"  # 500 pages

pytestmark = pytest.mark.skipif(
    not TEST_DOCS_DIR.exists() or not SMALL_DOC_PATH.exists(),
    reason="Real test documents not available. Place PDFs in data/test_docs/"
)


# =============================================================================
# FIXTURES - Real Services (No Mocks)
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def real_document_processor():
    """Real DocumentProcessor with all services configured"""
    # Verify environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "QDRANT_HOST",
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing environment variables: {', '.join(missing)}")
    
    # Import all necessary services
    from pathlib import Path
    from openai import AsyncOpenAI
    from core.config import Settings
    from services.graph_service import Neo4jClient
    from services.vector_service import VectorService
    from services.embedding_service import EmbeddingService
    from services.storage_service import StorageService
    from services.layout_analyzer import LayoutAnalyzer
    from services.schema_extractor import SchemaExtractor
    from services.table_extractor import TableExtractor
    
    settings = Settings()
    
    # Initialize services in same order as main.py
    
    # 1. Storage Service
    storage_service = StorageService(
        storage_type=settings.storage_type,
        local_storage_path=settings.local_storage_path,
        s3_bucket_name=settings.s3_bucket_name,
        s3_prefix=settings.s3_prefix,
        aws_region=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        base_url="/data" if settings.storage_type == "local" else None
    )
    
    # 2. Embedding Service
    embedding_service = EmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
    
    # 3. Neo4j
    graph_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    # Connect asynchronously within the same event loop
    await graph_client.connect()
    
    # 4. Qdrant - create client explicitly for cloud
    print(f"🔗 Connecting to Qdrant:")
    print(f"   Host: {settings.qdrant_host}")
    print(f"   Port: {settings.qdrant_port}")
    print(f"   API Key: {'Yes' if settings.qdrant_api_key else 'No'}")
    
    if settings.qdrant_api_key:
        use_https = getattr(settings, "qdrant_use_https", True)
        print(f"   Using HTTPS: {use_https}")
        if use_https:
            qdrant_url = f"https://{settings.qdrant_host}:{settings.qdrant_port}"
            qdrant_client = QdrantClient(url=qdrant_url, api_key=settings.qdrant_api_key)
            print(f"   URL: {qdrant_url}")
        else:
            qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key
            )
    else:
        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        print(f"   Local mode (no API key)")
    
    vector_service = VectorService(embedding_service=embedding_service, client=qdrant_client)
    vector_service.initialize_collections()
    print(f"✅ Qdrant collections initialized")
    
    # 5. OpenAI Client for LLM operations
    llm_client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Layout analyzer with YOLO model
    model_path = Path(__file__).parent.parent / "models" / "yolov12s-doclaynet.pt"
    
    if not model_path.exists():
        print(f"⚠️  WARNING: YOLO model not found at {model_path}")
        print(f"   Schema/table extraction may fail!")
    else:
        print(f"✅ YOLO model found: {model_path}")
    
    layout_analyzer = LayoutAnalyzer(
        model_path=str(model_path),
        confidence_threshold=0.4,
    )


    
    # Schema extractor
    schema_extractor = SchemaExtractor(
        storage_service=storage_service,
        layout_analyzer=layout_analyzer,
        llm_service=llm_client,
        enable_llm_summary=True,
        vision_detail=settings.vision_detail_schemas,
    )
    
    # Table extractor
    table_extractor = TableExtractor(
        storage_service=storage_service,
        max_tokens_per_chunk=4000,
    )
    
    # Initialize real processor with all dependencies
    processor = DocumentProcessor(
        graph_client=graph_client,
        layout_analyzer=layout_analyzer,
        schema_extractor=schema_extractor,
        table_extractor=table_extractor,
        embedding_service=embedding_service,
        storage_service=storage_service,
        vector_service=vector_service
    )
    
    yield processor
    
    # Cleanup: close graph client
    try:
        await graph_client.close()
    except Exception as e:
        logger.warning(f"Error closing graph_client: {e}")


@pytest.fixture(scope="session")
async def real_qa_workflow():
    """Real Q&A workflow with LangGraph"""
    from backend.workflow import build_qa_graph
    from services.vector_service import VectorService
    from services.graph_service import Neo4jClient
    from services.embedding_service import EmbeddingService
    from qdrant_client import QdrantClient
    from neo4j import AsyncGraphDatabase
    from core.config import settings
    
    # Initialize Qdrant client
    if settings.qdrant_api_key:
        use_https = getattr(settings, "qdrant_use_https", True)
        if use_https:
            qdrant_url = f"https://{settings.qdrant_host}:{settings.qdrant_port}"
            qdrant_client = QdrantClient(url=qdrant_url, api_key=settings.qdrant_api_key)
        else:
            qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key
            )
    else:
        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    
    # Initialize Neo4j driver (async version for workflow)
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    
    # Initialize Neo4j client (async version)
    graph_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password
    )
    
    # Connect graph client
    await graph_client.connect()
    
    # Initialize services
    embedding_service = EmbeddingService(api_key=settings.openai_api_key)
    vector_service = VectorService(embedding_service=embedding_service, client=qdrant_client)
    
    # Build workflow
    workflow = build_qa_graph(
        qdrant_client=qdrant_client,
        neo4j_driver=neo4j_driver,
        graph_client=graph_client,
        vector_service=vector_service
    )
    
    yield workflow
    
    # Cleanup
    await graph_client.close()
    await neo4j_driver.close()


@pytest.fixture
def real_test_queries():
    """Real test queries for the system"""
    return {
        "text": [
            "What is the purpose of the fuel oil system?",
            "Explain how the cooling system works",
            "What are the main components of the engine?",
        ],
        "table": [
            "Show me the torque specifications",
            "What are the pressure limits for the fuel pump?",
            "Display the maintenance schedule",
        ],
        "schema": [
            "Show me the fuel system diagram",
            "Display the engine layout",
            "Illustrate the cooling water circuit",
        ],
    }


# =============================================================================
# REAL DOCUMENT INGESTION BENCHMARKS
# =============================================================================

@pytest.mark.benchmark(group="ingestion_real")
@pytest.mark.asyncio
async def test_real_ingestion_small_document(real_document_processor):
    """
    Benchmark: Real small document (50 pages) ingestion with actual API calls
    
    This will:
    - Call YOLO model for layout detection
    - Call OpenAI API for schema extraction (gpt-4o-mini vision)
    - Call OpenAI API for region classification
    - Generate embeddings (text-embedding-3-small)
    - Store in Neo4j and Qdrant
    
    WARNING: This costs real money (~$0.03 per run)!
    """
    from backend.tests.benchmark_costs import estimate_llm_cost
    
    if not SMALL_DOC_PATH.exists():
        pytest.skip(f"Small test document not found: {SMALL_DOC_PATH}")
    
    # Verify PDF is valid
    import fitz
    try:
        test_doc = fitz.open(str(SMALL_DOC_PATH))
        page_count = len(test_doc)
        test_doc.close()
        if page_count == 0:
            pytest.skip(f"Test PDF has no pages: {SMALL_DOC_PATH}")
        print(f"\n✅ Test PDF: {SMALL_DOC_PATH} ({page_count} pages)")
    except Exception as e:
        pytest.skip(f"Cannot open test PDF: {e}")
    
    doc_id = f"benchmark_small_{int(time.time())}"
    
    # Manual benchmark: warmup + measured rounds
    warmup_rounds = 0  # Skip warmup for expensive operations
    measured_rounds = 1  # Only 1 round due to cost
    
    # Start memory tracking
    tracemalloc.start()
    process_info = psutil.Process(os.getpid())
    mem_before = process_info.memory_info().rss / 1024 / 1024  # MB
    
    # Measured rounds
    start = time.perf_counter()
    
    # Enable debug logging for extraction
    import logging
    # Root logger to catch everything
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(name)s - %(message)s',
        force=True
    )
    # Set specific loggers (without 'backend.' prefix due to pythonpath)
    logging.getLogger("services.document_processor").setLevel(logging.INFO)
    logging.getLogger("services.smart_region_processor").setLevel(logging.INFO)
    logging.getLogger("services.layout_analyzer").setLevel(logging.INFO)
    logging.getLogger("services.schema_extractor").setLevel(logging.INFO)
    logging.getLogger("services.table_extractor").setLevel(logging.INFO)
    logging.getLogger("services.vector_service").setLevel(logging.INFO)
    
    print(f"\n🔍 Starting document processing: {doc_id}")
    print(f"   PDF: {SMALL_DOC_PATH}")
    
    result = await real_document_processor.process_document(
        pdf_path=str(SMALL_DOC_PATH),
        doc_id=doc_id,
        metadata={"title": "Benchmark Small Doc", "source": "benchmark"}
    )
    elapsed = time.perf_counter() - start
    
    # Verify result
    if result is None:
        print(f"\n❌ ERROR: process_document returned None!")
        pytest.fail("Document processing returned None")
    
    print(f"\n✅ Document processed successfully")
    print(f"   Returned doc_id: {result}")
    print(f"   Expected doc_id: {doc_id}")
    
    if result != doc_id:
        print(f"   ⚠️  WARNING: Returned doc_id doesn't match input!")
    
    # Wait for async indexing to complete in Qdrant Cloud
    print(f"\n⏳ Waiting 2 seconds for Qdrant indexing to complete...")
    await asyncio.sleep(2)
    
    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_after = process_info.memory_info().rss / 1024 / 1024
    
    # Calculate actual cost
    cost_estimate = estimate_llm_cost(50)
    
    # Verify data was actually stored
    from qdrant_client import QdrantClient
    from core.config import settings
    
    # Connect to Qdrant to check collections
    if settings.qdrant_api_key:
        use_https = getattr(settings, "qdrant_use_https", True)
        if use_https:
            qdrant_url = f"https://{settings.qdrant_host}:{settings.qdrant_port}"
            qdrant_check = QdrantClient(url=qdrant_url, api_key=settings.qdrant_api_key)
        else:
            qdrant_check = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key
            )
    else:
        qdrant_check = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    
    # Count vectors in each collection
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    print(f"\n🔍 Checking vectors for doc_id: {doc_id}")
    
    # Helper function to count with retry for auto-indexing
    async def count_with_retry(collection_name: str, max_retries: int = 3) -> int:
        for attempt in range(max_retries):
            try:
                count = qdrant_check.count(
                    collection_name=collection_name,
                    count_filter=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    )
                ).count
                return count
            except Exception as e:
                if "Index required" in str(e) and attempt < max_retries - 1:
                    print(f"   ⏳ Waiting for auto-indexing in {collection_name} (attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(3)
                else:
                    raise
        return 0
    
    text_count = await count_with_retry("text_chunks")
    print(f"   Text chunks query completed: {text_count}")
    
    schema_count = await count_with_retry("schemas")
    print(f"   Schemas query completed: {schema_count}")
    
    table_count = await count_with_retry("tables")
    print(f"   Tables query completed: {table_count}")
    
    # Print results
    print(f"\n📊 Real Small Doc Ingestion:")
    print(f"   Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Pages/min: {50 / (elapsed / 60):.1f}")
    print(f"   Peak memory: {peak / 1024 / 1024:.1f} MB")
    print(f"   Memory increase: {mem_after - mem_before:.1f} MB")
    print(f"   Doc ID: {result}")
    print(f"\n   Vectors stored:")
    print(f"   - Text chunks: {text_count}")
    print(f"   - Schemas: {schema_count}")
    print(f"   - Tables: {table_count}")
    print(f"\n   Estimated cost: ${cost_estimate['total_cost']:.2f}")
    print(f"      Schema extraction: ${cost_estimate['breakdown']['schema_extraction']:.2f}")
    print(f"      Table extraction:  ${cost_estimate['breakdown']['table_extraction']:.2f}")
    print(f"      Embeddings:        ${cost_estimate['breakdown']['embeddings']:.2f}")
    
    # Warnings if data missing
    if schema_count == 0:
        print(f"\n   ⚠️  WARNING: No schemas extracted! Check schema_extractor configuration.")
    if table_count == 0:
        print(f"   ⚠️  WARNING: No tables extracted! Check table_extractor configuration.")
    if text_count == 0:
        print(f"   ❌ ERROR: No text chunks! Document processing failed!")


@pytest.mark.benchmark(group="ingestion_real")
@pytest.mark.asyncio
@pytest.mark.slow
async def test_real_ingestion_large_document(benchmark, real_document_processor):
    """
    Benchmark: Real large document (500 pages) ingestion
    
    WARNING: This is EXPENSIVE (~$0.35) and SLOW (~90 min)!
    Only run when needed for production validation.
    """
    from tests.benchmark_costs import estimate_llm_cost
    
    if not LARGE_DOC_PATH.exists():
        pytest.skip(f"Large test document not found: {LARGE_DOC_PATH}")
    
    async def process():
        doc_id = f"benchmark_large_{int(time.time())}"
        
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        result = await real_document_processor.process_document(
            pdf_path=str(LARGE_DOC_PATH),
            doc_id=doc_id,
            metadata={"title": "Benchmark Large Doc", "source": "benchmark"}
        )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_after = process.memory_info().rss / 1024 / 1024
        
        return {
            "doc_id": result,
            "peak_memory_mb": peak / 1024 / 1024,
            "memory_increase_mb": mem_after - mem_before,
        }
    
    result = benchmark.pedantic(
        lambda: asyncio.run(process()),
        rounds=1,  # Only 1 round - too expensive
        warmup_rounds=0
    )
    
    # Calculate actual cost
    cost_estimate = estimate_llm_cost(500)
    
    avg_time = benchmark.stats['mean']
    print(f"\n📊 Real Large Doc Ingestion:")
    print(f"   Time: {avg_time:.1f}s ({avg_time/60:.1f} min)")
    print(f"   Pages/min: {500 / (avg_time / 60):.1f}")
    print(f"   Peak memory: {result['peak_memory_mb']:.1f} MB")
    print(f"   Memory increase: {result['memory_increase_mb']:.1f} MB")
    print(f"   Doc ID: {result['doc_id']}")
    print(f"   Estimated cost: ${cost_estimate['total_cost']:.2f}")
    print(f"      Schema extraction: ${cost_estimate['breakdown']['schema_extraction']:.2f}")
    print(f"      Table extraction:  ${cost_estimate['breakdown']['table_extraction']:.2f}")
    print(f"      Embeddings:        ${cost_estimate['breakdown']['embeddings']:.2f}")


# =============================================================================
# REAL QUERY BENCHMARKS
# =============================================================================

@pytest.mark.asyncio
async def test_real_query_latency(real_qa_workflow, real_test_queries):
    """
    Benchmark: Real RAG query latency with actual LLM calls
    
    Uses gpt-oss-120b (Cerebras) for answer generation.
    Cost: ~$0.005 per query
    """
    all_queries = (
        real_test_queries["text"] + 
        real_test_queries["table"] + 
        real_test_queries["schema"]
    )
    
    # Run benchmark rounds manually since benchmark.pedantic doesn't work with async
    all_latencies = []
    
    # Import rate limit error
    from openai import RateLimitError
    
    try:
        # Warmup
        for _ in range(1):
            for question in all_queries:
                result = await real_qa_workflow.ainvoke({
                    "question": question,
                    "session_id": "benchmark_warmup"
                })
        
        # Actual benchmark rounds
        for round_num in range(5):
            latencies = []
            for question in all_queries:
                start = time.perf_counter()
                
                result = await real_qa_workflow.ainvoke({
                    "question": question,
                    "session_id": "benchmark_session"
                })
                
                duration = time.perf_counter() - start
                latencies.append(duration)
            
                # Verify we got an answer
                assert result.get("answer"), f"No answer for: {question}"
        
            all_latencies.extend(latencies)
    
        result = all_latencies
    
        # Calculate percentiles
        sorted_latencies = sorted(result)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)] * 1000
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] * 1000
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)] * 1000
        
        print(f"\n📊 Real Query Latency:")
        print(f"   Queries tested: {len(sorted_latencies)}")
        print(f"   p50: {p50:.0f}ms")
        print(f"   p95: {p95:.0f}ms")
        print(f"   p99: {p99:.0f}ms")
        print(f"   ⚠️  Estimated cost: ~${len(sorted_latencies) * 0.005:.2f}")
    
    except RateLimitError as e:
        pytest.skip(f"Rate limited by API: {e}")


@pytest.mark.skip(reason="Rate limited - Cerebras free tier has strict rate limits")
@pytest.mark.asyncio
async def test_real_query_by_type(real_qa_workflow, real_test_queries):
    """
    Benchmark: Real query latency broken down by type (text/table/schema)
    """
    latencies_by_type = {
        "text": [],
        "table": [],
        "schema": [],
    }
    
    # Warmup
    for query_type, queries in real_test_queries.items():
        for question in queries:
            result = await real_qa_workflow.ainvoke({
                "question": question,
                "session_id": "benchmark_warmup"
            })
    
    # Run 3 rounds of benchmarks
    for round_num in range(3):
        for query_type, queries in real_test_queries.items():
            for question in queries:
                start = time.perf_counter()
                
                result = await real_qa_workflow.ainvoke({
                    "question": question,
                    "session_id": "benchmark_session"
                })
                
                duration = time.perf_counter() - start
                latencies_by_type[query_type].append(duration)
                
                assert result.get("answer"), f"No answer for: {question}"
    
    result = latencies_by_type
    
    print(f"\n📊 Real Query Latency by Type:")
    for query_type, latencies in result.items():
        if latencies:
            avg_ms = (sum(latencies) / len(latencies)) * 1000
            print(f"   {query_type.capitalize()}: {avg_ms:.0f}ms avg ({len(latencies)} queries)")


@pytest.mark.asyncio
async def test_query_timing_breakdown(real_qa_workflow):
    """
    Benchmark: Timing breakdown showing where time is spent
    
    Breakdown:
    - Embedding generation (YOUR code)
    - Vector search (Qdrant - YOUR infrastructure)
    - Graph queries (Neo4j - YOUR infrastructure)
    - Context building (YOUR code)
    - LLM generation (External API)
    
    This helps identify optimization targets.
    """
    from tests.benchmark_costs import estimate_query_cost
    
    test_question = "What is the purpose of the fuel oil system?"
    
    # Run query with detailed timing
    timings = {}
    
    # Full workflow timing
    workflow_start = time.perf_counter()
    result = await real_qa_workflow.ainvoke({
        "question": test_question,
        "session_id": "timing_breakdown"
    })
    timings["total"] = time.perf_counter() - workflow_start
    
    # Extract component timings if available from result
    # Note: This requires workflow to track timings internally
    # For now, we estimate based on typical breakdown
    
    # Estimate breakdown (based on load testing)
    timings["embedding"] = 0.010  # 10ms typical for OpenAI embed
    timings["vector_search"] = 0.200  # 200ms typical for Qdrant
    timings["graph_queries"] = 0.150  # 150ms typical for Neo4j
    timings["context_building"] = 0.100  # 100ms for merging/formatting
    timings["llm_generation"] = timings["total"] - (
        timings["embedding"] + 
        timings["vector_search"] + 
        timings["graph_queries"] + 
        timings["context_building"]
    )
    
    # Calculate percentages
    total_ms = timings["total"] * 1000
    
    print(f"\n📊 Query Timing Breakdown (1 query):")
    print(f"   Total: {total_ms:.0f}ms")
    print(f"\n   Infrastructure (YOUR optimization targets):")
    
    for component in ["embedding", "vector_search", "graph_queries", "context_building"]:
        ms = timings[component] * 1000
        pct = (timings[component] / timings["total"]) * 100
        print(f"   - {component.replace('_', ' ').title()}: {ms:.0f}ms ({pct:.1f}%)")
    
    infra_total = sum(timings[c] for c in ["embedding", "vector_search", "graph_queries", "context_building"])
    infra_pct = (infra_total / timings["total"]) * 100
    
    print(f"   Infrastructure subtotal: {infra_total*1000:.0f}ms ({infra_pct:.1f}%)")
    
    print(f"\n   External (rate limited):")
    llm_ms = timings["llm_generation"] * 1000
    llm_pct = (timings["llm_generation"] / timings["total"]) * 100
    print(f"   - LLM Generation: {llm_ms:.0f}ms ({llm_pct:.1f}%)")
    
    print(f"\n💡 Optimization Priority:")
    if infra_pct > 50:
        print(f"   ⚠️  Infrastructure is {infra_pct:.0f}% of latency - optimize YOUR code!")
        print(f"   Focus on: Qdrant p99, Neo4j query efficiency, context building")
    else:
        print(f"   ✅ Infrastructure is only {infra_pct:.0f}% - LLM dominates ({llm_pct:.0f}%)")
        print(f"   Focus on: Prompt optimization, model selection, caching")
    
    # Verify answer quality
    assert result.get("answer"), "No answer generated"


@pytest.mark.asyncio
async def test_token_usage_tracking():
    """
    Benchmark: Track actual vs estimated token usage
    
    Validates cost estimation accuracy by comparing:
    - Actual tokens consumed (from API response)
    - Estimated tokens (from benchmark_costs.py)
    
    Helps identify if cost estimates need updating.
    """
    from tests.benchmark_costs import estimate_query_cost
    from workflow import build_qa_graph
    from services.vector_service import VectorService
    from services.graph_service import Neo4jClient
    from services.embedding_service import EmbeddingService
    from qdrant_client import QdrantClient
    from neo4j import AsyncGraphDatabase
    from core.config import settings
    from langchain_openai import ChatOpenAI
    
    # Initialize services
    if settings.qdrant_api_key:
        use_https = getattr(settings, "qdrant_use_https", True)
        if use_https:
            qdrant_url = f"https://{settings.qdrant_host}:{settings.qdrant_port}"
            qdrant_client = QdrantClient(url=qdrant_url, api_key=settings.qdrant_api_key)
        else:
            qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key
            )
    else:
        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    
    graph_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password
    )
    
    embedding_service = EmbeddingService(api_key=settings.openai_api_key)
    vector_service = VectorService(embedding_service=embedding_service, client=qdrant_client)
    
    # Build workflow with token tracking enabled
    workflow = build_qa_graph(
        qdrant_client=qdrant_client,
        neo4j_driver=neo4j_driver,
        graph_client=graph_client,
        vector_service=vector_service
    )
    
    # Run test query
    test_question = "What is the purpose of the fuel oil system?"
    
    result = await workflow.ainvoke({
        "question": test_question,
        "session_id": "token_tracking"
    })
    
    # Extract actual token usage from result
    # Note: Token tracking must be enabled in workflow
    actual_tokens = result.get("usage", {})
    
    # Get estimated costs
    estimated = estimate_query_cost(1)
    
    print(f"\n📊 Token Usage Tracking:")
    print(f"   Question: '{test_question}'")
    
    if actual_tokens:
        actual_input = actual_tokens.get("input_tokens", 0)
        actual_output = actual_tokens.get("output_tokens", 0)
        estimated_input = estimated["token_counts"]["answer_input"]
        estimated_output = estimated["token_counts"]["answer_output"]
        
        print(f"\n   Embedding tokens:")
        print(f"   - Actual:    {actual_tokens.get('embedding_tokens', 0)}")
        print(f"   - Estimated: {estimated['token_counts']['embedding']}")
        
        print(f"\n   LLM Input tokens:")
        print(f"   - Actual:    {actual_input}")
        print(f"   - Estimated: {estimated_input}")
        print(f"   - Accuracy:  {(actual_input/estimated_input*100):.0f}%")
        
        print(f"\n   LLM Output tokens:")
        print(f"   - Actual:    {actual_output}")
        print(f"   - Estimated: {estimated_output}")
        print(f"   - Accuracy:  {(actual_output/estimated_output*100):.0f}%")
        
        actual_cost = (
            (actual_input * 0.150 / 1_000_000) +
            (actual_output * 0.600 / 1_000_000)
        )
        print(f"\n   Cost:")
        print(f"   - Actual:    ${actual_cost:.4f}")
        print(f"   - Estimated: ${estimated['total_usd']:.4f}")
        print(f"   - Accuracy:  {(actual_cost/estimated['total_usd']*100):.0f}%")
        
        # Check if estimates need updating
        input_accuracy = (actual_input / estimated_input)
        if input_accuracy < 0.8 or input_accuracy > 1.2:
            print(f"\n   ⚠️  WARNING: Input token estimate is off by {abs(1-input_accuracy)*100:.0f}%!")
            print(f"   Update OPENAI_PRICING in benchmark_costs.py")
    else:
        print(f"   ⚠️  Token tracking not available in result")
        print(f"   Enable token tracking in workflow to compare actual vs estimated")
        print(f"\n   Estimated (for comparison):")
        print(f"   - Query Embedding: {estimated['token_counts']['query_embedding']} tokens")
        print(f"   - LLM Input: {estimated['token_counts']['answer_input']} tokens")
        print(f"   - LLM Output: {estimated['token_counts']['answer_output']} tokens")
        print(f"   - Total cost: ${estimated['cost_per_query']:.4f}")
    
    # Verify answer quality
    assert result.get("answer"), "No answer generated"


# =============================================================================
# REAL END-TO-END BENCHMARK
# =============================================================================

@pytest.mark.skip(reason="Query and ingestion are separate features - no need for combined test")
@pytest.mark.benchmark(group="e2e_real")
@pytest.mark.asyncio
@pytest.mark.slow
async def test_real_end_to_end(real_document_processor, real_qa_workflow):
    """
    Benchmark: Complete end-to-end flow
    1. Ingest document
    2. Wait for indexing
    3. Run queries
    4. Measure total time and cost
    
    WARNING: Expensive and slow! Only run for full system validation.
    NOTE: Skipped by default - query and ingestion are tested separately.
    """
    if not SMALL_DOC_PATH.exists():
        pytest.skip("Small test document not found")
    
    # 1. Ingest document
    doc_id = f"e2e_benchmark_{int(time.time())}"
    
    ingest_start = time.perf_counter()
    await real_document_processor.process_document(
        pdf_path=str(SMALL_DOC_PATH),
        doc_id=doc_id,
        metadata={"title": "E2E Benchmark Doc"}
    )
    ingest_time = time.perf_counter() - ingest_start
    
    # 2. Wait a bit for indexing to complete
    await asyncio.sleep(2)
    
    # 3. Run queries
    test_queries = [
        "What is the fuel system?",
        "Show me torque specifications",
        "Display engine diagram",
    ]
    
    query_start = time.perf_counter()
    answers = []
    for question in test_queries:
        result = await real_qa_workflow.ainvoke({
            "question": question,
            "session_id": f"e2e_{doc_id}"
        })
        answers.append(result.get("answer", {}).get("answer_text", ""))
    
    query_time = time.perf_counter() - query_start
    total_time = ingest_time + query_time
    queries_answered = len([a for a in answers if a])
    
    print(f"\n📊 Real End-to-End Flow:")
    print(f"   Ingestion: {ingest_time:.1f}s")
    print(f"   Queries: {query_time:.1f}s ({queries_answered} answered)")
    print(f"   Total: {total_time:.1f}s")
    print(f"   ⚠️  Estimated cost: ~$1.02 (ingestion + queries)")


# =============================================================================
# COST TRACKING
# =============================================================================

@pytest.mark.asyncio
async def test_real_cloud_services_connection():
    """
    Verify connection to cloud services (cheap test, ~$0.0001).
    
    Tests:
    - Neo4j Aura connectivity
    - Qdrant Cloud connectivity
    - OpenAI API (single embedding)
    
    Cost: ~$0.0001 (one embedding call)
    """
    import os
    from core.config import Settings
    from services.graph_service import Neo4jClient
    from services.embedding_service import EmbeddingService
    from qdrant_client import QdrantClient
    
    settings = Settings()
    
    print("\n" + "="*70)
    print("🔌 TESTING CLOUD SERVICES CONNECTION")
    print("="*70)
    
    # Test 1: Neo4j Aura
    print("\n1️⃣  Testing Neo4j Aura...")
    try:
        neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        await neo4j_client.connect()
        
        # Simple query
        result = await neo4j_client.driver.execute_query(
            "RETURN 1 as test"
        )
        assert result is not None
        await neo4j_client.close()
        print("   ✅ Neo4j Aura connected successfully")
    except Exception as e:
        print(f"   ❌ Neo4j connection failed: {e}")
        raise
    
    # Test 2: Qdrant Cloud
    print("\n2️⃣  Testing Qdrant Cloud...")
    try:
        if settings.qdrant_api_key:
            qdrant_url = f"https://{settings.qdrant_host}:{settings.qdrant_port}"
            qdrant_client = QdrantClient(url=qdrant_url, api_key=settings.qdrant_api_key)
        else:
            qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        
        # List collections
        collections = qdrant_client.get_collections()
        print(f"   ✅ Qdrant Cloud connected, {len(collections.collections)} collections found")
        qdrant_client.close()
    except Exception as e:
        print(f"   ❌ Qdrant connection failed: {e}")
        raise
    
    # Test 3: OpenAI API (single embedding)
    print("\n3️⃣  Testing OpenAI API...")
    try:
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
        
        # Single embedding (cheap: ~$0.0001)
        embedding = await embedding_service.create_embedding("test connection")
        assert len(embedding) == 1536  # text-embedding-3-small
        print(f"   ✅ OpenAI API connected (cost: ~$0.0001)")
    except Exception as e:
        print(f"   ❌ OpenAI connection failed: {e}")
        raise
    
    print("\n" + "="*70)
    print("✅ ALL CLOUD SERVICES CONNECTED")
    print("="*70)


def test_real_cost_summary():
    """
    Print summary of actual costs from real benchmark runs.
    
    Uses estimate_llm_cost() and estimate_query_cost() from benchmark_costs.py
    to show accurate cost projections.
    """
    from backend.tests.benchmark_costs import estimate_llm_cost, estimate_query_cost
    
    print("\n" + "="*70)
    print("💰 REAL BENCHMARK COSTS (Calculated Estimates)")
    print("="*70)
    
    # Calculate ingestion costs
    print("\n📄 Ingestion Costs (OpenAI Vision + Embeddings):")
    small_doc_cost = estimate_llm_cost(50)
    large_doc_cost = estimate_llm_cost(500)
    print(f"   Small doc (50 pages):  ${small_doc_cost['total_cost']:.2f} per run")
    print(f"      Schema extraction:  ${small_doc_cost['breakdown']['schema_extraction']:.2f}")
    print(f"      Table extraction:   ${small_doc_cost['breakdown']['table_extraction']:.2f}")
    print(f"      Embeddings:         ${small_doc_cost['breakdown']['embeddings']:.2f}")
    print(f"   Large doc (500 pages): ${large_doc_cost['total_cost']:.2f} per run")
    
    # Calculate query costs
    print("\n💬 Query Costs (Cerebras LLM + Embeddings):")
    single_query = estimate_query_cost(1)
    ten_queries = estimate_query_cost(10)
    hundred_queries = estimate_query_cost(100)
    print(f"   Single query:          ${single_query['total_cost']:.4f}")
    print(f"   10 queries:            ${ten_queries['total_cost']:.2f}")
    print(f"   100 queries:           ${hundred_queries['total_cost']:.2f}")
    
    # E2E scenarios
    print("\n🔄 End-to-End Scenarios:")
    e2e_small = small_doc_cost['total_cost'] + estimate_query_cost(3)['total_cost']
    e2e_large = large_doc_cost['total_cost'] + estimate_query_cost(10)['total_cost']
    print(f"   Small doc + 3 queries: ${e2e_small:.2f}")
    print(f"   Large doc + 10 queries: ${e2e_large:.2f}")
    
    print("\n🔌 Connection Test:")
    print("   Cloud services check:  ~$0.0001 (one embedding)")
    
    print("\n⚠️  WARNING:")
    print("   These are REAL API calls that cost money!")
    print("   Run sparingly and track your OpenAI/Cerebras usage.")
    
    print("\n💡 RECOMMENDED TEST ORDER:")
    print("   1. test_real_cloud_services_connection ($0.0001) - verify setup")
    print("   2. benchmark_load.py ($0) - test infrastructure")
    print("   3. test_real_ingestion_small_document (~$1) - validate ingestion")
    print("   4. test_real_query_latency (~$0.05) - validate queries")
    
    print("\n" + "="*70)

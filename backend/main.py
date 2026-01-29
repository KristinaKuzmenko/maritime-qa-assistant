"""
FastAPI main application for maritime technical documentation system.
Provides endpoints for document upload, processing, search, and Q&A.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from middleware.rate_limiter import RateLimitExceeded
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import logging
import asyncio
from pathlib import Path
import uuid
import shutil
import os

from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase

from services.document_processor import DocumentProcessor
from services.schema_extractor import SchemaExtractor
from services.table_extractor import TableExtractor
from services.layout_analyzer import LayoutAnalyzer
from services.storage_service import StorageService
from services.graph_service import Neo4jClient
from services.vector_service import VectorService
from services.embedding_service import EmbeddingService
from workflow import build_qa_graph, preload_entities

from routes import health, documents, chat
from core.config import settings
from core.dependencies import set_services
from core.exceptions import MaritimeQAException


# Configure logging with file output
log_dir = Path("logs")
try:
    log_dir.mkdir(exist_ok=True)
except (OSError, PermissionError):
    # Fallback to /tmp if logs/ is not writable (e.g., in read-only containers)
    log_dir = Path("/tmp/logs")
    log_dir.mkdir(exist_ok=True)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
simple_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Only add handlers if not already present (avoid duplicates on reload)
if not root_logger.handlers:
    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # File handler for all logs (detailed format)
    file_handler = logging.FileHandler(log_dir / "maritime_api.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # File handler for document processing only (very detailed)
    processing_handler = logging.FileHandler(log_dir / "document_processing.log", encoding='utf-8')
    processing_handler.setLevel(logging.DEBUG)
    processing_handler.setFormatter(detailed_formatter)
    processing_handler.addFilter(lambda record: any(name in record.name for name in [
        'document_processor', 'schema_extractor', 'table_extractor', 
        'layout_analyzer', 'region_classifier', 'smart_region_processor',
        'embedding_service', 'graph_service', 'vector_service'
    ]))

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(processing_handler)
else:
    # Suppress duplicate initialization logs from reloader
    os.environ['LOGGING_CONFIGURED'] = '1'

logger = logging.getLogger(__name__)

# Global service instances
graph_client: Optional[Neo4jClient] = None
vector_service: Optional[VectorService] = None
embedding_service: Optional[EmbeddingService] = None
storage_service: Optional[StorageService] = None
qa_graph = None
schema_extractor = None
table_extractor = None
document_processor = None
layout_analyzer = None

qdrant_client: Optional[QdrantClient] = None
neo4j_driver = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global graph_client, vector_service, embedding_service, storage_service
    global qa_graph, schema_extractor, table_extractor, document_processor, layout_analyzer
    global qdrant_client, neo4j_driver
    
    # ========== STARTUP ==========
    logger.info("=" * 80)
    logger.info("🚀 Starting Maritime Documentation API...")
    logger.info("=" * 80)
    
    try:
        # 1. Storage Service
        logger.info("Initializing storage service...")
        storage_service = StorageService(
            storage_type=settings.storage_type,
            local_storage_path=settings.local_storage_path,
            s3_bucket_name=settings.s3_bucket_name,
            s3_prefix=settings.s3_prefix,  # e.g., 'data' if files are in data/schemas/...
            aws_region=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            base_url="/data" if settings.storage_type == "local" else None
        )
        logger.info(f"✅ Storage service initialized ({settings.storage_type})")
        
        # 2. Embedding Service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
        logger.info("✅ Embedding service initialized")
        
        # 2.5. OpenAI Client for LLM operations (vision, etc.)
        from openai import AsyncOpenAI
        llm_client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("✅ OpenAI LLM client initialized")
        
        # 3. Neo4j (with graceful fallback)
        logger.info("Connecting to Neo4j...")
        try:
            graph_client = Neo4jClient(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
                database=settings.neo4j_database,
            )
            await graph_client.connect()

            # Configure Neo4j driver with automatic recovery
            # liveness_check_timeout enables detection of dead connections
            neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=300, 
                max_connection_pool_size=10,
                connection_acquisition_timeout=90.0,  # 90 seconds
                connection_timeout=30.0,  # 30 seconds
                keep_alive=True,
                liveness_check_timeout=5.0,  # CHECK connection before use 
            )

            logger.info("✅ Neo4j connected")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            logger.warning("⚠️  API will start without Neo4j")
            logger.info("💡 Start Neo4j: docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
            graph_client = None
            neo4j_driver = None
        
        # 4. Qdrant 
        logger.info("Connecting to Qdrant...")
        try:
            vector_service = VectorService(embedding_service=embedding_service)
            vector_service.initialize_collections()
            logger.info("✅ Qdrant initialized")

            # Initialize Qdrant client (reuse logic from VectorService)
            if settings.qdrant_api_key:
                use_https = getattr(settings, "qdrant_use_https", True)
                if use_https:
                    qdrant_url = f"https://{settings.qdrant_host}:{settings.qdrant_port}"
                    qdrant_client = QdrantClient(
                        url=qdrant_url,
                        api_key=settings.qdrant_api_key,
                    )
                else:
                    qdrant_client = QdrantClient(
                        host=settings.qdrant_host,
                        port=settings.qdrant_port,
                        api_key=settings.qdrant_api_key,
                    )
            else:
                qdrant_client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                )
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")
            logger.warning("⚠️  API will start without Qdrant")
            logger.info("💡 Start Qdrant: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest")
            vector_service = None
            qdrant_client = None
        
        # 5. Q&A Workflow (only if both Neo4j and Qdrant available)
        if graph_client and vector_service:
            if neo4j_driver and qdrant_client:
                logger.info("Initializing Q&A workflow...")
                try:
                    # ⚡ OPTIMIZATION: Preload entities at startup (not on first query)
                    from workflow import preload_entities, tool_ctx
                    
                    # Set driver and graph_client in tool_ctx BEFORE calling preload_entities
                    tool_ctx.neo4j_driver = neo4j_driver
                    tool_ctx.graph_client = graph_client  # Neo4jClient for high-level operations
                    
                    known_entities = await preload_entities(neo4j_driver)
                    tool_ctx.known_entities = known_entities
                    tool_ctx.entities_loaded = True
                    logger.info(f"✅ Preloaded {len(known_entities)} entities into tool_ctx")
                    
                    qa_graph = build_qa_graph(
                        qdrant_client=qdrant_client,
                        neo4j_driver=neo4j_driver,
                        vector_service=vector_service,
                        neo4j_uri=settings.neo4j_uri,
                        neo4j_auth=(settings.neo4j_user, settings.neo4j_password),
                        graph_client=graph_client,
                    )
                    logger.info("✅ Q&A workflow initialized")
                    
                except Exception as e:
                    logger.error(f"❌ Q&A workflow failed: {e}", exc_info=True)
                    qa_graph = None
            else:
                logger.warning("⚠️  Q&A workflow disabled (Neo4j or Qdrant unavailable)")
        
        # 6. Document Processing Components
        if graph_client and vector_service:
            logger.info("Initializing document processors...")
            try:
                # Use path relative to this file (works locally and in Docker)
                model_path = Path(__file__).parent / "models" / "yolov12s-doclaynet.pt"
                layout_analyzer = LayoutAnalyzer(
                    model_path=str(model_path),
                    confidence_threshold=0.4,
                )

                schema_extractor = SchemaExtractor(
                    storage_service=storage_service,
                    layout_analyzer=layout_analyzer,
                    llm_service=llm_client,
                    enable_llm_summary=True,
                    vision_detail=settings.vision_detail_schemas,  # Cost optimization
                )
                
                table_extractor = TableExtractor(
                    storage_service=storage_service,
                    max_tokens_per_chunk=4000,
                )
                
                document_processor = DocumentProcessor(
                    graph_client=graph_client,
                    layout_analyzer=layout_analyzer,
                    schema_extractor=schema_extractor,
                    embedding_service=embedding_service,
                    storage_service=storage_service,
                    vector_service=vector_service,
                    table_extractor=table_extractor,
                )
                logger.info("✅ Document processors initialized")
            except Exception as e:
                logger.error(f"❌ Document processors failed: {e}")
        
        
        # ✅ Set services for dependency injection
        set_services(
            graph_client=graph_client,
            vector_service=vector_service,
            embedding_service=embedding_service,
            storage_service=storage_service,
            document_processor=document_processor,
            qa_graph=qa_graph
        )
        
        logger.info("=" * 80)
        logger.info("✅ API is ready!")
        logger.info("📖 Docs: http://localhost:8000/docs")
        logger.info("🏥 Health: http://localhost:8000/health")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Critical startup error: {e}", exc_info=True)
    
    yield  # API runs here
    
    # ========== SHUTDOWN ==========
    logger.info("=" * 80)
    logger.info("👋 Shutting down...")
    logger.info("=" * 80)
    
    try:
        if graph_client:
            try:
                await graph_client.close()
                logger.info("✅ Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j: {e}")

        # Close async driver
        if neo4j_driver:
            try:
                await neo4j_driver.close()
                logger.info("✅ Neo4j async driver closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")
    except asyncio.CancelledError:
        # Normal shutdown behavior - ignore
        logger.info("✅ Shutdown complete")
        pass


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Maritime Technical Documentation API",
    description="API for processing and querying maritime technical manuals",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Exception handler for rate limit
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers
    )

# Exception handler for Maritime QA custom exceptions
@app.exception_handler(MaritimeQAException)
async def maritime_exception_handler(request: Request, exc: MaritimeQAException):
    """Handle all Maritime QA custom exceptions."""
    logger.error(
        f"{exc.__class__.__name__}: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )

@app.middleware("http")
async def attach_user_to_request(request: Request, call_next):
    user_id = request.headers.get("X-User-Id")
    user_role = request.headers.get("X-User-Role", "guest")

    request.state.user_id = user_id or request.client.host
    request.state.user_role = user_role or "guest"

    response = await call_next(request)
    return response


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/qa", tags=["Q&A"])


# Static Files - Only mount for local storage (S3 uses direct links)
if settings.storage_type == "local" and not os.environ.get('LOGGING_CONFIGURED'):
    BASE_DIR = Path(__file__).parent.parent  
    DATA_DIR = BASE_DIR / "data"
    
    # Ensure directories exist
    (DATA_DIR / "schemas").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "tables").mkdir(parents=True, exist_ok=True)
    
    # Mount static file directories with absolute paths
    app.mount(
        "/schemas", 
        StaticFiles(directory=str(DATA_DIR / "schemas")), 
        name="schemas"
    )
    
    app.mount(
        "/tables", 
        StaticFiles(directory=str(DATA_DIR / "tables")), 
        name="tables"
    )
    
    logger.info(f"📁 Static files mounted (local storage):")
    logger.info(f"   /schemas -> {DATA_DIR / 'schemas'}")
    logger.info(f"   /tables -> {DATA_DIR / 'tables'}")
elif settings.storage_type == "s3" and not os.environ.get('LOGGING_CONFIGURED'):
    logger.info(f"📁 Storage configured for S3 (bucket: {settings.s3_bucket_name})")
    logger.info(f"   Direct S3 links will be used for schemas and tables")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Maritime Documentation Q&A API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "documents": "/documents",
            "upload": "/documents/upload",
            "qa": "/qa/answer",
            "schemas": "/schemas/original/{doc_id}/{filename}",
            "tables": "/tables/original/{doc_id}/{filename}",
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
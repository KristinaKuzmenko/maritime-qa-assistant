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
from workflow import build_qa_graph

from routes import health, documents, chat
from core.config import settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
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
            storage_type="local",
            local_storage_path="./data",
            base_url="/data"
        )
        logger.info("✅ Storage service initialized")
        
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

            neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
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
                    qa_graph = build_qa_graph(
                        qdrant_client=qdrant_client,
                        neo4j_driver=neo4j_driver,
                        vector_service=vector_service,
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
                layout_analyzer = LayoutAnalyzer(
                    model_path="backend/models/yolov10s_best.pt",  # Path from project root
                    confidence_threshold=0.4,
                )

                schema_extractor = SchemaExtractor(
                    storage_service=storage_service,
                    layout_analyzer=layout_analyzer,
                    llm_service=llm_client,
                    enable_llm_summary=True,
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
        
        
        # Make services globally available to routes
        import routes.documents as doc_routes
        import routes.chat as chat_routes
        import routes.health as health_routes

        doc_routes.graph_client = graph_client
        doc_routes.vector_service = vector_service
        doc_routes.embedding_service = embedding_service
        doc_routes.storage_service = storage_service
        doc_routes.document_processor = document_processor
        doc_routes.schema_extractor = schema_extractor
        doc_routes.table_extractor = table_extractor
        doc_routes.layout_analyzer = layout_analyzer

        chat_routes.qa_graph = qa_graph
        chat_routes.graph_client = graph_client
        chat_routes.qdrant_client = qdrant_client
        chat_routes.neo4j_driver = neo4j_driver


        health_routes.graph_client = graph_client
        health_routes.vector_service = vector_service
        health_routes.storage_service = storage_service
        health_routes.qa_graph = qa_graph
        
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


# Static Files - Serve schemas, tables, and other media
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

logger.info(f"📁 Static files mounted:")
logger.info(f"   /schemas -> {DATA_DIR / 'schemas'}")
logger.info(f"   /tables -> {DATA_DIR / 'tables'}")


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
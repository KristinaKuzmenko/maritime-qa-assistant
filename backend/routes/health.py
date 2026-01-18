"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends
from datetime import datetime
import logging

from core.dependencies import HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check(status: HealthStatus = Depends()):
    """
    Comprehensive health check endpoint.
    Returns API status and all service availability.
    """
    
    
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Maritime Documentation API",
        "version": "1.0.0",
        "services": {}
    }
    
    # Check Neo4j
    try:
        if status.graph:
            stats = await status.graph.get_document_stats(doc_id=None)
            health["services"]["neo4j"] = {
                "status": "healthy",
                "stats": stats
            }
        else:
            health["services"]["neo4j"] = {
                "status": "unavailable",
                "message": "Neo4j not initialized"
            }
            health["status"] = "degraded"
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        health["services"]["neo4j"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "degraded"
    
    # Check Qdrant
    try:
        if status.vector:
            # get_collection_info() now returns a dict with collection labels as keys
            # Format: {"text_chunks": {...}, "figures": {...}, "tables": {...}, "summary": {...}}
            info = status.vector.get_collection_info()
            
            collections_info = {}
            
            # Extract collection data (exclude 'summary' key)
            for label, data in info.items():
                if label == "summary":
                    continue  # Skip summary, we'll add it separately
                
                if isinstance(data, dict):
                    collections_info[label] = {
                        "collection_name": data.get("collection_name", label),
                        "points_count": data.get("points_count", 0),
                        "status": data.get("status", "unknown"),
                    }
                    
                    # Add error if present
                    if "error" in data:
                        collections_info[label]["error"] = data["error"]
            
            # Build response
            qdrant_status = {
                "status": "healthy",
                "collections": collections_info,
            }
            
            # Add summary if available
            if "summary" in info:
                qdrant_status["summary"] = info["summary"]
            
            # Check if any collection has errors
            has_errors = any(
                coll.get("status") == "error" 
                for coll in collections_info.values()
            )
            
            if has_errors:
                qdrant_status["status"] = "degraded"
                health["status"] = "degraded"
            
            health["services"]["qdrant"] = qdrant_status
        else:
            health["services"]["qdrant"] = {
                "status": "unavailable",
                "message": "Qdrant not initialized",
            }
            health["status"] = "degraded"

    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        health["services"]["qdrant"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health["status"] = "degraded"
    
    # Check Storage
    try:
        if status.storage:
            storage_health = await status.storage.health_check()
            health["services"]["storage"] = {
                "status": "healthy",
                "info": storage_health
            }
        else:
            health["services"]["storage"] = {
                "status": "unavailable",
                "message": "Storage service not initialized"
            }
            health["status"] = "degraded"
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        health["services"]["storage"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "degraded"
    
    # Check Q&A Workflow
    if status.qa:
        health["services"]["qa_workflow"] = {
            "status": "available",
            "message": "LangGraph workflow ready"
        }
    else:
        health["services"]["qa_workflow"] = {
            "status": "unavailable",
            "message": "Q&A workflow not initialized (requires Neo4j + Qdrant)"
        }
        health["status"] = "degraded"
    
    return health


@router.get("/")
async def root():
    """API information."""
    return {
        "name": "Maritime Technical Documentation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "documents": "/documents",
            "chat": "/qa/answer",
        }
    }
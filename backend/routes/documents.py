"""
Document management endpoints.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Header, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import uuid
from datetime import datetime
from pathlib import Path

from middleware.rate_limiter import role_rate_limit
from core.dependencies import DocumentServices, get_graph_client, get_storage_service
from core.exceptions import FileUploadError, DocumentNotFoundError, ProcessingError
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Processing status tracking
processing_status: Dict[str, Dict[str, Any]] = {}


# Pydantic Models

class DocumentMetadata(BaseModel):
    """Document metadata model."""
    title: str
    doc_type: str = "manual"
    owner: str = "global"
    tags: List[str] = []

class DocumentResponse(BaseModel):
    """Document response model."""
    doc_id: str
    title: str
    status: str
    created_at: str
    total_pages: Optional[int] = None
    owner: Optional[str] = None


# Background Processing

async def process_document_background(
    task_id: str,
    doc_id: str,
    file_path: str,
    metadata: DocumentMetadata,
    document_processor,
    storage_service,
    is_s3_only: bool = False,
):
    """Background task for document processing."""
    # initial status
    processing_status[task_id] = {
        "status": "processing",
        "progress": 0,
        "doc_id": doc_id,
        "message": "Starting document processing...",
    }

    if not document_processor:
        processing_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "doc_id": doc_id,
            "error": "Document processor not initialized",
            "message": "Processing failed: Document processor not initialized",
        }
        logger.error("Document processor not initialized")
        return
    
    # If file is in S3 only, download it temporarily for processing
    temp_file = None
    actual_file_path = file_path
    
    if is_s3_only and storage_service:
        try:
            import tempfile
            logger.info(f"📥 Downloading file from S3 for processing: {file_path}")
            
            # Download from S3
            content = await storage_service.get_file(file_path)
            
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(content)
            temp_file.close()
            actual_file_path = temp_file.name
            
            logger.info(f"📄 File downloaded to temp location: {actual_file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download file from S3: {e}")
            processing_status[task_id] = {
                "status": "failed",
                "progress": 0,
                "doc_id": doc_id,
                "error": f"Failed to download file from S3: {str(e)}",
                "message": f"Processing failed: Could not access file",
            }
            return

    # prepare metadata dict (pydantic v2-friendly)
    meta = (
        metadata.model_dump()
        if hasattr(metadata, "model_dump")
        else {
            "title": metadata.title,
            "doc_type": metadata.doc_type,
            "owner": metadata.owner,
            "tags": metadata.tags,
        }
    )
    # ensure default fields
    meta.setdefault("version", "1.0")
    meta.setdefault("language", "en")

    # progress callback used by DocumentProcessor
    async def progress_cb(progress: float):
        # guard in case task_id was removed
        if task_id in processing_status:
            processing_status[task_id]["progress"] = int(progress)
            processing_status[task_id]["message"] = f"Processing... {int(progress)}%"

    try:
        logger.info(f"Processing document {doc_id}: {metadata.title}")

        result = await document_processor.process_document(
            pdf_path=actual_file_path,
            doc_id=doc_id,
            metadata=meta,
            progress_callback=progress_cb,  # make sure process_document accepts this
        )

        processing_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "doc_id": doc_id,
            "message": "Processing completed successfully",
            "result": result,
        }

        logger.info(f"✅ Document {doc_id} processed successfully")
        
        # Invalidate entity cache so it reloads on next query
        try:
            from workflow import tool_ctx
            tool_ctx.entities_loaded = False
            logger.info("♻️ Entity cache invalidated - will reload on next query")
        except Exception as e:
            logger.warning(f"Failed to invalidate entity cache: {e}")

    except Exception as e:
        logger.error(f"❌ Document processing failed for {doc_id}: {e}", exc_info=True)
        processing_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "doc_id": doc_id,
            "error": str(e),
            "message": f"Processing failed: {str(e)}",
        }
    finally:
        # Clean up temporary file if it was created
        if temp_file and Path(actual_file_path).exists():
            try:
                import os
                os.unlink(actual_file_path)
                logger.info(f"🗑️ Temporary file deleted: {actual_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


# Document Upload

@router.post("/upload")
@role_rate_limit("upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    doc_type: str = Form("manual"),
    owner: str = Form("global"),
    tags: str = Form(""),
    services: DocumentServices = Depends()
):
    """
    Upload and process a PDF document.
    
    Processing happens in background:
    1. Extract schema and structure
    2. Extract tables and figures
    3. Generate embeddings
    4. Store in Neo4j and Qdrant
    """
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise FileUploadError(
            message="Only PDF files are allowed",
            filename=file.filename
        )
    
    # Generate IDs
    task_id = str(uuid.uuid4())
    doc_id = f"doc_{int(datetime.now().timestamp())}"
    
    try:
        # Read file content once
        content = await file.read()
        
        # Determine storage strategy based on storage_type
        is_s3_only = settings.storage_type == "s3"
        file_path = None
        storage_path = f"uploads/{doc_id}.pdf"
        
        if is_s3_only:
            # S3 storage: save only to S3
            if not services.storage:
                raise FileUploadError(
                    message="S3 storage is configured but not initialized",
                    filename=file.filename
                )
            try:
                await services.storage.save_file(content, storage_path, content_type="application/pdf")
                logger.info(f"📤 File uploaded to S3: {storage_path}")
                file_path = storage_path  # Will be used to download later
            except Exception as e:
                logger.error(f"❌ Failed to upload to S3: {e}")
                raise FileUploadError(
                    message=f"Failed to upload to S3: {str(e)}",
                    filename=file.filename
                )
        else:
            # Local storage: save to local filesystem
            upload_dir = Path("data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            local_file_path = upload_dir / f"{doc_id}.pdf"
            
            with open(local_file_path, "wb") as f:
                f.write(content)
            
            logger.info(f"📄 File uploaded locally: {local_file_path}")
            file_path = str(local_file_path)
            
            # Optionally also save to S3 if storage service is available
            if services.storage:
                try:
                    await services.storage.save_file(content, storage_path, content_type="application/pdf")
                    logger.info(f"📤 File also backed up to S3: {storage_path}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to backup to S3 (non-critical): {e}")
        
        # Parse tags
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        # Create metadata
        metadata = DocumentMetadata(
            title=title,
            doc_type=doc_type,
            owner=owner,
            tags=tag_list
        )
        
        # Initialize processing status
        processing_status[task_id] = {
            "status": "queued",
            "progress": 0,
            "doc_id": doc_id,
            "message": "Document queued for processing"
        }
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            task_id=task_id,
            doc_id=doc_id,
            file_path=file_path,
            metadata=metadata,
            document_processor=services.processor,
            storage_service=services.storage,
            is_s3_only=is_s3_only
        )
        
        return {
            "task_id": task_id,
            "doc_id": doc_id,
            "message": "Document upload started",
            "status_endpoint": f"/documents/upload/status/{task_id}"
        }
        
    except FileUploadError:
        # Re-raise already typed exceptions
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise ProcessingError(
            message=f"Document upload failed: {file.filename}"
        )


@router.get("/upload/status/{task_id}")
async def get_upload_status(task_id: str):
    """Get document processing status."""
    if task_id not in processing_status:
        raise DocumentNotFoundError(task_id)
    
    return processing_status[task_id]


# Document Management

@router.get("/list", response_model=List[DocumentResponse])
async def list_documents(owner: Optional[str] = None, graph: Any = Depends(get_graph_client)):
    """
    List all documents.
    Optional filter by owner.
    """
    
    try:
        # Use get_all_documents from graph_service
        documents = await graph.get_all_documents(owner=owner)
        
        return [
            DocumentResponse(
                doc_id=doc["id"],
                title=doc["title"],
                status=doc.get("status", "completed"),
                created_at=str(doc.get("created_at", "")),
                total_pages=doc.get("total_pages"),
                owner=doc.get("owner")
            )
            for doc in documents
        ]
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise ProcessingError(
            message=f"Failed to list documents for owner: {owner}"
        )

@router.get("/{doc_id}")
async def get_document(doc_id: str, graph: Any = Depends(get_graph_client)):
    try:
        # metadata
        meta = await graph.get_document_metadata(doc_id)
        if not meta:
            raise DocumentNotFoundError(doc_id)

        # Try to get saved stats from metadata first
        stats = None
        metadata = meta.get("metadata", {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        saved_stats = metadata.get("stats") if metadata else None
        
        if saved_stats:
            # Use saved statistics from processing completion
            stats = {
                "chapters": saved_stats.get("chapters", 0),
                "sections": saved_stats.get("sections", 0),
                "text_chunks": saved_stats.get("text_chunks", 0),
                "tables": saved_stats.get("tables", 0),
                "table_chunks": saved_stats.get("table_chunks", 0),
                "schemas": saved_stats.get("schemas", 0),
                "entities": saved_stats.get("entities", 0),
            }
        else:
            # Fallback: calculate stats dynamically (slower)
            stats_raw = await graph.get_document_stats(doc_id)
            if stats_raw:
                stats = {
                    "chapters": stats_raw.get("chapters", 0),
                    "sections": stats_raw.get("sections", 0),
                    "text_chunks": 0,  # Not available in dynamic query
                    "tables": stats_raw.get("tables", 0),
                    "table_chunks": stats_raw.get("table_chunks", 0),
                    "schemas": stats_raw.get("schemas", 0),
                    "entities": stats_raw.get("entities", 0),
                }

        return {
            "doc_id":       meta.get("id"),
            "title":        meta.get("title"),
            "status":       meta.get("status", "completed"),
            "doc_type":     meta.get("doc_type"),
            "owner":        meta.get("owner"),
            "page_count":   meta.get("total_pages"),
            "uploaded_at":  meta.get("created_at") or "",
            "processed_at": meta.get("processed_at") or "",
            "tags":         meta.get("tags"),
            "metadata":     meta.get("metadata") or {},
            "stats":        stats,
        }

    except DocumentNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to get document details",
            doc_id=doc_id
        )

@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    request: Request,
    user_id: str = Header(None, alias="X-User-Id"),
    user_role: str = Header(None, alias="X-User-Role"),
    services: DocumentServices = Depends()
):
    """
    Delete a document and all associated data.
    Admins can delete any document, users can only delete their own.
    """
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    logger.info(f"Delete request for document {doc_id} by user {user_id} (role: {user_role})")
    
    try:
        # Get document metadata
        doc_metadata = await services.graph.get_document_metadata(doc_id)
        
        if not doc_metadata:
            raise DocumentNotFoundError(doc_id)
        
        doc_owner = doc_metadata.get("owner", "")
        doc_title = doc_metadata.get("title", "Unknown")
        
        # Authorization check: global documents can only be deleted by admin
        if doc_owner == "global" and user_role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Global documents can only be deleted by administrators"
            )
        
        # Regular authorization: users can only delete their own documents
        if user_role != "admin" and doc_owner != user_id:
            raise HTTPException(
                status_code=403,
                detail=f"You can only delete your own documents. This document belongs to '{doc_owner}'"
            )
        
        logger.info(f"Deleting document '{doc_title}' (owner: {doc_owner})")
        
        # ===== 1. DELETE FROM NEO4J =====
        deleted = await services.graph.delete_document(doc_id)
        
        if not deleted:
            raise DocumentNotFoundError(doc_id)
        
        logger.info(f"✅ Deleted document {doc_id} from Neo4j")
        
        # ===== 2. DELETE FROM QDRANT =====
        if services.vector:
            try:
                import asyncio
                await asyncio.to_thread(services.vector.delete_document_vectors, doc_id)
                logger.info(f"✅ Deleted vectors for document {doc_id} from Qdrant")
            except Exception as e:
                logger.error(f"⚠️ Failed to delete Qdrant vectors: {e}")
                # Don't fail the whole operation if Qdrant deletion fails
        
        # ===== 3. DELETE PHYSICAL FILES =====
        # Use storage service to delete all files (PDF, schemas, tables) from both local and S3
        try:
            deleted_count = await services.storage.delete_document_files(doc_id)
            logger.info(f"✅ Deleted {deleted_count} files for document {doc_id} from storage")
        except Exception as e:
            logger.error(f"⚠️ Failed to delete files from storage: {e}")
            deleted_count = 0
        
        # Also clean up local directories as fallback (in case of hybrid storage)
        deleted_files = []
        data_dir = Path(__file__).parent.parent.parent / "data"
        
        # Delete local PDF copies
        pdf_locations = [
            data_dir / "uploads" / f"{doc_id}.pdf",
            data_dir / "storage" / "pdfs" / f"{doc_id}.pdf",
        ]
        
        for pdf_path in pdf_locations:
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                    deleted_files.append(str(pdf_path))
                    logger.info(f"Deleted local PDF: {pdf_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {pdf_path}: {e}")
        
        # Delete schemas - stored in directories: data/schemas/original/{doc_id}/ and data/schemas/thumbnail/{doc_id}/
        import shutil
        
        schemas_dirs = [
            data_dir / "schemas" / "original" / doc_id,
            data_dir / "schemas" / "thumbnail" / doc_id,
        ]
        
        for schema_dir in schemas_dirs:
            if schema_dir.exists() and schema_dir.is_dir():
                try:
                    file_count = len(list(schema_dir.iterdir()))
                    shutil.rmtree(schema_dir)
                    deleted_files.append(f"{schema_dir} ({file_count} files)")
                    logger.info(f"Deleted schema directory: {schema_dir} ({file_count} files)")
                except Exception as e:
                    logger.error(f"Failed to delete schema directory {schema_dir}: {e}")
        
        # Delete tables - stored in directories: data/tables/original/{doc_id}/, data/tables/thumbnail/{doc_id}/, data/tables/csv/{doc_id}/
        tables_dirs = [
            data_dir / "tables" / "original" / doc_id,
            data_dir / "tables" / "thumbnail" / doc_id,
            data_dir / "tables" / "csv" / doc_id,
        ]
        
        for table_dir in tables_dirs:
            if table_dir.exists() and table_dir.is_dir():
                try:
                    file_count = len(list(table_dir.iterdir()))
                    shutil.rmtree(table_dir)
                    deleted_files.append(f"{table_dir} ({file_count} files)")
                    logger.info(f"Deleted table directory: {table_dir} ({file_count} files)")
                except Exception as e:
                    logger.error(f"Failed to delete table directory {table_dir}: {e}")
        
        logger.info(f"✅ Deleted {len(deleted_files)} local files for document {doc_id}")
        
        return {
            "status": "success",
            "message": f"Document '{doc_title}' deleted successfully",
            "doc_id": doc_id,
            "deleted_files": deleted_count + len(deleted_files),
            "details": {
                "neo4j": "✅ Deleted",
                "qdrant": "✅ Deleted" if services.vector else "⚠️ Skipped (not configured)",
                "storage": f"✅ Deleted {deleted_count} files from storage (S3/local)",
                "local_cleanup": f"✅ Deleted {len(deleted_files)} local files"
            }
        }
    
    except DocumentNotFoundError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting document {doc_id}: {e}", exc_info=True)
        raise ProcessingError(
            message="Failed to delete document",
            doc_id=doc_id
        )


@router.get("/{doc_id}/download")
async def download_document(doc_id: str, storage: Any = Depends(get_storage_service)):
    """Download original PDF file."""
    file_path = Path(f"data/uploads/{doc_id}.pdf")
    
    if not file_path.exists():
        raise DocumentNotFoundError(doc_id)
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"{doc_id}.pdf"
    )
"""
Response transformer for converting file paths to accessible URLs.
Handles both local and S3 storage, generating presigned URLs when needed.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.storage_service import StorageService

logger = logging.getLogger(__name__)


async def transform_response_urls(
    response: Dict[str, Any], 
    storage_service: 'StorageService',
    use_presigned: bool = True,
    expiration: int = 3600
) -> Dict[str, Any]:
    """
    Transform file paths in API response to accessible URLs.
    
    For S3 storage: Converts paths to presigned URLs
    For local storage: Keeps paths as-is (served by FastAPI static files)
    
    :param response: API response dictionary containing file paths
    :param storage_service: StorageService instance
    :param use_presigned: Generate presigned URLs for S3 (recommended for private buckets)
    :param expiration: Presigned URL expiration in seconds
    :return: Transformed response with accessible URLs
    """
    
    # Only transform if using S3 storage
    if storage_service.storage_type != "s3":
        return response
    
    # Transform figures/schemas
    if "figures" in response and isinstance(response["figures"], list):
        for figure in response["figures"]:
            if isinstance(figure, dict):
                # Transform main image URL
                if "url" in figure and isinstance(figure["url"], str):
                    if not figure["url"].startswith("http"):
                        # It's a relative path, convert to presigned URL
                        if use_presigned:
                            figure["url"] = await storage_service.get_presigned_url(
                                figure["url"], 
                                expiration=expiration
                            )
                        else:
                            figure["url"] = storage_service.get_file_url(figure["url"])
                
                # Transform thumbnail URL if present
                if "thumbnail_url" in figure and isinstance(figure["thumbnail_url"], str):
                    if not figure["thumbnail_url"].startswith("http"):
                        if use_presigned:
                            figure["thumbnail_url"] = await storage_service.get_presigned_url(
                                figure["thumbnail_url"],
                                expiration=expiration
                            )
                        else:
                            figure["thumbnail_url"] = storage_service.get_file_url(figure["thumbnail_url"])
    
    # Transform tables
    if "tables" in response and isinstance(response["tables"], list):
        for table in response["tables"]:
            if isinstance(table, dict):
                # Transform table image URL
                if "url" in table and isinstance(table["url"], str):
                    if not table["url"].startswith("http"):
                        if use_presigned:
                            table["url"] = await storage_service.get_presigned_url(
                                table["url"],
                                expiration=expiration
                            )
                        else:
                            table["url"] = storage_service.get_file_url(table["url"])
                
                # Transform CSV URL if present
                if "csv_url" in table and isinstance(table["csv_url"], str):
                    if not table["csv_url"].startswith("http"):
                        if use_presigned:
                            table["csv_url"] = await storage_service.get_presigned_url(
                                table["csv_url"],
                                expiration=expiration
                            )
                        else:
                            table["csv_url"] = storage_service.get_file_url(table["csv_url"])
    
    # Transform citations with file_path (if they contain file references)
    if "citations" in response and isinstance(response["citations"], list):
        for citation in response["citations"]:
            if isinstance(citation, dict) and "file_path" in citation:
                if isinstance(citation["file_path"], str) and not citation["file_path"].startswith("http"):
                    if use_presigned:
                        citation["file_path"] = await storage_service.get_presigned_url(
                            citation["file_path"],
                            expiration=expiration
                        )
                    else:
                        citation["file_path"] = storage_service.get_file_url(citation["file_path"])
    
    logger.debug(f"Transformed response URLs for S3 storage (presigned={use_presigned})")
    return response


def transform_response_urls_sync(
    response: Dict[str, Any],
    storage_service: 'StorageService',
    expiration: int = 3600
) -> Dict[str, Any]:
    """
    Synchronous version of transform_response_urls.
    Uses synchronous presigned URL generation.
    
    :param response: API response dictionary
    :param storage_service: StorageService instance
    :param expiration: Presigned URL expiration in seconds
    :return: Transformed response
    """
    
    if storage_service.storage_type != "s3":
        return response
    
    logger.info(f"🔄 Starting S3 URL transformation (bucket={storage_service.bucket_name}, prefix={storage_service.s3_prefix})")
    
    # Transform figures/schemas
    if "figures" in response and isinstance(response["figures"], list):
        for idx, figure in enumerate(response["figures"], 1):
            if isinstance(figure, dict):
                if "url" in figure and isinstance(figure["url"], str):
                    if not figure["url"].startswith("http"):
                        # Remove leading slash for S3 (storage_service expects relative paths)
                        original_path = figure["url"]
                        clean_path = original_path.lstrip('/')
                        transformed_url = storage_service.get_file_url(
                            clean_path, 
                            presigned=True, 
                            expiration=expiration
                        )
                        figure["url"] = transformed_url
                        # Log full URL for first figure to debug
                        if idx == 1:
                            logger.info(f"   📷 Figure {idx} FULL URL: {transformed_url}")
                        else:
                            logger.info(f"   📷 Figure {idx}: {original_path} → {transformed_url[:100]}...")
                
                if "thumbnail_url" in figure and isinstance(figure["thumbnail_url"], str):
                    if not figure["thumbnail_url"].startswith("http"):
                        clean_path = figure["thumbnail_url"].lstrip('/')
                        figure["thumbnail_url"] = storage_service.get_file_url(
                            clean_path,
                            presigned=True,
                            expiration=expiration
                        )
    
    # Transform tables
    if "tables" in response and isinstance(response["tables"], list):
        for table in response["tables"]:
            if isinstance(table, dict):
                if "url" in table and isinstance(table["url"], str):
                    if not table["url"].startswith("http"):
                        clean_path = table["url"].lstrip('/')
                        table["url"] = storage_service.get_file_url(
                            clean_path,
                            presigned=True,
                            expiration=expiration
                        )
                
                if "csv_url" in table and isinstance(table["csv_url"], str):
                    if not table["csv_url"].startswith("http"):
                        clean_path = table["csv_url"].lstrip('/')
                        table["csv_url"] = storage_service.get_file_url(
                            clean_path,
                            presigned=True,
                            expiration=expiration
                        )
    
    # Transform citations
    if "citations" in response and isinstance(response["citations"], list):
        for citation in response["citations"]:
            if isinstance(citation, dict) and "file_path" in citation:
                if isinstance(citation["file_path"], str) and not citation["file_path"].startswith("http"):
                    clean_path = citation["file_path"].lstrip('/')
                    citation["file_path"] = storage_service.get_file_url(
                        clean_path,
                        presigned=True,
                        expiration=expiration
                    )
    
    figures_count = len(response.get("figures", []))
    tables_count = len(response.get("tables", []))
    logger.info(f"🔄 Transformed S3 URLs: {figures_count} figures, {tables_count} tables (presigned, exp={expiration}s)")
    
    return response

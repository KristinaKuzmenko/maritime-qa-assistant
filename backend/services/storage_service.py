# services/storage_service.py
"""
Storage service for handling file operations (local and S3).
Optimized version with only used methods and improved error handling.
"""

import os
from pathlib import Path
from typing import Optional, Union
import logging
import aioboto3
from botocore.exceptions import ClientError
import aiofiles
import mimetypes
import asyncio

logger = logging.getLogger(__name__)


class StorageService:
    """Unified storage service supporting local and S3 storage."""
    
    def __init__(self, storage_type: str = "local", **kwargs):
        """
        Initialize storage service.
        
        :param storage_type: 'local' or 's3'
        :param kwargs: Storage-specific configuration
        """
        self.storage_type = storage_type
        
        if storage_type == "local":
            # Use absolute path relative to this file
            default_path = Path(__file__).parent.parent.parent / "data"
            self.base_path = Path(
                kwargs.get('local_storage_path', default_path)
            ).resolve()
            
            # Base URL for serving files (if behind nginx/apache)
            self.base_url = kwargs.get('base_url', '/files')
            
            self._ensure_directories()
            
            logger.info(f"Local storage initialized at: {self.base_path}")
            
        elif storage_type == "s3":
            self.bucket_name = kwargs.get('s3_bucket_name')
            if not self.bucket_name:
                raise ValueError("s3_bucket_name is required for S3 storage")
            
            self.region = kwargs.get('s3_region', 'us-east-1')
            self.aws_access_key = kwargs.get('aws_access_key_id')
            self.aws_secret_key = kwargs.get('aws_secret_access_key')
            
            # Create session
            self.session = aioboto3.Session(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region
            )
            
            # CloudFront distribution (optional, for faster access)
            self.cloudfront_domain = kwargs.get('cloudfront_domain')
            
            logger.info(f"S3 storage initialized: {self.bucket_name}")
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def _ensure_directories(self):
        """Ensure required directories exist for local storage."""
        directories = [
            'schemas/original',
            'schemas/thumbnail',
            'tables/original',
            'tables/csv',
            'tables/thumbnail',
            'uploads',
            'temp'
        ]
        
        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {dir_path}")
    
    # -------------------------------------------------------------------------
    # Core methods (production)
    # -------------------------------------------------------------------------
    
    async def save_file(
        self, 
        content: Union[bytes, 'BinaryIO'], 
        file_path: str, 
        content_type: Optional[str] = None,
        retry_count: int = 3,
    ) -> str:
        """
        Save file to storage with retry logic.
        
        :param content: File content (bytes or file-like object)
        :param file_path: Relative file path within storage
        :param content_type: MIME type (auto-detected if None)
        :param retry_count: Number of retries on failure
        :return: File URL or path for retrieval
        """
        for attempt in range(retry_count):
            try:
                if self.storage_type == "local":
                    return await self._save_local(content, file_path)
                else:
                    return await self._save_s3(content, file_path, content_type)
            
            except Exception as e:
                logger.error(
                    f"Error saving file (attempt {attempt + 1}/{retry_count}): {e}"
                )
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to save file after {retry_count} attempts") from e
    
    async def _save_local(self, content: Union[bytes, 'BinaryIO'], file_path: str) -> str:
        """Save file to local filesystem."""
        full_path = self.base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        if isinstance(content, bytes):
            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(content)
        else:
            # File-like object
            async with aiofiles.open(full_path, 'wb') as f:
                if hasattr(content, 'seek'):
                    content.seek(0)
                data = content.read()
                if asyncio.iscoroutine(data):
                    data = await data
                await f.write(data)
        
        logger.debug(f"File saved locally: {full_path}")
        
        # Return relative path (not absolute!)
        # This makes paths portable across different servers
        return file_path
    
    async def _save_s3(
        self, 
        content: Union[bytes, 'BinaryIO'], 
        file_path: str, 
        content_type: Optional[str] = None
    ) -> str:
        """Save file to S3 bucket."""
        # Auto-detect content type
        if not content_type:
            content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        
        # Prepare content
        if isinstance(content, bytes):
            body = content
        else:
            if hasattr(content, 'seek'):
                content.seek(0)
            body = content.read()
            if asyncio.iscoroutine(body):
                body = await body
        
        # Upload to S3
        async with self.session.client('s3') as s3:
            await s3.put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=body,
                ContentType=content_type,
                # Make publicly readable (adjust based on your security needs)
                # ACL='public-read',
            )
        
        logger.debug(f"File saved to S3: s3://{self.bucket_name}/{file_path}")
        
        # Return URL for file access
        if self.cloudfront_domain:
            return f"https://{self.cloudfront_domain}/{file_path}"
        else:
            return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{file_path}"
    
    # -------------------------------------------------------------------------
    # Optional methods (add only if needed)
    # -------------------------------------------------------------------------
    
    async def get_file(self, file_path: str) -> bytes:
        """
        Retrieve file content from storage.
        
        :param file_path: Relative file path or full URL
        :return: File content as bytes
        """
        if self.storage_type == "local":
            return await self._get_local(file_path)
        else:
            return await self._get_s3(file_path)
    
    async def _get_local(self, file_path: str) -> bytes:
        """Get file from local storage."""
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        async with aiofiles.open(full_path, 'rb') as f:
            content = await f.read()
        
        return content
    
    async def _get_s3(self, file_path: str) -> bytes:
        """Get file from S3."""
        # Handle full S3 URLs
        if file_path.startswith('s3://'):
            parts = file_path.replace('s3://', '').split('/', 1)
            if len(parts) == 2:
                file_path = parts[1]
        elif file_path.startswith('http'):
            # Extract key from HTTP URL
            file_path = file_path.split('/', 3)[-1]
        
        async with self.session.client('s3') as s3:
            response = await s3.get_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            content = await response['Body'].read()
        
        return content
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage.
        
        :param file_path: Relative file path
        :return: True if deleted successfully
        """
        try:
            if self.storage_type == "local":
                full_path = self.base_path / file_path
                if full_path.exists():
                    full_path.unlink()
                    logger.info(f"File deleted: {file_path}")
                    return True
                return False
            else:
                async with self.session.client('s3') as s3:
                    await s3.delete_object(
                        Bucket=self.bucket_name,
                        Key=file_path
                    )
                logger.info(f"File deleted from S3: {file_path}")
                return True
        
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in storage.
        
        :param file_path: Relative file path
        :return: True if file exists
        """
        try:
            if self.storage_type == "local":
                full_path = self.base_path / file_path
                return full_path.exists()
            else:
                async with self.session.client('s3') as s3:
                    await s3.head_object(
                        Bucket=self.bucket_name,
                        Key=file_path
                    )
                return True
        except:
            return False
    
    def get_file_url(self, file_path: str) -> str:
        """
        Get public URL for file access.
        
        :param file_path: Relative file path
        :return: Public URL
        """
        if self.storage_type == "local":
            # Return URL that can be served by your web server
            return f"{self.base_url}/{file_path}"
        else:
            # S3 public URL
            if self.cloudfront_domain:
                return f"https://{self.cloudfront_domain}/{file_path}"
            else:
                return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{file_path}"
    
    async def get_presigned_url(
        self, 
        file_path: str, 
        expiration: int = 3600
    ) -> str:
        """
        Generate presigned URL for temporary file access (S3 only).
        For local storage, returns regular URL.
        
        :param file_path: Relative file path
        :param expiration: URL expiration time in seconds
        :return: Presigned URL
        """
        if self.storage_type == "local":
            return self.get_file_url(file_path)
        
        try:
            async with self.session.client('s3') as s3:
                url = await s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': file_path},
                    ExpiresIn=expiration
                )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            # Fallback to public URL
            return self.get_file_url(file_path)
    
    # -------------------------------------------------------------------------
    # Batch operations 
    # -------------------------------------------------------------------------
    
    async def delete_document_files(self, doc_id: str) -> int:
        """
        Delete all files associated with a document.
        
        :param doc_id: Document ID
        :return: Number of files deleted
        """
        deleted_count = 0
        
        # Delete schemas
        for subdir in ['original', 'thumbnail']:
            prefix = f"schemas/{subdir}/{doc_id}/"
            
            if self.storage_type == "local":
                dir_path = self.base_path / prefix
                if dir_path.exists():
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_count += 1
                    # Remove empty directory
                    try:
                        dir_path.rmdir()
                    except:
                        pass
            else:
                # S3: list and delete
                async with self.session.client('s3') as s3:
                    try:
                        paginator = s3.get_paginator('list_objects_v2')
                        async for page in paginator.paginate(
                            Bucket=self.bucket_name,
                            Prefix=prefix
                        ):
                            if 'Contents' in page:
                                for obj in page['Contents']:
                                    await s3.delete_object(
                                        Bucket=self.bucket_name,
                                        Key=obj['Key']
                                    )
                                    deleted_count += 1
                    except ClientError as e:
                        logger.error(f"Error deleting S3 files: {e}")
        
        logger.info(f"Deleted {deleted_count} files for document {doc_id}")
        return deleted_count
    
    # -------------------------------------------------------------------------
    # Health check and stats
    # -------------------------------------------------------------------------
    
    async def health_check(self) -> dict:
        """
        Check storage health.
        
        :return: Health status dictionary
        """
        status = {
            "type": self.storage_type,
            "healthy": False,
            "error": None
        }
        
        try:
            if self.storage_type == "local":
                # Check if base path is writable
                test_file = self.base_path / "temp" / ".health_check"
                test_file.parent.mkdir(parents=True, exist_ok=True)
                
                async with aiofiles.open(test_file, 'w') as f:
                    await f.write("test")
                
                test_file.unlink()
                status["healthy"] = True
                status["path"] = str(self.base_path)
            
            else:
                # S3: try to list bucket
                async with self.session.client('s3') as s3:
                    await s3.head_bucket(Bucket=self.bucket_name)
                
                status["healthy"] = True
                status["bucket"] = self.bucket_name
        
        except Exception as e:
            status["error"] = str(e)
            logger.error(f"Storage health check failed: {e}")
        
        return status
    
    def get_stats(self) -> dict:
        """Get storage statistics."""
        stats = {
            "type": self.storage_type,
        }
        
        if self.storage_type == "local":
            stats["base_path"] = str(self.base_path)
            
            # Calculate disk usage
            total_size = 0
            file_count = 0
            
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            stats["total_size_bytes"] = total_size
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            stats["file_count"] = file_count
        
        else:
            stats["bucket"] = self.bucket_name
            stats["region"] = self.region
        
        return stats


# -------------------------------------------------------------------------
# Factory function for easy initialization
# -------------------------------------------------------------------------

def get_storage_service(
    storage_type: str = "local",
    **kwargs
) -> StorageService:
    """
    Factory function to create storage service instance.
    
    Example:
        # Local storage
        storage = get_storage_service("local", local_storage_path="/data")
        
        # S3 storage
        storage = get_storage_service(
            "s3",
            s3_bucket_name="my-bucket",
            aws_access_key_id="...",
            aws_secret_access_key="..."
        )
    """
    return StorageService(storage_type=storage_type, **kwargs)
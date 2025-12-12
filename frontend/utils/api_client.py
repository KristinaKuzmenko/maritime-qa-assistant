"""
API client for backend communication with rate limit handling.
"""

import requests
import time
import streamlit as st
from typing import Dict, Any, List, Optional, Callable
from config import API_BASE_URL



# Exceptions

class APIError(Exception):
    """Base API error."""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded error."""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")


# Rate Limit Handler

def handle_api_request(
    func: Callable, 
    *args, 
    max_retries: int = 3,
    show_progress: bool = True,
    **kwargs
) -> Any:
    """
    Handle API requests with rate limit retry logic.
    
    Args:
        func: Function to call
        max_retries: Maximum retry attempts (default: 3)
        show_progress: Show progress messages in Streamlit (default: True)
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Function result
    
    Raises:
        RateLimitError: If max retries exceeded
        APIError: For other API errors
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(e.response.headers.get('Retry-After', 60))
                
                if attempt < max_retries - 1:
                    if show_progress:
                        st.warning(
                            f"⏱️ Rate limit exceeded. "
                            f"Retrying in {retry_after} seconds... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                    time.sleep(retry_after)
                else:
                    raise RateLimitError(retry_after)
            else:
                raise APIError(f"API error: {e.response.status_code} - {e.response.text}")
        
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {e}")
        
        except Exception as e:
            raise APIError(f"Unexpected error: {e}")
    
    raise APIError("Max retries exceeded")


class APIClient:
    """Client for backend API communication."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    

    # Document Management
    
    def upload_document(
        self,
        file,
        title: str,
        doc_type: str,
        owner: str,
        tags: str,
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Upload document to API with rate limit handling."""
        def _upload():
            files = {"file": (file.name, file, "application/pdf")}
            data = {
                "title": title,
                "doc_type": doc_type,
                "owner": owner,
                "tags": tags,
            }
            
            response = requests.post(
                f"{self.base_url}/documents/upload",
                files=files,
                data=data,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        
        return handle_api_request(_upload, max_retries=3, show_progress=True)
    
    def get_processing_status(self, task_id: str) -> Dict[str, Any]:
        """Get document processing status."""
        response = requests.get(
            f"{self.base_url}/documents/upload/status/{task_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def list_documents(
        self,
        owner: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """List documents."""
        params = {}
        if owner:
            params["owner"] = owner
        if doc_ids:
            params["doc_ids"] = ",".join(doc_ids)
        
        response = requests.get(
            f"{self.base_url}/documents/list",
            params=params,
            headers=headers or {}
        )
        response.raise_for_status()
        return response.json()
    
    def get_document(self, doc_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Get document details."""
        response = requests.get(
            f"{self.base_url}/documents/{doc_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    def delete_document(self, doc_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Delete document."""
        response = requests.delete(
            f"{self.base_url}/documents/{doc_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
 
    # Q&A

    def ask_question(
        self,
        question: str,
        user_id: str,
        owner: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Ask a question with rate limit handling."""
        def _ask():
            data = {
                "question": question,
                "user_id": user_id,
                "owner": owner,
                "doc_ids": doc_ids,
                "chat_history": chat_history or []
            }
            
            response = requests.post(
                f"{self.base_url}/qa/answer",
                json=data,
                headers=headers or {}
            )
            response.raise_for_status()
            return response.json()
        
        return handle_api_request(_ask, max_retries=3, show_progress=True)
    
    def debug_question(
        self,
        question: str,
        user_id: str,
        owner: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Ask a question with debug info (admin only) with rate limit handling."""
        def _debug():
            data = {
                "question": question,
                "user_id": user_id,
                "owner": owner,
                "doc_ids": doc_ids,
                "chat_history": chat_history or []
            }
            
            response = requests.post(
                f"{self.base_url}/qa/debug",
                json=data,
                headers=headers or {}
            )
            response.raise_for_status()
            return response.json()
        
        return handle_api_request(_debug, max_retries=3, show_progress=True)

    # Health
    
    def health_check(self) -> bool:
        """Check API health."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


# Global API client instance
api_client = APIClient()
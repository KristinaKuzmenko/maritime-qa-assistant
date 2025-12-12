"""
Helper functions for Streamlit app.
"""

import requests
import streamlit as st
from typing import Dict, Any, List
from config import API_BASE_URL
from auth_config import credentials


def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_user_role(username: str) -> str:
    """Get user role from credentials."""
    return credentials['usernames'].get(username, {}).get('role', 'user')


def can_access_document(username: str, doc_owner: str) -> bool:
    """Check if user can access document."""
    role = get_user_role(username)
    if role == 'admin':
        return True
    return username == doc_owner


def get_request_headers(username: str = None, role: str = None) -> Dict[str, str]:
    """Build request headers with user context."""
    if username is None:
        username = st.session_state.get('username', 'anonymous')
    if role is None:
        role = st.session_state.get('role', 'guest')
    
    return {
        "X-User-Id": username,
        "X-User-Role": role,
    }


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def display_error(error: Exception, context: str = ""):
    """Display error message in standardized format."""
    from utils.api_client import RateLimitError, APIError
    
    error_msg = str(error)
    
    # Handle specific error types
    if isinstance(error, RateLimitError):
        st.error(
            f"⏱️ **Rate Limit Exceeded**\n\n"
            f"Too many requests. Please wait {error.retry_after} seconds before trying again.\n\n"
            f"💡 **Tip:** Reduce request frequency or upgrade your plan."
        )
    elif isinstance(error, APIError):
        st.error(f"❌ **API Error:** {error_msg}")
    elif "403" in error_msg:
        st.error(f"⛔ **Access Denied:** You don't have permission to {context}")
    elif "404" in error_msg:
        st.error(f"🔍 **Not Found:** {context}")
    elif "500" in error_msg:
        st.error(f"💥 **Server Error:** {context}\n\nPlease try again later or contact support.")
    elif "timeout" in error_msg.lower():
        st.error(f"⏰ **Timeout:** Request took too long. Please try again.")
    else:
        st.error(f"❌ **Error:** {error_msg}")


def paginate_items(items: List[Any], page: int, items_per_page: int) -> tuple:
    """
    Paginate a list of items.
    Returns: (items_for_page, total_pages)
    """
    total_pages = (len(items) + items_per_page - 1) // items_per_page
    start_idx = page * items_per_page
    end_idx = start_idx + items_per_page
    
    return items[start_idx:end_idx], total_pages


def display_pagination_controls(total_pages: int, current_page: int, key_prefix: str):
    """Display pagination controls."""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", disabled=current_page == 0, key=f"{key_prefix}_first"):
            st.session_state[f"{key_prefix}_page"] = 0
            st.rerun()
    
    with col2:
        if st.button("◀️ Prev", disabled=current_page == 0, key=f"{key_prefix}_prev"):
            st.session_state[f"{key_prefix}_page"] = current_page - 1
            st.rerun()
    
    with col3:
        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    
    with col4:
        if st.button("Next ▶️", disabled=current_page >= total_pages - 1, key=f"{key_prefix}_next"):
            st.session_state[f"{key_prefix}_page"] = current_page + 1
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", disabled=current_page >= total_pages - 1, key=f"{key_prefix}_last"):
            st.session_state[f"{key_prefix}_page"] = total_pages - 1
            st.rerun()
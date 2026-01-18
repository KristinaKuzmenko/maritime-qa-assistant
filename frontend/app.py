"""Streamlit web application for maritime technical documentation system.

Main entry point with authentication and page routing.
"""

import sys
from pathlib import Path

# Ensure the project root is importable so we can use stable package imports
# like `frontend.utils.*` regardless of how Streamlit sets sys.path.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import streamlit_authenticator as stauth

from frontend.auth_config import get_credentials
from frontend.config import AUTH_COOKIE_NAME, AUTH_SECRET_KEY, AUTH_COOKIE_EXPIRY_DAYS
from frontend.utils.style import apply_minimal_style

# Configure page
st.set_page_config(
    page_title="Maritime QA Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_minimal_style()


# Authentication Setup
# Load fresh credentials from DynamoDB on each page load
# Create authenticator on each run to ensure fresh credentials after user changes
if 'auth_reload_needed' in st.session_state:
    # Force reload after user management changes
    del st.session_state['auth_reload_needed']
    if 'authenticator' in st.session_state:
        del st.session_state['authenticator']

if 'authenticator' not in st.session_state:
    credentials = get_credentials()
    st.session_state['authenticator'] = stauth.Authenticate(
        credentials,
        cookie_name=AUTH_COOKIE_NAME,
        key=AUTH_SECRET_KEY,
        cookie_expiry_days=AUTH_COOKIE_EXPIRY_DAYS,
    )

authenticator = st.session_state['authenticator']


# Login

# Try to login - handle both old and new streamlit-authenticator versions
try:
    login_result = authenticator.login(location='main')
    
    if login_result is None:
        name = st.session_state.get("name")
        authentication_status = st.session_state.get("authentication_status")
        username = st.session_state.get("username")
    else:
        name, authentication_status, username = login_result
        
except TypeError:
    # Fallback for any version issues
    name = st.session_state.get("name")
    authentication_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

if authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
elif authentication_status:
    # Store user info in session state (always overwrite to avoid stale role across logins)
    st.session_state['username'] = username
    st.session_state['name'] = name
    # Get fresh credentials to ensure role is up-to-date
    fresh_credentials = get_credentials()
    st.session_state['role'] = fresh_credentials['usernames'][username].get('role', 'user')
    
    # Sidebar navigation
    with st.sidebar:
        st.title(f"Welcome, {name}")
        st.caption(f"Role: {st.session_state.get('role', 'user')}")
        
        # Navigation
        st.markdown("---")
        st.subheader("Navigation")
        
        # Import pages
        from frontend.app_pages import chat_page, upload_page, documents_page, admin_page, help_page
        from utils.helpers import check_api_health
        
        # Check API health
        api_status = check_api_health()
        if api_status:
            st.success("API: online")
        else:
            st.error("API: offline")
            st.caption("Start backend: `uvicorn main:app`")
        
        # Page selection
        pages = {
            "Search": chat_page,
            "Upload": upload_page,
            "Documents": documents_page,
            "Help": help_page,
        }
        
        # Add admin page for admins
        if st.session_state['role'] == 'admin':
            pages["Admin"] = admin_page

        # Button-based navigation (modern, no radio)
        if "selected_page" not in st.session_state:
            st.session_state["selected_page"] = next(iter(pages.keys()))

        # If user role changed (e.g., admin -> user), previously selected page may be invalid.
        if st.session_state.get("selected_page") not in pages:
            st.session_state["selected_page"] = next(iter(pages.keys()))

        for page_name in pages.keys():
            is_active = st.session_state.get("selected_page") == page_name
            if st.button(
                page_name,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["selected_page"] = page_name
                st.rerun()

        selected_page = st.session_state.get("selected_page", next(iter(pages.keys())))
        
        st.markdown("---")
        
        st.subheader("Account")
        authenticator.logout('Logout', 'sidebar')
    
    # Render selected page
    pages[selected_page].render()

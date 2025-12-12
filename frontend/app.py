"""
Streamlit web application for maritime technical documentation system.
Main entry point with authentication and page routing.
"""

import streamlit as st
import streamlit_authenticator as stauth
from auth_config import credentials, cookie

# Configure page
st.set_page_config(
    page_title="Maritime Documentation",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide the top navigation menu
hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


# Authentication Setup

authenticator = stauth.Authenticate(
    credentials,
    cookie['name'],
    cookie['key'],
    cookie['expiry_days'],
)


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
    # Store user info in session state
    if 'username' not in st.session_state:
        st.session_state['username'] = username
    if 'name' not in st.session_state:
        st.session_state['name'] = name
    if 'role' not in st.session_state:
        st.session_state['role'] = credentials['usernames'][username].get('role', 'user')
    
    # Sidebar navigation
    with st.sidebar:
        st.title(f"👋 Welcome, {name}!")
        st.caption(f"Role: {st.session_state.get('role', 'user')}")
        
        # Navigation
        st.markdown("---")
        st.subheader("Navigation")
        
        # Import pages
        from app_pages import chat_page, upload_page, documents_page, admin_page
        from utils.helpers import check_api_health
        
        # Check API health
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Offline")
            st.caption("Start backend: `uvicorn main:app`")
        
        # Page selection
        pages = {
            "🔍 Search & Q&A": chat_page,
            "📤 Upload Document": upload_page,
            "📚 My Documents": documents_page,
        }
        
        # Add admin page for admins
        if st.session_state['role'] == 'admin':
            pages["⚙️ Admin Panel"] = admin_page
        
        # Page selector
        selected_page = st.radio(
            "Go to",
            list(pages.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Logout button
        authenticator.logout('Logout', 'sidebar')
    
    # Render selected page
    pages[selected_page].render()

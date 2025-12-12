"""
Admin panel page.
"""

import streamlit as st
import pandas as pd
from auth_config import credentials, add_user, remove_user, list_users, change_password
from utils.api_client import api_client
from utils.helpers import get_request_headers, display_error


def render():
    """Render admin panel."""
    st.title("⚙️ Admin Panel")
    st.caption("System administration and user management")
    
    # Check if user is admin
    role = st.session_state.get('role', 'user')
    if role != 'admin':
        st.error("⛔ Access Denied: Admin privileges required")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["👥 Users", "📊 System Stats", "🔧 Settings"])
    
    with tab1:
        render_user_management()
    
    with tab2:
        render_system_stats()
    
    with tab3:
        render_settings()


def render_user_management():
    """Render user management section."""
    st.subheader("User Management")
    
    # User list
    st.markdown("### Current Users")
    
    users_data = list_users()
    
    if users_data:
        df = pd.DataFrame(users_data)
        df.columns = ["Username", "Name", "Role", "Email"]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No users found")
    
    # User statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", len(users_data))
    
    with col2:
        admin_count = sum(1 for u in users_data if u.get('role') == 'admin')
        st.metric("Admins", admin_count)
    
    with col3:
        user_count = sum(1 for u in users_data if u.get('role') == 'user')
        st.metric("Regular Users", user_count)
    
    # Add new user
    st.markdown("---")
    st.subheader("➕ Add New User")
    
    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username *")
            new_name = st.text_input("Full Name *")
        
        with col2:
            new_email = st.text_input("Email *")
            new_role = st.selectbox("Role *", ["user", "admin"])
        
        new_password = st.text_input("Password *", type="password")
        confirm_password = st.text_input("Confirm Password *", type="password")
        
        submitted = st.form_submit_button("Add User", type="primary", use_container_width=True)
        
        if submitted:
            if not all([new_username, new_name, new_email, new_password]):
                st.error("All fields are required")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                result = add_user(new_username, new_password, new_name, new_role, new_email)
                if "✅" in result:
                    st.success(result)
                    st.rerun()
                else:
                    st.error(result)
    
    # Remove user
    st.markdown("---")
    st.subheader("🗑️ Remove User")
    
    current_username = st.session_state.get("username")
    available_users = [u["username"] for u in users_data if u["username"] != current_username]
    
    if available_users:
        with st.form("remove_user_form"):
            user_to_remove = st.selectbox(
                "Select user to remove",
                options=available_users
            )
            
            confirm_remove = st.checkbox("⚠️ I confirm I want to remove this user")
            
            submitted = st.form_submit_button("Remove User", type="secondary", use_container_width=True)
            
            if submitted:
                if not confirm_remove:
                    st.warning("Please confirm removal by checking the box")
                else:
                    result = remove_user(user_to_remove)
                    if "✅" in result:
                        st.success(result)
                        st.rerun()
                    else:
                        st.error(result)
    else:
        st.info("No users available to remove (cannot remove yourself)")
    
    # Change password
    st.markdown("---")
    st.subheader("🔐 Change Password")
    
    with st.form("change_password_form"):
        user_for_password = st.selectbox(
            "Select user",
            options=[u["username"] for u in users_data]
        )
        new_pwd = st.text_input("New Password *", type="password")
        confirm_pwd = st.text_input("Confirm New Password *", type="password")
        
        submitted = st.form_submit_button("Change Password", use_container_width=True)
        
        if submitted:
            if not new_pwd:
                st.error("Password is required")
            elif new_pwd != confirm_pwd:
                st.error("Passwords do not match")
            elif len(new_pwd) < 6:
                st.error("Password must be at least 6 characters")
            else:
                result = change_password(user_for_password, new_pwd)
                if "✅" in result:
                    st.success(result)
                else:
                    st.error(result)


def render_system_stats():
    """Render system statistics."""
    st.subheader("System Statistics")
    
    username = st.session_state.get('username')
    role = st.session_state.get('role')
    
    try:
        headers = get_request_headers(username, role)
        
        # Get all documents
        all_docs = api_client.list_documents(headers=headers)
        
        # Calculate stats
        total_docs = len(all_docs)
        total_pages = sum(d.get('page_count', 0) for d in all_docs)
        processing_docs = sum(1 for d in all_docs if d.get('status') == 'processing')
        completed_docs = sum(1 for d in all_docs if d.get('status') == 'completed')
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", total_docs)
        
        with col2:
            st.metric("Total Pages", total_pages)
        
        with col3:
            st.metric("Processing", processing_docs)
        
        with col4:
            st.metric("Completed", completed_docs)
        
        # Document types
        st.markdown("---")
        st.subheader("📊 Documents by Type")
        
        doc_types = {}
        for doc in all_docs:
            doc_type = doc.get('doc_type', 'Unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        if doc_types:
            df_types = pd.DataFrame([
                {"Type": k, "Count": v}
                for k, v in sorted(doc_types.items(), key=lambda x: x[1], reverse=True)
            ])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(df_types, width='stretch')
            
            with col2:
                st.bar_chart(df_types.set_index('Type'))
        
        # Document ownership
        st.markdown("---")
        st.subheader("👥 Documents by Owner")
        
        owner_counts = {}
        for doc in all_docs:
            owner = doc.get('owner', 'unknown')
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
        
        if owner_counts:
            df_owners = pd.DataFrame([
                {"Owner": k, "Documents": v}
                for k, v in sorted(owner_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(df_owners, width='stretch')
            
            with col2:
                st.bar_chart(df_owners.set_index('Owner'))
    
    except Exception as e:
        display_error(e, "load system statistics")


def render_settings():
    """Render system settings."""
    st.subheader("System Settings")
    
    st.info("⚙️ Settings management coming soon...")
    
    # API endpoint configuration
    st.markdown("### 🔗 API Configuration")
    
    from config import API_BASE_URL
    st.code(f"API Base URL: {API_BASE_URL}")
    
    # Check API health
    if api_client.health_check():
        st.success("✅ API is online and responding")
    else:
        st.error("❌ API is offline or not responding")
        st.code("Start backend: uvicorn main:app --reload")
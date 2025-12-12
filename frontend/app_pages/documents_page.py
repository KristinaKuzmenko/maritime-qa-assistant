"""
Documents management page.
"""

import streamlit as st
from utils.api_client import api_client
import time
from utils.helpers import (
    get_request_headers, 
    display_error, 
    can_access_document,
    paginate_items,
    display_pagination_controls
)
from config import ITEMS_PER_PAGE


def render():
    """Render documents page."""
    st.title("📚 My Documents")
    st.caption("View and manage your uploaded documents")
    
    # Get user info
    username = st.session_state.get('username')
    role = st.session_state.get('role', 'user')
    
    # Initialize page state
    if 'docs_page' not in st.session_state:
        st.session_state['docs_page'] = 0
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if role == 'admin':
            owner_filter = st.text_input(
                "Filter by owner",
                placeholder="Leave empty for all documents"
            )
        else:
            owner_filter = username
            st.info(f"Showing: Your documents")
    
    with col2:
        doc_type_filter = st.selectbox(
            "Filter by type",
            options=["All", "Manual", "Technical Specification", "Maintenance Guide", "Safety Documentation", "Training Material", "Other"]
        )
    
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Load documents
    try:
        headers = get_request_headers(username, role)
        
        # Get documents
        if role == 'admin' and owner_filter:
            documents = api_client.list_documents(owner=owner_filter, headers=headers)
        elif role == 'admin':
            documents = api_client.list_documents(headers=headers)
        else:
            documents = api_client.list_documents(owner=username, headers=headers)
        
        # Filter by type
        if doc_type_filter != "All":
            documents = [d for d in documents if d.get('doc_type') == doc_type_filter]
        
        if not documents:
            st.info("📭 No documents found")
            return
        
        # Display stats
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Documents", len(documents))
        
        with col2:
            processing = sum(1 for d in documents if d.get('status') == 'processing')
            st.metric("Processing", processing)
        
        st.markdown("---")
        
        # Pagination
        current_page = st.session_state['docs_page']
        paginated_docs, total_pages = paginate_items(documents, current_page, ITEMS_PER_PAGE)
        
        # Display documents
        for doc in paginated_docs:
            with st.expander(f"📄 {doc.get('title', 'Untitled')}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**ID:** `{doc.get('doc_id')}`")
                    st.markdown(f"**Type:** {doc.get('doc_type', 'Unknown')}")
                    st.markdown(f"**Owner:** {doc.get('owner', 'Unknown')}")
                    st.markdown(f"**Status:** {doc.get('status', 'Unknown')}")
                    
                    if doc.get('tags'):
                        tags = doc['tags'].split(',')
                        st.markdown(f"**Tags:** {', '.join([f'`{t.strip()}`' for t in tags])}")
                    
                    if doc.get('page_count'):
                        st.markdown(f"**Pages:** {doc['page_count']}")
                    
                    if doc.get('uploaded_at'):
                        st.markdown(f"**Uploaded:** {doc['uploaded_at'][:10]}")
                
                with col2:
                    # Actions
                    st.markdown("**Actions:**")
                    
                    # View details button
                    if st.button("👁️ View Details", key=f"view_{doc['doc_id']}", use_container_width=True):
                        st.session_state['selected_doc'] = doc['doc_id']
                        st.rerun()
                    
                    # Delete button (if user has permission)
                    if can_access_document(username, doc.get('owner', '')):
                        if st.button("🗑️ Delete", key=f"delete_{doc['doc_id']}", use_container_width=True, type="secondary"):
                            st.session_state['confirm_delete'] = doc['doc_id']
                            st.rerun()
                
                # Show delete confirmation
                if st.session_state.get('confirm_delete') == doc['doc_id']:
                    st.warning(f"⚠️ Are you sure you want to delete '{doc.get('title')}'?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Yes, Delete", key=f"confirm_delete_{doc['doc_id']}", type="primary", use_container_width=True):
                            try:
                                api_client.delete_document(doc['doc_id'], headers)
                                st.success(f"✅ Document deleted successfully!")
                                del st.session_state['confirm_delete']
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                display_error(e, "delete document")
                    
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_delete_{doc['doc_id']}", use_container_width=True):
                            del st.session_state['confirm_delete']
                            st.rerun()
        
        # Pagination controls
        if total_pages > 1:
            st.markdown("---")
            display_pagination_controls(total_pages, current_page, "docs")
        
    except Exception as e:
        display_error(e, "load documents")
    
    # Document details modal
    if st.session_state.get('selected_doc'):
        show_document_details(st.session_state['selected_doc'], username, role)


def show_document_details(doc_id: str, username: str, role: str):
    """Show detailed document information in modal."""
    st.markdown("---")
    st.subheader("📄 Document Details")
    
    try:
        headers = get_request_headers(username, role)
        doc = api_client.get_document(doc_id, headers)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Title:** {doc.get('title')}")
            st.markdown(f"**ID:** `{doc.get('doc_id')}`")
            st.markdown(f"**Type:** {doc.get('doc_type')}")
        
        with col2:
            st.markdown(f"**Owner:** {doc.get('owner')}")
            st.markdown(f"**Status:** {doc.get('status')}")
            st.markdown(f"**Pages:** {doc.get('page_count', 0)}")
        
        if doc.get('tags'):
            st.markdown(f"**Tags:** {doc['tags']}")
        
        # Statistics
        if doc.get('stats'):
            st.markdown("### 📊 Statistics")
            stats = doc['stats']
            
            # Structure: Chapters → Sections → Text Chunks
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📚 Chapters", stats.get('chapters', 0))
            with col2:
                st.metric("📄 Sections", stats.get('sections', 0))
            with col3:
                st.metric("📝 Text Chunks", stats.get('text_chunks', 0))
            
            # Visual elements: Schemas and Tables
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("🔧 Schemas", stats.get('schemas', 0))
            with col5:
                st.metric("📊 Tables", stats.get('tables', 0))
            with col6:
                st.metric("📋 Table Chunks", stats.get('table_chunks', 0))
            
            # Entities
            if stats.get('entities', 0) > 0:
                st.metric("🏷️ Entities", stats.get('entities', 0))
        
        # Close button
        if st.button("✖️ Close", use_container_width=True):
            del st.session_state['selected_doc']
            st.rerun()
    
    except Exception as e:
        display_error(e, "load document details")
        if st.button("✖️ Close"):
            del st.session_state['selected_doc']
            st.rerun()
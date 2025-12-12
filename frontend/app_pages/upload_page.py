"""
Document upload page.
"""

import streamlit as st
import time
from utils.api_client import api_client
from utils.helpers import get_request_headers, display_error, format_file_size
from config import MAX_FILE_SIZE_MB, ALLOWED_FILE_TYPES


def render():
    """Render document upload page."""
    st.title("📤 Upload Document")
    st.caption("Upload maritime technical documentation for processing")
    
    # Get user info
    username = st.session_state.get('username')
    role = st.session_state.get('role', 'user')
    
    # Upload form
    with st.form("upload_form", clear_on_submit=True):
        st.subheader("Document Information")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=ALLOWED_FILE_TYPES,
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Document title
            title = st.text_input(
                "Document Title *",
                help="Descriptive title for the document"
            )
            
            # Document type
            doc_type = st.selectbox(
                "Document Type *",
                options=[
                    "Manual",
                    "Technical Specification",
                    "Maintenance Guide",
                    "Safety Documentation",
                    "Training Material",
                    "Other"
                ]
            )
        
        with col2:
            # Owner (admin can set any owner)
            if role == 'admin':
                owner = st.text_input(
                    "Owner *",
                    value=username,
                    help="Document owner username"
                )
            else:
                owner = username
                st.text_input(
                    "Owner",
                    value=username,
                    disabled=True,
                    help="You are the document owner"
                )
            
            # Tags
            tags = st.text_input(
                "Tags",
                placeholder="engine, maintenance, safety",
                help="Comma-separated tags"
            )
        
        # Submit button
        submitted = st.form_submit_button("📤 Upload & Process", type="primary", use_container_width=True)
        
        if submitted:
            # Validation
            if not uploaded_file:
                st.error("Please select a file to upload")
            elif not title:
                st.error("Please provide a document title")
            elif not owner:
                st.error("Please specify document owner")
            elif uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File size exceeds maximum of {MAX_FILE_SIZE_MB}MB")
            else:
                # Upload document
                try:
                    with st.spinner("Uploading document..."):
                        headers = get_request_headers(username, role)
                        
                        response = api_client.upload_document(
                            file=uploaded_file,
                            title=title,
                            doc_type=doc_type,
                            owner=owner,
                            tags=tags,
                            headers=headers
                        )
                        
                        task_id = response.get('task_id')
                        doc_id = response.get('doc_id')
                        
                        st.success(f"✅ Document uploaded successfully!")
                        st.info(f"📄 Document ID: `{doc_id}`")
                        st.info(f"🔄 Processing Task ID: `{task_id}`")
                        
                        # Show processing status
                        st.markdown("---")
                        st.subheader("📊 Processing Status")
                        
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Poll for status
                        max_attempts = 60  # 5 minutes max
                        attempt = 0
                        
                        while attempt < max_attempts:
                            try:
                                status = api_client.get_processing_status(task_id)
                                
                                current_status = status.get('status', 'unknown')
                                progress = status.get('progress', 0)
                                message = status.get('message', '')
                                
                                # Update UI
                                status_placeholder.info(f"**Status:** {current_status}\n\n{message}")
                                progress_bar.progress(progress / 100)
                                
                                # Check if complete
                                if current_status == 'completed':
                                    status_placeholder.success(f"✅ Processing completed!\n\n{message}")
                                    progress_bar.progress(1.0)
                                    st.balloons()
                                    break
                                elif current_status == 'failed':
                                    status_placeholder.error(f"❌ Processing failed\n\n{message}")
                                    break
                                
                                time.sleep(5)
                                attempt += 1
                                
                            except Exception as e:
                                status_placeholder.warning(f"Could not fetch status: {e}")
                                break
                        
                        if attempt >= max_attempts:
                            st.warning("⏰ Status polling timeout. Processing continues in background.")
                        
                except Exception as e:
                    display_error(e, "upload document")
    
    # Instructions
    st.markdown("---")
    st.subheader("📋 Upload Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Supported Documents:**
        - PDF format only
        - Maritime technical manuals
        - Maintenance guides
        - Safety documentation
        - Training materials
        """)
    
    with col2:
        st.markdown("""
        **Processing includes:**
        - Text extraction
        - Table detection & extraction
        - Schema/diagram extraction
        - Vector embedding generation
        - Graph database indexing
        """)
    
    st.info("💡 **Tip:** Processing time depends on document size and complexity. Large documents may take several minutes.")
"""
Search and Q&A page.
"""

import streamlit as st

from frontend.utils.api_client import api_client
from frontend.utils.helpers import get_request_headers, display_error

MAX_CONTEXT_MESSAGES = 20


def get_image_url(url: str) -> str:
    """Convert relative path to full URL, or return presigned URL as-is."""
    if url.startswith('http://') or url.startswith('https://'):
        return url  # Already a full URL (presigned S3 URL)
    return f"http://localhost:8000{url}"  # Local path, add base URL

def render():
    """Render search and Q&A page."""
    st.title("Search")
    st.caption("Ask questions about your documents")
    
    # Get user info
    username = st.session_state.get('username')
    role = st.session_state.get('role', 'user')
    
    # Clear chat history if user changed (to avoid seeing previous user's chats)
    if 'last_chat_username' not in st.session_state or st.session_state['last_chat_username'] != username:
        st.session_state['chat_history'] = []
        st.session_state['last_chat_username'] = username
        st.rerun()
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Settings in sidebar
    with st.sidebar:
        st.subheader("Search settings")
        
        # Owner filter
        if role == 'admin':
            owner_filter = st.text_input(
                "Filter by owner",
                value="",
                help="Leave empty to search all documents"
            )
        else:
            owner_filter = username
            st.info("Scope: your documents")
        
        # Document filter
        try:
            headers = get_request_headers()
            docs_list = api_client.list_documents(
                owner=owner_filter if owner_filter else None,
                headers=headers
            )
            
            if docs_list and len(docs_list) > 0:
                doc_options = {
                    "All documents": None,
                    **{
                        f"{doc['title']} ({doc.get('total_pages', '?')} pages)": doc['doc_id']
                        for doc in docs_list
                    }
                }
                
                selected_doc = st.selectbox(
                    "Filter by document",
                    options=list(doc_options.keys()),
                    help="Search only in selected document(s)"
                )
                
                doc_ids_filter = [doc_options[selected_doc]] if doc_options[selected_doc] else None
            else:
                doc_ids_filter = None
                st.info("No documents available")
        except Exception as e:
            doc_ids_filter = None
            st.warning(f"Could not load documents: {str(e)}")
        
        # Debug mode (admin only)
        debug_mode = False
        if role == 'admin':
            debug_mode = st.checkbox(
                "Debug mode",
                value=False,
                help="Show detailed workflow information"
            )
        
        # Clear chat button
        if st.button("Clear chat history", use_container_width=True):
            st.session_state['chat_history'] = []
            st.rerun()
    
    # Display chat history
    st.markdown("### Conversation")
    
    for i, msg in enumerate(st.session_state['chat_history']):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display figures/tables if available
            if msg["role"] == "assistant":
                if msg.get("figures"):
                    with st.expander("Figures", expanded=False):
                        for fig in msg["figures"]:
                            st.image(
                                get_image_url(fig['url']),
                                caption=f"{fig['title']} - Page {fig['page']}"
                            )
                
                if msg.get("tables"):
                    with st.expander("Tables", expanded=False):
                        for table in msg["tables"]:
                            st.markdown(f"**{table['title']}** (Page {table['page']})")
                            if table.get('url'):
                                st.image(get_image_url(table['url']))
    
    # Chat input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Add user message to chat
        st.session_state['chat_history'].append({
            "role": "user",
            "content": question
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    headers = get_request_headers(username, role)

                    # Prepare chat history for API (last 10 messages, exclude current question)
                    # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                    api_chat_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state['chat_history'][:-1]  # Exclude current question
                    ][-MAX_CONTEXT_MESSAGES:]  # Last N messages max for context
                    
                    if debug_mode:
                        # Use debug endpoint
                        result = api_client.debug_question(
                            question=question,
                            user_id=username,
                            owner=owner_filter if owner_filter else None,
                            doc_ids=doc_ids_filter,
                            chat_history=api_chat_history,
                            headers=headers
                        )
                        
                        # Show debug info
                        with st.expander("Debug information", expanded=True):
                            st.markdown("### Query analysis")
                            st.json(result.get('step_1_query_analysis', {}))
                            
                            st.markdown("### Agent tool selection")
                            st.json(result.get('step_2_router_agent', {}))
                            
                            st.markdown("### Tool execution")
                            st.json(result.get('step_3_tool_execution', {}))
                            
                            st.markdown("### Anchor selection")
                            st.json(result.get('step_4_anchor_selection', {}))
                            
                            st.markdown("### Context building")
                            st.json(result.get('step_5_context_building', {}))
                            
                            st.markdown("### Answer generation")
                            st.json(result.get('step_6_answer', {}))
                        
                        # Extract answer from debug response
                        answer_data = result.get('answer', {})
                        answer_text = answer_data.get('answer_text', 'No answer')
                        figures = answer_data.get('figures', [])
                        tables = answer_data.get('tables', [])
                    else:
                        # Use normal endpoint
                        result = api_client.ask_question(
                            question=question,
                            user_id=username,
                            owner=owner_filter if owner_filter else None,
                            doc_ids=doc_ids_filter,
                            chat_history=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state['chat_history'][:-1]
                            ],
                            headers=headers
                        )
                        
                        answer_text = result.get('answer', 'No answer')
                        figures = result.get('figures', [])
                        tables = result.get('tables', [])
                    
                    # Display answer
                    st.markdown(answer_text)
                    
                    # Display figures
                    if figures:
                        with st.expander("Figures", expanded=False):
                            for fig in figures:
                                st.image(
                                    get_image_url(fig['url']),
                                    caption=f"{fig['title']} - Page {fig['page']}"
                                )
                    
                    # Display tables
                    if tables:
                        with st.expander("Tables", expanded=False):
                            for table in tables:
                                st.markdown(f"**{table['title']}** (Page {table['page']})")
                                if table.get('url'):
                                    st.image(get_image_url(table['url']))
                    
                    # Add assistant message to chat
                    st.session_state['chat_history'].append({
                        "role": "assistant",
                        "content": answer_text,
                        "figures": figures,
                        "tables": tables
                    })
                    
                except Exception as e:
                    display_error(e, "get answer")
                    st.session_state['chat_history'].pop()  # Remove user message on error
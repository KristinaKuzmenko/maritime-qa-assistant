"""Help / FAQ page."""

import streamlit as st


def render() -> None:
    st.title("Help")
    st.caption("How to search, upload, and manage documents")

    st.markdown("### FAQ")

    with st.expander("How to use search", expanded=True):
        st.markdown(
            """
- Open **Search**.
- Type your question in the chat input.
- Use **Filter by document** in the sidebar if you want to search inside one document only.
- If you are an admin, you can also set **Filter by owner**.

Tips:
- Ask one question at a time.
- If the answer looks incomplete, ask a follow-up and reference the section/page you need.
            """
        )

    with st.expander("How to upload a document"):
        st.markdown(
            """
- Open **Upload**.
- Select a PDF file.
- Fill **Document Title** and **Document Type**.
- Click **Upload & Process** and wait for processing to finish.

Notes:
- Processing time depends on file size and layout complexity.
- After upload, the document will appear in **Documents**.
            """
        )

    with st.expander("How to delete a document"):
        st.markdown(
            """
- Open **Documents**.
- Find the document in the list.
- Click **Delete**.
- Confirm deletion.

Only the document owner (or an admin) can delete a document.
            """
        )

    with st.expander("How to use the document filter for search"):
        st.markdown(
            """
- Open **Search**.
- In the sidebar, use **Filter by document**.
- Choose **All documents** to search across everything available to you.
- Choose a specific document to narrow results and reduce noise.

When to use it:
- When you know which manual/spec contains the answer.
- When you need consistent terminology and fewer cross-document matches.
            """
        )

    with st.expander("What questions work best (and how to phrase them)"):
        st.markdown(
            """
Best:
- Specific, technical questions with a clear target.
- Questions that include relevant context (component name, system, condition).

Good examples:
- "What is the recommended torque for the cylinder head bolts for model X?"
- "Which safety checks are required before starting maintenance procedure Y?"
- "What is the alarm threshold for parameter Z and what actions are recommended?"

Less effective:
- Very broad prompts like "Explain the whole manual".
- Questions without a subject or device context.

Formatting tips:
- Include identifiers: model, part number, chapter/section name if you know it.
- If you need a table value, ask for the exact row/column criteria.
            """
        )

import streamlit as st

from vector_database import index_documents
from rag_pipeline import answer_query

st.set_page_config(
    page_title="AI Legal Assistant",
)

st.title("AI Legal Assistant")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


uploaded_files = st.file_uploader(
    "Upload Legal PDFs",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            st.session_state.vector_db = index_documents(uploaded_files)

        st.success("PDFs processed successfully!")


user_query = st.text_area(
    "Ask a Legal Question",
    height=120,
    placeholder="Ask anything about uploaded documents..."
)


if st.button("Ask AI Lawyer"):

    if st.session_state.vector_db is None:
        st.error("Please upload and process PDFs first.")
    elif not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = answer_query(user_query,st.session_state.vector_db)

        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            st.write(response["answer"])
            st.subheader("Retrieved Legal Sources")

            for idx, doc in enumerate(response["documents"]):
                source = doc.metadata.get("source","Unknown Source")

                page = doc.metadata.get("page","N/A")

                with st.expander(f"Source {idx+1} | {source} | Page {page}"):
                    st.write(doc.page_content)

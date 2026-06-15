import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

embedding_model = load_embedding_model()


# LOAD PDF — written to a temp file, deleted immediately after parsing
def load_uploaded_pdf(uploaded_file):
    temp_path = None

    try:
        # case1 : for local file path(evaluation)
        if isinstance(uploaded_file, str):
            loader = PDFPlumberLoader(uploaded_file)
            documents = loader.load()
            return documents

        # case2: for streamlit uploaded file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        loader = PDFPlumberLoader(temp_path)
        documents = loader.load()

        for document in documents:
            document.metadata["source"] = uploaded_file.name

        return documents

    finally:
        # pass
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# CHUNK DOCUMENTS
def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    return chunks


# CORE PIPELINE VECTORDB
def index_documents(uploaded_files):
    all_documents = []

    for uploaded_file in uploaded_files:
        documents = load_uploaded_pdf(uploaded_file)
        all_documents.extend(documents)

    chunks = create_chunks(all_documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    bm25_retriever=BM25Retriever.from_documents(chunks)
    bm25_retriever.k=5

    return {
        "vector_db": vector_db,
        "bm25_retriever": bm25_retriever
    }

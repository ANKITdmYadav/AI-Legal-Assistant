# AI Legal Assistant

An intelligent Legal RAG (Retrieval-Augmented Generation) system that enables users to query legal documents in natural language and receive context-grounded responses.

## Features

* Upload and process legal PDF documents
* Semantic search using Chroma Vector Database
* Hybrid Retrieval (Dense Retrieval + BM25)
* Reciprocal Rank Fusion (RRF) for result aggregation
* Cross-Encoder Reranking for improved retrieval quality
* Source-grounded legal responses with citations
* Streamlit-based interactive user interface
* RAGAS evaluation for measuring retrieval and generation quality

## Tech Stack

* Python
* LangChain
* ChromaDB
* HuggingFace Embeddings (all-MiniLM-L6-v2)
* BM25 Retriever
* BAAI bge-reranker-base
* Groq LLM (GPT-OSS-120B)
* Streamlit
* RAGAS

## System Architecture

PDF Documents → Chunking → Embeddings → Chroma Vector Store

User Query → Vector Retrieval + BM25 Retrieval → RRF Fusion → Cross-Encoder Reranking → LLM → Grounded Answer

## Evaluation

The system is evaluated using RAGAS metrics:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

## Run Application

```bash
streamlit run app.py
```

## Project Structure

```text
├── app.py
├── config.py
├── rag_pipeline.py
├── vector_database.py
├── evaluate_rag.py
├── evaluation_dataset.py
├── requirements.txt
└── .env
```

## Disclaimer

This project is intended for legal information retrieval and document-based question answering. Responses should not be considered professional legal advice.

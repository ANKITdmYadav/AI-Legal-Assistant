import os
from dotenv import load_dotenv

load_dotenv()

# ==========================
# MODEL CONFIG
# ==========================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

# ==========================
# RETRIEVAL CONFIG
# ==========================

TOP_K_RESULTS = 4

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# RERANKER CONFIG
RERANKER_MODEL = "BAAI/bge-reranker-base"

INITIAL_RETRIEVAL_K = 10
FINAL_TOP_K = 4

# PATH
EMBEDDING_PATH = "artifacts/embedding_model"
RERANKER_PATH = "artifacts/reranker_model"


# ==========================
# ENV VARIABLES
# ==========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

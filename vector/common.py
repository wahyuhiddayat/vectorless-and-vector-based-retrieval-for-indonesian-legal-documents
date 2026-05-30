"""Shared utilities for vector RAG retrieval pipelines."""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.environ.get("VECTOR_EMBEDDING_MODEL", "bge-m3")
COLLECTION_NAME = os.environ.get("VECTOR_COLLECTION", "law-pasal-bgem3")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.environ.get("QDRANT_PATH", None)
GRANULARITY = os.environ.get("VECTOR_GRANULARITY", "pasal")
RERANKER = os.environ.get("VECTOR_RERANKER", "none")
LOG_DIR = Path("data/retrieval_logs")

RERANKER_TOP_N = int(os.environ.get("VECTOR_RERANKER_TOP_N", "50"))
"""First-stage candidates fed to the reranker."""

HNSW_EF_SEARCH = int(os.environ.get("VECTOR_HNSW_EF_SEARCH", "128"))
"""Qdrant HNSW search-time exploration depth. Higher means better recall, slower query."""

RERANKER_FP32 = os.environ.get("VECTOR_RERANKER_FP32", "0") == "1"
"""Force reranker weights to float32 on CUDA. Default is bfloat16 (smaller, faster)."""

_EMBEDDING_MODEL_MAP: dict[str, dict] = {
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dim": 1024,
        "backend": "sentence_transformers",
    },
    "multilingual-e5-large-instruct": {
        "model_id": "intfloat/multilingual-e5-large-instruct",
        "dim": 1024,
        "backend": "sentence_transformers",
        "query_instruction": (
            "Given a legal question in Indonesian, retrieve relevant legal "
            "document sections that answer the question"
        ),
    },
    "all-nusabert-large-v4": {
        "model_id": "LazarusNLP/all-nusabert-large-v4",
        "dim": 1024,
        "backend": "sentence_transformers",
    },
}

_RERANKER_REGISTRY: dict[str, dict] = {
    "none": {
        "model_id": None,
        "backend": "none",
    },
    "bge-reranker-v2-m3": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "backend": "cross_encoder",
        "predict_batch_size": 128,
    },
    "qwen3-reranker-0.6b": {
        # Smaller batch size because decoder KV-cache scales with batch * seqlen
        "model_id": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        "backend": "cross_encoder",
        "predict_batch_size": 16,
    },
    "bge-reranker-v2-gemma": {
        # 2.5B-param Gemma-based reranker. Larger than v2-m3 (568M), needs more VRAM.
        "model_id": "BAAI/bge-reranker-v2-gemma",
        "backend": "cross_encoder",
        "predict_batch_size": 32,
    },
}


_qdrant_client_cache = None


def get_qdrant_client():
    """Return a cached Qdrant client for local-path or server mode."""
    global _qdrant_client_cache
    if _qdrant_client_cache is not None:
        return _qdrant_client_cache
    from qdrant_client import QdrantClient
    if QDRANT_PATH:
        _qdrant_client_cache = QdrantClient(path=QDRANT_PATH)
    else:
        _qdrant_client_cache = QdrantClient(url=QDRANT_URL)
    return _qdrant_client_cache


_st_model_cache: dict = {}


def _get_st_model(model_id: str):
    """Create and cache a SentenceTransformer model."""
    if model_id not in _st_model_cache:
        from sentence_transformers import SentenceTransformer
        _st_model_cache[model_id] = SentenceTransformer(model_id)
    return _st_model_cache[model_id]


def embed_query(query: str) -> list[float]:
    """Embed a query with the configured SentenceTransformer model."""
    cfg = _EMBEDDING_MODEL_MAP.get(EMBEDDING_MODEL)
    if not cfg:
        raise ValueError(f"Unknown embedding model: {EMBEDDING_MODEL!r}")

    st = _get_st_model(cfg["model_id"])
    instruction = cfg.get("query_instruction")
    text = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
    vec = st.encode(text, normalize_embeddings=True)
    return [float(x) for x in vec]


def embed_queries(queries: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed multiple queries in one batched forward pass."""
    cfg = _EMBEDDING_MODEL_MAP.get(EMBEDDING_MODEL)
    if not cfg:
        raise ValueError(f"Unknown embedding model: {EMBEDDING_MODEL!r}")

    st = _get_st_model(cfg["model_id"])
    instruction = cfg.get("query_instruction")
    texts = [
        f"Instruct: {instruction}\nQuery: {q}" if instruction else q
        for q in queries
    ]
    vecs = st.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [[float(x) for x in v] for v in vecs]


def save_log(result: dict):
    """Persist a retrieval result under `data/retrieval_logs`."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy = result.get("strategy", "unknown").replace(" ", "_")
    log_path = LOG_DIR / f"{timestamp}_{strategy}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path.name}")

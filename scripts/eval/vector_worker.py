"""Subprocess worker for vector RAG evaluation.

Two operating modes.

Single-query mode (legacy, useful for debugging).
    Pass --query "..." and the worker runs once and exits with one JSON
    payload on stdout. Same as the original behaviour.

Batch mode (default when --query is omitted).
    Worker reads newline-delimited JSON objects from stdin, one per query,
    keeps the embedding model and reranker resident in GPU memory across
    queries, and writes one JSON payload per query to stdout (flushed each
    line). A line equal to "--DONE--" ends the loop cleanly. Eliminates the
    per-query model-reload overhead that dominated wall time in earlier runs.

stdin schema (batch mode):
    {"qid": "q033", "query": "Apa syarat penyadapan?", "top_k": 10}

stdout schema (one payload per input line, qid preserved for matching):
    {"qid": "q033", "ok": true, "system": "vector-dense",
     "granularity": "pasal", "embedding_model": "bge-m3",
     "reranker": "none", "collection": "law-pasal-bgem3",
     "llm_model": null, "result": {...}}

Usage:
    # Single-query (debug)
    python scripts/eval/vector_worker.py \\
        --system vector-dense --granularity pasal \\
        --embedding-model bge-m3 --query "Apa syarat penyadapan?" \\
        --qdrant-path ./qdrant_local

    # Batch (driven by the orchestrator)
    echo '{"qid":"q1","query":"Apa syarat?","top_k":10}\\n--DONE--' | \\
      python scripts/eval/vector_worker.py \\
        --system vector-dense --granularity pasal \\
        --embedding-model bge-m3 --reranker none \\
        --qdrant-path ./qdrant_local
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MODEL_SHORT = {
    "bge-m3": "bgem3",
    "multilingual-e5-large-instruct": "e5",
    "all-nusabert-large-v4": "nusabert",
}

_RERANKER_CHOICES = ["none", "bge-reranker-v2-m3", "qwen3-reranker-0.6b"]

_BATCH_END_SENTINEL = "--DONE--"


def _run_retrieval(retrieve_fn, query: str, top_k: int) -> dict:
    """Call retrieve_vector.retrieve with verbose disabled, return raw result."""
    return retrieve_fn(query, top_k=top_k, verbose=False)


def _make_payload(args, llm_model, qid: str | None, top_k: int, query: str,
                  retrieve_fn) -> dict:
    """Run one retrieval and wrap the response in the standard payload shape."""
    try:
        result = _run_retrieval(retrieve_fn, query, top_k)
        payload = {
            "ok": True,
            "system": args.system,
            "granularity": args.granularity,
            "embedding_model": args.embedding_model,
            "reranker": args.reranker,
            "collection": f"law-{args.granularity}-{_MODEL_SHORT[args.embedding_model]}",
            "llm_model": llm_model,
            "result": result,
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "system": args.system,
            "granularity": args.granularity,
            "embedding_model": args.embedding_model,
            "reranker": args.reranker,
            "llm_model": llm_model,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    if qid is not None:
        payload["qid"] = qid
    return payload


def main() -> int:
    """Parse args, set env vars before imports, then dispatch single or batch mode."""
    ap = argparse.ArgumentParser(
        description="Run vector retrieval calls in a fresh process."
    )
    ap.add_argument("--system", required=True, choices=["vector-dense"])
    ap.add_argument("--granularity", required=True,
                    choices=["pasal", "ayat", "rincian"])
    ap.add_argument("--embedding-model", required=True,
                    choices=list(_MODEL_SHORT),
                    help="Embedding model. bge-m3 | multilingual-e5-large-instruct | "
                         "all-nusabert-large-v4")
    ap.add_argument("--reranker", default="none", choices=_RERANKER_CHOICES,
                    help="Reranker. none | bge-reranker-v2-m3 | qwen3-reranker-0.6b")
    ap.add_argument("--query", default=None,
                    help="Single-query mode. If omitted, worker reads batch from stdin.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--qdrant-path", default=None,
                    help="Path to local Qdrant storage directory")
    args = ap.parse_args()

    # Env vars must be set BEFORE importing vector modules, they read at import time.
    model_short = _MODEL_SHORT[args.embedding_model]
    collection = f"law-{args.granularity}-{model_short}"
    os.environ["VECTOR_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["VECTOR_COLLECTION"] = collection
    os.environ["VECTOR_GRANULARITY"] = args.granularity
    os.environ["VECTOR_RERANKER"] = args.reranker
    if args.qdrant_path:
        os.environ["QDRANT_PATH"] = args.qdrant_path

    # Import once. First retrieve() call loads the embedding model and (optionally)
    # the reranker into GPU memory. Subsequent calls reuse the module-level caches
    # in vector/common.py and vector/rerank.py.
    from vector.retrieve_vector import retrieve

    try:
        from vectorless.llm import MODEL as _ans_model
        llm_model = str(_ans_model)
    except Exception:
        llm_model = None

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.query is not None:
        # Single-query mode, preserved for manual debugging.
        payload = _make_payload(args, llm_model, qid=None, top_k=args.top_k,
                                query=args.query, retrieve_fn=retrieve)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    # Batch mode. One JSON object per line, "--DONE--" ends the stream.
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        if line == _BATCH_END_SENTINEL:
            break
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err_payload = {
                "ok": False,
                "error": f"Worker could not parse stdin line as JSON: {exc}",
                "raw_line": line[:200],
            }
            print(json.dumps(err_payload, ensure_ascii=False))
            sys.stdout.flush()
            continue

        qid = req.get("qid")
        query = req.get("query", "")
        top_k = int(req.get("top_k", args.top_k))
        if not query:
            err_payload = {
                "ok": False,
                "qid": qid,
                "error": "Missing 'query' field in stdin payload",
            }
            print(json.dumps(err_payload, ensure_ascii=False))
            sys.stdout.flush()
            continue

        payload = _make_payload(args, llm_model, qid=qid, top_k=top_k,
                                query=query, retrieve_fn=retrieve)
        print(json.dumps(payload, ensure_ascii=False))
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Subprocess worker for vector RAG evaluation.

Supports single-query mode (--query) and batch mode (stdin). Batch mode
keeps the embedding model and reranker resident in GPU memory across
queries to avoid per-query reload overhead.

Usage:
    python scripts/eval/vector_worker.py \\
        --system vector-dense --granularity pasal \\
        --embedding-model bge-m3 --query "Apa syarat penyadapan?" \\
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

_RERANKER_CHOICES = ["none", "bge-reranker-v2-m3", "qwen3-reranker-0.6b", "bge-reranker-v2-gemma"]

_BATCH_END_SENTINEL = "--DONE--"


def _run_retrieval(retrieve_fn, query: str, top_k: int,
                   query_vec: list[float] | None = None) -> dict:
    """Call retrieve_vector.retrieve with verbose disabled, return raw result."""
    return retrieve_fn(query, top_k=top_k, verbose=False, query_vec=query_vec)


def _make_payload(args, llm_model, qid: str | None, top_k: int, query: str,
                  retrieve_fn, query_vec: list[float] | None = None) -> dict:
    """Run one retrieval and wrap the response in the standard payload shape."""
    try:
        result = _run_retrieval(retrieve_fn, query, top_k, query_vec=query_vec)
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
                    help="Reranker. none | bge-reranker-v2-m3 | qwen3-reranker-0.6b | bge-reranker-v2-gemma")
    ap.add_argument("--query", default=None,
                    help="Single-query mode. If omitted, worker reads batch from stdin.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--qdrant-path", default=None,
                    help="Path to local Qdrant storage directory")
    args = ap.parse_args()

    # Env vars must be set before importing vector modules.
    model_short = _MODEL_SHORT[args.embedding_model]
    collection = f"law-{args.granularity}-{model_short}"
    os.environ["VECTOR_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["VECTOR_COLLECTION"] = collection
    os.environ["VECTOR_GRANULARITY"] = args.granularity
    os.environ["VECTOR_RERANKER"] = args.reranker
    if args.qdrant_path:
        os.environ["QDRANT_PATH"] = args.qdrant_path

    # Suppress library warnings to keep stdout clean for JSON parsing.
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    # Import once, subsequent calls reuse module-level caches.
    from vector.retrieve_vector import retrieve
    from vector.common import embed_queries

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

    # Batch mode. Collect stdin, batch-encode in one forward pass, then retrieve per query.
    pending: list[dict] = []
    parse_errors: list[dict] = []
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        if line == _BATCH_END_SENTINEL:
            break
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append({
                "ok": False,
                "error": f"Worker could not parse stdin line as JSON: {exc}",
                "raw_line": line[:200],
            })
            continue

        qid = req.get("qid")
        query = req.get("query", "")
        top_k = int(req.get("top_k", args.top_k))
        if not query:
            parse_errors.append({
                "ok": False,
                "qid": qid,
                "error": "Missing 'query' field in stdin payload",
            })
            continue
        pending.append({"qid": qid, "query": query, "top_k": top_k})

    # Emit parse errors first so the orchestrator still records them.
    for err in parse_errors:
        print(json.dumps(err, ensure_ascii=False))
        sys.stdout.flush()

    if not pending:
        return 0

    # One model forward pass for the entire combo. Loads the embedding model
    # on first call (same module-level cache used by per-query path).
    try:
        query_vecs = embed_queries([p["query"] for p in pending])
    except Exception as exc:
        # Hard failure — emit one error per pending qid so the orchestrator
        # marks them all rather than hanging on missing output.
        for p in pending:
            err_payload = {
                "ok": False,
                "qid": p["qid"],
                "system": args.system,
                "granularity": args.granularity,
                "embedding_model": args.embedding_model,
                "reranker": args.reranker,
                "llm_model": llm_model,
                "error": f"Batch embed failed: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(json.dumps(err_payload, ensure_ascii=False))
            sys.stdout.flush()
        return 1

    for p, vec in zip(pending, query_vecs):
        payload = _make_payload(
            args, llm_model,
            qid=p["qid"], top_k=p["top_k"], query=p["query"],
            retrieve_fn=retrieve, query_vec=vec,
        )
        print(json.dumps(payload, ensure_ascii=False))
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

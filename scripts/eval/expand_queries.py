"""Pre-compute LLM-expanded versions of eval queries.

Cache for downstream eval runs that pass --query-expansion <path>. One LLM
call per query, idempotent (skip if cache exists unless --force). Output
file records prompt version, model, and split fingerprint so eval
orchestrators can verify cache integrity.

Usage.
    python scripts/eval/expand_queries.py --split dev
    python scripts/eval/expand_queries.py --split test --force
    python scripts/eval/expand_queries.py --split dev --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core.io import load_testset, load_split_qids, split_fingerprint
from scripts.eval.expand_queries_prompt import PROMPT_VERSION, build_expansion_prompt
from vectorless.llm import call as llm_call
from vectorless.models import RETRIEVAL_MODEL


def expand_one(query: str, model: str) -> tuple[str, dict]:
    """Send one query to the LLM and return (expanded_text, usage_dict).

    Raises ValueError when the LLM returns an empty or missing expansion.
    """
    prompt = build_expansion_prompt(query)
    parsed, usage = llm_call(prompt, model=model, return_usage=True)
    expanded_text = (parsed.get("expanded_query") or "").strip()
    if not expanded_text:
        raise ValueError(f"Empty expansion for query, {query[:80]}...")
    return expanded_text, usage


def main() -> int:
    """Parse args, expand each query in the chosen split, write cache JSON."""
    ap = argparse.ArgumentParser(
        description="Pre-compute LLM-expanded queries for downstream eval runs."
    )
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--testset", default="data/validated_testset.pkl")
    ap.add_argument(
        "--out",
        default=None,
        help="Output path. Default, data/query_expansion/<split>_expanded.json",
    )
    ap.add_argument(
        "--model",
        default=RETRIEVAL_MODEL,
        help=f"LLM model. Default, {RETRIEVAL_MODEL}",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-compute even if the cache file already exists.",
    )
    args = ap.parse_args()

    out_path = Path(
        args.out or f"data/query_expansion/{args.split}_expanded.json"
    )
    if out_path.exists() and not args.force:
        print(f"Cache exists at {out_path}. Use --force to re-compute.")
        return 0

    testset = load_testset(Path(args.testset))
    qids = load_split_qids(args.split)
    selected = [(qid, testset[qid]) for qid in qids if qid in testset]
    print(
        f"Expanding {len(selected)} queries from split={args.split} "
        f"using model={args.model} prompt={PROMPT_VERSION}..."
    )

    expansions: dict[str, dict] = {}
    total_in_tok = 0
    total_out_tok = 0
    for i, (qid, item) in enumerate(selected, 1):
        original = item["query"]
        try:
            expanded, usage = expand_one(original, args.model)
            expansions[qid] = {"original": original, "expanded": expanded}
            total_in_tok += usage.get("input_tokens", 0)
            total_out_tok += usage.get("output_tokens", 0)
        except Exception as exc:
            print(f"  [{i}/{len(selected)}] {qid} FAILED, {exc}")
            # Fallback to original so downstream eval can still run on this qid.
            expansions[qid] = {
                "original": original,
                "expanded": original,
                "error": str(exc),
            }
        if i % 25 == 0:
            print(
                f"  [{i}/{len(selected)}] tokens so far, "
                f"in={total_in_tok} out={total_out_tok}"
            )

    num_failed = sum(1 for v in expansions.values() if "error" in v)
    out = {
        "metadata": {
            "split": args.split,
            "num_queries": len(expansions),
            "num_failed": num_failed,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "split_fingerprint": split_fingerprint(args.split),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_input_tokens": total_in_tok,
            "total_output_tokens": total_out_tok,
        },
        "queries": expansions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(
        f"Saved {out_path} sha256={digest[:16]} "
        f"failed={num_failed}/{len(expansions)} "
        f"tokens_in={total_in_tok} tokens_out={total_out_tok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

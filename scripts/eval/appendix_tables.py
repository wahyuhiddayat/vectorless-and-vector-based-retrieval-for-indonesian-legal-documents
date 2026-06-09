"""Generate the Appendix B full-results LaTeX tables from the Stage 1 eval records.

For each paradigm and query type, emit a table of MAP@10, R@2, R@10, MRR@10, and
H@1 across all three granularities, recomputed from the same development-partition
runs used in Chapter 4. Errored queries count as zero, matching the table
convention. The output file is included by the thesis appendix.

Usage:
    python scripts/eval/appendix_tables.py --out "../../laporan-skripsi/src/99-backMatter/appendix-b-tables.tex"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "data" / "eval_runs"

VL_RUNS = {
    "bm25-flat": "stage1_vectorless/run23_20260520_vectorless_dev_357q_bm25_flat",
    "bm25-tree": "stage1_vectorless/run24_20260520_vectorless_dev_357q_bm25_tree",
    "hybrid-flat": "stage1_vectorless/run25_20260524_vectorless_dev_357q_hybrid_flat_deepseek",
    "hybrid-tree": "stage1_vectorless/run26_20260524_vectorless_dev_357q_hybrid_tree_deepseek",
    "llm-tree": "stage1_vectorless/run27_20260525_vectorless_dev_357q_llm_tree_deepseek",
}
LLM_FLAT_RUNS = {
    "ayat": "stage1_vectorless/run28a_20260526_vectorless_dev_357q_llm_flat_ayat_deepseek",
    "pasal": "stage1_vectorless/run28b_20260526_vectorless_dev_357q_llm_flat_pasal_deepseek",
    "rincian": "stage1_vectorless/run28c_20260527_vectorless_dev_357q_llm_flat_rincian_deepseek",
}
VECTOR_RUN = "stage1_vector/run29_20260526_vector_dev_357q_full"

GRANS = ["pasal", "ayat", "rincian"]
METRICS = ["map@10", "recall@2", "recall@10", "mrr@10", "hit@1"]
VL_METHODS = ["bm25-flat", "bm25-tree", "hybrid-flat", "hybrid-tree", "llm-flat", "llm-tree"]
EMBEDS = [("bge-m3", "BGE-M3"), ("multilingual-e5-large-instruct", "E5"), ("all-nusabert-large-v4", "NusaBERT")]
RERANKERS = [("none", "None"), ("bge-reranker-v2-m3", "BGE v2 M3"), ("qwen3-reranker-0.6b", "Qwen3 0.6B")]
TYPES = [("factual", "Factual"), ("paraphrased", "Paraphrased"), ("multihop", "Multihop")]


def load(rel: str) -> list[dict]:
    with open(RUNS / rel, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def vl_rows(method: str, gran: str) -> list[dict]:
    if method == "llm-flat":
        rel = f"{LLM_FLAT_RUNS[gran]}/records/llm-flat__{gran}.jsonl"
    else:
        rel = f"{VL_RUNS[method]}/records/{method}__{gran}.jsonl"
    return load(rel)


def vec_rows(gran: str, emb: str, rer: str) -> list[dict]:
    return load(f"{VECTOR_RUN}/records/vector-dense__{gran}__{emb}__{rer}.jsonl")


def cell(rows: list[dict], qt: str, metric: str) -> float:
    sub = [r for r in rows if r.get("query_type") == qt]
    return sum((r.get(metric) or 0) for r in sub) / len(sub)


def fmt(vals: list[float]) -> str:
    return " & ".join(f"{v:.4f}" for v in vals)


def vl_table(qt: str, qt_name: str, label: str, out) -> None:
    out.append(r"\begin{table}[H]")
    out.append(r"  \centering")
    out.append(r"  \setstretch{1.0}")
    out.append(r"  \renewcommand{\arraystretch}{1.15}")
    out.append(r"  \footnotesize")
    out.append(r"  \caption{Vectorless results on %s queries across all granularities.}" % qt_name.lower())
    out.append(r"  \label{%s}" % label)
    out.append(r"  \begin{tabular}{@{}ll|ccccc@{}}")
    out.append(r"    \toprule")
    out.append(r"    \textbf{Granularity} & \textbf{Method} & \textbf{MAP@10} & \textbf{R@2} & \textbf{R@10} & \textbf{MRR@10} & \textbf{H@1} \\")
    out.append(r"    \midrule")
    for gi, gran in enumerate(GRANS):
        for mi, method in enumerate(VL_METHODS):
            vals = [cell(vl_rows(method, gran), qt, m) for m in METRICS]
            gname = r"\multirow{6}{*}{%s}" % gran.capitalize() if mi == 0 else ""
            out.append("    %s & %s & %s \\\\" % (gname, method, fmt(vals)))
        if gi < len(GRANS) - 1:
            out.append(r"    \cmidrule(l){2-7}")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out.append("")


def vec_table(qt: str, qt_name: str, label: str, out) -> None:
    out.append(r"\begin{table}[H]")
    out.append(r"  \centering")
    out.append(r"  \setstretch{1.0}")
    out.append(r"  \renewcommand{\arraystretch}{1.15}")
    out.append(r"  \scriptsize")
    out.append(r"  \caption{Vector results on %s queries across all granularities.}" % qt_name.lower())
    out.append(r"  \label{%s}" % label)
    out.append(r"  \begin{tabular}{@{}lll|ccccc@{}}")
    out.append(r"    \toprule")
    out.append(r"    \textbf{Granularity} & \textbf{Embedding} & \textbf{Reranker} & \textbf{MAP@10} & \textbf{R@2} & \textbf{R@10} & \textbf{MRR@10} & \textbf{H@1} \\")
    out.append(r"    \midrule")
    for gi, gran in enumerate(GRANS):
        first = True
        for emb_key, emb_name in EMBEDS:
            for rer_key, rer_name in RERANKERS:
                vals = [cell(vec_rows(gran, emb_key, rer_key), qt, m) for m in METRICS]
                gname = r"\multirow{9}{*}{%s}" % gran.capitalize() if first else ""
                out.append("    %s & %s & %s & %s \\\\" % (gname, emb_name, rer_name, fmt(vals)))
                first = False
        if gi < len(GRANS) - 1:
            out.append(r"    \cmidrule(l){2-8}")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out.append("")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out: list[str] = ["% Generated by scripts/eval/appendix_tables.py. Do not edit by hand."]
    for qt, qt_name in TYPES:
        vl_table(qt, qt_name, f"tab:appendix-vl-{qt}", out)
    for qt, qt_name in TYPES:
        vec_table(qt, qt_name, f"tab:appendix-vec-{qt}", out)
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {len([t for t in TYPES]) * 2} appendix tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

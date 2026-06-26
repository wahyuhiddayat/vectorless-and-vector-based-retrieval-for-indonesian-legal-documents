"""Generate the by-query-type full-results LaTeX tables from the Stage 1 eval records.

For each paradigm and query type, emit a table of MAP@10, R@2, R@10, MRR@10, and
H@1 across all three granularities, recomputed from the same development-partition
runs used in Chapter 4. Errored queries count as zero, matching the table
convention. The output file is included by the thesis appendix.

Usage:
    python scripts/eval/appendix_tables.py --out "../../laporan-skripsi/src/99-backMatter"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RUNS = REPO_ROOT / "data" / "eval_runs"

from scripts.eval.core.significance import compare_paired  # noqa: E402

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
    """Load a records JSONL file relative to the eval runs directory."""
    with open(RUNS / rel, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def vl_rows(method: str, gran: str) -> list[dict]:
    """Load vectorless records for one method and granularity.

    llm-flat runs are stored per granularity, so they resolve to a separate
    run path from the other methods.
    """
    if method == "llm-flat":
        rel = f"{LLM_FLAT_RUNS[gran]}/records/llm-flat__{gran}.jsonl"
    else:
        rel = f"{VL_RUNS[method]}/records/{method}__{gran}.jsonl"
    return load(rel)


def vec_rows(gran: str, emb: str, rer: str) -> list[dict]:
    """Load vector records for one granularity, embedding, and reranker combination."""
    return load(f"{VECTOR_RUN}/records/vector-dense__{gran}__{emb}__{rer}.jsonl")


def cell(rows: list[dict], qt: str, metric: str) -> float:
    """Mean of one metric over the records matching the given query type."""
    sub = [r for r in rows if r.get("query_type") == qt]
    return sum((r.get(metric) or 0) for r in sub) / len(sub)


def fmt(vals: list[float]) -> str:
    """Format a metric row as ampersand-separated four-decimal LaTeX cells."""
    return " & ".join(f"{v:.4f}" for v in vals)


def fmt_int(n: float) -> str:
    """Format a token count with LaTeX thousands separators, e.g. 138{,}055."""
    return f"{int(round(n)):,}".replace(",", "{,}")


def vl_table(qt: str, qt_name: str, label: str, out) -> None:
    """Append a vectorless results table for one query type to out.

    Reports the five effectiveness metrics plus per-query LLM calls and tokens.
    The cost columns are configuration-level and vary little across query types.
    """
    out.append(r"\begin{table}[H]")
    out.append(r"  \centering")
    out.append(r"  \setstretch{1.0}")
    out.append(r"  \renewcommand{\arraystretch}{1.15}")
    out.append(r"  \scriptsize")
    out.append(r"  \caption{Vectorless results on %s queries across all granularities.}" % qt_name.lower())
    out.append(r"  \label{%s}" % label)
    out.append(r"  \begin{tabular}{@{}ll|rrrrr|rr@{}}")
    out.append(r"    \toprule")
    out.append(r"    \textbf{Granularity} & \textbf{Method} & \textbf{MAP@10} & \textbf{R@2} & \textbf{R@10} & \textbf{MRR@10} & \textbf{H@1} & \textbf{LLM calls} & \textbf{LLM tokens} \\")
    out.append(r"    \midrule")
    for gi, gran in enumerate(GRANS):
        for mi, method in enumerate(VL_METHODS):
            rows = vl_rows(method, gran)
            vals = [cell(rows, qt, m) for m in METRICS]
            calls = cell(rows, qt, "llm_calls")
            tokens = cell(rows, qt, "total_tokens")
            gname = r"\multirow{6}{*}{%s}" % gran.capitalize() if mi == 0 else ""
            out.append("    %s & %s & %s & %.1f & %s \\\\" % (gname, method, fmt(vals), calls, fmt_int(tokens)))
        if gi < len(GRANS) - 1:
            out.append(r"    \midrule")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out.append("")


def vec_table(qt: str, qt_name: str, label: str, out) -> None:
    """Append a vector results table for one query type to out."""
    out.append(r"\begin{table}[H]")
    out.append(r"  \centering")
    out.append(r"  \setstretch{1.0}")
    out.append(r"  \renewcommand{\arraystretch}{1.15}")
    out.append(r"  \scriptsize")
    out.append(r"  \caption{Vector results on %s queries across all granularities.}" % qt_name.lower())
    out.append(r"  \label{%s}" % label)
    out.append(r"  \begin{tabular}{@{}lll|rrrrr@{}}")
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
            out.append(r"    \midrule")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out.append("")


TEST_RUNS = [
    ("Vectorless", "stage3_test/rq4_test_hybrid_tree/records/hybrid-tree__pasal.jsonl"),
    ("Vector", "stage3_test/rq4_test_v2m3_qe/records/vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl"),
]


def test_table(out) -> None:
    """Append the Stage 3 test-partition results broken down by query type.

    Both improved configurations are reported per query type with the five
    effectiveness metrics and mean latency. The language-model token cost is
    configuration-level and reported in the cost tables of Chapter 4.
    """
    out.append(r"\begin{table}[H]")
    out.append(r"  \centering")
    out.append(r"  \setstretch{1.0}")
    out.append(r"  \renewcommand{\arraystretch}{1.15}")
    out.append(r"  \footnotesize")
    out.append(r"  \caption{Test-partition results by query type for the improved vectorless and vector configurations.}")
    out.append(r"  \label{tab:appendix-test-bytype}")
    out.append(r"  \begin{tabular}{@{}ll|rrrrr|r@{}}")
    out.append(r"    \toprule")
    out.append(r"    \textbf{Query type} & \textbf{Configuration} & \textbf{MAP@10} & \textbf{R@2} & \textbf{R@10} & \textbf{MRR@10} & \textbf{H@1} & \textbf{Latency (s)} \\")
    out.append(r"    \midrule")
    for ti, (qt, qt_name) in enumerate(TYPES):
        for ci, (cfg_name, path) in enumerate(TEST_RUNS):
            rows = load(path)
            vals = [cell(rows, qt, m) for m in METRICS]
            lat = cell(rows, qt, "elapsed_s")
            tname = r"\multirow{2}{*}{%s}" % qt_name if ci == 0 else ""
            out.append("    %s & %s & %s & %.2f \\\\" % (tname, cfg_name, fmt(vals), lat))
        if ti < len(TYPES) - 1:
            out.append(r"    \midrule")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out.append("")


VL_TUNE = "stage2_vectorless/tune_vectorless_log.json"
VEC_TUNE = "stage2_vector/tune_vector_log.json"
INDEXING = REPO_ROOT / "data" / "indexing_cost_summary.json"


def load_json(rel: str) -> dict:
    """Load a JSON log relative to the eval runs directory."""
    with open(RUNS / rel, encoding="utf-8") as f:
        return json.load(f)


def last_step(log: list, name: str) -> dict:
    """Return the last decision-log entry for a step.

    The tuning sweeps were re-run, so the final entry holds the clean values the
    chapter reports, after transient language-model errors were resolved.
    """
    return [e for e in log if e.get("step") == name][-1]


def _stage2_table(caption: str, label: str, groups, out) -> None:
    """Append a Stage 2 tuning table. R@2 was not logged during the search."""
    out.append(r"\begin{table}[H]")
    out.append(r"  \centering")
    out.append(r"  \setstretch{1.0}")
    out.append(r"  \renewcommand{\arraystretch}{1.15}")
    out.append(r"  \footnotesize")
    out.append(r"  \caption{%s}" % caption)
    out.append(r"  \label{%s}" % label)
    out.append(r"  \begin{tabular}{@{}ll|rrrr@{}}")
    out.append(r"    \toprule")
    out.append(r"    \textbf{Step} & \textbf{Value} & \textbf{MAP@10} & \textbf{R@10} & \textbf{MRR@10} & \textbf{H@1} \\")
    out.append(r"    \midrule")
    for gi, (step, vals) in enumerate(groups):
        for vi, (vlabel, d) in enumerate(vals):
            sname = r"\multirow{%d}{*}{%s}" % (len(vals), step) if vi == 0 else ""
            out.append("    %s & %s & %.4f & %.4f & %.4f & %.4f \\\\" % (
                sname, vlabel, d["map@10"], d["recall@10"], d["mrr@10"], d["hit@1"]))
        if gi < len(groups) - 1:
            out.append(r"    \midrule")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out.append("")


def stage2_vl_table(out) -> None:
    """Append the vectorless Stage 2 tuning table with full effectiveness metrics."""
    log = load_json(VL_TUNE)["decision_log"]
    model = last_step(log, "model_upgrade")
    qe = last_step(log, "query_expansion")
    groups = [
        ("Candidate count", [(str(r["value"]), r) for r in last_step(log, "sweep_bm25_top_k")["results"]]),
        ("Document-pick count", [(str(r["value"]), r) for r in last_step(log, "sweep_doc_pick_top_k")["results"]]),
        ("Retrieval model", [("deepseek-v4-flash", model["flash"]), ("deepseek-v4-pro", model["v4_pro"])]),
        ("Query expansion", [("original query", qe["without_qe"]), ("expanded query", qe["with_qe"])]),
    ]
    _stage2_table("Sequential improvement of the best vectorless configuration, full effectiveness metrics.",
                  "tab:appendix-vl-tuning", groups, out)


def stage2_vec_table(out) -> None:
    """Append the vector Stage 2 tuning table with full effectiveness metrics."""
    log = load_json(VEC_TUNE)["decision_log"]
    rer = last_step(log, "reranker_upgrade")
    qe = last_step(log, "query_expansion")
    groups = [
        ("First-stage depth", [(str(r["value"]), r) for r in last_step(log, "sweep_top_n")["results"]]),
        (r"HNSW \texttt{ef}", [(str(r["value"]), r) for r in last_step(log, "sweep_ef_search")["results"]]),
        ("Reranker", [("BGE v2 M3", rer["v2_m3_tuned"]), ("BGE v2 Gemma", rer["v2_gemma"])]),
        ("Query expansion", [("original query", qe["without_qe"]), ("expanded query", qe["with_qe"])]),
    ]
    _stage2_table("Sequential improvement of the best vector configuration, full effectiveness metrics.",
                  "tab:appendix-vec-tuning", groups, out)


def effect_size_table(out) -> None:
    """Append the test-partition effect-size table, vectorless minus vector.

    Recomputed directly from the two improved configurations' per-query records
    with the two-sided paired procedure of Chapter 3, so the values match the
    records exactly. Cohen's d_z is the paired effect size, and the final row
    restricts R@2 to the two-anchor multihop subset.
    """
    f = {r["query_id"]: r for r in load(TEST_RUNS[0][1])}
    g = {r["query_id"]: r for r in load(TEST_RUNS[1][1])}
    shared = sorted(set(f) & set(g))
    label_map = {"map@10": "MAP@10", "recall@2": "R@2", "recall@10": "R@10", "mrr@10": "MRR@10", "hit@1": "H@1"}
    out.append(r"\begin{table}[H]")
    out.append(r"  \centering")
    out.append(r"  \setstretch{1.0}")
    out.append(r"  \renewcommand{\arraystretch}{1.15}")
    out.append(r"  \footnotesize")
    out.append(r"  \caption{Effect sizes for the test-partition comparison, vectorless minus vector.}")
    out.append(r"  \label{tab:appendix-effect-size}")
    out.append(r"  \begin{tabular}{@{}l|rrrc@{}}")
    out.append(r"    \toprule")
    out.append(r"    \textbf{Metric} & \textbf{Difference} & \textbf{p-value} & \textbf{Cohen's $d_z$} & \textbf{Magnitude} \\")
    out.append(r"    \midrule")
    for m in METRICS:
        a = [float(f[q].get(m) or 0) for q in shared]
        b = [float(g[q].get(m) or 0) for q in shared]
        r = compare_paired(a, b, alternative="two-sided", B=10000, seed=42)
        out.append("    %s & $%+.4f$ & %.4f & %.2f & %s \\\\" % (
            label_map[m], r["mean_diff"], r["paired_randomization"]["p_value"],
            r["cohens_d"]["d"], r["cohens_d"]["label"].capitalize()))
    mh = [q for q in shared if f[q].get("query_type") == "multihop" and (f[q].get("num_relevant") or 0) == 2]
    a = [float(f[q].get("recall@2") or 0) for q in mh]
    b = [float(g[q].get("recall@2") or 0) for q in mh]
    r = compare_paired(a, b, alternative="two-sided", B=10000, seed=42)
    out.append("    R@2 (multihop) & $%+.4f$ & %.4f & %.2f & %s \\\\" % (
        r["mean_diff"], r["paired_randomization"]["p_value"], r["cohens_d"]["d"], r["cohens_d"]["label"].capitalize()))
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out.append("")


def indexing_table(out) -> None:
    """Append the indexing token-cost table by processing stage and granularity."""
    with open(INDEXING, encoding="utf-8") as f:
        ic = json.load(f)
    parse = ic["pasal"]["per_stage"]["parse"]["tokens"]
    repair = ic["pasal"]["per_stage"]["ocr_clean"]["tokens"]
    summ = {g: ic[g]["per_stage"]["summary"]["tokens"] for g in ("pasal", "ayat", "rincian")}
    shared = parse + repair
    vl_specific = sum(summ.values())
    total = shared + vl_specific
    out.append(r"\begin{table}[H]")
    out.append(r"  \centering")
    out.append(r"  \setstretch{1.0}")
    out.append(r"  \renewcommand{\arraystretch}{1.15}")
    out.append(r"  \footnotesize")
    out.append(r"  \caption{Indexing token cost by stage and granularity.}")
    out.append(r"  \label{tab:appendix-indexing}")
    out.append(r"  \begin{tabular*}{0.66\textwidth}{@{\extracolsep{\fill}}ll|r@{}}")
    out.append(r"    \toprule")
    out.append(r"    \textbf{Cost} & \textbf{Stage} & \textbf{Tokens} \\")
    out.append(r"    \midrule")
    out.append("    \\multirow{3}{*}{Shared} & Structural parsing & %s \\\\" % fmt_int(parse))
    out.append("     & Text repair & %s \\\\" % fmt_int(repair))
    out.append("     & \\textbf{Subtotal} & \\textbf{%s} \\\\" % fmt_int(shared))
    out.append(r"    \midrule")
    out.append("    \\multirow{4}{*}{Vectorless} & Summary, \\textit{pasal} & %s \\\\" % fmt_int(summ["pasal"]))
    out.append("     & Summary, \\textit{ayat} & %s \\\\" % fmt_int(summ["ayat"]))
    out.append("     & Summary, \\textit{rincian} & %s \\\\" % fmt_int(summ["rincian"]))
    out.append("     & \\textbf{Subtotal} & \\textbf{%s} \\\\" % fmt_int(vl_specific))
    out.append(r"    \midrule")
    out.append("    \\multicolumn{2}{@{}l|}{\\textbf{Total}} & \\textbf{%s} \\\\" % fmt_int(total))
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular*}")
    out.append(r"\end{table}")
    out.append("")


HEADER = "% Generated by scripts/eval/appendix_tables.py. Do not edit by hand."


def main() -> int:
    """Render the appendix tables and write them to the --out directory.

    Stage 1 per-type tables go to lampiran-bytype-tables.tex (Appendix 3)
    and the Stage 3 test table goes to lampiran-test-tables.tex (Appendix 5).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="thesis src/99-backMatter directory")
    args = ap.parse_args()
    out_dir = Path(args.out)

    stage1: list[str] = [HEADER]
    for qt, qt_name in TYPES:
        vl_table(qt, qt_name, f"tab:appendix-vl-{qt}", stage1)
    for qt, qt_name in TYPES:
        vec_table(qt, qt_name, f"tab:appendix-vec-{qt}", stage1)
    (out_dir / "lampiran-bytype-tables.tex").write_text("\n".join(stage1), encoding="utf-8")

    stage2: list[str] = [HEADER]
    stage2_vl_table(stage2)
    stage2_vec_table(stage2)
    (out_dir / "lampiran-stage2-tables.tex").write_text("\n".join(stage2), encoding="utf-8")

    test: list[str] = [HEADER]
    test_table(test)
    effect_size_table(test)
    (out_dir / "lampiran-test-tables.tex").write_text("\n".join(test), encoding="utf-8")

    indexing: list[str] = [HEADER]
    indexing_table(indexing)
    (out_dir / "lampiran-indexing-tables.tex").write_text("\n".join(indexing), encoding="utf-8")

    print(f"Wrote Stage 1, Stage 2, test, and indexing appendix tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

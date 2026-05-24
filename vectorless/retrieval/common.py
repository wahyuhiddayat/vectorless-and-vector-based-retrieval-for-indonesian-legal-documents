"""Shared helpers for the vectorless retrieval pipelines."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from ..ids import doc_category

load_dotenv()

DATA_INDEX = Path(os.environ.get("DATA_INDEX", "data/index_pasal"))
LOG_DIR = Path("data/retrieval_logs")

DOC_PICK_TOP_K = 3
"""Number of documents selected at stage 1 for tree-based variants."""


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens.

    No stopword removal or stemming. Single-letter tokens are dropped
    but single digits are kept to preserve legal citations like
    Pasal 3 or ayat 1.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 or t.isdigit()]


def load_catalog() -> list[dict]:
    """Load the document catalog at the active DATA_INDEX."""
    with open(DATA_INDEX / "catalog.json", encoding="utf-8") as f:
        return json.load(f)


def doc_corpus_string(doc_meta: dict) -> str:
    """Build the BM25 doc-level corpus string for one catalog entry.

    Concatenates metadata fields and the aggregated doc_summary_text
    when available.

    Args:
        doc_meta: One entry from catalog.json.

    Returns:
        Space-joined corpus string.
    """
    parts = [
        doc_meta.get("judul") or "",
        doc_meta.get("bidang") or "",
        doc_meta.get("subjek") or "",
        doc_meta.get("materi_pokok") or "",
    ]
    summary_text = doc_meta.get("doc_summary_text") or ""
    if summary_text:
        parts.append(summary_text)
    return " ".join(parts)


def catalog_for_llm_prompt(catalog: list[dict], summary_cap: int = 600) -> list[dict]:
    """Return a slim catalog projection for LLM doc-pick prompts.

    Keeps metadata fields intact and truncates doc_summary_text to
    summary_cap characters per document.

    Args:
        catalog: Full catalog list from catalog.json.
        summary_cap: Maximum characters for the summary field per doc.

    Returns:
        List of slim catalog entry dicts.
    """
    slim = []
    for doc in catalog:
        entry = {
            "doc_id": doc.get("doc_id"),
            "judul": doc.get("judul"),
            "bidang": doc.get("bidang"),
            "subjek": doc.get("subjek"),
            "materi_pokok": doc.get("materi_pokok"),
        }
        summary_text = doc.get("doc_summary_text") or ""
        if summary_text:
            entry["doc_summary_text"] = summary_text[:summary_cap]
        slim.append(entry)
    return slim


def load_doc(doc_id: str) -> dict:
    """Load one indexed document."""
    path = DATA_INDEX / doc_category(doc_id) / f"{doc_id}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect_leaf_nodes(nodes: list[dict]) -> list[dict]:
    """Recursively collect all leaf nodes that carry text."""
    leaves = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            leaves.extend(_collect_leaf_nodes(node["nodes"]))
        elif node.get("text"):
            leaves.append(node)
    return leaves


def load_all_leaf_nodes() -> list[dict]:
    """Flat list of every leaf node across all docs in the active index."""
    catalog = load_catalog()
    all_leaves = []
    for doc_meta in catalog:
        doc_id = doc_meta["doc_id"]
        path = DATA_INDEX / doc_category(doc_id) / f"{doc_id}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        for node in _collect_leaf_nodes(doc.get("structure", [])):
            all_leaves.append({
                "doc_id": doc_id,
                "doc_title": doc_meta["judul"],
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "navigation_path": node.get("navigation_path", ""),
                "text": node.get("text", ""),
                "penjelasan": node.get("penjelasan"),
                "summary": node.get("summary", ""),
            })
    return all_leaves


def find_node(nodes: list[dict], node_id: str) -> dict | None:
    """Locate one node in the tree by node_id."""
    for node in nodes:
        if node.get("node_id") == node_id:
            return node
        if "nodes" in node:
            found = find_node(node["nodes"], node_id)
            if found:
                return found
    return None


def extract_nodes(doc: dict, node_ids: list[str]) -> list[dict]:
    """Resolve node_ids in doc.structure to compact dicts with text + penjelasan."""
    out = []
    for nid in node_ids:
        node = find_node(doc["structure"], nid)
        if node:
            out.append({
                "node_id": nid,
                "title": node.get("title", ""),
                "navigation_path": node.get("navigation_path", ""),
                "text": node.get("text", ""),
                "penjelasan": node.get("penjelasan"),
            })
    return out


def agentic_finalize(submitted_ids: list[str],
                     top_k: int) -> tuple[list[str], list[str]]:
    """Deduplicate and truncate the agent's submitted ids to top_k.

    The output may be shorter than top_k when the agent submits fewer
    candidates.

    Args:
        submitted_ids: Ordered ids from the agent, most relevant first.
        top_k: Maximum output length.

    Returns:
        Tuple of (final_ranking, source_labels). source_labels is a
        parallel list labelling each slot as "agent_submit".
    """
    seen: set[str] = set()
    final: list[str] = []
    labels: list[str] = []
    for nid in submitted_ids:
        if nid and nid not in seen:
            final.append(nid)
            labels.append("agent_submit")
            seen.add(nid)
            if len(final) >= top_k:
                break
    return final, labels


def validate_llm_ranking(llm_ranking: list[str], candidates: list[dict]) -> list[str]:
    """Validate and complete an LLM-generated ranking.

    Drops hallucinated and duplicate IDs, then appends missing candidate
    IDs in their original order so the output covers all candidates.

    Args:
        llm_ranking: Node IDs from the LLM in descending relevance order.
        candidates: Candidate dicts with node_id, in first-stage order.

    Returns:
        Complete list of node_ids covering all candidates, LLM ranking
        first then missing candidates in original order.
    """
    valid_order = [c["node_id"] for c in candidates]
    valid_set = set(valid_order)
    seen: set[str] = set()
    cleaned: list[str] = []
    for nid in llm_ranking:
        if nid in valid_set and nid not in seen:
            cleaned.append(nid)
            seen.add(nid)
    for nid in valid_order:
        if nid not in seen:
            cleaned.append(nid)
            seen.add(nid)
    return cleaned


def save_log(result: dict) -> None:
    """Persist a retrieval result under data/retrieval_logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

"""LLM tree retrieval for Indonesian legal QA.

The LLM acts as an agent that navigates the document tree using
expand(), read(), and submit() tools. The full tree outline (titles
and summaries) is shown upfront. Documents are selected first via
LLM doc search, then the agent explores the picked documents to
find relevant leaf nodes. Inspired by PageIndex (Vectify AI, 2024).

Usage:
    python -m vectorless.retrieval.llm.tree "Apa syarat penyadapan?"
"""

from __future__ import annotations

import argparse
import json
import os
import time

from rank_bm25 import BM25Okapi

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    load_catalog, load_doc, find_node, save_log,
    tree_finalize, tokenize,
    doc_corpus_string, catalog_for_llm_prompt,
    DOC_PICK_TOP_K,
)


def _bm25_doc_search(query: str, catalog: list[dict], top_k: int = DOC_PICK_TOP_K) -> list[str]:
    """Rank catalog entries with BM25 and return top doc_ids above zero.

    Args:
        query: Legal question in Indonesian.
        catalog: List of document metadata dicts.
        top_k: Number of top documents to return.

    Returns:
        List of doc_id strings.
    """
    corpus = [tokenize(doc_corpus_string(doc)) for doc in catalog]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [catalog[idx]["doc_id"] for idx, score in ranked[:top_k] if score > 0]


def doc_search(query: str, catalog: list[dict], top_k: int = DOC_PICK_TOP_K,
               verbose: bool = True) -> dict:
    """Pick up to top_k relevant documents via LLM selection.

    The catalog is placed in the system message to enable prefix caching
    across queries.

    Args:
        query: Legal question in Indonesian.
        catalog: List of document metadata dicts.
        top_k: Maximum number of documents to select.
        verbose: Print selected documents to stdout.

    Returns:
        Dict with doc_ids, llm_doc_ids, and thinking.
    """
    slim_catalog = catalog_for_llm_prompt(catalog)
    docs_text = json.dumps(slim_catalog, ensure_ascii=False, indent=2)

    system = f"""\
Kamu diberi daftar Undang-Undang Indonesia beserta metadata dan ringkasan isi-nya.
Pilih UU yang paling mungkin mengandung jawaban untuk pertanyaan hukum yang diberikan user.

Daftar UU:
{docs_text}

Aturan:
- Pilih 1 sampai {top_k} UU yang paling mungkin mengandung jawaban (recall-oriented).
- Lebih baik over-include sedikit daripada miss UU yang relevan.
- Pertimbangkan judul, bidang, subjek, materi_pokok, dan doc_summary_text (kalau tersedia).
- Hanya kembalikan doc_ids kosong [] jika benar-benar tidak ada satupun yang dekat dengan topik pertanyaan.
- Kembalikan HANYA JSON, tanpa teks lain."""

    prompt = f"""\
Pertanyaan: {query}

Balas dalam format JSON:
{{
  "thinking": "<penalaran singkat mengapa UU tersebut kemungkinan relevan>",
  "doc_ids": ["doc_id_1", "doc_id_2", "doc_id_3"]
}}
"""

    llm_result = llm_call(prompt, system=system)

    valid_ids = {d["doc_id"] for d in catalog}
    llm_doc_ids = [doc_id for doc_id in llm_result.get("doc_ids", []) if doc_id in valid_ids]

    result = {
        "doc_ids": llm_doc_ids[:top_k],
        "llm_doc_ids": llm_doc_ids,
        "thinking": llm_result.get("thinking", ""),
    }

    if verbose:
        print(f"\n[Doc Search] LLM picks: {llm_doc_ids[:top_k]}")
        if llm_result.get("thinking"):
            print(f"  Reasoning: {llm_result['thinking'][:200]}")

    return result


MAX_ACTIONS = int(os.environ.get("LLM_TREE_MAX_ACTIONS", "30"))
MAX_READS = int(os.environ.get("LLM_TREE_MAX_READS", "18"))
OBSERVATION_RENDER_CAP = 1800
SCRATCHPAD_RECENT_FULL = 2
DEFAULT_TOP_K = 10
VALID_ACTIONS = ("expand", "read", "submit")


def _node_view(node: dict) -> dict:
    """Render one node and its descendants, titles and summaries only."""
    out = {
        "node_id": node.get("node_id", ""),
        "title": node.get("title", ""),
    }
    if node.get("summary"):
        out["summary"] = node["summary"]
    children = node.get("nodes") or []
    if children:
        out["nodes"] = [_node_view(c) for c in children]
    return out


def _resolve_doc(picked_docs: dict[str, dict], doc_id: str) -> dict | None:
    """Return the picked doc by id, or None if not in scope."""
    if not doc_id:
        return None
    return picked_docs.get(doc_id)


def _tool_inspect_doc(picked_docs: dict[str, dict], doc_id: str) -> dict:
    """Full recursive tree structure of one picked document, no leaf text."""
    doc = _resolve_doc(picked_docs, doc_id)
    if doc is None:
        return {
            "error": f"doc_id '{doc_id}' is not in the picked set; valid: {list(picked_docs.keys())}",
        }
    return {
        "doc_id": doc.get("doc_id", ""),
        "judul": doc.get("judul", ""),
        "structure": [_node_view(n) for n in doc.get("structure", [])],
    }


def _render_tree_outline(nodes: list[dict], depth: int = 0) -> str:
    """Render one document's tree as an indented outline."""
    lines: list[str] = []
    indent = "  " * depth
    for n in nodes:
        node_id = n.get("node_id", "")
        title = (n.get("title") or "").strip()
        summary = (n.get("summary") or "").strip()
        head = f"{indent}- [{node_id}] {title}" if title else f"{indent}- [{node_id}]"
        if summary:
            head += f" :: {summary}"
        lines.append(head)
        children = n.get("nodes") or []
        if children:
            lines.append(_render_tree_outline(children, depth + 1))
    return "\n".join(lines)


def _render_multidoc_outline(picked_docs: dict[str, dict]) -> str:
    """Render outlines for all picked docs, concatenated with doc headers.

    Args:
        picked_docs: Mapping of doc_id to loaded document dict.

    Returns:
        Concatenated outline string with doc headers.
    """
    parts: list[str] = []
    for doc_id, doc in picked_docs.items():
        judul = doc.get("judul", "")
        parts.append(f"=== DOC: {doc_id} -- {judul} ===")
        parts.append(_render_tree_outline(doc.get("structure", [])))
        parts.append("")
    return "\n".join(parts).rstrip()


def _tool_expand(picked_docs: dict[str, dict], doc_id: str, node_id: str) -> dict:
    """Children of one internal node in the named picked doc."""
    doc = _resolve_doc(picked_docs, doc_id)
    if doc is None:
        return {
            "error": f"doc_id '{doc_id}' is not in the picked set; valid: {list(picked_docs.keys())}",
        }
    node = find_node(doc.get("structure", []), node_id)
    if node is None:
        return {"error": f"node_id '{node_id}' not found in doc '{doc_id}'"}
    children = node.get("nodes") or []
    if not children:
        return {
            "error": f"node_id '{node_id}' has no children, it is already a leaf",
            "doc_id": doc_id,
            "title": node.get("title", ""),
        }
    return {
        "doc_id": doc_id,
        "parent": {"node_id": node_id, "title": node.get("title", ""),
                   "navigation_path": node.get("navigation_path", "")},
        "children": [_node_view(c) for c in children],
    }


def _tool_read(picked_docs: dict[str, dict], doc_id: str, node_id: str) -> dict:
    """Full text of one leaf node in the named picked doc, plus penjelasan."""
    doc = _resolve_doc(picked_docs, doc_id)
    if doc is None:
        return {
            "error": f"doc_id '{doc_id}' is not in the picked set; valid: {list(picked_docs.keys())}",
        }
    node = find_node(doc.get("structure", []), node_id)
    if node is None:
        return {"error": f"node_id '{node_id}' not found in doc '{doc_id}'"}
    out = {
        "doc_id": doc_id,
        "node_id": node_id,
        "title": node.get("title", ""),
        "navigation_path": node.get("navigation_path", ""),
        "text": node.get("text") or "",
    }
    penjelasan = node.get("penjelasan")
    if penjelasan and penjelasan != "Cukup jelas.":
        out["penjelasan"] = penjelasan
    return out


def _parse_node_ref(ref, default_doc_id: str | None = None) -> tuple[str | None, str | None]:
    """Parse a node reference into (doc_id, node_id).

    Accepts a dict with doc_id/node_id fields, a "doc_id/node_id" string,
    or a bare node_id (resolved with default_doc_id).

    Args:
        ref: Dict, compound string, or bare node_id.
        default_doc_id: Fallback doc_id for bare node_ids.

    Returns:
        Tuple of (doc_id, node_id). Either may be None if unresolvable.
    """
    if isinstance(ref, dict):
        return ref.get("doc_id") or default_doc_id, ref.get("node_id")
    if isinstance(ref, str):
        if "/" in ref:
            doc_id, node_id = ref.split("/", 1)
            return doc_id, node_id
        return default_doc_id, ref
    return None, None


def _render_scratchpad(scratchpad: list[dict]) -> str:
    """Render the scratchpad for the next prompt, recent steps full, older truncated."""
    lines = []
    n = len(scratchpad)
    for i, entry in enumerate(scratchpad):
        recent = (n - i) <= SCRATCHPAD_RECENT_FULL
        cap = OBSERVATION_RENDER_CAP if recent else 600
        action = entry.get("action", "?")
        args_str = json.dumps(entry.get("args") or {}, ensure_ascii=False)[:200]
        obs = entry.get("observation")
        obs_str = json.dumps(obs, ensure_ascii=False)
        if len(obs_str) > cap:
            obs_str = obs_str[:cap] + "...[truncated]"
        lines.append(f"[{entry.get('step', i)}] {action}({args_str}) -> {obs_str}")
        if entry.get("thinking") and recent:
            lines.append(f"     thinking: {entry['thinking'][:300]}")
    return "\n".join(lines) if lines else "(belum ada tindakan)"


def _anti_loop_hints(scratchpad: list[dict], actions_used: int, reads_used: int,
                     max_actions: int, max_reads: int) -> list[str]:
    """Return steering hints when the agent may be stalling.

    Detects repeated identical actions, approaching action budget
    exhaustion, and near-depleted read budget.

    Args:
        scratchpad: List of past action/observation entries.
        actions_used: Number of actions taken so far.
        reads_used: Number of read calls used so far.
        max_actions: Total action budget.
        max_reads: Total read budget.

    Returns:
        List of hint strings. Empty if no issues detected.
    """
    hints: list[str] = []
    submitted = any(e.get("action") == "submit"
                    and not (e.get("observation") or {}).get("error")
                    for e in scratchpad)

    recent = [e for e in scratchpad[-3:]
              if e.get("action") in ("expand", "read", "inspect_doc")]
    if len(recent) >= 3:
        sig = (recent[0].get("action"),
               json.dumps(recent[0].get("args") or {}, sort_keys=True))
        if all(
            (e.get("action"), json.dumps(e.get("args") or {}, sort_keys=True)) == sig
            for e in recent
        ):
            hints.append(
                "Kamu sudah melakukan action yang sama 3 kali berturut-turut. "
                "Ganti strategi: pilih node lain, atau submit ranking final sekarang."
            )

    if not submitted and actions_used >= int(max_actions * 0.6):
        remaining = max_actions - actions_used
        hints.append(
            f"Sudah {actions_used}/{max_actions} actions tanpa submit, sisa {remaining}. "
            "Submit ranking final sekarang berdasarkan info yang ada."
        )

    if not submitted and reads_used >= int(max_reads * 0.7):
        remaining = max_reads - reads_used
        hints.append(
            f"Anggaran read tersisa {remaining}/{max_reads}. "
            "Stop reading, mulai pertimbangkan submit ranking."
        )

    return hints


def _build_prompt_parts(query: str, scratchpad: list[dict],
                        actions_left: int, reads_left: int,
                        picked_docs: dict[str, dict],
                        multidoc_outline: str,
                        steering_hints: list[str] | None = None) -> tuple[str, str]:
    """Build the system and user messages for the next agent step.

    Static context (instructions, query, outlines, tools) goes in the
    system message. Dynamic content (budget, hints, scratchpad) goes in
    the user message.

    Args:
        query: Legal question in Indonesian.
        scratchpad: List of past action/observation entries.
        actions_left: Remaining action budget.
        reads_left: Remaining read budget.
        picked_docs: Mapping of doc_id to loaded document dict.
        multidoc_outline: Pre-rendered outline of all picked docs.
        steering_hints: Optional anti-loop hints to inject.

    Returns:
        Tuple of (system message, user message).
    """
    doc_headers = "\n".join(
        f"  - {did} -- {doc.get('judul', '')}"
        for did, doc in picked_docs.items()
    )
    hints_block = ""
    if steering_hints:
        hints_text = "\n".join(f"  - {h}" for h in steering_hints)
        hints_block = f"── PERINGATAN ──\n{hints_text}\n\n"
    system = (
        "Kamu adalah agen retrieval dokumen hukum.\n"
        "Tugas: temukan node paling relevan (pasal/ayat/rincian) untuk menjawab pertanyaan.\n"
        "Kamu diberi beberapa UU kandidat, dan harus eksplor satu atau beberapa untuk\n"
        "menemukan jawaban paling tepat. Setiap tool wajib menyertakan doc_id.\n\n"
        f"Pertanyaan: {query}\n\n"
        f"── KANDIDAT UU ({len(picked_docs)} doc) ──\n"
        f"{doc_headers}\n\n"
        "── STRUKTUR PER DOKUMEN (outline lengkap) ──\n"
        f"{multidoc_outline}\n\n"
        "Outline di atas sudah berisi semua node dengan title + ringkasan.\n"
        "Kamu TIDAK perlu request outline lagi -- langsung lompat ke expand/read/submit.\n\n"
        "── TOOLS ──\n"
        "- expand(doc_id, node_id)        lihat anak-anak node di doc itu (jika perlu detail tambahan)\n"
        "- read(doc_id, node_id)          baca teks LENGKAP satu node (untuk verifikasi)\n"
        "- submit(node_ids)               finalisasi ranking, format `doc_id/node_id`,\n"
        "                                  terurut paling relevan dulu (max 10)\n\n"
        "── ALUR KERJA ──\n"
        "1. Scan outline lintas doc -> identifikasi node tematis relevan (boleh > 1 doc)\n"
        "2. read(doc_id, node_id) -> verifikasi teks lengkap kandidat utama\n"
        "3. expand(doc_id, node_id) -> jika butuh lihat sub-node\n"
        "4. submit([\"doc_id/node_id\", ...]) -> kirim ranking final ASAP\n\n"
        "── CATATAN PENTING SUBMIT ──\n"
        "Output retrieval kamu HANYALAH apa yang kamu submit, tidak ada padding otomatis.\n"
        "Submit beberapa kandidat (sampai 10) jika ada lebih dari satu node yang relevan\n"
        "atau jika kamu tidak yakin mana yang paling tepat. Urutkan paling yakin dulu.\n"
        "Tetap selektif, jangan submit node yang jelas tidak relevan hanya untuk mengisi slot.\n\n"
        "── FORMAT ──\n"
        "{\n"
        '  "thinking": "...",\n'
        '  "action": "expand" | "read" | "submit",\n'
        '  "args": { ... }\n'
        "}\n"
    )
    prompt = (
        f"── SISA ANGGARAN ──\n"
        f"Action: {actions_left} | Read: {reads_left}\n"
        "Gunakan read() secara selektif — hanya untuk verifikasi.\n\n"
        f"{hints_block}"
        "── RIWAYAT TINDAKAN ──\n"
        f"{_render_scratchpad(scratchpad)}\n\n"
        "Apa langkah selanjutnya? Kembalikan HANYA JSON.\n"
    )
    return system, prompt


def _coerce_id(value):
    """Unwrap a schema-loose id that a model may return as a single-item list.

    Some models return node_id or doc_id as a one-element list rather than a
    string. Unwrapping it keeps read and expand from crashing or failing on a
    valid intent, matching the leniency already applied to submit.
    """
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _siblings_hint_multidoc(picked_docs: dict[str, dict], doc_id: str,
                            missing_id: str, limit: int = 5) -> list[str]:
    """Best-effort list of node_ids near the missing id within one doc."""
    doc = picked_docs.get(doc_id)
    if doc is None:
        return []
    all_ids: list[str] = []

    def _walk(nodes):
        """Collect every node_id in the subtree into all_ids, depth-first."""
        for n in nodes:
            nid = n.get("node_id")
            if nid:
                all_ids.append(nid)
            if n.get("nodes"):
                _walk(n["nodes"])

    _walk(doc.get("structure", []))
    prefix = missing_id.split("_")[0] if isinstance(missing_id, str) and missing_id else ""
    near = [nid for nid in all_ids if prefix and nid.startswith(prefix)]
    return near[:limit] if near else all_ids[:limit]


def retrieve(query: str,
             max_actions: int = MAX_ACTIONS, max_reads: int = MAX_READS,
             top_k: int = DEFAULT_TOP_K, top_k_docs: int = DOC_PICK_TOP_K,
             verbose: bool = True) -> dict:
    """Run the full LLM tree retrieval pipeline.

    Selects documents via LLM doc search, then runs an agent loop
    where the LLM explores document trees using expand, read, and
    submit tools. The agent's submitted refs are deduplicated and
    truncated to top_k.

    Args:
        query: Legal question in Indonesian.
        max_actions: Hard cap on total agent steps.
        max_reads: Hard cap on read tool calls.
        top_k: Maximum number of results to return.
        top_k_docs: Number of documents selected at stage 1.
        verbose: Print progress to stdout.

    Returns:
        Dict with query, strategy, doc search, agent trace, sources,
        and metrics.
    """
    reset_counters()
    t_start = time.time()
    timings: dict = {}

    if verbose:
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Strategy: llm-tree (top_k_docs={top_k_docs}, "
              f"max_actions={max_actions}, max_reads={max_reads})")
        print("=" * 60)

    catalog = load_catalog()
    doc_cache: dict[str, dict] = {}

    def _get_doc(doc_id: str) -> dict:
        """Lazy-load and memoize a document by id."""
        if doc_id not in doc_cache:
            doc_cache[doc_id] = load_doc(doc_id)
        return doc_cache[doc_id]

    scratchpad: list[dict] = []

    snap = snapshot_counters()
    t_step = time.time()
    doc_result = doc_search(query, catalog, top_k=top_k_docs, verbose=verbose)
    timings["doc_search"] = step_metrics(t_step, snap)

    doc_ids = doc_result.get("doc_ids", [])[:top_k_docs]
    if not doc_ids:
        return {
            "query": query,
            "strategy": "llm-tree",
            "picked_doc_ids": [],
            "doc_search": doc_result,
            "error": "No relevant documents found",
            "metrics": {**get_stats(), "elapsed_s": round(time.time() - t_start, 2),
                        "step_metrics": timings},
        }

    picked_docs: dict[str, dict] = {did: _get_doc(did) for did in doc_ids}
    multidoc_outline = _render_multidoc_outline(picked_docs)

    scratchpad.append({
        "step": 0,
        "action": "doc_search",
        "args": {},
        "observation": {
            "doc_ids": doc_ids,
            "doc_titles": {did: doc.get("judul", "") for did, doc in picked_docs.items()},
        },
    })

    snap = snapshot_counters()
    t_step = time.time()

    actions_used = 0
    reads_used = 0
    submitted = False
    selected: list[dict] = []
    parse_failures = 0

    while actions_used < max_actions and not submitted:
        steering_hints = _anti_loop_hints(
            scratchpad, actions_used, reads_used, max_actions, max_reads,
        )
        system, prompt = _build_prompt_parts(
            query, scratchpad,
            actions_left=max_actions - actions_used,
            reads_left=max_reads - reads_used,
            picked_docs=picked_docs,
            multidoc_outline=multidoc_outline,
            steering_hints=steering_hints,
        )

        try:
            response = llm_call(prompt, system=system)
            parse_failures = 0
        except json.JSONDecodeError:
            parse_failures += 1
            scratchpad.append({
                "step": len(scratchpad),
                "action": "(invalid_json)",
                "args": {},
                "observation": {"error": "Response was not valid JSON, try again with valid JSON."},
            })
            actions_used += 1
            if parse_failures >= 3:
                break
            continue

        action = response.get("action") or ""
        args = response.get("args") or {}
        thinking = response.get("thinking", "")

        observation: dict = {}

        if action == "inspect_doc":
            doc_id = args.get("doc_id") or (doc_ids[0] if doc_ids else "")
            observation = _tool_inspect_doc(picked_docs, doc_id)

        elif action == "expand":
            doc_id = _coerce_id(args.get("doc_id"))
            node_id = _coerce_id(args.get("node_id"))
            if not doc_id:
                observation = {"error": f"expand requires doc_id (one of {doc_ids})."}
            elif not node_id:
                observation = {"error": "expand requires node_id."}
            else:
                obs = _tool_expand(picked_docs, doc_id, node_id)
                if "error" in obs and doc_id in picked_docs:
                    obs["hint_nearby"] = _siblings_hint_multidoc(picked_docs, doc_id, node_id)
                observation = obs

        elif action == "read":
            if reads_used >= max_reads:
                observation = {"error": f"Read budget exhausted ({max_reads}). Submit soon."}
            else:
                doc_id = _coerce_id(args.get("doc_id"))
                node_id = _coerce_id(args.get("node_id"))
                if not doc_id:
                    observation = {"error": f"read requires doc_id (one of {doc_ids})."}
                elif not node_id:
                    observation = {"error": "read requires node_id."}
                else:
                    obs = _tool_read(picked_docs, doc_id, node_id)
                    if "error" in obs and doc_id in picked_docs:
                        obs["hint_nearby"] = _siblings_hint_multidoc(picked_docs, doc_id, node_id)
                    observation = obs
                    if "error" not in obs:
                        reads_used += 1

        elif action == "submit":
            # Some models return args as a bare list of node refs instead
            # of the documented {"node_ids": [...], "reasoning": "..."} dict.
            # Accept both shapes so a schema-loose model is not penalized.
            if isinstance(args, list):
                refs = args
                reasoning = ""
            else:
                refs = args.get("node_ids") or []
                reasoning = args.get("reasoning", "")
            resolved: list[dict] = []
            invalid: list[str] = []
            for ref in refs:
                doc_id, node_id = _parse_node_ref(ref, default_doc_id=None)
                if not (doc_id and node_id):
                    invalid.append(f"{ref!r} (missing doc_id, use 'doc_id/node_id' format)")
                    continue
                if doc_id not in picked_docs:
                    invalid.append(f"{doc_id}/{node_id} (doc not in picked set)")
                    continue
                if find_node(picked_docs[doc_id].get("structure", []), node_id) is None:
                    invalid.append(f"{doc_id}/{node_id} (node not found)")
                    continue
                key = (doc_id, node_id)
                if key not in {(s["doc_id"], s["node_id"]) for s in resolved}:
                    resolved.append({"doc_id": doc_id, "node_id": node_id})
            if resolved:
                selected = resolved
                submitted = True
                observation = {
                    "submitted": True,
                    "count": len(resolved),
                    "node_ids": [f"{s['doc_id']}/{s['node_id']}" for s in resolved],
                    "reasoning": reasoning[:300],
                }
                if invalid:
                    observation["dropped"] = invalid
            else:
                observation = {
                    "error": "submit produced no valid node_ids, refine and try again.",
                    "invalid": invalid,
                }

        else:
            observation = {
                "error": f"Unknown action '{action}'.",
                "valid_actions": list(VALID_ACTIONS),
            }

        scratchpad.append({
            "step": len(scratchpad),
            "thinking": thinking[:300],
            "action": action or "(empty)",
            "args": args,
            "observation": observation,
        })
        actions_used += 1

        if verbose:
            obs_preview = json.dumps(observation, ensure_ascii=False)[:200]
            print(f"\n[Step {actions_used}] {action} -> {obs_preview}")
            if thinking:
                print(f"  thinking: {thinking[:200]}")

    timings["agent_loop"] = step_metrics(t_step, snap)

    submitted_refs = [f"{s['doc_id']}/{s['node_id']}" for s in selected]

    final_refs, slot_labels = tree_finalize(
        submitted_ids=submitted_refs,
        top_k=top_k,
    )

    if not final_refs:
        return {
            "query": query,
            "strategy": "llm-tree",
            "picked_doc_ids": doc_ids,
            "doc_search": doc_result,
            "agent": {
                "actions_used": actions_used,
                "reads_used": reads_used,
                "submitted": submitted,
                "scratchpad": scratchpad,
            },
            "error": (
                f"Agent submitted no valid node "
                f"(submitted_refs={len(submitted_refs)})"
            ),
            "metrics": {**get_stats(), "elapsed_s": round(time.time() - t_start, 2),
                        "step_metrics": timings},
        }

    sources: list[dict] = []
    for pos, (ref, src_label) in enumerate(zip(final_refs, slot_labels)):
        doc_id, _, nid = ref.partition("/")
        doc = picked_docs.get(doc_id) or {}
        node = find_node(doc.get("structure", []), nid) or {}
        sources.append({
            "doc_id": doc_id,
            "node_id": nid,
            "title": node.get("title", ""),
            "navigation_path": node.get("navigation_path", ""),
            "rerank_position": pos,
            "submission_source": src_label,
        })

    final_ids = [s["node_id"] for s in sources]
    submitted_ids = [s["node_id"] for s in selected]

    submission_source_counts = {
        "agent_submit": slot_labels.count("agent_submit"),
    }

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "llm-tree",
        "picked_doc_ids": doc_ids,
        "doc_search": doc_result,
        "agent": {
            "actions_used": actions_used,
            "reads_used": reads_used,
            "submitted": submitted,
            "submitted_count": len(submitted_ids),
            "submission_source_counts": submission_source_counts,
            "scratchpad": scratchpad,
        },
        "node_ids": final_ids,
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": timings},
    }

    save_log(result)

    if verbose:
        print("\n" + "=" * 60)
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens  |  submitted={submitted}")
        print("=" * 60)

    return result


def main() -> None:
    """CLI entry point for the LLM tree retrieval module."""
    ap = argparse.ArgumentParser(description="LLM tree retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS,
                    help=f"Hard cap on agent steps (default {MAX_ACTIONS})")
    ap.add_argument("--max-reads", type=int, default=MAX_READS,
                    help=f"Hard cap on read tool calls (default {MAX_READS})")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"Final ranked output length after fallback (default {DEFAULT_TOP_K})")
    ap.add_argument("--top-k-docs", type=int, default=DOC_PICK_TOP_K,
                    help=f"Number of docs picked at stage 1 (default {DOC_PICK_TOP_K})")
    args = ap.parse_args()

    result = retrieve(args.query,
                      max_actions=args.max_actions, max_reads=args.max_reads,
                      top_k=args.top_k, top_k_docs=args.top_k_docs)

    print("\n" + "-" * 60)
    print("DASAR HUKUM:")
    for src in result.get("sources", []):
        path = src.get("navigation_path") or src.get("node_id", "")
        print(f"  > {src.get('doc_id', '')} :: {path}")
    print("-" * 60)


if __name__ == "__main__":
    main()

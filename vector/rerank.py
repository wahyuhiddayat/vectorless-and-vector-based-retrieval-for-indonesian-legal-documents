"""Reranker stage for vector RAG.

Scores first-stage candidates using a CrossEncoder model and reorders
them by relevance. Supports encoder cross-attention and decoder LLM
pointwise backends via the sentence-transformers CrossEncoder API.
"""

from .common import _RERANKER_REGISTRY, RERANKER_FP32


_QWEN_INSTRUCTION = (
    "Given a legal question in Indonesian, retrieve relevant legal "
    "document sections that answer the question"
)


_ce_model_cache: dict = {}


def _get_cross_encoder(model_id: str):
    """Load and cache a CrossEncoder model.

    Defaults to bfloat16 weights on CUDA to reduce VRAM usage. Setting
    `VECTOR_RERANKER_FP32=1` forces float32 weights, which roughly
    doubles VRAM and halves throughput but eliminates bf16 rounding
    that may shift borderline scores. The dtype is set via model_kwargs
    rather than post-hoc casting because the latter breaks input
    handling on some decoder models.
    """
    if model_id not in _ce_model_cache:
        import torch
        from sentence_transformers import CrossEncoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            dtype = torch.float32 if RERANKER_FP32 else torch.bfloat16
            ce = CrossEncoder(
                model_id,
                model_kwargs={"torch_dtype": dtype},
            )
        else:
            ce = CrossEncoder(model_id)
        _ce_model_cache[model_id] = ce
    return _ce_model_cache[model_id]


def _is_qwen_model(model_id: str) -> bool:
    """Check whether a model ID refers to a Qwen reranker."""
    return "qwen" in model_id.lower()


def _format_qwen_pairs(query: str, candidates: list[dict]) -> list[list[str]]:
    """Build prompt-formatted pairs for Qwen3-Reranker.

    The seq-cls variant requires the full chat template with system,
    user, and assistant blocks. Without this formatting the model
    outputs near-uniform scores and reranking becomes random.
    """
    formatted_query = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on "
        'the Query and the Instruct provided. Note that the answer '
        'can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
        f"<Instruct>: {_QWEN_INSTRUCTION}\n"
        f"<Query>: {query}\n"
    )
    pairs = []
    for c in candidates:
        formatted_doc = (
            f"<Document>: {c['text']}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        pairs.append([formatted_query, formatted_doc])
    return pairs


def rerank(query: str, candidates: list[dict], reranker_name: str,
           top_k: int = 10) -> list[dict]:
    """Rerank candidates and return top_k by descending score.

    Args:
        query: Legal question in Indonesian.
        candidates: Candidate dicts with at least a "text" key.
        reranker_name: Key in _RERANKER_REGISTRY. "none" passes through unchanged.
        top_k: Number of candidates to return.

    Returns:
        Reranked list of candidate dicts with a rerank_score field added.
    """
    cfg = _RERANKER_REGISTRY.get(reranker_name)
    if cfg is None:
        raise ValueError(f"Unknown reranker: {reranker_name!r}")

    if cfg["backend"] == "none":
        out = []
        for c in candidates[:top_k]:
            c_copy = dict(c)
            c_copy["rerank_score"] = None
            out.append(c_copy)
        return out

    if cfg["backend"] == "cross_encoder":
        ce = _get_cross_encoder(cfg["model_id"])
        if _is_qwen_model(cfg["model_id"]):
            pairs = _format_qwen_pairs(query, candidates)
        else:
            pairs = [(query, c["text"]) for c in candidates]
        batch_size = cfg.get("predict_batch_size", 32)
        scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored = [(float(s), c) for s, c in zip(scores, candidates)]
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, c in scored[:top_k]:
            c_copy = dict(c)
            c_copy["rerank_score"] = s
            out.append(c_copy)
        return out

    raise ValueError(f"Unsupported reranker backend: {cfg['backend']!r}")

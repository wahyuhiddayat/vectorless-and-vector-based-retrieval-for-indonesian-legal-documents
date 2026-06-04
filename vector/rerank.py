"""Reranker stage for vector RAG.

Scores first-stage candidates and reorders them by relevance. Two scoring
backends exist. The cross_encoder backend covers encoder rerankers and
seq-cls decoder conversions via the sentence-transformers CrossEncoder API.
The llm_yes_logit backend covers causal-LM rerankers such as
bge-reranker-v2-gemma, which are scored by the logit of the Yes token at
the final position, following the official BAAI usage.
"""

from .common import (
    _RERANKER_REGISTRY,
    RERANKER_BATCH_SIZE,
    RERANKER_FP32,
    RERANKER_MAX_LENGTH,
)


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

    When `RERANKER_MAX_LENGTH` is unset the model reads at its native context
    window, so a full pasal is never truncated. Setting it reproduces the older
    capped runs. Pasals top out near 3200 tokens, well inside the native 8192,
    so the native window is the memory-safe default.
    """
    if model_id not in _ce_model_cache:
        import torch
        from sentence_transformers import CrossEncoder
        kwargs = {}
        if RERANKER_MAX_LENGTH is not None:
            kwargs["max_length"] = RERANKER_MAX_LENGTH
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            dtype = torch.float32 if RERANKER_FP32 else torch.bfloat16
            kwargs["model_kwargs"] = {"torch_dtype": dtype}
        _ce_model_cache[model_id] = CrossEncoder(model_id, **kwargs)
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


_GEMMA_INSTRUCTION = (
    "Given a query A and a passage B, determine whether the passage "
    "contains an answer to the query by providing a prediction of "
    "either 'Yes' or 'No'."
)


_llm_reranker_cache: dict = {}


def _get_llm_reranker(model_id: str):
    """Load and cache a causal-LM reranker plus its tokenizer and Yes-token id.

    The tokenizer padding side is forced to left so that the final position
    of every padded row is the true last token, which is where the Yes logit
    is read. Respects the same dtype env switch as the CrossEncoder loader.
    """
    if model_id not in _llm_reranker_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        kwargs = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            kwargs["torch_dtype"] = torch.float32 if RERANKER_FP32 else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        yes_loc = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        _llm_reranker_cache[model_id] = (model, tokenizer, yes_loc)
    return _llm_reranker_cache[model_id]


def _last_position_logits(model, batch):
    """Run a forward pass and return logits at the final position only.

    Newer transformers versions accept logits_to_keep or num_logits_to_keep,
    which avoids materialising the full sequence-length logits tensor. With a
    256k vocabulary that tensor would not fit in memory at native lengths, so
    the keyword is tried first and the full pass is only a fallback.
    """
    for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}):
        try:
            out = model(**batch, return_dict=True, **kw)
            return out.logits[:, -1, :]
        except TypeError:
            continue
    out = model(**batch, return_dict=True)
    return out.logits[:, -1, :]


def _yes_logit_scores(query: str, candidates: list[dict], model_id: str,
                      batch_size: int) -> list[float]:
    """Score query-candidate pairs with a causal-LM reranker.

    Builds the official BAAI prompt layout. The packed input is the bos
    token, the prefixed query, a separator, and the prefixed passage with
    passage-side truncation, followed by the instruction prompt. The score
    of each pair is the logit of the Yes token at the last position. The
    passage budget defaults to 4096 tokens, which covers the longest pasal
    in the corpus, and VECTOR_RERANKER_MAX_LENGTH overrides it.

    Args:
        query: Legal question in Indonesian.
        candidates: Candidate dicts with at least a "text" key.
        model_id: HuggingFace model id of the causal-LM reranker.
        batch_size: Number of pairs scored per forward pass.

    Returns:
        One float score per candidate, in input order.
    """
    import torch
    model, tokenizer, yes_loc = _get_llm_reranker(model_id)
    max_length = RERANKER_MAX_LENGTH or 4096
    prompt_ids = tokenizer(_GEMMA_INSTRUCTION, add_special_tokens=False)["input_ids"]
    sep_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    query_ids = tokenizer(
        f"A: {query}", add_special_tokens=False,
        truncation=True, max_length=max_length * 3 // 4)["input_ids"]
    # Build the packed input manually, since tokenizer.prepare_for_model was
    # removed in transformers 5.x. Layout follows the official BAAI usage,
    # bos, query, sep, passage, sep, instruction, and only the passage is
    # truncated so the fixed parts always survive.
    fixed_len = len(bos_ids) + len(query_ids) + 2 * len(sep_ids) + len(prompt_ids)
    passage_budget = max(0, max_length - fixed_len)
    items = []
    for c in candidates:
        passage_ids = tokenizer(
            f"B: {c['text']}", add_special_tokens=False)["input_ids"][:passage_budget]
        input_ids = bos_ids + query_ids + sep_ids + passage_ids + sep_ids + prompt_ids
        items.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        })
    device = next(model.parameters()).device
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = tokenizer.pad(
                items[start:start + batch_size], padding=True,
                pad_to_multiple_of=8, return_tensors="pt").to(device)
            logits = _last_position_logits(model, batch)
            scores.extend(logits[:, yes_loc].float().cpu().tolist())
    return scores


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

    batch_size = RERANKER_BATCH_SIZE or cfg.get("predict_batch_size", 32)
    if cfg["backend"] == "cross_encoder":
        ce = _get_cross_encoder(cfg["model_id"])
        if _is_qwen_model(cfg["model_id"]):
            pairs = _format_qwen_pairs(query, candidates)
        else:
            pairs = [(query, c["text"]) for c in candidates]
        scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    elif cfg["backend"] == "llm_yes_logit":
        scores = _yes_logit_scores(query, candidates, cfg["model_id"], batch_size)
    else:
        raise ValueError(f"Unsupported reranker backend: {cfg['backend']!r}")

    scored = [(float(s), c) for s, c in zip(scores, candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, c in scored[:top_k]:
        c_copy = dict(c)
        c_copy["rerank_score"] = s
        out.append(c_copy)
    return out

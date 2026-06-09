# Vectorless and Vector-Based Retrieval for Indonesian Legal Documents

Code for the thesis "Evaluating Vectorless and Vector-Based Retrieval for Indonesian Legal Documents". The project compares two retrieval paradigms on the same Indonesian legal corpus. Vectorless RAG combines BM25 with LLM navigation over hierarchical document trees, and vector RAG uses Qdrant dense retrieval with an optional reranker. Both paradigms read the same indexed corpus, so any difference in measured effectiveness is attributable to the retrieval design rather than to data variation. The pipeline is retrieval-only, there is no answer generation, and every system is evaluated at k=10 against a held-out test split.

## Research questions

1. **RQ1, vectorless effectiveness at default settings.** Which vectorless configuration is most effective. Six methods (`bm25-flat`, `bm25-tree`, `hybrid-flat`, `hybrid-tree`, `llm-flat`, `llm-tree`) crossed with three granularities (pasal, ayat, rincian).
2. **RQ2, vector effectiveness at default settings.** Which vector configuration is most effective. Embedding models (BGE-M3, multilingual-E5-large-instruct, NusaBERT-large-v4) crossed with rerankers (none, bge-reranker-v2-m3, qwen3-reranker-0.6b) crossed with granularities.
3. **RQ3, boost improvement on the paradigm winners.** How much the RQ1 and RQ2 winners improve under tuning interventions on the development split, such as query expansion, hyperparameter tuning, and retrieval-LLM upgrades.
4. **RQ4, cross-paradigm effectiveness and cost.** The tuned winners evaluated once on the sealed test split, with a paired-randomization significance test on Recall@10 and MRR@10, bootstrap confidence intervals, and cost reported alongside effectiveness.

## Architecture

```
scraper/ -> data/raw/  ->  vectorless/indexing/ -> data/index_{pasal,ayat,rincian}/
                                                          |
                                            vectorless/retrieval/ (BM25, LLM, Hybrid)
                                            vector/ (Qdrant dense, reads the same index JSON)
                                                          |
                                            scripts/ (ground truth generation and evaluation)
```

The corpus is parsed once into three granularities. The `pasal` index is produced by the LLM parser followed by OCR cleanup and per-node summaries. The `ayat` and `rincian` indexes are derived deterministically from `pasal`. Both retrieval paradigms consume these shared index files.

## Repository structure

| Path | Contents |
|---|---|
| `scraper/` | BPK JDIH scraper that acquires source PDFs and metadata into `data/raw/` |
| `vectorless/indexing/` | LLM-first indexing, OCR cleanup, deterministic re-split, per-node summaries, verification |
| `vectorless/retrieval/` | The six vectorless methods, one subpackage per family (`bm25`, `hybrid`, `llm`) |
| `vector/` | Vector indexing and retrieval over Qdrant with an optional reranker |
| `scripts/gt/` | Ground-truth construction pipeline, from document selection to the final test split |
| `scripts/eval/` | Evaluation harness, tuning flows, and the significance test, with `core/` as the shared engine |
| `scripts/parser/` | Thin CLI wrappers around the indexing library, plus quality and granularity checks |
| `scripts/aggregation/`, `scripts/analysis/` | Cost aggregation and corpus statistics |

## Setup

```bash
pip install -r requirements.txt
gcloud auth application-default login    # Vertex AI application default credentials
cp .env.example .env                     # then fill in the API keys
```

## Pipeline

All commands run from the project root.

### 1. Scrape

Acquire documents for one regulation type from BPK JDIH (peraturan.bpk.go.id). The corpus samples across the 31 categories defined in `vectorless/categories.py`.

```bash
python scraper/bpk_scraper.py --jenis 8 --pages 1-5
```

### 2. Index

Parse a category end to end into all three granularities. The `pasal` index is the only one that calls the LLM parser, the rest are re-derived from it.

```bash
python -m vectorless.indexing.build --category UU
python -m vectorless.indexing.status --refresh-verify   # inspect indexing progress
```

### 3. Ground truth

The test set is built by an eleven-step pipeline under `scripts/gt/`. Steps 3 and 5 are LLM annotation and judging, which can be run manually or through the API runners. Every other step is a local script.

```text
0. select_gt_docs.py     choose GT-source documents per category
1. allocate_quotas.py    assign a query quota per document and type
2. prompt.py             build annotator prompts
3. annotate              write raw GT JSON (manual or auto_annotate.py)
4. build_validate.py     run local gates and build the judge prompt
5. judge                 clean semantic issues (manual or auto_judge.py)
6. apply_validation.py   extract the cleaned judge output
7. log_review.py         author spot-check
8. collect.py            merge accepted raw GT into ground_truth.json
9. finalize.py           write validated_testset.pkl
10. split_dataset.py     write the dev and test qid splits
```

Three query types are used. `factual` and `paraphrased` have one gold anchor each, `multihop` has two anchors inside one document. The split is 50/50 dev and test, deterministic with seed 42, stratified by category and query type.

### 4. Evaluate

Each run writes a directory under `data/eval_runs/<label>` with an overall summary, per-slice CSVs, per-query records, a progress log, and an error log. Decisions in the tuning flows use MAP@10 as the primary metric because the ground truth is partly multi-gold.

```bash
# Development split.
python scripts/eval/vectorless.py --label my_run --split dev
python scripts/eval/vector.py     --label my_run --split dev --qdrant-path ./qdrant_local

# Sealed test split, only after the winner is picked. Requires the seal flag.
EVAL_ALLOW_TEST=1 python scripts/eval/vectorless.py --label my_test_run --split test

# Paired significance test between two runs, for RQ4.
python scripts/eval/significance.py --help
```

A single retrieval call can also be run directly for a sanity check.

```bash
python -m vectorless.retrieval.hybrid.flat "query"
python -m vector.retrieve_vector "query" --reranker bge-reranker-v2-m3
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI key for the parser and the GT judge |
| `ANTHROPIC_API_KEY` | required | Anthropic key for the GT annotator |
| `DEEPSEEK_API_KEY` | required | DeepSeek key for the retrieval LLM |
| `GOOGLE_CLOUD_PROJECT` | required | Vertex AI project for the parser-judge, summary, and OCR roles |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI region |
| `DATA_INDEX` | `data/index_pasal` | Which granularity index retrieval reads from |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server for vector RAG |
| `EVAL_ALLOW_TEST` | unset | Must be set to `1` to evaluate on the sealed test split |

## LLM model pins

Roles are distributed across three vendors so that no comparison boundary shares a model family. The parser and its judge differ, the GT annotator and the retrieval LLM differ, and the GT annotator and the GT judge differ. The pins live in `vectorless/models.py`, `vectorless/categories.py`, and the GT runner defaults.

| Role | Vendor | Model |
|---|---|---|
| Parser | OpenAI | `gpt-5` |
| Summary and OCR clean | Vertex Gemini | `gemini-2.5-flash-lite` |
| Parser judge | Vertex Gemini | `gemini-2.5-pro` |
| Retrieval LLM | DeepSeek | `deepseek-v4-flash` |
| GT annotator | Anthropic | `claude-sonnet-4-6` |
| GT judge | OpenAI | `gpt-5` |

## Notes

The `data/` directory is not tracked in git. It holds the scraped corpus, the indexes, the ground truth, and the evaluation runs, and is backed up separately. The pipeline is retrieval-only, so no answer-generation component is included.

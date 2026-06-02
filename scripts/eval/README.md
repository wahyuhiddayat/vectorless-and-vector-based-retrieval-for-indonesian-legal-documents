# Evaluation harness

Retrieval-only evaluation for the vectorless and vector paradigms, plus the
Stage 2 tuning flows and the RQ4 significance test. Every run writes a
directory under `data/eval_runs/<label>` with `summary_overall.json`,
per-slice CSVs, `records/`, `progress.log`, and `errors.jsonl`.

Decisions in both tuning flows use MAP@10 as the primary metric, because the
ground truth is partly multi-gold. Multihop queries have two required pasals,
and MAP credits retrieving all of them while MRR ignores the second.

## Entry points

| Script | Purpose |
|---|---|
| `vectorless.py` | Run vectorless systems (bm25, hybrid, llm x flat, tree) on a split |
| `vector.py` | Run the dense vector pipeline (Qdrant plus optional reranker) on a split |
| `tune_vl_step1..5_*.py` | Stage 2 vectorless tuning, hybrid-tree, one script per step |
| `tune_bm25_params.py` | Free BM25 k1/b grid on candidate recall, no LLM |
| `tune_vector.py` | Stage 2 vector tuning, all steps in one script |
| `significance.py` | Paired significance test between two runs for RQ4 |
| `expand_queries.py` | Build the cached query expansion JSON used by both paradigms |

## Running a single evaluation

```bash
# Vectorless, dev split.
python scripts/eval/vectorless.py --label my_run --systems hybrid-tree --granularities pasal --split dev

# Vector, dev split.
python scripts/eval/vector.py --label my_run --split dev --qdrant-path ./qdrant_local

# Sealed test split, requires the seal flag.
EVAL_ALLOW_TEST=1 python scripts/eval/vectorless.py --label my_test_run --split test
```

Useful `vectorless.py` flags. `--query-expansion <path>` substitutes expanded
queries, `--resume` re-runs only the missing or errored queries in an existing
run directory, `--overwrite` starts the run fresh, `--query-limit N` with
`--random-seed S` samples a subset, `--per-type-limit N` samples per query
type.

## Stage 2 vectorless tuning, five steps

The vectorless winner is hybrid-tree at pasal. Its tuning is split into one
script per step so each step can be launched and monitored on its own, and a
crash in one step does not discard the others. Each step carries its winner
forward through `data/eval_runs/tune_vectorless_state.json`, decides on
MAP@10, and prints the next command. Run them in order.

```bash
python scripts/eval/tune_vl_step1_bm25params.py   # BM25 k1/b, free, no LLM
python scripts/eval/tune_vl_step2_bm25topk.py     # HYBRID_BM25_TOP_K in {10,20,30,50}
python scripts/eval/tune_vl_step3_docpick.py      # HYBRID_DOC_PICK_TOP_K in {1,2,3,5}
python scripts/eval/tune_vl_step4_model.py        # deepseek-v4-flash to deepseek-v4-pro
python scripts/eval/tune_vl_step5_qe.py           # query expansion, writes final log
```

Behavior worth knowing.

- Step 1 creates the state file, so re-running it resets the whole tuning.
- Each step auto-resumes if its run directory already exists, so after a
  crash or an API balance running out, re-run the same step and it continues
  from where it stopped.
- A run that comes back with more than ten errored queries aborts the winner
  pick, so a drained balance cannot silently corrupt a decision.
- Ties within MAP@10 tolerance resolve to the cheaper setting (smaller pool).
  A model upgrade or query expansion is accepted only if its MAP@10 lift
  clears the intervention threshold.
- Step 5 writes the final tuned config and the full decision history to
  `data/eval_runs/tune_vectorless_log.json`.

The hybrid-tree knobs these steps set are read from the environment,
`HYBRID_BM25_TOP_K`, `HYBRID_DOC_PICK_TOP_K`, `HYBRID_BM25_K1`,
`HYBRID_BM25_B`, and `RETRIEVAL_MODEL_OVERRIDE` for the model upgrade.

## Stage 2 vector tuning

Vector tuning runs all steps in one script, since the dense pipeline has no
LLM cost per query and a single run is fast.

```bash
python scripts/eval/tune_vector.py --qdrant-path ./qdrant_local
```

## RQ4 significance test

After both tuned winners have been evaluated on the sealed test split, compare
them with a paired test. See the script's own `--help` for the exact argument
names.

```bash
python scripts/eval/significance.py --help
```

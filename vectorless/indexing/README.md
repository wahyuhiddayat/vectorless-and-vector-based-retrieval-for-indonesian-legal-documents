# Indexing Playbook

Operational guide for the vectorless indexing pipeline.

Run all commands from the project root.

## What each granularity means

- `pasal`: main parsed index from LLM parse + OCR clean + summary
- `ayat`: derived from pasal by deterministic re-split
- `rincian`: finest derived index, also derived from pasal

Only pasal talks to the LLM parser. Ayat and rincian are always regenerated from pasal.

## Status first

Check current indexing progress before doing anything else.

```powershell
python -m vectorless.indexing.status --refresh-verify
python -m vectorless.indexing.status --category PMK --refresh-verify
python -m vectorless.indexing.status --doc-id pmk-10-2026 --refresh-verify
```

Important fields in the output:

- `Pasal/Ayat/Full split`: how many docs have each granularity indexed
- `Stale parse`: docs that need rebuild after a parser-version bump
- `Stale derived`: docs whose ayat or rincian are not synced to the latest pasal
- `GT candidates`: docs structurally safe enough for ground-truth work
- `Clean retrieval candidates`: docs fully synced and cleaned for retrieval experiments

## CLI reference

### build (python -m vectorless.indexing.build)

Flags:

| Flag | Description |
|------|-------------|
| `--doc-id X` | Index a specific document (repeatable) |
| `--category Y` | Index every document in this jenis_folder (e.g. UU, PMK) |
| `--resplit-only` | Skip LLM parse, only re-derive ayat + rincian from existing pasal |
| `--dry-run` | Preview LLM parse without overwriting index files |
| `--skip-existing` | Skip docs already indexed across all 3 granularities |
| `--rebuild-catalog-only` | Only rebuild catalog.json for all granularities |

### status (python -m vectorless.indexing.status)

Flags:

| Flag | Description |
|------|-------------|
| `--doc-id X` | Show status for one document |
| `--category Y` | Filter by category (comma-separated, e.g. UU,PP,PMK) |
| `--refresh-verify` | Run verification checks before printing status |
| `--json` | Output as JSON |

### verify (python -m vectorless.indexing.verify)

Flags:

| Flag | Description |
|------|-------------|
| `--granularity X` | Verify one granularity (pasal, ayat, or rincian) |
| `--all` | Verify all granularities + cross-compare |
| `--doc-id X` | Verify a single document |
| `--category Y` | Filter by category |
| `--json` | Output as JSON |

## Daily workflows

### 1. New documents after scraping

```powershell
python -m vectorless.indexing.build --category PMK
```

This runs the full pipeline per doc: LLM parse, OCR clean, resplit, and summary annotation across all three granularities.

### 2. Resplit only (after a parser.py fix to the re-splitter)

```powershell
python -m vectorless.indexing.build --category PMK --resplit-only
```

Skips LLM parse. Re-derives ayat and rincian from existing pasal index files and re-annotates summaries on the derived granularities.

### 3. Rebuild one problematic document

```powershell
python -m vectorless.indexing.build --doc-id pmk-6-2026
```

### 4. Skip already indexed documents

```powershell
python -m vectorless.indexing.build --category PMK --skip-existing
```

### 5. Rebuild catalog only

```powershell
python -m vectorless.indexing.build --rebuild-catalog-only
```

## Interpreting verify results

- `OK`: structurally clean
- `WARN`: usable, but has parser anomalies worth noting
- `FAIL`: broken enough to avoid for experiments until fixed
- `MISSING`: index file does not exist yet

WARN does not mean you must rebuild immediately. It usually means the doc is usable but worth revisiting after a parser fix.

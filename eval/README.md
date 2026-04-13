# Evaluation Guide

This folder contains benchmark datasets for the `mmrag eval` command.

## Dataset Format

Use JSONL (`.jsonl`) or JSON list (`.json`).

Each case supports:
- `case_id` (optional)
- `question` (required)
- `query_image_path` (optional, relative paths resolve from dataset folder)
- `collection` (optional override)
- `top_k` (optional override)
- `expected_chunk_ids` (optional list)
- `expected_source_paths` (optional list)

At least one of `expected_chunk_ids` or `expected_source_paths` should be present for retrieval scoring.

## Example

```json
{
  "case_id": "sample_case",
  "question": "What is the contract number?",
  "expected_source_paths": ["contracts.pdf"]
}
```

## Run Evaluation

From repo root:

```bash
mmrag eval ./eval/datasets/starter_eval.jsonl --ingest-path ./data
```

Optional:
- `--k-values 1,3,5,10`
- `--collection your_collection`
- `--output ./eval/report.json`

Run ablations:

```bash
mmrag eval ./eval/datasets/starter_eval.jsonl --ingest-path ./data --ablation
```

Optional ablation controls:
- `--ablation-modes dense_only,hybrid,hybrid_rerank`
- `--ablation-baseline dense_only`

## Recommended Benchmark Size

For interview-grade reporting, curate 30-50 cases split across:
- text-heavy questions
- table lookup questions
- image-grounded questions
- multi-hop cross-document questions

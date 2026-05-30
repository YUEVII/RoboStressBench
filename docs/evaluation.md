# Evaluation

## CLI

```bash
python run.py --config configs/local.yaml --gpu-ids 0
```

Start with one GPU and `batch_size: 1`. If the run is stable, scale up `configs/model_registry.yaml` and pass more GPU ids.

Useful overrides:

```bash
python run.py --config configs/local.yaml --models Qwen3-VL-4B-Instruct --gt-types mcq --limit 30
python run.py --config configs/api.yaml --datasets 01,02 --limit 30
```

## Smoke Tests

Use these on a GPU node after setting up the dataset path and checkpoints.

Dry-run smoke test, no model loading:

```bash
python run.py --config configs/local_smoke.yaml --dry-run
```

Local checkpoint smoke test:

```bash
python run.py --config configs/local_smoke.yaml --gpu-ids 0
```

API smoke test:

```bash
python run.py --config configs/api_smoke.yaml
```

Molmo smoke test from the Molmo environment:

```bash
python run.py --config configs/molmo_smoke.yaml --gpu-ids 0
```

## Outputs

Each model writes:

```text
outputs/<run>/<model_name>/predictions.json
outputs/<run>/<model_name>/submission.json
outputs/<run>/<model_name>/summary.json
outputs/<run>/run_statistics.log
```

`summary.json` contains overall metrics, per-dataset and per-stress breakdowns, unparsed response counts, and timing/profiling fields.

`submission.json` is the compact leaderboard format. It does not include ground truth, metrics, prompts, images, or raw model responses:

```json
{
  "schema_version": "1.0",
  "benchmark": "RoboStressBench",
  "dataset_version": "1.0",
  "model_name": "model-name",
  "predictions": [
    {
      "sample_id": "000001",
      "parsed_answer": {
        "kind": "mcq",
        "answer_index": 0,
        "answer": "A"
      }
    }
  ]
}
```

For unparseable responses, `parsed_answer` is `null`.

## Metrics

- MCQ: exact match on parsed answer index.
- BBox: IoU between predicted and ground-truth boxes. Summary includes mean IoU, IoU >= 0.5, and IoU >= 0.95.
- Mask: point hit against the thresholded ground-truth mask.
- Total accuracy: MCQ and mask are 0/1; bbox contributes 1 only when IoU >= 0.95.

## Troubleshooting

If a run fails with `exit_codes=[-9]`, a worker was killed by the OS or job scheduler rather than raising a normal Python exception. The most common cause is CPU RAM or GPU memory pressure. Lower `batch_size`, use fewer GPU workers, or test with `--limit` before running the full benchmark.

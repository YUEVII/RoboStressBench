# Dataset Format

A RoboStressBench dataset directory contains:

```text
manifest.jsonl
metadata.jsonl
<dataset>/<subcategory>/records/<sample_id>.json
<dataset>/<subcategory>/media/rgb/<image>
<dataset>/<subcategory>/media/gt_mask/<mask>   # only for mask samples
```

## manifest.jsonl

Each row is a compact index entry:

```json
{
  "sample_id": "000001",
  "dataset": "01",
  "subcategory": "01",
  "task": "spatial_mcq",
  "gt_type": "mcq",
  "stress_tags": ["V__truncated_out_of_frame"],
  "record_relpath": "01/01/records/000001.json",
  "rgb_relpath": "01/01/media/rgb/000001_rgb.jpg",
  "mask_relpath": null
}
```

## records/*.json

Each record contains the prompt, image path, stress labels, and ground truth:

```json
{
  "sample_id": "000001",
  "status": "annotated",
  "annotation": {
    "sample_id": "000001",
    "task": "spatial_mcq",
    "stress": {"V": ["truncated_out_of_frame"]},
    "instruction": "...",
    "rgb": "01/01/media/rgb/000001_rgb.jpg",
    "gt": {"type": "mcq", "mcq": {"options": ["..."], "answer_index": 0}}
  }
}
```

## Ground Truth Types

MCQ:

```json
{"type": "mcq", "mcq": {"options": ["A text", "B text"], "answer_index": 0}}
```

Bounding box:

```json
{"type": "bbox", "bbox": {"format": "xyxy", "space": "permille", "box": [100, 200, 600, 800]}}
```

Mask point target:

```json
{"type": "mask", "mask": {"path": ".../gt_mask/000001_mask.jpg", "encoding": "jpg/thresholded_mask"}}
```

Coordinates are stored in permille space: `0` is the left/top image boundary and `1000` is the right/bottom image boundary.

## metadata.jsonl

`metadata.jsonl` is a flattened convenience file. It duplicates the useful fields from `manifest.jsonl` and `records/*.json`, including `file_name`, `instruction`, `stress`, `ground_truth`, and task-specific fields such as `options`, `answer`, `bbox`, or `mask_relpath`. The evaluation code does not require it, but it is useful for Hugging Face browsing and external analysis.

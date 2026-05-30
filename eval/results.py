from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final


@dataclass
class PredictionRecord:
    question_id: int | str
    sample_id: str
    dataset: str
    subcategory: str
    task: str
    gt_type: str
    stress: Any | None
    image_path: str
    record_path: str | None
    prompt: str
    raw_response: str | None
    parsed_answer: dict[str, Any] | None
    ground_truth: dict[str, Any]
    metrics: dict[str, Any]
    model_name: str
    error: str | None
    rectified_image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_breakdown_key(record: PredictionRecord) -> str:
    return f"{record.dataset}/{record.subcategory}" if record.subcategory else record.dataset


def ensure_model_output_dir(output_root: Path, model_name: str) -> Path:
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_predictions(predictions_path: Path, records: list[PredictionRecord]) -> None:
    payload = [record.to_dict() for record in records]
    with predictions_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_submission_payload(
    records: list[PredictionRecord],
    model_name: str,
    dataset_version: str = "1.0",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "benchmark": "RoboStressBench",
        "dataset_version": dataset_version,
        "model_name": model_name,
        "predictions": [
            {
                "sample_id": record.sample_id,
                "parsed_answer": record.parsed_answer,
            }
            for record in records
        ],
    }


def write_submission(
    submission_path: Path,
    records: list[PredictionRecord],
    model_name: str,
    dataset_version: str = "1.0",
) -> None:
    payload = build_submission_payload(
        records=records,
        model_name=model_name,
        dataset_version=dataset_version,
    )
    with submission_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _has_parsed_answer(record: PredictionRecord) -> bool:
    return record.parsed_answer is not None


def _sample_score(record: PredictionRecord) -> float:
    """Scalar contribution for total_acc: mcq/mask are 0/1; bbox is 1 iff IoU >= 0.95."""
    gt = record.gt_type
    if gt == "mcq":
        return 1.0 if record.metrics.get("mcq_correct") is True else 0.0
    if gt == "mask":
        return 1.0 if record.metrics.get("mask_correct") is True else 0.0
    if gt == "bbox":
        iou = record.metrics.get("bbox_iou")
        if iou is None:
            return 0.0
        return 1.0 if float(iou) >= _STRESS_BREAKDOWN_BBOX_HIT_IOU else 0.0
    return 0.0


# Stress L1/L2 terminal breakdown: mean score within the bucket only; bbox counts as 1 iff IoU >= threshold.
_STRESS_BREAKDOWN_BBOX_HIT_IOU: Final[float] = 0.95


def _stress_breakdown_sample_score(record: PredictionRecord) -> float:
    gt = record.gt_type
    if gt == "mcq":
        return 1.0 if record.metrics.get("mcq_correct") is True else 0.0
    if gt == "mask":
        return 1.0 if record.metrics.get("mask_correct") is True else 0.0
    if gt == "bbox":
        iou = record.metrics.get("bbox_iou")
        if iou is None:
            return 0.0
        return 1.0 if float(iou) >= _STRESS_BREAKDOWN_BBOX_HIT_IOU else 0.0
    return 0.0


def _stress_breakdown_mean_score(bucket: list[PredictionRecord]) -> float:
    return sum(_stress_breakdown_sample_score(record) for record in bucket) / len(bucket)


def _stress_primary_keys_present(stress: Any) -> set[str]:
    if not isinstance(stress, dict):
        return set()
    out: set[str] = set()
    for key in ("G", "M", "V", "L"):
        val = stress.get(key)
        if isinstance(val, list) and len(val) > 0:
            out.add(key)
    return out


# Sixteen secondary stresses under their primary dimension (annotation keys G/M/V/L).
# Must match annotation["stress"] in dataset records — verified against RoboStressBench manifests:
# G includes occlusion; low_contrast_blend is material (M), not viewpoint (V).
CANONICAL_SECOND_LEVEL_BY_PRIMARY: Final[dict[str, tuple[str, ...]]] = {
    "G": ("cluttered_layout", "non_rigid_deform", "occlusion", "stacked_layout"),
    "M": (
        "complex_texture",
        "dark_absorptive",
        "low_contrast_blend",
        "specular_confusion",
        "transparent",
    ),
    "V": ("extreme_viewpoint", "small_scale", "truncated_out_of_frame"),
    "L": ("global_overexposure", "global_underexposure", "local_overexposure", "local_underexposure"),
}


def _canonical_secondary_composite_keys() -> list[str]:
    keys: list[str] = []
    for pk in ("G", "M", "V", "L"):
        for tag in CANONICAL_SECOND_LEVEL_BY_PRIMARY[pk]:
            keys.append(f"{pk} {tag}")
    return keys


def _composite_primary_key(composite: str) -> str | None:
    parts = composite.split(" ", 1)
    if len(parts) != 2:
        return None
    letter = parts[0]
    return letter if letter in {"G", "M", "V", "L"} else None


def _ordered_secondary_composite_keys(observed_second_keys: set[str]) -> list[str]:
    """L2 keys grouped by primary (G, then M, V, L): canonical secondaries first, then extras."""
    global_canonical_list = _canonical_secondary_composite_keys()
    global_canonical_set = set(global_canonical_list)
    ordered: list[str] = []
    for pk in ("G", "M", "V", "L"):
        for tag in CANONICAL_SECOND_LEVEL_BY_PRIMARY[pk]:
            ordered.append(f"{pk} {tag}")
        extras = sorted(
            k
            for k in observed_second_keys
            if k not in global_canonical_set and _composite_primary_key(k) == pk
        )
        ordered.extend(extras)
    placed = set(ordered)
    orphans = sorted(k for k in observed_second_keys if k not in placed)
    ordered.extend(orphans)
    return ordered


def summarize_predictions(records: list[PredictionRecord]) -> dict[str, Any]:
    total_questions = len(records)
    parsed_count = sum(1 for record in records if _has_parsed_answer(record))
    unparsed_count = total_questions - parsed_count
    error_count = sum(1 for record in records if record.error)

    mcq_unparsed = sum(
        1 for record in records if record.gt_type == "mcq" and not _has_parsed_answer(record)
    )
    mask_unparsed = sum(
        1 for record in records if record.gt_type == "mask" and not _has_parsed_answer(record)
    )
    bbox_unparsed = sum(
        1 for record in records if record.gt_type == "bbox" and not _has_parsed_answer(record)
    )

    per_dataset: dict[str, list[PredictionRecord]] = {}
    per_gt_type: dict[str, list[PredictionRecord]] = {}
    primary_to_records: dict[str, list[PredictionRecord]] = {"G": [], "M": [], "V": [], "L": []}
    secondary_to_records: dict[str, list[PredictionRecord]] = defaultdict(list)

    for record in records:
        per_dataset.setdefault(record.dataset, []).append(record)
        per_gt_type.setdefault(record.gt_type, []).append(record)

        stress = record.stress
        for pk in _stress_primary_keys_present(stress):
            if pk in primary_to_records:
                primary_to_records[pk].append(record)
        if isinstance(stress, dict):
            for key in ("G", "M", "V", "L"):
                val = stress.get(key)
                if not isinstance(val, list):
                    continue
                for item in val:
                    composite = f"{key} {str(item)}"
                    secondary_to_records[composite].append(record)

    observed_second_keys = set(secondary_to_records.keys())
    ordered_secondary_keys = _ordered_secondary_composite_keys(observed_second_keys)

    def _summarize_group(group_records: list[PredictionRecord]) -> dict[str, Any]:
        total = len(group_records)
        parsed = sum(1 for record in group_records if _has_parsed_answer(record))
        errors = sum(1 for record in group_records if record.error)
        mcq_records = [record for record in group_records if record.gt_type == "mcq"]
        mask_records = [record for record in group_records if record.gt_type == "mask"]
        bbox_records = [record for record in group_records if record.gt_type == "bbox"]
        mcq_correct = sum(1 for record in mcq_records if record.metrics.get("mcq_correct") is True)
        mask_correct = sum(1 for record in mask_records if record.metrics.get("mask_correct") is True)
        bbox_ious = [
            float(record.metrics["bbox_iou"])
            for record in bbox_records
            if record.metrics.get("bbox_iou") is not None
        ]
        bbox_ge_05 = sum(1 for iou in bbox_ious if iou >= 0.5)
        bbox_ge_095 = sum(1 for iou in bbox_ious if iou >= 0.95)
        bbox_iou_ge_0_5_ratio = (bbox_ge_05 / len(bbox_ious)) if bbox_ious else None
        bbox_iou_ge_0_95_ratio = (bbox_ge_095 / len(bbox_ious)) if bbox_ious else None
        return {
            "total_questions": total,
            "parsed_count": parsed,
            "unparsed_count": total - parsed,
            "error_count": errors,
            "unparsed_ratio": ((total - parsed) / total) if total else 0.0,
            "mcq_count": len(mcq_records),
            "mcq_correct_count": mcq_correct,
            "mcq_accuracy": (mcq_correct / len(mcq_records)) if mcq_records else None,
            "mask_count": len(mask_records),
            "mask_correct_count": mask_correct,
            "mask_accuracy": (mask_correct / len(mask_records)) if mask_records else None,
            "bbox_count": len(bbox_records),
            "bbox_mean_iou": (sum(bbox_ious) / len(bbox_ious)) if bbox_ious else None,
            "bbox_iou_ge_0_5_ratio": bbox_iou_ge_0_5_ratio,
            "bbox_iou_ge_0_95_ratio": bbox_iou_ge_0_95_ratio,
        }

    def _total_acc(bucket: list[PredictionRecord]) -> float:
        if not bucket:
            return 0.0
        return sum(_sample_score(record) for record in bucket) / len(bucket)

    def _stress_stats(bucket: list[PredictionRecord]) -> dict[str, Any]:
        base = _summarize_group(bucket)
        base["stress_bucket_size"] = len(bucket)
        if len(bucket) == 0:
            base["total_acc"] = None
        else:
            base["total_acc"] = _stress_breakdown_mean_score(bucket)
        return base

    base_overall = _summarize_group(records)
    base_overall["total_acc"] = _total_acc(records)

    per_1st: dict[str, Any] = {}
    for key in ("G", "M", "V", "L"):
        per_1st[key] = _stress_stats(primary_to_records[key])

    per_2nd: dict[str, Any] = {}
    for tag in ordered_secondary_keys:
        per_2nd[tag] = _stress_stats(secondary_to_records.get(tag, []))

    overall_mcq_n = base_overall.get("mcq_count") or 0
    overall_mask_n = base_overall.get("mask_count") or 0
    overall_bbox_n = base_overall.get("bbox_count") or 0

    return {
        "overall_statistics": {
            **base_overall,
            "total_questions": total_questions,
            "parsed_count": parsed_count,
            "unparsed_count": unparsed_count,
            "error_count": error_count,
            "unparsed_by_gt_type": {
                "mcq": mcq_unparsed,
                "mask": mask_unparsed,
                "bbox": bbox_unparsed,
            },
            "unparsed_ratio_by_gt_type": {
                "mcq": (mcq_unparsed / overall_mcq_n) if overall_mcq_n else 0.0,
                "mask": (mask_unparsed / overall_mask_n) if overall_mask_n else 0.0,
                "bbox": (bbox_unparsed / overall_bbox_n) if overall_bbox_n else 0.0,
            },
        },
        "per_gt_type_statistics": {
            gt_type: _summarize_group(group_records)
            for gt_type, group_records in sorted(per_gt_type.items())
        },
        "per_dataset_statistics": {
            dataset: _summarize_group(group_records)
            for dataset, group_records in sorted(per_dataset.items())
        },
        "per_1st_level_stress": per_1st,
        "per_2nd_level_stress": per_2nd,
    }


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

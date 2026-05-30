from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import AppConfig, ConfigError

_LOGGER = logging.getLogger(__name__)

_GT_ORDER = ("mcq", "bbox", "mask")


@dataclass(frozen=True)
class QuestionSample:
    sample_id: str
    question_id: int | str
    dataset: str
    subcategory: str
    task: str
    gt_type: str
    image_rel_path: str
    image_abs_path: str
    instruction: str
    ground_truth: dict[str, Any]
    record_relpath: str | None = None
    mask_relpath: str | None = None
    mask_abs_path: str | None = None
    question_text: str = ""
    options: list[str] | None = None
    answer: str | None = None
    answer_index: int | None = None
    stress: Any | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_relpath(root: Path, relpath: str, field_name: str) -> Path:
    if not isinstance(relpath, str) or not relpath.strip():
        raise ConfigError(f"{field_name} must be a non-empty relative path.")
    path = Path(relpath)
    if path.is_absolute():
        raise ConfigError(f"{field_name} must be relative to dataset_root: {relpath}")
    return (root / path).resolve()


def _validate_record_payload(record_payload: Any, record_path: Path) -> dict[str, Any]:
    if not isinstance(record_payload, dict):
        raise ConfigError(f"{record_path} must contain a JSON object.")
    annotation = record_payload.get("annotation")
    if not isinstance(annotation, dict):
        raise ConfigError(f"{record_path} is missing annotation object.")
    return annotation


def _normalize_mcq_gt(gt_payload: dict[str, Any], record_path: Path) -> tuple[dict[str, Any], int, list[str]]:
    mcq_payload = gt_payload.get("mcq")
    if not isinstance(mcq_payload, dict):
        raise ConfigError(f"{record_path} has gt.type=mcq but no gt.mcq object.")
    options = mcq_payload.get("options")
    answer_index = mcq_payload.get("answer_index")
    if not isinstance(options, list) or not options or not all(isinstance(option, str) for option in options):
        raise ConfigError(f"{record_path} gt.mcq.options must be a non-empty string list.")
    if not isinstance(answer_index, int) or answer_index < 0 or answer_index >= len(options):
        raise ConfigError(f"{record_path} gt.mcq.answer_index is invalid.")
    return {"type": "mcq", "mcq": {"options": options, "answer_index": answer_index}}, answer_index, options


def _normalize_bbox_gt(gt_payload: dict[str, Any], record_path: Path) -> dict[str, Any]:
    bbox_payload = gt_payload.get("bbox")
    if not isinstance(bbox_payload, dict):
        raise ConfigError(f"{record_path} has gt.type=bbox but no gt.bbox object.")
    box = bbox_payload.get("box")
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) for value in box):
        raise ConfigError(f"{record_path} gt.bbox.box must contain four numbers.")
    return {
        "type": "bbox",
        "bbox": {
            "format": bbox_payload.get("format", "xyxy"),
            "space": bbox_payload.get("space", "permille"),
            "box": [float(value) for value in box],
        },
    }


def _normalize_mask_gt(gt_payload: dict[str, Any], dataset_root: Path, record_path: Path) -> tuple[dict[str, Any], str, str]:
    mask_payload = gt_payload.get("mask")
    if not isinstance(mask_payload, dict):
        raise ConfigError(f"{record_path} has gt.type=mask but no gt.mask object.")
    mask_relpath = mask_payload.get("path")
    mask_abs_path = _resolve_relpath(dataset_root, mask_relpath, f"{record_path} gt.mask.path")
    if not mask_abs_path.exists():
        raise ConfigError(f"Mask file not found for {record_path}: {mask_abs_path}")
    return (
        {
            "type": "mask",
            "mask": {
                "path": mask_relpath,
                "encoding": mask_payload.get("encoding"),
            },
        },
        mask_relpath,
        str(mask_abs_path),
    )


def _sample_from_manifest_entry(
    manifest_entry: dict[str, Any],
    dataset_root: Path,
) -> QuestionSample:
    record_relpath = manifest_entry.get("record_relpath")
    record_path = _resolve_relpath(dataset_root, record_relpath, "manifest.record_relpath")
    if not record_path.exists():
        raise ConfigError(f"Record file not found: {record_path}")

    record_payload = _read_json(record_path)
    annotation = _validate_record_payload(record_payload, record_path)
    instruction = annotation.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ConfigError(f"{record_path} annotation.instruction must be a non-empty string.")

    rgb_relpath = annotation.get("rgb") or manifest_entry.get("rgb_relpath")
    image_abs_path = _resolve_relpath(dataset_root, rgb_relpath, f"{record_path} annotation.rgb")
    if not image_abs_path.exists():
        raise ConfigError(f"Image file not found for {record_path}: {image_abs_path}")

    gt_payload = annotation.get("gt")
    if not isinstance(gt_payload, dict):
        raise ConfigError(f"{record_path} is missing annotation.gt object.")
    gt_type = gt_payload.get("type") or manifest_entry.get("gt_type")
    if gt_type not in {"mcq", "bbox", "mask"}:
        raise ConfigError(f"{record_path} has unsupported gt.type: {gt_type!r}")

    stress = annotation.get("stress")

    answer_index: int | None = None
    options: list[str] | None = None
    answer: str | None = None
    mask_relpath: str | None = None
    mask_abs_path: str | None = None
    if gt_type == "mcq":
        ground_truth, answer_index, options = _normalize_mcq_gt(gt_payload, record_path)
        answer = chr(ord("A") + answer_index) if answer_index < 26 else str(answer_index)
    elif gt_type == "bbox":
        ground_truth = _normalize_bbox_gt(gt_payload, record_path)
    else:
        ground_truth, mask_relpath, mask_abs_path = _normalize_mask_gt(gt_payload, dataset_root, record_path)

    dataset = str(manifest_entry.get("dataset") or annotation.get("provenance", {}).get("dataset") or "")
    subcategory = str(manifest_entry.get("subcategory") or annotation.get("provenance", {}).get("subcategory") or "")
    task = str(manifest_entry.get("task") or annotation.get("task") or "")
    sample_id = str(manifest_entry.get("sample_id") or annotation.get("sample_id") or record_path.stem)

    return QuestionSample(
        sample_id=sample_id,
        question_id=sample_id,
        dataset=dataset,
        subcategory=subcategory,
        task=task,
        gt_type=str(gt_type),
        image_rel_path=str(rgb_relpath),
        image_abs_path=str(image_abs_path),
        instruction=instruction.strip(),
        ground_truth=ground_truth,
        record_relpath=str(record_relpath),
        mask_relpath=mask_relpath,
        mask_abs_path=mask_abs_path,
        question_text=instruction.strip(),
        options=options,
        answer=answer,
        answer_index=answer_index,
        stress=stress,
    )


def discover_available_datasets(app_config: AppConfig) -> list[str]:
    if app_config.run_settings.dataset_root is None or app_config.run_settings.manifest_path is None:
        raise ConfigError("dataset_root and manifest_path are required for stress_bench mode.")
    manifest_path = app_config.run_settings.dataset_root / app_config.run_settings.manifest_path
    if not manifest_path.exists():
        raise ConfigError(f"Manifest file not found: {manifest_path}")
    datasets: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and isinstance(payload.get("dataset"), str):
                datasets.add(payload["dataset"])
    return sorted(datasets)


def _iter_filtered_samples(
    manifest_path: Path,
    materialized_root: Path,
    dataset_filter: set[str] | None,
    gt_type_filter: set[str] | None,
) -> Iterator[QuestionSample]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                manifest_entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{manifest_path}:{line_number} contains invalid JSON.") from exc
            if not isinstance(manifest_entry, dict):
                raise ConfigError(f"{manifest_path}:{line_number} must contain a JSON object.")
            if dataset_filter is not None and manifest_entry.get("dataset") not in dataset_filter:
                continue
            if gt_type_filter is not None and manifest_entry.get("gt_type") not in gt_type_filter:
                continue
            yield _sample_from_manifest_entry(manifest_entry, materialized_root)


def _stratified_balanced_pick(samples: list[QuestionSample], limit: int, sample_seed: int) -> list[QuestionSample]:
    buckets: dict[str, list[QuestionSample]] = {gt: [] for gt in _GT_ORDER}
    for sample in samples:
        if sample.gt_type not in buckets:
            raise ConfigError(f"Unexpected gt_type in stratified sampling: {sample.gt_type!r}")
        buckets[sample.gt_type].append(sample)

    active_types = [gt for gt in _GT_ORDER if buckets.get(gt)]
    if not active_types:
        return []

    n_types = len(active_types)
    quota = limit // n_types
    rng = random.Random(sample_seed)
    selected: list[QuestionSample] = []
    for gt in active_types:
        pool = list(buckets[gt])
        rng.shuffle(pool)
        take = min(quota, len(pool))
        selected.extend(pool[:take])

    if len(selected) < limit:
        _LOGGER.warning(
            "Stratified balanced sampling produced %s samples (requested %s); "
            "at least one gt_type had fewer than quota=%s items.",
            len(selected),
            limit,
            quota,
        )
    return selected


def load_stress_bench_samples(
    app_config: AppConfig,
    selected_datasets: str | list[str],
    selected_gt_types: str | list[str],
    limit: int | None = None,
    stratified: bool = False,
    sample_seed: int = 42,
) -> list[QuestionSample]:
    dataset_root = app_config.run_settings.dataset_root
    manifest_relpath = app_config.run_settings.manifest_path
    if dataset_root is None or manifest_relpath is None:
        raise ConfigError("dataset_root and manifest_path are required for stress_bench mode.")
    if not dataset_root.exists():
        raise ConfigError(f"Dataset root not found: {dataset_root}")

    manifest_path = dataset_root / manifest_relpath
    if not manifest_path.exists():
        raise ConfigError(f"Manifest file not found: {manifest_path}")
    materialized_root = manifest_path.parent

    dataset_filter = None if selected_datasets == "all" else set(selected_datasets)
    gt_type_filter = None if selected_gt_types == "all" else set(selected_gt_types)

    if stratified:
        if limit is None:
            raise ConfigError("stratified_balanced_sample requires run_limit or --limit to be set.")
        all_samples = list(
            _iter_filtered_samples(manifest_path, materialized_root, dataset_filter, gt_type_filter)
        )
        if not all_samples:
            raise ConfigError("No StressBench samples matched the selected filters.")
        return _stratified_balanced_pick(all_samples, limit, sample_seed)

    samples: list[QuestionSample] = []
    for sample in _iter_filtered_samples(manifest_path, materialized_root, dataset_filter, gt_type_filter):
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise ConfigError("No StressBench samples matched the selected filters.")
    return samples


def load_evaluation_samples(
    app_config: AppConfig,
    selected_datasets: str | list[str] = "all",
    selected_gt_types: str | list[str] = "all",
    limit: int | None = None,
    stratified: bool = False,
    sample_seed: int = 42,
) -> list[QuestionSample]:
    return load_stress_bench_samples(
        app_config,
        selected_datasets=selected_datasets,
        selected_gt_types=selected_gt_types,
        limit=limit,
        stratified=stratified,
        sample_seed=sample_seed,
    )

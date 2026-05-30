from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .dataset import QuestionSample


def _resolve_bbox_pred_space(parsed_answer: dict[str, Any], gt_space: str) -> str:
    """Coordinate space for model bbox outputs before normalization."""
    explicit = parsed_answer.get("space")
    if explicit is not None:
        return str(explicit)
    box = parsed_answer.get("box") or []
    if len(box) != 4:
        return str(gt_space)
    floats = [float(value) for value in box]
    mx = max(abs(value) for value in floats)
    # Older evaluations assumed normalized [0, 1] predictions vs permille GT.
    if str(gt_space) == "permille" and mx <= 1.0:
        return "fraction"
    return str(gt_space)


def _normalize_bbox(box: list[float], space: str) -> list[float]:
    """Express bbox corners on a ``[0, 1]`` image-axis scale for IoU.

    StressBench GT stores corners as **千分制 (permille)**: each coordinate is on ``[0, 1000]``
    along width/height (``gt.bbox.space`` is ``permille``). That convention is unchanged.

    For ``space == "permille"``, we apply ``value / 1000`` so GT and predictions sit on the same
    normalized axes inside ``_bbox_iou``. IoU is scale-invariant; this matches computing IoU in
    raw permille space.

    Clamp to ``[0, 1000]`` before dividing only guards malformed model outputs; clean GT from
    records is already valid permille.

    Other ``space`` values pass floats through as ``[0, 1]`` axis coordinates (legacy preds).
    """
    if space == "permille":
        return [max(0.0, min(1000.0, float(value))) / 1000.0 for value in box]
    return [float(value) for value in box]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _resolve_point_space(parsed_answer: dict[str, Any]) -> str:
    explicit = parsed_answer.get("space")
    if explicit is not None:
        return str(explicit)
    point = parsed_answer.get("point") or []
    if len(point) != 2:
        return "permille"
    floats = [float(value) for value in point]
    if max(abs(value) for value in floats) <= 1.0:
        return "fraction"
    return "permille"


def _normalize_point(point: list[float], space: str) -> list[float]:
    if space == "permille":
        return [max(0.0, min(1000.0, float(value))) / 1000.0 for value in point]
    return [float(value) for value in point]


def _bbox_iou(pred_box: list[float], gt_box: list[float]) -> float:
    px1, py1, px2, py2 = [_clip01(float(value)) for value in pred_box]
    gx1, gy1, gx2, gy2 = [_clip01(float(value)) for value in gt_box]

    if px2 < px1:
        px1, px2 = px2, px1
    if py2 < py1:
        py1, py2 = py2, py1
    if gx2 < gx1:
        gx1, gx2 = gx2, gx1
    if gy2 < gy1:
        gy1, gy2 = gy2, gy1

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _mask_point_hit(sample: QuestionSample, point: list[float]) -> bool:
    if sample.mask_abs_path is None:
        return False
    x_norm = _clip01(float(point[0]))
    y_norm = _clip01(float(point[1]))
    with Image.open(Path(sample.mask_abs_path)) as mask_image:
        mask = mask_image.convert("L")
        width, height = mask.size
        x = min(width - 1, max(0, int(round(x_norm * (width - 1)))))
        y = min(height - 1, max(0, int(round(y_norm * (height - 1)))))
        return mask.getpixel((x, y)) > 0


def compute_metrics(sample: QuestionSample, parsed_answer: dict[str, Any] | None) -> dict[str, Any]:
    metrics = {
        "mcq_correct": None,
        "mask_correct": None,
        "bbox_iou": None,
    }

    if sample.gt_type == "mcq":
        answer_index = sample.answer_index
        if answer_index is None:
            answer_index = sample.ground_truth.get("mcq", {}).get("answer_index")
        metrics["mcq_correct"] = (
            parsed_answer is not None
            and parsed_answer.get("kind") == "mcq"
            and parsed_answer.get("answer_index") == answer_index
        )
        return metrics

    if sample.gt_type == "bbox":
        if parsed_answer is None or parsed_answer.get("kind") != "bbox":
            metrics["bbox_iou"] = 0.0
            return metrics
        gt_payload = sample.ground_truth.get("bbox", {})
        gt_space = gt_payload.get("space", "permille")
        gt_box = _normalize_bbox(gt_payload.get("box", []), str(gt_space))
        if len(gt_box) != 4:
            metrics["bbox_iou"] = 0.0
            return metrics
        pred_space = _resolve_bbox_pred_space(parsed_answer, str(gt_space))
        pred_box = _normalize_bbox(parsed_answer.get("box", []), pred_space)
        if len(pred_box) != 4:
            metrics["bbox_iou"] = 0.0
            return metrics
        metrics["bbox_iou"] = _bbox_iou(pred_box, gt_box)
        return metrics

    if sample.gt_type == "mask":
        metrics["mask_correct"] = (
            parsed_answer is not None
            and parsed_answer.get("kind") == "mask"
            and _mask_point_hit(
                sample,
                _normalize_point(
                    parsed_answer.get("point", []),
                    _resolve_point_space(parsed_answer),
                ),
            )
        )
        return metrics

    return metrics

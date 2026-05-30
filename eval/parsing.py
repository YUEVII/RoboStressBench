from __future__ import annotations

import ast
import re
from typing import Any

from .dataset import QuestionSample

_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_LETTER_CUE_RE = re.compile(r"\b(?:answer|option|choice)\b[^A-Z]*([A-Z])\b", re.IGNORECASE)
_SINGLE_LETTER_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def _parse_number_sequence(raw_text: str | None, expected_len: int) -> list[float] | None:
    if raw_text is None:
        return None
    text = raw_text.strip()
    if not text:
        return None

    candidates = [text]
    bracket_match = re.search(r"(\[[^\]]+\]|\([^\)]+\))", text)
    if bracket_match:
        candidates.insert(0, bracket_match.group(1))

    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            value = None
        if isinstance(value, (list, tuple)) and len(value) == expected_len:
            try:
                numbers = [float(item) for item in value]
            except (TypeError, ValueError):
                numbers = []
            if len(numbers) == expected_len:
                return numbers

    matches = [float(match.group(0)) for match in _NUMBER_RE.finditer(text)]
    if len(matches) >= expected_len:
        return matches[:expected_len]
    return None


def _parse_mcq_index(raw_text: str | None, option_count: int) -> int | None:
    if raw_text is None:
        return None
    text = raw_text.strip()
    if not text:
        return None

    compact = text.upper()
    if len(compact) == 1 and "A" <= compact <= "Z":
        index = ord(compact) - ord("A")
        if 0 <= index < option_count:
            return index

    cue_match = _LETTER_CUE_RE.search(text)
    if cue_match:
        index = ord(cue_match.group(1).upper()) - ord("A")
        if 0 <= index < option_count:
            return index

    matches = _SINGLE_LETTER_RE.findall(text)
    if len(matches) == 1:
        index = ord(matches[0].upper()) - ord("A")
        if 0 <= index < option_count:
            return index

    first_line = text.splitlines()[0].strip()
    if first_line and "A" <= first_line[0].upper() <= "Z":
        index = ord(first_line[0].upper()) - ord("A")
        if 0 <= index < option_count:
            return index

    numeric_match = re.search(r"\b(\d+)\b", text)
    if numeric_match:
        value = int(numeric_match.group(1))
        if 1 <= value <= option_count:
            return value - 1
        if 0 <= value < option_count:
            return value
    return None


def parse_answer(sample: QuestionSample, raw_text: str | None) -> dict[str, Any] | None:
    if sample.gt_type == "mcq":
        options = sample.options or sample.ground_truth.get("mcq", {}).get("options", [])
        option_count = len(options)
        parsed_index = _parse_mcq_index(raw_text, option_count)
        if parsed_index is None:
            return None
        return {
            "kind": "mcq",
            "answer_index": parsed_index,
            "answer": chr(ord("A") + parsed_index) if parsed_index < 26 else str(parsed_index),
        }

    if sample.gt_type == "bbox":
        box = _parse_number_sequence(raw_text, expected_len=4)
        if box is None:
            return None
        bbox_gt = sample.ground_truth.get("bbox") or {}
        pred_space = str(bbox_gt.get("space", "permille"))
        return {"kind": "bbox", "box": box, "space": pred_space}

    if sample.gt_type == "mask":
        point = _parse_number_sequence(raw_text, expected_len=2)
        if point is None:
            return None
        return {"kind": "mask", "point": point, "space": "permille"}

    return None

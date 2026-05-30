from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import torch

from ..dataset import QuestionSample
from ..prompting import get_model_prompt
from .base import BaseLocalHFAdapter


class InternVLAdapter(BaseLocalHFAdapter):
    _IGNORED_IMAGE_INPUT_KEYS = {"num_patches"}

    def _get_padding_side_for_generation(self) -> str | None:
        return "left"

    def _load_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(
            self.context.model_config.ckpt_path,
            trust_remote_code=self.context.model_config.trust_remote_code,
        )

    def _load_model(self):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            self.context.model_config.ckpt_path,
            dtype=self._resolve_dtype(),
            device_map=self._resolve_device_map(),
            trust_remote_code=self.context.model_config.trust_remote_code,
        )

    def _configure_padding_side(self) -> None:
        padding_side = self._get_padding_side_for_generation()
        tokenizer = self._get_tokenizer()
        if padding_side is not None and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = padding_side
        if padding_side is not None and hasattr(self.processor, "padding_side"):
            self.processor.padding_side = padding_side

    def _get_image_processor(self):
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("InternVL processor does not expose image_processor for cacheable preprocessing.")
        return image_processor

    def _get_cache_dir(self) -> Path | None:
        return self.context.model_config.cache_path

    def _get_cache_file_path(self, sample: QuestionSample) -> Path | None:
        cache_dir = self._get_cache_dir()
        if cache_dir is None:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(sample.image_abs_path)
        stat = image_path.stat()
        cache_key = hashlib.sha1(
            f"{image_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        return cache_dir / f"{cache_key}.pt"

    @staticmethod
    def _normalize_cache_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if torch.is_tensor(value):
                normalized[key] = value.detach().cpu()
            else:
                normalized[key] = value
        return normalized

    def _prepare_single_image_inputs(self, image) -> dict[str, Any]:
        image_inputs = self._get_image_processor()(images=[image], crop_to_patches=True, return_tensors="pt")
        return self._normalize_cache_payload(dict(image_inputs))

    @staticmethod
    def _extract_num_patches(cached_item: dict[str, Any]) -> int:
        num_patches = cached_item.get("num_patches")
        if isinstance(num_patches, int):
            return num_patches
        if isinstance(num_patches, (list, tuple)) and num_patches:
            return int(num_patches[0])
        if torch.is_tensor(num_patches):
            if num_patches.numel() == 0:
                raise RuntimeError("InternVL cached num_patches tensor is empty.")
            return int(num_patches.reshape(-1)[0].item())
        raise RuntimeError(f"Unsupported InternVL num_patches payload: {type(num_patches)!r}")

    def _load_or_prepare_image_inputs(self, sample: QuestionSample, image) -> dict[str, Any]:
        cache_file_path = self._get_cache_file_path(sample)
        if cache_file_path is not None and cache_file_path.exists():
            return torch.load(cache_file_path, map_location="cpu", weights_only=False)

        image_inputs = self._prepare_single_image_inputs(image)
        if cache_file_path is not None:
            torch.save(image_inputs, cache_file_path)
        return image_inputs

    def _collate_cached_image_inputs(self, cached_items: list[dict[str, Any]]) -> dict[str, Any]:
        if not cached_items:
            return {}

        collated: dict[str, Any] = {}
        for key in cached_items[0]:
            if key in self._IGNORED_IMAGE_INPUT_KEYS:
                continue
            values = [item[key] for item in cached_items]
            first_value = values[0]
            if torch.is_tensor(first_value):
                try:
                    collated[key] = torch.cat(values, dim=0)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Failed to collate cached image input '{key}' for InternVL. "
                        f"Shapes={[tuple(value.shape) for value in values]}"
                    ) from exc
            elif isinstance(first_value, list):
                merged: list[Any] = []
                for value in values:
                    merged.extend(value)
                collated[key] = merged
            else:
                collated[key] = values
        return collated

    def _expand_prompt_with_image_tokens(self, prompt: str, num_patches: int) -> str:
        image_token = getattr(self.processor, "image_token", None)
        start_image_token = getattr(self.processor, "start_image_token", None)
        end_image_token = getattr(self.processor, "end_image_token", None)
        image_seq_length = getattr(self.processor, "image_seq_length", None)
        if not all(value is not None for value in (image_token, start_image_token, end_image_token, image_seq_length)):
            raise RuntimeError("InternVL processor is missing image token metadata required for cached prompts.")
        replacement = f"{start_image_token}{image_token * (int(image_seq_length) * num_patches)}{end_image_token}"
        if image_token not in prompt:
            raise RuntimeError("InternVL prompt is missing image placeholder token required for multimodal generation.")
        return prompt.replace(image_token, replacement, 1)

    def _prepare_text_inputs(self, prompts: list[str], cached_items: list[dict[str, Any]]) -> dict[str, Any]:
        self._configure_padding_side()
        text_inputs = [
            self._expand_prompt_with_image_tokens(self._build_text_input(prompt), self._extract_num_patches(cached_item))
            for prompt, cached_item in zip(prompts, cached_items)
        ]
        tokenizer = self._get_tokenizer()
        return dict(tokenizer(text_inputs, padding=True, return_tensors="pt"))

    def generate_answers(self, samples: list[QuestionSample]) -> list[str]:
        if not samples:
            return []

        self._profiling_stats["batch_count"] += 1
        self._profiling_stats["sample_count"] += len(samples)

        load_images_start = time.perf_counter()
        images = self._load_images(samples)
        self._profiling_stats["load_images_seconds"] += time.perf_counter() - load_images_start

        prepare_inputs_start = time.perf_counter()
        prompt_texts = [get_model_prompt(sample) for sample in samples]
        cached_items = [
            self._load_or_prepare_image_inputs(sample, image)
            for sample, image in zip(samples, images)
        ]
        del images
        text_batch = self._prepare_text_inputs(prompt_texts, cached_items)
        image_batch = self._collate_cached_image_inputs(cached_items)
        overlapping_keys = set(text_batch) & set(image_batch)
        if overlapping_keys:
            raise RuntimeError(f"InternVL text/image inputs have overlapping keys: {sorted(overlapping_keys)}")
        inputs = self._move_inputs_to_device({**text_batch, **image_batch})
        self._profiling_stats["prepare_inputs_seconds"] += time.perf_counter() - prepare_inputs_start

        generation_kwargs = self._build_generation_kwargs()
        generate_start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        self._profiling_stats["generate_seconds"] += time.perf_counter() - generate_start

        decode_start = time.perf_counter()
        decoded_outputs = self._decode_outputs(generated_ids, inputs)
        self._profiling_stats["decode_seconds"] += time.perf_counter() - decode_start
        self._release_generation_tensors(inputs, generated_ids)
        return decoded_outputs

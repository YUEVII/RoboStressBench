from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..dataset import QuestionSample
from ..prompting import get_model_prompt
from .base import BaseLocalHFAdapter


class Molmo2Adapter(BaseLocalHFAdapter):
    def _get_padding_side_for_generation(self) -> str | None:
        return "left"

    def _load_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(
            self.context.model_config.ckpt_path,
            trust_remote_code=True,
            dtype="auto",
            device_map=self._resolve_device_map(),
        )

    def _load_model(self):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            self.context.model_config.ckpt_path,
            dtype=self._resolve_dtype(),
            device_map=self._resolve_device_map(),
            trust_remote_code=True,
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
            raise RuntimeError("Molmo2 processor does not expose image_processor for cacheable preprocessing.")
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
        image_inputs = self._get_image_processor()(images=[image], return_tensors="pt")
        return self._normalize_cache_payload(dict(image_inputs))

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
            values = [item[key] for item in cached_items]
            first_value = values[0]
            if torch.is_tensor(first_value):
                try:
                    collated[key] = torch.cat(values, dim=0)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Failed to collate cached image input '{key}' for Molmo2. "
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

    def _expand_prompt_with_image_tokens(self, prompt: str, cached_item: dict[str, Any]) -> str:
        image_placeholder_token = getattr(self.processor, "image_placeholder_token", None)
        if image_placeholder_token is None:
            raise RuntimeError("Molmo2 processor is missing image_placeholder_token.")
        image_grids = cached_item.get("image_grids")
        if image_grids is None:
            raise RuntimeError("Molmo2 cached image input is missing image_grids.")

        if torch.is_tensor(image_grids):
            image_grids = image_grids.detach().cpu().numpy()
        image_grids = np.asarray(image_grids)
        if image_grids.ndim == 1:
            image_grids = image_grids[None, :]

        expanded_prompt = prompt
        for image_grid in image_grids:
            image_tokens = self.processor.get_image_tokens(np.asarray(image_grid))
            image_string = "".join(image_tokens.tolist() if hasattr(image_tokens, "tolist") else image_tokens)
            if image_placeholder_token not in expanded_prompt:
                raise RuntimeError("Molmo2 prompt is missing image placeholder token required for multimodal generation.")
            expanded_prompt = expanded_prompt.replace(image_placeholder_token, image_string, 1)
        return expanded_prompt

    def _prepare_text_inputs(self, prompts: list[str], cached_items: list[dict[str, Any]]) -> dict[str, Any]:
        self._configure_padding_side()
        tokenizer = self._get_tokenizer()
        expanded_prompts = [
            self._expand_prompt_with_image_tokens(self._build_text_input(prompt), cached_item)
            for prompt, cached_item in zip(prompts, cached_items)
        ]

        text_inputs = tokenizer(expanded_prompts, padding=True, return_tensors="np")
        input_ids = np.array(text_inputs["input_ids"])
        attention_mask = np.array(text_inputs["attention_mask"])

        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        input_ids, attention_mask = self.processor.insert_bos(
            input_ids,
            attention_mask,
            bos_token_id,
            tokenizer.pad_token_id,
        )

        image_token_ids = np.array(self.processor.image_token_ids).astype(input_ids.dtype)
        token_type_ids = np.any(input_ids[:, :, None] == image_token_ids[None, None, :], axis=-1)

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "token_type_ids": torch.from_numpy(token_type_ids),
        }

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
            raise RuntimeError(f"Molmo2 text/image inputs have overlapping keys: {sorted(overlapping_keys)}")
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

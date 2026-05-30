from __future__ import annotations

import copy
import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from PIL import Image

LOGGER = logging.getLogger(__name__)


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        torch_mod = __import__("torch")
        oom_cls = getattr(torch_mod.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    except Exception:
        pass
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False

from ..config import ModelConfig, RunSettings
from ..dataset import QuestionSample
from ..prompting import get_model_prompt
from ..runtime import configure_runtime_noise


@dataclass(frozen=True)
class AdapterContext:
    model_config: ModelConfig
    run_settings: RunSettings
    gpu_id: int | None
    dry_run: bool = False


@dataclass(frozen=True)
class GenerationResult:
    raw_response: str | None
    error: str | None = None


class BaseModelAdapter(ABC):
    def __init__(self, context: AdapterContext) -> None:
        self.context = context

    @abstractmethod
    def generate_answer(self, sample: QuestionSample) -> str:
        raise NotImplementedError

    def generate_answers(self, samples: list[QuestionSample]) -> list[str]:
        return [self.generate_answer(sample) for sample in samples]

    def generate_batch_results(self, samples: list[QuestionSample]) -> list[GenerationResult]:
        return [GenerationResult(raw_response=raw_response) for raw_response in self.generate_answers(samples)]

    def get_profiling_stats(self) -> dict[str, float]:
        return {}


class APIPlaceholderAdapter(BaseModelAdapter):
    def generate_answer(self, sample: QuestionSample) -> str:
        raise NotImplementedError(
            f"API model '{self.context.model_config.name}' is registered but real API calls "
            "are intentionally not implemented in this iteration."
        )


class DryRunAdapter(BaseModelAdapter):
    def generate_answer(self, sample: QuestionSample) -> str:
        if sample.gt_type == "bbox":
            bbox_meta = sample.ground_truth.get("bbox", {})
            gt_box = list(bbox_meta.get("box", [100, 100, 900, 900]))
            space = bbox_meta.get("space", "permille")
            if space == "permille":
                return f"({gt_box[0]}, {gt_box[1]}, {gt_box[2]}, {gt_box[3]})"
            gt_norm = [float(value) for value in gt_box]
            return (
                f"({gt_norm[0]:.3f}, {gt_norm[1]:.3f}, {gt_norm[2]:.3f}, {gt_norm[3]:.3f})"
            )
        if sample.gt_type == "mask":
            return "(500, 500)"
        letters = ["A", "B", "C", "D"]
        index = hash((self.context.model_config.name, sample.question_id)) % len(letters)
        return letters[index]


class BaseLocalHFAdapter(BaseModelAdapter):
    def __init__(self, context: AdapterContext) -> None:
        super().__init__(context)
        self.model = None
        self.processor = None
        self.device = None
        self._profiling_stats = {
            "batch_count": 0.0,
            "sample_count": 0.0,
            "load_images_seconds": 0.0,
            "prepare_inputs_seconds": 0.0,
            "generate_seconds": 0.0,
            "decode_seconds": 0.0,
        }
        self._load()

    @staticmethod
    def _import_torch():
        import torch

        return torch

    def _resolve_dtype(self):
        torch = self._import_torch()
        dtype_name = self.context.model_config.torch_dtype or "bfloat16"
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported torch dtype in config: {dtype_name}")
        return getattr(torch, dtype_name)

    def _resolve_device_map(self) -> Any:
        if self.context.gpu_id is None:
            return None
        strategy = self.context.model_config.worker_strategy
        if strategy == "single_instance":
            return self.context.model_config.device_map
        return {"": f"cuda:{self.context.gpu_id}"}

    def _resolve_runtime_device(self):
        torch = self._import_torch()
        if self.context.gpu_id is not None:
            return torch.device(f"cuda:{self.context.gpu_id}")
        return torch.device("cpu")

    def _load_processor(self):
        raise NotImplementedError

    def _load_model(self):
        raise NotImplementedError

    def _load(self) -> None:
        configure_runtime_noise()
        self.processor = self._load_processor()
        self.model = self._load_model()
        self.device = self._resolve_runtime_device()

    def _build_messages(self, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _get_chat_template_kwargs(self) -> dict[str, Any]:
        return {}

    def _get_padding_side_for_generation(self) -> str | None:
        return None

    def _build_text_input(self, prompt: str) -> str:
        messages = self._build_messages(prompt)
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self._get_chat_template_kwargs(),
            )
        return prompt

    def _prepare_inputs(self, images: list[Image.Image], prompts: list[str]):
        padding_side = self._get_padding_side_for_generation()
        tokenizer = self._get_tokenizer()
        if padding_side is not None and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = padding_side
        if padding_side is not None and hasattr(self.processor, "padding_side"):
            self.processor.padding_side = padding_side

        text_inputs = [self._build_text_input(prompt) for prompt in prompts]
        inputs = self.processor(
            text=text_inputs,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        return self._move_inputs_to_device(inputs)

    def _decode_outputs(self, generated_ids, inputs) -> list[str]:
        if hasattr(generated_ids, "detach"):
            generated_ids = generated_ids.detach().cpu()

        prompt_width = None
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            prompt_width = input_ids.shape[-1]
        else:
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                prompt_width = attention_mask.shape[-1]

        if prompt_width is not None:
            generated_ids = generated_ids[:, prompt_width:]

        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return [item.strip() for item in decoded]

        return [str(item) for item in generated_ids]

    def _load_images(self, samples: list[QuestionSample]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for sample in samples:
            with Image.open(Path(sample.image_abs_path)) as pil_image:
                images.append(pil_image.convert("RGB"))
        return images

    def get_profiling_stats(self) -> dict[str, float]:
        return dict(self._profiling_stats)

    def _recover_after_cuda_oom(self) -> None:
        gc.collect()
        torch = self._import_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _release_generation_tensors(inputs: Any, generated_ids: Any) -> None:
        try:
            del inputs
            del generated_ids
        except Exception:
            pass

    def generate_batch_results(self, samples: list[QuestionSample]) -> list[GenerationResult]:
        if not samples:
            return []
        try:
            answers = self.generate_answers(samples)
            return [GenerationResult(raw_response=text) for text in answers]
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            self._recover_after_cuda_oom()
            if len(samples) == 1:
                LOGGER.warning(
                    "CUDA OOM during inference on a single sample; recording failure for this item (%s). "
                    "This sample cannot fit in current VRAM (resolution / prompt length / model size). "
                    "Optional: reduce max_new_tokens in YAML, lower image resolution upstream, "
                    "or PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.",
                    exc.__class__.__name__,
                )
                return [
                    GenerationResult(
                        raw_response=None,
                        error=f"{exc.__class__.__name__}: {exc}",
                    )
                ]
            LOGGER.warning(
                "CUDA OOM during inference (batch_size=%s); splitting batch and retrying (%s). "
                "Peak VRAM grows with image resolution, prompt length, and batch size; "
                "optional mitigations: lower model_registry batch_size or "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.",
                len(samples),
                exc.__class__.__name__,
            )
            mid = max(1, len(samples) // 2)
            left = self.generate_batch_results(samples[:mid])
            right = self.generate_batch_results(samples[mid:])
            return left + right

    def _move_inputs_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def _get_tokenizer(self):
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        return self.processor

    def _sync_special_token_ids(self, generation_config) -> None:
        tokenizer = self._get_tokenizer()
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)

        if getattr(generation_config, "pad_token_id", None) is None and pad_token_id is not None:
            generation_config.pad_token_id = pad_token_id
        if getattr(generation_config, "eos_token_id", None) is None and eos_token_id is not None:
            generation_config.eos_token_id = eos_token_id
        if getattr(generation_config, "bos_token_id", None) is None and bos_token_id is not None:
            generation_config.bos_token_id = bos_token_id

        if getattr(self.model, "generation_config", None) is not None:
            if getattr(self.model.generation_config, "pad_token_id", None) is None and pad_token_id is not None:
                self.model.generation_config.pad_token_id = pad_token_id
            if getattr(self.model.generation_config, "eos_token_id", None) is None and eos_token_id is not None:
                self.model.generation_config.eos_token_id = eos_token_id
            if getattr(self.model.generation_config, "bos_token_id", None) is None and bos_token_id is not None:
                self.model.generation_config.bos_token_id = bos_token_id

    def _build_generation_kwargs(self) -> dict[str, Any]:
        do_sample = self.context.run_settings.temperature > 0
        generation_config = None
        if getattr(self.model, "generation_config", None) is not None:
            generation_config = copy.deepcopy(self.model.generation_config)
            self._sync_special_token_ids(generation_config)
            generation_config.max_new_tokens = self.context.run_settings.max_new_tokens
            generation_config.do_sample = do_sample
            if do_sample:
                generation_config.temperature = self.context.run_settings.temperature
                generation_config.top_p = self.context.run_settings.top_p
            else:
                generation_config.temperature = 1.0
                generation_config.top_p = 1.0
                if hasattr(generation_config, "top_k"):
                    generation_config.top_k = 50

        if generation_config is not None:
            return {"generation_config": generation_config}

        generation_kwargs = {
            "max_new_tokens": self.context.run_settings.max_new_tokens,
            "do_sample": do_sample,
        }
        tokenizer = self._get_tokenizer()
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            generation_kwargs["temperature"] = self.context.run_settings.temperature
            generation_kwargs["top_p"] = self.context.run_settings.top_p
        return generation_kwargs

    def generate_answer(self, sample: QuestionSample) -> str:
        return self.generate_answers([sample])[0]

    def generate_answers(self, samples: list[QuestionSample]) -> list[str]:
        if not samples:
            return []

        torch = self._import_torch()
        prompts = [get_model_prompt(sample) for sample in samples]
        self._profiling_stats["batch_count"] += 1
        self._profiling_stats["sample_count"] += len(samples)

        load_images_start = time.perf_counter()
        images = self._load_images(samples)
        self._profiling_stats["load_images_seconds"] += time.perf_counter() - load_images_start

        prepare_inputs_start = time.perf_counter()
        inputs = self._prepare_inputs(images, prompts)
        del images
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

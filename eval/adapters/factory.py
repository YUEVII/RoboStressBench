from __future__ import annotations

from ..config import ModelConfig, RunSettings
from .base import APIPlaceholderAdapter, AdapterContext, BaseModelAdapter, DryRunAdapter


def create_adapter(
    model_config: ModelConfig,
    run_settings: RunSettings,
    gpu_id: int | None,
    dry_run: bool = False,
) -> BaseModelAdapter:
    context = AdapterContext(
        model_config=model_config,
        run_settings=run_settings,
        gpu_id=gpu_id,
        dry_run=dry_run,
    )
    if dry_run:
        return DryRunAdapter(context)
    if model_config.mode == "api":
        if model_config.provider == "openai":
            from .openai_api import OpenAIAPIAdapter

            return OpenAIAPIAdapter(context)
        if model_config.provider == "gemini":
            from .gemini_api import GeminiAPIAdapter

            return GeminiAPIAdapter(context)
        return APIPlaceholderAdapter(context)

    family = model_config.family
    if family == "qwen3_vl":
        from .qwen3_vl import Qwen3VLAdapter

        return Qwen3VLAdapter(context)
    if family == "qwen3_5":
        from .qwen3_5 import Qwen35Adapter

        return Qwen35Adapter(context)
    if family == "qwen3_6":
        from .qwen3_5 import Qwen35Adapter

        return Qwen35Adapter(context)
    if family == "molmo2":
        from .molmo2 import Molmo2Adapter

        return Molmo2Adapter(context)
    if family == "internvl":
        from .internvl import InternVLAdapter

        return InternVLAdapter(context)

    raise ValueError(f"Unsupported local model family: {family}")

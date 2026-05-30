from __future__ import annotations

from .base import BaseLocalHFAdapter


class Qwen3VLAdapter(BaseLocalHFAdapter):
    def _get_padding_side_for_generation(self) -> str | None:
        return "left"

    def _load_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(self.context.model_config.ckpt_path)

    def _load_model(self):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            self.context.model_config.ckpt_path,
            dtype=self._resolve_dtype(),
            device_map=self._resolve_device_map(),
            trust_remote_code=self.context.model_config.trust_remote_code,
        )

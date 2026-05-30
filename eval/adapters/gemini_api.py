from __future__ import annotations

from typing import Any

from .api_base import BaseAPIAdapter, PreparedImage


class GeminiAPIAdapter(BaseAPIAdapter):
    def _build_request(
        self,
        prompt: str,
        prepared_image: PreparedImage,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        api_key = self._get_api_key()
        model_name = self.context.model_config.api_model_name
        if not model_name:
            raise RuntimeError(
                f"API model '{self.context.model_config.name}' is missing api_model_name in config."
            )
        api_url = self.context.model_config.api_url
        if not api_url:
            raise RuntimeError(f"API model '{self.context.model_config.name}' is missing api_url in config.")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": prepared_image.mime_type,
                                "data": prepared_image.data_base64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.context.run_settings.temperature,
                "topP": self.context.run_settings.top_p,
                "maxOutputTokens": self.context.run_settings.max_new_tokens,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        return api_url, headers, payload

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(
                f"Gemini API model '{self.context.model_config.name}' returned no candidates: {payload}"
            )
        content = candidates[0].get("content", {})
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise RuntimeError(
                f"Gemini API model '{self.context.model_config.name}' returned malformed content: {payload}"
            )
        texts: list[str] = []
        for item in parts:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    texts.append(text_value)
        return "\n".join(texts).strip()

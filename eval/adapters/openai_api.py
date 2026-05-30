from __future__ import annotations

from typing import Any

from .api_base import BaseAPIAdapter, PreparedImage


class OpenAIAPIAdapter(BaseAPIAdapter):
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
            "model": model_name,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{prepared_image.mime_type};base64,{prepared_image.data_base64}",
                        },
                    ],
                }
            ],
            "max_output_tokens": self.context.run_settings.max_new_tokens,
            "temperature": self.context.run_settings.temperature,
            "top_p": self.context.run_settings.top_p,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return api_url, headers, payload

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            return output_text.strip()

        output = payload.get("output")
        if isinstance(output, list):
            texts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        texts.append(text_value)
            if texts:
                return "\n".join(texts).strip()

        raise RuntimeError(
            f"OpenAI API model '{self.context.model_config.name}' returned no text output: {payload}"
        )

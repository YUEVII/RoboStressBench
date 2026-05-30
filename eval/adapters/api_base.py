from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from ..dataset import QuestionSample
from ..prompting import get_model_prompt
from .base import BaseModelAdapter, GenerationResult


@dataclass(frozen=True)
class PreparedImage:
    data_base64: str
    mime_type: str


class BaseAPIAdapter(BaseModelAdapter):
    _DEFAULT_TIMEOUT_SECONDS = 60.0
    _DEFAULT_MAX_RETRIES = 3
    _DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
    _RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, context) -> None:
        super().__init__(context)
        self._profiling_stats = {
            "batch_count": 0.0,
            "sample_count": 0.0,
            "load_images_seconds": 0.0,
            "prepare_inputs_seconds": 0.0,
            "generate_seconds": 0.0,
            "decode_seconds": 0.0,
        }

    def get_profiling_stats(self) -> dict[str, float]:
        return dict(self._profiling_stats)

    def generate_answer(self, sample: QuestionSample) -> str:
        result = self.generate_batch_results([sample])[0]
        if result.error:
            raise RuntimeError(result.error)
        return result.raw_response or ""

    def generate_batch_results(self, samples: list[QuestionSample]) -> list[GenerationResult]:
        if not samples:
            return []

        self._profiling_stats["batch_count"] += 1
        self._profiling_stats["sample_count"] += len(samples)
        concurrency = max(
            1,
            min(len(samples), int(self.context.run_settings.max_concurrent_api_requests)),
        )

        if concurrency == 1:
            outputs = [self._generate_single_result(sample) for sample in samples]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                outputs = list(executor.map(self._generate_single_result, samples))

        for _, metrics in outputs:
            for key, value in metrics.items():
                self._profiling_stats[key] += float(value)
        return [result for result, _ in outputs]

    def _generate_single_result(
        self,
        sample: QuestionSample,
    ) -> tuple[GenerationResult, dict[str, float]]:
        metrics = {
            "load_images_seconds": 0.0,
            "prepare_inputs_seconds": 0.0,
            "generate_seconds": 0.0,
            "decode_seconds": 0.0,
        }
        try:
            prompt = get_model_prompt(sample)

            load_start = time.perf_counter()
            prepared_image = self._load_and_prepare_image(sample)
            metrics["load_images_seconds"] += time.perf_counter() - load_start

            prepare_start = time.perf_counter()
            url, headers, payload = self._build_request(prompt, prepared_image)
            body = json.dumps(payload).encode("utf-8")
            metrics["prepare_inputs_seconds"] += time.perf_counter() - prepare_start

            generate_start = time.perf_counter()
            response_payload = self._send_request_with_retries(url, headers, body)
            metrics["generate_seconds"] += time.perf_counter() - generate_start

            decode_start = time.perf_counter()
            raw_response = self._extract_response_text(response_payload)
            metrics["decode_seconds"] += time.perf_counter() - decode_start
            return GenerationResult(raw_response=raw_response or "", error=None), metrics
        except Exception as exc:
            return GenerationResult(raw_response=None, error=str(exc)), metrics

    def _load_and_prepare_image(self, sample: QuestionSample) -> PreparedImage:
        image_path = Path(sample.image_abs_path)
        image_bytes = image_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        return PreparedImage(
            data_base64=base64.b64encode(image_bytes).decode("ascii"),
            mime_type=mime_type,
        )

    def _get_timeout_seconds(self) -> float:
        return self._DEFAULT_TIMEOUT_SECONDS

    def _get_max_retries(self) -> int:
        return self._DEFAULT_MAX_RETRIES

    def _get_retry_backoff_seconds(self) -> float:
        return self._DEFAULT_RETRY_BACKOFF_SECONDS

    def _get_api_key(self) -> str:
        env_var = self.context.model_config.api_key_env_var
        if not env_var:
            raise RuntimeError(
                f"API model '{self.context.model_config.name}' is missing api_key_env_var in config."
            )
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{env_var}' is not set for API model '{self.context.model_config.name}'."
            )
        return api_key

    def _send_request_with_retries(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self._get_max_retries() + 1):
            try:
                return self._send_request(url, headers, body)
            except Exception as exc:
                last_error = exc
                if not self._should_retry(exc) or attempt >= self._get_max_retries():
                    break
                time.sleep(self._get_retry_backoff_seconds() * attempt)
        assert last_error is not None
        raise last_error

    def _send_request(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, Any]:
        req = request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self._get_timeout_seconds()) as response:
                response_bytes = response.read()
        except error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(self._build_http_error_message(exc.code, response_text)) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Network error while calling API model '{self.context.model_config.name}': {exc}") from exc

        try:
            return json.loads(response_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"API model '{self.context.model_config.name}' returned invalid JSON: "
                f"{response_bytes.decode('utf-8', errors='replace')[:500]}"
            ) from exc

    def _build_http_error_message(self, status_code: int, response_text: str) -> str:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            payload = None
        detail = self._extract_error_detail(payload) if isinstance(payload, dict) else response_text.strip()
        detail = detail or response_text.strip() or "unknown error"
        return f"HTTP {status_code} from API model '{self.context.model_config.name}': {detail}"

    @classmethod
    def _should_retry(cls, exc: Exception) -> bool:
        message = str(exc)
        for status_code in cls._RETRYABLE_STATUS_CODES:
            if f"HTTP {status_code} " in message:
                return True
        return "Network error while calling API model" in message

    @staticmethod
    def _extract_error_detail(payload: dict[str, Any]) -> str | None:
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            for key in ("message", "status", "code"):
                value = error_payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(error_payload, str) and error_payload.strip():
            return error_payload.strip()
        return None

    def _build_request(
        self,
        prompt: str,
        prepared_image: PreparedImage,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        raise NotImplementedError

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError

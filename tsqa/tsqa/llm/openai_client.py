"""OpenAI-compatible LLM client."""
from __future__ import annotations

import base64
import mimetypes
import random
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .artifacts import normalize_artifacts
from .base import BaseLLMClient


def _image_data_url(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 503}:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "429",
            "503",
            "rate limit",
            "resource_exhausted",
            "timeout",
            "timed out",
            "deadline",
            "connection",
            "temporarily unavailable",
            "service unavailable",
        )
    )


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        client: Any = None,
        timeout: float = 90.0,
        max_retries: int = 3,
        min_call_interval: float = 0.02,
    ):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.timeout = timeout
        self._max_retries = max_retries
        self._min_call_interval = min_call_interval
        self._call_lock = threading.Lock()
        self._last_call_time = 0.0
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("openai is required for OpenAIClient. Install it with: pip install openai") from e
            self._client = OpenAI(api_key=api_key, timeout=timeout)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[Any] = None,
        artifacts: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        norm_artifacts = normalize_artifacts(artifacts, image=image)
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for art in norm_artifacts:
            path = art.get("path")
            if path and Path(path).exists():
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                with self._call_lock:
                    elapsed = time.monotonic() - self._last_call_time
                    if elapsed < self._min_call_interval:
                        time.sleep(self._min_call_interval - elapsed)
                    self._last_call_time = time.monotonic()

                resp = self._client.chat.completions.create(
                    model=kwargs.get("model", self.model_name),
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content if len(content) > 1 else user_prompt},
                    ],
                    timeout=kwargs.get("timeout", self.timeout),
                )
                choices = getattr(resp, "choices", None) or []
                if not choices:
                    raise RuntimeError("OpenAI API call returned no choices")
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self._add_usage(
                        prompt=getattr(usage, "prompt_tokens", 0),
                        completion=getattr(usage, "completion_tokens", 0),
                        total=getattr(usage, "total_tokens", 0),
                    )
                return (choices[0].message.content or "").strip()
            except Exception as exc:
                if isinstance(exc, RuntimeError) and "no choices" in str(exc).lower():
                    raise
                last_error = exc
                if not _is_retryable(exc) or attempt >= self._max_retries:
                    # Preserve the original exception type (auth/schema/network)
                    # instead of collapsing everything into RuntimeError. (#10;
                    # mirrors GeminiClient.)
                    raise
                backoff = min(20.0, (2**attempt) * 2.0) + random.random()
                time.sleep(backoff)

        raise RuntimeError(f"OpenAI API call failed: {last_error}")


__all__ = ["OpenAIClient"]

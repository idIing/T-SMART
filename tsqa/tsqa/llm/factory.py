"""Provider factory for LLM clients."""
from __future__ import annotations

import os
from typing import Any

from .client import GeminiClient
from .openai_client import OpenAIClient


DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "openai": "gpt-4o-mini",
}


def create_llm_client(
    *,
    provider: str = "gemini",
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
):
    provider = provider.strip().lower()
    model_name = model or DEFAULT_MODELS.get(provider)
    if provider == "gemini":
        return GeminiClient(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
            model_name=model_name or DEFAULT_MODELS["gemini"],
            **kwargs,
        )
    if provider == "openai":
        return OpenAIClient(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model_name=model_name or DEFAULT_MODELS["openai"],
            **kwargs,
        )
    raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(DEFAULT_MODELS)}")


def add_provider_args(parser):
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="gemini")
    parser.add_argument("--model", default=None)
    return parser


__all__ = ["DEFAULT_MODELS", "create_llm_client", "add_provider_args"]

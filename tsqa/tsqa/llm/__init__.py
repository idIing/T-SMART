from .base import BaseLLMClient
from .client import GeminiClient
from .factory import create_llm_client

try:
    from .openai_client import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from .qwen_client import QwenClient
except ImportError:
    QwenClient = None  # torch/transformers not installed in this env

from .prompt import (
    build_answer_prompt,
    build_llm_only_prompt,
    build_llm_numeric_prompt,
    LLM_ONLY_SYSTEM_PROMPT,
    LLM_NUMERIC_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from .prompt_oneshot import build_oneshot_prompt
from .parser import parse_answer
from .fallback import build_fallback_prompt
from .vision import analyze_image
from .prompt_numeric import (
    build_numeric_prompt,
    build_categorical_prompt,
    parse_numeric_answer,
    parse_categorical_answer,
    build_numeric_planner_prompt,
    parse_numeric_plan_json,
)

__all__ = [
    "BaseLLMClient",
    "GeminiClient",
    "OpenAIClient",
    "QwenClient",
    "create_llm_client",
    "build_answer_prompt",
    "build_oneshot_prompt",
    "build_llm_only_prompt",
    "build_llm_numeric_prompt",
    "LLM_ONLY_SYSTEM_PROMPT",
    "LLM_NUMERIC_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "parse_answer",
    "build_fallback_prompt",
    "analyze_image",
    "build_numeric_prompt",
    "build_categorical_prompt",
    "parse_numeric_answer",
    "parse_categorical_answer",
    "build_numeric_planner_prompt",
    "parse_numeric_plan_json",
]

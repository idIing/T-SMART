import threading
from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseLLMClient(ABC):
    """
    Generic interface for all LLM backends.

    Every model backend should implement:
      - generate(system_prompt, user_prompt)
      - optional multimodal/image handling later
      - consistent raw text output
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Thread-safe API token accounting. Every subclass calls super().__init__,
        # so the accumulators exist on all backends. generate() implementations
        # call _add_usage() after a successful response; usage_snapshot() reads a
        # consistent copy. Used by the harnesses to persist real token cost into
        # the run metrics (the autonomous search's budget ledger).
        self._usage_lock = threading.Lock()
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def _add_usage(self, prompt=0, completion=0, total=0):
        """Accumulate one call's token usage (thread-safe)."""
        with self._usage_lock:
            self.total_prompt_tokens += int(prompt or 0)
            self.total_completion_tokens += int(completion or 0)
            self.total_tokens += int(total or 0)
            self.total_calls += 1

    def usage_snapshot(self) -> dict:
        """Consistent snapshot of cumulative token usage across all calls."""
        with self._usage_lock:
            return {
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "calls": self.total_calls,
            }

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[Any] = None,
        artifacts: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a raw text response from the model.

        Args:
            system_prompt: high-level instructions/rules
            user_prompt: task-specific input
            image: DEPRECATED legacy single-image arg; prefer ``artifacts``.
            artifacts: canonical multimodal payload — a list of
                ``{"kind": "image", "path": ...}`` dicts (see
                ``tsqa.llm.artifacts.normalize_artifacts``). Multimodal backends attach
                each image; text-only backends ignore it.
            **kwargs: backend-specific generation options

        Returns:
            Raw response text as a string.
        """
        raise NotImplementedError

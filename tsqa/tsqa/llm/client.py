from .base import BaseLLMClient
from .artifacts import normalize_artifacts
import random
import threading
import time


class GeminiClient(BaseLLMClient):
    """
    Thin wrapper around the Google GenAI SDK.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        max_tokens: int = 3000,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "google-genai is required for GeminiClient. "
                "Install it with: pip install google-genai"
            ) from e

        # Per-request timeout (ms) so a silently-dropped TCP connection can never
        # hang a long batch run forever (the SDK has no default timeout). A normal
        # call is <2s; vision calls a few seconds — 90s is a generous ceiling.
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=90_000),
        )
        self._types = genai_types
        self._call_lock = threading.Lock()
        self._last_call_time = 0.0
        # ~3000 RPM ceiling (0.02 s spacing), comfortably under the 4000 RPM cap.
        # The old 2.0 s value pinned the whole harness to ~30 RPM and made every
        # run fully serial. The spacing is now enforced in a tiny critical section
        # while the actual network call runs OUTSIDE the lock (see generate()).
        self._min_call_interval = 0.02
        self._max_retries = 3

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image=None,
        artifacts=None,
        **kwargs,
    ) -> str:
        """
        Generate text response from Gemini.
        If artifacts are provided, they are appended to the payload (used by the Vision sensor).

        A per-call ``temperature`` kwarg overrides ``self.temperature`` for this
        call ONLY (mirrors OpenAIClient). Default is ``self.temperature`` (0.0),
        so every existing call site stays byte-identical — the override exists
        solely for the Self-Consistency control arm's k>1 sampling (temp>0).
        """
        call_temperature = kwargs.get("temperature", self.temperature)
        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=call_temperature,
            max_output_tokens=self.max_tokens,
        )

        api_contents = [user_prompt]
        if artifacts is None:
            artifacts = kwargs.get("artifacts")
        artifacts = normalize_artifacts(artifacts, image=image)

        if artifacts:
            for art in artifacts:
                if not isinstance(art, dict):
                    continue
                if art.get("kind") == "image":
                    import os
                    from PIL import Image

                    path = art.get("path")
                    if path and os.path.exists(path):
                        try:
                            api_contents.append(Image.open(path))
                        except Exception:
                            continue

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                # Tiny critical section: enforce inter-call spacing and stamp the
                # send time, then release the lock BEFORE the (slow) network call
                # so concurrent worker threads have overlapping in-flight requests
                # instead of serializing on the HTTP round-trip.
                with self._call_lock:
                    elapsed = time.monotonic() - self._last_call_time
                    if elapsed < self._min_call_interval:
                        time.sleep(self._min_call_interval - elapsed)
                    self._last_call_time = time.monotonic()

                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=api_contents,
                    config=config,
                )
                um = getattr(resp, "usage_metadata", None)
                if um is not None:
                    self._add_usage(
                        prompt=getattr(um, "prompt_token_count", 0),
                        completion=getattr(um, "candidates_token_count", 0),
                        total=getattr(um, "total_token_count", 0),
                    )
                return (resp.text or "").strip()
            except Exception as e:
                last_error = e
                message = str(e).lower()
                retryable = any(
                    code in message
                    for code in (
                        "503", "429", "unavailable", "resource_exhausted",
                        "timeout", "timed out", "deadline", "connection",
                    )
                )
                if not retryable or attempt >= self._max_retries:
                    # Preserve the original exception type (auth/schema/network)
                    # instead of collapsing everything into RuntimeError. (#10;
                    # mirrors OpenAIClient — both clients re-raise the original so
                    # callers/monitoring can distinguish failure modes.)
                    raise
                backoff = min(20.0, (2**attempt) * 2.0) + random.random()
                time.sleep(backoff)

        raise RuntimeError(f"Gemini API call failed: {last_error}")

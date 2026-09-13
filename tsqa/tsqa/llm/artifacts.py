"""Canonical multimodal payload — **frozen Contract C** (gate-zero).

The live vision path passes ``artifacts=[{"kind": "image", "path": ...}]`` (see
``runner.py`` and ``llm/vision.py``), while the historical ``BaseLLMClient.generate``
signature only advertised ``image=``. This module makes ``artifacts`` the ONE canonical
multimodal channel every client consumes, so a new ``OpenAIClient`` and the existing
``GeminiClient`` share a single payload contract — and text-only clients (``QwenClient``)
simply ignore it.

This is purely additive: :func:`normalize_artifacts` returns exactly the
``list[{"kind","path"}]`` shape the current code already produces, so wiring it into a
client changes no existing behavior. The frozen-identity guard (``tests/test_frozen_identity``)
protects the Gemini vision path while Worker D adds the new client.
"""
from __future__ import annotations

from typing import Any, List, Optional, TypedDict


class ImageArtifact(TypedDict):
    kind: str   # "image"
    path: str


def normalize_artifacts(
    artifacts: Optional[List[dict]] = None,
    image: Optional[Any] = None,
) -> List[ImageArtifact]:
    """Coerce ``(artifacts, legacy image)`` into the canonical multimodal payload:
    ``[{"kind": "image", "path": ...}, ...]``.

    Rules (match ``GeminiClient``'s existing defensive handling):
      * Only well-formed image artifacts (``dict`` with ``kind == "image"`` and a truthy
        ``path``) are kept; malformed entries are dropped silently.
      * ``image`` is the deprecated single-image arg, honored only when ``artifacts`` yields
        nothing, and only for the path-bearing forms (``{"path": ...}`` or a ``str`` path).
        A non-path ``image`` (e.g. a raw PIL object) is ignored — callers should migrate to
        ``artifacts``.

    A client that cannot do multimodal calls passes the result and ignores it; a multimodal
    client iterates it and attaches each image.
    """
    out: List[ImageArtifact] = []
    if artifacts:
        for a in artifacts:
            if isinstance(a, dict) and a.get("kind") == "image" and a.get("path"):
                out.append({"kind": "image", "path": a["path"]})
    if not out and image is not None:
        if isinstance(image, dict) and image.get("path"):
            out.append({"kind": "image", "path": image["path"]})
        elif isinstance(image, str) and image:
            out.append({"kind": "image", "path": image})
    return out


__all__ = ["ImageArtifact", "normalize_artifacts"]

"""Trace helpers for optional runner instrumentation."""
from __future__ import annotations

from typing import Any, Optional

from .hooks import TraceFields, compute_trace_hash


def build_trace_fields(
    *,
    routing: Optional[dict] = None,
    branch: Optional[str] = None,
    evidence: Optional[dict] = None,
    answer: Optional[str] = None,
    model_id: Optional[str] = None,
    prompt_template_ids: Optional[list[str]] = None,
    verifier_state: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TraceFields:
    routing = routing or {}
    evidence = evidence or {}
    flags = evidence.get("flags") or []
    return TraceFields(
        route=routing.get("branch"),
        branch=branch or evidence.get("branch"),
        tool_sequence=[branch or evidence.get("branch")] if (branch or evidence.get("branch")) else [],
        tool_args={"scope": routing.get("scope"), "subtype": routing.get("subtype")},
        tool_outputs=evidence,
        prompt_template_ids=list(prompt_template_ids or []),
        model_id=model_id,
        answer=answer,
        verifier_state=verifier_state or ",".join(sorted(map(str, flags))),
    )


def attach_trace(
    result: dict,
    *,
    routing: Optional[dict] = None,
    evidence: Optional[dict] = None,
    model_id: Optional[str] = None,
    prompt_template_ids: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict:
    fields = build_trace_fields(
        routing=routing or result.get("routing"),
        branch=result.get("branch_used"),
        evidence=evidence or result.get("evidence"),
        answer=result.get("predicted_letter") or result.get("predicted_value"),
        model_id=model_id,
        prompt_template_ids=prompt_template_ids,
        metadata=metadata,
    )
    result["trace_hash"] = compute_trace_hash(fields)
    result["trace_fields"] = fields
    return result


__all__ = ["build_trace_fields", "attach_trace"]

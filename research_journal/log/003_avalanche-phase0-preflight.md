# 003 — The Graceful Avalanche: Phase 0 pre-flight (env pinned, baselines reconciled, Gate 0 green)

---
id: 003
date: 2026-06-22
stage: 1
claim: none   # pre-flight for the staged Stage 1→3 transition; moves no experimental claim yet
status: note
evidence_level: scaffold
claim_scope: n/a
overall_significance: n/a
prereg: research_journal/02_react_stages.md (Stage 1–3 trajectory); per-stage PREREGISTRATION_*.md (pending)
commit: agentic-tsmart @ Gate 0 (see git log)
artifact_present: n/a   # pre-flight; no run yet
required_caveats: "no experiment has run; this only establishes the validated substrate for the staged paired runs."
---

## Why this entry exists
Kickoff of **The Graceful Avalanche** — the staged, *measured* decision-tree → ReAct transition
(Stages 1–3; ADR-007 / `02_react_stages.md`), executed as a meta-orchestrator → stage-orchestrator
program. Per repo norm (no result without a log entry) this records the Phase-0 substrate every
subsequent paired run depends on, and resolves one stale premise carried in from the cloud plan.

## What was established (Gate 0)
- **Branch:** `agentic-tsmart` off `numeric-head-visual-trust` (frozen surfaces untouched).
- **Interpreter pinned:** system `/usr/bin/python3` already carries all deps; `tsqa` re-linked
  **editable** (`pip install -e agentic-setup --break-system-packages`) so orchestrator edits to
  `tsqa/` are live everywhere — guards against the stale-code / silent-wrong-result failure mode.
  Every orchestrator uses this interpreter.
- **Suites green:** agentic-setup **72**, MMTS tools **89** + smoke **20/20**, TSRBench **20**.
  MCQ byte-identity invariant intact.
- **Vision path live:** `test_vision_pipeline.py` end-to-end (router → branch → trigger →
  **structured JSON** topology → reasoner, correct). SVI intact (PNG to tmp; no raw pixels to the
  reasoner).
- **Backbones reachable:** `GEMINI_API_KEY` present; OpenAI key (Track A) verified working via a
  1-token `gpt-4o-mini` call (HTTP 200). Secrets never surfaced (loaded in-process; only
  presence/length/prefix printed).

## Stale-premise correction (cloud plan vs on-disk reality)
The cloud-authored plan (written on a fresh container) stated *"MMTS `outputs/` does not exist — a
baseline must be generated."* **False locally:** `mmts_bench/outputs/generalization/` already holds
the full paired grid {Base,InWild,Match,Align} × {baseline,nu_ad_fix} with per-row result CSVs
(2026-06-18). TSExam `outputs/baseline_full` (746) and the TSRBench baseline (6 runs) are likewise on
disk. **Decision:** *reuse* these as the pairing baselines (join on row id), never regenerate — this
preserves the pairing and ~$4–5 of Gemini budget. Phase 0's "long pole" is therefore already retired.

## What changed next
- Gate 0 cleared → Stage-1 orchestrator authorized to build the two still-open promotion
  prerequisites (Self-Consistency control arm #4; verifier-flag precision/recall #2 — see log/002)
  and run the first *measured* Stage-1 paired experiment.

## Artifacts
- Living scoreboard: `AGENTIC_TSMART_RESULTS.md` (repo root).
- Task ledger: session tasks #1–#8 (one per phase/track).

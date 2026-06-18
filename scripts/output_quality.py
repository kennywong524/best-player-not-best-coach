#!/usr/bin/env python3
"""Per-row output quality checks (truncation, empties, task deliverable rules)."""

from __future__ import annotations

import re
from typing import Any

from audit_run import (  # noqa: E402
    DEFAULT_MIN_OUTPUT_CHARS,
    _deliverable_failure_reasons,
    _strip_reasoning_wrappers,
)


def hard_truncation_signals(text: str) -> list[str]:
    """Heuristic truncation signals (same rules as audit_all_outputs)."""
    out = _strip_reasoning_wrappers(text)
    if not out:
        return []
    signals: list[str] = []
    tail = out.rstrip()
    if len(tail) < 400:
        return signals
    if re.search(r"[,;:\-]\s*$", tail):
        signals.append("ends_with_punctuation_gap")
    if re.search(r"\b(and|or|the|to|for|with|in|on|of|that|which)\s*$", tail, flags=re.I):
        signals.append("ends_with_conjunction")
    if re.search(r"\b\w{1,3}\s*$", tail) and not re.search(r"[.!?][\"')\]]*\s*$", tail[-40:]):
        signals.append("ends_mid_fragment")
    return signals


def audit_output_row(
    raw: str,
    *,
    task_slug: str,
    mode: str,
    condition: str,
    min_output_chars: int = DEFAULT_MIN_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Return ok/issues/n_chars for a single generated output."""
    out = _strip_reasoning_wrappers(raw)
    quality = _deliverable_failure_reasons(
        raw,
        task_slug=task_slug,
        condition=condition,
        min_output_chars=min_output_chars,
    )
    trunc = hard_truncation_signals(raw)
    issues = list(dict.fromkeys(quality + trunc))
    return {
        "ok": not issues,
        "issues": issues,
        "n_chars": len(out),
        "tail_preview": out[-120:] if out else "",
    }


def audit_csv_row(
    row: dict[str, Any],
    *,
    task_slug: str,
    mode: str,
    min_output_chars: int = DEFAULT_MIN_OUTPUT_CHARS,
) -> dict[str, Any]:
    raw = str(row.get("output", "") or "")
    condition = str(row.get("condition", ""))
    result = audit_output_row(
        raw,
        task_slug=task_slug,
        mode=mode,
        condition=condition,
        min_output_chars=min_output_chars,
    )
    result.update(
        {
            "task": task_slug,
            "mode": mode,
            "model_id": str(row.get("model_id", "")),
            "model_label": str(row.get("model_label", "")),
            "replicate_id": row.get("replicate_id"),
            "condition": condition,
        }
    )
    return result

#!/usr/bin/env python3
"""Audit every row in augmentation/automation outputs for truncation and deliverable quality."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from centaur_benchmark.io import results_base

from audit_run import (  # noqa: E402
    DEFAULT_MIN_OUTPUT_CHARS,
    TASK_REQUIRED_TERMS,
    TASK_SCAFFOLD_OUTPUT_MIN_CHARS,
    _deliverable_failure_reasons,
    _strip_reasoning_wrappers,
)


def _hard_truncation_signals(text: str) -> list[str]:
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


def audit_all_outputs(
    run_id: str,
    *,
    modes: list[str] | None = None,
    exclude_model_ids: list[str] | None = None,
    min_output_chars: int = DEFAULT_MIN_OUTPUT_CHARS,
) -> dict:
    modes = modes or ["augmentation", "automation"]
    excluded = set(exclude_model_ids or [])
    rows: list[dict] = []
    base = results_base()

    for task_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        task_slug = task_dir.name
        for mode in modes:
            csv_path = task_dir / run_id / mode / "outputs.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            for _, rec in df.iterrows():
                model_id = str(rec.get("model_id", ""))
                if mode == "augmentation" and model_id in excluded:
                    continue
                raw = str(rec.get("output", "") or "")
                out = _strip_reasoning_wrappers(raw)
                condition = str(rec.get("condition", ""))
                reasons = _deliverable_failure_reasons(
                    raw,
                    task_slug=task_slug,
                    condition=condition,
                    min_output_chars=min_output_chars,
                )
                trunc = _hard_truncation_signals(raw)
                row = {
                    "task": task_slug,
                    "mode": mode,
                    "model_id": model_id,
                    "model_label": str(rec.get("model_label", "")),
                    "condition": condition,
                    "replicate_id": rec.get("replicate_id"),
                    "n_chars": len(out),
                    "ok": not reasons and not trunc,
                    "quality_issues": reasons,
                    "truncation_signals": trunc,
                    "tail_preview": out[-100:] if out else "",
                }
                rows.append(row)

    failing = [r for r in rows if not r["ok"]]
    by_kind: dict[str, int] = {}
    for r in failing:
        for issue in r["quality_issues"] + r["truncation_signals"]:
            by_kind[issue] = by_kind.get(issue, 0) + 1

    return {
        "run_id": run_id,
        "modes": modes,
        "excluded_models": sorted(excluded),
        "n_rows": len(rows),
        "n_ok": len(rows) - len(failing),
        "n_failing": len(failing),
        "issue_counts": dict(sorted(by_kind.items(), key=lambda x: (-x[1], x[0]))),
        "rows": rows,
        "failures": failing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit every output row in a run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--modes", default="augmentation,automation")
    parser.add_argument("--exclude-models", default=None)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_OUTPUT_CHARS)
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    excluded = [x.strip() for x in args.exclude_models.split(",") if x.strip()] if args.exclude_models else None
    report = audit_all_outputs(
        args.run_id,
        modes=modes,
        exclude_model_ids=excluded,
        min_output_chars=args.min_chars,
    )
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"rows={report['n_rows']} ok={report['n_ok']} failing={report['n_failing']}")
    if report["issue_counts"]:
        print("issue_counts:", report["issue_counts"])
    for r in report["failures"][:40]:
        issues = r["quality_issues"] + r["truncation_signals"]
        print(
            f"  {r['task']}/{r['mode']} {r['model_label']} rep={r['replicate_id']} "
            f"{r['n_chars']}c {issues}"
        )


if __name__ == "__main__":
    main()

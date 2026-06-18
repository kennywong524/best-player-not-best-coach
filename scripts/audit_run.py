#!/usr/bin/env python3
"""Audit a benchmark run for missing or empty outputs and scaffolds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import results_base, write_json

DEFAULT_MIN_OUTPUT_CHARS = 250
DEFAULT_MAX_SCAFFOLD_CHARS = 2600

# Minimum chars for scaffold-conditioned worker outputs (plain baseline uses min_output_chars only).
TASK_SCAFFOLD_OUTPUT_MIN_CHARS: dict[str, int] = {
    "meal_plan": 1500,
    "tax_prep": 1800,
    "travel_planning": 1500,
    "operations_research": 1200,
}

TASK_REQUIRED_TERMS: dict[str, tuple[str, ...]] = {
    "meal_plan": ("breakfast", "lunch", "dinner", "grocery"),
    "tax_prep": ("schedule", "federal", "california"),
    "travel_planning": ("itinerary", "budget", "hotel"),
}


def _strip_reasoning_wrappers(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"(?is)<think>.*?</think>\s*", "", t)
    t = re.sub(r"(?is)<think>.*\Z", "", t)
    return t.strip()


def _output_failure_reasons(text: str, *, min_chars: int = DEFAULT_MIN_OUTPUT_CHARS) -> list[str]:
    out = _strip_reasoning_wrappers(text)
    if not out:
        return ["empty_output"]
    reasons: list[str] = []
    if out.startswith("{'format':") or out.startswith('{"format":'):
        reasons.append("format_stub")
    if len(out) < min_chars:
        reasons.append(f"short<{min_chars}")
    return reasons


def _scaffold_failure_reasons(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_OUTPUT_CHARS,
    max_chars: int = DEFAULT_MAX_SCAFFOLD_CHARS,
) -> list[str]:
    out = str(text or "").strip()
    reasons = _output_failure_reasons(out, min_chars=min_chars)
    if not out:
        return reasons
    low = (
        out.lower()
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    normalized_start = low.lstrip("*# \n")
    if "assistant guidance: three-phase workflow" not in low:
        reasons.append("missing_scaffold_heading")
    elif not normalized_start.startswith("assistant guidance: three-phase workflow"):
        reasons.append("heading_not_first")
    if len(out) > max_chars:
        reasons.append(f"too_long>{max_chars}")
    return reasons


def _deliverable_failure_reasons(
    text: str,
    *,
    task_slug: str,
    condition: str,
    min_output_chars: int,
) -> list[str]:
    reasons = _output_failure_reasons(text, min_chars=min_output_chars)
    out = _strip_reasoning_wrappers(text)
    if not out:
        return reasons

    low = out.lower()
    is_scaffold_row = str(condition).startswith("scaffold_")
    if is_scaffold_row:
        min_scaffold_out = TASK_SCAFFOLD_OUTPUT_MIN_CHARS.get(task_slug, min_output_chars)
        if len(out) < min_scaffold_out:
            reasons.append(f"short_scaffold_output<{min_scaffold_out}")

        missing = [term for term in TASK_REQUIRED_TERMS.get(task_slug, ()) if term not in low]
        if missing:
            reasons.append(f"missing_terms:{','.join(missing)}")

        meta_markers = (
            "three-phase workflow",
            "requirements check",
            "follow the structured workflow",
            "following the outlined structure",
            "following this structured approach",
            "by following the three-phase workflow",
        )
        if any(marker in low for marker in meta_markers):
            reasons.append("meta_scaffold_echo")

    if task_slug == "meal_plan":
        days = {int(d) for d in re.findall(r"\bDay\s*(\d+)\b", out, flags=re.I)}
        if is_scaffold_row and days and max(days) < 7:
            reasons.append(f"incomplete_meal_plan_days<7(max={max(days)})")

    # Hard truncation: long output that ends mid-thought (not merely missing final period).
    tail = out.rstrip()
    if len(tail) >= 1200:
        if re.search(r"[,;:\-]\s*$", tail):
            reasons.append("likely_hard_truncation")
        elif re.search(r"\b(and|or|the|to|for|with|in|on|of|that|which)\s*$", tail, flags=re.I):
            reasons.append("likely_hard_truncation")

    return list(dict.fromkeys(reasons))


def audit_run(
    run_id: str,
    *,
    task_slugs: list[str] | None = None,
    min_output_chars: int = DEFAULT_MIN_OUTPUT_CHARS,
    augmentation_only: bool = False,
    exclude_model_ids: list[str] | None = None,
) -> dict:
    task_paths = sorted(default_tasks_dir().glob("*.yaml"))
    tasks = [load_task(p) for p in task_paths]
    if task_slugs:
        wanted = set(task_slugs)
        tasks = [t for t in tasks if t.slug in wanted]

    excluded = set(exclude_model_ids or [])
    report: dict = {
        "run_id": run_id,
        "min_output_chars": min_output_chars,
        "augmentation_only": augmentation_only,
        "excluded_models": sorted(excluded),
        "tasks": {},
        "failures": [],
        "missing_tasks": [],
        "ready_for_judging": True,
    }

    for task in tasks:
        root = results_base() / task.slug / run_id
        task_info: dict = {
            "augmentation_outputs": False,
            "automation_outputs": False,
            "bad_outputs": [],
            "bad_scaffolds": [],
            "empty_scaffolds": [],
            "missing_models": {"augmentation": [], "automation": []},
        }

        aug_csv = root / "augmentation" / "outputs.csv"
        auto_csv = root / "automation" / "outputs.csv"
        if not aug_csv.exists():
            report["missing_tasks"].append(task.slug)
            report["ready_for_judging"] = False
            report["tasks"][task.slug] = task_info
            continue
        if not augmentation_only and not auto_csv.exists():
            report["missing_tasks"].append(task.slug)
            report["ready_for_judging"] = False
            report["tasks"][task.slug] = task_info
            continue

        task_info["augmentation_outputs"] = True
        task_info["automation_outputs"] = auto_csv.exists()

        mode_specs: list[tuple[str, Path, set[str]]] = [
            ("augmentation", aug_csv, set(task.assistants.keys()) | {"plain"}),
        ]
        if not augmentation_only and auto_csv.exists():
            mode_specs.append(
                ("automation", auto_csv, set((task.automation_models or {}).keys()))
            )

        for mode, path, expected in mode_specs:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                model_id = str(row.get("model_id", ""))
                if mode == "augmentation" and model_id in excluded:
                    continue
                out = str(row.get("output", "") or "")
                reasons = _deliverable_failure_reasons(
                    out,
                    task_slug=task.slug,
                    condition=str(row.get("condition", "")),
                    min_output_chars=min_output_chars,
                )
                if not reasons:
                    continue
                failure = {
                    "task": task.slug,
                    "mode": mode,
                    "model_id": model_id,
                    "model_label": str(row.get("model_label", "")),
                    "condition": str(row.get("condition", "")),
                    "kind": reasons[0],
                    "reasons": reasons,
                    "n_chars": len(_strip_reasoning_wrappers(out)),
                    "preview": _strip_reasoning_wrappers(out)[:120],
                }
                task_info["bad_outputs"].append(failure)
                report["failures"].append(failure)
                report["ready_for_judging"] = False

            present = set(df["model_id"].astype(str))
            expected_present = expected - excluded if mode == "augmentation" else expected
            missing = sorted(expected_present - present)
            if missing:
                task_info["missing_models"][mode] = missing
                for mid in missing:
                    failure = {
                        "task": task.slug,
                        "mode": mode,
                        "model_id": mid,
                        "model_label": (task.assistants if mode == "augmentation" else task.automation_models or {}).get(mid, mid),
                        "kind": "missing_model",
                    }
                    report["failures"].append(failure)
                    report["ready_for_judging"] = False

        from centaur_benchmark.runner import _safe_slug

        scaffolds_dir = root / "augmentation" / "scaffolds"
        for model_id, model_label in task.assistants.items():
            if model_id in excluded:
                continue
            base = scaffolds_dir / f"{_safe_slug(model_label)}.md"
            scaffold_paths = [base] if base.exists() else sorted(
                scaffolds_dir.glob(f"{_safe_slug(model_label)}_rep*.md")
            )
            if not scaffold_paths or any(p.stat().st_size == 0 for p in scaffold_paths):
                failure = {
                    "task": task.slug,
                    "mode": "augmentation",
                    "model_id": model_id,
                    "model_label": model_label,
                    "kind": "empty_scaffold",
                }
                task_info["empty_scaffolds"].append(failure)
                if not any(
                    f["task"] == task.slug
                    and f["model_id"] == model_id
                    and f["kind"] == "empty_scaffold"
                    for f in report["failures"]
                ):
                    report["failures"].append(failure)
                report["ready_for_judging"] = False
                continue

            for p in scaffold_paths:
                scaffold_text = p.read_text(encoding="utf-8", errors="replace")
                scaffold_reasons = _scaffold_failure_reasons(
                    scaffold_text, min_chars=min_output_chars
                )
                if scaffold_reasons:
                    failure = {
                        "task": task.slug,
                        "mode": "augmentation",
                        "model_id": model_id,
                        "model_label": model_label,
                        "kind": scaffold_reasons[0],
                        "reasons": scaffold_reasons,
                        "n_chars": len(scaffold_text.strip()),
                        "preview": scaffold_text.strip()[:120],
                        "path": str(p),
                    }
                    task_info["bad_scaffolds"].append(failure)
                    report["failures"].append(failure)
                    report["ready_for_judging"] = False

        report["tasks"][task.slug] = task_info

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit run outputs for completeness.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tasks", default=None, help="Comma-separated task slugs")
    parser.add_argument("--json-out", default=None, help="Write audit JSON path")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_OUTPUT_CHARS)
    parser.add_argument(
        "--augmentation-only",
        action="store_true",
        help="Do not require automation outputs; audit augmentation + scaffolds only.",
    )
    parser.add_argument(
        "--exclude-models",
        default=None,
        help="Comma-separated assistant model_ids to skip (e.g. google/gemini-3.1-pro).",
    )
    args = parser.parse_args()
    slugs = [x.strip() for x in args.tasks.split(",") if x.strip()] if args.tasks else None
    excluded = [x.strip() for x in args.exclude_models.split(",") if x.strip()] if args.exclude_models else None
    report = audit_run(
        args.run_id,
        task_slugs=slugs,
        min_output_chars=args.min_chars,
        augmentation_only=args.augmentation_only,
        exclude_model_ids=excluded,
    )
    out = Path(args.json_out) if args.json_out else results_base() / f"audit_{args.run_id}.json"
    write_json(out, report)
    print(f"Audit written to {out}")
    print(f"ready_for_judging={report['ready_for_judging']}")
    print(f"failures={len(report['failures'])} missing_tasks={report['missing_tasks']}")
    for f in report["failures"]:
        print(f"  {f['task']}/{f['mode']}: {f['model_label']} ({f['kind']})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate panel judging outputs for a benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import results_base
from centaur_benchmark.judge_pairwise import validate_judge_batch

REQUIRED_FILES = (
    "pairwise_judgments_by_judge.csv",
    "pairwise_ranked_by_judge.csv",
    "leaderboard_by_judge.csv",
    "leaderboard_aggregate.csv",
    "leaderboard_matrix_win_rate_by_judge.csv",
    "leaderboard_matrix_rank_by_judge.csv",
    "leaderboard_matrix_aggregate.csv",
    "rubric_scores_long.csv",
    "rubric_scores_summary.csv",
)


def validate_task_mode(run_id: str, task_slug: str, mode: str) -> dict:
    root = results_base() / task_slug / run_id / mode
    info: dict = {
        "task": task_slug,
        "mode": mode,
        "complete": False,
        "missing_files": [],
        "judges": {},
        "excluded_judges": [],
        "rubric_judges": [],
    }
    if not root.exists():
        info["missing_files"] = list(REQUIRED_FILES)
        return info

    for name in REQUIRED_FILES:
        if not (root / name).exists():
            info["missing_files"].append(name)

    validation_path = root / "judge_validation.json"
    if validation_path.exists():
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        info["judges"] = payload.get("judges", {})
        info["excluded_judges"] = payload.get("excluded_from_aggregate", [])
        info["aggregate_judges"] = payload.get("aggregate_judges", [])

    judgments_path = root / "pairwise_judgments_by_judge.csv"
    if judgments_path.exists():
        df = pd.read_csv(judgments_path)
        if "judge_label" in df.columns:
            for label in sorted(df["judge_label"].dropna().unique()):
                sub = df[df["judge_label"] == label]
                batch = validate_judge_batch(sub)
                info["judges"][label] = {
                    **{k: v for k, v in batch.items() if k != "issues_by_row"},
                    "revalidated": True,
                }

    summary_path = root / "rubric_scores_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "judge_model" in summary.columns:
            info["rubric_judges"] = sorted(summary["judge_model"].dropna().unique().tolist())

    info["complete"] = not info["missing_files"]
    return info


def validate_run(run_id: str, *, task_slugs: list[str] | None = None) -> dict:
    tasks = [load_task(p) for p in sorted(default_tasks_dir().glob("*.yaml"))]
    if task_slugs:
        wanted = set(task_slugs)
        tasks = [t for t in tasks if t.slug in wanted]

    report = {
        "run_id": run_id,
        "task_modes": [],
        "complete_task_modes": [],
        "incomplete_task_modes": [],
        "excluded_judges": [],
        "ready_for_summarize": True,
    }

    for task in tasks:
        for mode in ("augmentation", "automation"):
            info = validate_task_mode(run_id, task.slug, mode)
            report["task_modes"].append(info)
            key = f"{task.slug}/{mode}"
            if info["complete"]:
                report["complete_task_modes"].append(key)
            else:
                report["incomplete_task_modes"].append(key)
                report["ready_for_summarize"] = False
            for ex in info.get("excluded_judges", []):
                report["excluded_judges"].append({**ex, "task_mode": key})

    return report


def write_notes(run_id: str, report: dict, out_path: Path) -> None:
    lines = [
        f"# Judging notes — {run_id}",
        "",
        "## Complete task/mode folders",
    ]
    if report["complete_task_modes"]:
        lines.extend(f"- {x}" for x in report["complete_task_modes"])
    else:
        lines.append("- none")

    lines.extend(["", "## Incomplete or missing artifacts", ""])
    if report["incomplete_task_modes"]:
        for key in report["incomplete_task_modes"]:
            info = next(
                x for x in report["task_modes"] if f"{x['task']}/{x['mode']}" == key
            )
            missing = ", ".join(info.get("missing_files", [])) or "unknown"
            lines.append(f"- {key}: missing {missing}")
    else:
        lines.append("- none")

    lines.extend(["", "## Judges excluded from aggregate", ""])
    if report["excluded_judges"]:
        for ex in report["excluded_judges"]:
            lines.append(
                f"- {ex.get('task_mode', '?')}: {ex.get('judge_label', '?')} "
                f"({ex.get('reason', ex.get('exclude_reason', 'unknown'))})"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Rubric score coverage by task/mode", ""])
    for info in report["task_modes"]:
        judges = info.get("rubric_judges") or []
        lines.append(f"- {info['task']}/{info['mode']}: {', '.join(judges) or 'none'}")

    lines.extend(
        [
            "",
            f"ready_for_summarize={report['ready_for_summarize']}",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate judging artifacts for a run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tasks", default=None, help="Comma-separated task slugs")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--notes-out", default=None)
    args = parser.parse_args()

    slugs = [x.strip() for x in args.tasks.split(",") if x.strip()] if args.tasks else None
    report = validate_run(args.run_id, task_slugs=slugs)

    json_out = (
        Path(args.json_out)
        if args.json_out
        else results_base() / f"judge_validation_{args.run_id}.json"
    )
    json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    notes_out = (
        Path(args.notes_out)
        if args.notes_out
        else results_base() / f"judging_notes_{args.run_id}.md"
    )
    write_notes(args.run_id, report, notes_out)

    print(f"Validation written to {json_out}")
    print(f"Notes written to {notes_out}")
    print(f"complete={len(report['complete_task_modes'])} incomplete={len(report['incomplete_task_modes'])}")
    print(f"ready_for_summarize={report['ready_for_summarize']}")


if __name__ == "__main__":
    main()

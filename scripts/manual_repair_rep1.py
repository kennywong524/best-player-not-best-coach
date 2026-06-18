#!/usr/bin/env python3
"""Targeted repair for remaining failed 20260612_fresh_rep1 automation rows."""

from __future__ import annotations

from pathlib import Path

from audit_all_outputs import audit_all_outputs
from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import ensure_run_dir, write_json
from centaur_benchmark.runner import generate_automation_replicate
from output_quality import audit_output_row


RUN_ID = "20260612_fresh_rep1"
REPLICATE_ID = 0


def main() -> None:
    targets: list[tuple[str, str, list[int]]] = [
        ("meal_plan", "gpt-5-mini-2025-08-07", [0, 0, 0, 0, 0, 0]),
        ("meal_plan", "deepseek-ai/DeepSeek-V3.1", [1, 2, 3, 0, 1, 2]),
        ("tax_prep", "gpt-5-mini-2025-08-07", [0, 0, 0, 0, 0, 0]),
        ("tax_prep", "deepseek-ai/DeepSeek-V3.1", [1, 2, 3, 0, 1, 2]),
        ("travel_planning", "deepseek-ai/DeepSeek-V3.1", [1, 2, 3, 0, 1, 2]),
    ]

    for slug, model_id, attempts in targets:
        task = load_task(default_tasks_dir() / f"{slug}.yaml")
        root = ensure_run_dir(slug, RUN_ID)
        label = (task.automation_models or {})[model_id]
        condition = f"automation_{label.replace(' ', '_')}"
        print(f"\n=== REPAIR {slug} {label} ===", flush=True)
        repaired = False
        last_audit = None
        for index, attempt in enumerate(attempts, 1):
            print(f"try {index}/{len(attempts)} attempt={attempt}", flush=True)
            try:
                raw = generate_automation_replicate(
                    task,
                    root,
                    model_id,
                    REPLICATE_ID,
                    attempt=attempt,
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic repair script
                print(f"EXCEPTION {type(exc).__name__}: {exc}", flush=True)
                continue
            last_audit = audit_output_row(
                raw,
                task_slug=slug,
                mode="automation",
                condition=condition,
            )
            print(
                f"chars={last_audit['n_chars']} ok={last_audit['ok']} "
                f"issues={last_audit['issues']}",
                flush=True,
            )
            if last_audit["ok"]:
                repaired = True
                break
        if not repaired:
            print(f"FAILED_REPAIR {slug} {label} last={last_audit}", flush=True)

    report = audit_all_outputs(RUN_ID)
    out = Path("results/logs") / f"audit_{RUN_ID}_manual_repair.json"
    write_json(out, report)
    print(
        f"\nAUDIT {report['n_ok']}/{report['n_rows']} ok; "
        f"failing={report['n_failing']}; report={out}",
        flush=True,
    )
    for failure in report.get("failures", [])[:20]:
        print(
            "FAIL",
            failure["task"],
            failure["mode"],
            failure["model_label"],
            failure.get("quality_issues"),
            failure.get("truncation_signals"),
            flush=True,
        )


if __name__ == "__main__":
    main()

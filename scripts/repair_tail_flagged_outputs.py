#!/usr/bin/env python3
"""Repair rows that pass mechanical audit but visibly end mid-output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from edsl import Agent, Model, QuestionFreeText, Scenario, ScenarioList, Survey

from audit_all_outputs import audit_all_outputs
from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.edsl_runtime import edsl_run_kwargs
from centaur_benchmark.io import ensure_run_dir, write_json
from output_quality import audit_output_row


RUN_ID = "20260612_fresh_rep1"

TARGETS: list[dict[str, Any]] = [
    {
        "task": "counselling",
        "model_id": "gpt-5-mini-2025-08-07",
        "variants": [
            ("gpt5_max6000", {"max_tokens": 6000}),
            ("gpt5_max12000", {"max_tokens": 12000}),
        ],
        "min_chars": 2500,
    },
    {
        "task": "travel_planning",
        "model_id": "gpt-5-mini-2025-08-07",
        "variants": [
            ("gpt5_max8000", {"max_tokens": 8000}),
            ("gpt5_max12000", {"max_tokens": 12000}),
        ],
        "min_chars": 4500,
    },
    {
        "task": "tutoring",
        "model_id": "gpt-5-mini-2025-08-07",
        "variants": [
            ("gpt5_max8000", {"max_tokens": 8000}),
            ("gpt5_max12000", {"max_tokens": 12000}),
        ],
        "min_chars": 4500,
    },
    {
        "task": "meal_plan",
        "model_id": "deepseek-ai/DeepSeek-V3.1",
        "variants": [
            ("deepseek_max12000", {"max_tokens": 12000}),
            ("deepseek_max24000", {"max_tokens": 24000}),
            ("deepseek_default", {}),
        ],
        "min_chars": 5000,
    },
]


def _visually_ok(text: str, *, min_chars: int) -> tuple[bool, list[str]]:
    stripped = str(text or "").strip()
    issues: list[str] = []
    if len(stripped) < min_chars:
        issues.append(f"short_visual<{min_chars}")
    if stripped and stripped[-1] not in ".!?)]}\"'’”*":
        issues.append("tail_no_terminal_punct")
    tail = stripped[-260:].lower()
    bad_tail_fragments = [
        " - give",
        " day 2 ",
        " make",
        "      •",
        " - costs",
    ]
    if any(fragment in tail for fragment in bad_tail_fragments):
        issues.append("tail_looks_mid_list")
    return not issues, issues


def _call(task_slug: str, model_id: str, kwargs: dict[str, Any], tag: str) -> str | None:
    task = load_task(default_tasks_dir() / f"{task_slug}.yaml")
    label = (task.automation_models or {})[model_id]
    condition = f"automation_{label.replace(' ', '_')}"
    scenario = ScenarioList(
        [
            Scenario(
                {
                    "task_prompt": task.task_prompt,
                    "condition": condition,
                    "replicate_id": 0,
                }
            )
        ]
    )
    survey = Survey([QuestionFreeText("output", "{{ scenario.task_prompt }}")])
    agent = Agent(
        instruction=task.automation_worker_instruction
        or "Complete the task clearly and professionally. Return only the final deliverable."
    )
    print(f"TRY {task_slug} {model_id} {tag} kwargs={kwargs}", flush=True)
    results = (
        survey.by(scenario)
        .by(agent)
        .by(Model(model_id, **kwargs))
        .run(
            **edsl_run_kwargs(
                description=f"{RUN_ID}-tail-repair-{task_slug}-{tag}",
                visibility=task.remote_inference_visibility,
                n=1,
            )
        )
    )
    raw = results.select("answer.output").to_list()[0]
    audit = audit_output_row(raw, task_slug=task_slug, mode="automation", condition=condition)
    print(f"RESULT chars={audit['n_chars']} ok={audit['ok']} issues={audit['issues']}", flush=True)
    return raw if audit["ok"] else None


def _upsert(task_slug: str, model_id: str, raw: str) -> None:
    task = load_task(default_tasks_dir() / f"{task_slug}.yaml")
    root = ensure_run_dir(task_slug, RUN_ID)
    label = (task.automation_models or {})[model_id]
    condition = f"automation_{label.replace(' ', '_')}"
    out_csv = root / "automation" / "outputs.csv"
    df = pd.read_csv(out_csv)
    if "replicate_id" in df.columns:
        df["replicate_id"] = df["replicate_id"].astype(int)
    df = df[~((df["model_id"] == model_id) & (df["replicate_id"] == 0))]
    row = pd.DataFrame(
        [
            {
                "replicate_id": 0,
                "output": raw,
                "condition": condition,
                "model_id": model_id,
                "model_label": label,
            }
        ]
    )
    pd.concat([df, row], ignore_index=True).to_csv(out_csv, index=False)


def main() -> None:
    repaired: list[str] = []
    failed: list[str] = []
    for target in TARGETS:
        task_slug = target["task"]
        model_id = target["model_id"]
        min_chars = int(target["min_chars"])
        clean: str | None = None
        clean_tag: str | None = None
        for tag, kwargs in target["variants"]:
            try:
                raw = _call(task_slug, model_id, kwargs, tag)
            except Exception as exc:  # noqa: BLE001
                print(f"EXCEPTION {task_slug} {model_id} {tag}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if not raw:
                continue
            visual_ok, visual_issues = _visually_ok(raw, min_chars=min_chars)
            print(f"VISUAL ok={visual_ok} issues={visual_issues}", flush=True)
            if visual_ok:
                clean = raw
                clean_tag = tag
                break
        if clean is None:
            failed.append(f"{task_slug}/{model_id}")
            print(f"FAILED_VISUAL_REPAIR {task_slug} {model_id}", flush=True)
            continue
        _upsert(task_slug, model_id, clean)
        repaired.append(f"{task_slug}/{model_id}/{clean_tag}")
        print(f"UPSERTED {task_slug} {model_id} via {clean_tag}", flush=True)

    report = audit_all_outputs(RUN_ID)
    write_json(Path(f"results/logs/audit_{RUN_ID}_tail_repairs.json"), report)
    print(f"REPAIRED {repaired}", flush=True)
    print(f"FAILED {failed}", flush=True)
    print(f"AUDIT {report['n_ok']}/{report['n_rows']} ok failing={report['n_failing']}", flush=True)


if __name__ == "__main__":
    main()

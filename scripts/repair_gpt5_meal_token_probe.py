#!/usr/bin/env python3
"""Targeted GPT-5-Mini meal-plan repair with token/service variants."""

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
TASK = "meal_plan"
MODEL_ID = "gpt-5-mini-2025-08-07"


def _call_model(task, kwargs: dict[str, Any], tag: str) -> str | None:
    label = (task.automation_models or {})[MODEL_ID]
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
    print(f"TRY {tag} kwargs={kwargs}", flush=True)
    results = (
        survey.by(scenario)
        .by(agent)
        .by(Model(MODEL_ID, **kwargs))
        .run(
            **edsl_run_kwargs(
                description=f"{RUN_ID}-meal-gpt5mini-token-probe-{tag}",
                visibility=task.remote_inference_visibility,
                n=1,
            )
        )
    )
    raw = results.select("answer.output").to_list()[0]
    audit = audit_output_row(raw, task_slug=TASK, mode="automation", condition=condition)
    print(
        f"RESULT {tag} chars={audit['n_chars']} ok={audit['ok']} issues={audit['issues']}",
        flush=True,
    )
    if not audit["ok"]:
        preview = str(raw or "")[:200].replace("\n", " ")
        print(f"PREVIEW {tag}: {preview}", flush=True)
    return raw if audit["ok"] else None


def main() -> None:
    task = load_task(default_tasks_dir() / f"{TASK}.yaml")
    variants: list[tuple[str, dict[str, Any]]] = [
        ("default_no_max_no_temp", {}),
        ("default_max6000", {"max_tokens": 6000}),
        ("default_max12000", {"max_tokens": 12000}),
        ("openai_max6000", {"service_name": "openai", "max_tokens": 6000}),
        ("openai_max12000", {"service_name": "openai", "max_tokens": 12000}),
        ("openai_temp05_max6000", {"service_name": "openai", "temperature": 0.5, "max_tokens": 6000}),
    ]

    raw_good: str | None = None
    winner: str | None = None
    for tag, kwargs in variants:
        try:
            raw_good = _call_model(task, kwargs, tag)
        except Exception as exc:  # noqa: BLE001
            print(f"EXCEPTION {tag}: {type(exc).__name__}: {exc}", flush=True)
            raw_good = None
        if raw_good:
            winner = tag
            break

    if raw_good:
        root = ensure_run_dir(TASK, RUN_ID)
        out_csv = root / "automation" / "outputs.csv"
        label = (task.automation_models or {})[MODEL_ID]
        condition = f"automation_{label.replace(' ', '_')}"
        df = pd.read_csv(out_csv)
        if "replicate_id" in df.columns:
            df["replicate_id"] = df["replicate_id"].astype(int)
        df = df[~((df["model_id"] == MODEL_ID) & (df["replicate_id"] == 0))]
        row = pd.DataFrame(
            [
                {
                    "replicate_id": 0,
                    "output": raw_good,
                    "condition": condition,
                    "model_id": MODEL_ID,
                    "model_label": label,
                }
            ]
        )
        pd.concat([df, row], ignore_index=True).to_csv(out_csv, index=False)
        print(f"UPSERTED clean GPT-5-Mini meal row via {winner}", flush=True)
    else:
        print("FAILED all GPT-5-Mini meal token/service variants", flush=True)

    report = audit_all_outputs(RUN_ID)
    write_json(Path(f"results/logs/audit_{RUN_ID}_gpt5_meal_token_probe.json"), report)
    print(f"AUDIT {report['n_ok']}/{report['n_rows']} ok failing={report['n_failing']}", flush=True)
    for failure in report.get("failures", []):
        print(
            "FAIL",
            failure["task"],
            failure["mode"],
            failure["model_label"],
            failure["quality_issues"],
            failure["truncation_signals"],
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Try repairing tax_prep GPT-5-Mini automation by forcing the openai service route."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from edsl import Agent, Model, QuestionFreeText, Scenario, ScenarioList, Survey

from audit_all_outputs import audit_all_outputs
from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.edsl_runtime import edsl_run_kwargs
from centaur_benchmark.io import ensure_run_dir, write_json
from output_quality import audit_output_row


RUN_ID = "20260612_fresh_rep1"
TASK = "tax_prep"
MODEL_ID = "gpt-5-mini-2025-08-07"


def try_model(model_name: str, service_name: str | None, attempt: int) -> str | None:
    task = load_task(default_tasks_dir() / f"{TASK}.yaml")
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
    kwargs = {"temperature": 0.5}
    if service_name:
        kwargs["service_name"] = service_name
    print(f"TRY model={model_name} service={service_name or 'default'} attempt={attempt}", flush=True)
    results = (
        survey.by(scenario)
        .by(agent)
        .by(Model(model_name, **kwargs))
        .run(
            **edsl_run_kwargs(
                description=f"{RUN_ID}-tax-gpt5mini-{service_name or 'default'}-{model_name}-{attempt}",
                visibility=task.remote_inference_visibility,
                n=1,
            )
        )
    )
    raw = results.select("answer.output").to_list()[0]
    audit = audit_output_row(raw, task_slug=TASK, mode="automation", condition=condition)
    print(f"chars={audit['n_chars']} ok={audit['ok']} issues={audit['issues']}", flush=True)
    return raw if audit["ok"] else None


def main() -> None:
    task = load_task(default_tasks_dir() / f"{TASK}.yaml")
    root = ensure_run_dir(TASK, RUN_ID)
    label = (task.automation_models or {})[MODEL_ID]
    condition = f"automation_{label.replace(' ', '_')}"
    candidates = [
        ("gpt-5-mini-2025-08-07", "openai"),
        ("gpt-5-mini", "openai"),
        ("gpt-5-mini-2025-08-07", None),
    ]
    raw_good: str | None = None
    for model_name, service_name in candidates:
        for attempt in range(1, 4):
            try:
                raw_good = try_model(model_name, service_name, attempt)
            except Exception as exc:  # noqa: BLE001
                print(f"EXCEPTION {type(exc).__name__}: {exc}", flush=True)
                continue
            if raw_good:
                break
        if raw_good:
            break

    if raw_good:
        out_csv = root / "automation" / "outputs.csv"
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
        print("UPSERTED clean GPT-5-Mini tax row", flush=True)
    else:
        print("FAILED to repair GPT-5-Mini tax with forced routes", flush=True)

    report = audit_all_outputs(RUN_ID)
    write_json(Path(f"results/logs/audit_{RUN_ID}_gpt5_tax_force_openai.json"), report)
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

#!/usr/bin/env python3
"""One-off no-cap repair for meal_plan DeepSeek automation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from edsl import Agent, Model, QuestionFreeText, Scenario, ScenarioList, Survey

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.edsl_runtime import edsl_run_kwargs
from centaur_benchmark.io import ensure_run_dir, write_json
from output_quality import audit_output_row
from audit_all_outputs import audit_all_outputs


RUN_ID = "20260612_fresh_rep1"
TASK = "meal_plan"
MODEL_ID = "deepseek-ai/DeepSeek-V3.1"


def main() -> None:
    task = load_task(default_tasks_dir() / f"{TASK}.yaml")
    root = ensure_run_dir(TASK, RUN_ID)
    label = (task.automation_models or {})[MODEL_ID]
    condition = f"automation_{label.replace(' ', '_')}"
    q = QuestionFreeText("output", "{{ scenario.task_prompt }}")
    survey = Survey([q])
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
    agent = Agent(
        instruction=task.automation_worker_instruction
        or "Complete the task clearly and professionally. Return only the final deliverable."
    )
    for attempt in range(1, 7):
        print(f"NO-CAP DeepSeek meal_plan attempt {attempt}", flush=True)
        try:
            results = (
                survey.by(scenario)
                .by(agent)
                .by(Model(MODEL_ID, temperature=0.5))
                .run(
                    **edsl_run_kwargs(
                        description=f"{RUN_ID}-meal-plan-deepseek-no-cap-{attempt}",
                        visibility=task.remote_inference_visibility,
                        n=1,
                    )
                )
            )
            raw = results.select("answer.output").to_list()[0]
        except Exception as exc:  # noqa: BLE001
            print(f"EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            continue
        audit = audit_output_row(raw, task_slug=TASK, mode="automation", condition=condition)
        print(f"chars={audit['n_chars']} ok={audit['ok']} issues={audit['issues']}", flush=True)
        if not audit["ok"]:
            continue
        out_csv = root / "automation" / "outputs.csv"
        df = pd.read_csv(out_csv)
        if "replicate_id" in df.columns:
            df["replicate_id"] = df["replicate_id"].astype(int)
        df = df[~((df["model_id"] == MODEL_ID) & (df["replicate_id"] == 0))]
        row = pd.DataFrame(
            [
                {
                    "replicate_id": 0,
                    "output": raw,
                    "condition": condition,
                    "model_id": MODEL_ID,
                    "model_label": label,
                }
            ]
        )
        pd.concat([df, row], ignore_index=True).to_csv(out_csv, index=False)
        print("UPSERTED clean DeepSeek meal_plan row", flush=True)
        break

    report = audit_all_outputs(RUN_ID)
    write_json(Path(f"results/logs/audit_{RUN_ID}_meal_deepseek_no_cap.json"), report)
    print(f"AUDIT {report['n_ok']}/{report['n_rows']} ok failing={report['n_failing']}", flush=True)
    for failure in report.get("failures", []):
        print("FAIL", failure["task"], failure["mode"], failure["model_label"], failure["quality_issues"], failure["truncation_signals"], flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Try exact DeepInfra route variants for DeepSeek-V3.1 meal planning."""

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
MODEL_ID = "deepseek-ai/DeepSeek-V3.1"


def _visual_ok(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 5000:
        return False
    if stripped[-1] not in ".!?)]}\"'’”*":
        return False
    low = stripped[-500:].lower()
    if "grocery" not in stripped.lower() or "prep" not in stripped.lower():
        return False
    if " day 7 " in low and ("make" in low or "roast chicken" in low):
        return False
    return True


def _try(task, kwargs: dict[str, Any], tag: str) -> str | None:
    label = (task.automation_models or {})[MODEL_ID]
    condition = f"automation_{label.replace(' ', '_')}"
    scenario = ScenarioList(
        [Scenario({"task_prompt": task.task_prompt, "condition": condition, "replicate_id": 0})]
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
                description=f"{RUN_ID}-deepseek-meal-exact-{tag}",
                visibility=task.remote_inference_visibility,
                n=1,
            )
        )
    )
    raw = results.select("answer.output").to_list()[0]
    audit = audit_output_row(raw, task_slug=TASK, mode="automation", condition=condition)
    print(f"RESULT chars={audit['n_chars']} ok={audit['ok']} issues={audit['issues']} visual={_visual_ok(raw)}", flush=True)
    return raw if audit["ok"] and _visual_ok(raw) else None


def main() -> None:
    task = load_task(default_tasks_dir() / f"{TASK}.yaml")
    variants = [
        ("deepinfra_6000", {"service_name": "deep_infra", "max_tokens": 6000, "temperature": 0.5}),
        ("deepinfra_12000", {"service_name": "deep_infra", "max_tokens": 12000, "temperature": 0.5}),
        ("deepinfra_24000", {"service_name": "deep_infra", "max_tokens": 24000, "temperature": 0.5}),
        ("deepinfra_default", {"service_name": "deep_infra", "temperature": 0.5}),
    ]
    clean = None
    winner = None
    for tag, kwargs in variants:
        try:
            clean = _try(task, kwargs, tag)
        except Exception as exc:  # noqa: BLE001
            print(f"EXCEPTION {tag}: {type(exc).__name__}: {exc}", flush=True)
            clean = None
        if clean:
            winner = tag
            break
    if clean:
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
                    "output": clean,
                    "condition": condition,
                    "model_id": MODEL_ID,
                    "model_label": label,
                }
            ]
        )
        pd.concat([df, row], ignore_index=True).to_csv(out_csv, index=False)
        print(f"UPSERTED via {winner}", flush=True)
    else:
        print("FAILED exact-route DeepSeek meal repair", flush=True)

    report = audit_all_outputs(RUN_ID)
    write_json(Path(f"results/logs/audit_{RUN_ID}_deepseek_meal_exact_route.json"), report)
    print(f"AUDIT {report['n_ok']}/{report['n_rows']} ok failing={report['n_failing']}", flush=True)


if __name__ == "__main__":
    main()

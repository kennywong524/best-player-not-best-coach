"""Optional rubric-based numeric grading (single judge model)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from centaur_benchmark.config import TaskConfig
from centaur_benchmark.edsl_runtime import edsl_run_kwargs

_RUBRIC_Q = """
Task / evaluation context:
{{ scenario.task_context }}

Rubric:
___RUBRIC___

Candidate output to score:
{{ scenario.candidate }}

Return only the numeric score (no other text).
"""


def grade_outputs_rubric(
    task: TaskConfig,
    run_root: Path,
    *,
    subset: str = "augmentation",
    eval_model: str | None = None,
) -> Path:
    """Score each row's output using task.rubric_prompt; writes rubric_scores.csv."""
    if not task.rubric_prompt:
        raise ValueError("task.rubric_prompt is not set; nothing to grade.")

    from edsl import Agent, Model, Scenario, ScenarioList, Survey, QuestionFreeText

    eval_model = eval_model or task.default_evaluator
    inp = run_root / subset / "outputs.csv"
    if not inp.exists():
        raise FileNotFoundError(f"Missing {inp}")
    df = pd.read_csv(inp)
    df = df[df["output"].notna()].reset_index(drop=True)

    rubric_block = task.rubric_prompt.strip()
    question_text = _RUBRIC_Q.replace("___RUBRIC___", rubric_block)

    scenarios = ScenarioList(
        [
            Scenario(
                {
                    "row_id": int(i),
                    "task_context": task.pairwise_task_context,
                    "candidate": str(row["output"]),
                }
            )
            for i, row in df.iterrows()
        ]
    )
    agent = Agent(
        instruction=(
            "You are an evaluator. Read the rubric carefully. "
            "Respond with ONLY a single number on the scale requested in the rubric."
        )
    )
    q = QuestionFreeText(question_name="score", question_text=question_text)
    survey = Survey([q])
    results = (
        survey.by(scenarios)
        .by(agent)
        .by(Model(eval_model))
        .run(
            **edsl_run_kwargs(
                description=f"centaur-rubric-{task.slug}-{subset}",
                visibility=task.remote_inference_visibility,
                progress_bar=True,
                verbose=True,
                n=1,
            ),
        )
    )
    out = results.select("scenario.row_id", "answer.score").to_pandas()
    out.columns = ["row_id", "rubric_score_raw"]
    merged = df.reset_index(drop=True).reset_index().rename(columns={"index": "_idx"})
    merged = merged.merge(out, left_on="_idx", right_on="row_id", how="left")
    outp = run_root / subset / "rubric_scores.csv"
    merged.to_csv(outp, index=False)
    print(f"Wrote {outp}")
    return outp

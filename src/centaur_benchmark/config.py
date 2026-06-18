"""Load task YAML definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskConfig:
    """One benchmark task: prompts, models, and pairwise judge settings."""

    slug: str
    title: str
    task_prompt: str
    scaffold_prompt_template: str
    worker_instruction: str
    pairwise_task_context: str
    pairwise_eval_prompt: str
    default_worker: str = "gpt-3.5-turbo"
    default_evaluator: str = "gpt-5"
    replicates: int = 1
    assistants: dict[str, str] = field(default_factory=dict)
    automation_models: dict[str, str] | None = None
    evaluator_models: dict[str, str] = field(default_factory=dict)
    scaffold_model_max_tokens: int | None = None
    pairwise_n_evals: int = 3
    pairwise_n_outer_runs: int = 1
    rubric_prompt: str | None = None
    remote_inference_visibility: str = "private"
    automation_worker_instruction: str | None = None
    automation_model_max_tokens: int | None = None
    worker_model_max_tokens: int | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TaskConfig:
        models = data.get("models") or {}
        assistants = models.get("assistants") or {}
        automation = models.get("automation")
        if automation is not None and not isinstance(automation, dict):
            raise TypeError("models.automation must be a mapping or null")
        return TaskConfig(
            slug=data["slug"],
            title=data.get("title") or data["slug"],
            task_prompt=data["task_prompt"],
            scaffold_prompt_template=data["scaffold_prompt_template"],
            worker_instruction=data["worker_instruction"],
            pairwise_task_context=data.get("pairwise_task_context") or data["task_prompt"],
            pairwise_eval_prompt=data["pairwise_eval_prompt"],
            default_worker=data.get("default_worker", "gpt-3.5-turbo"),
            default_evaluator=data.get("default_evaluator", "gpt-5"),
            replicates=int(data.get("replicates", 1)),
            assistants=dict(assistants),
            automation_models=dict(automation) if automation else None,
            evaluator_models=dict(models.get("evaluators") or {}),
            scaffold_model_max_tokens=data.get("scaffold_model_max_tokens"),
            pairwise_n_evals=int(data.get("pairwise", {}).get("n_evals_per_pair", 3)),
            pairwise_n_outer_runs=int(data.get("pairwise", {}).get("n_outer_runs", 1)),
            rubric_prompt=data.get("rubric_prompt"),
            remote_inference_visibility=data.get("remote_inference_visibility", "private"),
            automation_worker_instruction=data.get("automation_worker_instruction"),
            automation_model_max_tokens=data.get("automation_model_max_tokens"),
            worker_model_max_tokens=data.get("worker_model_max_tokens"),
        )


def load_task(path: str | Path) -> TaskConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task file must be a mapping: {p}")
    return TaskConfig.from_dict(raw)


def default_tasks_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tasks"

"""EDSL runners for augmentation (scaffold + worker) and automation (model solves)."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from centaur_benchmark.config import TaskConfig
from centaur_benchmark.edsl_runtime import edsl_run_kwargs
from centaur_benchmark.io import write_json
from centaur_benchmark.scaffold_prompt import SCAFFOLD_SURVEY_QUESTION

MIN_SCAFFOLD_CHARS = 250
MAX_SCAFFOLD_CHARS = 2600
# Replicate variance needs stochastic sampling; 0.01 is effectively deterministic.
# cache=False in edsl_run_kwargs is the primary anti-cache fix (see edsl_runtime.py).
GENERATION_TEMPERATURE = 0.5
# Per-model automation output caps — Expected Parrot verified ceilings.
# Task YAML automation_model_max_tokens is an upper bound; runner uses min(yaml, cap).
# Goal: v4-style headroom (no truncation) without HTTP 500 on models EP rejects at 128k.
# Re-probe: python scripts/probe_automation_max_tokens.py
_AUTOMATION_MAX_TOKENS_CAP: dict[str, int] = {
    "gpt-3.5-turbo": 4_096,  # EP hard limit
    "gpt-4.1": 32_768,  # EP hard limit (128k → HTTP 500)
    "deepseek-ai/DeepSeek-V3.1": 128_000,
    "o4-mini-2025-04-16": 128_000,
    "o3-mini-2025-01-31": 128_000,
    "openai/gpt-oss-120b": 128_000,
    "anthropic/claude-sonnet-4-6": 128_000,
    "anthropic/claude-opus-4-8": 128_000,
    "google/gemini-3.1-pro": 65_536,  # EP rejects 128k
}
_DEFAULT_AUTOMATION_MAX_TOKENS = 128_000
# Fresh local-proxy calls time out when max_tokens is set to the provider ceiling but
# deliverables are much shorter (counselling ~3–5k chars). Cap the request per task.
_TASK_AUTOMATION_REQUEST_CAP: dict[str, int] = {
    "counselling": 16_384,
    "market_trends": 16_384,
    "tutoring": 16_384,
    "operations_research": 24_576,
    "travel_planning": 32_768,
    "meal_plan": 32_768,
    "tax_prep": 12_000,
}
_GPT5_AUTOMATION_TASK_MAX_TOKENS: dict[str, int] = {
    # Default GPT-5-Mini calls can return empty outputs/format stubs on these
    # longer deliverables. These values passed in 20260612_fresh_rep1 repairs.
    "counselling": 12_000,
    "meal_plan": 6_000,
    "tax_prep": 12_000,
    "travel_planning": 8_000,
    "tutoring": 8_000,
}
_AUTOMATION_MODEL_TASK_MAX_TOKENS: dict[tuple[str, str], int] = {
    # DeepSeek meal planning failed or truncated with generic high/default caps;
    # exact deep_infra route with 6k passed in 20260612_fresh_rep1.
    ("deepseek-ai/DeepSeek-V3.1", "meal_plan"): 6_000,
    ("deepseek-ai/DeepSeek-V3.1", "tax_prep"): 6_000,
}


def _generation_model_kwargs(model_id: str, base: dict[str, Any] | None = None) -> dict[str, Any]:
    kwargs = dict(base or {})
    if "temperature" not in kwargs:
        kwargs["temperature"] = GENERATION_TEMPERATURE
    return kwargs


def _cache_bust_description(desc: str) -> str:
    """Prefix EDSL job descriptions with CENTAUR_RUN_ID so separate runs cannot hit v4 cache."""
    run_id = os.environ.get("CENTAUR_RUN_ID", "").strip()
    if run_id:
        return f"{run_id}-{desc}"[:200]
    return desc[:200]


def _automation_model_kwargs(
    task: TaskConfig,
    model_id: str,
    *,
    attempt: int = 0,
) -> dict[str, Any]:
    """Model kwargs for automation runs; caps max_tokens per provider + task headroom."""
    if "gpt-5" in model_id:
        cap = _GPT5_AUTOMATION_TASK_MAX_TOKENS.get(task.slug)
        if cap is None:
            return {}
        return {"max_tokens": cap}
    special_cap = _AUTOMATION_MODEL_TASK_MAX_TOKENS.get((model_id, task.slug))
    if special_cap is not None:
        return {"max_tokens": special_cap}
    if task.automation_model_max_tokens is None:
        return {}
    model_cap = _AUTOMATION_MAX_TOKENS_CAP.get(model_id, _DEFAULT_AUTOMATION_MAX_TOKENS)
    cap = min(int(task.automation_model_max_tokens), model_cap)
    task_cap = _TASK_AUTOMATION_REQUEST_CAP.get(task.slug)
    if task_cap is not None:
        cap = min(cap, task_cap)
    if attempt > 0:
        cap = max(4_096, cap // (2**attempt))
    return {"max_tokens": cap}


def _strip_reasoning_wrappers(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"(?is)<think>.*?</think>\s*", "", t)
    t = re.sub(r"(?is)<think>.*\Z", "", t)
    return t.strip()


def _compose_worker_prompt(scaffold: str, task_prompt: str) -> str:
    task_specific_completion = ""
    if "Assume tax year 2025 and state: California" in task_prompt:
        task_specific_completion = (
            "\n\nTAX REVIEW COMPLETENESS REQUIREMENT:\n"
            "Write a complete, sectioned tax-preparation review, not a short client letter. "
            "Include: (1) discrepancy list, (2) federal and California rules, "
            "(3) recalculation framework or estimates using stated assumptions, "
            "(4) forms/schedules to correct, (5) dependent analysis, and "
            "(6) missing information needed for exact liability."
        )
    return (
        "INTERNAL ASSISTANT GUIDANCE (planning only — do not copy or restate in your output):\n"
        f"{scaffold}\n\n"
        "---\n\n"
        "CLIENT TASK (write the complete final deliverable for this request only):\n"
        f"{task_prompt}"
        f"{task_specific_completion}"
    )


def _safe_slug(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("\\", "_")
        .replace("|", "_")
    )


def _scaffold_validation_errors(text: str) -> list[str]:
    out = _strip_reasoning_wrappers(text)
    if not out:
        return ["empty"]

    errors: list[str] = []
    low = (
        out.lower()
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    if out.startswith("{'format':") or out.startswith('{"format":'):
        errors.append("format_stub")
    if len(out) < MIN_SCAFFOLD_CHARS:
        errors.append(f"short<{MIN_SCAFFOLD_CHARS}")
    if len(out) > MAX_SCAFFOLD_CHARS:
        errors.append(f"too_long>{MAX_SCAFFOLD_CHARS}")
    normalized_start = low.lstrip("*# \n")
    if "assistant guidance: three-phase workflow" not in low:
        errors.append("missing_required_heading")
    elif not normalized_start.startswith("assistant guidance: three-phase workflow"):
        errors.append("heading_not_first")

    return errors


def _generate_scaffold(
    task: TaskConfig,
    task_prompt: str,
    assistant_model: str,
    *,
    replicate_id: int | None = None,
    cache_attempt: int = 0,
) -> str:
    from edsl import Agent, Model, Scenario, Survey, QuestionFreeText

    assistant_agent = Agent(instruction=task.scaffold_prompt_template)
    scaffold_question = SCAFFOLD_SURVEY_QUESTION
    if "gemini" in assistant_model:
        scaffold_question += """
Gemini-specific constraints:
- Do NOT output <think> tags or any hidden reasoning blocks.
- Your response must begin with exactly: **Assistant Guidance: Three-Phase Workflow**
- Finish all three phases completely before stopping.
"""
    rep = replicate_id if replicate_id is not None else 0
    q = QuestionFreeText("scaffold", scaffold_question)
    survey = Survey([q])
    scenario = Scenario({"task_prompt": task_prompt})
    kwargs: dict[str, Any] = {}
    # EDSL currently returns a format-parameter stub for GPT-5-style models when
    # max_tokens is supplied through Model(...). Let those models use defaults;
    # the scaffold validator still rejects overlong or malformed outputs.
    if "gpt-5" in assistant_model:
        pass  # GPT-5 returns format stubs when max_tokens is set via Model(...)
    elif "gemini" in assistant_model:
        kwargs["max_tokens"] = 1200  # headroom after occasional thinking wrappers
    elif task.scaffold_model_max_tokens is not None:
        kwargs["max_tokens"] = int(task.scaffold_model_max_tokens)
    model = Model(assistant_model, **_generation_model_kwargs(assistant_model, kwargs))
    print(f"Generating scaffold from {assistant_model} rep={rep}...")
    last_text = ""
    last_errors: list[str] = []
    max_attempts = 5 if "gemini" in assistant_model else 3
    for attempt in range(1, max_attempts + 1):
        if replicate_id is not None:
            desc = (
                f"{task.slug}-scaffold-{assistant_model}-rep{replicate_id}"
                f"-a{cache_attempt}-v{attempt}"
            )
        else:
            desc = f"centaur-scaffold-{assistant_model}-attempt-{attempt}"
        results = survey.by(scenario).by(assistant_agent).by(model).run(
            **edsl_run_kwargs(
                description=_cache_bust_description(desc),
                visibility=task.remote_inference_visibility,
            ),
        )
        scaffold_text = _strip_reasoning_wrappers(results.select("scaffold").to_list()[0] or "")
        errors = _scaffold_validation_errors(scaffold_text)
        if not errors:
            print(f"Scaffold OK ({len(scaffold_text or '')} chars)")
            return scaffold_text

        last_text = scaffold_text
        last_errors = errors
        print(
            f"Scaffold rejected from {assistant_model} attempt={attempt} "
            f"chars={len(scaffold_text or '')} errors={errors}"
        )

    preview = str(last_text or "").strip().replace("\n", " ")[:200]
    raise RuntimeError(
        f"Failed to generate valid scaffold from {assistant_model}; "
        f"errors={last_errors}; preview={preview!r}"
    )


def _run_worker_batch(
    task: TaskConfig,
    *,
    worker_model: str,
    condition: str,
    full_prompts: list[str],
    remote_description: str,
    remote_visibility: str,
    replicate_id: int = 0,
) -> pd.DataFrame:
    from edsl import Agent, Model, Scenario, ScenarioList, Survey, QuestionFreeText

    scenario_rows: list[dict[str, Any]] = []
    for i, fp in enumerate(full_prompts):
        rep = replicate_id if len(full_prompts) == 1 else i
        scenario_rows.append(
            {"task_prompt": fp, "condition": condition, "replicate_id": rep}
        )
    scenarios = ScenarioList([Scenario(row) for row in scenario_rows])
    q = QuestionFreeText("output", "{{ scenario.task_prompt }}")
    survey = Survey([q])
    worker = Agent(instruction=task.worker_instruction)
    wkwargs: dict[str, Any] = {}
    if task.worker_model_max_tokens is not None:
        wkwargs["max_tokens"] = int(task.worker_model_max_tokens)
    interactive = sys.stdout.isatty()
    print(f"Running worker={worker_model} condition={condition} n={len(full_prompts)} rep={replicate_id}...")
    results = (
        survey.by(scenarios)
        .by(worker)
        .by(Model(worker_model, **_generation_model_kwargs(worker_model, wkwargs)))
        .run(
            **edsl_run_kwargs(
                description=_cache_bust_description(remote_description),
                visibility=remote_visibility,
                progress_bar=interactive,
                verbose=interactive,
                n=1,
            ),
        )
    )
    df = results.select("scenario.replicate_id", "answer.output", "scenario.condition").to_pandas()
    df.rename(
        columns={
            "scenario.replicate_id": "replicate_id",
            "answer.output": "output",
            "scenario.condition": "condition",
        },
        inplace=True,
    )
    return df


def run_augmentation(
    task: TaskConfig,
    run_root: Path,
    *,
    worker_model: str | None = None,
    assistants: dict[str, str] | None = None,
    replicates: int | None = None,
) -> Path:
    """Plain baseline + one scaffold per assistant model. Writes augmentation/outputs.csv."""
    worker_model = worker_model or task.default_worker
    assistants = assistants or task.assistants
    n_rep = replicates if replicates is not None else task.replicates
    if not assistants:
        raise ValueError("No assistants configured for this task.")

    out_csv = run_root / "augmentation" / "outputs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scaffolds_dir = run_root / "augmentation" / "scaffolds"
    scaffolds_dir.mkdir(parents=True, exist_ok=True)
    scaffolds_records: list[dict[str, Any]] = []

    all_parts: list[pd.DataFrame] = []

    plain_parts: list[pd.DataFrame] = []
    for rep in range(n_rep):
        plain_df = _run_worker_batch(
            task,
            worker_model=worker_model,
            condition="plain",
            full_prompts=[task.task_prompt],
            remote_description=f"{task.slug}-{worker_model}-plain-rep{rep}",
            remote_visibility=task.remote_inference_visibility,
            replicate_id=rep,
        )
        plain_df["replicate_id"] = rep
        plain_parts.append(plain_df)
    plain_df = pd.concat(plain_parts, ignore_index=True)
    plain_df["assistant_model"] = "plain"
    plain_df["worker_model"] = worker_model
    plain_df["model_id"] = "plain"
    plain_df["model_label"] = "plain"
    plain_df["scaffold_path"] = ""
    plain_df["scaffold_sha256"] = ""
    plain_df["scaffold_text"] = ""
    all_parts.append(plain_df)

    for model_id, model_label in assistants.items():
        cond = f"scaffold_{_safe_slug(model_label)}"
        scaffold_parts: list[pd.DataFrame] = []
        for rep in range(n_rep):
            scaffold = _generate_scaffold(task, task.task_prompt, model_id, replicate_id=rep)
            scaffold_path = scaffolds_dir / f"{_safe_slug(model_label)}_rep{rep}.md"
            scaffold_path.write_text(scaffold, encoding="utf-8")
            scaffold_sha256 = hashlib.sha256(scaffold.encode("utf-8")).hexdigest()
            scaffolds_records.append(
                {
                    "model_id": model_id,
                    "model_label": model_label,
                    "replicate_id": rep,
                    "scaffold_path": str(scaffold_path),
                    "scaffold_sha256": scaffold_sha256,
                    "n_chars": len(scaffold),
                }
            )
            worker_prompt = _compose_worker_prompt(scaffold, task.task_prompt)
            sdf = _run_worker_batch(
                task,
                worker_model=worker_model,
                condition=cond,
                full_prompts=[worker_prompt],
                remote_description=f"{task.slug}-{worker_model}-{cond}-rep{rep}",
                remote_visibility=task.remote_inference_visibility,
                replicate_id=rep,
            )
            sdf["replicate_id"] = rep
            sdf["scaffold_path"] = str(scaffold_path)
            sdf["scaffold_sha256"] = scaffold_sha256
            sdf["scaffold_text"] = scaffold
            scaffold_parts.append(sdf)
        sdf = pd.concat(scaffold_parts, ignore_index=True)
        sdf["assistant_model"] = model_label
        sdf["worker_model"] = worker_model
        sdf["model_id"] = model_id
        sdf["model_label"] = model_label
        all_parts.append(sdf)

    if scaffolds_records:
        pd.DataFrame(scaffolds_records).to_csv(run_root / "augmentation" / "scaffolds.csv", index=False)

    final_df = pd.concat(all_parts, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(final_df)} rows)")
    return out_csv


def _run_automation_single_model(
    task: TaskConfig,
    *,
    model_id: str,
    model_label: str,
    prompts: list[str],
    remote_description: str,
    remote_visibility: str,
    replicate_id: int = 0,
    attempt: int = 0,
) -> pd.DataFrame:
    from edsl import Agent, Model, Scenario, ScenarioList, Survey, QuestionFreeText

    condition = f"automation_{model_label.replace(' ', '_')}"
    scenario_rows: list[dict[str, Any]] = []
    for i, fp in enumerate(prompts):
        rep = replicate_id if len(prompts) == 1 else i
        scenario_rows.append(
            {"task_prompt": fp, "condition": condition, "replicate_id": rep}
        )
    scenarios = ScenarioList([Scenario(row) for row in scenario_rows])
    q = QuestionFreeText("output", "{{ scenario.task_prompt }}")
    survey = Survey([q])
    auto_instr = task.automation_worker_instruction or (
        "Complete the task clearly and professionally. Return only the final deliverable."
    )
    worker = Agent(instruction=auto_instr)
    mkwargs = _automation_model_kwargs(task, model_id, attempt=attempt)
    max_tok = mkwargs.get("max_tokens", "default")
    print(
        f"Running automation model={model_id} n={len(prompts)} rep={replicate_id} "
        f"max_tokens={max_tok} attempt={attempt}..."
    )
    results = (
        survey.by(scenarios)
        .by(worker)
        .by(Model(model_id, **_generation_model_kwargs(model_id, mkwargs)))
        .run(
            **edsl_run_kwargs(
                description=_cache_bust_description(remote_description),
                visibility=remote_visibility,
                n=1,
            ),
        )
    )
    df = results.select("scenario.replicate_id", "answer.output", "scenario.condition").to_pandas()
    df.rename(
        columns={
            "scenario.replicate_id": "replicate_id",
            "answer.output": "output",
            "scenario.condition": "condition",
        },
        inplace=True,
    )
    df["model_id"] = model_id
    df["model_label"] = model_label
    return df


def run_automation(
    task: TaskConfig,
    run_root: Path,
    *,
    models: dict[str, str] | None = None,
    replicates: int | None = None,
) -> Path | None:
    """Each model solves the task directly. Writes automation/outputs.csv."""
    models = models or task.automation_models
    if not models:
        print("No automation_models in task config; skipping automation.")
        return None
    n_rep = replicates if replicates is not None else task.replicates
    out_csv = run_root / "automation" / "outputs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    for model_id, model_label in models.items():
        rep_parts: list[pd.DataFrame] = []
        for rep in range(n_rep):
            df = _run_automation_single_model(
                task,
                model_id=model_id,
                model_label=model_label,
                prompts=[task.task_prompt],
                remote_description=f"{task.slug}-automation-{model_id}-rep{rep}",
                remote_visibility=task.remote_inference_visibility,
                replicate_id=rep,
            )
            df["replicate_id"] = rep
            rep_parts.append(df)
        parts.append(pd.concat(rep_parts, ignore_index=True))
    final_df = pd.concat(parts, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(final_df)} rows)")
    return out_csv


def patch_augmentation_models(
    task: TaskConfig,
    run_root: Path,
    model_ids: list[str],
    *,
    worker_model: str | None = None,
    replicates: int | None = None,
) -> Path:
    """Regenerate scaffolds + worker outputs for specific assistant models and merge into outputs.csv."""
    worker_model = worker_model or task.default_worker
    n_rep = replicates if replicates is not None else task.replicates
    out_csv = run_root / "augmentation" / "outputs.csv"
    if not out_csv.exists():
        raise FileNotFoundError(f"Missing {out_csv}")
    existing = pd.read_csv(out_csv)
    scaffolds_dir = run_root / "augmentation" / "scaffolds"
    scaffolds_dir.mkdir(parents=True, exist_ok=True)
    scaffolds_records: list[dict[str, Any]] = []
    if (run_root / "augmentation" / "scaffolds.csv").exists():
        scaffolds_records = pd.read_csv(run_root / "augmentation" / "scaffolds.csv").to_dict("records")

    new_parts: list[pd.DataFrame] = []
    for model_id in model_ids:
        model_label = task.assistants.get(model_id)
        if not model_label:
            raise ValueError(f"Unknown assistant model_id: {model_id}")
        scaffold = _generate_scaffold(task, task.task_prompt, model_id)
        scaffold_path = scaffolds_dir / f"{_safe_slug(model_label)}.md"
        scaffold_path.write_text(scaffold, encoding="utf-8")
        scaffold_sha256 = hashlib.sha256(scaffold.encode("utf-8")).hexdigest()
        rec = {
            "model_id": model_id,
            "model_label": model_label,
            "scaffold_path": str(scaffold_path),
            "scaffold_sha256": scaffold_sha256,
            "n_chars": len(scaffold),
        }
        scaffolds_records = [r for r in scaffolds_records if r.get("model_id") != model_id]
        scaffolds_records.append(rec)

        cond = f"scaffold_{_safe_slug(model_label)}"
        worker_prompt = _compose_worker_prompt(scaffold, task.task_prompt)
        scaffold_parts: list[pd.DataFrame] = []
        for rep in range(n_rep):
            sdf = _run_worker_batch(
                task,
                worker_model=worker_model,
                condition=cond,
                full_prompts=[worker_prompt],
                remote_description=f"{task.slug}-retry-{worker_model}-{cond}-rep{rep}",
                remote_visibility=task.remote_inference_visibility,
                replicate_id=rep,
            )
            sdf["replicate_id"] = rep
            scaffold_parts.append(sdf)
        sdf = pd.concat(scaffold_parts, ignore_index=True)
        sdf["assistant_model"] = model_label
        sdf["worker_model"] = worker_model
        sdf["model_id"] = model_id
        sdf["model_label"] = model_label
        sdf["scaffold_path"] = str(scaffold_path)
        sdf["scaffold_sha256"] = scaffold_sha256
        sdf["scaffold_text"] = scaffold
        new_parts.append(sdf)

    patched = existing[~existing["model_id"].isin(model_ids)].copy()
    if new_parts:
        patched = pd.concat([patched, *new_parts], ignore_index=True)
    patched.to_csv(out_csv, index=False)
    if scaffolds_records:
        pd.DataFrame(scaffolds_records).to_csv(run_root / "augmentation" / "scaffolds.csv", index=False)
    print(f"Patched augmentation models {model_ids} -> {out_csv}")
    return out_csv


def patch_augmentation_worker_outputs(
    task: TaskConfig,
    run_root: Path,
    model_ids: list[str],
    *,
    worker_model: str | None = None,
    replicates: int | None = None,
) -> Path:
    """Re-run worker only, keeping existing scaffolds on disk."""
    worker_model = worker_model or task.default_worker
    n_rep = replicates if replicates is not None else task.replicates
    out_csv = run_root / "augmentation" / "outputs.csv"
    if not out_csv.exists():
        raise FileNotFoundError(f"Missing {out_csv}")
    existing = pd.read_csv(out_csv)
    scaffolds_dir = run_root / "augmentation" / "scaffolds"

    new_parts: list[pd.DataFrame] = []
    for model_id in model_ids:
        model_label = task.assistants.get(model_id)
        if not model_label:
            raise ValueError(f"Unknown assistant model_id: {model_id}")
        scaffold_path = scaffolds_dir / f"{_safe_slug(model_label)}.md"
        if not scaffold_path.is_file():
            raise FileNotFoundError(f"Missing scaffold {scaffold_path}")
        scaffold = scaffold_path.read_text(encoding="utf-8")
        scaffold_sha256 = hashlib.sha256(scaffold.encode("utf-8")).hexdigest()
        cond = f"scaffold_{_safe_slug(model_label)}"
        worker_prompt = _compose_worker_prompt(scaffold, task.task_prompt)
        scaffold_parts: list[pd.DataFrame] = []
        for rep in range(n_rep):
            sdf = _run_worker_batch(
                task,
                worker_model=worker_model,
                condition=cond,
                full_prompts=[worker_prompt],
                remote_description=f"{task.slug}-worker-retry-{worker_model}-{cond}-rep{rep}",
                remote_visibility=task.remote_inference_visibility,
                replicate_id=rep,
            )
            sdf["replicate_id"] = rep
            scaffold_parts.append(sdf)
        sdf = pd.concat(scaffold_parts, ignore_index=True)
        sdf["assistant_model"] = model_label
        sdf["worker_model"] = worker_model
        sdf["model_id"] = model_id
        sdf["model_label"] = model_label
        sdf["scaffold_path"] = str(scaffold_path)
        sdf["scaffold_sha256"] = scaffold_sha256
        sdf["scaffold_text"] = scaffold
        new_parts.append(sdf)

    patched = existing[~existing["model_id"].isin(model_ids)].copy()
    if new_parts:
        patched = pd.concat([patched, *new_parts], ignore_index=True)
    patched.to_csv(out_csv, index=False)
    print(f"Patched augmentation worker outputs for {model_ids} -> {out_csv}")
    return out_csv


def patch_augmentation_plain_outputs(
    task: TaskConfig,
    run_root: Path,
    *,
    worker_model: str | None = None,
    replicates: int | None = None,
) -> Path:
    """Re-run plain baseline worker outputs (no scaffold)."""
    worker_model = worker_model or task.default_worker
    n_rep = replicates if replicates is not None else task.replicates
    out_csv = run_root / "augmentation" / "outputs.csv"
    if not out_csv.exists():
        raise FileNotFoundError(f"Missing {out_csv}")
    existing = pd.read_csv(out_csv)
    plain_parts: list[pd.DataFrame] = []
    for rep in range(n_rep):
        plain_df = _run_worker_batch(
            task,
            worker_model=worker_model,
            condition="plain",
            full_prompts=[task.task_prompt],
            remote_description=f"{task.slug}-plain-retry-{worker_model}-rep{rep}",
            remote_visibility=task.remote_inference_visibility,
            replicate_id=rep,
        )
        plain_df["replicate_id"] = rep
        plain_parts.append(plain_df)
    plain_df = pd.concat(plain_parts, ignore_index=True)
    plain_df["assistant_model"] = "plain"
    plain_df["worker_model"] = worker_model
    plain_df["model_id"] = "plain"
    plain_df["model_label"] = "plain"
    plain_df["scaffold_path"] = ""
    plain_df["scaffold_sha256"] = ""
    plain_df["scaffold_text"] = ""
    patched = existing[existing["model_id"] != "plain"].copy()
    patched = pd.concat([patched, plain_df], ignore_index=True)
    patched.to_csv(out_csv, index=False)
    print(f"Patched augmentation plain baseline -> {out_csv}")
    return out_csv


def patch_automation_models(
    task: TaskConfig,
    run_root: Path,
    model_ids: list[str],
    *,
    replicates: int | None = None,
) -> Path:
    """Regenerate automation outputs for specific models and merge into outputs.csv."""
    if not task.automation_models:
        raise ValueError("No automation_models configured")
    n_rep = replicates if replicates is not None else task.replicates
    out_csv = run_root / "automation" / "outputs.csv"
    if not out_csv.exists():
        raise FileNotFoundError(f"Missing {out_csv}")
    existing = pd.read_csv(out_csv)
    new_parts: list[pd.DataFrame] = []
    for model_id in model_ids:
        model_label = task.automation_models.get(model_id)
        if not model_label:
            raise ValueError(f"Unknown automation model_id: {model_id}")
        rep_parts: list[pd.DataFrame] = []
        for rep in range(n_rep):
            rep_parts.append(
                _run_automation_single_model(
                    task,
                    model_id=model_id,
                    model_label=model_label,
                    prompts=[task.task_prompt],
                    remote_description=f"{task.slug}-automation-retry-{model_id}-rep{rep}",
                    remote_visibility=task.remote_inference_visibility,
                    replicate_id=rep,
                ).assign(replicate_id=rep)
            )
        df = pd.concat(rep_parts, ignore_index=True)
        valid = df["output"].notna() & (df["output"].astype(str).str.strip().str.len() > 50)
        if not valid.all():
            bad = int((~valid).sum())
            print(
                f"WARNING: skipping patch for {model_id}: "
                f"{bad}/{len(df)} empty/invalid outputs (keeping existing rows)"
            )
            continue
        new_parts.append(df)

    patched = existing[~existing["model_id"].isin([d["model_id"].iloc[0] for d in new_parts])].copy()
    if new_parts:
        patched = pd.concat([patched, *new_parts], ignore_index=True)
    patched.to_csv(out_csv, index=False)
    print(f"Patched automation models {model_ids} -> {out_csv}")
    return out_csv


def _upsert_output_rows(
    out_csv: Path,
    new_rows: pd.DataFrame,
    *,
    model_id: str,
    replicate_id: int,
) -> None:
    """Replace rows for (model_id, replicate_id) and write outputs.csv."""
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        if "replicate_id" in existing.columns:
            existing["replicate_id"] = existing["replicate_id"].astype(int)
        mask = (existing["model_id"] == model_id) & (existing["replicate_id"] == replicate_id)
        existing = existing[~mask]
    else:
        existing = pd.DataFrame()
    merged = pd.concat([existing, new_rows], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)


def ensure_augmentation_scaffold(
    task: TaskConfig,
    run_root: Path,
    model_id: str,
) -> tuple[str, Path, str]:
    """Create or return validated scaffold for an assistant model."""
    model_label = task.assistants.get(model_id)
    if not model_label:
        raise ValueError(f"Unknown assistant model_id: {model_id}")
    scaffolds_dir = run_root / "augmentation" / "scaffolds"
    scaffolds_dir.mkdir(parents=True, exist_ok=True)
    scaffold_path = scaffolds_dir / f"{_safe_slug(model_label)}.md"
    if scaffold_path.is_file():
        scaffold = scaffold_path.read_text(encoding="utf-8")
        if not _scaffold_validation_errors(scaffold):
            return scaffold, scaffold_path, hashlib.sha256(scaffold.encode("utf-8")).hexdigest()

    scaffold = _generate_scaffold(task, task.task_prompt, model_id)
    scaffold_path.write_text(scaffold, encoding="utf-8")
    scaffold_sha256 = hashlib.sha256(scaffold.encode("utf-8")).hexdigest()

    scaffolds_csv = run_root / "augmentation" / "scaffolds.csv"
    records: list[dict[str, Any]] = []
    if scaffolds_csv.exists():
        records = pd.read_csv(scaffolds_csv).to_dict("records")
    records = [r for r in records if r.get("model_id") != model_id]
    records.append(
        {
            "model_id": model_id,
            "model_label": model_label,
            "scaffold_path": str(scaffold_path),
            "scaffold_sha256": scaffold_sha256,
            "n_chars": len(scaffold),
        }
    )
    pd.DataFrame(records).to_csv(scaffolds_csv, index=False)
    return scaffold, scaffold_path, scaffold_sha256


def generate_augmentation_plain_replicate(
    task: TaskConfig,
    run_root: Path,
    replicate_id: int,
    *,
    worker_model: str | None = None,
    attempt: int = 0,
) -> str:
    """Generate one plain-baseline replicate; returns raw output text."""
    worker_model = worker_model or task.default_worker
    out_csv = run_root / "augmentation" / "outputs.csv"
    plain_df = _run_worker_batch(
        task,
        worker_model=worker_model,
        condition="plain",
        full_prompts=[task.task_prompt],
        remote_description=f"{task.slug}-plain-{worker_model}-rep{replicate_id}-a{attempt}",
        remote_visibility=task.remote_inference_visibility,
        replicate_id=replicate_id,
    )
    plain_df["replicate_id"] = replicate_id
    plain_df["assistant_model"] = "plain"
    plain_df["worker_model"] = worker_model
    plain_df["model_id"] = "plain"
    plain_df["model_label"] = "plain"
    plain_df["scaffold_path"] = ""
    plain_df["scaffold_sha256"] = ""
    plain_df["scaffold_text"] = ""
    _upsert_output_rows(out_csv, plain_df, model_id="plain", replicate_id=replicate_id)
    return str(plain_df["output"].iloc[0] or "")


def generate_augmentation_worker_replicate(
    task: TaskConfig,
    run_root: Path,
    model_id: str,
    replicate_id: int,
    *,
    worker_model: str | None = None,
    attempt: int = 0,
) -> str:
    """Generate fresh scaffold + worker output for one augmentation replicate."""
    worker_model = worker_model or task.default_worker
    model_label = task.assistants.get(model_id)
    if not model_label:
        raise ValueError(f"Unknown assistant model_id: {model_id}")

    scaffold = _generate_scaffold(
        task,
        task.task_prompt,
        model_id,
        replicate_id=replicate_id,
        cache_attempt=attempt,
    )
    scaffolds_dir = run_root / "augmentation" / "scaffolds"
    scaffolds_dir.mkdir(parents=True, exist_ok=True)
    scaffold_path = scaffolds_dir / f"{_safe_slug(model_label)}_rep{replicate_id}.md"
    scaffold_path.write_text(scaffold, encoding="utf-8")
    scaffold_sha256 = hashlib.sha256(scaffold.encode("utf-8")).hexdigest()

    scaffolds_csv = run_root / "augmentation" / "scaffolds.csv"
    records: list[dict[str, Any]] = []
    if scaffolds_csv.exists():
        records = pd.read_csv(scaffolds_csv).to_dict("records")
    records = [
        r
        for r in records
        if not (r.get("model_id") == model_id and int(r.get("replicate_id", -1)) == replicate_id)
    ]
    records.append(
        {
            "model_id": model_id,
            "model_label": model_label,
            "replicate_id": replicate_id,
            "scaffold_path": str(scaffold_path),
            "scaffold_sha256": scaffold_sha256,
            "n_chars": len(scaffold),
        }
    )
    pd.DataFrame(records).to_csv(scaffolds_csv, index=False)

    cond = f"scaffold_{_safe_slug(model_label)}"
    worker_prompt = _compose_worker_prompt(scaffold, task.task_prompt)
    sdf = _run_worker_batch(
        task,
        worker_model=worker_model,
        condition=cond,
        full_prompts=[worker_prompt],
        remote_description=f"{task.slug}-{worker_model}-{cond}-rep{replicate_id}-a{attempt}",
        remote_visibility=task.remote_inference_visibility,
        replicate_id=replicate_id,
    )
    sdf["replicate_id"] = replicate_id
    sdf["assistant_model"] = model_label
    sdf["worker_model"] = worker_model
    sdf["model_id"] = model_id
    sdf["model_label"] = model_label
    sdf["scaffold_path"] = str(scaffold_path)
    sdf["scaffold_sha256"] = scaffold_sha256
    sdf["scaffold_text"] = scaffold
    out_csv = run_root / "augmentation" / "outputs.csv"
    _upsert_output_rows(out_csv, sdf, model_id=model_id, replicate_id=replicate_id)
    return str(sdf["output"].iloc[0] or "")


def generate_automation_replicate(
    task: TaskConfig,
    run_root: Path,
    model_id: str,
    replicate_id: int,
    *,
    attempt: int = 0,
) -> str:
    """Generate one automation replicate; returns raw output text."""
    model_label = (task.automation_models or {}).get(model_id)
    if not model_label:
        raise ValueError(f"Unknown automation model_id: {model_id}")
    df = _run_automation_single_model(
        task,
        model_id=model_id,
        model_label=model_label,
        prompts=[task.task_prompt],
        remote_description=f"{task.slug}-automation-{model_id}-rep{replicate_id}-a{attempt}",
        remote_visibility=task.remote_inference_visibility,
        replicate_id=replicate_id,
        attempt=attempt,
    )
    df["replicate_id"] = replicate_id
    out_csv = run_root / "automation" / "outputs.csv"
    _upsert_output_rows(out_csv, df, model_id=model_id, replicate_id=replicate_id)
    return str(df["output"].iloc[0] or "")


def write_run_config(
    run_root: Path,
    task: TaskConfig,
    *,
    modes: list[str],
    worker_model: str | None,
    replicates: int | None,
    assistants_used: dict[str, str] | None,
    automation_used: dict[str, str] | None,
) -> None:
    from centaur_benchmark import __version__

    cfg = {
        "package_version": __version__,
        "task_slug": task.slug,
        "task_title": task.title,
        "modes": modes,
        "worker_model": worker_model,
        "replicates": replicates if replicates is not None else task.replicates,
        "assistants": assistants_used or task.assistants,
        "automation_models": automation_used or task.automation_models,
        "evaluator_models": task.evaluator_models,
        "default_evaluator": task.default_evaluator,
        "pairwise_n_evals": task.pairwise_n_evals,
        "pairwise_n_outer_runs": task.pairwise_n_outer_runs,
    }
    write_json(run_root / "config.json", cfg)

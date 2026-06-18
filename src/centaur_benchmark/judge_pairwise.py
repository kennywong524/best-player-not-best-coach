"""Pairwise LLM-as-judge over outputs (augmentation or automation)."""

from __future__ import annotations

import itertools
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from centaur_benchmark.config import TaskConfig
from centaur_benchmark.edsl_runtime import edsl_run_kwargs

EXPECTED_SCORE_DIMENSIONS: tuple[str, ...] = (
    "task_dimension_1",
    "task_dimension_2",
    "task_dimension_3",
    "task_dimension_4",
    "task_dimension_5",
    "task_dimension_6",
    "general_instruction_following",
    "general_accuracy_specificity",
    "general_practical_usefulness",
    "general_organization_readability",
    "general_tone_audience_fit",
)

_PAIRWISE_JSON_SCHEMA = """{
  "final_choice": "option_1",
  "short_rationale": "1-3 sentence explanation grounded in the rubric",
  "option_1_scores": {
    "task_dimension_1": 7,
    "task_dimension_2": 6,
    "task_dimension_3": 8,
    "task_dimension_4": 7,
    "task_dimension_5": 6,
    "task_dimension_6": 7,
    "general_instruction_following": 8,
    "general_accuracy_specificity": 7,
    "general_practical_usefulness": 7,
    "general_organization_readability": 6,
    "general_tone_audience_fit": 7
  },
  "option_2_scores": {
    "task_dimension_1": 5,
    "task_dimension_2": 4,
    "task_dimension_3": 6,
    "task_dimension_4": 5,
    "task_dimension_5": 4,
    "task_dimension_6": 5,
    "general_instruction_following": 6,
    "general_accuracy_specificity": 5,
    "general_practical_usefulness": 5,
    "general_organization_readability": 5,
    "general_tone_audience_fit": 5
  },
  "option_1_average": 6.8,
  "option_2_average": 5.0
}"""

_JUDGE_RULES_SUFFIX = """

Important scoring rules:
- Score every dimension from 1 to 10.
- 1 means the response essentially fails that dimension.
- 5 means partially adequate but with clear weaknesses.
- 10 means excellent.
- Use the full 1-10 scale when warranted.
- Do not assign all 1s unless the response truly fails every dimension.
- A useful, coherent, mostly correct response should not receive all 1s.
- Score each option independently before choosing.
- The final choice must be consistent with the average scores unless there is a serious safety, legal, medical, or dietary issue; if so, explain briefly in the rationale.
- Return JSON only. No Markdown. No prose outside JSON.
- Use exactly the dimension keys shown in the schema (task_dimension_1 through task_dimension_6 and the five general_* keys).
"""


def _build_judge_instruction(task: TaskConfig) -> str:
    return (task.pairwise_eval_prompt.strip() + _JUDGE_RULES_SUFFIX).strip()


def _pair_order(left_idx: int, right_idx: int, replicate_id: int) -> tuple[int, int]:
    """Randomize which item appears as option_1 vs option_2 (deterministic per pair)."""
    rng = random.Random(f"{left_idx}-{right_idx}-{replicate_id}")
    if rng.random() < 0.5:
        return right_idx, left_idx
    return left_idx, right_idx


def _ensure_augmentation_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model_id" not in df.columns:
        df["model_id"] = df.get("assistant_model", "unknown")
    if "model_label" not in df.columns:
        df["model_label"] = df["model_id"]
    return df


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    d = d[d["output"].notna()].reset_index(drop=True)
    return d


def _build_scenarios_augmentation(
    df: pd.DataFrame,
    task_text: str,
    n_evals: int,
    *,
    run_id: int | None = None,
) -> list:
    from edsl import Scenario

    scenarios = []
    item_pairs = list(itertools.combinations(df.index.tolist(), 2))
    for i, j in item_pairs:
        if pd.isna(df.loc[i].get("output")) or pd.isna(df.loc[j].get("output")):
            continue
        if str(df.loc[i].get("condition", "")) == str(df.loc[j].get("condition", "")):
            continue
        for replicate_id in range(n_evals):
            opt_left, opt_right = _pair_order(int(i), int(j), int(replicate_id))
            row_left = df.loc[opt_left]
            row_right = df.loc[opt_right]
            d = {
                "left_idx": int(opt_left),
                "right_idx": int(opt_right),
                "replicate_id": int(replicate_id),
                "task_text": str(task_text),
                "option_1": str(row_left["output"]),
                "option_2": str(row_right["output"]),
                "left_worker_model": str(row_left.get("worker_model", "unknown")),
                "right_worker_model": str(row_right.get("worker_model", "unknown")),
                "left_assistant_model": str(row_left.get("assistant_model", "none")),
                "right_assistant_model": str(row_right.get("assistant_model", "none")),
                "left_condition": str(row_left.get("condition", "unknown")),
                "right_condition": str(row_right.get("condition", "unknown")),
                "left_model_id": str(row_left.get("model_id", "unknown")),
                "right_model_id": str(row_right.get("model_id", "unknown")),
                "left_model_label": str(row_left.get("model_label", "unknown")),
                "right_model_label": str(row_right.get("model_label", "unknown")),
            }
            if run_id is not None:
                d["run_id"] = int(run_id)
            scenarios.append(Scenario(d))
    return scenarios


def _build_scenarios_automation(df: pd.DataFrame, task_text: str, n_evals: int) -> list:
    from edsl import Scenario

    scenarios = []
    item_pairs = list(itertools.combinations(df.index.tolist(), 2))
    for i, j in item_pairs:
        if pd.isna(df.loc[i].get("output")) or pd.isna(df.loc[j].get("output")):
            continue
        if str(df.loc[i].get("condition", "")) == str(df.loc[j].get("condition", "")):
            continue
        for replicate_id in range(n_evals):
            opt_left, opt_right = _pair_order(int(i), int(j), int(replicate_id))
            row_left = df.loc[opt_left]
            row_right = df.loc[opt_right]
            scenarios.append(
                Scenario(
                    {
                        "left_idx": int(opt_left),
                        "right_idx": int(opt_right),
                        "replicate_id": int(replicate_id),
                        "task_text": str(task_text),
                        "option_1": str(row_left["output"]),
                        "option_2": str(row_right["output"]),
                        "left_model_id": str(row_left.get("model_id", "unknown")),
                        "right_model_id": str(row_right.get("model_id", "unknown")),
                        "left_model_label": str(row_left.get("model_label", "unknown")),
                        "right_model_label": str(row_right.get("model_label", "unknown")),
                        "left_condition": str(row_left.get("condition", "unknown")),
                        "right_condition": str(row_right.get("condition", "unknown")),
                        "left_replicate_id": int(row_left["replicate_id"])
                        if pd.notna(row_left.get("replicate_id"))
                        else None,
                        "right_replicate_id": int(row_right["replicate_id"])
                        if pd.notna(row_right.get("replicate_id"))
                        else None,
                    }
                )
            )
    return scenarios


def _run_pairwise_survey(
    scenarios_list: list,
    eval_prompt: str,
    eval_model: str,
    *,
    include_run_id: bool,
    remote_description: str,
    remote_visibility: str = "private",
) -> pd.DataFrame:
    from edsl import Agent, Model, ScenarioList, Survey, QuestionFreeText

    if not scenarios_list:
        return pd.DataFrame()
    scenarios = ScenarioList(scenarios_list)
    q = QuestionFreeText(
        question_name="judgment",
        question_text="""
Original request:
{{ task_text }}

--------------------------------------------------
OPTION 1
{{ option_1 }}

--------------------------------------------------
OPTION 2
{{ option_2 }}

Compare OPTION 1 and OPTION 2 using the rubric in your evaluator instructions.
Score each option independently on every dimension before choosing.
Replace the placeholder integers below with your real 1-10 scores. Do not copy the example values.

Return JSON only. No Markdown. No prose outside JSON. Do not include hidden reasoning, chain-of-thought, analysis tags, or <think> blocks. Use exactly this schema:
""" + _PAIRWISE_JSON_SCHEMA + """
""",
    )
    survey = Survey([q])
    agent = Agent(instruction=eval_prompt)
    model_kwargs = {}
    if "gemini" in eval_model.lower() or "google" in eval_model.lower():
        model_kwargs["max_tokens"] = 4096
    evaluator = Model(eval_model, **model_kwargs)
    results = (
        survey.by(scenarios)
        .by(agent)
        .by(evaluator)
        .run(
            **edsl_run_kwargs(
                description=remote_description,
                visibility=remote_visibility,
                n=1,
                cache=True,
            ),
        )
    )
    if include_run_id:
        pairwise_df = results.select(
            "run_id",
            "left_idx",
            "right_idx",
            "replicate_id",
            "answer.judgment",
        ).to_pandas()
    else:
        pairwise_df = results.select(
            "left_idx",
            "right_idx",
            "replicate_id",
            "answer.judgment",
        ).to_pandas()
    pairwise_df.columns = [c.split(".")[-1] for c in pairwise_df.columns]
    pairwise_df = _parse_pairwise_judgments(pairwise_df)
    pairwise_df["judge_model"] = eval_model
    return pairwise_df


def _clean_json_text(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _iter_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def _looks_like_pairwise_judgment(obj: dict[str, object]) -> bool:
    return bool(
        obj.get("final_choice")
        or obj.get("option_1_scores")
        or obj.get("option1_scores")
        or obj.get("scores_option_1")
        or obj.get("option_2_scores")
        or obj.get("option2_scores")
        or obj.get("scores_option_2")
    )


def _extract_json_object(text: str) -> dict[str, object]:
    cleaned = _clean_json_text(text)
    if not cleaned:
        return {}
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass
    parsed_objects: list[dict[str, object]] = []
    for blob in _iter_balanced_json_objects(cleaned):
        for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    parsed_objects.append(loaded)
                    break
            except json.JSONDecodeError:
                continue
    for obj in parsed_objects:
        if _looks_like_pairwise_judgment(obj):
            return obj
    if parsed_objects:
        return parsed_objects[-1]
    return {}


def _normalize_dimension_key(key: object) -> str | None:
    text = re.sub(r"[^a-z0-9]+", "_", str(key or "").strip().lower()).strip("_")
    if not text:
        return None
    aliases = {
        "task_dimension1": "task_dimension_1",
        "task_dimension2": "task_dimension_2",
        "task_dimension3": "task_dimension_3",
        "task_dimension4": "task_dimension_4",
        "task_dimension5": "task_dimension_5",
        "task_dimension6": "task_dimension_6",
        "instruction_following": "general_instruction_following",
        "accuracy_specificity": "general_accuracy_specificity",
        "accuracy_and_specificity": "general_accuracy_specificity",
        "practical_usefulness": "general_practical_usefulness",
        "organization_readability": "general_organization_readability",
        "organization_and_readability": "general_organization_readability",
        "tone_audience_fit": "general_tone_audience_fit",
        "tone_and_audience_fit": "general_tone_audience_fit",
    }
    if text in aliases:
        return aliases[text]
    if re.fullmatch(r"task_dimension_\d", text):
        return text[:-1] + text[-1]
    if text in EXPECTED_SCORE_DIMENSIONS:
        return text
    for dim in EXPECTED_SCORE_DIMENSIONS:
        if dim in text or text in dim:
            return dim
    return None


def _normalize_score_dict(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        dim = _normalize_dimension_key(key)
        if dim is None:
            continue
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        out[dim] = max(1, min(10, score))
    return out


def _scores_average(scores: object) -> float | None:
    if not isinstance(scores, dict):
        return None
    vals = []
    for value in scores.values():
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(np.mean(vals)) if vals else None


def _parse_one_judgment(raw: object) -> dict[str, object]:
    text = str(raw or "")
    parsed = _extract_json_object(text)

    choice_raw = str(parsed.get("final_choice", "") if parsed else "")
    choice_match = re.search(r"option[_\s-]*([12])", choice_raw, flags=re.I)
    if not choice_match:
        choice_match = re.search(r"option[_\s-]*([12])", text, flags=re.I)
    winner = f"option_{choice_match.group(1)}" if choice_match else ""

    option_1_scores = _normalize_score_dict(
        parsed.get("option_1_scores")
        or parsed.get("option1_scores")
        or parsed.get("scores_option_1")
    )
    option_2_scores = _normalize_score_dict(
        parsed.get("option_2_scores")
        or parsed.get("option2_scores")
        or parsed.get("scores_option_2")
    )

    try:
        option_1_average = float(parsed.get("option_1_average")) if parsed else None
    except (TypeError, ValueError):
        option_1_average = None
    try:
        option_2_average = float(parsed.get("option_2_average")) if parsed else None
    except (TypeError, ValueError):
        option_2_average = None

    computed_1 = _scores_average(option_1_scores)
    computed_2 = _scores_average(option_2_scores)
    if option_1_average is None:
        option_1_average = computed_1
    if option_2_average is None:
        option_2_average = computed_2

    return {
        "winner": winner,
        "short_rationale": str(parsed.get("short_rationale", "") if parsed else "").strip(),
        "option_1_scores_json": json.dumps(option_1_scores, sort_keys=True),
        "option_2_scores_json": json.dumps(option_2_scores, sort_keys=True),
        "option_1_average": option_1_average,
        "option_2_average": option_2_average,
        "parse_ok": bool(parsed),
    }


def _all_scores_are_one(scores_json: str) -> bool:
    try:
        scores = json.loads(scores_json)
    except json.JSONDecodeError:
        return False
    if not scores:
        return False
    return all(int(v) == 1 for v in scores.values())


def _validate_judgment_row(row: pd.Series) -> list[str]:
    issues: list[str] = []
    if not str(row.get("winner", "")).strip():
        issues.append("missing_winner")
    if not str(row.get("short_rationale", "")).strip():
        issues.append("missing_rationale")

    for side in ("option_1", "option_2"):
        raw = row.get(f"{side}_scores_json", "{}")
        try:
            scores = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            scores = {}
            issues.append(f"{side}_scores_invalid_json")
        if not scores:
            issues.append(f"{side}_scores_empty")
            continue
        missing = [d for d in EXPECTED_SCORE_DIMENSIONS if d not in scores]
        if missing:
            issues.append(f"{side}_missing_dimensions:{len(missing)}")
        for dim, value in scores.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                issues.append(f"{side}_non_numeric:{dim}")
                continue
            if score < 1 or score > 10:
                issues.append(f"{side}_out_of_range:{dim}")

        avg_key = f"{side}_average"
        try:
            reported = float(row.get(avg_key))
            computed = _scores_average(scores)
            if computed is not None and abs(reported - computed) > 1.5:
                issues.append(f"{side}_average_mismatch")
        except (TypeError, ValueError):
            issues.append(f"{side}_average_missing")

        if _all_scores_are_one(str(raw)):
            issues.append(f"{side}_all_ones")

    winner = str(row.get("winner", "")).strip().lower()
    try:
        o1 = float(row.get("option_1_average"))
        o2 = float(row.get("option_2_average"))
        rationale = str(row.get("short_rationale", "")).lower()
        safety_terms = ("safety", "legal", "medical", "dietary", "allerg", "constraint", "violation")
        if winner == "option_1" and o1 + 2 < o2 and not any(t in rationale for t in safety_terms):
            issues.append("choice_contradicts_averages")
        if winner == "option_2" and o2 + 2 < o1 and not any(t in rationale for t in safety_terms):
            issues.append("choice_contradicts_averages")
    except (TypeError, ValueError):
        pass
    return issues


def validate_judge_batch(pairwise_df: pd.DataFrame) -> dict[str, object]:
    if pairwise_df.empty:
        return {
            "batch_ok": False,
            "row_pass_rate": 0.0,
            "n_rows": 0,
            "n_passing": 0,
            "issues_by_row": [],
            "suspicious_all_ones_rate": 0.0,
            "exclude_reason": "empty_batch",
        }

    row_reports: list[dict[str, object]] = []
    passing = 0
    all_ones_rows = 0
    for idx, row in pairwise_df.iterrows():
        issues = _validate_judgment_row(row)
        strict_issues = [
            i
            for i in issues
            if not i.endswith("_all_ones") and i != "choice_contradicts_averages"
        ]
        ok = not strict_issues
        if ok:
            passing += 1
        if any(i.endswith("_all_ones") for i in issues):
            all_ones_rows += 1
        row_reports.append({"row_index": int(idx), "ok": ok, "issues": issues})

    n_rows = len(pairwise_df)
    pass_rate = passing / n_rows if n_rows else 0.0
    all_ones_rate = all_ones_rows / n_rows if n_rows else 0.0
    exclude_reason = ""
    batch_ok = True
    if pass_rate < 0.9:
        batch_ok = False
        exclude_reason = f"row_pass_rate={pass_rate:.2f}<0.90"
    elif all_ones_rate > 0.4:
        batch_ok = False
        exclude_reason = f"all_ones_rate={all_ones_rate:.2f}>0.40"
    elif _scores_long(pairwise_df).empty:
        batch_ok = False
        exclude_reason = "no_rubric_scores_parsed"

    return {
        "batch_ok": batch_ok,
        "row_pass_rate": pass_rate,
        "n_rows": n_rows,
        "n_passing": passing,
        "issues_by_row": row_reports,
        "suspicious_all_ones_rate": all_ones_rate,
        "exclude_reason": exclude_reason,
    }


def _parse_pairwise_judgments(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if "judgment" not in pairwise_df.columns:
        return pairwise_df
    parsed = pd.DataFrame([_parse_one_judgment(x) for x in pairwise_df["judgment"]])
    return pd.concat([pairwise_df, parsed], axis=1)


def _aggregate_one_pass(
    df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    n_evals: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df must be reset-indexed contiguous 0..n-1 matching left_idx/right_idx."""
    records = []
    for replicate_id in range(n_evals):
        sub = pairwise_df[pairwise_df["replicate_id"] == replicate_id].copy()
        wins = {i: 0 for i in df.index}
        games = {i: 0 for i in df.index}
        for _, row in sub.iterrows():
            left = int(row["left_idx"])
            right = int(row["right_idx"])
            winner = str(row["winner"]).strip()
            games[left] += 1
            games[right] += 1
            if winner == "option_1":
                wins[left] += 1
            elif winner == "option_2":
                wins[right] += 1
        rep_df = pd.DataFrame(
            {
                "item_id": list(df.index),
                "replicate_id": replicate_id,
                "wins": [wins[i] for i in df.index],
                "games": [games[i] for i in df.index],
            }
        )
        rep_df["win_rate"] = rep_df["wins"] / rep_df["games"].replace(0, np.nan)
        rep_df["rank"] = rep_df["wins"].rank(ascending=False, method="average")
        records.append(rep_df)
    long_rank_df = pd.concat(records, ignore_index=True)
    summary = (
        long_rank_df.groupby("item_id")
        .agg(
            avg_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            avg_win_rate=("win_rate", "mean"),
            std_win_rate=("win_rate", "std"),
            total_wins=("wins", "sum"),
            total_games=("games", "sum"),
        )
        .reset_index()
    )
    scored = df.copy()
    scored["item_id"] = scored.index
    scored = scored.merge(summary, on="item_id", how="left")
    scored = scored.sort_values(["avg_rank", "avg_win_rate"], ascending=[True, False])
    return scored, long_rank_df


def _aggregate_multirun(df: pd.DataFrame, all_scored_runs: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["item_id"] = base.index
    cross = (
        all_scored_runs.groupby("item_id")
        .agg(
            avg_rank=("run_avg_rank", "mean"),
            sd_rank=("run_avg_rank", "std"),
            avg_win_rate=("run_avg_win_rate", "mean"),
            sd_win_rate=("run_avg_win_rate", "std"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
    )
    scored = base.merge(cross, on="item_id", how="left")
    return scored.sort_values(["avg_rank", "avg_win_rate"], ascending=[True, False])


def _leaderboard_from_scored(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["model_id", "model_label", "condition"])
        .agg(
            avg_rank=("avg_rank", "mean"),
            avg_win_rate=("avg_win_rate", "mean"),
            std_rank=("avg_rank", "std"),
            std_win_rate=("avg_win_rate", "std"),
            total_wins=("total_wins", "mean"),
            total_games=("total_games", "mean"),
        )
        .sort_values(by=["avg_rank", "avg_win_rate"], ascending=[True, False])
        .reset_index()
    )


def _model_family(model_id: object, model_label: object = "") -> str:
    text = f"{model_id} {model_label}".lower()
    if any(x in text for x in ("gpt", "openai", "o3", "o4")):
        return "openai"
    if any(x in text for x in ("claude", "anthropic")):
        return "anthropic"
    if "gemini" in text or "google" in text:
        return "google"
    if "deepseek" in text:
        return "deepseek"
    return text.split()[0] if text.split() else "unknown"


def _filter_scenarios_for_judge(scenarios: list, judge_model: str, *, exclude_self_family: bool) -> list:
    if not exclude_self_family:
        return scenarios
    judge_family = _model_family(judge_model)
    kept = []
    for scenario in scenarios:
        data = getattr(scenario, "data", None) or dict(scenario)
        left_family = _model_family(data.get("left_model_id"), data.get("left_model_label"))
        right_family = _model_family(data.get("right_model_id"), data.get("right_model_label"))
        if judge_family in {left_family, right_family}:
            continue
        kept.append(scenario)
    return kept


def _score_from_pairwise(df: pd.DataFrame, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    wins = {i: 0 for i in df.index}
    games = {i: 0 for i in df.index}
    for _, row in pairwise_df.iterrows():
        winner = str(row.get("winner", "")).strip().lower()
        if winner not in {"option_1", "option_2"}:
            continue
        left = int(row["left_idx"])
        right = int(row["right_idx"])
        games[left] += 1
        games[right] += 1
        if winner == "option_1":
            wins[left] += 1
        else:
            wins[right] += 1

    scored = df.copy()
    scored["item_id"] = scored.index
    scored["total_wins"] = scored["item_id"].map(wins)
    scored["total_games"] = scored["item_id"].map(games)
    scored["avg_win_rate"] = scored["total_wins"] / scored["total_games"].replace(0, np.nan)
    scored["avg_rank"] = scored["avg_win_rate"].rank(ascending=False, method="average")
    scored["std_rank"] = np.nan
    scored["std_win_rate"] = np.nan
    return scored.sort_values(["avg_rank", "avg_win_rate"], ascending=[True, False])


def _safe_file_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _leaderboard_from_panel_scored(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    return (
        scored.groupby(["judge_model", "model_id", "model_label", "condition"])
        .agg(
            avg_rank=("avg_rank", "mean"),
            avg_win_rate=("avg_win_rate", "mean"),
            total_wins=("total_wins", "mean"),
            total_games=("total_games", "mean"),
        )
        .reset_index()
        .sort_values(["judge_model", "avg_rank", "avg_win_rate"], ascending=[True, True, False])
    )


def _aggregate_panel_leaderboard(leaderboard_by_judge: pd.DataFrame) -> pd.DataFrame:
    if leaderboard_by_judge.empty:
        return leaderboard_by_judge
    out = (
        leaderboard_by_judge.groupby(["model_id", "model_label", "condition"])
        .agg(
            avg_rank_across_judges=("avg_rank", "mean"),
            sd_rank_across_judges=("avg_rank", "std"),
            avg_win_rate_across_judges=("avg_win_rate", "mean"),
            sd_win_rate_across_judges=("avg_win_rate", "std"),
            n_judges=("judge_model", "nunique"),
        )
        .reset_index()
    )
    out["aggregate_rank"] = out["avg_win_rate_across_judges"].rank(ascending=False, method="average")
    return out.sort_values(["aggregate_rank", "avg_win_rate_across_judges"], ascending=[True, False])


def _write_panel_matrices(out_dir: Path, leaderboard_by_judge: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    if leaderboard_by_judge.empty:
        return
    rank_matrix = leaderboard_by_judge.pivot_table(
        index=["model_label", "condition"],
        columns="judge_model",
        values="avg_rank",
        aggfunc="mean",
    )
    win_matrix = leaderboard_by_judge.pivot_table(
        index=["model_label", "condition"],
        columns="judge_model",
        values="avg_win_rate",
        aggfunc="mean",
    )
    rank_matrix.to_csv(out_dir / "leaderboard_matrix_rank_by_judge.csv")
    win_matrix.to_csv(out_dir / "leaderboard_matrix_win_rate_by_judge.csv")
    if not aggregate.empty:
        aggregate_matrix = aggregate.pivot_table(
            index=["model_label", "condition"],
            values=["aggregate_rank", "avg_win_rate_across_judges"],
            aggfunc="mean",
        )
        aggregate_matrix.to_csv(out_dir / "leaderboard_matrix_aggregate.csv")


def _scores_long(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in pairwise_df.iterrows():
        for side, item_col in (("option_1", "left_idx"), ("option_2", "right_idx")):
            raw = row.get(f"{side}_scores_json", "{}")
            try:
                scores = json.loads(raw) if isinstance(raw, str) else {}
            except json.JSONDecodeError:
                scores = {}
            for dimension, score in scores.items():
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    continue
                records.append(
                    {
                        "judge_model": row.get("judge_model"),
                        "item_id": int(row[item_col]),
                        "left_idx": row.get("left_idx"),
                        "right_idx": row.get("right_idx"),
                        "replicate_id": row.get("replicate_id"),
                        "scored_option": side,
                        "dimension": dimension,
                        "score": numeric_score,
                    }
                )
    return pd.DataFrame(records)


def _write_score_summaries(out_dir: Path, df: pd.DataFrame, pairwise_df: pd.DataFrame) -> None:
    long_scores = _scores_long(pairwise_df)
    if long_scores.empty:
        return
    meta_cols = ["item_id", "model_id", "model_label", "condition"]
    meta = df.copy()
    meta["item_id"] = meta.index
    long_scores = long_scores.merge(meta[meta_cols], on="item_id", how="left")
    long_scores.to_csv(out_dir / "rubric_scores_long.csv", index=False)
    summary = (
        long_scores.groupby(["judge_model", "model_id", "model_label", "condition", "dimension"])
        .agg(mean_score=("score", "mean"), sd_score=("score", "std"), n_scores=("score", "count"))
        .reset_index()
    )
    summary.to_csv(out_dir / "rubric_scores_summary.csv", index=False)


def _run_panel_judging(
    *,
    df: pd.DataFrame,
    scenarios: list,
    task: TaskConfig,
    out_dir: Path,
    eval_models: dict[str, str],
    exclude_self_family: bool,
    include_run_id: bool = False,
    run_id: str | None = None,
    subset_mode: str = "judging",
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_prompt = _build_judge_instruction(task)
    all_pairwise: list[pd.DataFrame] = []
    all_scored: list[pd.DataFrame] = []
    validation_report: dict[str, object] = {"judges": {}, "excluded_from_aggregate": []}

    for eval_model, eval_label in eval_models.items():
        judge_scenarios = _filter_scenarios_for_judge(
            scenarios,
            eval_model,
            exclude_self_family=exclude_self_family,
        )
        if not judge_scenarios:
            validation_report["judges"][eval_label] = {
                "batch_ok": False,
                "exclude_reason": "no_scenarios_after_family_filter",
            }
            continue
        remote_description = (
            f"centaur-judge-{run_id or 'run'}-{task.slug}-{subset_mode}-{_safe_file_slug(eval_label)}"
        )
        pairwise_df = _run_pairwise_survey(
            judge_scenarios,
            eval_prompt,
            eval_model,
            include_run_id=include_run_id,
            remote_description=remote_description,
            remote_visibility=task.remote_inference_visibility,
        )
        pairwise_df["judge_label"] = eval_label
        batch_validation = validate_judge_batch(pairwise_df)
        validation_report["judges"][eval_label] = {
            "judge_model": eval_model,
            **{k: v for k, v in batch_validation.items() if k != "issues_by_row"},
            "failed_rows": [
                r for r in batch_validation["issues_by_row"] if not r["ok"]
            ][:20],
        }

        scored = _score_from_pairwise(df, pairwise_df)
        scored["judge_model"] = eval_model
        scored["judge_label"] = eval_label

        judge_slug = _safe_file_slug(eval_label)
        pairwise_df.to_csv(out_dir / f"pairwise_judgments_{judge_slug}.csv", index=False)
        scored.to_csv(out_dir / f"pairwise_ranked_{judge_slug}.csv", index=False)

        if batch_validation["batch_ok"]:
            all_pairwise.append(pairwise_df)
            all_scored.append(scored)
        else:
            validation_report["excluded_from_aggregate"].append(
                {
                    "judge_label": eval_label,
                    "judge_model": eval_model,
                    "reason": batch_validation["exclude_reason"],
                }
            )

    if not all_pairwise:
        (out_dir / "judge_validation.json").write_text(
            json.dumps(validation_report, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            "No valid pairwise judgments for aggregate ranking. "
            f"See {out_dir / 'judge_validation.json'}"
        )

    pairwise_all = pd.concat(all_pairwise, ignore_index=True)
    scored_all = pd.concat(all_scored, ignore_index=True)
    leaderboard_by_judge = _leaderboard_from_panel_scored(scored_all)
    aggregate = _aggregate_panel_leaderboard(leaderboard_by_judge)

    judgments_path = out_dir / "pairwise_judgments_by_judge.csv"
    ranked_path = out_dir / "pairwise_ranked_by_judge.csv"
    board_path = out_dir / "leaderboard_aggregate.csv"

    pairwise_all.to_csv(judgments_path, index=False)
    scored_all.to_csv(ranked_path, index=False)
    leaderboard_by_judge.to_csv(out_dir / "leaderboard_by_judge.csv", index=False)
    aggregate.to_csv(board_path, index=False)
    _write_panel_matrices(out_dir, leaderboard_by_judge, aggregate)
    _write_score_summaries(out_dir, df, pairwise_all)
    validation_report["aggregate_judges"] = sorted(
        {str(frame["judge_label"].iloc[0]) for frame in all_pairwise}
    )
    (out_dir / "judge_validation.json").write_text(
        json.dumps(validation_report, indent=2, default=str),
        encoding="utf-8",
    )
    return judgments_path, ranked_path, board_path


def judge_augmentation(
    task: TaskConfig,
    run_root: Path,
    *,
    eval_model: str | None = None,
    n_evals: int | None = None,
    n_outer_runs: int | None = None,
) -> tuple[Path, Path, Path]:
    """Read augmentation/outputs.csv; write pairwise_judgments, pairwise_ranked, leaderboard."""
    eval_model = eval_model or task.default_evaluator
    n_evals = n_evals if n_evals is not None else task.pairwise_n_evals
    n_outer = n_outer_runs if n_outer_runs is not None else task.pairwise_n_outer_runs

    inp = run_root / "augmentation" / "outputs.csv"
    if not inp.exists():
        raise FileNotFoundError(f"Missing {inp}; run augmentation first.")
    raw = pd.read_csv(inp)
    raw = _ensure_augmentation_columns(raw)
    df = _prep_df(raw)

    out_dir = run_root / "augmentation"
    judgments_path = out_dir / "pairwise_judgments.csv"
    ranked_path = out_dir / "pairwise_ranked.csv"
    board_path = out_dir / "leaderboard.csv"
    task_text = task.pairwise_task_context

    if n_outer <= 1:
        scenarios = _build_scenarios_augmentation(df, task_text, n_evals, run_id=None)
        pairwise_df = _run_pairwise_survey(
            scenarios,
            _build_judge_instruction(task),
            eval_model,
            include_run_id=False,
            remote_description=f"centaur-judge-{run_root.name}-{task.slug}-augmentation-{eval_model}",
            remote_visibility=task.remote_inference_visibility,
        )
        scored, long_df = _aggregate_one_pass(df, pairwise_df, n_evals)
        pairwise_df.to_csv(judgments_path, index=False)
        long_df.to_csv(out_dir / "pairwise_long_rank.csv", index=False)
        scored.to_csv(ranked_path, index=False)
        _leaderboard_from_scored(scored).to_csv(board_path, index=False)
        return judgments_path, ranked_path, board_path

    all_pairwise: list[pd.DataFrame] = []
    all_scored_runs: list[pd.DataFrame] = []
    all_long: list[pd.DataFrame] = []

    for run_id in range(n_outer):
        scenarios = _build_scenarios_augmentation(df, task_text, n_evals, run_id=run_id)
        pw = _run_pairwise_survey(
            scenarios,
            _build_judge_instruction(task),
            eval_model,
            include_run_id=True,
            remote_description=f"centaur-judge-{run_root.name}-{task.slug}-augmentation-outer{run_id}-{eval_model}",
            remote_visibility=task.remote_inference_visibility,
        )
        scored_run, long_rank = _aggregate_one_pass(df, pw, n_evals)
        scored_run = scored_run.rename(
            columns={
                "avg_rank": "run_avg_rank",
                "std_rank": "run_std_rank",
                "avg_win_rate": "run_avg_win_rate",
                "std_win_rate": "run_std_win_rate",
                "total_wins": "run_total_wins",
                "total_games": "run_total_games",
            }
        )
        scored_run["run_id"] = run_id
        long_rank["run_id"] = run_id
        all_pairwise.append(pw)
        all_long.append(long_rank)
        all_scored_runs.append(scored_run)

    all_pairwise_df = pd.concat(all_pairwise, ignore_index=True)
    all_scored_runs_df = pd.concat(all_scored_runs, ignore_index=True)
    all_pairwise_df.to_csv(judgments_path, index=False)
    pd.concat(all_long, ignore_index=True).to_csv(out_dir / "pairwise_long_rank.csv", index=False)
    scored_final = _aggregate_multirun(df, all_scored_runs_df)
    scored_final.to_csv(ranked_path, index=False)
    all_scored_runs_df.to_csv(out_dir / "pairwise_run_level.csv", index=False)
    lb = (
        all_scored_runs_df.groupby(["model_id", "model_label", "condition"])
        .agg(
            avg_rank=("run_avg_rank", "mean"),
            sd_rank=("run_avg_rank", "std"),
            avg_win_rate=("run_avg_win_rate", "mean"),
            sd_win_rate=("run_avg_win_rate", "std"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
        .sort_values(by=["avg_rank", "avg_win_rate"], ascending=[True, False])
    )
    lb.to_csv(board_path, index=False)
    return judgments_path, ranked_path, board_path


def judge_automation(
    task: TaskConfig,
    run_root: Path,
    *,
    eval_model: str | None = None,
    n_evals: int | None = None,
) -> tuple[Path, Path, Path]:
    eval_model = eval_model or task.default_evaluator
    n_evals = n_evals if n_evals is not None else task.pairwise_n_evals
    inp = run_root / "automation" / "outputs.csv"
    if not inp.exists():
        raise FileNotFoundError(f"Missing {inp}; run automation first.")
    df = _prep_df(pd.read_csv(inp))
    out_dir = run_root / "automation"
    task_text = task.pairwise_task_context
    scenarios = _build_scenarios_automation(df, task_text, n_evals)
    pairwise_df = _run_pairwise_survey(
        scenarios,
        _build_judge_instruction(task),
        eval_model,
        include_run_id=False,
        remote_description=f"centaur-judge-{run_root.name}-{task.slug}-automation-{eval_model}",
        remote_visibility=task.remote_inference_visibility,
    )
    scored, long_df = _aggregate_one_pass(df, pairwise_df, n_evals)
    judgments_path = out_dir / "pairwise_judgments.csv"
    ranked_path = out_dir / "pairwise_ranked.csv"
    board_path = out_dir / "leaderboard.csv"
    pairwise_df.to_csv(judgments_path, index=False)
    long_df.to_csv(out_dir / "pairwise_long_rank.csv", index=False)
    scored.to_csv(ranked_path, index=False)
    _leaderboard_from_scored(scored).to_csv(board_path, index=False)
    return judgments_path, ranked_path, board_path


def judge_augmentation_panel(
    task: TaskConfig,
    run_root: Path,
    *,
    eval_models: dict[str, str] | None = None,
    n_evals: int | None = None,
    exclude_self_family: bool = True,
) -> tuple[Path, Path, Path]:
    """Run multiple LLM judges and write by-judge plus aggregate matrices."""
    eval_models = eval_models or task.evaluator_models or {task.default_evaluator: task.default_evaluator}
    n_evals = n_evals if n_evals is not None else task.pairwise_n_evals
    inp = run_root / "augmentation" / "outputs.csv"
    if not inp.exists():
        raise FileNotFoundError(f"Missing {inp}; run augmentation first.")
    raw = pd.read_csv(inp)
    raw = _ensure_augmentation_columns(raw)
    df = _prep_df(raw)
    scenarios = _build_scenarios_augmentation(df, task.pairwise_task_context, n_evals, run_id=None)
    return _run_panel_judging(
        df=df,
        scenarios=scenarios,
        task=task,
        out_dir=run_root / "augmentation",
        eval_models=eval_models,
        exclude_self_family=exclude_self_family,
        include_run_id=False,
        run_id=run_root.name,
        subset_mode="augmentation",
    )


def judge_automation_panel(
    task: TaskConfig,
    run_root: Path,
    *,
    eval_models: dict[str, str] | None = None,
    n_evals: int | None = None,
    exclude_self_family: bool = True,
) -> tuple[Path, Path, Path]:
    """Run multiple LLM judges and write by-judge plus aggregate matrices."""
    eval_models = eval_models or task.evaluator_models or {task.default_evaluator: task.default_evaluator}
    n_evals = n_evals if n_evals is not None else task.pairwise_n_evals
    inp = run_root / "automation" / "outputs.csv"
    if not inp.exists():
        raise FileNotFoundError(f"Missing {inp}; run automation first.")
    df = _prep_df(pd.read_csv(inp))
    scenarios = _build_scenarios_automation(df, task.pairwise_task_context, n_evals)
    return _run_panel_judging(
        df=df,
        scenarios=scenarios,
        task=task,
        out_dir=run_root / "automation",
        eval_models=eval_models,
        exclude_self_family=exclude_self_family,
        include_run_id=False,
        run_id=run_root.name,
        subset_mode="automation",
    )

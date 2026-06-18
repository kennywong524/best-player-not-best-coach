"""Build a compact JSON bundle for the static results dashboard."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "20260610_scaffold_strict_v4"
RUNS = [
    {
        "id": "20260610_scaffold_strict_v4",
        "label": "Run 1",
        "description": "Independent run of the full pipeline.",
    },
    {
        "id": "20260612_fresh_rep1",
        "label": "Run 2",
        "description": "Independent run of the full pipeline.",
    },
    {
        "id": "20260612_fresh_rep2",
        "label": "Run 3",
        "description": "Independent run of the full pipeline.",
    },
]
TASKS = [
    "counselling",
    "market_trends",
    "meal_plan",
    "operations_research",
    "tax_prep",
    "travel_planning",
    "tutoring",
]
TASK_LABELS = {
    "counselling": "Counseling",
    "market_trends": "Market Trends",
    "meal_plan": "Menu Planning",
    "operations_research": "Operations Research",
    "tax_prep": "Tax Prep",
    "travel_planning": "Travel Agent",
    "tutoring": "Tutoring",
}
TASK_TYPES = {
    "counselling": "Human-facing interactive",
    "market_trends": "Professional / analytical",
    "meal_plan": "Structured planning",
    "operations_research": "Professional / analytical",
    "tax_prep": "Professional / analytical",
    "travel_planning": "Structured planning",
    "tutoring": "Human-facing interactive",
}
MODES = ["augmentation", "automation"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numberish(value):
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value
    return int(x) if x.is_integer() else x


def convert_numbers(row: dict) -> dict:
    return {k: numberish(v) for k, v in row.items()}


def load_task_yaml(slug: str) -> dict:
    path = ROOT / "tasks" / f"{slug}.yaml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        return {
            "slug": slug,
            "label": TASK_LABELS[slug],
            "type": TASK_TYPES[slug],
            "title": data.get("title", TASK_LABELS[slug]),
            "task_prompt": data.get("task_prompt", ""),
            "scaffold_prompt": data.get("scaffold_prompt_template", ""),
            "worker_instruction": data.get("worker_instruction", ""),
            "rubric": data.get("pairwise_eval_prompt", ""),
        }
    except Exception:
        def block(key: str) -> str:
            m = re.search(rf"^{key}:\s*(.*?)(?=^[a-zA-Z_]+:|\Z)", text, re.M | re.S)
            return (m.group(1).strip() if m else "").strip("'\"")

        return {
            "slug": slug,
            "label": TASK_LABELS[slug],
            "type": TASK_TYPES[slug],
            "title": block("title") or TASK_LABELS[slug],
            "task_prompt": block("task_prompt"),
            "scaffold_prompt": block("scaffold_prompt_template"),
            "worker_instruction": block("worker_instruction"),
            "rubric": block("pairwise_eval_prompt"),
        }


def slim_output(row: dict, idx: int) -> dict:
    return {
        "idx": idx,
        "replicate_id": row.get("replicate_id"),
        "condition": row.get("condition"),
        "assistant_model": row.get("assistant_model"),
        "worker_model": row.get("worker_model"),
        "model_id": row.get("model_id"),
        "model_label": row.get("model_label"),
        "output": row.get("output", ""),
        "scaffold_text": row.get("scaffold_text", ""),
        "scaffold_path": row.get("scaffold_path", ""),
    }


def parse_json_field(value: str) -> dict:
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def slim_judgment(row: dict) -> dict:
    return {
        "left_idx": numberish(row.get("left_idx")),
        "right_idx": numberish(row.get("right_idx")),
        "replicate_id": row.get("replicate_id"),
        "winner": row.get("winner"),
        "short_rationale": row.get("short_rationale", ""),
        "option_1_average": numberish(row.get("option_1_average")),
        "option_2_average": numberish(row.get("option_2_average")),
        "option_1_scores": parse_json_field(row.get("option_1_scores_json", "")),
        "option_2_scores": parse_json_field(row.get("option_2_scores_json", "")),
        "parse_ok": row.get("parse_ok"),
        "judge_model": row.get("judge_model"),
        "judge_label": row.get("judge_label"),
    }


def load_run_bundle(run_id: str) -> dict:
    cross = ROOT / "artifacts" / "cross_task" / run_id
    aggregate = [convert_numbers(r) for r in read_csv(cross / "all_leaderboards_long.csv")]
    by_judge = [convert_numbers(r) for r in read_csv(cross / "all_leaderboards_by_judge_long.csv")]
    corr = [convert_numbers(r) for r in read_csv(cross / "judge_rank_correlation_summary.csv")]
    scatter = [convert_numbers(r) for r in read_csv(cross / "judge_rank_scatter_points.csv")]
    runs: dict[str, dict] = {}
    rubric_scores: list[dict] = []
    validations: list[dict] = []
    for task in TASKS:
        for mode in MODES:
            key = f"{task}/{mode}"
            mode_dir = ROOT / "results" / task / run_id / mode
            outputs = [slim_output(r, i) for i, r in enumerate(read_csv(mode_dir / "outputs.csv"))]
            judgments = [slim_judgment(r) for r in read_csv(mode_dir / "pairwise_judgments_by_judge.csv")]
            leaderboard = [convert_numbers(r) for r in read_csv(mode_dir / "leaderboard_aggregate.csv")]
            judge_lb = [convert_numbers(r) for r in read_csv(mode_dir / "leaderboard_by_judge.csv")]
            runs[key] = {
                "task": task,
                "task_label": TASK_LABELS[task],
                "mode": mode,
                "outputs": outputs,
                "judgments": judgments,
                "leaderboard": leaderboard,
                "leaderboard_by_judge": judge_lb,
            }
            for r in read_csv(mode_dir / "rubric_scores_summary.csv"):
                d = convert_numbers(r)
                d["task_slug"] = task
                d["task_label"] = TASK_LABELS[task]
                d["mode"] = mode
                rubric_scores.append(d)
            validation_path = mode_dir / "judge_validation.json"
            if validation_path.exists():
                validations.append(
                    {
                        "task_slug": task,
                        "task_label": TASK_LABELS[task],
                        "mode": mode,
                        "validation": json.loads(validation_path.read_text(encoding="utf-8")),
                    }
                )
    return {
        "aggregate": aggregate,
        "by_judge": by_judge,
        "rubric_scores": rubric_scores,
        "validations": validations,
        "correlations": corr,
        "scatter_points": scatter,
        "runs": runs,
    }


def main() -> None:
    dashboard = ROOT / "dashboard"
    dashboard.mkdir(exist_ok=True)
    runs_by_id = {run["id"]: load_run_bundle(run["id"]) for run in RUNS}
    default_bundle = runs_by_id[DEFAULT_RUN_ID]

    data = {
        "meta": {
            "run_id": DEFAULT_RUN_ID,
            "default_run_id": DEFAULT_RUN_ID,
            "replicate_runs": RUNS,
            "generated_from": str(ROOT),
            "notes": "Static dashboard bundle generated from current Centaur benchmark artifacts.",
        },
        "tasks": [load_task_yaml(t) for t in TASKS],
        "modes": MODES,
        "model_sets": {
            "all": {
                "label": "All candidates",
                "exclude": [],
            },
            "no_baseline": {
                "label": "Exclude baselines (paper 1-9 view)",
                "exclude": ["plain", "GPT-3.5-Turbo"],
            },
            "frontier_current": {
                "label": "Current non-legacy set",
                "include": [
                    "GPT-5-Mini",
                    "Claude-Sonnet-4.6",
                    "Claude-Opus-4.8",
                    "Gemini-3.1-Pro",
                    "DeepSeek-V3.1",
                    "GPT-OSS-120B",
                ],
            },
        },
        "runs_by_id": runs_by_id,
        "aggregate": default_bundle["aggregate"],
        "by_judge": default_bundle["by_judge"],
        "rubric_scores": default_bundle["rubric_scores"],
        "validations": default_bundle["validations"],
        "correlations": default_bundle["correlations"],
        "scatter_points": default_bundle["scatter_points"],
        "runs": default_bundle["runs"],
    }
    out = dashboard / "dashboard-data.json"
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()

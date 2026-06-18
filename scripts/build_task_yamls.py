#!/usr/bin/env python3
"""One-off: extract TASK_PROMPT / eval prompts from legacy notebooks into tasks/*.yaml."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def triple(src: str, name: str) -> str | None:
    m = re.search(rf"{re.escape(name)}\s*=\s*\"\"\"", src, re.DOTALL)
    if not m:
        return None
    s = m.end()
    e = src.find('"""', s)
    if e < 0:
        return None
    return src[s:e]


def first_cell_src(nb: dict, needle: str) -> str | None:
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if needle in src:
            return src
    return None


def extract_dict(src: str, var: str) -> dict[str, str]:
    m = re.search(rf"{re.escape(var)}\s*=\s*\{{", src)
    if not m:
        return {}
    start = m.end() - 1
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                blob = src[start : j + 1]
                break
    else:
        return {}
    out: dict[str, str] = {}
    for km in re.finditer(r'"([^"]+)"\s*:\s*"([^"]*)"', blob):
        out[km.group(1)] = km.group(2)
    return out


def extract_int(src: str, name: str, default: int) -> int:
    m = re.search(rf"{re.escape(name)}\s*=\s*(\d+)", src)
    return int(m.group(1)) if m else default


def pairwise_cell(nb: dict, augmentation_csv_substring: str) -> str | None:
    """Pick the code cell whose INPUT_FILE references augmentation outputs."""
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "EVAL_PROMPT" not in src:
            continue
        m = re.search(r'INPUT_FILE\s*=\s*"([^"]+)"', src)
        if not m:
            continue
        if augmentation_csv_substring in m.group(1):
            return src
    # fallback: first cell with EVAL_PROMPT
    return first_cell_src(nb, "EVAL_PROMPT")


def worker_instruction(aug_src: str) -> str:
    m = re.search(
        r"Agent\(\s*instruction=\(\s*\"\"\"(.*?)\"\"\"\s*\)\s*\)",
        aug_src,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return (
        "If the task prompt includes an 'Assistant Guidance: Three-Phase Workflow' section at the top, "
        "read it carefully first and follow its planning and self-review steps before producing your response. "
        "Then complete the task itself clearly and professionally. "
        "Return only the final deliverable — do not restate the guidance."
    )


NOTEBOOKS: list[tuple[str, str, str, str]] = [
    # slug, title, notebook path, substring identifying augmentation results CSV in pairwise cell
    ("meal_plan", "Menu planning (GDPval-style)", "menu planning/menu-planning.ipynb", "mealplan_vary_scaffolds"),
    ("counselling", "Counseling session (Anthropic Econ Index)", "counselling/counselling", "counseling_augmentation"),
    ("market_trends", "Market trends analysis", "market trend analysis/market-trend-analysis", "market_trend_augmentation"),
    (
        "operations_research",
        "Operations research analyst memo",
        "operations research/operations-research",
        "operations-research_augmentation",
    ),
    ("tutoring", "Tutoring session planning", "tutoring/tutoring", "tutoring_vary_scaffolds"),
    ("travel_planning", "Travel agent itinerary", "travel planning/travel-planning", "travel_agent_augmentation"),
    ("tax_prep", "Tax preparation discrepancy spotting", "tax preparation/tax-prep", "tax_preparer_augmentation"),
]


def main() -> None:
    tasks_dir = ROOT / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    for slug, title, rel, csv_hint in NOTEBOOKS:
        path = ROOT / rel
        nb = json.loads(path.read_text(encoding="utf-8"))
        aug = first_cell_src(nb, "Varying Assistant Scaffolds") or first_cell_src(nb, "ASSISTANT_MODELS")
        if not aug:
            print("skip", slug, "no aug cell")
            continue
        pw = pairwise_cell(nb, csv_hint) or first_cell_src(nb, "EVAL_PROMPT")
        if not pw:
            print("skip", slug, "no pairwise")
            continue

        task_prompt = triple(aug, "TASK_PROMPT") or ""
        scaffold = triple(aug, "ASSISTANT_PROMPT_TEMPLATE") or ""
        task_text = triple(pw, "TASK_TEXT") or task_prompt
        eval_prompt = triple(pw, "EVAL_PROMPT") or ""
        assistants = extract_dict(aug, "ASSISTANT_MODELS")
        auto = extract_dict(aug, "TASK_MODELS")
        if not auto:
            auto_cell = first_cell_src(nb, "Direct Model Comparison") or first_cell_src(nb, "TASK_MODELS")
            if auto_cell:
                auto = extract_dict(auto_cell, "TASK_MODELS")

        n_rep = extract_int(aug, "N_REPLICATES", 3)
        n_evals = extract_int(pw, "N_EVALS", 3)
        n_outer = extract_int(pw, "N_RUNS", 1)

        cfg: dict = {
            "slug": slug,
            "title": title,
            "default_worker": "gpt-3.5-turbo",
            "default_evaluator": "gpt-5",
            "replicates": n_rep,
            "scaffold_model_max_tokens": None,
            "automation_model_max_tokens": None,
            "automation_worker_instruction": (
                "Complete the task clearly and professionally. Return only the final deliverable."
            ),
            "remote_inference_visibility": "private",
            "task_prompt": task_prompt,
            "scaffold_prompt_template": scaffold,
            "worker_instruction": worker_instruction(aug),
            "pairwise_task_context": task_text,
            "pairwise_eval_prompt": eval_prompt,
            "pairwise": {"n_evals_per_pair": n_evals, "n_outer_runs": n_outer},
            "models": {"assistants": assistants, "automation": auto or None},
            "rubric_prompt": None,
        }

        out = tasks_dir / f"{slug}.yaml"
        out.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=1000),
            encoding="utf-8",
        )
        print("wrote", out, "assistants", len(assistants), "auto", len(auto or {}))


if __name__ == "__main__":
    main()
    sys.exit(0)

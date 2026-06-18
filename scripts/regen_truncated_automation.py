#!/usr/bin/env python3
"""Regenerate truncated automation outputs for 20260610_scaffold_strict_v4."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import ensure_run_dir
from centaur_benchmark.runner import patch_automation_models

RUN_ID = "20260610_scaffold_strict_v4"

RETRY_BATCHES: dict[str, list[str]] = {
    "counselling": [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-oss-120b",
    ],
    "market_trends": [
        "anthropic/claude-opus-4-8",
    ],
    "meal_plan": [
        "anthropic/claude-sonnet-4-6",
        "deepseek-ai/DeepSeek-V3.1",
        "google/gemini-3.1-pro",
        "openai/gpt-oss-120b",
    ],
    "tax_prep": [
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
        "gpt-4.1",
        "openai/gpt-oss-120b",
    ],
    "travel_planning": [
        "anthropic/claude-sonnet-4-6",
        "google/gemini-3.1-pro",
        "gpt-4.1",
        "openai/gpt-oss-120b",
    ],
    "tutoring": [
        "anthropic/claude-sonnet-4-6",
        "gpt-4.1",
        "openai/gpt-oss-120b",
    ],
}

BATCHES: dict[str, list[str]] = {
    "counselling": [
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-oss-120b",
    ],
    "market_trends": [
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
    ],
    "meal_plan": [
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
        "deepseek-ai/DeepSeek-V3.1",
        "google/gemini-3.1-pro",
        "gpt-4.1",
        "o3-mini-2025-01-31",
        "openai/gpt-oss-120b",
        "gpt-3.5-turbo",
    ],
    "tax_prep": [
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
        "deepseek-ai/DeepSeek-V3.1",
        "gpt-4.1",
        "o3-mini-2025-01-31",
        "o4-mini-2025-04-16",
        "openai/gpt-oss-120b",
    ],
    "travel_planning": [
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
        "deepseek-ai/DeepSeek-V3.1",
        "google/gemini-3.1-pro",
        "gpt-4.1",
        "openai/gpt-oss-120b",
    ],
    "tutoring": [
        "anthropic/claude-sonnet-4-6",
        "deepseek-ai/DeepSeek-V3.1",
        "gpt-4.1",
        "openai/gpt-oss-120b",
    ],
    "operations_research": [
        "anthropic/claude-opus-4-8",
        "openai/gpt-oss-120b",
    ],
}

_THINKING_RE = re.compile(
    r"^<think>.*?</think>\s*",
    re.DOTALL | re.IGNORECASE,
)


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def strip_thinking_tags(run_id: str) -> int:
    """Remove Gemini thinking prefixes from all automation outputs."""
    n = 0
    for path in sorted(Path("results").glob(f"*/{run_id}/automation/outputs.csv")):
        df = pd.read_csv(path)
        changed = False
        for i, row in df.iterrows():
            out = str(row["output"])
            cleaned = _THINKING_RE.sub("", out)
            if cleaned != out:
                df.at[i, "output"] = cleaned
                changed = True
                n += 1
        if changed:
            df.to_csv(path, index=False)
            print(f"Stripped thinking tags in {path}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-only", action="store_true", help="Regen only still-bad models")
    parser.add_argument("--pause", type=int, default=10, help="Seconds between model calls")
    args = parser.parse_args()

    _load_dotenv()
    os.environ["EDSL_API_TIMEOUT"] = "1800"
    os.environ["REMOTE_PROXY_TIMEOUT"] = "1800"
    os.environ["EDSL_MAX_ATTEMPTS"] = "8"

    batches = RETRY_BATCHES if args.retry_only else BATCHES
    total = sum(len(v) for v in batches.values())
    done = 0
    for slug, model_ids in batches.items():
        task = load_task(default_tasks_dir() / f"{slug}.yaml")
        root = ensure_run_dir(slug, RUN_ID)
        print(f"\n=== BATCH {slug} ({len(model_ids)} models) ===", flush=True)
        for mid in model_ids:
            done += 1
            print(f"[{done}/{total}] {slug} :: {mid}", flush=True)
            patch_automation_models(task, root, [mid], replicates=1)
            if args.pause > 0:
                time.sleep(args.pause)

    stripped = strip_thinking_tags(RUN_ID)
    print(f"\nDone. Stripped thinking tags from {stripped} rows.", flush=True)


if __name__ == "__main__":
    main()

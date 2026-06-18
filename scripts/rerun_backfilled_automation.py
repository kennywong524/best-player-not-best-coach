#!/usr/bin/env python3
"""Re-run automation rows that were previously backfilled from an older run."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import ensure_run_dir, results_base
from centaur_benchmark.runner import patch_automation_models

SKIP_MODEL_IDS = {"gpt-5-mini-2025-08-07"}  # tax_prep row already native v4


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


def main() -> None:
    _load_dotenv()
    os.environ["EDSL_API_TIMEOUT"] = os.environ.get("EDSL_API_TIMEOUT", "1800")
    os.environ["REMOTE_PROXY_TIMEOUT"] = os.environ.get("REMOTE_PROXY_TIMEOUT", "1800")
    os.environ.setdefault("EDSL_MAX_ATTEMPTS", "8")

    parser = argparse.ArgumentParser(description="Re-run backfilled automation model groups.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", required=True, help="Backfill manifest JSON")
    parser.add_argument(
        "--skip-model-ids",
        default=",".join(sorted(SKIP_MODEL_IDS)),
        help="Comma-separated model_ids to skip",
    )
    args = parser.parse_args()
    skip = {x.strip() for x in args.skip_model_ids.split(",") if x.strip()}

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_task: dict[str, list[str]] = defaultdict(list)
    for row in manifest.get("backfilled", []):
        task = str(row["task"])
        mid = str(row["model_id"])
        if mid in skip and task == "tax_prep":
            print(f"SKIP {task} {mid} (already native)")
            continue
        if mid not in by_task[task]:
            by_task[task].append(mid)

    for task_slug in sorted(by_task):
        ids = sorted(by_task[task_slug])
        task = load_task(default_tasks_dir() / f"{task_slug}.yaml")
        root = ensure_run_dir(task_slug, args.run_id)
        print(f"=== RERUN {task_slug} automation models={ids} (no max_tokens except gpt-5) ===")
        patch_automation_models(task, root, ids)

    print(f"Done. Manifest source: {args.manifest}")


if __name__ == "__main__":
    main()

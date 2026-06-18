#!/usr/bin/env python3
"""Run automation generation only for a shared run id."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import ensure_run_dir
from centaur_benchmark.runner import run_automation, write_run_config


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
    os.environ.setdefault("EDSL_API_TIMEOUT", "1800")
    os.environ.setdefault("REMOTE_PROXY_TIMEOUT", "1800")
    os.environ.setdefault("EDSL_MAX_ATTEMPTS", "8")

    parser = argparse.ArgumentParser(description="Run automation for all or selected tasks.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tasks", default=None, help="Comma-separated task slugs (default: all)")
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args()
    slugs = {x.strip() for x in args.tasks.split(",") if x.strip()} if args.tasks else None

    for path in sorted(default_tasks_dir().glob("*.yaml")):
        task = load_task(path)
        if slugs and task.slug not in slugs:
            continue
        root = ensure_run_dir(task.slug, args.run_id)
        print(f"=== AUTOMATION {task.slug} -> {root / 'automation'} ===")
        p = run_automation(task, root, replicates=args.replicates)
        if p is None:
            continue
        write_run_config(
            root,
            task,
            modes=["automation"],
            worker_model=task.default_worker,
            replicates=args.replicates,
            assistants_used=None,
            automation_used=task.automation_models,
        )
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()

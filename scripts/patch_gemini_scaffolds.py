#!/usr/bin/env python3
"""Regenerate Gemini scaffold + worker outputs for a run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import ensure_run_dir
from centaur_benchmark.runner import patch_augmentation_models

GEMINI_ID = "google/gemini-3.1-pro"


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
    os.environ.setdefault("EDSL_API_TIMEOUT", "600")
    os.environ.setdefault("REMOTE_PROXY_TIMEOUT", "600")
    os.environ.setdefault("EDSL_MAX_ATTEMPTS", "8")

    parser = argparse.ArgumentParser(description="Patch Gemini augmentation for all tasks.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tasks", default=None, help="Comma-separated task slugs (default: all)")
    args = parser.parse_args()
    slugs = {x.strip() for x in args.tasks.split(",") if x.strip()} if args.tasks else None

    for path in sorted(default_tasks_dir().glob("*.yaml")):
        task = load_task(path)
        if slugs and task.slug not in slugs:
            continue
        if GEMINI_ID not in task.assistants:
            print(f"SKIP {task.slug}: no Gemini assistant")
            continue
        root = ensure_run_dir(task.slug, args.run_id)
        print(f"=== PATCH {task.slug} Gemini ===")
        patch_augmentation_models(task, root, [GEMINI_ID])


if __name__ == "__main__":
    main()

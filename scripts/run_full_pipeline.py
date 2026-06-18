#!/usr/bin/env python3
"""Run the full Centaur benchmark pipeline across tasks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.edsl_runtime import use_remote_inference
from centaur_benchmark.io import ensure_run_dir, results_base
from centaur_benchmark.judge_pairwise import judge_augmentation_panel, judge_automation_panel
from centaur_benchmark.runner import run_augmentation, run_automation, write_run_config
from centaur_benchmark.summarize import export_cross_task_matrices

from audit_run import audit_run
from validate_judging import validate_run, write_notes


def _task_paths(task_slugs: list[str] | None) -> list[Path]:
    tasks_dir = default_tasks_dir()
    if not task_slugs:
        return sorted(tasks_dir.glob("*.yaml"))
    return [tasks_dir / f"{slug}.yaml" for slug in task_slugs]


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
    os.environ.setdefault("EXPECTED_PARROT_URL", "https://www.expectedparrot.com")
    os.environ.setdefault("EDSL_API_TIMEOUT", "300")
    os.environ.setdefault("EDSL_MAX_ATTEMPTS", "5")
    os.environ.setdefault("REMOTE_PROXY_TIMEOUT", "300")

    parser = argparse.ArgumentParser(description="Run generation, panel judging, and cross-task matrices.")
    parser.add_argument("--run-id", required=True, help="Shared run id for every task.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task slugs; default all tasks.")
    parser.add_argument("--mode", choices=["all", "generation", "judge", "summarize"], default="all")
    parser.add_argument("--replicates", type=int, default=None, help="Override task replicates.")
    parser.add_argument("--n-evals", type=int, default=None, help="Override pairwise repeats per pair.")
    parser.add_argument(
        "--include-self-family",
        action="store_true",
        help="Allow same-family evaluator/output comparisons.",
    )
    parser.add_argument(
        "--only-mode",
        choices=["augmentation", "automation"],
        default=None,
        help="Judge only augmentation or automation (judge mode only).",
    )
    args = parser.parse_args()

    task_slugs = [x.strip() for x in args.tasks.split(",") if x.strip()] if args.tasks else None
    tasks = [load_task(path) for path in _task_paths(task_slugs)]

    if args.mode in {"all", "generation"}:
        for task in tasks:
            root = ensure_run_dir(task.slug, args.run_id)
            print(f"=== GENERATION {task.slug} -> {root} ===")
            run_augmentation(task, root, replicates=args.replicates)
            run_automation(task, root, replicates=args.replicates)
            write_run_config(
                root,
                task,
                modes=["augmentation", "automation"],
                worker_model=task.default_worker,
                replicates=args.replicates,
                assistants_used=task.assistants,
                automation_used=task.automation_models,
            )

    if args.mode in {"all", "judge"}:
        mode_label = "remote Expected Parrot Jobs" if use_remote_inference() else "local API proxy"
        print(f"EDSL execution mode: {mode_label} (CENTAUR_EDSL_REMOTE={os.environ.get('CENTAUR_EDSL_REMOTE', '0')})")
        audit = audit_run(args.run_id, task_slugs=task_slugs)
        if not audit["ready_for_judging"]:
            raise SystemExit(
                f"Run {args.run_id} failed audit ({len(audit['failures'])} failures). "
                "Patch/retry generations before judging."
            )
        for task in tasks:
            root = ensure_run_dir(task.slug, args.run_id)
            print(f"=== PANEL JUDGE {task.slug} -> {root} ===")
            if args.only_mode in {None, "augmentation"}:
                judge_augmentation_panel(
                    task,
                    root,
                    n_evals=args.n_evals,
                    exclude_self_family=not args.include_self_family,
                )
            if args.only_mode in {None, "automation"}:
                judge_automation_panel(
                    task,
                    root,
                    n_evals=args.n_evals,
                    exclude_self_family=not args.include_self_family,
                )
        validation = validate_run(args.run_id, task_slugs=task_slugs)
        notes_path = results_base() / f"judging_notes_{args.run_id}.md"
        write_notes(args.run_id, validation, notes_path)
        print(f"Judging notes written to {notes_path}")

    if args.mode in {"all", "summarize"}:
        validation = validate_run(args.run_id, task_slugs=task_slugs)
        if not validation["ready_for_summarize"]:
            raise SystemExit(
                f"Run {args.run_id} is not ready to summarize. "
                f"Incomplete: {validation['incomplete_task_modes']}"
            )
        export_cross_task_matrices(args.run_id, task_slugs=[t.slug for t in tasks])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Retry failed model×task combinations from an audit report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import ensure_run_dir, results_base
from centaur_benchmark.runner import (
    patch_augmentation_models,
    patch_augmentation_plain_outputs,
    patch_augmentation_worker_outputs,
    patch_automation_models,
)

PLAIN_RETRY_TASKS = {"meal_plan", "tax_prep"}

RETRYABLE_KINDS = {
    "empty_output",
    "format_stub",
    "empty_scaffold",
    "missing_model",
    "meta_scaffold_echo",
    "likely_hard_truncation",
    "incomplete_meal_plan_days<7(max=1)",
    "incomplete_meal_plan_days<7(max=2)",
    "incomplete_meal_plan_days<7(max=3)",
    "incomplete_meal_plan_days<7(max=4)",
    "incomplete_meal_plan_days<7(max=5)",
    "incomplete_meal_plan_days<7(max=6)",
}

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_run import audit_run  # noqa: E402


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


def retry_from_audit(
    run_id: str,
    audit: dict,
    *,
    modes: set[str] | None = None,
) -> list[dict]:
    """Retry each failure; returns list of actions taken."""
    actions: list[dict] = []
    by_task_mode: dict[tuple[str, str], dict[str, set[str]]] = {}
    for f in audit.get("failures", []):
        if modes and str(f.get("mode", "")) not in modes:
            continue
        kind = str(f.get("kind", ""))
        if (
            kind not in RETRYABLE_KINDS
            and not kind.startswith("short<")
            and not kind.startswith("short_scaffold_output<")
            and not kind.startswith("missing_terms:")
            and not kind.startswith("incomplete_meal_plan_days")
        ):
            continue
        key = (f["task"], f["mode"])
        bucket = by_task_mode.setdefault(key, {"scaffold": set(), "worker": set()})
        mid = str(f.get("model_id", ""))
        if mid == "plain":
            continue
        if kind in {"empty_scaffold", "missing_model"}:
            bucket["scaffold"].add(mid)
        else:
            bucket["worker"].add(mid)

    for (task_slug, mode), buckets in sorted(by_task_mode.items()):
        task_path = default_tasks_dir() / f"{task_slug}.yaml"
        task = load_task(task_path)
        root = ensure_run_dir(task.slug, run_id)
        if mode == "augmentation":
            scaffold_ids = sorted(buckets["scaffold"])
            worker_ids = sorted(buckets["worker"] - buckets["scaffold"])
            if scaffold_ids:
                print(f"=== RETRY {task_slug} augmentation scaffold models={scaffold_ids} ===")
                patch_augmentation_models(task, root, scaffold_ids)
                actions.append({"task": task_slug, "mode": mode, "model_ids": scaffold_ids, "patch": "scaffold"})
            if worker_ids:
                print(f"=== RETRY {task_slug} augmentation worker models={worker_ids} ===")
                patch_augmentation_worker_outputs(task, root, worker_ids)
                actions.append({"task": task_slug, "mode": mode, "model_ids": worker_ids, "patch": "worker"})
            if task_slug in PLAIN_RETRY_TASKS:
                print(f"=== RETRY {task_slug} augmentation plain baseline ===")
                patch_augmentation_plain_outputs(task, root)
                actions.append({"task": task_slug, "mode": mode, "model_ids": ["plain"], "patch": "plain"})
        else:
            ids = sorted(buckets["scaffold"] | buckets["worker"])
            if not ids:
                continue
            print(f"=== RETRY {task_slug} {mode} models={ids} ===")
            patch_automation_models(task, root, ids)
            actions.append({"task": task_slug, "mode": mode, "model_ids": ids})
    return actions


def main() -> None:
    _load_dotenv()
    os.environ.setdefault("EDSL_API_TIMEOUT", "600")
    os.environ.setdefault("REMOTE_PROXY_TIMEOUT", "600")
    os.environ.setdefault("EDSL_MAX_ATTEMPTS", "8")

    parser = argparse.ArgumentParser(description="Retry failed outputs from audit JSON or fresh audit.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audit-json", default=None, help="Use existing audit file")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated modes to retry (augmentation,automation). Default: both.",
    )
    args = parser.parse_args()
    mode_filter = {m.strip() for m in args.modes.split(",") if m.strip()} if args.modes else None

    audit_path = Path(args.audit_json) if args.audit_json else results_base() / f"audit_{args.run_id}.json"
    for round_i in range(1, args.max_rounds + 1):
        if round_i == 1 and args.audit_json and audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            audit = audit_run(args.run_id)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

        if audit.get("ready_for_judging"):
            print(f"Run {args.run_id} is ready for judging (round {round_i}).")
            return

        failures = audit.get("failures", [])
        if not failures:
            print("No failures but not ready — check missing_tasks.")
            return

        print(f"=== RETRY ROUND {round_i}/{args.max_rounds} ({len(failures)} failures) ===")
        retry_from_audit(args.run_id, audit, modes=mode_filter)
        audit = audit_run(args.run_id)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        if audit.get("ready_for_judging"):
            print(f"Ready for judging after round {round_i}.")
            return

    print(f"Still {len(audit.get('failures', []))} failures after {args.max_rounds} rounds. See {audit_path}")


if __name__ == "__main__":
    main()

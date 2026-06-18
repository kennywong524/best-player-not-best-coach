#!/usr/bin/env python3
"""Resumable batch runner with per-replicate quality gates.

Each replicate is one checkpointed API call + automated audit (truncation, empty,
task deliverable rules). Failed replicates retry in-place before moving on.

Checkpoint: results/logs/batch_{run_id}.json
Log:        results/logs/batch_{run_id}.log
Audit:      results/logs/audit_{run_id}.json (written after generation + repair)

Re-run the same command after disconnect; completed steps are skipped.
Use --verify on resume to re-audit completed rows and redo any that regressed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from centaur_benchmark.config import default_tasks_dir, load_task  # noqa: E402
from centaur_benchmark.edsl_runtime import use_remote_inference  # noqa: E402
from centaur_benchmark.io import ensure_run_dir, results_base, write_json  # noqa: E402
from centaur_benchmark.judge_pairwise import (  # noqa: E402
    judge_augmentation_panel,
    judge_automation_panel,
)
from centaur_benchmark.runner import (  # noqa: E402
    _safe_slug,
    generate_augmentation_plain_replicate,
    generate_augmentation_worker_replicate,
    generate_automation_replicate,
    write_run_config,
)
from centaur_benchmark.summarize import export_cross_task_matrices  # noqa: E402

from audit_all_outputs import audit_all_outputs  # noqa: E402
from audit_run import audit_run  # noqa: E402
from check_replicate_variability import check_variability  # noqa: E402
from output_quality import audit_csv_row, audit_output_row  # noqa: E402
from validate_judging import validate_run, write_notes  # noqa: E402

DEFAULT_RUN_ID = "20260611_replicates3_v2"

AUG_COLUMNS = [
    "replicate_id",
    "output",
    "condition",
    "assistant_model",
    "worker_model",
    "model_id",
    "model_label",
    "scaffold_path",
    "scaffold_sha256",
    "scaffold_text",
]
AUTO_COLUMNS = ["replicate_id", "output", "condition", "model_id", "model_label"]


class QualityGateError(RuntimeError):
    """Raised when a replicate fails audit after all retries."""


class FailureExplosionError(RuntimeError):
    """Raised when the same steps keep failing across resume cycles."""


def _load_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    # Local proxy: low timeouts + high max_tokens requests → LanguageModelNoResponseError.
    os.environ["EDSL_API_TIMEOUT"] = "1800"
    os.environ["REMOTE_PROXY_TIMEOUT"] = "1800"
    os.environ.setdefault("EDSL_MAX_ATTEMPTS", "8")
    os.environ.setdefault("CENTAUR_EDSL_REMOTE", "0")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_path(run_id: str) -> Path:
    return results_base() / "logs" / f"batch_{run_id}.json"


def _log_path(run_id: str) -> Path:
    return results_base() / "logs" / f"batch_{run_id}.log"


def _audit_report_path(run_id: str) -> Path:
    return results_base() / "logs" / f"audit_{run_id}.json"


def _load_checkpoint(run_id: str) -> dict[str, Any]:
    path = _checkpoint_path(run_id)
    if not path.is_file():
        return {
            "run_id": run_id,
            "started_at": _now(),
            "updated_at": _now(),
            "completed": [],
            "failed": {},
            "failure_counts": {},
            "current": None,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    counts = state.setdefault("failure_counts", {})
    for step_id in state.get("failed", {}):
        counts.setdefault(step_id, 1)
    return state


def _record_step_failure(state: dict[str, Any], step_id: str) -> int:
    counts = state.setdefault("failure_counts", {})
    counts[step_id] = int(counts.get(step_id, 0)) + 1
    return counts[step_id]


def _check_failure_guardrails(
    *,
    step_id: str,
    step_failures: int,
    consecutive_failures: int,
    max_step_failures: int,
    max_consecutive_failures: int,
) -> None:
    if step_failures >= max_step_failures:
        raise FailureExplosionError(
            f"Step {step_id} failed {step_failures} times (limit {max_step_failures}). "
            "Patch this row manually, then resume with --from-step or repair."
        )
    if consecutive_failures >= max_consecutive_failures:
        raise FailureExplosionError(
            f"{consecutive_failures} consecutive step failures this session "
            f"(limit {max_consecutive_failures}) with no successes. "
            "Check the log, patch hot rows, then resume."
        )


def _save_checkpoint(state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    write_json(_checkpoint_path(state["run_id"]), state)


def _log(run_id: str, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    log_file = _log_path(run_id)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _task_yaml_paths(task_slugs: list[str] | None) -> list[Path]:
    tasks_dir = default_tasks_dir()
    if not task_slugs:
        return sorted(tasks_dir.glob("*.yaml"))
    return [tasks_dir / f"{slug}.yaml" for slug in task_slugs]


def _ensure_outputs_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def _parse_gen_step(step_id: str) -> dict[str, Any] | None:
    """Parse gen/{task}/... step ids into components."""
    if not step_id.startswith("gen/"):
        return None
    parts = step_id.split("/")
    if len(parts) < 3:
        return None
    info: dict[str, Any] = {"task": parts[1], "kind": parts[2]}
    if info["kind"] == "init" or info["kind"] == "config":
        return info
    if info["kind"] == "repair" or info["kind"] == "final_audit":
        return info
    if info["kind"] == "aug" and len(parts) == 5 and parts[4].startswith("rep"):
        info["subkind"] = "plain" if parts[3] == "plain" else "scaffold_worker"
        info["model_id"] = "plain" if parts[3] == "plain" else parts[3]
        info["replicate_id"] = int(parts[4].replace("rep", ""))
        return info
    if info["kind"] == "auto" and len(parts) == 5 and parts[4].startswith("rep"):
        info["subkind"] = "auto"
        info["model_id"] = parts[3]
        info["replicate_id"] = int(parts[4].replace("rep", ""))
        return info
    return None


def _row_from_csv(
    task_slug: str,
    run_id: str,
    *,
    mode: str,
    model_id: str,
    replicate_id: int,
) -> dict[str, Any] | None:
    root = ensure_run_dir(task_slug, run_id)
    path = root / mode / "outputs.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["replicate_id"] = df["replicate_id"].astype(int)
    sub = df[(df["model_id"] == model_id) & (df["replicate_id"] == replicate_id)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _step_passes_quality(step_id: str, run_id: str) -> bool:
    parsed = _parse_gen_step(step_id)
    if not parsed:
        return True
    if parsed.get("kind") in {"init", "config", "repair", "final_audit"}:
        return True

    mode = "automation" if parsed.get("subkind") == "auto" else "augmentation"
    row = _row_from_csv(
        parsed["task"],
        run_id,
        mode=mode,
        model_id="plain" if parsed.get("subkind") == "plain" else parsed["model_id"],
        replicate_id=parsed["replicate_id"],
    )
    if not row:
        return False
    audit = audit_csv_row(row, task_slug=parsed["task"], mode=mode)
    return bool(audit["ok"])


def _is_upstream_api_failure(raw: str, audit: dict[str, Any]) -> bool:
    """nan / empty rows from LanguageModelNoResponseError — retries won't help for ~70s."""
    text = str(raw or "").strip().lower()
    if text in {"", "nan", "none"}:
        return True
    return audit.get("n_chars", 0) <= 3 and "short<250" in audit.get("issues", [])


def _generate_with_quality(
    run_id: str,
    step_id: str,
    *,
    task: Any,
    generate: Callable[[int], str],
    mode: str,
    condition: str,
) -> None:
    last_issues: list[str] = []
    for attempt in range(_MAX_RETRIES_CTX):
        raw = generate(attempt)
        audit = audit_output_row(
            raw,
            task_slug=task.slug,
            mode=mode,
            condition=condition,
        )
        if audit["ok"]:
            _log(
                run_id,
                f"QUALITY OK {step_id} rep chars={audit['n_chars']} attempt={attempt + 1}",
            )
            return
        last_issues = audit["issues"]
        _log(
            run_id,
            f"QUALITY FAIL {step_id} attempt={attempt + 1}/{_MAX_RETRIES_CTX} "
            f"issues={last_issues} chars={audit['n_chars']}",
        )
        if _is_upstream_api_failure(raw, audit):
            _log(
                run_id,
                f"  upstream API failure (proxy timeout) — skipping remaining retries for {step_id}",
            )
            break
    raise QualityGateError(
        f"{step_id} failed quality gate after {_MAX_RETRIES_CTX} attempts: {last_issues}"
    )


def _repair_single_failure(run_id: str, failure: dict[str, Any], task: Any) -> None:
    task_slug = failure["task"]
    mode = failure["mode"]
    model_id = failure["model_id"]
    replicate_id = int(failure.get("replicate_id", 0))
    root = ensure_run_dir(task_slug, run_id)

    if mode == "automation":
        _generate_with_quality(
            run_id,
            f"repair/{task_slug}/auto/{model_id}/rep{replicate_id}",
            task=task,
            generate=lambda attempt, mid=model_id, rep=replicate_id: generate_automation_replicate(
                task, root, mid, rep, attempt=attempt
            ),
            mode="automation",
            condition=f"automation_{failure.get('model_label', model_id).replace(' ', '_')}",
        )
        return

    if model_id == "plain":
        _generate_with_quality(
            run_id,
            f"repair/{task_slug}/aug/plain/rep{replicate_id}",
            task=task,
            generate=lambda attempt, rep=replicate_id: generate_augmentation_plain_replicate(
                task, root, rep, attempt=attempt
            ),
            mode="augmentation",
            condition="plain",
        )
        return

    model_label = task.assistants.get(model_id, model_id)
    cond = f"scaffold_{_safe_slug(model_label)}"
    _generate_with_quality(
        run_id,
        f"repair/{task_slug}/aug/{model_id}/rep{replicate_id}",
        task=task,
        generate=lambda attempt, mid=model_id, rep=replicate_id: generate_augmentation_worker_replicate(
            task, root, mid, rep, attempt=attempt
        ),
        mode="augmentation",
        condition=cond,
    )


def _build_steps(
    tasks: list[Any],
    *,
    phase: str,
    replicates: int | None = None,
) -> list[tuple[str, Callable[[], None]]]:
    steps: list[tuple[str, Callable[[], None]]] = []

    if phase in {"all", "generation"}:
        for task in tasks:
            slug = task.slug
            n_rep = replicates if replicates is not None else task.replicates

            def _init_task(t=task) -> None:
                r = ensure_run_dir(t.slug, _RUN_ID_CTX)
                _ensure_outputs_csv(r / "augmentation" / "outputs.csv", AUG_COLUMNS)
                _ensure_outputs_csv(r / "automation" / "outputs.csv", AUTO_COLUMNS)

            steps.append((f"gen/{slug}/init", _init_task))

            for rep in range(n_rep):
                def _plain(r=rep, t=task) -> None:
                    root = ensure_run_dir(t.slug, _RUN_ID_CTX)
                    _generate_with_quality(
                        _RUN_ID_CTX,
                        f"gen/{t.slug}/aug/plain/rep{r}",
                        task=t,
                        generate=lambda attempt, rep=r: generate_augmentation_plain_replicate(
                            t, root, rep, attempt=attempt
                        ),
                        mode="augmentation",
                        condition="plain",
                    )

                steps.append((f"gen/{slug}/aug/plain/rep{rep}", _plain))

            for model_id in task.assistants:
                for rep in range(n_rep):
                    def _aug_rep(r=rep, mid=model_id, t=task) -> None:
                        root = ensure_run_dir(t.slug, _RUN_ID_CTX)
                        label = t.assistants[mid]
                        cond = f"scaffold_{_safe_slug(label)}"
                        _generate_with_quality(
                            _RUN_ID_CTX,
                            f"gen/{t.slug}/aug/{mid}/rep{r}",
                            task=t,
                            generate=lambda attempt, rep=r, m=mid: generate_augmentation_worker_replicate(
                                t, root, m, rep, attempt=attempt
                            ),
                            mode="augmentation",
                            condition=cond,
                        )

                    steps.append((f"gen/{slug}/aug/{model_id}/rep{rep}", _aug_rep))

            def _variability_check(t=task) -> None:
                report = check_variability(
                    _RUN_ID_CTX,
                    task_slug=t.slug,
                    modes=["augmentation"],
                )
                var_path = results_base() / "logs" / f"variability_{_RUN_ID_CTX}_{t.slug}.json"
                write_json(var_path, report)
                _log(
                    _RUN_ID_CTX,
                    f"Variability {t.slug}: {report['n_models_checked']} models, "
                    f"{report['n_issues']} issues",
                )
                for warning in report.get("warnings", []):
                    _log(
                        _RUN_ID_CTX,
                        f"  VAR WARN {warning['model_id']}: {warning['issue']}",
                    )
                output_issues = [
                    i for i in report["issues"] if i["issue"] == "identical_outputs_across_replicates"
                ]
                if output_issues:
                    for issue in output_issues:
                        _log(
                            _RUN_ID_CTX,
                            f"  IDENTICAL OUTPUT {issue['model_id']}: {issue['n_chars']} chars",
                        )
                    raise QualityGateError(
                        f"Replicate outputs identical for {t.slug} — likely cache bug. "
                        f"See {var_path}. Regenerate augmentation for this task."
                    )

            steps.append((f"gen/{slug}/variability_check", _variability_check))

            if task.automation_models:
                for model_id in task.automation_models:
                    for rep in range(n_rep):
                        def _auto(r=rep, mid=model_id, t=task) -> None:
                            root = ensure_run_dir(t.slug, _RUN_ID_CTX)
                            label = t.automation_models[mid]
                            cond = f"automation_{label.replace(' ', '_')}"
                            _generate_with_quality(
                                _RUN_ID_CTX,
                                f"gen/{t.slug}/auto/{mid}/rep{r}",
                                task=t,
                                generate=lambda attempt, rep=r, m=mid: generate_automation_replicate(
                                    t, root, m, rep, attempt=attempt
                                ),
                                mode="automation",
                                condition=cond,
                            )

                        steps.append((f"gen/{slug}/auto/{model_id}/rep{rep}", _auto))

            def _config(t=task) -> None:
                r = ensure_run_dir(t.slug, _RUN_ID_CTX)
                write_run_config(
                    r,
                    t,
                    modes=["augmentation", "automation"],
                    worker_model=t.default_worker,
                    replicates=t.replicates,
                    assistants_used=t.assistants,
                    automation_used=t.automation_models,
                )

            steps.append((f"gen/{slug}/config", _config))

        def _repair() -> None:
            tasks_by_slug = {t.slug: t for t in tasks}
            for round_i in range(1, _MAX_REPAIR_ROUNDS_CTX + 1):
                report = audit_all_outputs(_RUN_ID_CTX)
                write_json(_audit_report_path(_RUN_ID_CTX), report)
                if report["n_failing"] == 0:
                    _log(_RUN_ID_CTX, f"Repair round {round_i}: all outputs clean")
                    return
                _log(
                    _RUN_ID_CTX,
                    f"Repair round {round_i}/{_MAX_REPAIR_ROUNDS_CTX}: "
                    f"{report['n_failing']} failing rows",
                )
                for failure in report["failures"]:
                    task = tasks_by_slug[failure["task"]]
                    _repair_single_failure(_RUN_ID_CTX, failure, task)
            report = audit_all_outputs(_RUN_ID_CTX)
            write_json(_audit_report_path(_RUN_ID_CTX), report)
            if report["n_failing"] > 0:
                raise QualityGateError(
                    f"Still {report['n_failing']} failing rows after "
                    f"{_MAX_REPAIR_ROUNDS_CTX} repair rounds; see {_audit_report_path(_RUN_ID_CTX)}"
                )

        steps.append(("gen/repair", _repair))

        def _final_audit() -> None:
            report = audit_all_outputs(_RUN_ID_CTX)
            write_json(_audit_report_path(_RUN_ID_CTX), report)
            _log(
                _RUN_ID_CTX,
                f"Final audit: {report['n_ok']}/{report['n_rows']} ok, "
                f"{report['n_failing']} failing",
            )
            if report["n_failing"] > 0:
                for row in report["failures"][:20]:
                    issues = row["quality_issues"] + row["truncation_signals"]
                    _log(
                        _RUN_ID_CTX,
                        f"  FAIL {row['task']}/{row['mode']} {row['model_label']} "
                        f"rep={row['replicate_id']} {issues}",
                    )
                raise QualityGateError(
                    f"Generation audit failed: {report['n_failing']} bad rows. "
                    f"Re-run with --phase generation (repair step will retry) or inspect "
                    f"{_audit_report_path(_RUN_ID_CTX)}"
                )

        steps.append(("gen/final_audit", _final_audit))

    if phase in {"all", "judge"}:
        for task in tasks:
            slug = task.slug

            def _judge_aug(t=task) -> None:
                r = ensure_run_dir(t.slug, _RUN_ID_CTX)
                judge_augmentation_panel(t, r, n_evals=_N_EVALS_CTX)

            def _judge_auto(t=task) -> None:
                r = ensure_run_dir(t.slug, _RUN_ID_CTX)
                judge_automation_panel(t, r, n_evals=_N_EVALS_CTX)

            steps.append((f"judge/{slug}/augmentation", _judge_aug))
            steps.append((f"judge/{slug}/automation", _judge_auto))

        def _judge_notes() -> None:
            validation = validate_run(_RUN_ID_CTX, task_slugs=[t.slug for t in tasks])
            notes_path = results_base() / f"judging_notes_{_RUN_ID_CTX}.md"
            write_notes(_RUN_ID_CTX, validation, notes_path)

        steps.append(("judge/notes", _judge_notes))

    if phase in {"all", "summarize"}:
        def _summarize() -> None:
            export_cross_task_matrices(_RUN_ID_CTX, task_slugs=[t.slug for t in tasks])

        steps.append(("summarize/cross_task", _summarize))

    return steps


_RUN_ID_CTX: str = ""
_N_EVALS_CTX: int | None = None
_MAX_RETRIES_CTX: int = 3
_MAX_REPAIR_ROUNDS_CTX: int = 5


def _print_status(state: dict[str, Any], steps: list[tuple[str, Callable[[], None]]]) -> None:
    completed = set(state.get("completed", []))
    total = len(steps)
    done = sum(1 for sid, _ in steps if sid in completed)
    _log(state["run_id"], f"Progress: {done}/{total} steps complete")
    pending = [sid for sid, _ in steps if sid not in completed]
    if pending:
        _log(state["run_id"], f"Next pending: {pending[0]}")
    if state.get("failed"):
        _log(state["run_id"], f"Failed steps: {list(state['failed'].keys())}")
    counts = state.get("failure_counts") or {}
    if counts:
        hot = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        _log(state["run_id"], f"Failure counts (hot): {hot}")
    audit_path = _audit_report_path(state["run_id"])
    if audit_path.is_file():
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        _log(
            state["run_id"],
            f"Latest audit: {report.get('n_ok', '?')}/{report.get('n_rows', '?')} ok, "
            f"{report.get('n_failing', '?')} failing",
        )


def main() -> None:
    global _RUN_ID_CTX, _N_EVALS_CTX, _MAX_RETRIES_CTX, _MAX_REPAIR_ROUNDS_CTX

    _load_dotenv()
    os.environ.setdefault("EXPECTED_PARROT_URL", "https://www.expectedparrot.com")
    os.environ.setdefault("EDSL_API_TIMEOUT", "1800")
    os.environ.setdefault("EDSL_MAX_ATTEMPTS", "5")
    os.environ.setdefault("REMOTE_PROXY_TIMEOUT", "1800")

    parser = argparse.ArgumentParser(
        description="Resumable per-replicate batch runner with quality gates."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Shared run id for all tasks.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task slugs; default all.")
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="Override task YAML replicates (use 1 for separate-run replicate passes).",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "generation", "judge", "summarize"],
        default="all",
        help="Which pipeline phase to run (default: all).",
    )
    parser.add_argument("--n-evals", type=int, default=None, help="Override pairwise n_evals.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Per-replicate generation retries when audit fails (default: 3).",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=5,
        help="Repair-loop rounds at end of generation (default: 5).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint; re-run every step (still updates checkpoint).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-audit completed generation steps and redo any that fail quality checks.",
    )
    parser.add_argument("--status", action="store_true", help="Print progress and exit.")
    parser.add_argument("--from-step", default=None, help="Start at this step id (inclusive).")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failures and continue instead of stopping.",
    )
    parser.add_argument(
        "--max-step-failures",
        type=int,
        default=6,
        help="With --continue-on-error, stop if any step fails this many times across "
        "resumes (default: 6). Prevents endless DeepSeek/GPT-OSS retry loops.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=30,
        help="With --continue-on-error, stop after this many consecutive step failures "
        "in one session with no successes (default: 30).",
    )
    parser.add_argument(
        "--skip-variability-gate",
        action="store_true",
        help="Do not require replicate variability before judging.",
    )
    args = parser.parse_args()

    _RUN_ID_CTX = args.run_id
    _MAX_RETRIES_CTX = max(1, args.max_retries)
    _MAX_REPAIR_ROUNDS_CTX = max(1, args.max_repair_rounds)
    os.environ["CENTAUR_RUN_ID"] = args.run_id

    task_slugs = [x.strip() for x in args.tasks.split(",") if x.strip()] if args.tasks else None
    tasks = [load_task(p) for p in _task_yaml_paths(task_slugs)]

    steps = _build_steps(tasks, phase=args.phase, replicates=args.replicates)
    state = _load_checkpoint(args.run_id)

    if args.status:
        _print_status(state, steps)
        return

    if args.phase in {"all", "judge"}:
        mode_label = "remote Expected Parrot Jobs" if use_remote_inference() else "local API proxy"
        _log(args.run_id, f"EDSL mode: {mode_label}")

    if args.phase in {"all", "judge"}:
        audit = audit_run(args.run_id, task_slugs=task_slugs)
        if args.phase == "judge" and not audit["ready_for_judging"]:
            _log(
                args.run_id,
                f"BLOCKED: audit not clean ({len(audit['failures'])} failures). "
                f"Inspect {_audit_report_path(args.run_id)} and re-run generation/repair.",
            )
            raise SystemExit(1)
        if not args.skip_variability_gate and args.phase == "judge":
            var = check_variability(args.run_id, task_slug=None)
            var_path = results_base() / "logs" / f"variability_{args.run_id}.json"
            write_json(var_path, var)
            if not var["ok"]:
                _log(
                    args.run_id,
                    f"BLOCKED: identical replicate outputs ({var['n_issues']} issues). "
                    f"See {var_path}",
                )
                raise SystemExit(1)

    completed: set[str] = set() if args.no_resume else set(state.get("completed", []))
    skip_until = args.from_step is not None
    consecutive_failures = 0

    _log(
        args.run_id,
        f"Starting batch phase={args.phase} ({len(steps)} steps, "
        f"max_retries={_MAX_RETRIES_CTX}, repair_rounds={_MAX_REPAIR_ROUNDS_CTX}"
        f"{', continue-on-error' if args.continue_on_error else ''})",
    )

    for step_id, fn in steps:
        if skip_until:
            if step_id != args.from_step:
                continue
            skip_until = False

        if step_id in completed and not args.no_resume:
            if args.verify and step_id.startswith("gen/") and _parse_gen_step(step_id):
                if _step_passes_quality(step_id, args.run_id):
                    continue
                _log(args.run_id, f"VERIFY: re-running failed quality step {step_id}")
            else:
                continue

        state["current"] = step_id
        _save_checkpoint(state)
        _log(args.run_id, f"STEP START {step_id}")

        try:
            if step_id.startswith("judge/") and step_id != "judge/notes":
                _N_EVALS_CTX = args.n_evals
            fn()
            completed.add(step_id)
            state["completed"] = sorted(completed)
            state["failed"].pop(step_id, None)
            state.setdefault("failure_counts", {}).pop(step_id, None)
            state["current"] = None
            consecutive_failures = 0
            _save_checkpoint(state)
            _log(args.run_id, f"STEP OK {step_id}")
        except Exception as exc:
            tb = traceback.format_exc()
            state["failed"][step_id] = {"error": str(exc), "at": _now(), "traceback": tb}
            state["current"] = None
            _save_checkpoint(state)
            _log(args.run_id, f"STEP FAIL {step_id}: {exc}")
            if args.continue_on_error:
                consecutive_failures += 1
                n_fail = _record_step_failure(state, step_id)
                _save_checkpoint(state)
                _log(args.run_id, f"  failure_count[{step_id}]={n_fail}")
                try:
                    _check_failure_guardrails(
                        step_id=step_id,
                        step_failures=n_fail,
                        consecutive_failures=consecutive_failures,
                        max_step_failures=max(1, args.max_step_failures),
                        max_consecutive_failures=max(1, args.max_consecutive_failures),
                    )
                except FailureExplosionError as guard:
                    _log(args.run_id, f"GUARDRAIL STOP: {guard}")
                    raise SystemExit(2) from guard
            else:
                raise SystemExit(1) from exc

    _log(args.run_id, f"Batch complete for run_id={args.run_id}")
    _print_status(state, steps)


if __name__ == "__main__":
    main()

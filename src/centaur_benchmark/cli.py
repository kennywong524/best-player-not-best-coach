"""CLI: login, run experiments, judge, export artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from centaur_benchmark import __version__
from centaur_benchmark.config import default_tasks_dir, load_task
from centaur_benchmark.io import ensure_run_dir, new_run_id


def _resolve_task_path(task: str) -> Path:
    p = Path(task)
    if p.is_file():
        return p
    td = default_tasks_dir() / f"{task}.yaml"
    if td.is_file():
        return td
    td2 = default_tasks_dir() / task
    if td2.is_file():
        return td2
    raise FileNotFoundError(f"Task not found: {task} (looked under {default_tasks_dir()})")


def _parse_model_subset(s: str | None, base: dict[str, str]) -> dict[str, str]:
    if not s:
        return base
    wanted = {x.strip() for x in s.split(",") if x.strip()}
    out = {k: v for k, v in base.items() if k in wanted or v in wanted}
    if not out:
        raise ValueError(f"No matching models for subset {wanted!r}")
    return out


def cmd_login(_: argparse.Namespace) -> None:
    """Open Expected Parrot login flow (EDSL)."""
    try:
        from edsl import login
    except ImportError as e:
        print(
            "EDSL is not installed or Python is too old. Install with:\n"
            "  pip install edsl\n"
            "Requires Python 3.10+ per PyPI. Then run:\n"
            "  centaur-benchmark login\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    login()


def cmd_run(args: argparse.Namespace) -> None:
    from centaur_benchmark.runner import run_augmentation, run_automation, write_run_config

    task_path = _resolve_task_path(args.task)
    task = load_task(task_path)
    run_id = args.run_id or new_run_id()
    root = ensure_run_dir(task.slug, run_id)

    assistants = _parse_model_subset(args.assistants, task.assistants)
    auto_base = task.automation_models or {}
    auto_models = _parse_model_subset(args.models, auto_base)

    modes: list[str] = []
    if args.mode in ("all", "augmentation"):
        run_augmentation(
            task,
            root,
            worker_model=args.worker or task.default_worker,
            assistants=assistants,
            replicates=args.replicates,
        )
        modes.append("augmentation")
    if args.mode in ("all", "automation"):
        p = run_automation(
            task,
            root,
            models=auto_models if auto_models else None,
            replicates=args.replicates,
        )
        if p is not None:
            modes.append("automation")

    write_run_config(
        root,
        task,
        modes=modes,
        worker_model=args.worker or task.default_worker,
        replicates=args.replicates,
        assistants_used=assistants,
        automation_used=auto_models if auto_models else None,
    )
    print(f"Run directory: {root}")


def cmd_judge(args: argparse.Namespace) -> None:
    from centaur_benchmark.judge_pairwise import (
        judge_augmentation,
        judge_augmentation_panel,
        judge_automation,
        judge_automation_panel,
    )

    task_path = _resolve_task_path(args.task)
    task = load_task(task_path)
    root = ensure_run_dir(task.slug, args.run)

    subset = args.subset
    aug_csv = root / "augmentation" / "outputs.csv"
    auto_csv = root / "automation" / "outputs.csv"
    eval_models = None
    if args.eval_models:
        wanted = [x.strip() for x in args.eval_models.split(",") if x.strip()]
        source = task.evaluator_models or {task.default_evaluator: task.default_evaluator}
        eval_models = {k: source.get(k, k) for k in wanted}
    elif args.panel:
        eval_models = task.evaluator_models or {task.default_evaluator: task.default_evaluator}

    if subset in ("augmentation", "both"):
        if not aug_csv.exists():
            if subset == "augmentation":
                raise FileNotFoundError(f"Missing {aug_csv}")
        else:
            if eval_models:
                judge_augmentation_panel(
                    task,
                    root,
                    eval_models=eval_models,
                    n_evals=args.n_evals,
                    exclude_self_family=not args.include_self_family,
                )
            else:
                judge_augmentation(
                    task,
                    root,
                    eval_model=args.eval_model,
                    n_evals=args.n_evals,
                    n_outer_runs=args.n_outer_runs,
                )
    if subset in ("automation", "both"):
        if not auto_csv.exists():
            if subset == "automation":
                raise FileNotFoundError(f"Missing {auto_csv}")
        else:
            if eval_models:
                judge_automation_panel(
                    task,
                    root,
                    eval_models=eval_models,
                    n_evals=args.n_evals,
                    exclude_self_family=not args.include_self_family,
                )
            else:
                judge_automation(task, root, eval_model=args.eval_model, n_evals=args.n_evals)


def cmd_rubric(args: argparse.Namespace) -> None:
    from centaur_benchmark.judge_rubric import grade_outputs_rubric

    task_path = _resolve_task_path(args.task)
    task = load_task(task_path)
    root = ensure_run_dir(task.slug, args.run)
    grade_outputs_rubric(
        task,
        root,
        subset=args.subset,
        eval_model=args.eval_model,
    )


def cmd_summarize(args: argparse.Namespace) -> None:
    from centaur_benchmark.summarize import export_run_artifacts

    export_run_artifacts(
        args.task,
        args.run,
        dest=Path(args.to) if args.to else None,
        plot=not args.no_plot,
    )


def cmd_summarize_all(args: argparse.Namespace) -> None:
    from centaur_benchmark.summarize import export_cross_task_matrices

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()] if args.tasks else None
    export_cross_task_matrices(
        args.run,
        task_slugs=tasks,
        dest=Path(args.to) if args.to else None,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="centaur-benchmark",
        description="Centaur benchmark harness (EDSL / Expected Parrot).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("login", help="Run edsl.login() to authenticate (terminal-friendly).")
    pl.set_defaults(func=cmd_login)

    pr = sub.add_parser("run", help="Run augmentation and/or automation for a task.")
    pr.add_argument("--task", required=True, help="Task slug (e.g. meal_plan) or path to YAML")
    pr.add_argument(
        "--mode",
        choices=["all", "augmentation", "automation"],
        default="all",
        help="Which pipeline(s) to execute",
    )
    pr.add_argument("--run-id", default=None, help="Optional run id (default: UTC timestamp)")
    pr.add_argument("--worker", default=None, help="Worker model id for augmentation")
    pr.add_argument("--replicates", type=int, default=None, help="Override task replicates")
    pr.add_argument(
        "--assistants",
        default=None,
        help="Comma-separated assistant model ids to subset (default: all in task YAML)",
    )
    pr.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids for automation subset (default: all in task YAML)",
    )
    pr.set_defaults(func=cmd_run)

    pj = sub.add_parser("judge", help="Run pairwise judging on outputs.csv for a run.")
    pj.add_argument("--task", required=True)
    pj.add_argument("--run", required=True, help="Run id folder under results/<task>/")
    pj.add_argument("--subset", choices=["augmentation", "automation", "both"], default="both")
    pj.add_argument("--eval-model", default=None)
    pj.add_argument(
        "--eval-models",
        default=None,
        help="Comma-separated evaluator model ids for panel judging (overrides --eval-model).",
    )
    pj.add_argument("--panel", action="store_true", help="Use all evaluator models configured in the task YAML.")
    pj.add_argument(
        "--include-self-family",
        action="store_true",
        help="Do not exclude same-family judge/output comparisons in panel mode.",
    )
    pj.add_argument("--n-evals", type=int, default=None, help="Pairwise repeats per pair")
    pj.add_argument(
        "--n-outer-runs",
        type=int,
        default=None,
        help="Repeat full pairwise tournament N times (augmentation only; for judge stability)",
    )
    pj.set_defaults(func=cmd_judge)

    prb = sub.add_parser("rubric", help="Optional rubric grading (requires rubric_prompt in task YAML).")
    prb.add_argument("--task", required=True)
    prb.add_argument("--run", required=True)
    prb.add_argument("--subset", choices=["augmentation", "automation"], default="augmentation")
    prb.add_argument("--eval-model", default=None)
    prb.set_defaults(func=cmd_rubric)

    ps = sub.add_parser("summarize", help="Copy leaderboards to artifacts/ and plot.")
    ps.add_argument("--task", required=True)
    ps.add_argument("--run", required=True)
    ps.add_argument("--to", default=None, help="Destination directory (default: artifacts/<task>/<run>)")
    ps.add_argument("--no-plot", action="store_true")
    ps.set_defaults(func=cmd_summarize)

    psa = sub.add_parser("summarize-all", help="Build cross-task model matrices for a shared run id.")
    psa.add_argument("--run", required=True, help="Run id shared across results/<task>/<run>/ folders")
    psa.add_argument("--tasks", default=None, help="Comma-separated task slugs (default: all task YAMLs)")
    psa.add_argument("--to", default=None, help="Destination directory (default: artifacts/cross_task/<run>)")
    psa.set_defaults(func=cmd_summarize_all)

    return p


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check that generation replicates differ (detect EDSL cache / temp=0 issues)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from centaur_benchmark.io import results_base  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(str(text).encode()).hexdigest()


def check_variability(
    run_id: str,
    *,
    task_slug: str | None = None,
    modes: list[str] | None = None,
    min_reps: int = 2,
) -> dict:
    modes = modes or ["augmentation", "automation"]
    base = results_base()
    issues: list[dict] = []
    warnings: list[dict] = []
    summary: list[dict] = []

    task_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if task_slug:
        task_dirs = [p for p in task_dirs if p.name == task_slug]

    for task_dir in task_dirs:
        slug = task_dir.name
        for mode in modes:
            path = task_dir / run_id / mode / "outputs.csv"
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            for model_id, g in df.groupby("model_id"):
                g = g.sort_values("replicate_id")
                n = len(g)
                if n < min_reps:
                    continue
                out_hashes = [_sha(x) for x in g["output"].astype(str)]
                uniq_out = len(set(out_hashes))
                scaf_col = g.get("scaffold_sha256")
                uniq_scaf = 0
                if scaf_col is not None and scaf_col.notna().any():
                    uniq_scaf = scaf_col.nunique()
                row = {
                    "task": slug,
                    "mode": mode,
                    "model_id": model_id,
                    "n_reps": n,
                    "unique_outputs": uniq_out,
                    "unique_scaffolds": int(uniq_scaf) if uniq_scaf else None,
                    "ok": uniq_out > 1 or n == 1,
                }
                summary.append(row)
                if n > 1 and uniq_out == 1:
                    issues.append(
                        {
                            **row,
                            "issue": "identical_outputs_across_replicates",
                            "n_chars": int(g["output"].astype(str).str.len().iloc[0]),
                        }
                    )
                if mode == "augmentation" and model_id != "plain" and n > 1 and uniq_scaf == 1:
                    warnings.append({**row, "issue": "identical_scaffolds_across_replicates"})

    return {
        "run_id": run_id,
        "task_slug": task_slug,
        "n_models_checked": len(summary),
        "n_issues": len(issues),
        "n_warnings": len(warnings),
        "ok": len(issues) == 0,
        "summary": summary,
        "issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify replicate outputs are not identical.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", default=None, help="Single task slug")
    parser.add_argument("--modes", default="augmentation,automation")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--fail", action="store_true", help="Exit 1 if any issues")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    report = check_variability(args.run_id, task_slug=args.task, modes=modes)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"checked={report['n_models_checked']} issues={report['n_issues']} "
        f"warnings={report['n_warnings']} ok={report['ok']}"
    )
    for issue in report["issues"]:
        print(
            f"  FAIL {issue['task']}/{issue['mode']} {issue['model_id']}: "
            f"{issue['issue']} ({issue['n_reps']} reps)"
        )
    for warning in report.get("warnings", []):
        print(
            f"  WARN {warning['task']}/{warning['mode']} {warning['model_id']}: "
            f"{warning['issue']} ({warning['n_reps']} reps)"
        )
    if args.fail and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

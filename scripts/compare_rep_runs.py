#!/usr/bin/env python3
"""Compare generation outputs between two run_ids (e.g. v4 rep0 vs rep1 pass)."""

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
    return hashlib.sha256(str(text).encode()).hexdigest()[:12]


def _preview(text: str, n: int = 180) -> str:
    s = str(text or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def compare_runs(
    *,
    base_run_id: str,
    new_run_id: str,
    base_replicate: int = 0,
    new_replicate: int = 0,
    task_slug: str | None = None,
    modes: list[str] | None = None,
) -> dict:
    modes = modes or ["augmentation", "automation"]
    base = results_base()
    rows: list[dict] = []
    identical = 0
    missing = 0

    task_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if task_slug:
        task_dirs = [p for p in task_dirs if p.name == task_slug]

    for task_dir in task_dirs:
        slug = task_dir.name
        for mode in modes:
            base_path = task_dir / base_run_id / mode / "outputs.csv"
            new_path = task_dir / new_run_id / mode / "outputs.csv"
            if not base_path.is_file() or not new_path.is_file():
                continue
            bdf = pd.read_csv(base_path)
            ndf = pd.read_csv(new_path)
            bdf = bdf[bdf["replicate_id"].astype(int) == base_replicate]
            ndf = ndf[ndf["replicate_id"].astype(int) == new_replicate]
            for model_id in sorted(set(bdf["model_id"].astype(str)) | set(ndf["model_id"].astype(str))):
                brow = bdf[bdf["model_id"].astype(str) == model_id]
                nrow = ndf[ndf["model_id"].astype(str) == model_id]
                if brow.empty or nrow.empty:
                    missing += 1
                    rows.append(
                        {
                            "task": slug,
                            "mode": mode,
                            "model_id": model_id,
                            "status": "missing_side",
                            "base_chars": int(brow["output"].astype(str).str.len().iloc[0]) if len(brow) else None,
                            "new_chars": int(nrow["output"].astype(str).str.len().iloc[0]) if len(nrow) else None,
                        }
                    )
                    continue
                bout = str(brow.iloc[0]["output"] or "")
                nout = str(nrow.iloc[0]["output"] or "")
                same = _sha(bout) == _sha(nout)
                if same:
                    identical += 1
                rows.append(
                    {
                        "task": slug,
                        "mode": mode,
                        "model_id": model_id,
                        "status": "IDENTICAL" if same else "different",
                        "base_chars": len(bout),
                        "new_chars": len(nout),
                        "base_sha": _sha(bout),
                        "new_sha": _sha(nout),
                        "base_preview": _preview(bout),
                        "new_preview": _preview(nout),
                    }
                )

    return {
        "base_run_id": base_run_id,
        "new_run_id": new_run_id,
        "base_replicate": base_replicate,
        "new_replicate": new_replicate,
        "n_compared": len(rows) - missing,
        "n_identical": identical,
        "n_missing": missing,
        "ok": identical == 0 and missing == 0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualitative compare: base run rep0 vs new run (separate replicate pass)."
    )
    parser.add_argument("--base-run-id", default="20260610_scaffold_strict_v4")
    parser.add_argument("--new-run-id", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--modes", default="augmentation,automation")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--fail", action="store_true", help="Exit 1 if any identical outputs")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    report = compare_runs(
        base_run_id=args.base_run_id,
        new_run_id=args.new_run_id,
        task_slug=args.task,
        modes=modes,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"compared={report['n_compared']} identical={report['n_identical']} "
        f"missing={report['n_missing']} ok={report['ok']}"
    )
    for row in report["rows"]:
        flag = row["status"]
        print(
            f"  [{flag}] {row['task']}/{row['mode']} {row['model_id']}: "
            f"base={row.get('base_chars')} new={row.get('new_chars')}"
        )
        if flag == "different":
            print(f"    base: {row.get('base_preview', '')}")
            print(f"    new:  {row.get('new_preview', '')}")
        elif flag == "IDENTICAL":
            print(f"    preview: {row.get('base_preview', '')}")
    if args.fail and report["n_identical"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

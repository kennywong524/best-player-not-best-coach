#!/usr/bin/env python3
"""Backfill broken automation rows in a target run from a source run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from centaur_benchmark.io import results_base


def _pairs_from_audit(audit_path: Path, *, mode: str = "automation") -> dict[str, set[str]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    by_task: dict[str, set[str]] = {}
    for f in audit.get("failures", []):
        if str(f.get("mode", "")) != mode:
            continue
        by_task.setdefault(str(f["task"]), set()).add(str(f["model_id"]))
    return by_task


def backfill_automation(
    *,
    target_run_id: str,
    source_run_id: str,
    model_ids_by_task: dict[str, set[str]],
) -> dict:
    manifest: dict = {
        "target_run_id": target_run_id,
        "source_run_id": source_run_id,
        "backfilled": [],
        "skipped": [],
    }
    base = results_base()

    for task_slug, model_ids in sorted(model_ids_by_task.items()):
        src = base / task_slug / source_run_id / "automation" / "outputs.csv"
        dst = base / task_slug / target_run_id / "automation" / "outputs.csv"
        if not src.is_file():
            manifest["skipped"].append({"task": task_slug, "reason": f"missing source {src}"})
            continue
        if not dst.is_file():
            manifest["skipped"].append({"task": task_slug, "reason": f"missing target {dst}"})
            continue

        src_df = pd.read_csv(src)
        dst_df = pd.read_csv(dst)
        replace_ids = sorted(model_ids)
        missing = sorted(set(replace_ids) - set(src_df["model_id"].astype(str)))
        if missing:
            manifest["skipped"].append({"task": task_slug, "reason": f"missing models in source: {missing}"})
            replace_ids = [m for m in replace_ids if m not in missing]
        if not replace_ids:
            continue

        donor = src_df[src_df["model_id"].astype(str).isin(replace_ids)].copy()
        kept = dst_df[~dst_df["model_id"].astype(str).isin(replace_ids)].copy()
        merged = pd.concat([kept, donor], ignore_index=True)
        # Stable order: model_id then replicate_id
        merged.sort_values(["model_id", "replicate_id"], inplace=True)
        merged.to_csv(dst, index=False)

        for mid in replace_ids:
            sub = donor[donor["model_id"].astype(str) == mid]
            n_chars = [len(str(x or "")) for x in sub["output"]]
            manifest["backfilled"].append(
                {
                    "task": task_slug,
                    "model_id": mid,
                    "model_label": str(sub.iloc[0]["model_label"]) if len(sub) else mid,
                    "n_rows": len(sub),
                    "n_chars": n_chars,
                    "source": str(src),
                    "target": str(dst),
                }
            )
        print(f"Backfilled {task_slug}: {replace_ids} ({len(donor)} rows from {source_run_id})")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill automation outputs from another run.")
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument(
        "--audit-json",
        required=True,
        help="Audit JSON listing failing automation rows to replace.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Write JSON manifest of backfilled rows (default: results/backfill_<target>_from_<source>.json).",
    )
    args = parser.parse_args()

    pairs = _pairs_from_audit(Path(args.audit_json), mode="automation")
    if not pairs:
        raise SystemExit("No automation failures found in audit JSON.")

    manifest = backfill_automation(
        target_run_id=args.target_run_id,
        source_run_id=args.source_run_id,
        model_ids_by_task=pairs,
    )
    out = Path(args.manifest_out or results_base() / f"backfill_{args.target_run_id}_automation_from_{args.source_run_id}.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out}")
    print(f"Backfilled {len(manifest['backfilled'])} model×task groups, skipped {len(manifest['skipped'])}")


if __name__ == "__main__":
    main()

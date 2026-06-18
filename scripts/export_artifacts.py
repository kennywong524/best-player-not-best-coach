#!/usr/bin/env python3
"""CLI wrapper: copy leaderboards/plots from results/ to artifacts/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from centaur_benchmark.summarize import export_run_artifacts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Export run outputs to artifacts/")
    p.add_argument("--task", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--to", default=None)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    export_run_artifacts(args.task, args.run, dest=Path(args.to) if args.to else None, plot=not args.no_plot)


if __name__ == "__main__":
    main()

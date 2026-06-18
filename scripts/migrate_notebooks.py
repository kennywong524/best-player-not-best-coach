#!/usr/bin/env python3
"""Copy legacy Colab notebooks into notebooks/legacy/*.ipynb with outputs stripped."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "legacy"

SOURCES = [
    ROOT / "menu planning" / "menu-planning.ipynb",
    ROOT / "counselling" / "counselling",
    ROOT / "market trend analysis" / "market-trend-analysis",
    ROOT / "operations research" / "operations-research",
    ROOT / "tutoring" / "tutoring",
    ROOT / "travel planning" / "travel-planning",
    ROOT / "tax preparation" / "tax-prep",
]


def strip_outputs(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        if not src.exists():
            print("missing", src, file=sys.stderr)
            continue
        nb = json.loads(src.read_text(encoding="utf-8"))
        nb = strip_outputs(nb)
        name = src.name if src.suffix == ".ipynb" else src.name + ".ipynb"
        out = DEST / name
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("wrote", out.relative_to(ROOT))

    # Optional: keep a copy of old paths (do not delete user data); README points to legacy/
    readme = DEST / "README.md"
    readme.write_text(
        "These notebooks are **archived** from the Colab workflow. Prefer the CLI:\n\n"
        "```bash\n"
        "centaur-benchmark run --task meal_plan --mode all\n"
        "```\n",
        encoding="utf-8",
    )
    print("wrote", readme.relative_to(ROOT))


if __name__ == "__main__":
    main()

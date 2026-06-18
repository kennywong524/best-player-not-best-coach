"""Run directory layout and metadata."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_base() -> Path:
    return repo_root() / "results"


def run_dir(task_slug: str, run_id: str) -> Path:
    return results_base() / task_slug / run_id


def ensure_run_dir(task_slug: str, run_id: str) -> Path:
    d = run_dir(task_slug, run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "augmentation").mkdir(exist_ok=True)
    (d / "automation").mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    return d


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")

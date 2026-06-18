#!/usr/bin/env python3
"""Wait for generation, audit, retry failures, then judge and summarize."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
RUN_ID = "20260609_stable_frontier_v2"
POLL_SEC = 60


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _pipeline_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "run_full_pipeline.py"], capture_output=True, text=True)
    return r.returncode == 0


def _task_status(run_id: str) -> str:
    tasks = ["counselling", "market_trends", "meal_plan", "operations_research", "tax_prep", "travel_planning", "tutoring"]
    done = 0
    lines = []
    for task in tasks:
        base = ROOT / "results" / task / run_id
        aug = (base / "augmentation" / "outputs.csv").exists()
        auto = (base / "automation" / "outputs.csv").exists()
        if aug and auto:
            done += 1
        lines.append(f"{task}: {'done' if aug and auto else 'pending'}")
    return f"{done}/7 tasks complete\n" + "\n".join(lines)


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {' '.join(cmd)} ===\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        return proc.wait()


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--poll-sec", type=int, default=POLL_SEC)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monitor_log = log_dir / f"post_generation_{args.run_id}.log"

    if not args.skip_wait:
        print(f"Waiting for generation pipeline (run_id={args.run_id})...")
        with monitor_log.open("a", encoding="utf-8") as log:
            while _pipeline_running():
                status = _task_status(args.run_id)
                msg = f"[{time.strftime('%H:%M:%S')}] still running\n{status}\n"
                print(msg)
                log.write(msg)
                log.flush()
                time.sleep(args.poll_sec)
            log.write(f"[{time.strftime('%H:%M:%S')}] generation process exited\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    rc = _run([str(PY), "scripts/audit_run.py", "--run-id", args.run_id], monitor_log)
    if rc != 0:
        sys.exit(rc)

    rc = _run([str(PY), "scripts/retry_failed.py", "--run-id", args.run_id, "--max-rounds", "3"], monitor_log)
    if rc != 0:
        sys.exit(rc)

    audit_path = ROOT / "results" / f"audit_{args.run_id}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("ready_for_judging"):
        print(f"NOT READY FOR JUDGING — see {audit_path}")
        sys.exit(1)

    print("Starting panel judging + summarize...")
    rc = _run(
        [
            str(PY),
            "scripts/run_full_pipeline.py",
            "--run-id",
            args.run_id,
            "--mode",
            "judge",
        ],
        monitor_log,
    )
    if rc != 0:
        sys.exit(rc)

    rc = _run(
        [
            str(PY),
            "scripts/run_full_pipeline.py",
            "--run-id",
            args.run_id,
            "--mode",
            "summarize",
        ],
        monitor_log,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe Expected Parrot for the highest max_tokens each automation model accepts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def _load_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
    os.environ["EDSL_API_TIMEOUT"] = "1800"
    os.environ["REMOTE_PROXY_TIMEOUT"] = "1800"


AUTOMATION_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4.1",
    "gpt-5-mini-2025-08-07",
    "deepseek-ai/DeepSeek-V3.1",
    "o4-mini-2025-04-16",
    "o3-mini-2025-01-31",
    "openai/gpt-oss-120b",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-8",
    "google/gemini-3.1-pro",
]

CANDIDATE_CAPS = [128_000, 100_000, 65_536, 32_768, 16_384, 8_192, 4_096]


def _probe(model_id: str, max_tokens: int) -> dict:
    from edsl import Agent, Model, Scenario, ScenarioList, Survey, QuestionFreeText

    from centaur_benchmark.edsl_runtime import edsl_run_kwargs
    from centaur_benchmark.runner import _generation_model_kwargs

    if "gpt-5" in model_id:
        return {"skipped": True, "reason": "gpt-5 omits max_tokens in runner"}

    q = QuestionFreeText("output", "{{ scenario.task_prompt }}")
    survey = Survey([q])
    worker = Agent(instruction="Reply briefly with exactly: OK")
    scenarios = ScenarioList([Scenario({"task_prompt": "Reply with exactly: OK"})])
    kwargs = _generation_model_kwargs(model_id, {"max_tokens": max_tokens})
    desc = f"probe-max-tokens-{model_id.replace('/', '_')}-{max_tokens}"
    try:
        results = (
            survey.by(scenarios)
            .by(worker)
            .by(Model(model_id, **kwargs))
            .run(**edsl_run_kwargs(description=desc, visibility="private", n=1))
        )
        out = str(results.select("answer.output").to_pandas().iloc[0, 0] or "")
        ok = out.lower() not in {"nan", ""} and len(out) >= 2
        return {"ok": ok, "n_chars": len(out), "preview": out[:80]}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200]}


def probe_model(model_id: str) -> dict:
    if "gpt-5" in model_id:
        return {
            "model_id": model_id,
            "omit_max_tokens": True,
            "recommended_cap": None,
            "tests": [],
        }
    tests: list[dict] = []
    highest_ok: int | None = None
    for cap in CANDIDATE_CAPS:
        result = _probe(model_id, cap)
        row = {"max_tokens": cap, **result}
        tests.append(row)
        print(f"  {model_id} max_tokens={cap}: {'OK' if result.get('ok') else 'FAIL'}")
        if result.get("ok"):
            highest_ok = cap
            break  # caps are descending; first OK is highest working
    return {
        "model_id": model_id,
        "omit_max_tokens": False,
        "recommended_cap": highest_ok,
        "tests": tests,
    }


def verify_config_caps() -> dict:
    """Test the exact caps in runner._AUTOMATION_MAX_TOKENS_CAP via EDSL."""
    from centaur_benchmark.config import default_tasks_dir, load_task
    from centaur_benchmark.runner import (
        _AUTOMATION_MAX_TOKENS_CAP,
        _automation_model_kwargs,
        _DEFAULT_AUTOMATION_MAX_TOKENS,
    )

    task = load_task(default_tasks_dir() / "counselling.yaml")
    rows: list[dict] = []
    for model_id in AUTOMATION_MODELS:
        kwargs = _automation_model_kwargs(task, model_id)
        if not kwargs:
            from edsl import Agent, Model, Scenario, ScenarioList, Survey, QuestionFreeText

            from centaur_benchmark.edsl_runtime import edsl_run_kwargs
            from centaur_benchmark.runner import _generation_model_kwargs

            q = QuestionFreeText("output", "{{ scenario.task_prompt }}")
            survey = Survey([q])
            worker = Agent(instruction="Reply briefly with exactly: OK")
            scenarios = ScenarioList([Scenario({"task_prompt": "Reply with exactly: OK"})])
            try:
                results = (
                    survey.by(scenarios)
                    .by(worker)
                    .by(Model(model_id, **_generation_model_kwargs(model_id)))
                    .run(
                        **edsl_run_kwargs(
                            description=f"verify-config-{model_id.replace('/', '_')}-omit",
                            visibility="private",
                            n=1,
                        )
                    )
                )
                out = str(results.select("answer.output").to_pandas().iloc[0, 0] or "")
                ok = out.lower() not in {"nan", ""} and len(out) >= 2
                row = {
                    "model_id": model_id,
                    "configured_cap": None,
                    "omit_max_tokens": True,
                    "ok": ok,
                    "n_chars": len(out),
                }
            except Exception as exc:
                row = {
                    "model_id": model_id,
                    "configured_cap": None,
                    "omit_max_tokens": True,
                    "ok": False,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:200],
                }
        else:
            cap = kwargs["max_tokens"]
            row = {
                "model_id": model_id,
                "configured_cap": cap,
                "omit_max_tokens": False,
                **_probe(model_id, cap),
            }
        status = "OK" if row.get("ok") else "FAIL"
        cap_label = row.get("configured_cap") if row.get("configured_cap") else "omit"
        print(f"  {model_id} cap={cap_label}: {status}")
        rows.append(row)

    ok_all = all(r.get("ok") for r in rows)
    return {
        "mode": "verify_config",
        "default_cap": _DEFAULT_AUTOMATION_MAX_TOKENS,
        "configured_caps": dict(_AUTOMATION_MAX_TOKENS_CAP),
        "ok": ok_all,
        "models": rows,
    }


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(AUTOMATION_MODELS))
    parser.add_argument("--json-out", default="results/logs/automation_max_tokens_probe.json")
    parser.add_argument(
        "--verify-config",
        action="store_true",
        help="Test runner._AUTOMATION_MAX_TOKENS_CAP values (not ceiling discovery).",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.verify_config:
        print("Verifying configured caps via EDSL...")
        report = verify_config_caps()
        out_path = Path(args.json_out.replace(".json", "_verify.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out_path}")
        print(f"All configured caps OK: {report['ok']}")
        if not report["ok"]:
            raise SystemExit(1)
        return

    report: dict = {"models": []}
    for model_id in models:
        print(f"Probing {model_id}...")
        report["models"].append(probe_model(model_id))

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("\nRecommended caps (EP ceiling discovery):")
    for row in report["models"]:
        if row.get("omit_max_tokens"):
            print(f"  {row['model_id']}: omit max_tokens")
        else:
            print(f"  {row['model_id']}: {row['recommended_cap']}")


if __name__ == "__main__":
    main()

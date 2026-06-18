#!/usr/bin/env bash
# Overnight-safe resumable batch: per-replicate generation + quality gates.
#
# Each replicate is audited immediately (empty, truncation, task deliverable rules).
# Failed replicates retry up to 3x before stopping. End-of-gen repair pass catches stragglers.
# With --continue-on-error, the runner keeps going unless the same step fails too many
# times (--max-step-failures, default 6) — then stop and patch those rows manually.
#
# Usage (run from your Mac terminal, not inside Cursor agent):
#   ./scripts/start_resumable_batch.sh --phase generation   # auto --continue-on-error
#   RUN_ID=20260611_replicates3_v1 ./scripts/start_resumable_batch.sh --status
#   ./scripts/start_resumable_batch.sh --phase judge   # only after audit + variability clean
#
# Overnight: start generation and go to sleep. Flaky rows (DeepSeek/GPT-OSS) fail fast
# and the run continues. Guardrails stop endless retries (6 fails/step). Repair at end.
# Morning only:
#   ./scripts/start_resumable_batch.sh --status
#   cat results/logs/audit_${RUN_ID}.json | python -m json.tool | less
#
# Before judging, require clean audit + variability:
#   PYTHONPATH=src:scripts .venv/bin/python scripts/check_replicate_variability.py \
#     --run-id ${RUN_ID} --fail

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-20260612_rep1}"
REPLICATES="${REPLICATES:-1}"
# v4 (20260610_scaffold_strict_v4) used local Expected Parrot API proxy, not remote Jobs.
CENTAUR_EDSL_REMOTE="${CENTAUR_EDSL_REMOTE:-0}"
export CENTAUR_EDSL_REMOTE
export CENTAUR_RUN_ID="$RUN_ID"
PHASE="${PHASE:-generation}"
EXTRA_ARGS=("$@")
HAS_PHASE=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--phase" ]]; then
    HAS_PHASE=true
    break
  fi
done
if ! $HAS_PHASE; then
  EXTRA_ARGS=(--phase "$PHASE" "${EXTRA_ARGS[@]}")
fi

# Generation overnight: skip flaky rows, don't require babysitting.
EFFECTIVE_PHASE="$PHASE"
HAS_CONTINUE=false
HAS_STOP=false
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[i]}"
  if [[ "$arg" == "--continue-on-error" ]]; then HAS_CONTINUE=true; fi
  if [[ "$arg" == "--stop-on-error" ]]; then HAS_STOP=true; fi
  if [[ "$arg" == "--phase" && -n "${EXTRA_ARGS[i + 1]:-}" ]]; then
    EFFECTIVE_PHASE="${EXTRA_ARGS[i + 1]}"
  fi
done
if [[ "$EFFECTIVE_PHASE" == "generation" ]] && ! $HAS_CONTINUE && ! $HAS_STOP; then
  EXTRA_ARGS=(--continue-on-error "${EXTRA_ARGS[@]}")
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-}:src:scripts"
# Override .env — 600s + 128k max_tokens causes proxy timeouts on fresh calls.
export EDSL_API_TIMEOUT=1800
export EDSL_MAX_ATTEMPTS="${EDSL_MAX_ATTEMPTS:-8}"
export REMOTE_PROXY_TIMEOUT=1800

mkdir -p results/logs
LOG="results/logs/batch_${RUN_ID}.log"
PIDFILE="results/logs/batch_${RUN_ID}.pid"

if [[ " ${EXTRA_ARGS[*]} " == *" --status "* ]]; then
  exec .venv/bin/python scripts/run_resumable_batch.py --run-id "$RUN_ID" --status "${EXTRA_ARGS[@]}"
fi

if [[ -f "$PIDFILE" ]]; then
  OLD_PID="$(cat "$PIDFILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Batch already running (pid $OLD_PID). Monitor: tail -f $LOG"
    exit 0
  fi
fi

EDSL_MODE="remote Expected Parrot Jobs"
if [[ "${CENTAUR_EDSL_REMOTE}" == "0" ]]; then
  EDSL_MODE="local API proxy"
fi
HAS_REPLICATES=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--replicates" ]]; then HAS_REPLICATES=true; break; fi
done

echo "Starting overnight batch run_id=$RUN_ID replicates=$REPLICATES (EDSL: $EDSL_MODE)"
echo "Log: $LOG"
echo "Audit report: results/logs/audit_${RUN_ID}.json"
BATCH_CMD=(
  .venv/bin/python scripts/run_resumable_batch.py
  --run-id "$RUN_ID"
  --verify
  --max-retries 3
  --max-repair-rounds 5
)
if ! $HAS_REPLICATES; then
  BATCH_CMD+=(--replicates "$REPLICATES")
fi
BATCH_CMD+=("${EXTRA_ARGS[@]}")
nohup "${BATCH_CMD[@]}" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "PID $(cat "$PIDFILE"). Disconnect-safe — re-run this script to resume."

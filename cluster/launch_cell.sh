#!/bin/bash
# One cell on the VM, detached:
#   SEEDS=5 BUDGET=16 setsid nohup bash cluster/launch_cell.sh <cell> <tag> [arm,arm,...] > /dev/null 2>&1 < /dev/null & disown
# Env passed through: SEEDS, FIRST_SEED, BUDGET, ANALYST_MODEL (default gemini-2.5-flash), MEMORY_PATH.
# SMOKE_FIRST=1 runs the smoke cell (one round, every arm) before the cell and stops if it fails.
# POWER_OFF=1 powers the VM off when everything is done (a stopped VM bills only its disk).
# Log: results/<cell>_<tag>.log (one line per design; read it with `python -m loop.summarize progress`).
cd ~/hackathon
CELL="$1"; TAG="$2"; ARMS="${3:-}"
export ANALYST_MODEL="${ANALYST_MODEL:-gemini-2.5-flash}"
export RAY_DEDUP_LOGS=0          # every run prints its own rows; Ray would collapse identical lines
LOG=results/${CELL}_${TAG}.log
echo $$ > results/${CELL}_${TAG}.pid
if [ "${SMOKE_FIRST:-0}" = "1" ]; then
  SEEDS=1 BUDGET=2 LOOP_DISPATCH=chia .venv/bin/python -m loop.run smoke ${TAG} > results/smoke_${TAG}.log 2>&1
  if ! grep -q "0 failed" results/smoke_${TAG}.log; then
    echo "SMOKE_FAILED $(date)" >> $LOG
    if [ "${POWER_OFF:-0}" = "1" ]; then sudo shutdown -h now; fi
    exit 1
  fi
fi
LOOP_DISPATCH=chia .venv/bin/python -m loop.run $CELL $TAG $ARMS > $LOG 2>&1
.venv/bin/python -m loop.summarize results/run_${CELL}_${TAG}.json >> $LOG 2>&1
echo "CELL_DONE $(date)" >> $LOG
if [ "${POWER_OFF:-0}" = "1" ]; then
  sleep 600            # time to fetch the results before the machine goes down
  sudo shutdown -h now
fi

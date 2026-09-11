#!/bin/bash
# One cell on the VM, detached:
#   SEEDS=2 setsid nohup bash cluster/launch_cell.sh <cell> <tag> [arm,arm,...] > /dev/null 2>&1 < /dev/null & disown
# Env passed through: SEEDS, FIRST_SEED, BUDGET, ANALYST_MODEL (default gemini-2.5-flash), MEMORY_PATH.
# Log: results/<cell>_<tag>.log (one line per design; read it with `python -m loop.summarize progress`).
cd ~/hackathon
CELL="$1"; TAG="$2"; ARMS="${3:-}"
export ANALYST_MODEL="${ANALYST_MODEL:-gemini-2.5-flash}"
export RAY_DEDUP_LOGS=0          # every run prints its own rows; Ray would collapse identical lines
LOG=results/${CELL}_${TAG}.log
echo $$ > results/${CELL}_${TAG}.pid
LOOP_DISPATCH=chia .venv/bin/python -m loop.run $CELL $TAG $ARMS > $LOG 2>&1
.venv/bin/python -m loop.summarize results/run_${CELL}_${TAG}.json >> $LOG 2>&1
echo "CELL_DONE $(date)" >> $LOG

#!/bin/bash
# One cell on the VM, detached: bash cluster/launch_cell.sh <cell> <tag> <arms> [SEEDS=n FIRST_SEED=n ROUNDS=n MEMORY_PATH=... PLAYBOOK=...]
# PLAYBOOK reuses another cell's playbook (w2 reads w1's: same training tables, no second distillation).
cd ~/hackathon
CELL="$1"; TAG="$2"; ARMS="$3"
export ANALYST_MODEL="${ANALYST_MODEL:-gemini-2.5-pro}"
export RAY_DEDUP_LOGS=0          # every arm prints its own rows; Ray would collapse identical lines
export MEMORY_PATH="${MEMORY_PATH:-results/experiment_${TAG}_memory.json}"
PLAYBOOK="${PLAYBOOK:-results/experiment_${TAG}_playbook.json}"
echo $$ > results/${TAG}_${CELL}_launch.pid
LOG=results/${TAG}_${CELL}_s${FIRST_SEED:-0}.log
.venv/bin/python -m loop.run_chia local $CELL $TAG $PLAYBOOK $ARMS > $LOG 2>&1
.venv/bin/python -m loop.summarize results/experiment_${TAG}_${CELL}.json --uniform 300 > results/${TAG}_${CELL}_summary.txt 2>&1
echo "CELL_DONE $(date)" >> $LOG

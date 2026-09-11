#!/bin/bash
# One unattended session on the VM, in dependency order:
#   bash cluster/chain.sh <tag>
# Steps: smoke (every arm, one round, through CHIA) -> learn w1 (playbook + pool + memory
# from chip C's SPEC tables) -> cell w1 -> consolidate w1's memory -> [RUN_W2=1] cell w2
# with the consolidated memory (needs GAP set 2 traces and their reference rows on disk)
# -> [RUN_K=1] learn k (from chips A+B) and cell k. Each step logs to results/<tag>_<step>.log;
# a failed step never aborts the chain; the VM shuts itself down at the end unless NO_SHUTDOWN=1.
# Switches: SKIP_SMOKE=1, SKIP_LEARN=1 (reuse LEARN_TAG's playbook/pool/memory), SKIP_W1=1,
# SEEDS (w1/w2, default 2), SEEDS_K (default 2), ROUNDS (default 24), ARMS (subset for w1/w2).
# Stop a chain with: kill $(cat results/<tag>_chain.pid)
set -u
TAG="${1:-w}"
cd ~/hackathon
echo $$ > results/${TAG}_chain.pid
LEARN_TAG="${LEARN_TAG:-$TAG}"
export ANALYST_MODEL="${ANALYST_MODEL:-gemini-2.5-pro}"
PY=.venv/bin/python
T=traces
GAP2="$T/bfs.road-99B.champsimtrace.xz $T/pr.web-16B.champsimtrace.xz $T/pr.road-28B.champsimtrace.xz"
ARMS="${ARMS:-}"

step() {  # step <name> <command...>: run, log, record the outcome, never abort the chain
  local name="$1"; shift
  echo "== $(date '+%F %T') start $name" | tee -a results/${TAG}_chain.log
  "$@" > results/${TAG}_${name}.log 2>&1
  local code=$?
  echo "== $(date '+%F %T') end $name (exit $code)" | tee -a results/${TAG}_chain.log
  return $code
}

# 1. Smoke: every arm for one round on two workloads, through CHIA, with an uncached seed.
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  step smoke env FIRST_SEED=3 SEEDS=1 $PY -m loop.run_chia local smoke ${TAG}
fi

# 2. w1's knowledge: playbook (rules) and pool from chip C's SPEC tables, memory from the same tables.
if [ "${SKIP_LEARN:-0}" != "1" ]; then
  step learn_w1 $PY -m loop.run learn w1 ${LEARN_TAG}
fi
PLAYBOOK=results/experiment_${LEARN_TAG}_playbook.json
MEMORY=results/experiment_${LEARN_TAG}_memory.json
MEMORY_W2=results/experiment_${TAG}_memory_after_w1.json

# 3. Cell w1, then the consolidated memory for w2.
if [ "${SKIP_W1:-0}" != "1" ] && [ -f $PLAYBOOK ]; then
  step w1 env SEEDS=${SEEDS:-2} MEMORY_PATH=$MEMORY $PY -m loop.run_chia local w1 ${TAG} $PLAYBOOK $ARMS
  step summary_w1 $PY -m loop.summarize results/experiment_${TAG}_w1.json --uniform 300
  step consolidate $PY -m loop.memory consolidate $MEMORY $MEMORY_W2 w1
fi

# 4. Cell w2 with the consolidated memory (opt-in; needs GAP set 2 on disk).
if [ "${RUN_W2:-0}" = "1" ] && [ -f $MEMORY_W2 ]; then
  MISSING=0
  for trace in $GAP2; do [ -f "$trace" ] || { echo "MISSING $trace" | tee -a results/${TAG}_chain.log; MISSING=1; }; done
  if [ $MISSING -eq 0 ]; then
    step w2 env SEEDS=${SEEDS:-2} MEMORY_PATH=$MEMORY_W2 $PY -m loop.run_chia local w2 ${TAG} $PLAYBOOK $ARMS
    step summary_w2 $PY -m loop.summarize results/experiment_${TAG}_w2.json --uniform 300
  fi
fi

# 5. Cell k: the chip-boundary control (opt-in).
if [ "${RUN_K:-0}" = "1" ]; then
  step learn_k $PY -m loop.run learn k ${TAG}_k
  step k env SEEDS=${SEEDS_K:-2} $PY -m loop.run_chia local k ${TAG} results/experiment_${TAG}_k_playbook.json
  step summary_k $PY -m loop.summarize results/experiment_${TAG}_k.json --uniform 300
fi

echo "== $(date '+%F %T') chain done" | tee -a results/${TAG}_chain.log
if [ "${NO_SHUTDOWN:-0}" != "1" ]; then
  echo "== shutting down" | tee -a results/${TAG}_chain.log
  sudo shutdown -h now
fi

#!/bin/bash
# One unattended night on the VM: the pilot of the third plan review, in dependency order.
#   bash cluster/chain.sh <tag>
# Steps: smoke (every arm, one round, through CHIA) -> learn (pooled playbook + memory
# + gate G-language) -> cell gap with SEEDS seeds (default 2) -> consolidate the memory
# -> [RUN_REF=1] 300-design uniform reference on the GAP suite -> [RUN_2B=1] cell gap2b
# with the consolidated memory. Each step logs to results/<tag>_<step>.log; a failed
# step never aborts the chain; the VM shuts itself down at the end.
set -u
TAG="${1:-v2}"
cd ~/hackathon
export ANALYST_MODEL="${ANALYST_MODEL:-gemini-2.5-pro}"
PY=.venv/bin/python
T=traces
GAP3="$T/bfs.urand-36B.champsimtrace.xz $T/pr.urand-129B.champsimtrace.xz $T/bfs.kron-128B.champsimtrace.xz"
GAP2B="$T/bfs.road-99B.champsimtrace.xz $T/pr.web-16B.champsimtrace.xz $T/pr.road-28B.champsimtrace.xz"

step() {  # step <name> <command...>: run, log, record the outcome, never abort the chain
  local name="$1"; shift
  echo "== $(date '+%F %T') start $name" | tee -a results/${TAG}_chain.log
  "$@" > results/${TAG}_${name}.log 2>&1
  local code=$?
  echo "== $(date '+%F %T') end $name (exit $code)" | tee -a results/${TAG}_chain.log
  return $code
}

# 1. Smoke: every arm for one round on two workloads, through CHIA, with an uncached seed.
step smoke env FIRST_SEED=3 SEEDS=1 $PY -m loop.run_chia local smoke ${TAG} - random,bo,rules,llm_direct,handoff,memory

# 2. Playbook (pooled over A+B), memory (from the same tables), frozen pool, gate G-language.
step learn $PY -m loop.run learn ${TAG}
PLAYBOOK=results/experiment_${TAG}_playbook.json
MEMORY=results/experiment_${TAG}_memory.json

# 3. The pilot of the headline cell, every arm, from chip B's best design.
if [ -f $PLAYBOOK ]; then
  step gap env SEEDS=${SEEDS:-2} $PY -m loop.run_chia local gap ${TAG} $PLAYBOOK
  step summary $PY -m loop.summarize results/experiment_${TAG}_gap.json
  # 4. The memory that includes what cell gap taught.
  step consolidate $PY -m loop.memory consolidate $MEMORY results/experiment_${TAG}_memory_after_gap.json
fi

# 5. Reference and null curve for the GAP suite (opt-in: ~2.5 h, ~4 USD).
if [ "${RUN_REF:-0}" = "1" ]; then
  step reference env COLLECT_SOCS=C_server PARALLEL_PAIRS=3 SIM_THREADS=10 $PY -m loop.collect uniform 300 $GAP3
fi

# 6. Cell 2b with the consolidated memory (opt-in; needs its traces on disk).
if [ "${RUN_2B:-0}" = "1" ] && [ -f results/experiment_${TAG}_memory_after_gap.json ]; then
  MISSING=0
  for trace in $GAP2B; do [ -f "$trace" ] || { echo "MISSING $trace" | tee -a results/${TAG}_chain.log; MISSING=1; }; done
  if [ $MISSING -eq 0 ]; then
    step gap2b env SEEDS=${SEEDS_2B:-5} MEMORY_PATH=results/experiment_${TAG}_memory_after_gap.json \
        $PY -m loop.run_chia local gap2b ${TAG} $PLAYBOOK
    step summary2b $PY -m loop.summarize results/experiment_${TAG}_gap2b.json
  fi
fi

echo "== $(date '+%F %T') chain done; shutting down" | tee -a results/${TAG}_chain.log
sudo shutdown -h now

#!/bin/bash
# Both memories built from scratch by the declared procedure, then the VM off:
#   POWER_OFF=1 setsid nohup bash cluster/launch_membuild.sh > /dev/null 2>&1 < /dev/null & disown
# Budgets come from loop/memory_build.py (MB_SEARCH, MB_CONFIRM, MB_ANCHOR_KNOBS).
# Log: results/membuild.log; the smoke's own log is results/membuild_smoke.log.
cd ~/hackathon
export ANALYST_MODEL="${ANALYST_MODEL:-gemini-2.5-flash}"
export RAY_DEDUP_LOGS=0
export LOOP_DISPATCH=chia
SPEC="traces/605.mcf_s-665B.champsimtrace.xz traces/620.omnetpp_s-874B.champsimtrace.xz traces/619.lbm_s-2676B.champsimtrace.xz"
GAP="traces/bfs.urand-36B.champsimtrace.xz traces/pr.urand-129B.champsimtrace.xz traces/bfs.kron-128B.champsimtrace.xz"
LOG=results/membuild.log
echo $$ > results/membuild.pid

# 1. the gate: the whole procedure at toy budgets on two workloads
MB_SEARCH=4 MB_CONFIRM=2 MB_ANCHOR_KNOBS=2 .venv/bin/python -m loop.memory_build bo results/memory_smoke.json \
  traces/605.mcf_s-665B.champsimtrace.xz -- traces/619.lbm_s-2676B.champsimtrace.xz > results/membuild_smoke.log 2>&1
if ! grep -q "designs requested" results/membuild_smoke.log; then
  echo "SMOKE_FAILED $(date)" >> $LOG
  if [ "${POWER_OFF:-0}" = "1" ]; then sudo shutdown -h now; fi
  exit 1
fi
echo "SMOKE_OK $(date)" >> $LOG

# 2. from scratch: the memory workloads' old tables move out of the way (the test
#    suites' tables stay, so nothing already published can move).
mkdir -p results_archive
for trace in 605.mcf_s-665B 620.omnetpp_s-874B 619.lbm_s-2676B bfs.urand-36B pr.urand-129B bfs.kron-128B; do
  mv "results/table_${trace}.json" results_archive/ 2>/dev/null
  rm -f "results/table_${trace}.json.lock"
done

# 3. the two memories: identical anchor and confirm stages, different searcher.
#    They share the simulation cache; each memory holds only its own designs.
.venv/bin/python -m loop.memory_build bo  results/memory_bo.json  $SPEC -- $GAP >> $LOG 2>&1
echo "BO_DONE $(date)" >> $LOG
.venv/bin/python -m loop.memory_build llm results/memory_llm.json $SPEC -- $GAP >> $LOG 2>&1
echo "MEMBUILD_DONE $(date)" >> $LOG

if [ "${POWER_OFF:-0}" = "1" ]; then
  sleep 600            # time to fetch the results before the machine goes down
  sudo shutdown -h now
fi

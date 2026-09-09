#!/bin/bash
# One unattended night on the VM: the proof-of-concept chain, in dependency order.
#   bash cluster/chain.sh <tag>
# Each step logs to results/<tag>_<step>.log; an independent step runs even if an
# earlier one failed; the VM shuts itself down at the end. Assumes ~/hackathon with
# .venv, champsim trees, traces/ and results/ in place (cluster/README.md).
set -u
TAG="${1:-v1}"
cd ~/hackathon
export ANALYST_MODEL="${ANALYST_MODEL:-gemini-2.5-pro}"
PY=.venv/bin/python
T=traces
SPEC3="$T/605.mcf_s-665B.champsimtrace.xz $T/620.omnetpp_s-874B.champsimtrace.xz $T/619.lbm_s-2676B.champsimtrace.xz"
TRAIN5="$SPEC3 $T/649.fotonik3d_s-10881B.champsimtrace.xz $T/627.cam4_s-490B.champsimtrace.xz"
GAP3="$T/bfs.urand-36B.champsimtrace.xz $T/pr.urand-129B.champsimtrace.xz $T/bfs.kron-128B.champsimtrace.xz"

step() {  # step <name> <command...>: run, log, record the outcome, never abort the chain
  local name="$1"; shift
  echo "== $(date '+%F %T') start $name" | tee -a results/${TAG}_chain.log
  "$@" > results/${TAG}_${name}.log 2>&1
  local code=$?
  echo "== $(date '+%F %T') end $name (exit $code)" | tee -a results/${TAG}_chain.log
  return $code
}

# 0. Every trace the chain needs must exist, or the dependent steps are skipped.
MISSING=0
for trace in $TRAIN5 $GAP3; do
  if [ ! -f "$trace" ]; then echo "MISSING $trace" | tee -a results/${TAG}_chain.log; MISSING=1; fi
done

# 1. Smoke through CHIA with an uncached seed: real builds and simulations as CHIA tasks.
step smoke env FIRST_SEED=3 SEEDS=1 $PY -m loop.run_chia local smoke ${TAG} - random,bo

if [ $MISSING -eq 0 ]; then
  # 2. The learn set on A and B: baseline, every one-knob design, key pairs, the
  #    cached designs, on the five training workloads (only missing rows simulate).
  step learnset env COLLECT_SOCS=A_mobile,B_midrange PARALLEL_PAIRS=10 SIM_THREADS=3 \
      $PY -m loop.collect structured 130 $TRAIN5
  # 3. Distill the playbook from those tables, freeze the pool, gate G-language.
  step learn $PY -m loop.run learn ${TAG}
  # 4. First look at the headline cell: every arm, two seeds, through CHIA.
  if [ -f results/experiment_${TAG}_playbook.json ]; then
    step gap2 env SEEDS=2 $PY -m loop.run_chia local gap ${TAG} results/experiment_${TAG}_playbook.json
    step summary $PY -m loop.summarize results/experiment_${TAG}_gap.json
  fi
fi

# 5. Fidelity check: the best cached designs at 50M/50M, own tables (independent of the above).
step fidelity_spec env COLLECT_SOCS=C_server LOOP_WARMUP=50000000 LOOP_SIM=50000000 PARALLEL_PAIRS=3 SIM_THREADS=10 \
    $PY -m loop.collect top 20 $SPEC3
if [ $MISSING -eq 0 ]; then
  step fidelity_gap env COLLECT_SOCS=C_server LOOP_WARMUP=50000000 LOOP_SIM=50000000 PARALLEL_PAIRS=3 SIM_THREADS=10 \
      $PY -m loop.collect top 10 $GAP3
  # 6. Held-out reference: ~300 designs on the three GAP workloads. Opt-in
  #    (RUN_HELDOUT=1): the ruler for the full gap cell, bought once the playbook
  #    has shown something.
  if [ "${RUN_HELDOUT:-0}" = "1" ]; then
    step heldout_ref env COLLECT_SOCS=C_server PARALLEL_PAIRS=3 SIM_THREADS=10 \
        $PY -m loop.collect structured 300 $GAP3
  fi
fi

echo "== $(date '+%F %T') chain done; shutting down" | tee -a results/${TAG}_chain.log
sudo shutdown -h now

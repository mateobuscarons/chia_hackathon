#!/bin/sh
# Overnight pipeline, Sep 2-3 2026. Everything logs to results/overnight.log.
cd "$(dirname "$0")"
PY=.venv/bin/python
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "1. waiting for the sanitizer validation run to finish"
while pgrep -f "champsim_asan/bin" > /dev/null; do sleep 20; done
if grep -q "ERROR: AddressSanitizer" results/asan_run2.txt; then
  log "   sanitizer STILL reports an error - see results/asan_run2.txt (continuing anyway)"
else
  log "   sanitizer run clean: $(grep -c Heartbeat results/asan_run2.txt) heartbeats"
fi

log "2. stopping the current sweep queue"
pkill -f run_sweeps.sh; pkill -f "loop.sweep"; sleep 2; pkill -f "champsim/bin/[ABC]_"; sleep 2

log "3. dropping every spp_dev result and every binary so SPP is recomputed with the patch"
$PY - <<'PYEOF'
import glob, json
for path in glob.glob("results/sweep_*.json"):
    table = json.load(open(path))
    before = len(table)
    for name in list(table.keys()):
        if "pf-spp_dev" in name or table[name]["metrics"] is None:
            del table[name]
    json.dump(table, open(path, "w"), indent=2)
    print("   ", path, before, "->", len(table))
PYEOF
rm -f champsim/bin/A_mobile_* champsim/bin/B_midrange_* champsim/bin/C_server_*

log "4. sweeps, pass 1"
./run_sweeps.sh
log "5. sweeps, pass 2 (retries anything that failed)"
./run_sweeps.sh
$PY - <<'PYEOF'
import glob, json
for path in sorted(glob.glob("results/sweep_*.json")):
    table = json.load(open(path))
    failed = [n for n in table if table[n]["metrics"] is None]
    print("   ", path, len(table), "configs,", len(failed), "failed")
PYEOF

log "6. headline experiment: learn on A+B (3 traces), test on unseen C (3 traces), 3 seeds"
$PY -c "
from loop import experiment
traces = ['traces/605.mcf_s-665B.champsimtrace.xz', 'traces/619.lbm_s-2676B.champsimtrace.xz', 'traces/620.omnetpp_s-874B.champsimtrace.xz']
experiment.run_experiment(train_socs=['A_mobile', 'B_midrange'], test_soc='C_server', traces=traces,
                          rounds=10, per_round=2, seeds=3, output_path='results/experiment_crosssoc_C.json')
"
log "7. figures and summary"
$PY -m loop.plots results/experiment_crosssoc_C.json
$PY -m loop.summarize results/experiment_crosssoc_C.json > results/summary_crosssoc_C.txt
cat results/summary_crosssoc_C.txt
$PY -c "import json; print('LLM total USD: %.3f' % json.load(open('loop/llm_usage.json'))['total_cost_usd'])"
git add results/*.json results/*.png results/*.txt 2>/dev/null; git commit -qm "Overnight: patched SPP sweeps, cross-SoC experiment on C" && log "committed"
log "OVERNIGHT DONE"

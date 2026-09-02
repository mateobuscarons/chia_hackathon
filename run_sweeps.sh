#!/bin/sh
# Runs every remaining dense sweep, one after another (builds share champsim/).
# Waits for any sweep already running to finish first.
cd "$(dirname "$0")"
while pgrep -f "loop.sweep" > /dev/null; do sleep 30; done
for soc in B_midrange A_mobile C_server; do
  for trace in 605.mcf_s-665B 619.lbm_s-2676B 620.omnetpp_s-874B; do
    echo "=== $soc $trace $(date) ==="
    .venv/bin/python -m loop.sweep $soc traces/$trace.champsimtrace.xz
  done
done
echo "ALL SWEEPS DONE $(date)"

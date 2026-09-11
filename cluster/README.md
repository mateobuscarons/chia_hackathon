# Running on GCP

One VM with Ray in-process (`LOOP_DISPATCH=chia python -m loop.run`). Project `project-c23a6080-f5d0-4871-9cb`,
VM `champsim-1`, c2d-standard-32 (~1.5 USD/h), europe-west4-a. The project-wide cap is
32 vCPUs (`CPUS-ALL-REGIONS`), so this is the biggest VM; a quota raise is the only way
to shorten a round. No spot quota. The VM's service account has `roles/aiplatform.user`,
so the loop calls Gemini through Vertex with no key. ChampSim, 8 build trees, the traces
and `.venv` live on its disk; a stopped VM bills only the disk.

```bash
P=project-c23a6080-f5d0-4871-9cb; Z=europe-west4-a; VM=champsim-1
gcloud compute instances start $VM --project $P --zone $Z
# code and data: the loop, the launcher, the tables and the cell's memory (traces are already there)
gcloud compute scp --project $P --zone $Z --recurse loop cluster $VM:~/hackathon/
gcloud compute scp --project $P --zone $Z results/table_*.json results/memory_*.json results/profile_*.json $VM:~/hackathon/results/
# the gate: every arm, one round, flash model
gcloud compute ssh $VM --project $P --zone $Z --command 'cd ~/hackathon && SEEDS=1 LOOP_DISPATCH=chia .venv/bin/python -m loop.run smoke s1 2>&1 | tail -20'
# the cell, detached (survives the Mac sleeping)
gcloud compute ssh $VM --project $P --zone $Z --command \
  'cd ~/hackathon && SEEDS=2 setsid nohup bash cluster/launch_cell.sh dc d1 > /dev/null 2>&1 < /dev/null & disown'
# watch, fetch, merge, stop
gcloud compute ssh $VM --project $P --zone $Z --command 'cd ~/hackathon && .venv/bin/python -m loop.summarize progress results/dc_d1.log'
gcloud compute scp --project $P --zone $Z --recurse $VM:~/hackathon/results ./results_vm && python -m loop.workloads merge results_vm
gcloud compute instances stop $VM --project $P --zone $Z
```

Stop a running cell by its process group (the launcher's pid is in `results/<cell>_<tag>.pid`):
`kill -- -$(cat results/<cell>_<tag>.pid)`. Never `pkill -f` a broad pattern.

Parallelism: `PARALLEL_RUNS` runs at once (default 6), each simulating `per_round x suite size`
designs through `SIM_THREADS` threads per workload; builds run across the `champsim_N` tree
copies. Rounds are synchronous, so more concurrent runs is the only utilisation lever.

A fresh VM: `cluster/vm_bootstrap.sh` (ChampSim pinned to the commit the tables were made with,
the SPP patch from `upstream/`, build trees, the Python env, the SPEC traces). GAP traces come
from Zenodo with `loop.workloads fetch_gap`, datacenter traces as 100 MB prefixes with
`loop.workloads fetch <url> <out> 100`.

## CHIA cluster (`gcp.yaml`, kept for the integration story)

`local.yaml` is this Mac; `gcp.yaml` is a Mac head plus GCP workers on `ghcr.io/ucb-bar/chia-champsim`
over Tailscale (`chia up cluster/gcp.yaml`, then `LOOP_DISPATCH=chia CHIA_ADDRESS=auto python -m loop.run <cell> <tag>`, then
`chia down`). The workers have no traces: mount a bucket, or use `gs://` URIs once
`upstream/0001-champsim-gs-trace-resolver.patch` is applied.

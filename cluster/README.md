# Running on GCP

One VM. Project `project-c23a6080-f5d0-4871-9cb`, VM `champsim-1`, c2d-standard-32
(~1.5 USD/h), europe-west4-a. The project-wide cap is 32 vCPUs (`CPUS-ALL-REGIONS`), so this
is the biggest VM; a quota raise is the only way to shorten a round. No spot quota. The VM's
service account has `roles/aiplatform.user`, so the loop calls Gemini through Vertex with no
key. ChampSim, 8 build trees, the traces and `.venv` live on its disk; a stopped VM bills only
the disk.

```bash
P=project-c23a6080-f5d0-4871-9cb; Z=europe-west4-a; VM=champsim-1
gcloud compute instances start $VM --project $P --zone $Z
# code and data: the loop, the launcher, the tables and the memory (traces are already there)
gcloud compute scp --project $P --zone $Z --recurse loop cluster $VM:~/loop/
gcloud compute scp --project $P --zone $Z --recurse results/memory.json results/tables $VM:~/loop/results/
# the gate: every arm, one round
gcloud compute ssh $VM --project $P --zone $Z --command 'cd ~/loop && SEEDS=1 .venv/bin/python -m loop.search smoke s1 2>&1 | tail -20'
# the cell, detached (survives the Mac sleeping)
gcloud compute ssh $VM --project $P --zone $Z --command \
  'cd ~/loop && SEEDS=5 BUDGET=16 setsid nohup bash cluster/launch_cell.sh dc2 h1 > /dev/null 2>&1 < /dev/null & disown'
# watch, fetch, merge, stop
gcloud compute ssh $VM --project $P --zone $Z --command 'cd ~/loop && grep "ipc=" results/dc2_h1.log | tail -20'
gcloud compute scp --project $P --zone $Z --recurse $VM:~/loop/results ./results_vm && python -m loop.workloads merge results_vm
gcloud compute instances stop $VM --project $P --zone $Z
```

Stop a running cell by its process group (the launcher's pid is in `results/<cell>_<tag>.pid`):
`kill -- -$(cat results/<cell>_<tag>.pid)`. Never `pkill -f` a broad pattern.

Parallelism: `PARALLEL_RUNS` runs at once (default 6), each simulating its round's designs
through `SIM_THREADS` threads per workload; builds run across the `champsim_N` tree copies.
Rounds are synchronous, so more concurrent runs is the only utilisation lever.

A fresh VM: `cluster/vm_bootstrap.sh` (ChampSim pinned to the commit the tables were made with,
the SPP patch from `upstream/`, build trees, the Python env, the SPEC traces). GAP traces come
from Zenodo with `loop.workloads fetch_gap`, datacenter traces as 100 MB prefixes with
`loop.workloads fetch <url> <out> 100`.

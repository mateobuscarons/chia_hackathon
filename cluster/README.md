# Running on GCP

Two ways to run: one big VM with our own parallel runner (fast to set up, used
for the hard tier), or a CHIA cluster (`gcp.yaml`, Ray + Tailscale; kept for the
CHIA-integration story, see the end of this file).

## One VM for the hard tier (Tier C)

Project `project-c23a6080-f5d0-4871-9cb`. No spot quota (PREEMPTIBLE_CPUS = 0), so
the VM is on-demand: c2d-standard-56 (~$2.5/h) or n2-standard-64 (~$3.1/h).
europe-west1-b had no c2d stock on Sep 4; try europe-west4-a, then us-central1-a.

Auth on the Mac: `gcloud auth login` (or reuse ADC with
`--access-token-file=<(gcloud auth application-default print-access-token)`).

```bash
P=project-c23a6080-f5d0-4871-9cb; Z=europe-west4-a; VM=champsim-1
# 1. create (retry another zone on "stockout")
gcloud compute instances create $VM --project $P --zone $Z \
  --machine-type c2d-standard-56 --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 200GB --boot-disk-type pd-balanced --scopes cloud-platform
# 2. sync the repo (code only; tables are the result cache and are worth carrying)
gcloud compute scp --project $P --zone $Z --recurse \
  loop cluster upstream $VM:~/hackathon/
# 3. bootstrap: ChampSim + 12 build trees + traces (~30-45 min)
gcloud compute ssh $VM --project $P --zone $Z --command 'cd ~/hackathon && mkdir -p results && bash cluster/vm_bootstrap.sh'
# 4. run (detached; the VM keeps running if the Mac sleeps)
gcloud compute ssh $VM --project $P --zone $Z --command \
  'cd ~/hackathon && PARALLEL_RUNS=6 SIM_THREADS=8 CHAMPSIM_BUILD_SHARE=6 nohup .venv/bin/python -m loop.run C tierC > results/tierC.log 2>&1 &'
# 5. watch / fetch
gcloud compute ssh $VM --project $P --zone $Z --command 'cd ~/hackathon && .venv/bin/python -m loop.early results/experiment_tierC_C.json results/tierC.log'
gcloud compute scp --project $P --zone $Z --recurse $VM:~/hackathon/results ./results_vm
# 6. STOP or DELETE when done (a stopped VM only bills its disk)
gcloud compute instances delete $VM --project $P --zone $Z --quiet
```

Parallelism knobs (env): `PARALLEL_RUNS` arm runs at once, each simulating
`per_round x suite-size` designs through `SIM_THREADS` threads per workload;
builds run in parallel across the `champsim_N` tree copies (`TREES` in the
bootstrap, capped by `CHAMPSIM_TREES`), each `make -j(nproc/CHAMPSIM_BUILD_SHARE)`.

Cost/time estimate for the PoC (4 workloads, 5 arms, 3 seeds, 24 designs):
~1,400 simulations + ~350 builds; ~4 h on 56 vCPUs including bootstrap;
~$12 compute + ~$2 LLM.

## CHIA cluster (`gcp.yaml`)

- `local.yaml` — this Mac only. `python -m loop.run_chia local ...` does the same
  without `chia up` (starts Ray in-process with the same resources).
- `gcp.yaml` — Mac head + GCP VMs running `ghcr.io/ucb-bar/chia-champsim`.

Bring-up (one-time user steps):

1. `uv pip install -p .venv/bin/python google-cloud-compute`
2. `gcloud auth application-default login` and
   `gcloud services enable compute.googleapis.com --project project-c23a6080-f5d0-4871-9cb`
3. Tailscale account (free) -> Settings -> Keys -> generate an auth key
   (reusable + ephemeral). `export TS_AUTHKEY=tskey-...`
4. `ssh-add ~/.ssh/id_ed25519`; `export GCP_PRIVATE_KEY_PATH=~/.ssh/id_ed25519`
5. `export HEAD_IP=$(hostname) GCP_PROJECT=project-c23a6080-f5d0-4871-9cb`
6. `.venv/bin/chia up cluster/gcp.yaml`, then
   `python -m loop.run_chia auto traces/...` and `chia down cluster/gcp.yaml`.

Traces on the VMs: the image has none. Either mount a bucket or use `gs://`
trace URIs once `upstream/0001-champsim-gs-trace-resolver.patch` is applied
(the DPC-4 traces CHIA's own case study uses are in `gs://dpc4-all-traces`).

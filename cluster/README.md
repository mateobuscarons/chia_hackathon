# Running on GCP

Two ways to run: one big VM with Ray in-process (`loop.run_chia local`, fast to
set up), or a CHIA cluster (`gcp.yaml`, Ray + Tailscale; kept for the
CHIA-integration story, see the end of this file).

## One VM

Project `project-c23a6080-f5d0-4871-9cb`. No spot quota (PREEMPTIBLE_CPUS = 0), so
the VM is on-demand. The project-wide cap is **32 vCPUs** (CPUS-ALL-REGIONS, found
Sep 7; the per-region C2D quota of 100 is not the binding one), so the biggest VM is
c2d-standard-32 (~$1.5/h). europe-west4-a had stock on Sep 7.

The analyst calls Gemini through Vertex AI with the VM's default service account,
which needs `roles/aiplatform.user` once per project (granted Sep 7):
`gcloud projects add-iam-policy-binding $P --member=serviceAccount:<project-number>-compute@developer.gserviceaccount.com --role=roles/aiplatform.user`

Auth on the Mac: `gcloud auth login` (or reuse ADC with
`--access-token-file=<(gcloud auth application-default print-access-token)`).

```bash
P=project-c23a6080-f5d0-4871-9cb; Z=europe-west4-a; VM=champsim-1
# 1. create (retry another zone on "stockout")
gcloud compute instances create $VM --project $P --zone $Z \
  --machine-type c2d-standard-32 --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 200GB --boot-disk-type pd-balanced --scopes cloud-platform
# 2. sync the repo (code only; tables are the result cache and are worth carrying)
gcloud compute scp --project $P --zone $Z --recurse \
  loop cluster upstream $VM:~/hackathon/
# 3. bootstrap: ChampSim + 12 build trees + traces (~30-45 min)
gcloud compute ssh $VM --project $P --zone $Z --command 'cd ~/hackathon && mkdir -p results && TREES=8 nohup bash cluster/vm_bootstrap.sh > bootstrap.log 2>&1 &'
# 4. run (detached; the VM keeps running if the Mac sleeps)
gcloud compute ssh $VM --project $P --zone $Z --command \
  'cd ~/hackathon && SEEDS=1 setsid nohup bash cluster/launch_cell.sh w1 v1 bo,llm_direct,memory > /dev/null 2>&1 < /dev/null & disown'
# 5. watch / fetch
gcloud compute ssh $VM --project $P --zone $Z --command 'cd ~/hackathon && .venv/bin/python -m loop.summarize results/experiment_v1_w1.json --uniform 300'
gcloud compute scp --project $P --zone $Z --recurse $VM:~/hackathon/results ./results_vm
# 6. STOP or DELETE when done (a stopped VM only bills its disk)
gcloud compute instances delete $VM --project $P --zone $Z --quiet
```

Parallelism knobs (env): `PARALLEL_RUNS` arm runs at once, each simulating
`per_round x suite-size` designs through `SIM_THREADS` threads per workload;
builds run in parallel across the `champsim_N` tree copies (`TREES` in the
bootstrap, capped by `CHAMPSIM_TREES`), each `make -j(nproc/CHAMPSIM_BUILD_SHARE)`.

Cost/time estimate for the PoC (4 workloads, 5 arms, 3 seeds, 24 designs):
~1,400 simulations + ~350 builds; ~4 h on 56 vCPUs or ~6-7 h on 32 vCPUs
including bootstrap; ~$10-12 compute + ~$2 LLM either way.

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
   `python -m loop.run_chia auto w1 v1 results/experiment_v1_playbook.json` and `chia down cluster/gcp.yaml`.

Traces on the VMs: the image has none. Either mount a bucket or use `gs://`
trace URIs once `upstream/0001-champsim-gs-trace-resolver.patch` is applied
(the DPC-4 traces CHIA's own case study uses are in `gs://dpc4-all-traces`).

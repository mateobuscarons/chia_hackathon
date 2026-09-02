# Clusters

- `local.yaml` — this Mac only. `python -m loop.run_chia local ...` does the same
  without `chia up` (starts Ray in-process with the same resources).
- `gcp.yaml` — Mac head + GCP spot VMs running `ghcr.io/ucb-bar/chia-champsim`.

## Bringing up `gcp.yaml` (one-time user steps)

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

Cost: two n2-standard-8 spot VMs are roughly $0.20/h together; a full
3-SoC x 3-trace dense sweep (729 sims at ~2 min on 12 slots) is about 2 h.

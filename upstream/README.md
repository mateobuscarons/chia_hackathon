# Upstream contributions to CHIA (planned PRs)

The judges are the CHIA team; these are the pieces of our loop that belong in
the framework itself. `chia/` is a shallow clone, so PRs go through a fork.

1. **`0001-champsim-gs-trace-resolver.patch`** — `chia.simulators.champsim._resolve_trace`
   raises `NotImplementedError` for `gs://` traces; GCP clusters need them.
   Mirrors the existing `s3://` branch with `google-cloud-storage`. Open early:
   merge time is not ours.
2. **Config-space ChampSim build node** — `ChampSimNode.build_champsim` only
   accepts a prefetcher module; `loop/chia_nodes.py::build_from_config` builds
   from an arbitrary config JSON (cache sizes, policies, prefetchers, core
   parameters). Candidate for `chia.simulators.champsim`.
3. **Bet ledger / forecaster / disagreement-selector blocks** —
   `loop/playbook.py`, `loop/forecast.py`, `loop/loop.py` are simulator-agnostic:
   any `evaluate(config) -> metrics` fits. Candidate for a `chia.analysis`
   "calibration layer" any agentic CHIA loop can wrap around its simulator node.

## ChampSim (not CHIA)

4. **`0002-champsim-spp-dev-ghr-victim.patch`** — `prefetcher/spp_dev/spp_dev.cc`
   asserts "[GHR] Cannot find a replacement victim!" when every GHR entry has
   confidence 100: the victim search starts at `min_conf = 100`, so a saturated
   table yields no victim. Deterministic on lbm with a slow-memory core profile.
   Fix: start the search above any legal confidence. Found Sep 2 2026; no
   existing issue found upstream.

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

4. **`0002-champsim-spp-dev-ghr-victim.patch`** — two bugs in `prefetcher/spp_dev/spp_dev.cc`,
   both found Sep 2 2026 running SPP on a small-core profile (L2 MSHR 16, DDR-1600) with lbm:
   - **Heap-buffer-overflow in the lookahead loop.** `confidence_q`/`delta_q` are sized to the
     L2 MSHR count, but `read_pattern` appends up to `PT_WAY + 1` entries per lookahead step
     with no bounds check; a long confident chain overruns them (AddressSanitizer: READ of
     size 4 past a 64-byte region at `confidence_q[i]`, spp_dev.cc:78). Silent SIGTRAP on
     macOS, heap corruption elsewhere. Fix: stop the lookahead when the next step cannot fit.
   - **GHR victim search never finds a victim** when every entry has confidence 100
     (`min_conf` starts at 100): `assert(0)` "[GHR] Cannot find a replacement victim!".
     Fix: start the search above any legal confidence.
   No existing upstream issue found for either.

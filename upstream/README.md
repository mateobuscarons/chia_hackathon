# Patches for CHIA and ChampSim

Two patches against frameworks this repo only clones, so they go through a fork.

### `0001-champsim-gs-trace-resolver.patch` — CHIA

`chia/simulators/champsim.py::_resolve_trace` handles local paths and `s3://` URIs and raises
`NotImplementedError` for `gs://`, so a loop running on a GCP cluster cannot read its traces from
a bucket. The patch mirrors the existing `s3://` branch with `google-cloud-storage`, downloading
a remote trace once to the worker's temp directory and reusing it.

### `0002-champsim-spp-dev-ghr-victim.patch` — ChampSim

Two bugs in `prefetcher/spp_dev/spp_dev.cc`, both found running SPP on a small-core profile
(L2 MSHR 16, DDR-1600) with `lbm`. No existing upstream issue was found for either.

- **Heap-buffer-overflow in the lookahead loop.** `confidence_q` and `delta_q` are sized to the L2
  MSHR count, but `read_pattern` appends up to `PT_WAY + 1` entries per lookahead step with no
  bounds check, so a long confident chain overruns them (AddressSanitizer: READ of size 4 past a
  64-byte region at `confidence_q[i]`, `spp_dev.cc:78`). Silent SIGTRAP on macOS, heap corruption
  elsewhere. The fix stops the lookahead when the next step cannot fit.
- **The GHR victim search never finds a victim** when every entry has confidence 100, because
  `min_conf` starts at 100 and the comparison is strict, so `assert(0)` — "[GHR] Cannot find a
  replacement victim!" — fires. The fix starts the search above any legal confidence value.

Three further gaps were found in CHIA and are not yet patched: its ChampSim build node accepts
only a prefetcher module (so a configuration-space search cannot be expressed) and its run node
accepts a single trace — `loop/chia_nodes.py` carries the general versions; its Vertex Gemini
layer forwards no generation config, so temperature, response type and thinking budget never
reach the model; and `loop/memory.py` is a case-memory mechanism that no CHIA block provides.

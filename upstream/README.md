# What goes upstream

Fixes and findings only; the blocks stay in this repository (`council_loop/chia_blocks/`, in CHIA's
layout, with `install.sh` for the released package). Nothing has been opened yet; this is the work order.

| # | target | what | state |
|---|---|---|---|
| 1 | CHIA | `chia/models/vertex.py`: a `MALFORMED_FUNCTION_CALL` finish (the model attempts a tool the prompt mentions but none is declared) returns `success=True` with an empty result; treat it as a failure and surface the finish reason | found on CHIA's own CIRCT assess prompt (14 of 16 issues blank); worked around in our replica |
| 2 | CHIA | `chia/models/vertex.py`: a 429 raises at once ("never retry"), so a batch dies on the first rate limit; an opt-in backoff as the other backends have | worked around in our replica (`council_loop/circt_assess.py` retries) |
| 3 | CHIA | `chia/models/vertex.py`: the usage record adds `candidates_token_count` and never reads `thoughts_token_count`, so a thinking model's spend is undercounted (37% of our bill); and the generation config is fixed (no temperature, JSON mode or thinking budget). Two lines in `_run_generate_async` and a config argument | our `vertex_json.py` node does both, in the repository |
| 4 | CHIA | `chia/simulators/champsim.py`: `run_champsim` parses the first DRAM channel only (`dram_list[0]`) and no `miss_merge` counts; `patches/0002-run-champsim-raw-stats.patch` (7 lines) returns the raw record beside the typed stats | patch written and applied by `install.sh`; the loop runs on it |
| 5 | CHIA | `make -j$(nproc)` in both ChampSim build paths: `nproc` is GNU, a Mac has none, so `make -j` runs unbounded; `$(nproc 2>/dev/null \|\| sysctl -n hw.ncpu)` serves both | fixed in our build node; one line upstream |
| 6 | CHIA | `docs/user_guides/profiling.rst`: events are recorded fire-and-forget, so read the log after a blocking `get_events` on the collector; a profiler created before the collector stays disabled, so `reset_profiler()` after `start_collector()` | written in our docs page and used by the loop |
| 7 | CHIA | `0001-champsim-gs-trace-resolver.patch`: `_resolve_trace` raises `NotImplementedError` for `gs://` | patch written, to rebase |
| 8 | ChampSim | `0002-champsim-spp-dev-ghr-victim.patch`: a heap-buffer-overflow in SPP's lookahead loop and a GHR victim search that asserts when every entry has confidence 100 | patch written; a setup step here (without it `spp_dev` designs crash) |

Items 1 to 7 go to `ucb-bar/chia` through a fork, as issues with a patch each; item 8 to ChampSim.
AI assistance is disclosed in each, as CHIA's `CONTRIBUTING.md` asks.

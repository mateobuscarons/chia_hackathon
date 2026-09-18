# Blocks for CHIA, and fixes for ChampSim

Two things belong upstream rather than in this repo: **reusable blocks** for cache-hierarchy
design-space exploration, and **fixes** for gaps and bugs hit while building this loop. This file
is the plan for both: what each block is, where it lands in CHIA's tree, what it exposes, how it
is tested, and what this repo learned building its own version. The two patches are written;
the blocks are not.

**How CHIA is built, as read in the clone (`chia/`, gitignored).** A node is a Python function
under `@ChiaFunction(resources={...})`; called directly it runs in-process, called as
`fn.chia_remote(...)` it is scheduled by Ray onto a worker holding the resource, and `get(ref)`
collects it. Nodes for a tool are grouped in a class (`ChampSimNode`, `Gem5Node`) whose members
are `@staticmethod @ChiaFunction` functions re-bound per instance to a placement group. Results
are dataclasses and travel through the object store; a ChampSim binary travels as `bytes`. LLM
backends subclass `LLMCallBase` and expose `prompt(user_message, tools) -> QueryResult` as a
`ChiaFunction`. `cache`/`bypass` persist and replay one call's result by a `_chia_tag` inside a
loop; `chia_kv_store` is a detached in-memory actor for operator knobs. There is no design space,
no per-target measurement table and no memory between loops. Tests live in
`chia/<package>/tests/`, offline "Tier 0" tests on pure functions and env-gated live tests; public
functions carry docstrings; `docs/api/<package>.rst` is an `automodule` list. `CONTRIBUTING.md`
and `AGENTS.md` ask for tests with every contribution, no new dependencies without review, and
disclosure of AI assistance in the PR description.

| # | block or fix | lands in | state |
|---|---|---|---|
| 1 | configuration-space build, one trace per core | `chia/simulators/champsim.py` | planned; first |
| 2 | design space, measurement store, case memory, report tool, evaluation harness | `chia/dse/` (new package) | planned |
| 3 | generation config that reaches Gemini, thinking tokens, opt-in 429 backoff | `chia/models/vertex.py` | planned |
| 4 | `gs://` trace resolution | `chia/simulators/champsim.py` | patch written (`upstream/0001`), to rebase |
| 5 | two bugs in `prefetcher/spp_dev` | ChampSim upstream | patch written (`upstream/0002`), not opened |
| 6 | this loop in CHIA-example layout, running through 1, 2, 3 and 7 | this repo | planned; the gate is the smoke cell |
| 7 | CACTI in cache mode: associativity, block, tag array | `chia/vlsi/sram_cacti/cacti_runner.py` | planned; CACTI 7 built and run locally at 32 and 45 nm, ladders read (`CLAUDE.md`, The redesign) |

Decisions taken: the DSE blocks form a new package `chia/dse/`; item 1 is built first; the loop
example stays in this repo in the layout of `chia/examples/*` (loop file, `cluster.yaml`,
`README.md`) rather than as a pull request to `examples/`.

---

## 1. A configuration-space build, and a run that takes one trace per core

**The gap, verified in `chia/simulators/champsim.py`.**

```
build_champsim(champsim_root, prefetcher_src, module_name, *, cache_level="L2C", timeout_s=600, incremental=False)
run_champsim(binary, trace, *, warmup_instructions=5_000_000, simulation_instructions=25_000_000, timeout_s=300)
```

`build_champsim` writes `prefetcher_src` as a header and a two-key config
(`{"executable_name": "champsim", cache_level: {"prefetcher": module_name}}`), then runs
`make clean && python3 ./config.sh <cfg> && make -j$(nproc)`. It supports one kind of loop: an
agent writes a policy in C++ and CHIA compiles it. Cache sizes, associativity, replacement
policies, MSHR depths, the core and the memory are not reachable. `run_champsim` takes a single
trace; `_parse_champsim_json` reads `roi.cores[0]`, so a multi-core chip cannot be simulated or
read. Everything else the node needs is already there: `_run_logged` (process group, timeout),
the JSON parser with per-cache `CacheStats` and derived prefetch metrics, the content-hashed
binary transport, the placement-group binding. The worker image is
`ghcr.io/ucb-bar/chia-champsim` with a checkout at `/home/ray/champsim`.

**What lands.**

```
build_champsim_config(champsim_root, config, *, timeout_s=600) -> ChampSimBuildResult
    config: the complete ChampSim configuration as a dict, written as JSON and passed to
    config.sh unchanged. `executable_name` is taken from the dict, so the caller controls the
    binary's identity. Returns the existing result type; `module_name` carries executable_name.
run_champsim(binary, trace: str | list[str], ...) -> ChampSimRunResult
    one trace per core, in order (ChampSim requires exactly one per core). Per-core IPC is kept
    (`core_ipc: list[float]`), `ipc` is their geometric mean; cache counters of private caches
    (cpu0_L2C, cpu1_L2C, ...) are summed per level. A single string behaves as today.
```

Rendering the dict is the caller's job: what a knob is and where it lives in ChampSim's JSON is
the loop's vocabulary (`loop/simulate.py`: chip overrides, knob locations, latencies dropped so
ChampSim derives them from size). The node stays ignorant of it.

**What this repo learned.** A build holds the whole tree, so `champsim_build: 1` per checkout is
the scheduling contract; the node gets that for free from its resource. One configuration name
means one exact binary, so a binary is worth caching by name with no verification; a rebuild of
a configured tree is seconds, only the per-config object and the link. Fixed cache latencies
must be dropped from the config or capacity is free and the search finds nonsense.

**Tests** (`chia/simulators/tests/test_champsim_live.py`, Tier 0): the config dict written and
read back unchanged, `executable_name` honoured; a synthetic two-core stats JSON parsed into
per-core IPC, the geometric mean and summed private-cache counters; the single-trace path
unchanged. Tier 1 live test env-gated like the existing ones.
**Docs:** `docs/api/simulators.rst` lists `gem5` only; add the `champsim` automodule.

## 2. `chia/dse/`: the vocabulary of a design-space search

Five modules. `space`, `cases` and `harness` are pure Python with no Ray import, so a loop can
use them on a laptop and the same code runs on a cluster; `store` subclasses CHIA's `SQLiteNode`
and `report_tool` its `ChiaTool`. Each is a lift from this repo made target-agnostic: nothing in
them knows what a cache, a trace or a simulator is.

**`space.py`, from `loop/space.py` (128 lines).**

```
DesignSpace(knobs: dict[str, list], feasible: Callable[[dict], bool] | None, name: Callable | None)
    .contains(design)  .typed(design)  .sample(n, seed)  .neighbours(design)  .name(design)  .changed(a, b)
```

Learned: the name must carry the whole configuration, be filesystem-safe and free of `=` (which
`make` reads as an assignment), so a result table, a binary cache and a build record share one
key. Feasibility is part of the space, not a filter after sampling; here it is an area budget
coupling two levels.

**`store.py`, from `loop/suite.py`, on `chia.database.SQLiteNode`.**

```
MeasurementStore(SQLiteNode)            # one database file on the head, WAL, colocated members
    .get(soc, target, name)  .put(soc, target, name, design, metrics)  .rows(soc, target)
    .measured_on_all(soc, targets)  .merge(other_db)  .spawn_query_tool(name)   # inherited
```

Their node already gives colocation, WAL, `BEGIN IMMEDIATE` writes and a read-only SQL tool for
agents; this subclass adds the measurement schema and the composite reads, the way
`examples/gem5_align/alignment_db.py` does. A `metrics` of null records a design the simulator
could not measure, so it is never paid for twice. A row carrying an older `version` is a miss
and is re-measured. `measured_on_all` is the query the metric rests on: the best design
measured on every target of a suite is the denominator every share divides by. This is not
`chia.base.cache`: that replays one call by tag inside a loop; this is the experiment's dataset,
shared by parallel runs and read long after them. The JSON tables in `results/tables/`, one per
workload and fidelity, are the first version of the same store; `merge` imports them.

**`cases.py`, the memory between searches.**

```
CaseMemory(store)
    .write(soc, run, sheet, concern, move, outcome)   # after every measured design
    .nearest(features, concern, sources, k=5) -> list[case]   # k nearest among the (soc, run) sources given
```

A case is: its source (SoC descriptor, run id), the bottleneck sheet the diagnosis wrote, the
concern that moved, the move (knob, from, to), and the outcome (delta IPC, delta area, delta
energy). The feature vector is per-level hit ratios, prefetch coverage, miss latency, budget use
and core width class; retrieval is Euclidean on those, with no embeddings and no model. The block
stores every case with its source and reads only the sources the caller names, so which runs a
loop learns from is the loop's policy, not the block's, and every returned case carries its
source. This loop reads its own run and the SoCs run before it (`CLAUDE.md`, "The redesign").
The first version of this repo carried one design across searches instead (in git history):
measured, it bought a head start and nothing after it, and it cannot be refitted across SoCs.
Cases are what an architect remembers: a situation, a move, what happened.

**`report_tool.py`.** A `ChiaTool` (MCP server on a worker, `chia/base/tools/ChiaTool.py`) whose
functions are queries over the store and the cases: rows for a target, the ledger for an
assignment of a knob subset, the best design known, the nearest cases. A loop subclasses it with
its own views; this loop adds prefetch statistics by mechanism at a level and the SoC's latency
and energy ladder. Agents call it through the model backends' existing MCP loop; it is the CHIA
mechanism the first version of this loop never used.

**`harness.py`, from `loop/search.py` (`run_cell`, `score`, `ceiling`).**

```
run(problem, arms: dict[str, Callable], seeds, budget) -> report
score(report, best_known) -> designs-to-level table
ceiling(problem, designs, batch)
```

`problem` is the interface every arm searches: `stock`, `space`, `is_candidate`, `name_of`,
`evaluate`, `evaluate_many`, `objective`, `targets`. Budget counts designs, not rounds; a design
already in the store is read, not simulated, and still counts; a design the simulator cannot
measure is dropped and does not count. The ceiling is the same optimizer run long from the stock
point with no memory, and it must stay a mechanism separate from the arms: scoring against a
design one arm found bounds the metric at that arm, which happened once here and cost every arm
five points. `problem` carries the SoC, so one harness call runs the same arms on several SoCs. **Offline
mode:** `problem` backed by the store alone, `is_candidate` restricted to designs measured on
every target, simulation disabled; a run then needs no simulator and no trace, and any optimizer
can be compared against the arms here from the store.

**Tests** (`chia/dse/tests/`, all offline): sampling is deterministic under a seed and never
leaves the feasible set; names round-trip; concurrent `put` loses no row; an older version reads
as a miss; `merge` never downgrades; `nearest` returns the planted neighbour on synthetic cases;
the report tool's queries answer from a temporary store; `run` with a synthetic `evaluate` and
two arms yields a `score` table; offline mode refuses a design not in the store.
**Docs:** `docs/api/dse.rst`; a user guide `docs/user_guides/design_space_exploration.rst` with
the offline mode as the worked example.

## 3. A generation config that reaches Gemini, thinking tokens, opt-in backoff

**The gap, verified in `chia/models/vertex.py`.** `VertexGeminiLLM.__init__` takes `model`,
`system_message`, `timeout_seconds`, `retries`, `max_tokens=16000`, `max_tool_iterations`,
`client_kwargs`. The request config is built as `max_output_tokens`, `automatic_function_calling`
disabled, `system_instruction`, `tools`; temperature, response MIME type and thinking budget
never reach the model. `prompt()` re-raises `RateLimitError` immediately ("Never retry"),
retries a truncated answer once, and backs off on 5xx. `_last_metadata` counts input and output
tokens and turns; `thoughts_token_count` is not read. The module docstring says the backend is
experimental and not validated in production.

**Why it is not cosmetic, measured here.** On Gemini 2.5 Flash thinking tokens are billed and
counted as output and consume the same cap as the answer. An arm of this loop asked for eight
designs as JSON; the answer plus the model's thinking exceeded 16k, the JSON was truncated, the
layer raised, and a run that had simulated for an hour ended. Temperature stayed at the model's
default rather than the value set, so a run through that path cannot be reproduced.

**What lands.** `generation_config: dict | None = None` on `__init__`, merged into the request
config with the SDK's own field names (`temperature`, `top_p`, `seed`, `response_mime_type`,
`response_schema`, `thinking_config`), unchanged; `thoughts_token_count` accumulated into
`_last_metadata["thinking_tokens"]`; `rate_limit_retries: int = 0`, and when it is above zero a
429 waits until the `reset_time` the error already carries and retries that many times. Defaults
leave every current behaviour as it is.

**Tests** (`chia/models/tests/test_vertex.py`, the mocked-loop layer): the config reaches
`generate_content`; thinking tokens are counted; a 429 is raised at the default and retried when
opted in. `loop/analyst.py` carries the same fixes as a wrapper, which §6 drops for the block.

## 4. `gs://` trace resolution

**The gap.** `_resolve_trace` handles local paths and `s3://` and raises `NotImplementedError`
for `gs://`, so a loop on a GCP cluster cannot read traces from a bucket.

**What lands.** `upstream/0001-champsim-gs-trace-resolver.patch` mirrors the `s3://` branch with
`google-cloud-storage`, downloading once to the worker's temp directory under a key hash. To
rebase onto the current file. `google-cloud-storage` is not in `pyproject.toml`
(`google-cloud-compute` is), so the import stays lazy and the dependency goes in as an optional
extra, named in the pull request. **Test:** Tier 0 with the storage client monkeypatched.

## 5. Two bugs in ChampSim's `prefetcher/spp_dev`

Both found running SPP on a small-core profile (L2 MSHR 16, DDR-1600) with `lbm`.
`upstream/0002-champsim-spp-dev-ghr-victim.patch` fixes both, and applying it is a setup step
here: without it `spp_dev` designs crash, and `spp_dev` is in the search space.

**Heap-buffer-overflow in the lookahead loop.** `confidence_q` and `delta_q` are sized to the L2
MSHR count, but `read_pattern` appends up to `PT_WAY + 1` entries per lookahead step with no
bounds check; AddressSanitizer reports a READ of size 4 past a 64-byte region at
`confidence_q[i]`, `spp_dev.cc:78`. Silent SIGTRAP on macOS, heap corruption elsewhere. The fix
stops the lookahead when the next step cannot fit.

**The GHR victim search never finds a victim** when every entry has confidence 100: `min_conf`
starts at 100 and the comparison is strict, so `assert(0)` fires. The fix starts the search above
any legal confidence value.

Goes to ChampSim upstream as a pull request with the sanitizer report. CHIA's image builds its
own ChampSim ref, so the fix reaches CHIA through upstream.

## 6. This loop in CHIA-example layout

The layout of `chia/examples/*`: one loop file, `cluster.yaml`, `README.md`, the constants and
prompts beside them. The loop is the redesign in `CLAUDE.md`: SoC profile, CACTI ladder,
rendered config, build, run per trace, store, analyst, report tool, specialists in team
rounds with sketches, case memory. What changes in `loop/`:

- `loop/socs.py` returns from git history as the SoC profiles (core, frequency, memory, core
  count, process node, area budget).
- `loop/simulate.py` keeps the chip and the config rendering (that is the vocabulary), reads each
  level's latency from the CACTI ladder (§7) instead of ChampSim's derived formula, and hands the
  dict to `ChampSimNode.build_champsim_config`; the run goes through `ChampSimNode.run_champsim`
  with one trace per core. The tree pool and the binary cache become the node's resource and
  CHIA's cache.
- `loop/suite.py` becomes the `MeasurementStore` plus the `problem` from `chia.dse.harness`.
- `loop/analyst.py` becomes `VertexGeminiLLM(generation_config=...)` with `rate_limit_retries`
  set and the report tool in `tools`; the JSON contract stays here.
- `loop/space.py` becomes the `chia.dse` space with this loop's knobs and budget.
- `loop/council.py` stays the loop's own code (the analyst, the team round, the sketches) and
  gains the tool and the cases. It is the contribution, not a block.
- `loop/search.py` keeps the two arms, `council` and `bo`, and calls `chia.dse.harness.run`
  and `.score` per SoC.

One switch chooses local calls or `chia_remote`, so the loop runs on the Mac without Ray and on
the cluster with it. `cluster.yaml` describes one machine: a `champsim_build` worker with eight
CPUs, N `champsim` workers on `ghcr.io/ucb-bar/chia-champsim` with the pinned ChampSim commit
and the spp_dev patch applied in `worker_setup_commands`, a `cacti` worker on
`ghcr.io/ucb-bar/chia-cacti`, and a `vertex_creds` worker for the model. The gate is the smoke
cell through both paths, with the Gemini tool loop exercised and the precomputed-views fallback
ready. CACTI latency changes the chip, so every table is rebuilt for the grid; the tables in
`results/tables/` stay as the dataset.

## 7. CACTI in cache mode

**The gap, verified in `chia/vlsi/sram_cacti/cacti_runner.py`.** `run_cacti(spec: SRAMSpec,
technology_um=0.130, cacti_path)` is a `ChiaFunction` on the `cacti` resource that writes a
CACTI 7 config for an SRAM macro (`-cache type "ram"`, `-associativity 1`, block = the word
width, cell type itrs-hp, 360 K) and parses access time, cycle time, read energy, leakage and
height x width into `CACTIResult`. It serves the Chipyard SRAM characterization flow; a cache is
not expressible: no associativity, no 64-byte block, no tag array.

**What lands.**

```
CacheSpec(name, size_bytes, block_bytes, associativity, rw_ports=1, technology_um)
run_cacti_cache(spec, cacti_path) -> CACTIResult
```

The same runner with `-cache type "cache"`, `-associativity`, `-block size (bytes) 64`,
`-tag size (b) "default"`, and the cache level set by the caller. Nothing else moves; the parse
and the result type are shared. CACTI 7 documents 90, 65, 45 and 32 nm; the node's default of
130 nm serves Sky130 and is kept.

**What this loop does with it.** For an SoC at a node and a frequency, every (sets, ways) in the
space becomes one CACTI run, cached by (size, ways, block, node): a ladder of latency in cycles,
read energy and leakage per level, written into ChampSim's config and shown to the geometry
specialist in place of the formula ladder. Area in mm² is reported beside the KB budget.

**Tests** (`chia/vlsi/sram_cacti/test/`): the cache-mode config renders the associativity, block
and tag lines; the parser reads a recorded CACTI cache output. Live test gated on a `cacti`
binary. **Docs:** the `sram_cacti` page gains the cache-mode entry.

# Blocks for CHIA, and fixes for ChampSim

Two things belong upstream rather than in this repo: **reusable blocks** for cache-hierarchy design-space
exploration, and **fixes** for gaps and bugs hit while building this loop.

The blocks are the goal. A cache-DSE loop needs the same five pieces whatever it is searching for,
and this project wrote all five as one-offs. Writing them properly means writing them so that a
different cache project — a different simulator, a different space, a different objective — can use
them without editing them. Each section below states the gap upstream, what this repo learned
building its own version, and what the block has to look like to be general.

**State.** Nothing here is submitted. Two patches are written (§6, §7). The block code exists in this
repo in loop-specific form; the CHIA wiring that once carried it was deleted when the loop was pruned,
deliberately, so the blocks are rebuilt rather than salvaged.

| # | block or fix | target | state |
|---|---|---|---|
| 1 | a design space as a first-class object | CHIA | loop-specific, in `loop/space.py` |
| 2 | a ChampSim node that takes a configuration and several traces | CHIA | written, then removed; git history |
| 3 | a measurement store keyed by configuration | CHIA | loop-specific, in `loop/suite.py` |
| 4 | a memory of earlier searches | CHIA | loop-specific, in `loop/memory.py` |
| 5 | a generation config that reaches the model, and backoff | CHIA | described, not written |
| 6 | `gs://` trace resolution | CHIA | patch written, not submitted |
| 7 | two bugs in `prefetcher/spp_dev` | ChampSim | patch written, not submitted |

---

## 1. A design space as a first-class object

**The gap.** CHIA has no representation of "the set of configurations a loop may search". Every loop
invents one, and with it its own notion of which configurations are legal, how to sample them, and
how to name one so results can be keyed by it.

**What this repo learned.** `loop/space.py` is 128 lines and holds the whole of it: named knobs with
allowed values, a feasibility predicate, uniform sampling by rejection, and a stable name. Two details
turned out to matter more than expected.

- **The name must carry the whole configuration.** `config_name` emits the chip plus a sorted
  knob-value listing, so a result table, a compiled-binary cache and a build record can all be keyed
  by the same string and nothing has to store the knobs twice. It also has to be filesystem-safe and
  free of `=`, which `make` reads as an assignment.
- **The feasibility constraint is not a filter, it is part of the space.** Here it is an area budget
  coupling L2 and LLC capacity. A loop that samples first and filters second wastes most of its
  samples; one that treats feasibility as part of the definition does not.

**What it must look like to be general.** Knob values, a `feasible(config) -> bool` callable and a
naming function, injected. No assumption that a knob is a cache parameter, and no import of anything
that knows about a simulator.

## 2. A ChampSim node that takes a configuration, and a run node that takes several traces

**The gap.** `chia/chia/simulators/champsim.py` exposes:

```
build_champsim(champsim_root, prefetcher_src, module_name, *, cache_level="L2C")
run_champsim(binary, trace, *, warmup_instructions, simulation_instructions)
```

The build node takes a **prefetcher source module** and nothing else, so the block supports loops in
which an agent writes a policy in C++ and CHIA compiles it. A loop that searches the **configuration
space** cannot be expressed at all: cache sizes, associativity, replacement policies and MSHR depths
are not reachable. `run_champsim` takes a **single trace**, so a multi-core chip cannot be simulated.

**What this repo learned.** The working versions were `build_from_config(config, champsim_root)` and
`simulate(binary, trace_paths, ...)`, returning the binary as bytes through the object store and a flat
metrics dict. Three practical points survived from that:

- **Resources are the scheduling contract.** `champsim_build: 1` per source checkout, because a build
  holds the whole tree and two builds in one tree collide; `champsim: 1` per simulation core. Getting
  this wrong is the difference between a machine that is full and one that is thrashing.
- **A build is worth caching by name, not by content hash.** One configuration name always means one
  exact binary, so an existing binary can be reused with no verification. 939 binaries here, ~2 minutes
  each — the cache is worth about 31 hours of compiling.
- **Dropping the fixed cache latencies matters.** ChampSim will derive latency from size if the config
  does not name it. A search that does not do this gets bigger caches for free and finds nonsense.

**What it must look like to be general.** Configuration in, metrics dict out, with the build cache and
the tree pool as injected policy rather than hard-coded paths.

## 3. A measurement store keyed by configuration

**The gap.** CHIA caches task results, but a DSE loop needs something narrower and more durable: a
table per workload, keyed by configuration name, that several processes append to concurrently and
that survives as the experiment's dataset long after the run that produced it.

**What this repo learned.** `loop/suite.py` holds it in about 60 lines. Every process reads the table
before proposing, and appends under a file lock with an atomic rename, because a reader that catches a
half-written file kills a run that has been simulating for hours. The store is also what makes the
*metric* possible: "the best design measured on every workload of the suite" is the denominator every
published share divides by, and it is read straight out of the tables.

**What it must look like to be general.** A key-value store over configuration names with concurrent
append, plus one query — every configuration measured on all of a given set of targets. Nothing about
caches in it.

## 4. A memory of earlier searches

**The gap.** CHIA has no notion of a loop remembering earlier loops. Every search starts from nothing.
This is the piece with the most evidence behind it and the least prior art to copy.

**What this repo learned.** The whole of what survived measurement is small: **a memory is one
configuration, chosen as the one with the best mean share of the stock-to-best gap across every
remembered target, counting only configurations measured on all but one of them.** That last clause is
worth four points of gap share and is the reason the build has a confirm stage at all.

Everything richer was built and then removed after measuring it (`REPORT.md` §3): retrieval by
descriptor distance, a digest of conclusions for an agent to read, per-case effect tables. A plain
optimizer started from the single handover configuration matched or beat the agent reading the full
digest. **The block to contribute is the small one.**

**What it must look like to be general.** A target is a name plus a measurement store; a memory is
built from a set of targets and hands over one configuration. No trace, no simulator, no model. The
build procedure — search each group, then confirm the best everywhere — is generic over any
`evaluate(config, target)`.

## 5. A generation config that reaches the model, and backoff around it

**The gap.** `chia/chia/models/vertex.py` builds its request as:

```python
config_kwargs = {"max_output_tokens": self.max_tokens,
                 "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True)}
if self.system_message: config_kwargs["system_instruction"] = self.system_message
config = types.GenerateContentConfig(**config_kwargs)
```

Temperature, response MIME type and thinking budget never reach the model, so every call runs at the
model's defaults and the caller cannot see how the output cap is being spent. `max_tokens` defaults to
16000.

**Why it is not cosmetic, measured here.** On Gemini 2.5 Flash, thinking tokens are billed and counted
as output, so they consume the same cap as the answer. An arm of this loop asked for eight ranked
candidate designs as JSON; that answer plus the model's own thinking exceeded the 16k default, the
response was truncated, and the JSON failed to parse. The layer raised rather than retrying, so one
truncated call ended a run that had been simulating for an hour. Every run of one cell was lost that
way; one seed died twice, which is why that arm carries four seeds where the others carry five.

It also made the experiment **not reproducible**: temperature stayed at the model default rather than
the value the caller set, so the memory built through that path cannot be rebuilt bit-for-bit.

**Two separable fixes, both small and general.**

1. A `generation_config` passthrough — temperature, `response_mime_type`, `thinking_config` — plus the
   thinking-token count in `_last_metadata`, so a caller can see what the cap buys.
2. Waiting and retrying on rate limits (429), on 5xx, and on truncated or malformed answers, instead of
   raising immediately. This repo carries it as a wrapper in `analyst.generate_with_backoff`; it belongs
   under the model layer, where every CHIA loop would get it.

## 6. `gs://` trace resolution

**The gap.** `chia/chia/simulators/champsim.py::_resolve_trace` handles local paths and `s3://` URIs and
raises `NotImplementedError` for `gs://`, so a loop running on a GCP cluster cannot read its traces from
a bucket.

**What fills it.** `upstream/0001-champsim-gs-trace-resolver.patch` mirrors the existing `s3://` branch
with `google-cloud-storage`, downloading a remote trace once to the worker's temp directory and reusing
it. Nothing technical remains; it has not been opened as a pull request.

## 7. Two bugs in ChampSim's `prefetcher/spp_dev`

Both found running SPP on a small-core profile (L2 MSHR 16, DDR-1600) with `lbm`.
`upstream/0002-champsim-spp-dev-ghr-victim.patch` fixes both. No existing upstream issue was found for
either, and applying the patch is a setup step here — without it, `spp_dev` designs crash.

**Heap-buffer-overflow in the lookahead loop.** `confidence_q` and `delta_q` are sized to the L2 MSHR
count, but `read_pattern` appends up to `PT_WAY + 1` entries per lookahead step with no bounds check, so
a long confident chain overruns them — AddressSanitizer reports a READ of size 4 past a 64-byte region at
`confidence_q[i]`, `spp_dev.cc:78`. Silent SIGTRAP on macOS, heap corruption elsewhere. The fix stops the
lookahead when the next step cannot fit.

**The GHR victim search never finds a victim** when every entry has confidence 100, because `min_conf`
starts at 100 and the comparison is strict, so `assert(0)` — "[GHR] Cannot find a replacement victim!" —
fires. The fix starts the search above any legal confidence value.

# Findings worth a paper

Curated. A finding earns a place here only if it is **measured**, **reproducible by a
command in this repo**, and **says something a reader did not already know**. Intermediate
checkpoints, ops detail and anything still pending live in `CLAUDE.md`, not here.

---

## 1. An LLM searching produces the evidence a memory needs. An optimiser searching does not.

**confirmed**

The memory-building procedure spends a declared budget searching each workload suite before it
sweeps. Running that search stage twice at an identical budget — 41 designs per workload, same
space, same stages either side of it — the searcher decides whether the result is usable at all:

| stage-1 searcher | single-knob effects | **supported by ≥2 controlled pairs** |
|---|---|---|
| Gaussian process + expected improvement | 36 | **0** |
| LLM agent | 63 | **36** |

Zero against thirty-six, for the same simulation spend. A transferable claim about a knob needs
two controlled pairs — two designs differing in exactly that knob, measured twice — and the
optimiser supplied none.

The reason is structural rather than incidental. Expected improvement selects scattered points
out of a 20 000-design sample, and two points drawn from a 13-dimensional space almost never
differ in exactly one knob, so an optimiser's trace contains no controlled comparisons by
construction. An LLM proposes around a theme, varying one or two knobs at a time, and so
manufactures controlled pairs as a side effect of how it reasons.

This is what the LLM contributes to *building* a memory, as distinct from consuming one, and it
is measurable rather than asserted: at equal cost the agent's search leaves behind evidence and
the optimiser's leaves behind an optimisation trace.

**And the better evidence produces a better memory in use.** A memory's most consequential output
is the single design it hands a search that has never seen the workloads — the design that decides
the opening move. Simulating those handed-over designs on the two held-out datacenter suites, on
one scale, with no search involved:

| memory hands over | designs per workload | `dc` | `dc2` (shaped nothing) |
|---|---|---|---|
| agent-built | 122 | **95 %** | **93 %** |
| accumulated over the project | ~950 | 93 % | 80 % |
| optimiser-built | 122 | 89 % | 81 % |

Shares of each suite's stock-to-best-known gap. The agent-built memory wins on both, and by
thirteen points on `dc2` — the suite that shaped neither memory, nor the designs they hand over,
nor the choice of the first suite. It does so from one eighth of the simulations that the
accumulated memory took.

For scale: that single design, costing nothing at test time, reaches 0.4883 on `dc`, where the
best *sixteen-design run* of a memoryless LLM agent reached 0.4879.

**Owed before this is published as an arm result**: the memory arm must be re-run end to end on
the agent-built memory, five seeds, both cells. What is measured here is the handed-over design in
isolation; the arm's published opening of 81 % on `dc2` came from the accumulated memory's
handover, and a 93 % handover should move it, but that is an expectation until the arm is run.

Reproduce: `python -m loop.memory_build compare dc results/memory_bo.json results/memory_llm.json`
for the evidence counts; the handover scores are rows in the cached test tables.

---

## 2. How the memory is written matters more than what is in it.

**measured, and owed a re-run**: the raw-dump variant was removed from the code and its runs
deleted, so the numbers below are recorded rather than reproducible by a command. Publishing
them means re-adding the variant and re-running the cell — a cheap ablation, and one we should
run precisely because the effect is large.

The first version handed the agent the memory as it is stored: one case per remembered workload,
each with its descriptors, its best design and every measured single-knob effect. It is complete,
faithful, and the agent could barely use it. Restructuring **exactly the same information** as
conclusions — the moves that paid, the moves to avoid, the knobs where the remembered designs
disagree, and one design to copy and adapt — moved the memory arm's first simulated design from
**40 % to 65 %** of the stock-to-best gap.

Same memory, same budget, same model. Only the presentation changed, and it bought 25 points on
the metric the whole few-shot claim rests on.

The useful form of the claim is not "summarise your context". It is that a memory intended for an
LLM has to be **pre-reduced to conclusions, with the reasoning already performed**, because the
agent will not reliably perform that reduction itself inside a budget. A store that is correct and
complete can still be unusable, and the reduction is a design problem in its own right — which is
why `memory.digest` is a component with its own logic rather than a formatting step.

---

## 3. Two independent procedures converged on the same chip.

**confirmed**

The design the accumulated memory hands over and the design a from-scratch build hands over were
produced by entirely different routes — one from 1918 designs gathered over the project's
history, one from a single declared build in an afternoon. They agree on seven of thirteen knobs,
and the agreements are the ones that carry the performance:

| agree | differ |
|---|---|
| L1D prefetcher `next_line` | L1D sets/ways (near-inert on this chip) |
| **L2 prefetcher `va_ampm_lite`** | L2 size 512 KB vs 256 KB |
| L2 ways 16 | L2 replacement `lru` vs `drrip` |
| LLC prefetcher `none`, replacement `ship` | LLC geometry 4096×16 vs 8192×8 |
| L2 and LLC MSHR depths | |
| **LLC capacity: 4096 KB in both** | |

The LLC disagreement is geometry at identical capacity. This is evidence that the memory captures
structure in the problem rather than an artefact of how its table happened to be filled — the
single most useful check we have that a rebuild is faithful.

---

## 4. One design from memory beats sixteen designs of a tuned optimizer.

**confirmed on two suites, matched budget and matched configuration**

The comparison is like for like: the same cells, independent ceilings (each from a 100-design
random-forest search carrying no memory and no LLM — scoring against a design one of the arms
found would bound the metric at the best arm), and the same surrogate and settings on both
sides. Five seeds per arm, except `bo` on `dc` which has four.

`dc` (ceiling 0.4994) then `dc2` (ceiling 0.5485), mean over seeds:

| design | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| best memoryless optimizer from the stock chip | 22 % / 22 % | 22 % / 26 % | 42 % / 46 % | 64 % / 72 % | 82 % / 78 % | **89 % / 84 %** |
| `pooled_bo` from the memory's design | **90 % / 93 %** | 90 % / 93 % | 90 % / 93 % | 91 % / 95 % | 91 % / 95 % | 92 % / **96 %** |
| `memory`, the agent reading the digest | **90 % / 93 %** | 91 % / 93 % | 91 % / 93 % | 92 % / 93 % | 93 % / 93 % | 93 % / 95 % |

**The memory's first design is at 90 % and 93 %. The best memoryless optimizer reaches 89 %
and 84 % after sixteen designs, and never reaches the opening on either suite.** One
simulation against more than sixteen, with nothing differing between the second and third
rows but where the search begins.

The first row takes the **stronger of the two surrogates run on each suite**, deliberately.
Both a tuned Gaussian process and a random forest searched from the stock chip: they tie on
`dc` (89 % against 88 %) and differ by sixteen points on `dc2` (84 % against 68 %). Taking
the weaker one would inflate the margin by a seed draw, and the claim does not need it.

Two things fall out of the same table:

- **`pooled_bo` matches `memory` on `dc` (92 % against 93 %) and beats it on `dc2`
  (96 % against 95 %).** An optimizer handed the memory's design gets where the agent gets,
  with no digest, no prompt and no model call. **At test time the LLM is worth about a point
  on one suite and nothing on the other.** What it is demonstrably worth is at *build* time
  (§1), which is where the contribution sits.
- **Which surrogate a memoryless search uses cannot be resolved at these seed counts.** The
  Gaussian process scores 89 % / 84 %, the random forest 88 % / 68 %. On `dc2` that gap is
  sixteen points, but per seed the forest lands at 48, 62, 72, 91 and the process at 73, 76,
  88, 90, 96 — a difference of 16 points against a standard error of 10, t = 1.6, and across
  all nine cold-start runs the outcomes span 48 % to 96 %. An offline replay had predicted the
  forest four points *ahead* on a fair draw from the space, and it predicted neither arm.
  **At this budget the surrogate is not what decides the outcome, and cold-start search is
  unstable enough that a 16-point mean gap is a seed draw.** That is ArchGym's "all optimizers
  tie under tuned hyperparameters" arriving as noise rather than as a tie.

**Open, and owed before any "N times fewer simulations" claim is published.** The natural next
statement is a ratio — how many designs a memoryless search needs to reach what the memory arm
reaches at D16. We cannot state it yet. Measuring it against the 100-design reference gives D42
and D77 (a 4.8x ratio), but that reference is configured for a 100-design budget: a 10 %
initial design and batch 6, so by D16 it has made only about six model-guided picks and its
curve is flat at 79 % from D10 to D16. The arm's configuration — warm-up 3, batch 1 — climbs
roughly twice as fast early, so the true ratio is likely nearer 2.5-3x.

The measurement that would settle it: **run the `bo` arm's exact configuration long (about 100
designs) on both cells, and read off where it crosses the memory arm's D16.** Same warm-up,
same acquisition, same per-seed candidate pool; only the budget changes. Batch 1 would keep a
machine mostly idle, so batch 3 is the practical compromise and should be reported as such.
Until that exists, the matched-budget sentence above is the claim, and the ratio is not.

---

## 5. What transfers is a basin, not an optimum.

**confirmed, and it bounds the whole approach**

The memory hands over one design. Measuring every design in the cached tables by how many
knobs separate it from that handover:

| knobs from the handover | best design measured there | share of the gap |
|---|---|---|
| 0 (the handover) | 0.4883 | 90 % |
| 1 | 0.4910 | 93 % |
| 2 | 0.4922 | 94 % |
| 3 | 0.4936 | 95 % |
| 5 | 0.4932 | 95 % |
| **6** | **0.4994** | **100 %** |
| 8 | 0.4945 | 96 % |
| 10 | 0.4885 | 91 % |

**Everything within five knob changes of the handover is capped at 93–95 %.** The ceiling
design sits six knobs away and needs all six at once — a 64 KB L2 where the handover has
512 KB, `lru` at the last level where the handover has `ship`, and the two prefetchers
swapped between levels.

Three consequences:

1. The memory arm's 93 % is **not an optimizer failure**. It is the ceiling of the
   neighbourhood it was handed. No local refinement escapes it.
2. A search continued from the handover must **keep the whole space in view**. One- and
   two-knob steps provably cannot reach a six-knob move, which is why the memory-started arm
   keeps its global candidate pool instead of only expanding the incumbent's neighbours.
3. The digest's consensus is **locally right and globally wrong**. It reports `ship` as a
   move that pays and hands over a large L2; the best design known does the opposite of both.
   What the memory transfers reliably is a good region, not the optimum inside it.

This is the honest bound on few-shot transfer as we have built it: it buys the opening
almost for free, and it cannot by itself buy the last ten percent.

---

## 6. Contributions back to the frameworks

Five gaps hit while building the loop that belong in CHIA or ChampSim rather than in
this repo. Both are shallow clones here, so each goes through a fork. Patches for the
two that are already written live in `upstream/`.

### CHIA

**(a) `gs://` traces cannot be resolved.** `chia.simulators.champsim._resolve_trace`
raises `NotImplementedError` for `gs://` URIs while handling `s3://`, so a loop running
on a GCP cluster cannot read its traces from a bucket.
`upstream/0001-champsim-gs-trace-resolver.patch` mirrors the existing `s3://` branch
with `google-cloud-storage`.

**(b) The ChampSim build node accepts only a prefetcher.**
`ChampSimNode.build_champsim` takes a prefetcher module and nothing else, so a
design-space loop cannot build the configurations it wants to search.
`loop/chia_nodes.py::build_from_config` is the general version — any config JSON: cache
sizes, associativity, replacement policies, prefetchers at every level, MSHR depths,
core parameters — with `simulate` as the matching multi-trace run node. Candidate for
`chia.simulators.champsim`.

**(c) No generation config reaches Vertex Gemini, and this one cost us runs.**
`chia.models.vertex.VertexGeminiLLM` forwards only the maximum output tokens, the
system message and tools. A loop therefore cannot set temperature, JSON response mode,
or a thinking budget: every call runs at the model's defaults, and the caller cannot
see how many thinking tokens were spent.

This is not cosmetic. On Gemini 2.5 Flash thinking tokens are billed and counted as
output, so they consume the same cap as the answer. Our memory arm asks for eight
ranked candidate designs as JSON; that answer plus the model's own thinking exceeds the
layer's 16k output default, the response is truncated, and the JSON fails to parse.
The layer raises rather than retrying, so a single truncated call ends a run that has
been simulating for an hour. Every memory-arm run of cell `dc` was lost this way and had
to be repeated, and one seed died twice, which is why that arm carries four seeds where
the others carry five.

Two separable fixes, both small and general:
1. A `generation_config` passthrough (temperature, `response_mime_type`,
   `thinking_config`), plus the thinking-token count in `_last_metadata` so a caller can
   see what the cap is being spent on.
2. Waiting and retrying on rate limits (429) and truncated or malformed answers instead
   of raising immediately. We carry this as a backoff wrapper in `agent.ask`; it belongs
   under the model layer, where every CHIA loop would get it.

**(d) The case memory is simulator-agnostic and reusable.** `loop/memory.py` reads
knobs and descriptors from a `problem` dict and never touches ChampSim: cases built
from result tables, retrieval by standardised descriptor distance, the digest an agent
reads, and the design a memory hands over. Candidate for a `chia.analysis` block that
any agentic loop could wrap around its own simulator node.

### ChampSim

**(e) Two bugs in `prefetcher/spp_dev/spp_dev.cc`**, both found running SPP on a
small-core profile (L2 MSHR 16, DDR-1600) with lbm.
`upstream/0002-champsim-spp-dev-ghr-victim.patch` fixes both:
- *Heap-buffer-overflow in the lookahead loop.* `confidence_q` and `delta_q` are sized
  to the L2 MSHR count, but `read_pattern` appends up to `PT_WAY + 1` entries per
  lookahead step with no bounds check, so a long confident chain overruns them
  (AddressSanitizer: READ of size 4 past a 64-byte region at `confidence_q[i]`,
  spp_dev.cc:78). It is a silent SIGTRAP on macOS and heap corruption elsewhere. The fix
  stops the lookahead when the next step cannot fit.
- *The GHR victim search never finds a victim* when every entry has confidence 100,
  because `min_conf` starts at 100 and the search is strict — `assert(0)` "[GHR] Cannot
  find a replacement victim!". The fix starts the search above any legal confidence.

No existing upstream issue was found for either.

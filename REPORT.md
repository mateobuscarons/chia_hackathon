# Findings

Curated. A finding earns a place here only if it is **measured**, **reproducible by a
command in this repo**, and **says something a reader did not already know**. Intermediate
checkpoints, ops detail and anything still pending live in `CLAUDE.md`, not here.

---

## 1. An LLM's search generalises. An optimiser's overfits the workloads it was given.

**confirmed, and replicated six times**

The memory-building procedure spends a declared budget searching each workload suite before it
sweeps (`loop/memory_build.py`). Running that stage twice at an identical budget — 41 designs per
workload, same space, same stages either side of it — and changing only the searcher produces two
memories that differ in which designs they contain. The consequence shows up where it matters: in
the single design each memory hands to a search on workloads it has never seen.

| memory hands over | on the six workloads it searched | on unseen `dc` | on unseen `dc2` |
|---|---|---|---|
| optimiser-built | **91.0 %** | 84 % | 81 % |
| agent-built | 89.8 % | **90 %** | **93 %** |

Shares of each suite's stock-to-best-known gap. **The optimiser's design is the better fit to the
data it was given and the worse predictor of data it was not.** The agent's gives up a point at
home to gain six to twelve away.

This is not an artefact of the two test suites. Hold out one remembered workload, pick the handover
from the other five using only that searcher's own designs, and score it on the held-out one:

| held out | agent-built pool | optimiser-built pool |
|---|---|---|
| mcf | 78 % | **81 %** |
| omnetpp | **87 %** | 68 % |
| lbm | **88 %** | 74 % |
| bfs.urand | 64 % | 64 % |
| pr.urand | **88 %** | 85 % |
| bfs.kron | **99 %** | 96 % |
| **mean** | **84 %** | **78 %** |

Four wins to one, six replications of the same direction. Since the whole premise is a memory built
on some workloads being useful on *different* ones, a searcher that overfits the workloads it was
given is the wrong tool for filling it, however good its optimisation trace looks.

**Why the agent's designs transfer better is open.** Two candidate explanations were tested against
the cached tables and both fail. It is not the optimiser's missing LLC prefetcher: a prefetcher pays
+16 to +23 points on every suite, remembered and unseen alike, so it would cost the optimiser at home
too. It is not that expected improvement picks cornered designs: the correlation between how extreme
a design is and how much share it loses moving to unseen workloads flips sign across slices, +0.57 to
−0.35 on samples of 27 to 57 designs.

### The secondary result: controlled evidence

The same difference shows up in what each search leaves behind, and here the reason *is* understood.
Reading each memory as it stood immediately after the search stage, before any sweep:

| stage-1 searcher | single-knob effects | **supported by ≥2 controlled pairs** |
|---|---|---|
| Gaussian process + expected improvement | 36 | **0** |
| LLM agent | 57 | **30** |

Zero against thirty, for the same simulation spend. Expected improvement selects scattered points out
of a 20 000-design sample, and two points drawn from a 13-dimensional space almost never differ in
exactly one knob, so an optimiser's trace contains no controlled comparisons by construction. An LLM
proposes around a theme, varying one or two knobs at a time, and manufactures controlled pairs as a
side effect of how it reasons.

**This result currently buys nothing**, and the ablation in section 6 says why: the anchor stage hands
both searchers their controlled pairs afterwards (336 against 306 in the completed memories), so the
digest the agent reads comes out comparable either way. It is reported because the mechanism is real
and general, not because the loop exploits it.

### What is owed

The stage-1 search is **not reproducible** for the LLM: CHIA's Vertex layer forwards no generation
config, so temperature is the model default rather than the 0.7 in `loop/agent.py`, and `MB_SEED`
never reaches anything in that path. The optimiser build *is* bit-for-bit repeatable. Every number
above therefore rests on one build of each. A second build of both is what would settle it.

Reproduce: the handover scores and the leave-one-out table are rows in the cached tables, keyed by the
design in each memory's `pooled` field. For the evidence counts, rebuild each memory from the designs
its build record marks `stage == "search"` in the program's own group
(`results/memory_<searcher>_record.json`); reading the completed memories instead gives 306 and 336,
after the sweeps.

---

## 2. How the memory is written matters more than what is in it.

**measured, and owed a re-run**: the raw-dump variant was removed from the code and its runs
deleted, so the numbers below are recorded rather than reproducible by a command. Publishing
them means re-adding the variant and re-running the cell — a cheap ablation, and the effect is
large enough to be worth it.

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

**The head start, measured on both suites — and why it decays.** How many designs does a
memoryless search need to reach the levels the memory-started arms reach? Running the `bo`
arm's *exact* configuration long (same warm-up, acquisition and per-seed candidate pool; only
the budget changes; verified to reproduce the arm design-for-design on shared seeds), three
seeds per cell to 75 designs, the mean curve built exactly as every published row:

| level the memory-started arms reach | they reach it at | memoryless search, mean over seeds | per seed |
|---|---|---|---|
| `dc` 90 % — the handover itself | **D1** | **D27** | D14, D25, D30 |
| `dc` 92 % — `pooled_bo` at D16 | D16 | D30 | D19, D30, D34 |
| `dc` 93 % — `memory` at D16 | D16 | D30 | D21, D30, D39 |
| `dc2` 93 % — the handover itself | **D1** | **D25** | D13, D34, D69 |
| `dc2` 95 % — `memory` at D16 | D16 | D37 | D21, D49, D73 |
| `dc2` 96 % — `pooled_bo` at D16 | D16 | D49 | D21, D49, >75 |

**One simulation against twenty-five to twenty-seven for the opening, on both suites.** That
is the claim the memory supports, and it is stable across two independent workload sets.

**By the sixteen-design budget the advantage has decayed to 1.9x on `dc` (D30 against D16) and
2.3-3.1x on `dc2`.** A 27-design head start becomes a factor of two, because the
memory-started arms gain only 2-3 points across their whole budget (90 -> 92/93 on `dc`,
93 -> 95/96 on `dc2`) while a memoryless search grinding upward recovers most of the distance.
They stall because they cannot leave the handover's basin (section 5): everything within five
knob changes of it is capped at 93-95 %, and the optimum sits six away.

So the contribution has an honest shape: **what is bought is a head start, and it is worth only
as much as the continuation search can exploit. Today's continuation cannot exploit it.** The
open work is not a better memory — the handover already opens at 90-93 % on suites it has never
seen — but a search that, given a strong start, keeps going.

**The spread is a second result, and on `dc2` a larger one than the mean.** Per seed, a
memoryless search reaches the `dc2` handover's level at D13, D34 and D69, and one seed never
reaches `pooled_bo`'s D16 level inside 75 designs. On `dc` one seed opened *below the stock
chip* (-16 % of the gap) and sat at 82 % for sixteen designs. Every memory-started seed opened
at exactly 90 % or 93 %. **The memory does not only start higher, it removes the bad draw** —
which is what a fixed simulation budget actually buys a design team.

Reproduce: `loop/forest.py` is the searcher; the long runs are the `bo` arm at `budget=75`.

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

## 6. Half the memory build buys four lines of text.

**confirmed, and free to reproduce**

`loop/memory_build.py` tags every design with the stage that asked for it, and the result tables
hold every design. A stage can therefore be removed *after the fact* — rebuild the memory from a
filtered set of design names and read what changes. No simulation is involved. Four variants of
each memory, scored on `dc2`:

| build | variant | simulations | effects with ≥2 pairs | digest "moves that paid" | "traps" | handover | `dc2` share |
|---|---|---|---|---|---|---|---|
| agent | full | 715 | 336 | 3 lines | 1 line | — | 93 % |
| agent | no winner sweeps | 540 | 108 | **0** | **0** | unchanged | 93 % |
| agent | no stock sweep | 535 | 85 | **0** | **0** | unchanged | 93 % |
| agent | no anchor stage | 360 | 51 | **0** | **0** | unchanged | 93 % |
| optimiser | full | 711 | 306 | 4 lines | 1 line | — | 81 % |
| optimiser | no winner sweeps | 549 | 72 | 2 lines | 0 | unchanged | 81 % |
| optimiser | no stock sweep | 525 | 78 | 1 line | 0 | unchanged | 81 % |
| optimiser | no anchor stage | 363 | 6 | **0** | **0** | unchanged | 81 % |

**The anchor stage is minimal, not redundant.** Removing either of its two sweeps empties the
digest's conclusions. That is the design working as specified: `memory.digest` will not state an
effect on fewer than two controlled pairs, and each sweep supplies roughly one pair per single-knob
step, so the two are jointly necessary rather than alternatives.

**And it does not touch the design the memory hands over.** In all eight variants the handover is
identical — it comes from the confirm stage's pool, never from the sweeps. That design is what
carries the result: `pooled_bo`, which reads no digest at all and only starts from it, matches or
beats the `memory` arm that reads the whole digest (section 4).

So the anchor stage costs **355 simulations, half the build**, changes the handover not at all, and
everything it buys is four lines of digest text whose measured value at test time is about one
point.

This is a fact about the loop as it stands, not a verdict on the design. Traps and consensus would
become load-bearing for a continuation search that used them to *exclude* regions rather than to
suggest moves, which is open work. But a reader should know that today, the half of the memory
build that manufactures evidence is not what produces the result.

Reproduce: filter `results/memory_<searcher>_record.json` by `stage`, pass the surviving design
names to `memory.build`, and read `pooled` and the digest off each variant.

---

## 7. Contributions back to the frameworks

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

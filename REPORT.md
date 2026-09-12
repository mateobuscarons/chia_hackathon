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

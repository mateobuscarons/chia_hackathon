# Findings

Three sections. The first two are measured and reproducible by a command in this repo. The third
records what we removed and the measurement that justified removing it — those artifacts are in git
history, not in the tree, and are not rerunnable from it.

---

## 1. One design from memory against sixteen from a memoryless search

**Confirmed on two independent suites, matched budget, matched configuration.**

The comparison is like for like: the same cells, the same surrogate and settings on both sides, and
an independent ceiling for each suite from a 100-design random-forest search carrying no memory —
scoring against a design one of the arms found would bound the metric at the best arm. Five seeds
for `pooled_bo`, four for `bo`.

Share of the stock-to-best-known gap, mean over seeds:

| suite | arm | D1 | D2 | D4 | D8 | D12 | D16 |
|---|---|---|---|---|---|---|---|
| `dc` (ceiling 0.4994) | `bo` from the stock chip | 22 % | 22 % | 61 % | 72 % | 75 % | 88 % |
| | `pooled_bo` from the memory | **90 %** | 90 % | 90 % | 91 % | 91 % | 92 % |
| `dc2` (ceiling 0.5485) | `bo` from the stock chip | 19 % | 21 % | 46 % | 57 % | 61 % | 68 % |
| | `pooled_bo` from the memory | **93 %** | 93 % | 93 % | 95 % | 95 % | 96 % |

Nothing differs between the two rows of a pair but where the search begins.

**How many simulations that is worth.** Running the `bo` arm's *exact* configuration long — same
warm-up, acquisition and per-seed candidate pool, only the budget changes, verified to reproduce the
arm design-for-design on shared seeds — three seeds per cell to 75 designs:

| level | reached from memory at | memoryless, mean | per seed |
|---|---|---|---|
| `dc` 90 % — the handover itself | **D1** | **D27** | D14, D25, D30 |
| `dc` 92 % — `pooled_bo` at D16 | D16 | D30 | D19, D30, D34 |
| `dc2` 90 % | **D1** | D15 | D13, D15, D26 |
| `dc2` 93 % — the handover itself | **D1** | **D26** | D13, D34, D69 |
| `dc2` 95 % | D8 | D37 | D21, D49, D73 |
| `dc2` 96 % — `pooled_bo` at D16 | D16 | D49 | D21, D49, >75 |

**One simulation against twenty-six or twenty-seven for the opening, on both suites.** By the
sixteen-design budget the advantage has decayed to about 2x on `dc` and 3x on `dc2`, because the
memory-started arms gain only 2-3 points across their whole budget while a memoryless search
grinding upward recovers most of the distance. Section 2 says why they stall.

So the contribution has an honest shape: **what is bought is a head start, and it is worth only as
much as the continuation search can exploit.**

**The spread is a second result, and on `dc2` a larger one than the mean.** Per seed:

```
dc   bo         D1   31%, 45%, 14%,  0%      D16  87%, 92%, 84%, 89%
dc   pooled_bo  D1   90%, 90%, 90%, 90%, 90% D16  92%, 92%, 92%, 93%, 91%
dc2  bo         D1   37%,  3%, 22%, 12%      D16  72%, 91%, 48%, 62%
dc2  pooled_bo  D1   93%, 93%, 93%, 93%, 93% D16  94%, 94%, 99%, 95%, 98%
```

One memoryless seed on `dc` opened at the stock chip exactly, and one on `dc2` finished sixteen
designs at 48 % of the gap. Every memory-started seed opened at exactly 90 % or 93 %. **The memory
does not only start higher, it removes the bad draw** — which is what a fixed simulation budget
actually buys a design team.

Reproduce: `python -m loop.search score results/runs/dc2_g1.json`; the long curves are
`results/baseline/<cell>_s<seed>.json`, regenerated with `BUDGET=75 SEEDS=3 python -m loop.search
dc2 long bo`.

---

## 2. What transfers is a basin, and its depth is not the same on both suites

**Confirmed, and it bounds the whole approach.**

The memory hands over one design. Taking every design in the cached tables and grouping by how many
knobs separate it from that handover, the best share reached at each distance:

| knobs from the handover | `dc` | `dc2` |
|---|---|---|
| 0 (the handover) | 90 % | 93 % |
| 1 | 93 % | 95 % |
| 2 | 94 % | **99 %** |
| 3 | 95 % | 95 % |
| 4 | 94 % | **100 %** |
| 5 | 95 % | 100 % |
| 6 | **100 %** | 100 % |

**On `dc` the ceiling design sits six knob changes away and needs all six at once** — a 256-set L2
where the handover has 512, four ways where it has sixteen, `lru` at the last level where it has
`ship`, and the two prefetchers swapped between levels. Everything nearer is capped at 95 %.
**On `dc2` a two-knob move already reaches 99 % and a four-knob move reaches the ceiling.**

That difference is the explanation for the one thing the head start does not do equally on both
suites. The opening is the same (90 % and 93 %) and the budget is the same, but `pooled_bo` climbs
to 96 % on `dc2` and only 92 % on `dc`. It is not that the memory transfers worse to `dc` — it
transfers the same. It is that **the region it transfers into is poor on `dc` and rich on `dc2`.**

Three consequences:

1. A memory-started arm reaching 92 % on `dc` is **not an optimizer failure**. It is close to the
   ceiling of the neighbourhood it was handed, and no local refinement escapes it.
2. A search continued from the handover must **keep the whole space in view**. One- and two-knob
   steps provably cannot reach a six-knob move, which is why both forest arms keep a global
   candidate pool instead of only expanding the incumbent's neighbours.
3. What the memory transfers reliably is a good region, not the optimum inside it. This is the
   honest bound on few-shot transfer as we have built it: it buys the opening almost for free, and
   it cannot by itself buy the last ten percent.

Reproduce: group `suite.measured_designs(problem)` by `len(knobs_changed(design, memory.handover(...)))`.

---

## 3. What we removed, and what the measurement said

Four mechanisms were measured, found not to earn their cost, and deleted. Each is recorded here
because a reader will ask why the loop does not have it. The artifacts are in git history; none of
these is rerunnable from the current tree.

**The digest — a memory written as conclusions rather than as data.** The first version handed the
agent the memory as stored: one case per remembered workload with its descriptors, its best design
and every measured single-knob effect. Restructuring *exactly the same information* into conclusions
— what paid, what to avoid, where remembered designs disagree, one design to copy — moved the LLM
arm's first simulated design from **40 % to 65 %** of the gap. That is a real result about presenting
a memory to a model. It was removed anyway, because `pooled_bo` — which reads no digest and only
starts from the handover — matched or beat the arm that read the whole thing (96 % against 95 % on
`dc2`, 92 % against 93 % on `dc`). **At test time the digest was worth about one point.**

**The anchor stage — half the memory build.** The build used to run one-knob sweeps from the stock
chip and from each search's winner, to manufacture the controlled pairs the digest needed. Removing
either sweep emptied the digest's conclusions, so the stage was minimal rather than redundant. But it
cost **355 simulations, 87 of 163 designs**, and **the handover is byte-identical without it** — it
comes from the confirm pool, never from the sweeps. Half the build bought four lines of digest text
whose measured value at test time was about one point.

**The optimiser-built memory.** Building the memory twice at an identical budget, changing only the
stage-1 searcher, produced two different handovers:

| memory hands over | on the six workloads it searched | on unseen `dc` | on unseen `dc2` |
|---|---|---|---|
| optimiser-built | **91.0 %** | 84 % | 81 % |
| agent-built | 89.8 % | **90 %** | **93 %** |

The optimiser's design was the better fit to the data it was given and the worse predictor of data it
was not, and holding out each remembered workload in turn said the same thing (agent pool 84 %,
optimiser pool 78 %, four wins of six). The mechanism behind it was never explained: it is not the
optimiser's missing LLC prefetcher, and it is not that expected improvement picks cornered designs —
both were tested and both fail. Every number rests on one build of each, because the LLM build is not
reproducible. The optimiser searcher was removed when the memory was cut to its two essential stages,
so this comparison cannot be extended without rebuilding both.

**The workload descriptors.** The agent's prompt used to carry twelve numbers per workload, read from
the trace with no simulator — footprint, access rate, stride regularity, and a footprint-theory
estimate of the misses a bigger cache could remove. They were **48 % of a round-1 prompt**. Testing
each against the knob effect its own legend claimed it predicted, over the twelve workloads with
tables:

```
movable_llc_mpki  -> gain from doubling the LLC          +0.73   real
stride_regular    -> gain from the L2 stride prefetcher  +0.87   real
pred_llc_miss     -> gain from srrip                     +0.42   weak
movable_l2_mpki   -> gain from doubling the L2           -0.33   nothing
reuse_local       -> gain from srrip                     -0.17   nothing
movable_llc_mpki  -> total headroom                      -0.14   nothing
```

Two of twelve carried signal. The model read them and cited them by name — and in six of ten first
picks across both suites it read a small positive `movable_llc_mpki` as "capacity-bound" and proposed
the single-knob LLC doubling, worth 15 % of the gap on `dc2` and 35 % on `dc`. The four picks that
instead combined prefetchers, policy and capacity were worth 41-76 %. The block gave no scale, so
"positive" read as "large", and the descriptor that predicted best pointed at a knob the model did not
reach for. Removed entirely: the agent now sees the chip, the knobs and its own measurements, which
makes `llm_alone` a baseline with no information the optimizer lacks.

---

Contributions back to CHIA and ChampSim are in `CHIA_BLOCKS.md`.

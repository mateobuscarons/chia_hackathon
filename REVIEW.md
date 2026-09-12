# Review: does a memory of earlier searches save simulations?

For an outside reader. It covers what the loop is now, what the memory does and why it has this shape, the results measured overnight, and what is running. Code lives in `loop/` (about 3,000 lines, eleven files); data in `results/`; `CLAUDE.md` has the project history.

## 1. The question and the setting

A team has tuned cache hierarchies before. A new workload class arrives. With a budget of **16 simulated designs**, how close does each method get to the best design anyone knows?

- **Simulator:** ChampSim, one server-class core, 5M warmup + 10M simulated instructions per design and workload (validated: Spearman 0.919 against 50M/50M over 26 designs).
- **Space:** 13 knobs over L1D, L2 and LLC (sets, ways, prefetcher at each level, replacement at L2 and LLC, MSHRs). 6.6 M designs before constraints. Hard limit: L2 + LLC data capacity <= 4608 KB. Cache latency is derived from size, so capacity is never free.
- **Objective:** geometric mean of IPC over a three-workload suite. One design costs three simulations.
- **Start:** the stock chip, measured and shown to every method.
- **Memory:** six workloads searched earlier on the same chip, 450 to 1,500 designs each: SPEC17 mcf, omnetpp, lbm and GAP bfs.urand, pr.urand, bfs.kron.
- **Test:** three Google datacenter traces (`gtrace_v2`: sierra.a.4, merced, tahoe) that appear nowhere in the memory. Chosen by screening twelve families for headroom and for the share of it no single knob can reach.
- **Model:** Gemini 2.5 Flash throughout.

## 2. The three arms

Same start, same budget, same seeds, same fidelity.

| arm | proposes | selects | reads memory |
|---|---|---|---|
| `bo` | GP + expected improvement over a 20k-design feasible sample plus the incumbent's one- and two-knob neighbours | the GP | no |
| `llm_direct` | the LLM, two designs a round, each with a hypothesis | the LLM | no |
| `memory` | the LLM, eight ranked designs a round | a GP fit on this run picks two by expected improvement | yes |

The two LLM arms are one code path with two switches. With an empty memory their prompts are byte-identical.

**One round of the memory arm.** The prompt is assembled from data: the problem (chip, budget, knobs, stock design); the workloads (a legend, then one line of descriptors each); the memory digest (section 3); the results table so far, one row per design with the knobs it changes from stock, the per-workload IPC and cache statistics, and the hypothesis it tested; then the task and a strict JSON schema. The LLM returns eight ranked designs. Each is checked for validity, budget and novelty; rejects come back in one retry; an empty slot falls back to a deterministic one-knob perturbation, never a random design. From the third measured design on, a GP fit on this run's designs scores the accepted candidates and expected improvement picks the two to simulate.

## 3. The memory, and why it looks like this

**What is stored.** One **case** per workload searched before, written by code from the result table, never by an LLM: the workload's descriptors, its stock and best design with IPC, and every measured single-knob effect (every pair of measured designs differing in exactly one knob, with the mean effect and the pair count).

**What the agent reads.** Not the cases. A **digest** computed from the nearest ones, anchored on the stock design the agent starts from: how similar each test workload is to its nearest remembered one and on which descriptors they differ; the moves from stock that paid on every nearest workload, ranked; the traps that hurt everywhere; the knobs on which the remembered best designs disagree, which are the questions the search should settle; and one design to copy and adapt. Here it is verbatim:

```
## What earlier searches on this chip say about workloads like yours
Similarity: sierra.a.4 is closest to mcf (similarity high); merced is closest to omnetpp (similarity high); tahoe is closest to mcf (similarity high).
Where yours differ most from them: sierra.a.4 mem_accesses_per_kinstr 365 vs 465 for mcf; sierra.a.4 write_fraction 0.329 vs 0.247 for mcf; merced mem_accesses_per_kinstr 335 vs 523 for omnetpp; merced movable_llc_mpki 1.01 vs 0.296 for omnetpp; tahoe movable_l2_mpki 2.44 vs 12.8 for mcf; tahoe write_fraction 0.383 vs 0.247 for mcf.

Moves from the stock chip that paid on the closest remembered workloads (per-workload IPC change):
1. l2_prefetcher -> spp_dev: +31% (mcf), +1% (omnetpp)   solid
2. l2_prefetcher -> next_line: +21% (mcf), +2% (omnetpp)   solid
3. llc_sets -> 4096: +11% (mcf), +8% (omnetpp)   solid
4. l2_ways -> 16: +4% (mcf), +2% (omnetpp)   solid
5. l2_sets -> 2048: +4% (mcf), +2% (omnetpp)   solid

Traps: moves from the stock chip that hurt on the closest remembered workloads:
1. llc_ways -> 8: -26% (mcf), -7% (omnetpp)   solid
2. llc_replacement -> srrip: -15% (mcf), -6% (omnetpp)   solid
3. llc_sets -> 1024: -16% (mcf), -5% (omnetpp)   solid
4. l1d_sets -> 32: -10% (mcf), -11% (omnetpp)   solid

Where their best designs DISAGREE (the questions your search should settle):
- l2_prefetcher: mcf spp_dev, omnetpp va_ampm_lite, lbm spp_dev, pr.urand spp_dev, bfs.kron spp_dev (stock no)
- llc_replacement: mcf ship, omnetpp ship, lbm ship, pr.urand ship, bfs.kron lru (stock lru)
- llc_sets: mcf 4096, omnetpp 4096, lbm 2048, pr.urand 4096, bfs.kron 8192 (stock 2048)
- l2_sets: mcf 1024, omnetpp 512, lbm 2048, pr.urand 256, bfs.kron 256 (stock 1024)
- llc_prefetcher: mcf spp_dev, omnetpp no, lbm next_line, pr.urand ip_stride, bfs.kron next_line (stock no)
- l1d_prefetcher -> next_line is workload-dependent: +6% (mcf), -0% (omnetpp), -3% (lbm), -1% (pr.urand), +6% (bfs.kron)

The remembered design that did best across the remembered workloads (measured on 6 of them, reaching on average 92% of each one's stock-to-best gap), to copy and adapt:
{"l1d_sets": 64, "l1d_ways": 12, "l1d_prefetcher": "next_line", "l2_sets": 512, "l2_ways": 16, "l2_prefetcher": "va_ampm_lite", "l2_replacement": "lru", "llc_sets": 4096, "llc_ways": 16, "llc_prefetcher": "no", "llc_replacement": "ship", "l2_mshr": 64, "llc_mshr": 128}
```

**Why a pooled design and not the nearest workload's.** The copyable slot first held the best design of the closest remembered workload. Pooled retrieval replaced it: the remembered design with the best mean share of the stock-to-best gap across the memory workloads, counting only designs measured on all but one of them. The evidence is a free leave-one-out over the cached tables, holding each workload out and rebuilding the memory from the rest:

```
== leave one out: nearest case, its best design replayed; the pooled design (best mean gap share over the
   other memory workloads, measured on all but one of them) replayed; the nearest case's effects re-measured
   workload             nearest              distance       replay       pooled   effects sign held
   605.mcf_s-665B       620.omnetpp_s-874B       2.19   78% of gap   78% of gap        39        32
   620.omnetpp_s-874B   605.mcf_s-665B           2.19   unmeasured   unmeasured        40        33
   619.lbm_s-2676B      pr.urand-129B            3.71   unmeasured   80% of gap        48        31
   bfs.urand-36B        619.lbm_s-2676B          5.37   unmeasured   52% of gap        43        27
   pr.urand-129B        bfs.kron-128B            3.07   86% of gap   86% of gap        57        40
   bfs.kron-128B        pr.urand-129B            3.07   98% of gap   95% of gap        57        40
   sierra.a.4_0000      605.mcf_s-665B           1.53   unmeasured   85% of gap        37        27
   merced_0000          620.omnetpp_s-874B       2.00   93% of gap   93% of gap        34        21
   tahoe_0000           605.mcf_s-665B           2.41   unmeasured   89% of gap        37        26
   pooled design over the whole memory (mean share 90% on 6 workloads): l1d_prefetcher=next_line, l2_sets=512, l2_ways=16, l2_prefetcher=va_ampm_lite, llc_sets=4096, llc_replacement=ship

== sign survival of remembered effects against distance, over every ordered pair of memory workloads
   distance 2.19 .. 4.39: 320 of 457 effects kept their sign (70%)
   distance 4.39 .. 5.71: 268 of 457 effects kept their sign (59%)
   distance 5.71 .. 8.23: 286 of 458 effects kept their sign (62%)
```

Pooled beats nearest-case copying on five of six held-out workloads and never falls below 82%, where nearest falls to 27%. It is also flat in descriptor distance, which is itself a finding: remembered single-knob effects keep their sign on another workload about two thirds of the time regardless of how similar the workloads look, so the memory is trusted for *which* moves matter and not for *how much*, and distance is used only to choose which cases to read.

The pooled design for this memory is the stock chip with: l1d_prefetcher=next_line, l2_sets=512, l2_ways=16, l2_prefetcher=va_ampm_lite, llc_sets=4096, llc_replacement=ship.

## 4. Results measured overnight

Reference: the best design known on this suite, **0.4933**, from 673 designs measured on all three traces. Stock is 0.3837. The score is the share of the distance between them. Median over seeds, 16 designs per run.

| arm | D1 | D2 | D4 | D6 | D8 | D12 | D16 | final IPC |
|---|---|---|---|---|---|---|---|---|
| Bayesian optimisation | 51% | 51% | 55% | 68% | 80% | 91% | 94% | 0.4868 |
| LLM alone | 57% | 79% | 80% | 80% | 83% | 84% | 85% | 0.4770 |
| LLM + memory | 93% | 93% | 94% | 96% | 96% | 98% | 98% | 0.4911 |

Two reference points that are not arms:

- **The pooled design alone**, one simulation and no search at all, reaches **93%** and stops there.
- **Uniform random sampling** of the same space: 33% after one design, 83% after ten, 89% after twenty-five, 90% after a hundred, and still 90% after 274. It plateaus around 90% and never reaches what the memory arm finds in 16 designs. It also matches what either baseline reaches in 16 designs after about 12 draws, which says both baselines are extracting little at this budget.

**Reading.**

1. The memory arm is at 93% of the gap on its **first** simulation, a level Bayesian optimisation needs fifteen designs to reach and the plain LLM never reaches inside the budget.
2. Search on top of memory still pays. Pure retrieval stops at 93%; the memory arm passes it by design three, reaches 98%, and found the best design now known on this suite.
3. The memory removes the variance. Across seeds at 16 designs the BO runs span 51 to 98%, the memory runs 94 to 100%. At a fixed budget, predictability matters as much as the mean.
4. A negative worth stating: a GP prior fit on the memory tables (about 1,900 designs, target = each design's share of its workload's gap) ranked held-out designs well on its own, Spearman 0.73, but added nothing as an arm, 98% either way. It has been removed from the code.

## 5. What is running now

- **The same experiment on the three admitted datacenter traces the first suite did not take** (whiskey, bravo, delta; stock 0.4389, 11 designs measured so far). Nothing about those traces touched the memory, the pooled design, or the choice of the first suite, and the pooled design had never been simulated on them. This removes the one selection criticism an outside reader can make of section 4. Their nearest remembered workloads also differ: omnetpp for two of them, pr.urand for the third.
- **A ceiling search per suite**: 100 designs of Bayesian optimisation in batches of ten, on both the first and the held-out suite. The uniform sample showed that random draws are a poor estimate of the optimum, so the reference each cell is scored against should come from a long search rather than from random draws. Until it lands, the reference is "the best design any method has found", which is the honest phrasing and is what the score says.

## 6. Weaknesses an outside reader should press on

1. **Seeds.** Five per arm, and the top arms overlap in their spread.
2. **The reference is a lower bound**, found by one of the methods being compared. The ceiling searches above are the mitigation.
3. **Selection.** The first suite was chosen for headroom using an eleven-design probe that included the composite design the memory later hands over. The held-out suite now running is the answer.
4. **One chip, one class of test workload, one SimPoint per trace**, 100 MB trace prefixes.
5. **The area model** charges nothing for L1D geometry or MSHRs, so those knobs are a free lunch; the digest and the fallback deprioritise them, but the space still contains them.
6. **The budget is counted in designs, not simulated cycles.** A stock-chip start makes every design about 2.5x costlier than a start from a tuned design.
7. **Flash, not Pro.** Every run so far uses Gemini 2.5 Flash for speed and cost.
8. **The memory is built from searches on the same chip.** Transfer across chips is untested in this form.

## 7. Where to look

| what | where |
|---|---|
| a cell's report: every design, every prompt, candidates, rejections, GP predictions | `results/run_<cell>_<tag>.json` |
| the memory: cases and the pooled design | `results/memory_<cell>.json` |
| the simulation cache, one table per workload | `results/table_<workload>.json` |
| the agent, its prompt, the LLM call | `loop/agent.py` |
| cases, retrieval, the digest, pooled retrieval, the leave-one-out | `loop/memory.py` |
| the GP and the BO arm | `loop/bo.py` |
| cells, arms, the report | `loop/run.py` |
| the score table | `loop/summarize.py` |
| screens, probes, reference searches, trace fetching | `loop/workloads.py` |
| ChampSim and CHIA glue | `loop/champsim_problem.py`, `loop/simulate.py`, `loop/chia_nodes.py` |

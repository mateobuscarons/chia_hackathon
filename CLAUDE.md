# CHIA Hackathon — Rules that Bet (knowledge that survives an unseen chip *and* an unseen workload class)

A3 workshop hackathon (agentic-arch.org). **Real deadline Sep 24, 2026 AoE** (internal target Sep 20; Sep 21-24 buffer). Deliverable: 4-page paper + open-sourced CHIA loop with reproducible results; "reusable CHIA blocks upstreamed to mainline" is an explicit track. Judged by the A3 program committee, run by the six organizers: Gonzalez and Jain (Google, ArchAgent authors), Karandikar (UC Berkeley, CHIA PI), Yazdanbakhsh (Google DeepMind, ArchGym: "all optimizers tie under tuned hyperparameters"), Huang (NVIDIA, DOSA), Skarlatos (CMU). They reward novelty and quality of the agentic architecture, matched-budget baselines, seeds, realizability (they flagged ChampSim's lack of area/power feedback themselves), and reusable blocks + datasets. Not raw IPC.

## The mechanism (v7, headline set Sep 8 evening)

One sentence: **rules whose conditions are written on a workload's measurable habits - profiled once from its trace, never its identity - and whose claims are settled by the simulator on old chips, still fire correctly and still cut the search on a workload class and a chip they have never seen.**

This is a claim about the **rule language**, not about a learned model. Keep the two apart in every sentence we write: our own offline test shows a learned *surrogate* does **not** transfer to a held-out program (and the literature says it needs 7-30 training programs). A rule that says "footprint over LLC size > 4 and stride regularity < 0.2 -> SPP loses to next_line" references measurable habits, so it can be checked on a program nobody has ever profiled. That distinction is the paper.

- **Workload profile, once per trace, no simulator** (`loop/trace_profile.py`): working set, stride regularity, temporal locality, LRU miss-ratio curve (footprint theory, Xiang et al. ASPLOS 2013). Chip-independent by construction.
- **Conditions are chip-independent and program-agnostic**: profile descriptors read against the target chip's geometry (`champsim_problem.chip_descriptors`). The chip's own measured speed is never a condition. Thresholds fitted from per-workload evidence (v5.1).
- **Claims are measured, never stated**: controlled pair, |effect| >= 1%, sign matches the LLM's stated direction; rejections kept in `rejected_rules`. Losing rules are re-scoped, not deleted. Brier-scored ledger for every forecaster (surrogate, rules, analyst).
- **Rules in selection: soft, bounded categorical priors.** A speaking rule shifts the GP mean for its categorical value by at most (measured effect x credibility), decaying as real observations accumulate. **Never a filter.** Rules bet on every design they speak about. Dedicated one-knob claim tests only when EI brought no improvement last round (fixes v5.1's claim-test flood).
- **The LLM has two jobs, both ablatable:** (i) write conditions and rules from the measured one-knob effects; (ii) right of reply when EI stalls. The LLM never judges its own claims.
- **Multi-core mix descriptors** (decided Sep 8 evening): on a chip where N cores share the LLC, a workload is a mix of N programs. Capacity descriptors **sum** across the mix (N cores sharing an LLC collectively need the sum of their working sets, so "footprint / LLC size" keeps its meaning); rate descriptors (accesses per kilo-instruction, stride regularity, locality) are **access-weighted averages**. Reuses the mean/max machinery `make_suite_problem` already has.
- **Objective:** suite geomean of per-workload IPC under a hard area cap. On a multi-core chip the cap must count **one L2 per core and one shared LLC**, which makes private capacity N times more expensive than shared - a real interaction the rules can be tested on.
- **Guardrails:** rebuild per config, budget re-check, fresh-build re-run of the final best design, skeptic pass that tries to falsify the accepted hypothesis.

**Demoted by the headline change, deliberately** (say so if you disagree): the CPI-stack mean function (failed G1, see below - it survives as an offline finding and a paper section, not as the surrogate's mean); the `physics` arm (it would just be a worse GP); the LLM term proposer (it existed to feed the stack); CACTI area/energy (stretch only - the KB area proxy with correct per-core accounting already carries the interesting constraint).

## Decisions taken Sep 8 (user)

1. **Headline: "knowledge survives a workload class it never saw."** Rules learned on SPEC-class programs and chips A/B are tested on graph-analytics workloads (GAP) on unseen chips. Chosen because ArchAgent - written by four of the six judges - names "overfitting versus specialization, extrapolation from representative workloads" as an open community problem, and because margins in this field are 1-3% and shrinking, so methodology is the currency, not IPC.
2. Rules in selection: **categorical priors, soft and bounded**, as above. Not filters, not bets-only.
3. **Fidelity check** (baseline + top-20 cached + ~5 random designs at 50M warmup / 50M sim, ~1 h wall on 32 vCPUs, a few USD). Every downstream number depends on whether big-LLC ranks hold under a real warmup.
4. `textbook` arm stays (LLM prior knowledge alone, no LLM at test time): answers "isn't the LLM's prior doing the work?" and gives the thesis figure: stated rules' Brier vs measured rules' Brier.
5. Fixed reference for Tier C SPEC suite: **0.7491** (verified: best suite geomean of the 447 C designs measured on all 4 workloads; chip C baseline 0.5990, 25.1% headroom). Each new cell needs its own reference, collected the same way.
6. Freeze one A/B warm-start pool for `bo_pooled`, identical across cells and seeds (today the pool is that run's own learn history, so the baseline inherited the treatment's choices).
7. **Mechanism freezes Sep 15** (slipped from Sep 12 to fit the four setup changes); after that only measurement and writing. Hard stop Sep 24 AoE.
8. Spending: each launch needs the user's explicit OK with cost and time; VM stopped when idle.
9. **Workload hygiene, and a validated admission gate** (`python -m loop.offline admit`, zero cost, from the trace profile alone). ArchAgent's criterion (baseline LLC MPKI > 1) is necessary but not sufficient - it admits workloads whose misses are all compulsory. Ours: a workload is admitted for **capacity** only if, across the capacities the search space can actually buy (LLC read at 576-4608 KB on chip C), the miss-ratio curve moves at least **1 MPKI**. Validated against the Tier A full factorial: the gate's two ADMITs are exactly the two workloads with non-zero measured capacity variance, its two REJECTs exactly the two with ~zero.

| trace | MPKI@min | movable | unfixable (compulsory) | measured capacity variance | verdict |
|---|---|---|---|---|---|
| mcf | 23.5 | **12.6** | 10.7 | 0.46 | ADMIT |
| omnetpp | 11.0 | **2.6** | 8.3 | 0.86 | ADMIT |
| lbm | 45.2 | 0.6 | 42.7 | 0.01 | reject on capacity; **keep as the prefetcher workload** (l2_prefetcher owns 0.96 of its variance) |
| xalancbmk | 1.3 | -0.1 | 1.1 | - | **drop** (confirmed) |
| gcc | 69.5 | -0.7 | **69.5** | - | **reject** |

`xalancbmk` is dropped as planned. **`gcc` is rejected too, for the opposite reason:** its 43 MB footprint over a 10M-instruction window makes 100% of its 70 MPKI compulsory, and its curve is flat at 0.187 from 16 KB to 32 MB - no cache on any chip changes its miss count. (It has the highest stride regularity of any trace at 0.79, so it would be a strong *prefetcher* workload, but we already have lbm and the geomean would be swamped by a workload no capacity decision can move.) **Open: we still need a second capacity-sensitive workload**, to be screened with this gate from the 14 remaining SPEC17 candidates before anything is downloaded in bulk.

10. **Window length is not the lever; the workload's own reuse structure is.** Measured (Sep 8, all free): `bfs.kron` is capacity-flat at 10M, 30M and 45M windows alike (movable 0.01 / -0.00 / 0.00) because its footprint grows as fast as the window. Screening 4 GAP traces then showed the pattern is the **graph input**, not the kernel and not the window:

| trace | footprint | LLC MPKI | movable | verdict |
|---|---|---|---|---|
| `bfs.urand-36B` | 22.4 MB | 87.5 | **33.6** | **ADMIT** - 2.7x mcf's capacity headroom |
| `bfs.kron-128B` | 10.7 MB | 17.1 | 0.01 | reject |
| `pr.web-16B` | 8.1 MB | 12.8 | 0.20 | reject |
| `bfs.road-99B` | 4.3 MB | 6.9 | 0.00 | reject |

Explanation: in power-law (kron, twitter, web) and road graphs the reused hot set is small and already fits at 576 KB while the rest streams past once, so extra capacity buys nothing. In a **uniform-random** graph the reuse is spread evenly over the whole vertex array, so every extra megabyte captures proportionally more. **The held-out suite is therefore built from `urand` GAP traces.** Also screened and rejected as a 4th training workload: `perlbench` (76 KB footprint), `leela` (172 KB), `deepsjeng` (238 KB), `xz` (1.6 MB, movable 0.42 - the closest). Six SPEC17 traces profiled in total and only **mcf (7.5 MB) and omnetpp (5.6 MB)** are capacity-sensitive, which is our own measurement of ArchAgent's claim that SPEC is unrepresentative.

**Paper obligation:** held-out workloads are selected by this pre-registered, simulator-free gate, and both the gate and every rejection are reported. Selecting workloads where *no* design matters would measure nothing, and ArchAgent filters the same way (LLC MPKI > 1) - but ours must be stated openly, because choosing the held-out set is otherwise the easiest place to fool yourself.
11. **Held-out workload class:** GAP ChampSim traces (bfs, pr, cc) from Zenodo record 20043527 (BSC/UPC/TAMU, May 2026). Cited for exactly our gap: the trace set exists because replacement policy matters on big-data workloads and not on SPEC, which is why our `llc_replacement` knob explains only 0.2-11% of variance today.
12. **Multi-core is an extra cell, not the backbone** (assessment delegated to Claude, Sep 8): a 4-core shared-LLC chip is the most realistic setting and matches the CRC-2 configs ArchAgent uses, but it costs ~4x per design, and the only way to afford making the whole experiment multi-core is dropping from 5 seeds to 2-3. Yazdanbakhsh will read variance across seeds first, so that trade is the one this committee punishes. Main experiment stays single-core at 5 seeds; one added cell runs `rules` vs `bo_pooled` at 3 seeds on the 4-core chip. Two risks stated in the paper: a mix has no single footprint (hence decision on mix descriptors above), and footprint theory predicts a *dedicated* cache's miss ratio, so the miss-ratio curve is only an approximation on a shared LLC.
13. **Chips are not made structurally different.** The block-size trick was rejected: the miss-ratio curve is measured in 64-byte blocks, so per-chip block sizes would destroy the profile's chip-independence, which is a founding principle. Verified that the chip axis is nonetheless non-trivial: A's best design ranks 314th of 447 on C, B's best 181st, and C's best does not even fit A's or B's area budget. Stated limitation: **our chips differ in provisioning (core width, MSHRs, DRAM, area budget, core count on chip D), not in hierarchy shape.**
14. **We pre-register our own experimental predictions** before the paid run and Brier-score ourselves on them. See the paid-run section.

## Evidence gathered Sep 8 (all offline, zero spend)

**Offline transfer test** (`train on every cached A+B per-workload row, predict every cached C row`; 810 train / 1,834 test rows; harness `loop/offline.py`, reproducible with `python -m loop.offline compare | fair | learned`).

**The morning table was wrong and is corrected below.** A pooled Spearman over all four workloads mostly rewards knowing *which* workload a row is (mcf's log speedups span -0.31..+0.51, xalancbmk's -0.83..+0.02), so it flattered every model. The two metrics that mean something are the rank **inside** one workload and the **suite objective** the loop actually optimises (geomean over the 4 workloads, 447 designs measured on all of them). Same protocol, same rows, corrected columns:

| surrogate (fit on A+B, zero designs seen on C) | pooled Spearman (misleading) | within-workload Spearman | suite Spearman | best real IPC in its top 5 |
|---|---|---|---|---|
| knob columns + soc one-hot (today's `bo_pooled`) | 0.30 | 0.46 | 0.60 | 0.7204 (96.2%) |
| physics features | 0.86 | **0.31** | 0.22 | 0.7184 (95.9%) |
| minimal physics (design-vs-baseline miss deltas) | 0.87 | **0.56** | 0.60 | 0.7188 (96.0%) |
| CPI stack, curve-predicted miss counts | 0.46 | 0.47 | **0.72** | 0.7175 (95.8%) |
| CPI stack, learned miss counts (GP on A+B) | - | 0.45 | 0.47 | 0.7041 (94.0%) |
| *CPI stack, measured miss counts (oracle, not achievable in the loop)* | 0.46 | 0.54 | 0.75 | **0.7491 (100%)** |

Read it as: **G1 fails.** Every achievable model lands in the same 95.8-96.2% band on top-5, i.e. the CPI-stack mean as specified buys nothing over `bo_pooled`'s own GP. The G1 fallback ("minimal-physics GP as the mean") rests on the inflated 0.86; its real within-workload rank is 0.56. What *is* established: with exact miss counts the same 5-coefficient formula puts the true optimum in its top 5 from zero observations on C, so the whole mechanism reduces to one well-posed sub-problem - **predict per-level miss counts accurately** (the learned model reaches 34% relative error at the LLC; the stack multiplies that by ~180 DRAM cycles).

**Miss counts are chip-invariant, IPC is not** (36 designs simulated on both B and C): same design and workload, B -> C changes L1D/L2C/LLC MPKI by 1.8-4.2% (rank corr. 0.99+) while IPC moves 25.8%. So the problem factors: *how many misses* is a (program, design) property learnable on any chip, cheaply and without transfer; *what a miss costs* is a chip property a formula supplies. This is a better factorisation than either v6 extreme and is the next thing to test.

**Where the IPC variation lives** (Tier A full factorial, 81 designs, clean): capacity owns omnetpp (llc_sets 0.76 of variance) and half of mcf (0.41); the L2 prefetcher owns lbm (0.96); interactions are 2.5-23%. The split flips per workload, and the prefetcher's share grows with the chip (lbm l2_pref: A 0.34, B 0.84, C 0.96). This is the paper's motivation figure and the reason a capacity-only mean cannot carry a suite.

**A suite objective forgives model error** that per-workload accuracy condemns: per-workload errors partly cancel in the geomean, and what survives is common-mode, which does not change a ranking. This is why all four models were mis-ranked by the morning's metric.

**Interval model recovered from data:** fitting the stack's overlap fraction as `base + slope x stride_regular_fraction` returns base = 0.000, slope = 0.899 - a program with no regular strides gets no miss overlap at all, which is exactly the model's dependent-miss statement. It lifts mcf's within-workload rank from 0.73 to 0.84.

**Prefetchers hide misses, they do not remove them** (chip C, lbm): `spp_dev` cuts LLC misses 17% and raises IPC 34%; `ip_stride` cuts them 7% and raises IPC 2.7%. A miss-count model cannot see this. A fitted timeliness factor per prefetcher value recovers +0.05 suite Spearman and changes no top pick, so "physics for capacity, rules for policy" survives.

Held-out program (train A+B on 3 programs, predict C on the 4th): no model transfers (mean MAE ~0.17-0.18 for all; xalancbmk anti-correlated). Consistent with the literature (Concorde ISCA 2025 needs 2k-8k samples of a held-out program even with 29 training programs). Hence: physics for chips, not for unseen programs.

**Literature (Sep 8 scan; details in memory `research-sep8-ml-cache`).** Closest prior work: **AgentDSE** (MLArchSys @ ISCA 2026, arXiv 2606.21836): coding agent tunes a ChampSim cache hierarchy on mcf/lbm/omnetpp, ~50 sims vs 1,000 for BO; hypotheses as free text; no transfer, no scoring; its anonymization ablation showed the LLM is a capable black-box optimizer by itself. Also LUMINA (unverified LLM rules), MicroEvo (ICCAD 2026, refuses quantitative claims because of noise), ArchEval (agents' forecasts diverge from results). Learned cross-program transfer (MetaDSE, OneDSE, PerfVec, Concorde) needs 7-29 programs and 10^5 samples; RL for design selection loses to BO/GA at 10x our budget (ArchGym, OneDSE); deep trace surrogates need GPU-days. Mechanistic miss-cost models exist: interval model (TOCS 2009), Van den Steen et al. (ISPASS 2015 / IEEE TC 2016: profile once, predict CPI for any ROB/LLC/MSHR, 4-9% error), Aneto (MICRO 2026: blocking factor fit on 5-7 workloads). Nobody has simulator-settled, Brier-scored, pre-registered architectural claims; framing citations POPPER (ICML 2025), FOREAGENT (ACL 2026), Murphy 1973. Baseline fairness: budget-matched HPO study (2606.21641), MTBO pitfalls (2607.09073).

**Verdicts:** no RL, no deep learning, no learned workload embedding, no LLM-as-surrogate, no multi-fidelity search, no AlphaEvolve handoff, no Merlin arm, no cross-program claim.

## Roadmap with gates (Sep 8 -> Sep 24)

| when | do | gate |
|---|---|---|
| Sep 8 (done) | CPI-stack mean built (`loop/cpi_stack.py`) and tested offline (`loop/offline.py`, `loop/surrogates.py`). Reference verified. Offline evidence table corrected. Headline reset to workload-class transfer. | **G1 failed as specified**: suite top-5 reaches 95.8% of 0.7491 with predicted miss counts vs `bo_pooled`'s 96.2%; 100% with oracle counts. Formula sound, miss-count layer is the gap. Stack demoted to an offline finding. |
| Sep 8 (done) | Both free gates run. `gcc` profiled; GAP format checked from a 12 MB byte-range prefix of `bfs_traces.zip` (Zenodo serves HTTP 206, member is `bfs.kron-29B.champsimtrace.xz`, deflate inside the zip, xz inside that). Admission gate written and validated: `python -m loop.offline admit`. | **G-format: PASS.** 262,144 records parse under our 64-byte `input_instr` layout; `is_branch`/`branch_taken` strictly 0/1, branch rate 0.182, IPs in one hot loop at 0x404190-0x404273. Same naming convention as DPC3. Still to confirm by actually running ChampSim on a full trace. **G-MPKI: `gcc` FAILS** - see decision 9. |
| Sep 8 (done) | Screened 4 more SPEC17 candidates and 4 GAP traces with `loop.offline admit`, all free. **The zip-index trick:** a GAP kernel's zip is 10 GB but holds 31 separate traces (5 graph inputs x SimPoints) of 24-630 MB each; fetch the zip64 central directory from the last 256 KB, then byte-range exactly the member wanted. So one GAP trace costs ~150-600 MB, not 10 GB. | **G-window: the window is NOT the lever** - `bfs.kron` is capacity-flat at 10M, 30M and 45M alike, because its footprint grows as fast as the window. **What decides it is the graph input, not the kernel or the window.** See decision 9. |
| Sep 9 | Finish screening: remaining SPEC17 candidates (bwaves, fotonik3d, roms, cactuBSSN, wrf, imagick, nab, exchange2, x264) for a 4th training workload; more `urand` GAP traces for the held-out suite. Then the multi-core path. | a training suite of >= 3 admissible workloads and a held-out suite of >= 2 |
| Sep 9-10 | Multi-core path (~100 lines): `num_cores` per chip profile in `socs.py`; a trace **list** into `simulate.run_simulation` (ChampSim already requires exactly `NUM_CPUS` traces and reports per-core stats); per-core metric aggregation; area cap counts one L2 per core; mix descriptors (sum capacities, access-weighted rates). Chip D = 4-core CRC-2-shaped, shared LLC. | smoke: one 4-core design runs end to end and its per-core IPCs are sane |
| Sep 11 | Chores: fixed reference + 90/95/99% in `summarize`, unfinished-arm crash, frozen `bo_pooled` pool. Launch unattended collection: `gcc` rows on A/B/C, GAP rows on C, chip D rows (references for the new cells). | — |
| Sep 12-13 | Learn phase on A+B with the SPEC-class suite (mcf, lbm, omnetpp, gcc) -> **one frozen playbook**, used unchanged in every test cell. Smoke all cells. **Write the pre-registered predictions into the repo, timestamped, before any test cell runs.** | playbook has rules that speak on GAP descriptors at all (if every rule is mute on GAP, the descriptor language is the finding and the paper says so) |
| Sep 14 | Fidelity check (50M/50M on baseline + top-20 + 5 random). | **G0:** top-20 ranks hold at 50M warmup. Fail -> raise warmup, invalidate cache, re-plan budget. |
| Sep 15 | **Mechanism frozen.** Only measurement and writing after this. | — |
| Sep 16-17 | Test cell 1 (chip C, SPEC-class) and cell 2 (chip C, GAP), 6 arms x 5 seeds x 24 designs, same frozen playbook. Then the multi-core cell (chip D, GAP mixes, `rules` vs `bo_pooled`, 3 seeds). | **G-headline:** in cell 2, `rules`/`full` beat `bo_pooled` on designs-to-target, and degrade less than `bo_pooled` from cell 1 to cell 2. Fail -> report which conditions transferred and which did not; that still answers the judges' open question. |
| Sep 18 | 1B-instruction validation of the top ~5 designs per cell per key arm (ArchAgent's cascaded approach: search short, validate long). | designs that won at 10M still win at 1B |
| Sep 18-23 | Paper; CHIA node cleanup (the experiment must run through `run_chia.py` with profiles saved); upstream notes; dataset export. | submit (hard stop Sep 24 AoE) |

Budget: ~21 USD spent of ~257. Plan above ~59 USD: learn phase ~3, collection (gcc + GAP + chip D references) ~17, two test cells ~20, multi-core cell ~5, 1B validation ~14. ~37 h of VM time, mostly unattended. Compute is not the constraint; time and correctness are.

## Paid-run design and paper

**Three test cells, one frozen playbook, learned on chips A+B with the SPEC-class suite (mcf, lbm, omnetpp, gcc).** The playbook is never re-learned or re-tuned between cells; that is what makes the comparison mean anything.

| cell | chip | workloads | what it isolates | arms x seeds |
|---|---|---|---|---|
| 1 | C (unseen) | SPEC-class | new chip only | 6 x 5 |
| 2 | C (unseen) | **GAP** (bfs, pr, cc) | new chip **and** new workload class - the headline | 6 x 5 |
| 3 | D (unseen, **4-core shared LLC**) | GAP mixes | does it also survive a core-count change | 2 x 3 |

**Arms** (same budget, same baseline start, same seed set): `random`, `bo` (cold GP), `bo_pooled` (frozen A/B pool - the data-transfer baseline), `textbook` (LLM rules written before any result), `rules` (verified playbook as bounded categorical priors, no LLM at test time), `full` (+ analyst). The `physics` arm is dropped: G1 failed, so it would only be a worse GP. ChampSim is deterministic; variance comes from the optimizer and, for `rules`/`full`, from the analyst (temperature 0.3/0.7).

**Pre-registered predictions - written into the repo, timestamped, before cell 2 runs.** Applying the mechanism to ourselves; no paper in this space does it, and it is the cheapest credibility we can buy.

- **P1:** `bo_pooled` degrades sharply from cell 1 to cell 2, because its warm-start pool is SPEC rows. (Grounded: our held-out-program offline test showed a learned surrogate does not transfer to an unseen program.)
- **P2:** `rules` degrades markedly less than `bo_pooled`, because its conditions read measurable habits rather than program identity.
- **P3:** `textbook` is mediocre in both cells and degrades least of the three, having never been fitted to anything.

Report whether each held, with the numbers, and Brier-score ourselves alongside the surrogate, the rules and the analyst.

**Report:** best-so-far vs designs with bootstrap CIs over seeds (the plot Yazdanbakhsh reads first); censored medians to 90/95/99% of each cell's own reference; hit fraction at 12 and 24; final regret; Brier and reliability diagrams per forecaster (Murphy decomposition); **descriptor-range table** (training workloads' descriptor ranges vs GAP's, so the reader sees exactly which conditions are interpolating and which are extrapolating - this is the rigor move that pre-empts "you got lucky"); rule ledger on the held-out class (which rules spoke, win rate, Brier, SPEC vs GAP); corrected offline surrogate table with the CPI-stack oracle row; playbook excerpt with records; 1B-validation table.

**Motivation section** = the Sep 7 autopsies in three sentences (exploitation collapse, marginal-vs-interaction effects, simulator-relative thresholds), plus the variance decomposition (capacity owns omnetpp, the prefetcher owns lbm at 0.96, and the split flips per workload).

**What we can claim that ArchAgent cannot,** stated in the intro: (i) knowledge carried from old chips to a new one at all - they start every search from scratch; (ii) hypotheses pre-registered and settled by the simulator with a calibration score - theirs are free text, never scored; (iii) a direct test of extrapolation to an unseen workload class - they name it as an open problem; (iv) validation at competition length via their own cascaded approach. We do **not** compete on IPC: their margins are 1-3% because they design new policies, while we search a knob space, so the numbers are not comparable and we say so.

**Limitations, plainly:** chips differ in provisioning, not in hierarchy shape (block size and level count fixed; only chip D changes core count); footprint theory predicts a *dedicated* cache's miss ratio, so the miss-ratio curve is an approximation on chip D's shared LLC; 2-3 GAP kernels, not the full suite; placeholder SoC profiles, never team-reviewed; L1D miss-ratio prediction ~1.5-2x low; each cell's reference is the best cached design, not a true optimum; cell 3 has 3 seeds and 2 arms; warmup (with the fidelity result); no cross-program claim for the *surrogate* - only for the rule language.

## Repo layout

See `README.md` for the file table. `loop/loop.py` is simulator-agnostic; `loop/champsim_problem.py` is the only ChampSim glue; `loop/chia_nodes.py` + `loop/run_chia.py` run it as a CHIA loop (116 lines; **not yet exercised on Tier C**, the runs used `loop/run.py`'s own runner); `loop/collect.py` (new, Sep 8) simulates a fixed design list per chip on new traces; `loop/cpi_stack.py` (new, Sep 8) is the CPI-stack mean function (5 global coefficients, chip parameters read from the SoC profile so an unseen chip needs no work); `loop/offline.py` + `loop/surrogates.py` (new, Sep 8) are the zero-cost offline harness and the candidate feature sets; `loop/socs.py` / `configs.py` hold **placeholder** SoC profiles and search spaces; `proposal.tex` is the accepted proposal (its "line size" knob was never in any search space: ChampSim's block size is global); `paper/main.tex` the paper skeleton (must be rewritten to the v7 story). **To build (Sep 9-10):** multi-core path (`num_cores` in `socs.py`, trace list into `simulate.run_simulation`, per-core metric aggregation, per-core L2 in the area cap, mix descriptors).

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
./config.sh champsim_config.json && make -j8 && cd ..
git clone --depth 1 https://github.com/ucb-bar/chia.git chia
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python -e ./chia google-genai
mkdir traces && curl -o traces/605.mcf_s-665B.champsimtrace.xz \
  https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/605.mcf_s-665B.champsimtrace.xz
```

**GAP traces (the held-out workload class), Sep 9, on the VM only** — Zenodo record 20043527 (BSC/UPC/TAMU, May 2026; 46.4 GB for six kernels, so fetch only bfs/pr/cc, ~10 GB each). Cite *"Characterizing the impact of last-level cache replacement policies on big-data workloads"*, IISWC 2020. Verify the 64-byte `input_instr` record format before relying on them.

GCP: project `project-c23a6080-f5d0-4871-9cb`, ADC auth, `aiplatform.googleapis.com` enabled; project-wide cap 32 vCPUs (`CPUS_ALL_REGIONS`; the regional C2D quota of 100 is not binding); VM `champsim-1` (c2d-standard-32, europe-west4-a, ~1.5 USD/h) is STOPPED; recipe in `cluster/README.md`. `ANALYST_MODEL=gemini-2.5-pro` is the only Pro the project can call (default `gemini-2.5-flash`). 14 candidate SPEC17 traces for later expansion are listed in the DPC3 directory (gcc, x264, deepsjeng, leela, bwaves, fotonik3d, roms, xz, cactuBSSN, wrf, imagick, nab, perlbench, exchange2); profiling costs ~2 s per trace, downloading ~200 MB each.

## Results so far (read with the caveats)

- **Tier C v5 / v5.1, seed 0, Pro analyst** (`results/experiment_tierC_v5*_C.json` + `_playbook.json`): designs to 90% of gain, reference = best in that run (0.749 / 0.745): random >24 / >24; bo >24 / 18; bo_pooled **6** / >24; rules 11 / >24; full 9 / **8**. Final best: full 0.749 / 0.745. Caveats: one seed; run-dependent reference; `bo_pooled` pool differed between the runs (30 vs 20 learn designs). v5.1 `rules` collapsed from 5 claim tests in rounds 1-5 plus 8 rule priors pulling EI into the SPP-only basin; `full` escaped via the analyst's reply. Calibration on shared bets: rules Brier 0.21-0.27, surrogate 0.29-0.31, hypotheses 0.28-0.29 (win rates 0.37-0.42).
- **Tier B (46k designs, single-workload, mechanism v4.1, Sep 5):** median over 9 runs: full 8 (9/9 hit), rules 6 (8/9), bo 15, bo_pooled 15, random 19 (`results/experiment_v4*.json`). Chip-relative conditions; a Tier B data point, not evidence for v6.
- **Cached simulation tables** (`results/tierC_*.json`): 2,644 per-workload rows (A ~430, B ~380, C ~1,830 over mcf, lbm, omnetpp, xalancbmk); Tier A dense sweeps `results/sweep_*.json` (81 designs x 3 chips x 3 traces); profiles `results/profile_*.json`. These are the dataset artifact and the offline harness's input.

## History (condensed)

- **v2 (Sep 3):** disagreement selection, flat-confidence rules, claims never settled. **v3 (Sep 4):** controlled-pair verification, every bet counts; found that a rule's gain is a marginal effect at the baseline. **v4/v4.1 (Sep 5):** EI with rule/hypothesis priors, owed claim tests, direction claims, analyst credibility; Tier B result. **Sep 7 autopsies (Tier C):** exploitation collapse from priors; thin playbook; untestable capacity rules; analyst deadlock at credibility 0.5; sign contradictions; area coupling (bigger L2 makes the decisive LLC step infeasible); chip-relative thresholds silenced the good rules. Fixes: stall scan, right of reply, adjacent-pair verification, `direction` field, rewritten prompts, Pro analyst, **v5 trace-profile descriptors**, **v5.1 thresholds fitted from per-workload evidence**. **Sep 8:** honest review (one seed, pooled BO ties, CHIA integration thin, SoC profiles unreviewed); offline test shows physics features carry rank 0.86 across chips but nothing transfers to an unseen program from 3 programs; literature scan; external review adds fixed reference, frozen pool, fidelity check, CPI-stack as GP mean; decisions above.

## Problem scope: real problems are hard problems

Per cache level: size, associativity, replacement, prefetcher, MSHR/queue depths (block size is global in ChampSim and not a knob); plus coherence, interconnect, DRAM. Knobs interact through the area constraint. One detailed simulation costs hours per design per workload in industry; teams evaluate tens to low hundreds of designs per cycle across 20-50 workloads. Every new SoC starts from the previous one: chips A, B -> unseen C *is* the industrial loop. Hardness = wide space, sparse near-optimal band, large headroom, interacting knobs; Tier A (54 feasible) is a unit test, Tier C (415k designs, latency coupled to size, MSHRs, 4-workload suite, chip C baseline 0.599, best cached 0.7491) is the target. Baselines must scale with the problem, never be weakened.

## Working rules

- Budget ~257 EUR total GCP credits; ~21 USD spent. Every launch needs the user's explicit OK with cost and time. Nothing runs on the Mac (reserved). Stop the VM when idle.
- Code style: explicit, procedural, junior-readable; no clever one-liners; no flags for dead features, remove them. Commit messages informal, no authorship trailers.
- Domain decisions (search space, SoC profiles, descriptors, arms, prompt content, objective) belong to the user and their PhD teammates: surface options with tradeoffs, recommend, let them pick.
- Offline first: anything testable on the cached rows is tested there before a simulation is bought.

## How to read results and run things

`python -m loop.summarize <report.json> [more.json]` (censored medians, hit fraction, final best, AUC; to be fixed for the 0.7491 reference and unfinished arms); `python -m loop.early <report.json> <log>` while a run is in progress; `python -m loop.run C <tag>` (env `SEEDS`, `FIRST_SEED`, `ANALYST_MODEL`); `python -m loop.collect <designs_per_chip> <traces...>` to simulate a fixed design list on new traces; `python -m loop.trace_profile <trace.xz>` for one profile.

**Operational lessons.** Launch long runs detached on the VM (`setsid nohup ... &`); kill by process group, never `pkill -f` a broad pattern (it killed 4 live runs on Sep 5); result tables are the shared simulation cache, atomic-written, one lock per file; suites evaluate one design on 4 workloads at once; the process pool forks, so editing files under a running job is safe but the job keeps the old code. Measured throughput on c2d-standard-32: ~380 simulations/hour including builds (not 1 CPU-minute each).

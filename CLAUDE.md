# CHIA Hackathon — Rules that Bet (cross-SoC cache-hierarchy design)

A3 workshop hackathon (agentic-arch.org). **Real deadline Sep 24, 2026 AoE** (internal target Sep 20; Sep 21-24 buffer). Deliverable: 4-page paper + open-sourced CHIA loop with reproducible results; "reusable CHIA blocks upstreamed to mainline" is an explicit track. Judged by the A3 program committee, run by the six organizers: Gonzalez and Jain (Google, ArchAgent authors), Karandikar (UC Berkeley, CHIA PI), Yazdanbakhsh (Google DeepMind, ArchGym: "all optimizers tie under tuned hyperparameters"), Huang (NVIDIA, DOSA), Skarlatos (CMU). They reward novelty and quality of the agentic architecture, matched-budget baselines, seeds, realizability (they flagged ChampSim's lack of area/power feedback themselves), and reusable blocks + datasets. Not raw IPC.

## The mechanism we are building (v6, decided Sep 8)

One sentence: **the workload's physics (miss-ratio curve from the trace) times the chip's physics (a CPI stack) is the surrogate's mean; the LLM proposes what the physics is blind to, every proposal is a bet the simulator settles, and the scored rules transfer to the next chip.**

- **Workload profile, once per trace, no simulator** (`loop/trace_profile.py`): working set, stride regularity, temporal locality, LRU miss-ratio curve (footprint theory, Xiang et al. ASPLOS 2013). Chip-independent by construction.
- **CPI-stack mean function** (to build): per-level misses from the miss-ratio curve at the design's capacities x the chip's latency-from-size x an overlap factor from the interval model (Eyerman, Eeckhout, Karkhanis, Smith, TOCS 2009: independent LLC misses within one ROB window overlap and are paid once, capped by MSHRs; dependent misses never overlap). Structure is fixed physics; a handful of coefficients are fit on A+B and refit online on C. The GP models only the **residual**. No virtual points from rules or hypotheses in the GP (this removes the exploitation trap and the claim-test flood of v5.1 in one move).
- **Rules act only on what the physics cannot see: prefetcher, replacement, interaction terms.** A speaking rule shifts the GP mean for its categorical value by at most (measured effect x credibility); the shift decays as real observations on C accumulate. Soft and bounded. **Never a filter** (filters lose the optimum and are indefensible to this committee). Rules still bet on every design they speak about; bets are free. Dedicated one-knob claim tests only when EI brought no improvement last round.
- **The LLM has three jobs, each ablatable:** (i) propose terms the additive stack lacks (e.g. SPP x LLC capacity, prefetcher x stride regularity), admitted only if they improve a leave-one-chip-out fit on the cached rows, rejected ones logged; (ii) write conditions/rules from the fitted model and the measured one-knob claims; (iii) right of reply when EI stalls. The LLM never judges its own claims.
- **Conditions are chip-independent** (profile descriptors read against the chip's geometry: `champsim_problem.chip_descriptors`). **Claims are measured, never stated** (controlled pair, |effect| >= 1%, sign matches the LLM's stated direction; rejections kept in `rejected_rules`). Losing rules are re-scoped, not deleted. Brier-scored ledger for every forecaster (surrogate, rules, analyst).
- **Objective:** IPC under a **CACTI area cap** (CHIA's `run_cacti`, replaces the KB placeholder budget; cached rows stay valid). Energy (CACTI per-access energy x ChampSim access counts + DRAM traffic) as a secondary Pareto plot.
- **Guardrails:** rebuild per config, budget re-check, fresh-build re-run of the final best design, skeptic pass that tries to falsify the accepted hypothesis.

## Decisions taken Sep 8 (user)

1. Headline package: cross-**chip** transfer with a physics prior + measured, Brier-scored claims + CACTI realism. **No cross-program transfer claim** (evidence below: it needs 7-30 training programs; report the negative held-out-program column as a limitation).
2. Rules in selection: **categorical priors, soft and bounded**, as above. Not filters, not bets-only.
3. **Fidelity check is the first spend** (baseline + top-20 cached + ~5 random designs at 50M warmup / 50M sim, 4 workloads, ~25 CPU-h, ~1 h wall on 32 vCPUs, a few USD, then stop the VM). Every downstream number depends on whether big-LLC ranks hold under a real warmup.
4. `textbook` arm stays (LLM prior knowledge alone, no LLM at test time): answers "isn't the LLM's prior doing the work?" and gives the thesis figure: stated rules' Brier vs measured rules' Brier.
5. Fixed reference for Tier C: **0.7491**, the best suite geomean among the 447 C designs measured on all 4 workloads in the cached table. Report designs to 90/95/99%.
6. Freeze one A/B warm-start pool for `bo_pooled`, identical across versions and seeds (today the pool is that run's learn-phase history, so the baseline inherited the treatment's choices: v5 30 designs -> 6 to target, v5.1 20 designs -> >24).
7. Freeze the mechanism after Sep 12; after that only measurement and writing. gem5 port / second field only if everything else is done (it will not be).
8. Spending: each launch needs the user's explicit OK with cost and time; VM stopped when idle.

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

## Roadmap with gates (Sep 8 -> Sep 20)

| when | do | gate |
|---|---|---|
| Sep 8 | CLAUDE.md consolidated (this). Then: fixed reference in `summarize` (0.7491, 90/95/99%), `summarize` crash on unfinished arms fixed, frozen `bo_pooled` pool, line size removed from prose, fidelity-check job written (~40 lines: design list at 50M/50M into a separate table). **Launch fidelity check** (user presses; then stop VM). | — |
| Sep 8 (done) | CPI-stack mean built (`loop/cpi_stack.py`) and tested offline (`loop/offline.py`: `ceiling`, `timeliness`, `misses`, `compare`, `fair`, `learned`). | **G1 FAILED as specified:** suite top-5 reaches 95.8% of 0.7491 with curve-predicted miss counts, no better than `bo_pooled`'s 96.2%. Oracle miss counts reach 100%, so the formula is sound and the miss-count layer is the whole gap. Gate restated in suite terms (top-5 within 2% of 0.7491); the old "Spearman >= 0.86" was measuring between-workload spread. |
| Sep 9-10 | The one open sub-problem: **per-level miss counts to a few percent**, LLC first. Chip-invariance (above) means it trains on any chip's rows and needs no transfer, and Tier A/B rows are cheap. Prefetch coverage as a function of the profile is the missing term and is exactly what the LLM term proposer should be asked for (leave-one-chip-out admission, ledgered). Then re-run `python -m loop.offline learned`. | **G1':** suite top-5 within 2% of 0.7491 with predicted miss counts. Fail -> drop the CPI-stack mean, keep the trace profile as GP features (real within-workload rank 0.56), and the headline becomes calibration, not model transfer. |
| Sep 10 | Read fidelity result. | **G0:** top-20 ranks hold at 50M warmup. Fail -> raise warmup, invalidate cache, re-plan budget. |
| Sep 11-12 | Residual GP with physics mean; categorical rule priors (bounded, decaying); claim tests only on stall; `physics` arm; smoke Tier B; Tier C seed 0 (~3 USD + ~1 USD Pro). Mechanism frozen after this. | **G2:** `physics`/`rules`/`full` <= `bo_pooled` <= `bo` on designs to target, curves separate. Fail -> one day autopsy, run anyway, write what happened. |
| Sep 13 | CACTI node (area + energy per config); guardrails (fresh-build re-run of final best, skeptic pass). | — |
| Sep 14-15 | **Paid run:** Tier C, 7 arms x 5 seeds x 24 designs x 4 workloads ≈ 3,400 sims ≈ 9 h wall on 32 vCPUs at the measured ~380 sims/h, ~14 USD + ~5 USD Pro. Optional unattended 3-5k-design Tier C sweep (~10 h, ~12 USD) for the dataset. Export dataset: Tier A sweeps + Tier C tables + profiles + offline harness. | — |
| Sep 16-20 | Paper; CHIA node cleanup (experiment must run through `run_chia.py` with profiles saved); upstream notes. | submit (hard stop Sep 24) |

Budget: ~21 USD spent of ~280; the plan above is ~40-60 USD. Compute is not the constraint; time and correctness are.

## Paid-run design and paper

**Arms** (same budget, same baseline start, same seed set): `random`, `bo` (cold GP), `bo_pooled` (frozen A/B pool), `textbook` (LLM rules before any result), `physics` (CPI-stack mean fit on A/B, refit online on C, no LLM), `rules` (physics + verified playbook as categorical priors), `full` (+ analyst). ChampSim is deterministic; variance comes from the optimizer and, for `rules`/`full`, from the analyst (temperature 0.3/0.7). 5 seeds minimum.

**Report:** best-so-far vs designs with bootstrap CIs over seeds (the plot Yazdanbakhsh reads first); censored medians to 90/95/99% of 0.7491; hit fraction at 12 and 24; final regret; Brier and reliability diagrams per forecaster (Murphy decomposition); offline transfer table (knob-GP 0.30 / profile-GP 0.86 / CPI-stack / + LLM terms); playbook excerpt with records; CACTI Pareto if it fits.

**Headline, pick after G1/G2:** if `physics`/`rules` clearly beat `bo_pooled`: "Model transfer beats data transfer: workload physics x chip geometry as a prior for cross-SoC cache design." If they tie but `full` wins or calibration is the strong result: "Rules that bet: simulator-settled, Brier-scored architectural knowledge that survives a chip change." If `full` ≈ `physics`, write that sentence (AgentDSE earned credibility doing exactly this).

**Motivation section** = the Sep 7 autopsies in three sentences (exploitation collapse, marginal-vs-interaction effects, simulator-relative thresholds). **Limitations, plainly:** placeholder SoC profiles (never team-reviewed); L1D miss-ratio prediction ~2x low; chip transfer only, programs fixed; warmup (with the fidelity result); 5 seeds; reference = best cached design, not a true optimum.

## Repo layout

See `README.md` for the file table. `loop/loop.py` is simulator-agnostic; `loop/champsim_problem.py` is the only ChampSim glue; `loop/chia_nodes.py` + `loop/run_chia.py` run it as a CHIA loop (116 lines; **not yet exercised on Tier C**, the runs used `loop/run.py`'s own runner); `loop/collect.py` (new, Sep 8) simulates a fixed design list per chip on new traces; `loop/cpi_stack.py` (new, Sep 8) is the CPI-stack mean function (5 global coefficients, chip parameters read from the SoC profile so an unseen chip needs no work); `loop/offline.py` + `loop/surrogates.py` (new, Sep 8) are the zero-cost offline harness and the candidate feature sets; `loop/socs.py` / `configs.py` hold **placeholder** SoC profiles and search spaces; `proposal.tex` is the accepted proposal (its "line size" knob was never in any search space: ChampSim's block size is global); `paper/main.tex` the paper skeleton (must be rewritten to the v6 story).

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

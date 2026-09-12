# CHIA Hackathon: few-shot cache tuning from a memory of earlier searches

A3 workshop hackathon (agentic-arch.org). Deliverable: 4-page paper + open-sourced CHIA loop with reproducible results; "reusable CHIA blocks upstreamed to mainline" is an explicit track. Judged by the A3 program committee: Gonzalez and Jain (Google, ArchAgent authors), Karandikar (UC Berkeley, CHIA PI), Yazdanbakhsh (Google DeepMind, ArchGym: "all optimizers tie under tuned hyperparameters"), Huang (NVIDIA, DOSA), Skarlatos (CMU). They reward novelty and quality of the agentic architecture, matched-budget baselines, seeds, realizability, and reusable blocks + datasets. Not raw IPC.

**The open problem we answer.** ArchAgent (Gupta, Jain, Gonzalez et al., arXiv 2602.22425; four of the six organizers are authors) has an agent write cache-replacement policies inside ChampSim. Its authors state that it starts every search from scratch and that "extrapolation from representative workloads" is an open community problem. We test exactly that: an agent that remembers earlier searches on SPEC and graph workloads is handed Google datacenter traces it has never seen and a budget of 16 simulated designs, of which only the first few carry the claim.

## Start here (next session)

1. **Cell `dc` is done.** `python -m loop.summarize results/run_dc_d3.json results/run_dc_d3b.json` reproduces this table exactly.

```
share of the stock-to-best-known gap, mean over seeds (stock 0.3837, best known 0.4933, +28.6%)
arm          seeds    D1    D2    D4    D6    D8   D12   D16     final
bo               5   43%   43%   50%   58%   76%   84%   86%    0.4778
llm_direct       5   54%   76%   80%   81%   84%   87%   88%    0.4802
memory           4   90%   91%   94%   95%   95%   97%   98%    0.4910
spread at D16 (min..max over seeds): bo 51..98, llm_direct 84..95, memory 96..99
```

**Reading.** (a) The memory arm's **first** design is at 90% of the gap, above where either baseline ends after sixteen: that is the few-shot claim, and neither baseline closes it inside the budget. (b) It is the only arm past 90%, and it passes 95% by design 6. (c) It removes the variance (the spreads above), which is the practical argument for a design team: a fixed budget becomes predictable. (d) Internal evidence, not in the reported table: simulating the handed-over design alone, with no search, reaches 93% and stops there, and the loop passes it by design 3 — the loop is worth more than retrieval alone. (e) Random sampling reaches 90% after 219 designs and plateaus, matching what either baseline reaches in 16 designs after about 12: at this budget neither baseline extracts much more than chance. (f) The memory arm has 4 seeds, not 5: its seed 0 died on CHIA's 16k Vertex output cap in both attempts (the upstream gap below). The other four seeds are shared with the baselines. (g) The best design known on this suite, 0.4933, was not found by any of the three arms; their best single runs are bo 0.4906, llm_direct 0.4879, memory 0.4926.

Cell `dc2`, the same loop and the same memory on the three admitted datacenter traces the first suite did not take (whiskey, bravo, delta): nothing about them shaped the memory, the design it hands over, or the choice of the first suite, and that design had never been simulated on them. `python -m loop.summarize results/run_dc2_e1.json` reproduces it.

```
share of the stock-to-best-known gap, mean over seeds (stock 0.4389, best known 0.5463, +24.5%)
arm            D1    D2    D4    D6    D8   D12   D16     final
bo            43%   43%   50%   57%   58%   69%   83%    0.5276
llm_direct    26%   54%   72%   74%   76%   79%   85%    0.5305
memory        81%   82%   88%   93%   94%   96%   97%    0.5428
spread at D16 (min..max over seeds): bo 63..94, llm_direct 78..91, memory 92..100
```

**Reading `dc2`.** All five seeds of all three arms completed, no failed runs. The memory arm opens at 81%, again above where either baseline ends after sixteen, and its seed 3 found 0.5463 — **the best design known on this suite**, which no arm managed on `dc`. The plain LLM's first design is worse than BO's here (26% against 43%): with no memory, reading the descriptors alone is not reliably better than a GP's first guess. Caveat: this suite's ceiling rests on 300 designs against `dc`'s 713, and a reference search is still running on it, so 0.5463 can move up and every share with it.

2. **A tuned BO baseline, measured (scratch code, not in the repo: `bo_tune.py` and `bo_tuned_run.py` in the session scratchpad).** Our `bo` arm was textbook, not tuned. An offline replay against the uniform sample (8 seeds, candidate pool = the 300 random designs, so a fair proxy) says its settings are poor: batch 2 with no warm-up reaches 55% of that pool's gap at D16, where plain random search reaches 87%. The whole effect is the warm-up: with one measured design the GP has nothing to fit and its first picks are near-arbitrary.

   **Tuned configuration** (hyperparameters only; no memory, no warm start from cached tables): **batch 1** so every pick is made against measured outcomes, **expected improvement** (logEI is a monotone transform and picks identically, keep it only against underflow; UCB and greedy are worse; `honest_std=True` is worse), and a **random warm-up** before the GP chooses. Offline, warm-up 8 is best at a 16-design budget (96%); the two 50-design runs below used 5.

   **Result, one seed, 50 designs each, about 3-4 h per suite on the Mac:**

```
share of the stock-to-best-known gap, tuned BO, one seed
suite                D1    D4    D8   D12   D16   D20   D24   D32   D40    designs to 90/95/100%
dc  sierra/merced    43%   76%   76%   76%   77%   77%   83%   93%  100%    28 / 37 / 47
dc2 whiskey/bravo    65%   69%   69%   84%   84%   96%  100%  100%  100%    18 / 20 / 24
```

   **Reading.** (a) At the 16-design budget the tuning changes nothing decisive: 77% and 84% against the untuned arm's 86% and 81%, one seed against five, inside the noise. (b) Past the budget it clearly works: a tuned BO does reach the ceiling on both suites, needing about 40 designs on `dc` and 24 on `dc2` against the memory arm's 16. That ratio, roughly 1.5-2.5x the simulations, is the honest form of the claim. (c) The tuned run on `dc2` found 0.5485, above the memory arm's 0.5463, so **the `dc2` reference moved and its table above is scored against 0.5485**. (d) **Owed before quoting any margin over BO: re-run the `bo` arm at 16 designs, 5 seeds, tuned (batch 1, EI, warm-up 8), about an hour on the VM.** The margin over `llm_direct` is unaffected by all of this.

3. **The memory is rebuilt, and the agent-built one is now the default** (Sep 13). The 1918 designs that had accumulated per-workload over the project's history are gone from the tree (archived, gitignored); `loop/memory_build.py` rebuilt the shelf from scratch at ~122 designs per workload, twice, differing only in who searched in stage 1. Both memories are kept: `results/memory_llm.json` is what every cell now reads, `results/memory_bo.json` is the ablation.

```
                       designs/workload  facts  sign survival   handover on dc / dc2
   agent-searched (default)       122     336       69%            95% / 93%
   optimiser-searched             122     306       64%            89% / 81%
   the old accumulated one       ~950     290       64%            93% / 80%
```

   **Reading.** (a) At the same budget the LLM's search leaves behind **36** single-knob effects with the two controlled pairs a transferable claim needs; the GP's leaves **0**, because expected improvement picks scattered points and two points in 13 dimensions almost never differ in one knob. (b) The advantage survives the anchor sweeps that were meant to erase it. (c) The handover columns are the designs simulated on the held-out suites, one ruler, no search: the agent-built memory wins both and by 13 points on `dc2`, which shaped nothing. (d) Its handover reaches 0.4883 on `dc`, above the best 16-design run of the memoryless LLM arm (0.4879). (e) Pooling the two memories is worse than the better one alone (survival 67%), so they are not merged. (f) Each memory's internal `share` is against its own ceiling and is NOT comparable across memories — only the handover columns are.

   **Owed:** the arms have never been run *on* the agent-built memory. The published openings (90% on `dc`, 81% on `dc2`) came from the old accumulated memory's handover.

4. **Next, in this order.**

   **(4a) A repeatable process for building the memory — DONE**, see item 3 above. `loop/memory_build.py` is the block; the agent-built shelf is the default.

   **(4b) A fourth arm: BO started from the memory.** The same GP + expected improvement as `bo`, but its first design is the one the memory hands over instead of the stock chip; same budget, seeds and cells (`bo.run_bo` already takes a start design, so the change is small). It is the sharpest baseline this committee will ask for (ArchGym: "all optimizers tie under tuned hyperparameters"), and it is the arm that decides whether the LLM is load-bearing: if `memory` stays ahead, the LLM earns its place; if this arm ties or wins, the honest headline becomes "transfer works, the LLM is not what makes it work" and (4a) becomes the contribution.

   **(4c) Tune the BO baseline until it is a fair comparison.** `bo` runs at settings nobody tuned: two designs a round from a 20k uniform sample of the space plus the incumbent's one- and two-knob neighbours; expected improvement, batched by the kriging believer; Matern 5/2 with one length scale per column over log2-rescaled ordinals and one-hot categoricals; ConstantKernel * Matern + WhiteKernel, `normalize_y`, three restarts; and **no initial design at all** — it starts from the stock chip and its first round is already EI on a one-point GP, which is the likeliest reason its D1 is weak (43% on both cells) and its seed spread is enormous (51..98 on `dc`). Beating an untuned optimizer proves nothing to this committee, and Yazdanbakhsh is on it (ArchGym: "all optimizers tie under tuned hyperparameters"): until `bo` is tuned, the headline is not a fair comparison and a reviewer can say so in one sentence. Four knobs to sweep, the first two the most likely to matter: **initialisation** (a Latin-hypercube or Sobol design of the first 2-4 points instead of a cold start from stock); **batch size** (2 per round is a choice, not a default — 1 is more adaptive at the same budget, 4 wastes less wall time); **acquisition** (EI against log expected improvement, upper confidence bound, or Thompson sampling, and whether the kriging-believer batching is what hurts); **kernel** (Matern 5/2 against 3/2 and against an ARD RBF, the length-scale bounds `(5e-2, 5.0)`, and whether one-hot is the right encoding for prefetcher and replacement). Cheap to do honestly: the tuning sweep runs **offline on the cached tables** of a cell whose designs are already measured, scoring each configuration by the same gap-share-after-N-designs curve, with no new simulation — only the winning configuration is then re-run as the reported arm. Do this before (4b), since a BO-from-memory arm built on an untuned GP inherits the same objection. If a tuned `bo` closes the gap, that is the result and it must be reported as such.

   **The priority inside (3c), and the thing that is actually decided: one tuned surrogate, shared by every arm.** The `memory` arm already contains a GP — the LLM proposes 8 and a GP fit on this run picks 2 — so the sweep is not a gift to the baseline, it is a component upgrade our own arm collects too. The winning configuration becomes the GP in `loop/bo.py` **and** in `agent.select_by_gp`, and in any arm added later. Tuning `bo` while leaving our own surrogate at the old settings is the one version of this step that would hurt us, and it is also the one a reviewer would catch. Everything else in (3c) is the sweep that produces that configuration.

   **Ideas, not decided, for turning BO from a rival into a component (kept here so they are not lost; none costed, none chosen).** (i) *The memory shrinks the space*: today it shapes the LLM's candidate list; instead let traps exclude regions and consensus moves fix knobs, so a tuned GP searches a space orders of magnitude smaller at the same budget — the claim becomes "the memory tells the optimizer where not to look". (ii) *The memory replaces BO's initial design*: a tuned BO must spend its first 2-4 designs space-filling, which is what tuning tells it to do, while the memory hands over a design at D1 — this is (3b), and it is both the sharpest threat and the sharpest demonstration. (iii) *The LLM tunes the optimizer*: the agent reads the memory and chooses the GP's hyperparameters for this suite, which turns ArchGym's "all optimizers tie under tuned hyperparameters" into the open question of **who tunes them and on what budget**, with "the memory does, few-shot" as the answer — the most novel of these and the most on-theme for this workshop. (iv) *Report designs-to-quality, not quality-at-N*: "4.7x fewer simulations for the same design" survives BO catching up by D16 and is the number a design team actually buys; computable from the runs already in `results/`. (v) *The 2x2 framing*: {stock start, memory start} x {BO, LLM+GP}, which separates what transfer buys from what the LLM buys and stays informative whichever way it falls.

   **(4d) Evaluate and improve retrieval and the digest, the run-time half.** What the loop does on an unseen suite is untested taste: the 3 nearest cases per test workload by standardized descriptor distance (5 of 6 cases on `dc`), consensus and traps read from each workload's closest case only, the wider circle for workload-dependent moves, and the handed-over design computed over every case. One measurement already argues against the distance filter: effect signs survive on a new workload about two thirds of the time and **flat in descriptor distance**. Free offline checks first, no simulation: what the digest becomes with every case instead of the nearest 3; with another distance, or none; whether the handed-over design survives the "measured on all but one" guard; and which digest sections the agent actually uses — every prompt and every pick is in the reports. Spend designs on an ablation only after those.

   **(4e) After the ones above: an unseen chip.** The same few-shot question one level up — a memory of searches on chip C, a search on a *different* chip. `loop/configs.py` holds one chip today (`CHIP = "C_server"`), so a second profile has to come back (chip B). Cheaper than it sounds: the memory workloads are **not** re-simulated, since the whole point is that the memory stays chip C's; only the new chip's test suite needs its stock design, the arms' designs and a reference ceiling, because every cached table row was measured on chip C. The mechanism question to settle first, and the real content of the step: the digest anchors everything on "moves from the stock chip", and the two chips have different stock designs — so decide what is claimed to transfer, the move or the resulting design.

   **(4f) After the ones above: a stronger model, as an ablation.** The same arms, seeds and cells with `ANALYST_MODEL=gemini-2.5-pro` (the only Pro enabled on this project; Gemini 3 is not), everything else held fixed, to separate what the loop contributes from what the model contributes. Two things to plan around: Pro bills thinking as output, and under CHIA the Vertex layer forwards no thinking budget against a 16k output default — the same cap that killed a Flash run — so upstream fix (3) probably has to land first. Decide whether Pro runs on every arm or only on the memory arm (cost).

   **The roadmap, decided Sep 13:** tune the BO baseline (4c below), then re-run **every arm on the agent-built memory, 5 seeds, 20 rounds**, both cells. That single run replaces the published tables and settles every margin at once.

   Maintenance, small: the LLM re-proposes already-measured designs, 4 to 9 of 8 per round; the retry recovers most, and a compact listing of measured designs in the task text would remove it. After all of the above: `gap2` (graph set 2) with the same arms, and the paper.

4. **What the runs taught, in order.** (a) The verification probe verified true things and spent the budget doing so; removed. (b) The raw case dump was unreadable; the digest (conclusions first) lifted the memory arm's first design from 40% to 65% of the gap. (c) GP selection among LLM proposals pays only when the proposals are good: without the digest the LLM proposes single-knob candidates and the GP cannot rank them. (d) What the memory hands over decides the opening; the rest of the digest decides where the search goes next. (e) A remembered single-knob effect keeps its sign on another workload about two thirds of the time, flat in descriptor distance (`loop.memory leave_one_out`), so the memory is trusted for which moves matter, not for how much. (f) A prior fit on raw log speed-ups ranks the datacenter designs at Spearman 0.29 because GAP's gains dominate; fit on gap shares, 0.73 (the retired prior, below).

5. **Free checks, no VM:** `python -m loop.memory leave_one_out <6 memory traces> -- <3 test traces>` (sign survival) and `python -m loop.workloads headroom`. Check the VM's state (`gcloud compute instances list`) before anything else: it must be stopped whenever nothing runs on it.

## The design

**The question.** Given a stock chip, a workload suite the loop has never seen and a budget of 16 simulated designs, how much of the best known design does each search reach after 1, 2, 4, 8 designs? Memory is a few-shot claim: its value is in the first designs, not the last.

**Arms, same start (the stock chip), same budget, same seeds, same fidelity** (`loop/run.py`, ARMS):
- `bo`: Gaussian process + expected improvement over a seeded 20k-design feasible sample plus the incumbent's one- and two-knob neighbours (`loop/bo.py`).
- `llm_direct`: the plain LLM agent, two picks per round from the results table and the workload descriptors (`loop/agent.py`).
- `memory`: the same agent, its prompt carrying the memory's digest, proposing 8 ranked candidates per round; once 3 designs are measured, a GP fit on **this run only** picks the 2 to simulate by expected improvement (LLAMBO-style candidate sampling with surrogate selection). The GP never sees the memory: the memory shapes the candidate set, the GP ranks it.

One code path for the LLM arms: `agent.run_agent(..., use_memory, use_gp)`; with an empty memory the two produce the identical prompt (checked).

**The memory** (`results/memory_llm.json`, one shelf, built by `loop/memory_build.py`): one **case** per workload the chip has been searched on, written by the code from the result table: descriptors, stock and best design with IPC, every measured single-knob effect with its pair count. Retrieval is by standardized descriptor distance (log scale for counts and sizes): the 3 nearest cases per test workload (`NEAREST_CASES`), which on `dc` is 5 of the 6 cases. No LLM writes case text and there is nothing to consolidate: `memory.case_from_table` generates every case from the table. What the LLM decides is what is IN that table - it chooses which designs get simulated in stage 1 of the build - and that choice is the whole difference between a memory with 36 usable single-knob effects and one with none.

**Building the memory** (`loop/memory_build.py`, three stages, a declared budget of ~122 designs per workload, every design's stage and proposer recorded in `results/memory_<searcher>_record.json`): **search** one per suite by a declared searcher; **anchor** one-knob sweeps from the stock chip and from the search's own winner, which is what supplies the controlled pairs the digest needs (a search alone supplies almost none); **confirm** each suite's best 20 designs measured on every memory workload, which is the pool the handed-over design comes from. Searching first and sweeping around the winner keeps outside knowledge out of the anchor. **The default searcher is `llm`** and the shelf every cell reads is the agent-built one.

**The digest** (`memory.digest`, what the agent actually reads, conclusions first): similarity in words; the descriptors that differ most; the moves that paid and the traps, read from each test workload's **closest** case only; the workload-dependent moves and the knobs where the remembered best designs disagree, from the wider retrieved circle; and one design to copy and adapt — the remembered design with the best mean gap share across the memory workloads, counting only designs measured on all but one of them (`memory.pooled_design`, computed over every case and frozen in the memory file at build time, so distance plays no part in it).

**Prompt** (`agent.assemble_prompt`): problem (chip, budget, knobs, stock design); workloads (descriptor legend, then one line per workload); the digest (memory arm only); the results table (one row per design: index, source, knobs changed from stock, suite IPC, per-workload IPC / LLC MPKI / LLC hit ratio, then the hypothesis it tested); the task. A pick must be a real, in-budget, unmeasured design; else one retry with the rejected designs listed, then a deterministic one-knob perturbation of the incumbent. Never a random design.

**Scoring** (`loop/summarize.py`): per arm, the share of the stock-to-best-known gap reached after N designs, **mean over seeds** with min..max below, and the mean of each run's best design. The best known design is the best measured on every workload of the suite in the cached tables, so it moves as reference searches land.

**Search space** (`loop/configs.py`, 13 knobs, 6.6 M raw designs, area-coupled, latency derived from size): L1D sets/ways/prefetcher; L2 sets/ways/prefetcher/replacement/MSHR; LLC sets/ways/prefetcher/replacement/MSHR. Area = L2 + LLC data capacity <= 4608 KB; L1D and MSHRs cost nothing (known simplification). Chip C: wide core, two memory channels, placeholder profile. Fidelity 5M warmup / 10M simulated (Spearman 0.919 against 50M/50M over 26 designs; shorter failed).

## Cells and workloads

| cell | memory (deep tables, 450-1500 designs each) | test | role |
|---|---|---|---|
| `dc` | mcf, omnetpp, lbm, bfs.urand, pr.urand, bfs.kron | sierra.a.4, merced, tahoe (Google datacenter, DPC4 `gtrace_v2`) | the headline, done |
| `dc2` | the same | whiskey, bravo, delta | the confirmation on traces that shaped nothing; done |
| `gap2` | the same | sssp.kron, cc.urand, cc.twitter | fallback and development |
| `smoke` | omnetpp, bfs.urand | mcf, lbm, one round | the gate before any launch |

**Why the datacenter set.** Twelve families screened (one trace each, 100 MB prefix, no download), six admitted, all six probed with 11 designs. The three in `dc` have +26.6% suite headroom of which 55% no single knob reaches (GAP set 2: +33.5% / 62%; GAP set 1: +144% / 30%, and its arms tied). Their access streams are irregular (stride regularity 0.10-0.34), so a stride prefetcher does not hand over the gain. On every datacenter trace a replacement policy alone hurts or does nothing (srrip -13% on sierra.a.4) while the best design contains ship: the policy pays only after a prefetcher and a bigger LLC, the published SPEC-to-datacenter inversion, measured here. The other three admitted traces became `dc2`. One SimPoint per family so far; the bucket holds 4-89 per family.

| test trace | footprint | MPKI | movable LLC MPKI | stride | headroom (11 designs) | beyond one knob | closest remembered |
|---|---|---|---|---|---|---|---|
| sierra.a.4 | 6.2 MB | 16.4 | 6.46 | 0.34 | +29.5% | 47% | mcf (1.53) |
| merced | 4.5 MB | 9.6 | 4.38 | 0.10 | +33.6% | 52% | omnetpp (2.00) |
| tahoe | 12.9 MB | 26.3 | 2.66 | 0.10 | +17.3% | 65% | mcf (2.41) |

**Caution on the record.** The best of the 11 probed designs is the same composite on all nine graph and datacenter workloads (L2 va_ampm_lite, 4 MB LLC, ship, 512x16 L2, L1D next_line): the design GAP set 1's searches found. Either transfer works or this is a strong default for the whole space; the 11 designs cannot tell. What decides it is `dc2`, where nothing about those traces influenced the memory or that design.

## Evidence kept from earlier cells (condensed; the artifacts were deleted, the numbers live here)

- **Pilots and w1/w1b (GAP set 1 on chip C, 32-48 designs, memory of facts + strategies):** every arm within 1% of each other, seed variance larger than the gaps; SPEC facts held their sign on GAP 56-63% of the time; the agents were optimists (measured landed above forecast 2-15% of the time); a prompt that shows numbers but never the agent's own past hypotheses left the pollution evidence unused for rounds. The first 8 designs of w2a (GAP set 2 from stock, 20-round loop, cut at round 4-7) put every arm at +25-30% of a +33.5% ceiling: the 40-design framing measured noise. That is why the budget is small and the early designs are the claim.
- **Workload screening** (`loop.workloads admit`): capacity channel = movable MPKI >= 1 (validated against measured variance shares: capacity owns omnetpp 0.86, the L2 prefetcher owns lbm 0.96); policy channel = miss floor >= 5 MPKI. Most of SPEC17 does not touch memory hard enough (gcc 100% compulsory, xalancbmk 1.3 MPKI, exchange2 never leaves L1); AI inference traces (11 screened) touch it very hard but too predictably (stride 0.70-0.99, misses compulsory; four llama2 sizes give the same footprint in a 10M window). Only the profiles of the workloads in use are kept; the screen numbers live here.
- **Offline surrogate transfer** (train on other chips, predict this one): every learned surrogate lands at 96% top-5; miss counts are the whole gap; a held-out program cannot be predicted from three (consistent with the literature: 7-29 programs needed). Verdicts: no RL, no deep learning, no learned workload embedding, no LLM-as-surrogate.
- **Literature:** AgentDSE (MLArchSys @ ISCA 2026, arXiv 2606.21836) is the closest prior work: a coding agent tunes a ChampSim hierarchy in ~50 simulations, hypotheses as free text, no transfer. LUMINA, MicroEvo, ArchEval; MetaDSE / OneDSE / PerfVec / Concorde for learned transfer; the interval model (TOCS 2009) and Van den Steen (ISPASS 2015) for mechanistic miss costs.
- **Cost model:** wall time tracks cycles, not instructions: a stock-chip start at IPC 0.17 costs 2.5x a tuned start; more concurrent runs is the only utilisation lever; Pro bills thinking as output. Under CHIA the Vertex model layer forwards only the prompt, so temperature and thinking budget are Gemini's defaults there (an upstream candidate). The fidelity check (Spearman 0.919, 26 designs at 50M/50M) and the chip C variance decomposition (81-design sweeps) were measured; their tables were deleted, the numbers stand here and in `loop/workloads.py`.
- **Retired: the GP prior on the memory tables.** One GP fit per run on every design the memory workloads measured (about 1,900 rows, target = each design's share of that workload's stock-to-best gap in log units), with the run's GP fitting only the residual. It ranked the known datacenter designs well on its own (Spearman 0.73 on shares, against 0.29 on raw speed-ups) but added nothing as an arm: with the prior the memory arm matched its own 98% after 16 designs and was slightly slower in between. Removed from the code; the raw runs stay in `results/run_dc_d3b.json`. Other ways to bring a learned model into the loop are open, this one is answered.

## Upstream: PR candidates for CHIA and ChampSim

`upstream/` holds patches and notes for changes that belong in the frameworks themselves rather than in this repo (`chia/` and `champsim/` are shallow clones, so these go through a fork). Every one of them is a gap we hit while building the loop.

**CHIA.** (1) `chia.simulators.champsim._resolve_trace` raises `NotImplementedError` on `gs://` traces, which every GCP cluster needs; `0001-champsim-gs-trace-resolver.patch` mirrors the existing `s3://` branch with `google-cloud-storage`. (2) `ChampSimNode.build_champsim` only accepts a prefetcher module, so a design-space loop cannot build arbitrary configurations; `loop/chia_nodes.py::build_from_config` is the general version (any config JSON: sizes, policies, prefetchers, core parameters), and `simulate` is the matching multi-trace run node. (3) `chia.models.vertex.VertexGeminiLLM` forwards only max tokens, the system message and tools, so a loop cannot set temperature, JSON response mode or a thinking budget, and every call runs at the model's defaults; a `generation_config` passthrough plus the thinking-token count in `_last_metadata` is a small, general fix. This gap cost cell `dc` fifteen memory-arm runs: eight-candidate answers plus the model's thinking exceeded the layer's 16k output default, and the layer raised on rate limits without waiting. (4) `loop/memory.py` is simulator-agnostic (it reads knobs and descriptors from a `problem` dict): cases built from result tables, descriptor-distance retrieval, the digest and the handed-over design are a candidate for a `chia.analysis` block any agentic loop could wrap around its simulator node.

**ChampSim.** `0002-champsim-spp-dev-ghr-victim.patch` fixes two bugs in `prefetcher/spp_dev/spp_dev.cc`, both found running SPP on a small-core profile with lbm: a heap-buffer-overflow in the lookahead loop (`confidence_q` / `delta_q` are sized to the L2 MSHR count but `read_pattern` appends up to `PT_WAY + 1` entries per step with no bounds check), and a GHR victim search that never finds a victim when every entry has confidence 100, tripping `assert(0)`. No existing upstream issue was found for either.

## Rules of engagement

No dates or deadlines anywhere. Every launch needs the user's explicit OK with cost and time. The Mac runs only reference searches, overnight, under `caffeinate`. The VM is stopped when idle. Kill by process group, never a broad `pkill`. Code stays lean, one mechanism, no version archaeology; domain decisions are surfaced with options and left to the user. Deleted material is in git history, never in the tree.

## How to run

```bash
python -m loop.run memory <cell>                                 # rebuild the shelf (agent-searched, ~122 designs/workload; simulates)
python -m loop.memory_build compare <cell> <memory.json> ...      # read memories side by side, no simulation
python -m loop.memory leave_one_out <memory traces> -- <test traces>   # free transfer check on the tables
python -m loop.workloads admit                                   # the admission gate over every cached profile
python -m loop.workloads headroom <trace> <trace> <trace>        # headroom and share beyond one knob, from the tables
python -m loop.workloads probe <trace> ...                       # the 11-design probe for a new candidate (simulates)
python -m loop.workloads reference 100 10 <trace> <trace> <trace>  # the ceiling for a suite: one long BO search (simulates)
python -m loop.workloads uniform 300 <trace> <trace> <trace>      # the random-search null (simulates)
python -m loop.workloads fetch <url> <out> 100                   # a 100 MB trace prefix, enough to profile and simulate
python -m loop.trace_profile <trace>                             # one workload's profile (cached under results/)
SEEDS=1 LOOP_DISPATCH=chia python -m loop.run smoke s1           # the gate before any launch
SEEDS=5 BUDGET=16 setsid nohup bash cluster/launch_cell.sh dc2 e1 > /dev/null 2>&1 < /dev/null & disown   # on the VM
python -m loop.summarize progress results/dc2_e1.log             # while it runs
python -m loop.summarize results/run_dc2_e1.json                 # when it is done
python -m loop.workloads merge <fetched results dir>             # tables simulated on the VM into results/
```
Env: `SEEDS`, `FIRST_SEED`, `BUDGET` (default 8; the cells ran at 16), `ANALYST_MODEL` (default `gemini-2.5-flash`, the strongest flash this project can call: Gemini 3 is not enabled on it; `gemini-2.5-pro` is the only Pro), `MEMORY_PATH`, `PARALLEL_RUNS`, `SIM_THREADS`. Reports: `results/run_<cell>_<tag>.json` (every design, every prompt, every probe). Memory: `results/memory_<cell>.json`. Nothing else is written.

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
./config.sh champsim_config.json && make -j8 && cd ..
git clone --depth 1 https://github.com/ucb-bar/chia.git chia
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python -e ./chia google-genai
```
Traces (`traces/`, gitignored): SPEC17 from `https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/`; GAP from Zenodo record 20043527 with `loop.workloads fetch_gap` (cite *Characterizing the impact of last-level cache replacement policies on big-data workloads*, IISWC 2020); datacenter traces from the DPC4 bucket `https://pub-c31f67d79d1b4cd28ff320612b1a9f84.r2.dev/manifest.txt` (`gtrace_v2/<family>/<trace>.champsim.gz`, 359 traces in 12 families) as 100 MB prefixes with `loop.workloads fetch`. GCP: project `project-c23a6080-f5d0-4871-9cb`, VM `champsim-1` (c2d-standard-32, europe-west4-a, ~1.5 USD/h, 32-vCPU project cap), recipe in `cluster/README.md`. Budget: ~257 EUR of credits; ~75 USD spent.

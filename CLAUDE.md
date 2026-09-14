# Few-shot cache tuning from a memory of earlier searches

Cache-hierarchy design-space exploration on ChampSim, built as a CHIA loop. What the project produces: the loop, reproducible results from it, the cached simulation tables as a dataset, and the blocks that belong upstream in CHIA and ChampSim. Comparisons are matched-budget, seeded, and scored against ceilings found by an independent mechanism; the metric is share of the stock-to-best gap, not raw IPC. The prior work it is measured against is ArchAgent, ArchGym ("all optimizers tie under tuned hyperparameters"), and DOSA.

**The open problem we answer.** ArchAgent (Gupta, Jain, Gonzalez et al., arXiv 2602.22425) has an agent write cache-replacement policies inside ChampSim. Its authors state that it starts every search from scratch and that "extrapolation from representative workloads" is an open community problem. We test exactly that: an agent that remembers earlier searches on SPEC and graph workloads is handed Google datacenter traces it has never seen and a budget of 16 simulated designs, of which only the first few carry the claim.

**Findings live in `REPORT.md`**, not here. This file is state, design and operations.

## Start here

**The claim: how many simulations are saved to get within 5 % of the best design known.** Both
columns below are the same optimizer, same settings, three seeds each, averaged; only the starting
point differs. `dc2` (whiskey, bravo, delta), stock 0.4389, best known 0.5485 from an independent
100-design search:

```
how close to the best design    seeded from memory    from scratch    saved
90 % of the way                        D1                  D15          14
93 %                                   D5                  D25          20
95 %  (the claim)                      D11                 D37          26
96 %                                   D13                 D49          36
```

The memory's first design is at 92.5 %, which a from-scratch search needs ~15 simulations to match.
The ratio falls as the target rises (15x at 90 %, 3.4x at 95 %) because near the top the search has
to work for the rest and the memory stops helping — it puts you in a good region and cannot get you
out of it (the best design sits six knob changes from the handover, `REPORT.md` §5).

**The four arms on `dc2`, 5 seeds, 16 designs, against the same ceiling** (`run_dc2_f1.json` + `run_dc2_g1.json`):

```
arm            D1    D4    D8   D12   D16   final
bo_gp   (GP)  22%   46%   72%   78%   84%  0.5315
bo      (RF)  19%   45%   57%   61%   68%  0.5139
llm_direct    24%   72%   77%   79%   80%  0.5263
memory        93%   93%   93%   93%   95%  0.5433
pooled_bo     93%   93%   95%   95%   96%  0.5443
```

`pooled_bo` is the default arm to quote: it is the optimizer seeded from the memory, with no LLM at
test time, and it matches or beats the LLM agent reading the digest. Quote the **stronger**
from-scratch surrogate as the baseline (the GP, 84 %): the two differ by 16 points but t = 1.6 and
all nine cold-start runs span 48-96 %, so taking the forest's 68 % would inflate the margin by a
seed draw.

**PENDING — `dc` does not show the same saving.** Its memory arm tops out at 93 % and a from-scratch
search reaches that in 30 simulations, so the saving is ~14 rather than 26. The tables are complete
and correct; what is unresolved is why the head start converts so much worse there. `dc`'s ceiling
is 0.4994, stock 0.3837:

```
arm            D1    D4    D8   D12   D16   final      designs to reach, from scratch (mean of 3 seeds)
bo_gp   (GP)  22%   42%   64%   82%   89%  0.4861      90 % -> D27   92 % -> D30   93 % -> D30
bo      (RF)  22%   61%   72%   75%   88%  0.4856
llm_direct    49%   72%   80%   82%   84%  0.4810
memory        90%   91%   92%   93%   93%  0.4915
pooled_bo     90%   90%   91%   91%   92%  0.4899
```

**`bo` means different mechanisms in the two reports** — the GP in `f1`, the forest in `g1`. The
`f1` arm is stored as **`bo_gp`**, because `summarize` merges by arm name and seed and silently
filled `g1`'s missing `dc2` seed with the GP's, producing a contaminated row. Never let two
mechanisms share an arm name across reports of one cell.

**Next, in this order.**

1. **Resolve `dc`.** Same head start (90 % at D1), a third of the saving. Worth knowing whether it
   is the suite's smaller headroom above the handover, or the basin being tighter there.
2. **The LLM's value at build time is that its search generalises where the optimiser's overfits.**
   That is the claim to lead with, and the leave-one-out table is the evidence: hold out one remembered
   program, pick the handover from the other five out of each searcher's own pool, and the LLM pool
   reaches **84 %** of the held-out program's gap against the optimiser's **78 %**, winning four of six
   (`memory.md` §6). The handovers say the same thing in one line — the optimiser's design scores
   **91.0 %** on the six programs it was searched on and **81 %** on unseen `dc2` (−10 pt); the agent's
   scores **89.8 %** at home and **93 %** away (+3 pt). Better fit to the given data, worse predictor of
   new data. Since the entire premise is a memory built on some workloads being useful on different
   ones, a searcher that overfits is the wrong tool for filling it however good its optimisation trace.
   The **secondary** result — stage 1 with the LLM leaves 30 pair-backed single-knob effects against the
   GP's 0, structurally, because EI picks scattered points that never differ in one knob — is real but
   buys nothing today, because the anchor stage then hands both searchers their pairs. Do not lead with
   it. Two follow-ups; the first is now answered, the second only half:
   - **Converting it into a cost claim is dead, and the ablation that killed it found something else**
     (`memory.md` §7). Every stage can be removed after the fact for free, by rebuilding the memory from
     a filtered build record. Dropping *either* anchor sweep empties the digest's consensus and traps in
     the `llm` memory, because the digest needs two controlled pairs per step and each sweep supplies
     about one — so the anchor stage is minimal, not redundant. **But the handover design is identical
     in all eight variants**: it comes from the confirm pool, never from the sweeps. So the anchor stage
     is 355 simulations, half the build, and everything it buys is four lines of digest whose measured
     value at test time is about one point (`pooled_bo`, which reads no digest, matches `memory`).
     Whether to keep paying that is open, and turns on whether a continuation search can use traps to
     *exclude* regions (item 3).
   - **Why the agent's search generalises is open, and two candidates are closed** (`memory.md` §6).
     Not the optimiser's missing LLC prefetcher: a prefetcher pays +16 to +23 pt on every suite,
     remembered and unseen alike, so it would cost it at home too. Not extremeness either: the
     correlation between how cornered a design is and how much share it loses moving to unseen
     workloads flips sign across slices, +0.57 to −0.35 on 27-57 designs. The phenomenon replicates;
     the mechanism does not have an explanation yet.

   The remaining objection is the single draw: both pools come from one build each and the `llm` build
   is not reproducible (above). Only a re-run settles it.
3. **A continuation search that exploits a strong start** — the priority, because the head start
   decays on both suites. The handover already opens at 92-93 % on unseen suites; everything within
   five knob changes is capped at 93-95 % and the optimum sits six away. Candidates, none costed: a
   trust region that restarts globally when it stalls; using the memory's traps to *exclude* regions
   so the same budget covers a far smaller space; letting the digest fix the knobs it is confident
   about and search only the rest. Measure as designs-to-level, mean over seeds.

   **Future ablation, the leanest of them: a reasoned departure, not a random one.** Today the agent is
   asked only for "the designs most likely to raise the objective", which from a 90 %-opening incumbent
   means small steps, and the optimum is six knobs away. Change only the task text: tell the agent the
   handover is a good region that is capped, and require part of the round to be a **deliberate
   multi-knob departure justified from the digest's disagreements and traps** rather than from the
   incumbent. Run it with `ANALYST_MODEL=gemini-2.5-pro`, since the reasoning is the whole point, on the
   same arms, seeds and cells. One prompt change and one model swap; everything else fixed. It is the
   only candidate here that tests whether an LLM is worth anything **at test time**, where `pooled_bo`
   currently says it is worth about a point.
4. **Never measure a ratio against the 100-design reference.** It runs warm-up 10 and batch 6, which
   flattens it at 79 % from D10 to D16 — exactly where a comparison lives. That produced a retracted
   4.8x. Measure against the **arm's own configuration run long**, which is what the saving table
   above uses: `results/rflong_<cell>_s<seed>.json` holds those curves (75 designs, 3 seeds a cell,
   verified to reproduce the `bo` arm design-for-design). Regenerate one with
   `forest.run(make_suite_problem(run.CELLS[cell]["test"]), 75, 1, seed, tag)`.
5. **The memory arm's candidate filter is still a Gaussian process** (`agent.select_by_gp`) while
   every other search uses the forest. Test the swap for free first by replaying the ~43 logged
   rounds with a forest (the GP filter captures 55 % of each round's candidate spread against 45 %
   for taking the LLM's own order). Changing it makes the `f1` memory rows incomparable.
6. **Evaluate retrieval and the digest** — the run-time half nobody has measured: 3 nearest cases by
   standardized descriptor distance, consensus and traps from the closest case only, the handover
   over every case. One measurement already argues against the distance filter: effect signs survive
   on a new workload about two thirds of the time and **flat in descriptor distance**. Free offline
   checks first; spend designs only after.
7. **Push the test workloads further out.** What is measured today is transfer *inside* the memory's own
   neighbourhood. The six remembered programs sit 2.19 to 8.23 apart in standardized descriptor space
   (median 5.08), and every test workload's nearest remembered program is nearer than that: bravo 1.43,
   sierra.a.4 1.53, whiskey 1.78, merced 2.00, tahoe 2.41, delta 2.61, cc.twitter 2.94, cc.urand 3.26,
   sssp.kron 3.77 — five of the nine closer than **any** two remembered programs are to each other. The
   handover reaches 82-99 % on all twelve workloads with no visible split between seen and unseen, so no
   failure boundary is known. (It is still not trivial: the same build with a GP searcher hands over a
   design reaching 81 % on `dc2`, `gap2`'s own optimum reaches 80 % there, and the best single-knob change
   from stock reaches 47 %. And `dc`'s best design scores 99 % on `dc2`, so the second suite is a
   replication rather than an independent axis.) **The experiment:** take a family the admission gate
   rejected for being too stride-regular (0.70-0.99), re-fetch and re-profile it — free, no simulator —
   then measure the handover and an independent ceiling there. Either it holds, which widens the claim, or
   it breaks, which gives the claim a shape: transfer holds out to descriptor distance X. Cheap addition
   while there: the best single-knob design already sits in the tables and belongs in the score tables as
   a baseline row.
8. **An unseen chip**, then **a stronger model** (`ANALYST_MODEL=gemini-2.5-pro`; Pro bills thinking
   as output and CHIA forwards no thinking budget against a 16k default, `REPORT.md` §7c).

**What is in `results/`:** `table_<trace>.json` the shared simulation cache (the dataset, ~1300 designs on the dc traces, ~950 on dc2); `profile_<trace>.json` one per workload in use; `memory_llm.json` the shelf every cell reads and `memory_bo.json` the optimiser-searched ablation, each with a `_record.json` naming every design's stage and proposer; `run_<cell>_<tag>.json` the reports (`f1` = bo_gp/llm_direct/memory, `g1` = bo/pooled_bo); `rflong_<cell>_s<seed>.json` the from-scratch curves behind the saving table. Nothing else is written.

Small, known: the LLM re-proposes already-measured designs, 4 to 9 of 8 per round; listing measured
designs in the task text would remove it. After the above: `gap2`, then the paper.

**Free checks, no VM:** `python -m loop.memory leave_one_out <6 memory traces> -- <3 test traces>`
and `python -m loop.workloads headroom`. **Check `gcloud compute instances list` before anything
else: the VM must be stopped whenever nothing runs on it.**

## The design

**The question.** Given a stock chip and a workload suite the loop has never seen: **how many simulations does it take to get within 5 % of the best design known**, with and without a memory of earlier searches on other workloads? Designs-to-level, not level-at-N — the second moves whenever the ceiling moves, and it has moved twice.

**Arms, same start (the stock chip), same budget, same seeds, same fidelity** (`loop/run.py`, ARMS):
- `bo`: random forest + expected improvement from the stock chip (`loop/forest.py`), one design per round so every pick is made against measured outcomes.
- `pooled_bo`: the identical search, started from the design the memory hands over. One variable differs from `bo`.
- `llm_direct`: the plain LLM agent, two picks per round from the results table and the workload descriptors (`loop/agent.py`).
- `memory`: the same agent, its prompt carrying the memory's digest, proposing 8 ranked candidates per round; once 3 designs are measured, a GP fit on **this run only** picks the 2 to simulate by expected improvement. The GP never sees the memory: the memory shapes the candidate set, the GP ranks it.

One code path for the LLM arms: `agent.run_agent(..., use_memory, use_gp)`; with an empty memory the two produce the identical prompt (checked). One optimizer for the rest: `loop/forest.py` serves `bo`, `pooled_bo` and the reference, so no comparison rests on one side having a better model. `loop/bo.py` remains only for the memory arm's candidate filter and the optimiser-searched memory build.

**The memory** (`results/memory_llm.json`, built by `loop/memory_build.py`): one **case** per workload the chip has been searched on, written by the code from the result table — descriptors, stock and best design with IPC, every measured single-knob effect with its pair count. Retrieval is by standardized descriptor distance: the 3 nearest cases per test workload (`NEAREST_CASES`), which on `dc` is 5 of the 6 cases. No LLM writes case text; `memory.case_from_table` generates every case. What the LLM decides is what is IN that table — it chooses which designs get simulated in stage 1 of the build. Measured on the stage-1 snapshot that choice is the difference between 30 pair-backed single-knob effects and 0; measured on the completed memories it is 336 against 306, because the anchor stage hands both searchers their pairs.

**Building the memory** (`loop/memory_build.py`, three stages, ~122 designs per workload, every design's stage and proposer in `results/memory_<searcher>_record.json`): **search** one per suite by a declared searcher; **anchor** one-knob sweeps from the stock chip and from the search's own winner, which supplies the controlled pairs the digest needs; **confirm** each suite's best 20 designs measured on every memory workload, the pool the handover comes from. The default searcher is `llm`; `results/memory_bo.json` is the optimiser-searched ablation. `cluster/launch_membuild.sh` rebuilds both from scratch, and the **`bo` build is bit-for-bit repeatable** (`MB_SEED` seeds the pool and the acquisition, ChampSim is deterministic) while the **`llm` build is not** (CHIA's Vertex layer forwards no generation config, so temperature is the model default; `MB_SEED` is dead in that path because it only feeds a GP selection that never fires at 8 candidates for 8 slots). Each memory's internal `share` is against its own ceiling and is **not** comparable across memories — only the handover columns are.

**The digest** (`memory.digest`, conclusions first): similarity in words; the descriptors that differ most; the moves that paid and the traps, from each test workload's **closest** case only; the workload-dependent moves and the knobs where the remembered best designs disagree, from the wider circle; and one design to copy and adapt (`memory.pooled_design`: the remembered design with the best mean gap share across the memory workloads, counting only designs measured on all but one of them, frozen at build time so distance plays no part).

**Prompt** (`agent.assemble_prompt`): problem (chip, budget, knobs, stock design); workloads (descriptor legend, then one line each); the digest (memory arm only); the results table (index, source, knobs changed from stock, suite IPC, per-workload IPC / LLC MPKI / hit ratio, then the hypothesis it tested); the task. A pick must be a real, in-budget, unmeasured design; else one retry with the rejected designs listed, then a deterministic one-knob perturbation of the incumbent. Never a random design.

**Scoring** (`loop/summarize.py`): per arm, the share of the stock-to-best-known gap after N designs, **mean over seeds** with min..max below, and the mean of each run's best design. Best known = the best design measured on every workload of the suite in the cached tables, so it moves as searches land.

**BUDGET counts designs, not rounds** (`rounds = budget // PER_ROUND`): the cells at BUDGET=16 buy 16 designs over 8 rounds. **The unit of cost is the design, the unit of adaptation is the round** — the two designs of an LLM round are chosen together from the same information and listed in the arm's own rank order, so D1 is the design it ranked first and D2 the better of that pair, not a step of learning. `bo` and `pooled_bo` buy one design per round, so every one of their columns is a fresh decision.

**The reference** (`loop/forest.py`, `python -m loop.forest <designs> <batch> <traces>`): the ceiling a cell is scored against, and it must be an independent mechanism — scoring against a design one of the arms found bounds the metric at the best arm, which is exactly what happened on `dc` (0.4936, set by the memory arm, against 0.4994 from an independent 100-design search; re-scoring cost every arm about five points). Re-run it whenever the arms improve.

**Search space** (`loop/configs.py`, 13 knobs, 6.6 M raw designs, area-coupled, latency derived from size): L1D sets/ways/prefetcher; L2 sets/ways/prefetcher/replacement/MSHR; LLC sets/ways/prefetcher/replacement/MSHR. Area = L2 + LLC data capacity <= 4608 KB; L1D and MSHRs cost nothing (known simplification). Chip C: wide core, two memory channels, placeholder profile. Fidelity 5M warmup / 10M simulated (Spearman 0.919 against 50M/50M over 26 designs; shorter failed).

## Cells and workloads

| cell | memory (deep tables) | test | role |
|---|---|---|---|
| `dc` | mcf, omnetpp, lbm, bfs.urand, pr.urand, bfs.kron | sierra.a.4, merced, tahoe (Google datacenter, DPC4 `gtrace_v2`) | the headline |
| `dc2` | the same | whiskey, bravo, delta | the confirmation on traces that shaped nothing |
| `gap2` | the same | sssp.kron, cc.urand, cc.twitter | fallback and development |
| `smoke` | omnetpp, bfs.urand | mcf, lbm, one round | the gate before any launch |

**All six datacenter traces are held out from the memory in the same way** — no case, no simulation during the build, no influence on the handover. The difference is who else saw them: twelve families were screened, six admitted, all six probed with 11 designs, and then we *chose* which three to headline on their numbers (+26.6 % suite headroom, 55 % of it beyond any single knob, irregular access streams). `dc2` is the three we did not pick, so it is the control for **our** selection, not the memory's.

**Why the datacenter set.** Their access streams are irregular (stride regularity 0.10–0.34), so a stride prefetcher does not hand over the gain. On every datacenter trace a replacement policy alone hurts or does nothing (srrip −13 % on sierra.a.4) while the best designs contain one: the policy pays only after a prefetcher and a bigger LLC, the published SPEC-to-datacenter inversion, measured here. One SimPoint per family so far.

| test trace | footprint | MPKI | movable LLC MPKI | stride | headroom (11 designs) | beyond one knob | closest remembered |
|---|---|---|---|---|---|---|---|
| sierra.a.4 | 6.2 MB | 16.4 | 6.46 | 0.34 | +29.5% | 47% | mcf (1.53) |
| merced | 4.5 MB | 9.6 | 4.38 | 0.10 | +33.6% | 52% | omnetpp (2.00) |
| tahoe | 12.9 MB | 26.3 | 2.66 | 0.10 | +17.3% | 65% | mcf (2.41) |

**Do not merge the six into one suite.** Only 81 designs have ever been measured on all six, against 1035 on `dc` and 598 on `dc2`, so a merged ceiling would rest on 81 designs and every share would inflate. The compute is identical either way, so merging costs the replication and buys nothing.

## What earlier runs taught (condensed; the artifacts are deleted, the numbers stand here and in `REPORT.md`)

- **Pilots and GAP set 1:** every arm within 1 % of each other, seed variance larger than the gaps. That is why the budget is small and the early designs are the claim.
- **The loop's own history:** a verification probe verified true things and spent the budget doing so (removed); the raw case dump was unreadable and the digest lifted the memory arm's first design from 40 % to 65 % (`REPORT.md` §2); GP selection among LLM proposals pays only when the proposals are good; what the memory hands over decides the opening, the rest of the digest decides where the search goes next.
- **Workload screening** (`loop.workloads admit`): capacity channel = movable MPKI >= 1 (validated against measured variance shares: capacity owns omnetpp 0.86, the L2 prefetcher owns lbm 0.96); policy channel = miss floor >= 5 MPKI. Most of SPEC17 does not touch memory hard enough; AI inference traces (11 screened) touch it very hard but too predictably (stride 0.70–0.99).
- **Offline surrogate transfer** (train on other chips, predict this one): every learned surrogate lands at 96 % top-5; miss counts are the whole gap; a held-out program cannot be predicted from three (literature: 7–29 programs needed). Verdicts: no RL, no deep learning, no learned workload embedding, no LLM-as-surrogate.
- **Literature:** AgentDSE (MLArchSys @ ISCA 2026, arXiv 2606.21836) is the closest prior work — a coding agent tunes a ChampSim hierarchy in ~50 simulations, no transfer. MetaDSE / OneDSE / PerfVec / Concorde for learned transfer; SMAC and HyperMapper for random-forest DSE over categorical spaces with feasibility constraints; the interval model (TOCS 2009) and Van den Steen (ISPASS 2015) for mechanistic miss costs.
- **Retired, not to reappear as arms:** the digest variant that handed over the closest workload's best design; the one-design retrieval arm; the GP prior fit on the memory tables (it ranked unseen datacenter designs at Spearman 0.73 on gap shares against 0.29 on raw speed-ups, but added nothing as an arm). Other ways to bring a learned model into the loop are open; this one is answered.
- **Answered and closed:** a surrogate choosing the handover instead of `pooled_design`'s rule. Leave-one-out over the memory workloads gave +1 point (86 % → 87 %), and on the real test suites every rule picks the same design because only 16–25 memory designs have been measured there. The guard ("measured on all but one") is worth 4 points and justifies the confirm stage.

## Upstream: contributions to CHIA and ChampSim

`upstream/` holds the patches. **Every contribution is described in `REPORT.md` §7** (the `gs://` trace resolver, the config-space build node, the Vertex generation-config passthrough and its measured cost, the simulator-agnostic case memory, the two spp_dev bugs). Do not restate them here.

## Rules of engagement

No dates or deadlines anywhere. **Code and docs state plain facts only.** They describe the mechanism and the measurements; they do not speculate about how the work will be received, and they name no one outside the technical references. Every launch needs the user's explicit OK with cost and time. The Mac runs reference searches and overnight work under `caffeinate` (which stops idle sleep, **not** lid-close sleep). The VM is stopped when idle. Kill by process group or exact pid, never a broad `pkill`. Code stays lean, one mechanism, no version archaeology; domain decisions are surfaced with options and left to the user. Deleted material is in git history, never in the tree.

## How to run

```bash
python -m loop.run memory <cell>                                  # rebuild the shelf (agent-searched; simulates)
python -m loop.memory_build compare <cell> <memory.json> ...      # read memories side by side, no simulation
python -m loop.memory leave_one_out <memory traces> -- <test traces>   # free transfer check on the tables
python -m loop.workloads admit                                    # the admission gate over every cached profile
python -m loop.workloads headroom <trace> <trace> <trace>         # headroom and share beyond one knob
python -m loop.workloads probe <trace> ...                        # the 11-design probe for a new candidate (simulates)
python -m loop.forest 100 10 <trace> <trace> <trace>              # the suite's independent ceiling (simulates)
python -c "from loop import forest, run; from loop.champsim_problem import make_suite_problem; \
  forest.run(make_suite_problem(run.CELLS['dc2']['test']), 75, 1, 0, 'long-s0')"   # the bo arm run long, for designs-to-level
python -m loop.workloads uniform 300 <trace> <trace> <trace>      # the random-search null (simulates)
python -m loop.workloads fetch <url> <out> 100                    # a 100 MB trace prefix
python -m loop.trace_profile <trace>                              # one workload's profile (cached under results/)
SEEDS=1 LOOP_DISPATCH=chia python -m loop.run smoke s1            # the gate before any launch
SEEDS=5 BUDGET=16 CHAMPSIM_BUILD_SHARE=8 SMOKE_FIRST=1 POWER_OFF=1 \
  setsid nohup bash cluster/launch_cell.sh dc f1 [arm,arm] > /dev/null 2>&1 < /dev/null & disown   # on the VM
python -m loop.summarize progress results/dc_f1.log               # while it runs
python -m loop.summarize results/run_dc_f1.json                   # when it is done
python -m loop.workloads merge <fetched results dir>              # tables simulated on the VM into results/
```

Env: `SEEDS`, `FIRST_SEED`, `BUDGET` (designs, default 8; cells run at 16), `ANALYST_MODEL` (default `gemini-2.5-flash`; `gemini-2.5-pro` is the only Pro enabled), `MEMORY_PATH`, `PARALLEL_RUNS`, `SIM_THREADS` (concurrency = 3 x this), `CHAMPSIM_BUILD_SHARE` (set to 8 on the VM: the default gives each build `make -j$(nproc)` while 8 builds run at once). Reports: `results/run_<cell>_<tag>.json`. Memory: `results/memory_llm.json`.

**Ray's driver log stalls** while the driver blocks collecting results in submission order, so `summarize progress` can freeze mid-run while the cell is fine. Read the truth from the per-worker logs: `cat $(ls -dt /tmp/ray/session_*/logs | head -1)/worker-*.out | grep "ipc="`.

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
./config.sh champsim_config.json && make -j8 && cd ..
git clone --depth 1 https://github.com/ucb-bar/chia.git chia
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python -e ./chia google-genai
```

Traces (`traces/`, gitignored): SPEC17 from `https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/`; GAP from Zenodo record 20043527 with `loop.workloads fetch_gap` (cite *Characterizing the impact of last-level cache replacement policies on big-data workloads*, IISWC 2020); datacenter traces from the DPC4 bucket `https://pub-c31f67d79d1b4cd28ff320612b1a9f84.r2.dev/manifest.txt` (`gtrace_v2/<family>/<trace>.champsim.gz`, 359 traces in 12 families) as 100 MB prefixes with `loop.workloads fetch`.

GCP: project `project-c23a6080-f5d0-4871-9cb`, VM `champsim-1` (c2d-standard-32, europe-west4-a, ~1.5 USD/h), recipe in `cluster/README.md`. **The VM has no git checkout: code goes there with `gcloud compute scp loop/*.py`, never `git pull`.** Check `ls -l loop/forest.py` on it before a launch.

Budget: **~80-90 EUR of credits left** — the GCP console is the source of truth, this line goes stale and must not be extrapolated from. Cost of a run, measured over every VM run: **94-154 designs simulated per hour, ~110 typical**, at ~1.5 USD/h. Arms are not the cost, designs are: estimate a launch as (cells x arms x seeds x budget) / 110 hours. The binding limit is the global `CPUS_ALL_REGIONS` quota of 32 while europe-west4 allows 200; raising that one quota would cut wall time proportionally at the same total cost, and is refused while the billing account is on trial credits.

`results_archive/` (gitignored) is where `cluster/launch_membuild.sh` moves the memory workloads' existing tables before a rebuild, so the new tables hold only what that build measured. Whatever lands there is superseded by definition and must never be merged back into `results/`.

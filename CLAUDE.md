# CHIA Hackathon: few-shot cache tuning from a memory of earlier searches

A3 workshop hackathon (agentic-arch.org). Deliverable: 4-page paper + open-sourced CHIA loop with reproducible results; "reusable CHIA blocks upstreamed to mainline" is an explicit track. What the work has to deliver: novelty and quality of the agentic architecture, matched-budget baselines, seeds, realizable designs, and reusable blocks + datasets. Not raw IPC. The prior results it is measured against are ArchAgent, ArchGym ("all optimizers tie under tuned hyperparameters"), and DOSA.

**The open problem we answer.** ArchAgent (Gupta, Jain, Gonzalez et al., arXiv 2602.22425) has an agent write cache-replacement policies inside ChampSim. Its authors state that it starts every search from scratch and that "extrapolation from representative workloads" is an open community problem. We test exactly that: an agent that remembers earlier searches on SPEC and graph workloads is handed Google datacenter traces it has never seen and a budget of 16 simulated designs, of which only the first few carry the claim.

**Findings live in `REPORT.md`**, not here. This file is state, design and operations.

## Start here

**Both cells are complete, all four arms, against independent ceilings** (`results/run_<cell>_f1.json` + `run_<cell>_g1.json`; reproduce with `python -m loop.summarize <both reports of a cell>`).

```
dc   stock 0.3837, ceiling 0.4994 (independent)      dc2  stock 0.4389, ceiling 0.5485 (independent)
arm            D1    D4    D8   D12   D16   final     arm            D1    D4    D8   D12   D16   final
bo_gp   (GP)  22%   42%   64%   82%   89%  0.4861     bo_gp   (GP)  22%   46%   72%   78%   84%  0.5315
bo      (RF)  22%   61%   72%   75%   88%  0.4856     bo      (RF)  19%   45%   57%   61%   68%  0.5139
llm_direct    49%   72%   80%   82%   84%  0.4810     llm_direct    24%   72%   77%   79%   80%  0.5263
memory        90%   91%   92%   93%   93%  0.4915     memory        93%   93%   93%   93%   95%  0.5433
pooled_bo     90%   90%   91%   91%   92%  0.4899     pooled_bo     93%   93%   95%   95%   96%  0.5443
```

**Quote the STRONGER surrogate as the baseline** (GP on both suites: 89 % and 84 %). The two tie on `dc` and differ by 16 points on `dc2`, but per seed the forest lands at 48/62/72/91 and the process at 73/76/88/90/96 — t = 1.6, and all nine cold-start runs span 48-96 %. Taking the forest's 68 % would inflate the margin by a seed draw; the claim does not need it.

**`bo` means different mechanisms in the two reports** — the GP in `f1`, the forest in `g1`. The `f1` arm has been renamed **`bo_gp`** in the stored reports, because `summarize` merges by arm name and seed and silently filled `g1`'s missing `dc2` seed 1 with the GP's, producing a contaminated row. Never let two mechanisms share an arm name across reports of one cell.

**What this settles.** The memory's first design (90 % / 93 %) beats sixteen designs of the best memoryless optimizer (89 % / 84 %) on both suites. And `pooled_bo` — the same optimizer handed the memory's design, no LLM at test time — matches `memory` on `dc` and beats it on `dc2`. **The memory is the contribution; the agent reading it is worth about a point.** The LLM's measured value is at build time (`REPORT.md` §1). `REPORT.md` §4 is written on these tables.

**In flight (Mac, overnight):** the `bo` arm's exact configuration run to **75 designs**, 3 seeds on each cell, six processes under `caffeinate`, scratch code `rf_long.py` (not in the repo). It answers the open question in `REPORT.md` §4 — how many designs a memoryless forest search needs to reach the levels the memory-started arms reach at D16 — which the 100-design references cannot answer because they use warm-up 10 and batch 6. Curves land as `rflong_<cell>_s<seed>.json` in the session scratchpad; the simulations land in the shared tables either way.

**Next, in this order.**

1. **When the 75-design runs land**: read off where the memoryless forest crosses 90 % / 93 % (the memory arms' opening) and 93 % / 96 % (their D16). That is the ratio `REPORT.md` §4 leaves open — state it per suite with the seed range, never as a single number.
2. **The ratio is currently unpublishable** (`REPORT.md` §4, "Open"). Run the **`bo` arm's exact configuration** — warm-up 3, expected improvement, per-seed candidate pool — for about 100 designs on **both** cells, and read off where it crosses the memory arm's D16. Batch 1 leaves a machine mostly idle, so use batch 3 and say so. The existing 100-design references cannot answer this: they use warm-up 10 and batch 6, which flatten their curve at 79 % from D10 to D16 and inflate the apparent ratio to 4.8x when the true figure is probably 2.5-3x. Until this exists, claim the matched-budget sentence (one design against more than sixteen), never a ratio.
3. **The memory arm's candidate filter is still a Gaussian process** (`agent.select_by_gp`) while every other search uses the forest. Test the swap for free first by replaying the ~43 logged rounds in the reports with a forest (the GP filter captures 55 % of each round's candidate spread against 45 % for taking the LLM's own order). Only then change it — changing it makes the `f1` memory rows incomparable.
4. **Evaluate retrieval and the digest** — the run-time half nobody has measured. Today: the 3 nearest cases per test workload by standardized descriptor distance, consensus and traps from each workload's closest case only, the wider circle for workload-dependent moves, the handover computed over every case. One measurement already argues against the distance filter: effect signs survive on a new workload about two thirds of the time and **flat in descriptor distance**. Free offline checks first: the digest with every case instead of the nearest 3; with another distance, or none; which digest sections the agent actually uses (every prompt and pick is in the reports). Spend designs only after those.
5. **An unseen chip.** The same question one level up: a memory of searches on chip C, a search on a *different* chip. `loop/configs.py` holds one chip (`CHIP = "C_server"`), so a second profile has to come back. The memory workloads are **not** re-simulated — only the new chip's test suite needs a stock design, the arms' designs and a reference. Settle the mechanism question first: the digest anchors everything on "moves from the stock chip", and two chips have different stock designs, so decide whether the *move* or the *resulting design* is what transfers.
6. **A stronger model, as an ablation.** Same arms, seeds and cells with `ANALYST_MODEL=gemini-2.5-pro`, everything else fixed. Pro bills thinking as output and CHIA forwards no thinking budget against a 16k default (`REPORT.md` §6c), so that fix probably has to land first. Decide whether Pro runs on every arm or only the memory arm.

Small, known: the LLM re-proposes already-measured designs, 4 to 9 of 8 per round; the retry recovers most, and listing measured designs in the task text would remove it. After the above: `gap2` with the same arms, then the paper.

**Free checks, no VM:** `python -m loop.memory leave_one_out <6 memory traces> -- <3 test traces>` (sign survival) and `python -m loop.workloads headroom`. **Check `gcloud compute instances list` before anything else: the VM must be stopped whenever nothing runs on it.**

## The design

**The question.** Given a stock chip, a workload suite the loop has never seen and a budget of 16 simulated designs, how much of the best known design does each search reach after 1, 2, 4, 8 designs? Memory is a few-shot claim: its value is in the first designs, not the last.

**Arms, same start (the stock chip), same budget, same seeds, same fidelity** (`loop/run.py`, ARMS):
- `bo`: random forest + expected improvement from the stock chip (`loop/forest.py`), one design per round so every pick is made against measured outcomes.
- `pooled_bo`: the identical search, started from the design the memory hands over. One variable differs from `bo`.
- `llm_direct`: the plain LLM agent, two picks per round from the results table and the workload descriptors (`loop/agent.py`).
- `memory`: the same agent, its prompt carrying the memory's digest, proposing 8 ranked candidates per round; once 3 designs are measured, a GP fit on **this run only** picks the 2 to simulate by expected improvement. The GP never sees the memory: the memory shapes the candidate set, the GP ranks it.

One code path for the LLM arms: `agent.run_agent(..., use_memory, use_gp)`; with an empty memory the two produce the identical prompt (checked). One optimizer for the rest: `loop/forest.py` serves `bo`, `pooled_bo` and the reference, so no comparison rests on one side having a better model. `loop/bo.py` remains only for the memory arm's candidate filter and the optimiser-searched memory build.

**The memory** (`results/memory_llm.json`, built by `loop/memory_build.py`): one **case** per workload the chip has been searched on, written by the code from the result table — descriptors, stock and best design with IPC, every measured single-knob effect with its pair count. Retrieval is by standardized descriptor distance: the 3 nearest cases per test workload (`NEAREST_CASES`), which on `dc` is 5 of the 6 cases. No LLM writes case text; `memory.case_from_table` generates every case. What the LLM decides is what is IN that table — it chooses which designs get simulated in stage 1 of the build — and that choice is the whole difference between a memory with 36 usable single-knob effects and one with none.

**Building the memory** (`loop/memory_build.py`, three stages, ~122 designs per workload, every design's stage and proposer in `results/memory_<searcher>_record.json`): **search** one per suite by a declared searcher; **anchor** one-knob sweeps from the stock chip and from the search's own winner, which supplies the controlled pairs the digest needs; **confirm** each suite's best 20 designs measured on every memory workload, the pool the handover comes from. The default searcher is `llm`; `results/memory_bo.json` is the optimiser-searched ablation. Each memory's internal `share` is against its own ceiling and is **not** comparable across memories — only the handover columns are.

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

`upstream/` holds the patches. **Every contribution is described in `REPORT.md` §6** (the `gs://` trace resolver, the config-space build node, the Vertex generation-config passthrough and its measured cost, the simulator-agnostic case memory, the two spp_dev bugs). Do not restate them here.

## Rules of engagement

No dates or deadlines anywhere. **Code and docs state plain facts only: never name who will read or judge this work, and never write what a reviewer might say.** Anyone reading the repository should find the mechanism and the measurements, nothing about how the work expects to be received. Every launch needs the user's explicit OK with cost and time. The Mac runs reference searches and overnight work under `caffeinate` (which stops idle sleep, **not** lid-close sleep). The VM is stopped when idle. Kill by process group or exact pid, never a broad `pkill`. Code stays lean, one mechanism, no version archaeology; domain decisions are surfaced with options and left to the user. Deleted material is in git history, never in the tree.

## How to run

```bash
python -m loop.run memory <cell>                                  # rebuild the shelf (agent-searched; simulates)
python -m loop.memory_build compare <cell> <memory.json> ...      # read memories side by side, no simulation
python -m loop.memory leave_one_out <memory traces> -- <test traces>   # free transfer check on the tables
python -m loop.workloads admit                                    # the admission gate over every cached profile
python -m loop.workloads headroom <trace> <trace> <trace>         # headroom and share beyond one knob
python -m loop.workloads probe <trace> ...                        # the 11-design probe for a new candidate (simulates)
python -m loop.forest 100 10 <trace> <trace> <trace>              # the suite's independent ceiling (simulates)
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

`results_archive/` (gitignored) holds ~5 000 simulations on the memory workloads from before the rebuild — designs the current tables do **not** contain. Do not delete it.

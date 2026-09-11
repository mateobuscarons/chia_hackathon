# CHIA Hackathon: few-shot cache tuning from a memory of earlier searches

A3 workshop hackathon (agentic-arch.org). Deliverable: 4-page paper + open-sourced CHIA loop with reproducible results; "reusable CHIA blocks upstreamed to mainline" is an explicit track. Judged by the A3 program committee: Gonzalez and Jain (Google, ArchAgent authors), Karandikar (UC Berkeley, CHIA PI), Yazdanbakhsh (Google DeepMind, ArchGym: "all optimizers tie under tuned hyperparameters"), Huang (NVIDIA, DOSA), Skarlatos (CMU). They reward novelty and quality of the agentic architecture, matched-budget baselines, seeds, realizability, and reusable blocks + datasets. Not raw IPC.

**The open problem we answer.** ArchAgent (Gupta, Jain, Gonzalez et al., arXiv 2602.22425; four of the six organizers are authors) has an agent write cache-replacement policies inside ChampSim. Its authors state that it starts every search from scratch and that "extrapolation from representative workloads" is an open community problem. We test exactly that: an agent that remembers earlier searches on SPEC and graph workloads is handed Google datacenter traces it has never seen and a budget of 8 simulated designs.

## Start here (next session)

1. **The loop was rebuilt from scratch and has NOT been committed** (the state before the rebuild is the commit "checkpoint before the rebuild"). Everything in `loop/` is new; the old rules, bets, facts, strategies and surrogate studies are gone. Review, then commit.
2. **Cell `dc` has run twice on `gemini-2.5-flash`, 2 seeds, 8 designs, about 45 minutes and 1 USD each** (reports `results/run_dc_d1.json`, `run_dc_d2.json`; every prompt inside). d1 had the raw-case memory and a verification probe per round; d2 has the digest and the two hybrid arms. Share of the stock-to-best-known gap after 8 designs, d2: llm_bo_memory 95%, memory 90%, bo 88%, llm_direct 85%, replay 78% (one design), llm_bo 74%. llm_bo_memory seed 1 found 0.4901, the new best known design on this suite (L1D 128 sets + next_line, L2 spp_dev, 4 MB LLC with next_line, lru), above the GAP composite that had led at 0.4859. Two seeds: the ordering is suggestive, not settled.
3. **What the two runs taught.** (a) The verification probe cost the memory arm its budget: three of four probes confirmed effects remembered as harmful; removed. (b) The raw case dump was unreadable; the digest (conclusions first, moves from stock, traps, disagreements, a copyable design) lifted the memory arm from 87% to 90% and its round-1 design from 41% to 65% of the gap. (c) The GP over LLM proposals helps only with the memory on: without it the LLM proposed timid single-knob candidates and the GP, fit on 3-7 points, could not tell them apart (llm_bo seed 0 never left 56%). With the memory, the LLM's candidates are all in the good region and the GP picks among them well. (d) 4-9 of 8 proposals per round were rejected as duplicates in the hybrid arms: the LLM re-proposes measured designs; the retry recovers most.
4. **Check the VM first** (`gcloud compute instances list`): `champsim-1` must be stopped whenever nothing runs on it. The user's choice for this phase: iterate fast on `gemini-2.5-flash` with 2 seeds; Pro only when a result is worth confirming.
5. **Free result already in hand (leave-one-out over the cached tables):** remembered single-knob effects keep their sign on another workload about two thirds of the time, and the curve against descriptor distance is nearly flat (70% / 59% / 63% by tercile). All 8 probes of d1 held. The distance claim is weak; the few-shot score at 4 and 8 designs is the claim the cell can make.

## The design

**The question.** Given a stock chip, a new workload suite and 8 simulated designs, how much of the best known design does each search reach after 1, 2, 4, 8 designs? Memory is a few-shot claim: its value is in the first designs, not the last.

**Arms, same start (the stock chip), same budget, same seeds, same fidelity:**
- `bo`: Gaussian process + expected improvement over a seeded 20k-design feasible sample plus the incumbent's one- and two-knob neighbours (`loop/bo.py`).
- `llm_direct`: the plain LLM agent, two picks per round from the results table and the workload descriptors (`loop/agent.py`).
- `memory`: the same agent whose prompt carries the memory's **digest** (`memory.digest`): how similar each workload is to its nearest remembered one and on which descriptors they differ; the moves from stock that paid on every nearest workload, ranked; the traps; the knobs on which the nearest best designs disagree (the search's open questions); and the closest workload's best design as copyable JSON. Computed by code from the cases, about 25 lines, conclusions first.
- `llm_bo`: the agent proposes 8 ranked candidates per round; once 3 designs are measured, a GP fit on this run picks the 2 to simulate by expected improvement (LLAMBO-style candidate sampling with surrogate selection). `llm_bo_memory`: both switches on. One code path for the four LLM arms (`agent.run_agent(use_memory, use_gp)`).
- `replay`: one design, the nearest remembered workload's best design fitted to the budget. If this already wins, transfer is a lookup and the paper says so.

**The memory** (`results/memory_<cell>.json`, one shelf): one **case** per workload the chip has been searched on, written by the code from the result table: descriptors, stock and best design with IPC, every measured single-knob effect with its pair count. Retrieval by standardized descriptor distance (log scale for counts and sizes). Nothing written by an LLM, nothing to consolidate: `python -m loop.run memory <cell>` rebuilds it from the tables.

**Prompt** (`agent.assemble_prompt`): problem (chip, budget, knobs, stock design); workloads (descriptor legend, then one line per workload); the memory digest (memory arms only); the results table (one row per design: index, source, knobs changed from stock, suite IPC, per-workload IPC / LLC MPKI / LLC hit ratio, then the hypothesis it tested or the verification it settled); the task. With an empty memory the two LLM arms produce the identical prompt (checked). A pick must be a real, in-budget, unmeasured design; else one retry with the rejected designs listed, then a deterministic one-knob perturbation of the incumbent. Never a random design.

**Scoring** (`loop/summarize.py`): per arm, the share of the stock-to-best-known gap reached after N designs, mean over seeds with min..max; the best known design is the best measured on every workload of the suite in the cached tables.

**Search space** (`loop/configs.py`, 13 knobs, 6.6 M raw designs, area-coupled, latency derived from size): L1D sets/ways/prefetcher; L2 sets/ways/prefetcher/replacement/MSHR; LLC sets/ways/prefetcher/replacement/MSHR. Area = L2 + LLC data capacity <= 4608 KB; L1D and MSHRs cost nothing (known simplification). Chip C (`loop/configs.py`): wide core, two memory channels, placeholder profile. Fidelity 5M warmup / 10M simulated (Spearman 0.919 against 50M/50M over 26 designs; shorter failed).

## Cells and workloads

| cell | memory (deep tables, 450-1500 designs each) | test | role |
|---|---|---|---|
| `dc` | mcf, omnetpp, lbm, bfs.urand, pr.urand, bfs.kron | sierra.a.4, merced, tahoe (Google datacenter, DPC4 `gtrace_v2`) | the headline |
| `gap2` | the same | sssp.kron, cc.urand, cc.twitter | fallback and development |
| `smoke` | omnetpp, bfs.urand | mcf, lbm, one round | the gate before any launch |

**Why the datacenter set.** Twelve families screened (one trace each, 100 MB prefix, no download), six admitted, all six probed with 11 designs. The three chosen have +26.6% suite headroom of which 55% no single knob reaches (GAP set 2: +33.5% / 62%; GAP set 1: +144% / 30%, and its arms tied). Their access streams are irregular (stride regularity 0.10-0.34), so a stride prefetcher does not hand over the gain. On every datacenter trace a replacement policy alone hurts or does nothing (srrip -13% on sierra.a.4) while the best design contains ship: the policy pays only after a prefetcher and a bigger LLC, the published SPEC-to-datacenter inversion, measured here. Spare admitted traces with probe tables: whiskey, bravo, delta. One SimPoint per family so far; the bucket holds 4-89 per family.

| test trace | footprint | MPKI | movable LLC MPKI | stride | headroom (11 designs) | beyond one knob | nearest case |
|---|---|---|---|---|---|---|---|
| sierra.a.4 | 6.2 MB | 16.4 | 6.46 | 0.34 | +29.5% | 47% | mcf (1.53) |
| merced | 4.5 MB | 9.6 | 4.38 | 0.10 | +33.6% | 52% | omnetpp (2.00) |
| tahoe | 12.9 MB | 26.3 | 2.66 | 0.10 | +17.3% | 65% | mcf (2.41) |

**Caution on the record.** The best of the 11 probed designs is the same composite on all nine graph and datacenter workloads (L2 va_ampm_lite, 4 MB LLC, ship, 512x16 L2, L1D next_line): the design GAP set 1's searches found. Either transfer works or this is a strong default for the whole space; the 11 designs cannot tell. The `replay` arm and the plain agent's first four designs decide it.

## Evidence kept from earlier cells (condensed; the artifacts were deleted, the numbers live here)

- **Pilots and w1/w1b (GAP set 1 on chip C, 32-48 designs, memory of facts + strategies):** every arm within 1% of each other, seed variance larger than the gaps; SPEC facts held their sign on GAP 56-63% of the time; the agents were optimists (measured landed above forecast 2-15% of the time); a prompt that shows numbers but never the agent's own past hypotheses left the pollution evidence unused for rounds. The first 8 designs of w2a (GAP set 2 from stock, 20-round v2 loop, cut at round 4-7) put every arm at +25-30% of a +33.5% ceiling: the 40-design framing measured noise. That is why the budget is 8.
- **Workload screening** (`loop/screen.py admit`): capacity channel = movable MPKI >= 1 (validated against measured variance shares: capacity owns omnetpp 0.86, the L2 prefetcher owns lbm 0.96); policy channel = miss floor >= 5 MPKI. Most of SPEC17 does not touch memory hard enough (gcc 100% compulsory, xalancbmk 1.3 MPKI, exchange2 never leaves L1); AI inference traces (11 screened) touch it very hard but too predictably (stride 0.70-0.99, misses compulsory; four llama2 sizes give the same footprint in a 10M window). Only the profiles of the workloads in use are kept; the screen numbers live here.
- **Offline surrogate transfer** (train on other chips, predict this one): every learned surrogate lands at 96% top-5; miss counts are the whole gap; a held-out program cannot be predicted from three (consistent with the literature: 7-29 programs needed). Verdicts: no RL, no deep learning, no learned workload embedding, no LLM-as-surrogate.
- **Literature:** AgentDSE (MLArchSys @ ISCA 2026, arXiv 2606.21836) is the closest prior work: a coding agent tunes a ChampSim hierarchy in ~50 simulations, hypotheses as free text, no transfer. LUMINA, MicroEvo, ArchEval; MetaDSE / OneDSE / PerfVec / Concorde for learned transfer; the interval model (TOCS 2009) and Van den Steen (ISPASS 2015) for mechanistic miss costs.
- **Cost model:** wall time tracks cycles, not instructions: a stock-chip start at IPC 0.17 costs 2.5x a tuned start; more concurrent runs is the only utilisation lever; Pro bills thinking as output. Under CHIA the Vertex model layer forwards only the prompt, so temperature and thinking budget are Gemini's defaults there (an upstream candidate). The fidelity check (Spearman 0.919, 26 designs at 50M/50M) and the chip C variance decomposition (81-design sweeps) were measured; their tables were deleted, the numbers stand here and in `loop/workloads.py`.

## Rules of engagement

No dates or deadlines anywhere. Every launch needs the user's explicit OK with cost and time. The Mac runs only reference samples, overnight, under `caffeinate`. The VM is stopped when idle. Kill by process group, never a broad `pkill`. Code stays lean, one mechanism, no version archaeology; domain decisions are surfaced with options and left to the user. Deleted material is in git history, never in the tree.

## How to run

```bash
python -m loop.run memory dc                                   # the cell's memory from the cached tables
python -m loop.memory leave_one_out <memory traces> -- <test traces>   # free transfer check on the tables
python -m loop.workloads admit                                    # the admission gate over every cached profile
python -m loop.workloads headroom <trace> <trace> <trace>         # headroom and share beyond one knob, from the tables
python -m loop.workloads probe <trace> ...                       # the 11-design probe for a new candidate (simulates)
python -m loop.workloads fetch <url> <out> 100                   # a 100 MB trace prefix, enough to profile and simulate
python -m loop.trace_profile <trace>                           # one workload's profile (cached under results/)
SEEDS=1 LOOP_DISPATCH=chia python -m loop.run smoke s1         # the gate before any launch
SEEDS=2 setsid nohup bash cluster/launch_cell.sh dc d1 > /dev/null 2>&1 < /dev/null & disown   # on the VM
python -m loop.summarize progress results/dc_d1.log            # while it runs
python -m loop.summarize results/run_dc_d1.json                # when it is done
python -m loop.workloads merge <fetched results dir>             # tables simulated on the VM into results/
```
Env: `SEEDS`, `FIRST_SEED`, `BUDGET` (default 8), `ANALYST_MODEL` (default `gemini-2.5-flash`, the strongest flash this project can call: Gemini 3 is not enabled on it; `gemini-2.5-pro` is the only Pro), `MEMORY_PATH`, `PARALLEL_RUNS`, `SIM_THREADS`. Reports: `results/run_<cell>_<tag>.json` (every design, every prompt, every probe). Memory: `results/memory_<cell>.json`. Nothing else is written.

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
./config.sh champsim_config.json && make -j8 && cd ..
git clone --depth 1 https://github.com/ucb-bar/chia.git chia
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python -e ./chia google-genai
```
Traces (`traces/`, gitignored): SPEC17 from `https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/`; GAP from Zenodo record 20043527 with `loop.workloads fetch_gap` (cite *Characterizing the impact of last-level cache replacement policies on big-data workloads*, IISWC 2020); datacenter traces from the DPC4 bucket `https://pub-c31f67d79d1b4cd28ff320612b1a9f84.r2.dev/manifest.txt` (`gtrace_v2/<family>/<trace>.champsim.gz`, 359 traces in 12 families) as 100 MB prefixes with `loop.workloads fetch`. GCP: project `project-c23a6080-f5d0-4871-9cb`, VM `champsim-1` (c2d-standard-32, europe-west4-a, ~1.5 USD/h, 32-vCPU project cap), recipe in `cluster/README.md`. Budget: ~257 EUR of credits; ~75 USD spent.

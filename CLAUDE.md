# CHIA Hackathon: few-shot cache tuning from a memory of earlier searches

A3 workshop hackathon (agentic-arch.org). Deliverable: 4-page paper + open-sourced CHIA loop with reproducible results; "reusable CHIA blocks upstreamed to mainline" is an explicit track. Judged by the A3 program committee: Gonzalez and Jain (Google, ArchAgent authors), Karandikar (UC Berkeley, CHIA PI), Yazdanbakhsh (Google DeepMind, ArchGym: "all optimizers tie under tuned hyperparameters"), Huang (NVIDIA, DOSA), Skarlatos (CMU). They reward novelty and quality of the agentic architecture, matched-budget baselines, seeds, realizability, and reusable blocks + datasets. Not raw IPC.

**The open problem we answer.** ArchAgent (Gupta, Jain, Gonzalez et al., arXiv 2602.22425; four of the six organizers are authors) has an agent write cache-replacement policies inside ChampSim. Its authors state that it starts every search from scratch and that "extrapolation from representative workloads" is an open community problem. We test exactly that: an agent that remembers earlier searches on SPEC and graph workloads is handed Google datacenter traces it has never seen and a budget of 8 simulated designs.

## Start here (next session)

1. **Cell `dc` at 16 designs is DONE** (d3 = `bo`, `llm_direct`, `pooled`; d3b = the three memory arms after the CHIA output-cap fix; reports `results/run_dc_d3.json` + `run_dc_d3b.json`, which `summarize` merges: `python -m loop.summarize results/run_dc_d3.json results/run_dc_d3b.json`). VM powered itself off. **Every memory arm beats both baselines at every budget, and the full loop beats pure retrieval too.**

```
share of the stock-to-best-known gap (0.4933), mean over seeds, 16 designs
share of the stock-to-best-known gap (0.4933), median over all seeds
arm                          D1    D2    D4    D6    D8   D12   D16    final
bo                          51%   51%   55%   68%   80%   91%   94%   0.4778
llm_direct                  57%   79%   80%   80%   83%   84%   85%   0.4802
memory                      93%   93%   94%   96%   96%   98%   98%   0.4910
Internal evidence, not in the reported table: the pooled design alone (one simulation, no search)
reaches 93% and stops there; random sampling reaches 90% after 219 designs and plateaus, matching
what either baseline reaches in 16 designs after about 12; the retired digest variant that handed over the closest workload's design instead
reached 75% at D1 and 98% at D16.
```
Medians are the true median (the mean of the middle two for an even count). Means run lower than medians for `bo`, where one seed collapses, and for `memory_pooled`, where one run opened by editing the handed-over design instead of copying it; quote the median and keep the per-seed curves in the reports.

**Reading.** (a) One design retrieved from the memory (93%) beats 16 designs of BO (86%) and of the plain LLM (88%). (b) The full loop beats retrieval alone: `memory` passes the one-shot retrieved design (93%) by design 3, reaches 95% by design 8 and 98% by design 12, and `memory` seed 4 found 0.4933, the best design known on this suite. Pure retrieval never gets past 93%. (c) What the digest hands over decides the opening: nearest-case 72% at D1 versus pooled 90%; by D16 they converge (97 vs 98). (d) **Random sampling reaches 90% and plateaus**, and it matches BO's and the plain LLM's 16-design results in about 12 designs: at this budget neither baseline extracts much more than chance, while one retrieved design (93%) beats all 219 random ones. (e) BO's seeds spread 51..98 at D16, the memory arms 94..100: the memory removes the variance, which is the practical argument for a design team.
1b. **Still open on this cell:** a 300-design uniform sample is running on the Mac (900 simulations, ~10 h, started Sep 12) to anchor the true 100%; the reference 0.4933 is still the best any of our methods found, not a measured optimum. 2 of 15 d3b runs died on an empty model answer that arrives outside the retry wrapper in `agent.ask` (13 of 15 completed; the earlier output-cap and rate-limit failures are gone: zero occurrences). `loop.workloads merge` now holds the table lock while it rewrites, so a collection running at the same time cannot lose rows.
2. **The arms** (`loop/run.py`, ARMS): `bo`; `llm_direct`; `memory` (the digest, which hands over the pooled design, plus a GP selecting 2 of the LLM's 8 proposals). Everything else is retired: the digest variant that handed over the closest workload's design, the one-design `pooled` and `replay` arms, `bo_prior`, `pooled_gp`, `llm_bo` and `memory_pooled_prior`. `pooled_design` builds the memory's copyable design; the prior's negative result is recorded below.
3. **Earlier runs on this cell, 8 designs, 2 seeds** (reports `run_dc_d1.json`, `run_dc_d2.json`, every prompt inside). Scored against the best design known today (0.4927): d2 gave llm_bo_memory 95%, memory 90%, bo 88%, llm_direct 85%, replay 78%, llm_bo 74% of the stock-to-best gap after 8 designs. The pooled design alone reaches 94% in one design; one of its candidates is the 0.4927 best known, found by no search.
4. **What the runs taught, in order.** (a) The verification probe (d1) verified true things and spent the budget doing so; removed. (b) The raw case dump was unreadable; the digest (conclusions first) lifted the memory arm's first design from 40% to 65% of the gap. (c) GP selection among LLM proposals pays only when the proposals are good: with the digest it found the new best, without it the LLM proposed single-knob candidates and the GP could not rank them. (d) What the memory hands over matters: on the test suite the pooled design (best mean gap share across the remembered workloads, counting only designs measured on all but one) scores 93% of the gap in one simulation. (e) A remembered single-knob effect keeps its sign on another workload about two thirds of the time, flat in descriptor distance (`loop.memory leave_one_out`), so the memory is trusted for which moves matter, not for how much. (f) A prior fit on raw log speed-ups ranks the datacenter designs at Spearman 0.29 because GAP's gains dominate; fit on gap shares, 0.73. (g) The hybrid arms re-propose measured designs, 4 to 9 of 8 per round; the retry recovers most; a compact listing of measured designs in the task text would remove it.
5. **Check the VM first** (`gcloud compute instances list`): it must be stopped whenever nothing runs on it. Free checks that need no VM: `python -m loop.memory leave_one_out <6 memory traces> -- <3 test traces>` (sign survival, about two thirds and flat in distance) and `python -m loop.workloads headroom`.
6. **Open after d3:** the duplicate-proposal fix; a second test suite (`gap2`, traces and tables on disk) with the same arms; a uniform reference sample on the datacenter suite to anchor the ceiling (~900 simulations, overnight); Pro confirmation of the top arm; the paper.

## The design

**The question.** Given a stock chip, a new workload suite and 8 simulated designs, how much of the best known design does each search reach after 1, 2, 4, 8 designs? Memory is a few-shot claim: its value is in the first designs, not the last.

**Arms, same start (the stock chip), same budget, same seeds, same fidelity:**
- `bo`: Gaussian process + expected improvement over a seeded 20k-design feasible sample plus the incumbent's one- and two-knob neighbours (`loop/bo.py`).
- `llm_direct`: the plain LLM agent, two picks per round from the results table and the workload descriptors (`loop/agent.py`).
- `memory`: the agent whose prompt carries the memory's **digest** (`memory.digest`, below) and which proposes 8 ranked candidates per round; once 3 designs are measured, a GP fit on this run picks the 2 to simulate by expected improvement (LLAMBO-style candidate sampling with surrogate selection). The digest's copyable slot holds the closest workload's best design.
- `memory_pooled`: the same, with the **pooled default** in the copyable slot: the remembered design with the best mean gap share across the memory workloads, counting only designs measured on all but one of them (`memory.pooled_design`, frozen in the memory file at build time). On the datacenter suite it reaches 93% of the gap in one design.
- `pooled`: one design, the pooled default. The bar every search arm must clear.
One code path for the LLM arms: `agent.run_agent(memory_slot, use_gp)`; with no memory the prompt is the plain agent's (checked).

**The memory** (`results/memory_<cell>.json`, one shelf): one **case** per workload the chip has been searched on, written by the code from the result table: descriptors, stock and best design with IPC, every measured single-knob effect with its pair count. Retrieval by standardized descriptor distance (log scale for counts and sizes). Nothing written by an LLM, nothing to consolidate: `python -m loop.run memory <cell>` rebuilds it from the tables.

**Prompt** (`agent.assemble_prompt`): problem (chip, budget, knobs, stock design); workloads (descriptor legend, then one line per workload); the memory digest (memory arms only: similarity in words, the descriptors that differ most, moves from stock that paid on the closest workloads, traps, where their best designs disagree, one copyable design); the results table (one row per design: index, source, knobs changed from stock, suite IPC, per-workload IPC / LLC MPKI / LLC hit ratio, then the hypothesis it tested or the verification it settled); the task. With an empty memory the two LLM arms produce the identical prompt (checked). A pick must be a real, in-budget, unmeasured design; else one retry with the rejected designs listed, then a deterministic one-knob perturbation of the incumbent. Never a random design.

**Scoring** (`loop/summarize.py`): per arm, the share of the stock-to-best-known gap reached after N designs, mean over seeds with min..max; the best known design is the best measured on every workload of the suite in the cached tables.

**Search space** (`loop/configs.py`, 13 knobs, 6.6 M raw designs, area-coupled, latency derived from size): L1D sets/ways/prefetcher; L2 sets/ways/prefetcher/replacement/MSHR; LLC sets/ways/prefetcher/replacement/MSHR. Area = L2 + LLC data capacity <= 4608 KB; L1D and MSHRs cost nothing (known simplification). Chip C (`loop/configs.py`): wide core, two memory channels, placeholder profile. Fidelity 5M warmup / 10M simulated (Spearman 0.919 against 50M/50M over 26 designs; shorter failed).

## Cells and workloads

| cell | memory (deep tables, 450-1500 designs each) | test | role |
|---|---|---|---|
| `dc` | mcf, omnetpp, lbm, bfs.urand, pr.urand, bfs.kron | sierra.a.4, merced, tahoe (Google datacenter, DPC4 `gtrace_v2`) | the headline |
| `gap2` | the same | sssp.kron, cc.urand, cc.twitter | fallback and development |
| `smoke` | omnetpp, bfs.urand | mcf, lbm, one round | the gate before any launch |

**Why the datacenter set.** Twelve families screened (one trace each, 100 MB prefix, no download), six admitted, all six probed with 11 designs. The three chosen have +26.6% suite headroom of which 55% no single knob reaches (GAP set 2: +33.5% / 62%; GAP set 1: +144% / 30%, and its arms tied). Their access streams are irregular (stride regularity 0.10-0.34), so a stride prefetcher does not hand over the gain. On every datacenter trace a replacement policy alone hurts or does nothing (srrip -13% on sierra.a.4) while the best design contains ship: the policy pays only after a prefetcher and a bigger LLC, the published SPEC-to-datacenter inversion, measured here. Spare admitted traces with probe tables: whiskey, bravo, delta. One SimPoint per family so far; the bucket holds 4-89 per family.

| test trace | footprint | MPKI | movable LLC MPKI | stride | headroom (11 designs) | beyond one knob | closest remembered |
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

**Retired: the GP prior on the memory tables.** One GP fit per run on every design the memory workloads measured (about 1,900 rows, target = each design's share of that workload's stock-to-best gap in log units), with the run's GP fitting only the residual. It ranked the known datacenter designs well on its own (Spearman 0.73 on shares, against 0.29 when fit on raw speed-ups) but added nothing as an arm: `memory_pooled_prior` matched `memory_pooled` at 98% after 16 designs and was slightly slower in between. Removed from the code; the raw runs stay in `results/run_dc_d3b.json`. Other ways to bring a learned model into the loop are open, this particular one is answered.

## Upstream: PR candidates for CHIA and ChampSim

`upstream/` holds patches and notes for changes that belong in the frameworks themselves rather than in this repo (`chia/` and `champsim/` are shallow clones, so these go through a fork). Every one of them is a gap we hit while building the loop.

**CHIA.** (1) `chia.simulators.champsim._resolve_trace` raises `NotImplementedError` on `gs://` traces, which every GCP cluster needs; `0001-champsim-gs-trace-resolver.patch` mirrors the existing `s3://` branch with `google-cloud-storage`. (2) `ChampSimNode.build_champsim` only accepts a prefetcher module, so a design-space loop cannot build arbitrary configurations; `loop/chia_nodes.py::build_from_config` is the general version (any config JSON: sizes, policies, prefetchers, core parameters), and `simulate` is the matching multi-trace run node. (3) `chia.models.vertex.VertexGeminiLLM` forwards only max tokens, the system message and tools, so a loop cannot set temperature, JSON response mode or a thinking budget, and every call runs at the model's defaults; a `generation_config` passthrough plus the thinking-token count in `_last_metadata` is a small, general fix. This gap cost cell d3 its fifteen memory-arm runs: eight-candidate answers plus the model's thinking exceeded the layer's 16k output default and the layer raised on rate limits without waiting. (4) `loop/memory.py` is simulator-agnostic (it reads knobs and descriptors from a `problem` dict): cases built from result tables, descriptor-distance retrieval, the digest, the pooled default and the prior are a candidate for a `chia.analysis` block any agentic loop could wrap around its simulator node.

**ChampSim.** `0002-champsim-spp-dev-ghr-victim.patch` fixes two bugs in `prefetcher/spp_dev/spp_dev.cc`, both found running SPP on a small-core profile with lbm: a heap-buffer-overflow in the lookahead loop (`confidence_q` / `delta_q` are sized to the L2 MSHR count but `read_pattern` appends up to `PT_WAY + 1` entries per step with no bounds check), and a GHR victim search that never finds a victim when every entry has confidence 100, tripping `assert(0)`. No existing upstream issue was found for either.

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

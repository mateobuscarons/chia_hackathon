# CHIA Hackathon — Hypothesis-Driven Cache Optimization

A3 workshop hackathon (agentic-arch.org). Deadline **Sep 20, 2026**: 4-page paper + open-sourced CHIA loop + results. Judged by the CHIA team (UC Berkeley SLICE lab) — an academic program committee that rewards novel loop mechanisms and reusable CHIA blocks, not raw IPC numbers.

## Core design: "rules that bet"

An LLM agent optimizes cache hierarchies (ChampSim) and distills an **architect's playbook**: explicit rules of the form *(condition, claim, worked example)*, each carrying a win/loss record and Brier score.

- **Conditions are chip-independent.** A workload is profiled once from its trace (`loop/trace_profile.py`, no simulator: working set, stride regularity, temporal locality, LRU miss-ratio curve from footprint theory, Xiang et al. ASPLOS 2013) and read against the target chip's public cache geometry (`champsim_problem.chip_descriptors`: working set over L2 / LLC size, predicted miss ratio at the chip's L1D / L2 / LLC). "The L2 is 2x too small for this program" means the same on every chip. The chip's own measured speed is never a condition.
- **Claims are measured, never stated.** The LLM writes the condition and the words; the simulator measures the claim (one knob changed, everything else equal) before a rule is admitted. Claims with |effect| < 1%, the wrong sign against the LLM's stated direction, no controlled pair, or a duplicate knob+value are recorded in `rejected_rules`.
- **Every forecaster bets before each simulation** (surrogate, rules, LLM hypotheses); the simulator settles. Rules that lose get **re-scoped** (condition sharpened), not deleted. Rules below 0.5 credibility stay silent. The analyst earns credibility like a rule and, when silenced, keeps one design per round to bet on ("right of reply").
- **Selection = expected improvement** over a GP that carries every forecaster's belief as a prior, plus one owed claim test per round, plus a **stall scan** (rules / full arms only): when the last round brought no improvement, one slot goes to the incumbent with a never-tried categorical value or an ordinal knob one step up / down.
- **Arms** (same design budget): `random`, `bo` (textbook BO, cold), `bo_pooled` (BO warm-started with every A/B run: statistical transfer), `textbook` (rules the LLM wrote before any result), `rules` (BO + verified playbook, no LLM at test time), `full` (rules + analyst in the loop). Baselines stay textbook: no scan, no rules.
- **Headline claim:** rules learned on SoCs A/B cut designs-to-target on unseen SoC C vs pooled BO and cold BO.

## Repo layout

See `README.md` for the file table. `loop/loop.py` is simulator-agnostic; `loop/champsim_problem.py` is the only ChampSim glue; `loop/chia_nodes.py` + `loop/run_chia.py` run it as a CHIA loop; `loop/socs.py` / `configs.py` hold **placeholder** SoC profiles, budgets and search spaces pending team review; `proposal.tex` is the accepted proposal; `paper/main.tex` the paper skeleton.

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

GCP: project `project-c23a6080-f5d0-4871-9cb`, ADC auth (`gcloud auth application-default login`), `aiplatform.googleapis.com` enabled; project-wide cap 32 vCPUs; VM recipe in `cluster/README.md`. Gemini runs on the project's credits; `ANALYST_MODEL=gemini-2.5-pro` is the only Pro the project can call (default `gemini-2.5-flash`).

## Status (Sep 7, 2026, 23:00) and next steps

**Where we are.** Mechanism v5 (above) is built, smoke-tested on GCP, and running its first experiment: Tier C seed 0, all arms, Pro analyst (`results/tierC_v5.log` on the VM `champsim-1`, output `experiment_tierC_v5_C.json`; launched 22:43, ~2 h 15). The Mac must stay free for the user's other project: **nothing runs locally**; the VM is stopped (not deleted) whenever idle.

**Next steps, in order.**
1. Read seed 0 v5: playbook (do the conditions read like an architect's, do they speak on C), designs-to-target per arm (`python -m loop.summarize results/experiment_tierC_v5_C.json`), ledger calibration. Autopsy if `rules`/`full` do not beat `bo_pooled`.
2. If it holds: seeds 1-2 (`FIRST_SEED=1 SEEDS=2 ... loop.run C tierC_v5_s12 results/experiment_tierC_v5_C_playbook.json`), then the paper-scale run (8 workloads / 5 seeds; ~25-30 USD on 32 vCPUs, ~20 h; request a CPU quota increase first; the extra workloads are the team's choice).
3. Paper (`paper/main.tex`, 4 pages, Sep 20): story = chip-independent conditions + measured claims + calibrated ledger; the Sep 7 autopsies (below) as the mechanism's motivation; limitations (placeholder SoC profiles; L1D miss-ratio prediction ~2x low; bo_pooled is near-deterministic across seeds; 5M warmup).
4. Repo: `early.py` into `summarize.py`; CHIA upstream PR (gs:// resolver, config-space ChampSim node, ledger blocks, trace-profile node).
5. Later experiments: sequential learning (A then B with A's playbook, so conditions are exercised across two training chips before C); reuse-distance descriptors beyond footprint theory; power/area objectives.

**How to read results.** `python -m loop.summarize <report.json> [more.json]` (censored medians, hit fraction, final best, AUC); `python -m loop.early <report.json> <log>` while a run is in progress. Tier B table (mechanism v4.1): `python -m loop.summarize results/experiment_v4_C.json results/experiment_v4full_C.json results/experiment_v4full_{mcf1,lbm2,omnetpp0,omnetpp2}_C.json`.

**Operational lessons.** Launch long runs detached on the VM (`setsid nohup ... &`); kill by process group (`kill -- -PGID`), never `pkill -f` a broad pattern (it killed 4 live runs on Sep 5; `pgrep -f` also matches the ssh shell carrying the pattern); result tables are the simulation cache, atomic-written, shared by all processes (a killed run loses only its in-flight round); suites evaluate one design on 4 workloads at once, so every shared file needs a lock (config write, per-binary build lock, table); the process pool forks, so editing files under a running job is safe but the job keeps the old code. Cost so far: ~13 USD compute, ~8 USD LLM.

## Results so far

- **Tier B (46k designs, single-workload problems, mechanism v4.1, Sep 5):** designs to 90% gain, median over 9 runs: `full` 8 (9/9 hit), `rules` 6 (8/9), `bo` 15, `bo_pooled` 15, `random` 19. Best final quality: `full`. Files `results/experiment_v4*.json`. This result used chip-relative conditions (baseline MPKI); it stands as a Tier B data point, not as evidence for v5.
- **Tier C (415k designs, latency coupled to size, MSHRs, suite of mcf+lbm+omnetpp+xalancbmk, chip C baseline 0.599, best found 0.746):** with v4.1 conditions, seed 0: `bo_pooled` 14, `bo` 18, `rules`/`full`/`random` >24. With the Sep 7 exploration fixes (stall scan, right of reply, adjacent-pair verification), seed 0: `rules` 7, `full` 12, `bo` 18, `bo_pooled` >24; seed 1: `bo` 0.742 vs `rules`/`full` 0.721 (area trap: L2 grown to 2048x16 makes LLC 4096 infeasible). Both runs had rules silent on C because of chip-relative thresholds; that is what v5 removes. Tables `results/tierC_*.json` (cache) and `results/profile_*.json` (workload profiles) are kept; superseded reports were deleted, findings live here.

## History: how the mechanism got here (autopsies, condensed)

- **v2 (Sep 3):** disagreement-based selection, rules bet at a flat 0.70, claims never settled (they needed paired runs that never existed), the LLM credited one knob for multi-knob gains. Failed on omnetpp.
- **v3 (Sep 4):** verified claims from controlled pairs, sibling-relative forecasts, every bet counts, magnitude-aware probabilities, fair `bo` / `bo_pooled` arms. Fixes verified, outcome missed 2 of 3 criteria: a rule's gain is a marginal effect at the baseline; capacity knobs had no rules (bigger sizes over budget on A/B); pooled BO collapsed across chips (fixed by GP target = log speedup over own baseline).
- **v4 / v4.1 (Sep 5):** EI acquisition with rule / hypothesis priors and one owed claim test per round; direction claims; chip-adapted gain; the analyst earns credibility from its bets (v4 hypotheses won 32% and still steered the GP). Tier B result above.
- **Sep 7 autopsies (Tier C):** (1) exploration collapse: two rule priors turned EI into pure exploitation, `llc_prefetcher` never tried in 24 designs; (2) thin playbook: ~21 designs per training chip, knobs never explored; (3) capacity rules untestable: direction claims only counted beyond the baseline, and that step was over budget on A/B; (4) analyst deadlock: one lost bet at credibility 0.5 silenced it for good, learn-phase ledgers held 6 lost bets; (5) LLM text/number sign contradictions rejected good rules; (6) area coupling: growing L2 makes the decisive LLC step infeasible; (7) chip-relative conditions: the LLC-prefetcher rule (+2.7% on C) and Pro's +23% LLC-capacity rule were silent on C because their MPKI thresholds came from A. Fixes: stall scan (categorical, then ordinal), right of reply, any-adjacent-pair verification, `direction` field, one-claim-one-rule, measured one-knob effects shown to the distiller largest first, prompts rewritten per Google's guide (system instruction, sections, one fictional example, temperature 0.3 / 0.7, thinking 8192 / 2048), Pro analyst, and finally v5 descriptors. A reference-machine fingerprint and per-chip silent-rule claim tests were built and removed the same night (an imaginary machine; and with good descriptors a silent rule should be silent for the right reason).

## Problem scope: real problems are hard problems

Per cache level: size, associativity, line size, replacement, prefetcher, MSHR/queue depths; plus coherence, interconnect, DRAM. Millions of combinations under area and power budgets, so knobs interact (area spent on L2 is area not spent on LLC). One detailed simulation costs hours per design per workload; teams evaluate tens to low hundreds of designs per cycle across 20-50 workloads. Every new SoC starts from the previous one: the transfer setting (chips A, B -> unseen C) *is* the industrial loop.

A problem is hard when four things hold: a wide space (tens of thousands of designs or more), a sparse near-optimal band, large headroom (best vs untouched chip differs by tens of percent), and knobs that interact through a constraint. Tier A (54 feasible designs) is a control and unit test: every method converges in 2-7 simulations. Tier B and C are the targets; baselines must scale with the problem (fair BO, never weakened), and hardness must come from realism, never from hobbling the comparison.

## Stretch arm: Merlin compiler node (HW/SW co-design)

A PhD teammate is integrating a **Merlin compiler node** into CHIA. For us that is a second knob axis: the `problem` dict is search-space agnostic, so compiler choices become knobs next to cache knobs, and `evaluate` becomes compile -> re-trace -> simulate. Rules can then say "when the compiler does X, cache knob Y stops mattering". **Blocker to verify first:** ChampSim traces are recorded from a fixed binary, so every compiler change needs trace regeneration. Do not start this before the cross-SoC result is in hand.

## Working rules

- Budget: **~257 EUR total** GCP credits. Track LLM cost per call; expensive runs need the user's approval with a cost estimate first; the user approves every launch on the VM.
- Nothing runs on the Mac. Stop the VM when idle.
- Code style: explicit, procedural, junior-readable; no clever one-liners; no flags for dead features: remove them. Commit messages informal, no authorship trailers.
- Domain decisions (search space, SoC profiles, descriptors, arms, prompt content) belong to the user and their PhD teammates — surface options, don't decide.

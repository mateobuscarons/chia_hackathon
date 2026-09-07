# CHIA Hackathon — Hypothesis-Driven Cache Optimization

A3 workshop hackathon (agentic-arch.org). Deadline **Sep 20, 2026**: 4-page paper + open-sourced CHIA loop + results. Judged by the CHIA team (UC Berkeley SLICE lab) — an academic program committee that rewards novel loop mechanisms and reusable CHIA blocks, not raw IPC numbers.

## Core design (frozen): "rules that bet"

An LLM agent optimizes cache hierarchies (ChampSim) and distills an **architect's playbook**: explicit rules with the form *(condition, claim, worked example)*, each carrying a win/loss record and Brier score.

- Every hypothesis/rule logs a quantitative prediction **before** each simulation; the simulator settles the bet. The LLM never grades itself.
- Experiments are chosen where forecasters (rules, hypotheses, surrogate) **disagree most**; agreed-upon outcomes are skipped.
- Rules that lose bets get **re-scoped** (condition sharpened), not deleted.
- Rules act mechanically: they prune the search space / warm-start the surrogate, weighted by track record.
- **Headline claim:** rules learned on SoCs A/B cut simulations-to-target on unseen SoC C vs surrogate transfer and cold start.
- Key baseline arm: agent seeded with textbook-only knowledge, no loop — the delta measures the loop's value (and defends against "the LLM already knew this").

## Repo layout

See `README.md` for the file table. Key points: `loop/loop.py` is simulator-agnostic; `loop/champsim_problem.py` is the only ChampSim glue; `loop/chia_nodes.py` + `loop/run_chia.py` run it as a CHIA loop; `loop/socs.py`/`configs.py` hold **placeholder** SoC profiles, budgets and search space pending team review; `proposal.tex` is the accepted proposal.

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

GCP: project `project-c23a6080-f5d0-4871-9cb`, ADC auth (`gcloud auth application-default login`), `aiplatform.googleapis.com` enabled. Gemini runs on the project's credits.

## Status (Sep 5, 2026, 12:30) and next steps

**Where we are.** Tier B cross-SoC result is in hand (see v4.1 below): the full loop hits the target in 9/9 runs with a median of 8 designs vs 15 for a fair BO baseline; rules-only median 6. Tier C (hard tier: 415k designs, coupled latency, MSHRs, suite objective) is built and smoke-tested on the 5 paper arms; it has NOT been run. Nothing exists on GCP; the Mac must stay free for the user's other project (do not launch local runs without asking).

**Next steps, in order.**
1. GCP hard-tier proof of concept: user runs the 6 commands in `cluster/README.md` (login, create VM, scp, bootstrap, `python -m loop.smoke C` on the VM, then `python -m loop.run C tierC`). ~4 h, ~$15 compute + ~$2 LLM. Fetch results, `python -m loop.summarize`, read the playbook. Delete the VM.
2. If the PoC passes (full/rules beat BO on the suite): scale to 8 workloads / 5 seeds for the paper (needs user OK; ~$20).
3. Paper (`paper/main.tex`, 4 pages, deadline Sep 20): story = verified rules + calibrated ledger; autopsy of v2/v3 failures as the mechanism's motivation; honest limitations (capacity rules untestable on small-budget chips; conditions are chip-dependent MPKI; 5M warmup).
4. Repo consolidation toward 4-5 core files (user request): merge `early.py` into `summarize.py`, `distill_tables.py` into `experiment.py`, `run_chia.py`+`chia_nodes.py` stay as the CHIA block; then README table.
5. CHIA upstream PR (gs:// resolver, config-space ChampSim node, ledger blocks).

**How to read results.** `python -m loop.summarize <report.json> [more.json]` (censored medians, hit fraction, final best, AUC); `python -m loop.early <report.json> <log>` while a run is in progress. Current Tier B table: `python -m loop.summarize results/experiment_v4_C.json results/experiment_v4full_C.json results/experiment_v4full_{mcf1,lbm2,omnetpp0,omnetpp2}_C.json`. Old v2 outputs live in `results/archive/`.

**Operational lessons.** Launch long runs with `nohup caffeinate -i ... &`; never `pkill -f` a broad pattern (a `pkill -f multiprocessing.spawn` killed 4 live runs on Sep 5); result tables are the simulation cache, atomic-written, shared by all processes; every new design needs a ChampSim build (~2 min; parallel across `champsim_N` trees on the VM).

## Mechanism v3 (Sep 4, 2026, evening): rules earn their bets

Autopsy of the Tier B omnetpp failure: rules bet at a flat 0.70, zero claim bets settled in the whole test (claim bets needed a paired run that never existed), so losing rules were never re-scoped; distilled claims credited one knob for multi-knob gains (LLM said +20-35%, controlled pairs measured ~0%). Fixes, all in `loop/`:
- **Verified claims** (`loop.verify_claims`): a rule's `gain_pct` is measured from controlled pairs in the training history (or one extra sim of baseline+knob, max 5 per training problem); rules with |gain| < 1% or the wrong sign are recorded in `rejected_rules`, never admitted. The LLM writes conditions and words, never the number.
- **Sibling-relative forecasts**: a rule predicts sibling(config with knob at baseline) x (1+gain); sibling = measured if in history else GP mean. Runs that settle a claim as a controlled comparison get a selection bonus.
- **Every bet counts**: dispute and claim bets both update a rule's record; rules below 0.5 credibility stay silent; a rule that loses 3+ bets (or any claim bet) on a problem is re-scoped once there (analyst may add a 2nd clause; mechanical tightening as fallback).
- **Magnitude-aware probabilities**: P = credibility x Phi((forecast-threshold)/GP std) + (1-credibility) x 0.5.
- **Surrogate**: ordinal knobs encoded as scaled log2, categorical one-hot, ARD Matern 5/2, vectorised `predict_many`, no "unseen value" bonus, LOO std only for the betting surrogate; softer rule priors. Spurious Apple BLAS "divide by zero in matmul" warnings are silenced (outputs verified finite).
- **Fair baselines**: new `bo` / `bo_pooled` arms = GP + expected improvement + kriging believer (no bets). `surrogate` arm is now labelled as our selector's ablation.
- **Reporting** (`summarize.py`): censored medians, hit fraction, final best at budget, area under best-so-far curve. Per-arm ledgers no longer include learn-phase bets.
- Bug fixed: pooled arms tagged configs with `soc`, so they never hit the result cache and re-simulated (~140 duplicate table entries re-keyed).
- Condition metrics extended with `L2C_hit_ratio`, `LLC_hit_ratio`, `LLC_over_L2C_mpki` (derived at load, no re-simulation).

Experiment 1 = `python -m loop.run_tierb v3` (learn A+B on mcf/lbm; test C on mcf/lbm/omnetpp; arms random, bo, bo_pooled, rules, full; 3 seeds; 24 sims). Pass criteria written before launch: rules no longer worse than random on omnetpp; full still fastest on mcf/lbm; rule win-rate > 60%.

**Experiment 1 (v3) result, Sep 5 01:20 (results/experiment_v3_C.json):** mechanism fixes verified (fabricated claims rejected at the door; 61 re-scope events; surrogate bets Brier 0.185 / win 0.69) but the outcome missed 2 of 3 criteria. Designs to 90% gain, median of 3 seeds: mcf random >24 / BO 12 / BO-pooled 12 / rules >24 / full >24; lbm 9 / 9 / >24 / **3 / 3**; omnetpp 19 / 23 / 12 / >24 / >24. Rule win-rate 0.50, bimodal (good rules 70-75%, lbm-learned rules firing on mcf/omnetpp 5-25%). Autopsy (all C tables): (1) a rule's gain is a *marginal effect at the baseline*: SPP on C/mcf = +31% from the untouched chip but +8% averaged over all 8 controlled pairs; LLC next_line +23% vs +1.5%; forecasting the gain for every design carrying the knob is wrong by construction. (2) The decisive knobs on C (LLC/L2 capacity) had no rules: bigger sizes were over budget on A/B, and the LLM's "increase LLC sets" was rejected because claims had to name a fixed value. (3) BO-pooled collapsed on lbm (stuck at baseline 0.756 for 24 designs): C's IPC is 2x A/B's and the shared GP never recovered. (4) LLM hypotheses won 26/51 bets (coin flip). (5) Learning distilled from 20 runs while the tables hold 200-500 measured designs per training problem.

## Mechanism v4 (Sep 5, 2026, 02:30): BO + verified rules

- **Acquisition = expected improvement** for every model-driven arm (kriging-believer batches). Rules and hypotheses act only through GP priors and bets; one slot per round goes to an owed **claim test** (baseline moved one claim-step) while credible rules have untested claims on this chip. Textbook BO is the special case with no rules/analyst. The v2/v3 disagreement selector survives only as the `surrogate` ablation arm.
- **Rule scope**: a rule forecasts/bets only on designs within 1 other knob change of the baseline (`RULE_SCOPE_OTHER_KNOBS`); beyond that it is silent and the GP carries the information.
- **Direction claims** on ordinal knobs: `{"knob": "llc_sets", "value": "up"}` = "one step larger than the baseline helps by X%". Verified by controlled pairs one step apart; transfers across budgets.
- **Chip-adapted gain**: a transferred gain is used at half strength (`TRANSFER_SHRINK`) until the first controlled pair on the new chip, then the measured mean here.
- **GP target = log(objective / own baseline)** per run (`reference` field on history entries); pooled data from other chips enters as speedup, predictions convert back. Fixes the pooled collapse.
- **Verification evidence = every measured design of the training problem** (result tables), not just the 20-run history. `python -m loop.distill_tables <in> <out>` re-distills a playbook from the tables (few LLM calls, <=5 verification sims per problem).
- Hypothesis priors carry 0.7x the LLM's stated confidence. Negative claims bet on the other side of the half-gain event.
- Experiment: `python -m loop.run_tierb v4 results/experiment_v4_playbook.json` (v3 playbook + table-distilled rules incl. direction rules; random/bo replay from cache).
- Housekeeping: obsolete v2 outputs moved to `results/archive/`; `loop/audit.py`, `loop/surrogate.py` removed. Target: collapse loop/ toward 4-5 core files; do not add one-off scripts.

**v4 result (Sep 5 07:15, results/experiment_v4_C.json):** rules arm (BO + verified transferred rules, no LLM at test time) median 6 designs to target over 9 runs vs 15 for BO cold, 15 for BO pooled, 19 random; omnetpp 5 (3/3) vs BO 23; lbm 6 vs 14; mcf 15 vs 13. Rule bets 57% win, bimodal (good rules 73-86%, Brier 0.15-0.22). Pooled BO no longer collapses. `full` (LLM hypotheses in the loop) hurt: hypotheses won 32% of bets (2/17 on omnetpp) yet warmed the GP at the LLM's stated 0.7-0.9 confidence. Fix (v4.1): the analyst earns credibility like a rule (starts 0.5 per problem, updates from its bets, silent below 0.5); rerun of the `full` arm only = `python -m loop.run B v4full results/experiment_v4_playbook.json full`.

**v4.1 result (Sep 5 12:20, results/experiment_v4full_*.json merged with experiment_v4_C.json):** with the analyst earning credibility, `full` hits the target in 9/9 runs, median 8 designs (mcf 9, lbm 4, omnetpp 9), best final quality of every arm; `rules` median 6 (8/9); BO 15 (8/9); BO-pooled 15 (9/9); random 19 (5/9). Reproduce table: `python -m loop.summarize results/experiment_v4_C.json results/experiment_v4full_C.json results/experiment_v4full_{mcf1,lbm2,omnetpp0,omnetpp2}_C.json`. Note: 4 of the 9 full runs in experiment_v4full_C.json were killed by a stray pkill and re-run into the per-seed files. Local machine freed for the user from 12:22; everything further runs on GCP.

**Tier C (hard tier) built Sep 5:** `SEARCH_SPACE_C` = Tier B + l2/llc MSHR (414,720 designs), `LATENCY_FROM_SIZE` (ChampSim derives latency from sets*ways), suite problems (`make_suite_problem`: geomean IPC over workloads, conditions on suite mean/max descriptors), parallel ChampSim build trees (`champsim_N`, shared `champsim_bin/`), env knobs `PARALLEL_RUNS`, `SIM_THREADS`, `CHAMPSIM_TREES`, `CHAMPSIM_BUILD_SHARE`. Runner: `python -m loop.run C <tag>`; smoke: `python -m loop.smoke C`. GCP one-VM recipe in `cluster/README.md`.

**GCP gate:** no cloud resources are created until Experiment 1 shows the mechanism works; then hand the user cost + time + steps and let them trigger it. Compute API is enabled; PREEMPTIBLE_CPUS quota is 0 (on-demand only, N2 200 / C2D 100 vCPU); europe-west1-b was out of c2d stock on Sep 4.

## Problem scope (Sep 4, 2026): real problems are hard problems

**What a real memory-hierarchy design problem looks like.** Per cache level: size, associativity, line size, replacement, prefetcher and its parameters, MSHR/queue depths; plus coherence, interconnect and DRAM settings. Millions of combinations under area and power budgets, so knobs interact (area spent on L2 is area not spent on LLC). One detailed simulation (gem5, RTL) costs hours to a day per design per workload; teams evaluate tens to low hundreds of designs per cycle across 20-50 workloads. Every new SoC generation starts from the previous one: the transfer setting (chips A, B -> unseen C) *is* the industrial loop.

**Hardness, measured.** A problem is hard when four things hold together: a wide space (tens of thousands of designs or more), a sparse near-optimal band (a few percent or less of feasible designs within 2% of the best), large headroom (best vs untouched chip differs by tens of percent, so "good" and "best" are far apart), and knobs that interact through a constraint. Measure it with a small random probe: if random search reaches 90% of the gain in ~10 draws, the problem is easy and statistics alone suffice. Evidence: on Tier A (54 feasible designs, 8-30% of them near-optimal) every method converges in 2-7 simulations and nothing beats a GP; on the hard Tier B problems (46k designs, 1-3% near-optimal, +54-70% headroom) no statistical method reached within 2% of the best in 24 simulations and the full loop did.

**Scope going forward.** Tier A is a control and a unit test, not a target. The target is the hard regime: Tier B and harder, evaluated at explicit hardness levels, with the main claim made on the hard level. Baselines must scale with the problem (a fair Bayesian-optimization baseline, not a weakened one), and hardness must come from realism (more levels, more interacting knobs, more workloads, real budgets), never from hobbling the comparison.

## Stretch arm: Merlin compiler node (HW/SW co-design)

A PhD teammate (Berkeley, close to the CHIA project) is integrating a **Merlin compiler node** into CHIA so an agent can change the compiler/compilation agentically. For us that is a second knob axis: the `problem` dict in `loop/loop.py` is search-space agnostic, so compiler choices become knobs next to cache knobs, and `evaluate` becomes compile -> re-trace -> simulate. Rules can then say "when the compiler does X, cache knob Y stops mattering", which is the co-design story the CHIA team wants. **Blocker to verify before committing:** ChampSim traces are recorded from a fixed binary, so every compiler change needs trace regeneration (tracer availability, time per trace). Do not start this before the cross-SoC result is in hand.

## Working rules

- Budget: **~257 EUR total** GCP credits. Track LLM cost per call; expensive runs need the user's approval with a cost estimate first.
- Wrap long-running shell commands in `caffeinate -i`.
- Code style: explicit, procedural, junior-readable; no clever one-liners; small chunks with user approval between them.
- Domain decisions (search space, SoC profiles, claims) belong to the user and their PhD teammates — surface options, don't decide.

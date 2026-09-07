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
2. If the PoC passes (full/rules beat BO on the suite): scale to 8 workloads / 5 seeds for the paper (needs user OK; ~$25-30 compute on 32 vCPUs, ~20 h; request a CPU quota increase first).
2a. **Claim direction as its own field** (user decision Sep 7, after the PoC, not mid-run): in the Tier C learn phase the verifier rejected two rules ("more L2 sets/ways hurts when L2 hit ratio is already high", measured -1.4%/-1.6%) as "wrong sign" because the LLM wrote "hurts" in the text but a positive `gain_pct`. Fix: ask the LLM for `direction: helps|hurts` and check the measured sign against that, not against the sign of a number we never use; tidy the distill prompt at the same time. Keep the verifier from over-fitting to formatting details. No relaunch needed: re-distill from the tables (`loop.distill_tables`) and review `rejected_rules` for what was lost.
2c. **Analyst credibility deadlock** (found Sep 7 in the Tier C learn ledger: 6 settled bets in 20 rounds, all lost). The analyst starts each problem at 0.5 with PRIOR_WEIGHT 2; one lost bet puts it at 0.33 < CREDIBLE_CONFIDENCE and it goes silent; silent, its designs are rarely selected by EI, so it never bets again and can never recover. `full` then degrades to `rules`, and the distiller reads an empty ledger. Options for the user: (1) heavier prior / lower floor; (2) keep one claim-test slot per round for a silenced analyst so it can earn its way back (rules already have this path); (3) carry credibility across problems. Check the `full` ledgers of the Tier C runs before choosing.
2b. **Stronger analyst model** (user request, Sep 7): rerun with a pricier model than `gemini-2.5-flash` (e.g. Gemini 2.5 Pro; `MODEL`/`PRICES` in `loop/analyst.py`) as an ablation next to the flash runs, comparing hypothesis win-rate/Brier and designs-to-target. LLM spend to date is ~7 USD for the whole project; the user is happy to spend well beyond $2-5 per run on this. Keep the cost estimate before launch.
3. Paper (`paper/main.tex`, 4 pages, deadline Sep 20): story = verified rules + calibrated ledger; autopsy of v2/v3 failures as the mechanism's motivation; honest limitations (capacity rules untestable on small-budget chips; conditions are chip-dependent MPKI; 5M warmup).
4. Repo consolidation toward 4-5 core files (user request): merge `early.py` into `summarize.py`, `distill_tables.py` into `experiment.py`, `run_chia.py`+`chia_nodes.py` stay as the CHIA block; then README table.
5. CHIA upstream PR (gs:// resolver, config-space ChampSim node, ledger blocks).

**How to read results.** `python -m loop.summarize <report.json> [more.json]` (censored medians, hit fraction, final best, AUC); `python -m loop.early <report.json> <log>` while a run is in progress. Current Tier B table: `python -m loop.summarize results/experiment_v4_C.json results/experiment_v4full_C.json results/experiment_v4full_{mcf1,lbm2,omnetpp0,omnetpp2}_C.json`. Old v2 outputs live in `results/archive/`.

**Operational lessons.** Launch long runs with `nohup caffeinate -i ... &`; never `pkill -f` a broad pattern (a `pkill -f multiprocessing.spawn` killed 4 live runs on Sep 5); result tables are the simulation cache, atomic-written, shared by all processes; every new design needs a ChampSim build (~2 min; parallel across `champsim_N` trees on the VM).

## Tier C v2 (Sep 7, 2026, evening): autopsy fixes verified on seed 0

Fixes a/c/d from the autopsy below (commit `444fe99`): direction claims verified by any adjacent controlled pair (`forecast.claim_sibling`); a **stall scan** slot for rules/full only (`loop.EXPLORE_ON_STALL`: when the last round did not improve the best design, one slot goes to the incumbent with a never-tried categorical value, the GP-most-uncertain first; never the last slot); a silenced analyst keeps one bet per round on its most confident hypothesis ("right of reply"); round logs (hypotheses, chosen, claim tests, scans) persisted in the report. Baselines bo/bo_pooled stay textbook (user decision). `run.py` takes `SEEDS` and `FIRST_SEED` env overrides.

**Learn phase with the fixes:** 8 rules admitted, 2 rejected (was 1 / 6); the analyst replied 9 times instead of going mute after round 1; the stall scan on A found `llc_prefetcher=next_line` (0.323 -> 0.330), which became RULE-004 (+2.2%, 2 pairs). Direction rules now rest on 3-6 pairs. Still missing: an LLC-capacity rule (the LLM did not propose it despite a +33% pair on A; prompt should ask for the largest measured effects first, see 2b).

**Seed 0 result (`results/experiment_tierC_v2_C.json`, `python -m loop.summarize`):** designs to 90% of gain: **rules 7, full 12**, bo 18, bo_pooled >24, random >24; final best rules 0.7454 (= reference best), full 0.7423, bo 0.7316, bo_pooled 0.7292. Yesterday's seed 0: rules >24, full >24, bo_pooled 14. Calibration on shared dispute bets is weak for everyone (surrogate 0.50, rules 0.30 over 10, hypotheses 0.47) and should be looked at once seeds 1-2 are in. Seeds 1-2 of all 5 arms launched 19:47 with the same playbook (`results/tierC_v2_s12.log`, output `experiment_tierC_v2_s12_C.json`, ~2 h, ~$3); table = `python -m loop.summarize results/experiment_tierC_v2_C.json results/experiment_tierC_v2_s12_C.json`.

## Tier C proof of concept, seed 0 (Sep 7, 2026): rule transfer lost to raw-data transfer

Run on GCP (`champsim-1`, c2d-standard-32; 3 races fixed first: atomic config write, per-binary build lock, unique `.tmp`; suites evaluate one design on 4 workloads at once). Learn on A+B suites (10 rounds x 2 designs each), test on C suite, 24 designs. The user stopped it after seed 0 (baselines' seeds 1-2 partly cached). Files: `results/experiment_tierC_base_C.json` (random; bo/bo_pooled seed 0 only in the logs), `results/experiment_tierC_rules_C.json` (rules, full with playbook v2), `results/playbook_tierC_v1_thin.json` (learn run, 1 rule), `results/playbook_tierC_v2.json` (re-distilled with the direction field, 2 rules), tables `results/tierC_*.json`. Compute ~$6, LLM ~$0.10.

**Result (seed 0, C suite baseline 0.599, best found 0.746 = +24.6%).** Designs to 90% of gain: bo_pooled **14**, bo 18, random >24, rules >24 (stuck at 0.726 from design 11), full >24 (stuck at 0.718 from design 6). The 0.726 -> 0.746 gap is one knob: LLC `next_line` prefetcher (+2.7% on top of SPP + L2 512x16 + LLC 4096x16).

**Autopsy.**
1. *Exploration collapse.* rules and full tried `llc_prefetcher=no` in 24/24 designs and `spp_dev` in 23/24; bo cold tried `next_line` in 15/18, random 15/25, bo_pooled 8/20. The loop has no space-filling start: round 1 is EI over a one-point GP plus priors. With two rule priors the GP has an attractor from design 1 and EI exploits it; bo cold's round 1 is std-driven and diverse. `full` produced zero improvement over its last 18 designs.
2. *Thin playbook.* 2 verified rules from ~21 measured designs per training chip. On A/B `llc_prefetcher!=no` was measured in 2 designs (A) and 0 (B): no evidence, so no rule about the knob that decided C. The pooled GP still used A's two points.
3. *Capacity rules structurally untestable* (same as Tier B autopsy item 2, still open): `llc_sets` 1024->2048 is the largest single effect (+33% A, +9% B, +8.5% C; 2048->4096 +4.6% on C), but a direction claim is verified only on the claimed side of the baseline (`forecast.claim_sibling`), the baseline is 2048 on every chip, and 4096 is over budget on A and B. Rejected as "no controlled comparison" 3 times.
4. *Analyst deadlock confirmed.* Learn phase: hypothesis bets only in round 1 of each problem (all lost) -> credibility 0.33 -> silent for rounds 2-10; ledger shown to the distiller = 6 lost bets. On C (`full`): won its round-1 bet, then only 1 more bet in 23 designs (its designs rarely selected). Hypotheses are not persisted (`run_loop` returns round logs with them; `experiment` drops them), so what the analyst proposed cannot be audited.
5. *What worked.* Verifier decisions were right (no-effect claims rejected; direction field removed the formatting rejections; one genuine contradiction still caught). Rule bets on C: 8/10 won, Brier 0.18; both claims confirmed on C in rounds 1-2. Surrogate dispute bets: 2/7, Brier 0.39. 1 of 10 allowed verification sims used.

**Fix candidates (user decides).** (a) a space-filling initial batch for every model arm, or one exploration slot per round for an untried categorical value near the incumbent; (b) more learn-phase evidence (20 rounds on A/B) so knob coverage exists; (c) verify direction claims with any adjacent controlled pair, not only above the baseline; (d) analyst deadlock fix + persist round logs; (e) stronger model + cleaner prompts (2a/2b above).

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

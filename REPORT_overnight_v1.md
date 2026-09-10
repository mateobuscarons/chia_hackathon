# Overnight run `v1`: full report and autopsy

Purpose of this document: everything the first end-to-end run of the v7 mechanism produced, with an autopsy of every step and every arm, so an outside expert can judge what worked, what failed and why, and recommend next steps. Numbers are copied from the run's report (`results_vm/experiment_v1_gap.json`), the step logs (`results_vm/v1_*.log`) and the playbook (`results_vm/experiment_v1_playbook.json`). Nothing here is interpolated.

For the design being tested, see CLAUDE.md ("The mechanism (v7)", decisions 1-24). One-sentence version: rules whose conditions are written on a workload's measurable habits (profiled once from its trace) and whose claims are settled by the simulator on old chips A and B should still fire correctly, and still cut the search, on an unseen chip C and an unseen workload class (graph analytics, GAP traces), against matched-budget baselines.

---

## 1. What ran

One unattended chain on a 32-vCPU GCP VM (`cluster/chain.sh v1`), Gemini 2.5 Pro as the analyst, everything through the repo's code as committed at `a44560e`.

| step | what | wall time | notes |
|---|---|---|---|
| smoke | two arms, one round, chip C, mcf+lbm, through CHIA (Ray in-process, builds and simulations as CHIA tasks, arms as Ray tasks) | 5 min | exit 0; real simulations from a worker, profiler recorded them |
| learn set | structured design list on chips A and B over the five training workloads (mcf, omnetpp, lbm, fotonik3d, cam4): baseline, every feasible one-knob change, the L2-prefetcher x LLC-size and L2-size x LLC-size pairs, then the chips' cached designs; 130 designs on A, 117 on B; 625 new simulations | 2 h 49 min | fotonik3d is the slow trace (168 min for its column on A alone) |
| learn | distill the playbook from the A and B tables (one Pro call per chip), verify every claim by controlled comparison, freeze the pooled warm start (247 rows), run gate G-language | 1.5 min | 10 rules admitted, 0 rejected |
| gap2 | cell 2, the headline cell: chip C (unseen), suite of bfs.urand + pr.urand + bfs.kron (unseen class), 7 arms x 2 seeds x 24 designs (12 rounds of 2), through CHIA | 2 h 52 min | 0 failed jobs |
| fidelity SPEC | chip C, the 20 best cached designs by the mcf+omnetpp+lbm geomean plus baseline and 5 random, at 50M warmup / 50M simulated instead of 5M / 10M | 1 h 3 min | gate G0 |
| fidelity GAP | same with the top 10 GAP designs; finished, tables still on the VM disk (VM shut down as scripted) | ~30 min | not yet read |

Compute: about 7.5 h of VM, roughly 11 USD. LLM: 0.55 USD for the whole night (distillation, textbook rules, replies, the plain agent's 48 picks). The VM shut itself down at about 03:40.

---

## 2. The playbook that was learned

Distilled from the tables of chips A and B (five workloads each). The LLM wrote each rule's condition and words over the measured one-knob effects; every claim's magnitude is the mean of its controlled pairs, never the LLM's number. Thresholds were then fitted from the per-workload evidence: where the effect appeared ("present") and where it did not ("absent").

| rule | chip | condition (after fit) | claim | measured | pairs | present | absent |
|---|---|---|---|---|---|---|---|
| 001 | A | movable_l2_mpki >= -0.00007 | l2_prefetcher = spp_dev | +29.8% | 6 | mcf, lbm, fotonik3d, cam4 | omnetpp |
| 002 | A | movable_llc_mpki >= 0.072 | llc_sets up | +11.8% | 15 | mcf, omnetpp, lbm, cam4 | fotonik3d |
| 003 | A | movable_llc_mpki >= 0.072 | llc_prefetcher = next_line | +8.5% | 7 | omnetpp, lbm, fotonik3d, cam4 | mcf |
| 004 | A | mem_accesses_per_kinstr >= 219 | llc_replacement = ship | +6.4% | 5 | mcf, omnetpp, cam4 | lbm, fotonik3d |
| 005 | A | movable_l2_mpki >= 1.85 | l2_ways up | +4.0% | 13 | mcf, omnetpp, cam4 | lbm, fotonik3d |
| 006 | B | mem_accesses_per_kinstr >= 219 | l2_prefetcher = spp_dev | +20.9% | 13 | mcf, lbm, fotonik3d, cam4 | omnetpp |
| 007 | B | movable_llc_mpki < 0.489 | llc_ways down | -13.6% | 1 | mcf, omnetpp, fotonik3d, cam4 | lbm |
| 008 | B | stride_regular_fraction >= 0.0135 | l2_prefetcher = next_line | +14.9% | 7 | all five | none |
| 009 | B | movable_llc_mpki < 0.489 | llc_sets up | +12.6% | 9 | mcf, omnetpp, fotonik3d, cam4 | lbm |
| 010 | B | movable_l2_mpki >= 0.126 | l2_prefetcher = va_ampm_lite | +12.5% | 4 | mcf, lbm, fotonik3d, cam4 | omnetpp |

Observations on the playbook itself:

- **Every claim is real and correctly signed on its own chip.** The verifier admitted all ten; none of the LLM's magnitudes disagreed with the measurement by more than 1.6 points.
- **Several conditions are degenerate, i.e. always true.** Rules 004 and 006 condition on accesses per kilo-instruction >= 219, which is the training suite's minimum; rule 008 on stride regularity >= 0.0135, also the minimum; rule 001's threshold is effectively zero. The threshold fit does what it is told: when the effect is present on the workload with the lowest descriptor value and no absent workload lies below it, the threshold drops to that value and the rule speaks everywhere. These rules carry no workload discrimination at all.
- **Rule 009 is the failure that matters for the language.** Its text says "if a workload does not benefit from a larger LLC ... increasing LLC sets", its condition says movable LLC MPKI below 0.489, and its claim says a bigger LLC helps +12.6%. The cause is the training chip: B's area budget allows only 512 KB more cache than its baseline, so the movable-MPKI descriptor for every training workload on B lies between 0.00 (fotonik3d) and 0.49 (mcf), with lbm, the one workload where capacity did not help, at 0.08, in the middle. No threshold separates them; the fit produced "less than 0.49", which is "always" on B and "always" on any capacity-flat workload elsewhere. Chip A, whose budget doubles the cache, produced the sensible rule 002 (movable LLC MPKI >= 0.072). Lesson: a descriptor can only be fitted on a chip whose budget lets that descriptor vary.
- **The same claim appears twice across chips.** Rules 001 (A) and 006 (B) both say "spp_dev helps"; the playbook merge only de-duplicates within one chip. This has a mechanical consequence in selection (section 5).

---

## 3. Gate G-language (free, before any cell)

The pre-registered question: read against chip C and each held-out workload's own descriptors, do the rules fire where the class can use them and stay silent where it cannot, and does each speaking rule's sign match the one-knob effects already cached on that workload (40 designs each)?

| workload | movable LLC / L2 MPKI on C | rules speaking | verdict per rule |
|---|---|---|---|
| bfs.urand (capacity-sensitive, stride 0.27) | 14.1 / 15.2 | 8 of 10 | 001, 006 spp_dev: claimed +29.8 / +20.9, measured no->spp +13.6 (right sign, over-sized). 010 va_ampm_lite: +12.5 claimed, +16.5 measured. 002 llc_sets up: +11.8 claimed, **+10.5 measured**. 004 ship: +6.4 claimed, +11.5 measured. 008 next_line: +14.9 claimed; measured next_line vs no +5.2, vs ip_stride -4.5. 005 l2_ways up: +4.0 claimed, 16->8 measured +1.2 (i.e. down helps slightly: wrong, small). 003 llc next_line: no pair cached. |
| pr.urand (capacity-flat, stride 1.00) | 0.0 / 0.0 | 6 of 10 | prefetcher rules right, hugely under-sized (spp +186%, next_line vs no +30%). 004 ship: measured +0.5, below the 1% bar. **009 llc_sets up fires: measured -0.7%. Wrong.** 007 fires (no pair). 002 correctly silent. |
| bfs.kron (capacity-flat, stride 0.52) | 0.0 / 0.0 | 6 of 10 | same pattern: prefetcher rules right (spp +151%), ship +0.8 (below bar), **009 fires: measured -1.2%. Wrong.** 002 correctly silent. |

Verdict: the *firing* and the *sign* transferred for the prefetcher rules and for chip A's capacity rule (fires only where capacity moves the workload, exactly as pre-registered). Chip B's capacity rule failed for the reason in section 2. The *magnitudes* did not transfer in either direction: claimed +29.8% where +13.6% is measured on one workload and +186% on another.

Two caveats the gate itself exposes:

1. The gate is evaluated per workload, but the loop scores a **suite** and checks conditions on the suite's descriptor **means** (and maxima). On the suite the loop saw movable LLC MPKI mean 4.70, movable L2 MPKI mean 5.07, accesses 302, stride 0.595, footprint ratio 6.16. On those, rules 001, 002, 003, 004, 005, 006, 008, 010 spoke and rules 007 and 009 stayed silent (4.70 is not below 0.489). So the degenerate B rule never acted inside the cell; the per-workload failure is a finding about the language, not about this run's search.
2. "Stays silent on a capacity-flat workload" is not a property the suite objective can reward or punish: a bigger LLC is +10.5% on one third of the suite and about -1% on two thirds, so on the geomean it is right to grow the LLC anyway. The prediction instrument works at workload level; the search works at suite level. The paper must say which is which.

---

## 4. Cell 2 results (chip C, GAP suite, 2 seeds, 24 designs)

The reference used below is the best design any arm found in this run, suite geomean IPC 0.6426 (found by cold `bo`, seed 1, design 12: L1D next-line prefetcher, L2 256 sets x 4 ways with va_ampm_lite, 64 L2 MSHRs, LLC 8192 x 8 with no LLC prefetcher and SRRIP, 32 LLC MSHRs). Baseline: 0.2629. The provisional held-out reference from the screening sweep (0.6110) is below this run's best and is not used.

Designs bought until best-so-far reaches 90 / 95 / 99% of the gap between baseline and reference; median over the two seeds, with how many seeds got there; then final best and area under the best-so-far curve.

| arm | to 90% | to 95% | to 99% | final best (s0, s1) | AUC % |
|---|---|---|---|---|---|
| llm_direct | **6** (2/2) | 20.5 (2/2) | 20.5 (2/2) | 0.641 (0.641, 0.642) | 130.9 |
| rules | 12.5 (2/2) | 22 (1/2) | >24 (0/2) | 0.616 (0.609, 0.624) | 124.3 |
| full | 13.5 (2/2) | 20.5 (1/2) | >24 (0/2) | 0.620 (0.610, 0.630) | 124.9 |
| textbook | 14 (2/2) | >24 (0/2) | >24 (0/2) | 0.609 (0.609, 0.609) | 113.0 |
| bo (cold) | 14.5 (1/2) | 18.5 (1/2) | 18.5 (1/2) | 0.623 (0.603, 0.643) | 130.2 |
| random | >24 (0/2) | >24 | >24 | 0.584 (0.591, 0.576) | 112.6 |
| bo_pooled | >24 (0/2) | >24 | >24 | 0.546 (0.546, 0.546) | 107.5 |

Calibration (Brier score, 0 perfect, 0.25 coin flip; win rate = leaned the right way):

| forecaster | question | bets | Brier | wins |
|---|---|---|---|---|
| surrogate (GP) | shared dispute events | 112 | 0.225 | 69% |
| rules | shared dispute events | 176 | 0.278 | 35% |
| rules | their own claims (controlled pairs) | 7 | 0.09-0.20 | 7 won, 0 lost |
| analyst (replies) | shared dispute events | 10 | 0.244 | 50% |
| llm_direct | its own forecast within 5% | 46 | 0.181 | 76% |

Pre-registered predictions:

- **P1 held, hard:** the SPEC-pooled GP degraded to the worst arm and never recovered (section 5.4).
- **P2 partly:** `rules` degrades less than `bo_pooled` (trivially, since `bo_pooled` collapsed) and beats `random`, but only ties `textbook` and cold `bo`. Against the pre-registered gate ("rules / full beat bo_pooled and llm_direct on designs-to-target"): beats the first, loses to the second.
- **P3:** `textbook` mediocre, yes; "degrades least" cannot be judged without cell 1.
- **P4 fails:** `llm_direct` was not mediocre; it was the best arm on every measure, including its own calibration.

Two seeds and 24 designs are a direction, not a verdict. The direction is clear enough to act on before buying ten seeds.

---

## 5. Autopsy per arm

The best design region on this suite, found independently by cold `bo` (seed 1) and `llm_direct` (both seeds): an L1D next-line prefetcher, va_ampm_lite at the L2, a small L2, the largest LLC, no LLC prefetcher, SHIP or SRRIP. The second-best region, where `rules`, `full`, `textbook` and `bo` seed 0 all ended, is the same with spp_dev instead of va_ampm_lite and an LLC next-line prefetcher on: about 0.60-0.61 against 0.64.

### 5.1 `rules` (GP + playbook shifts, no LLM at test time)

Round 1 in both seeds went straight to the spp_dev basin: seed 0 designs 1-4 are L2 spp_dev + LLC 8192x8 + LLC next_line + SHIP, varying only L2 geometry (0.53-0.54); seed 1 design 1 was already 0.596 (same recipe with SRRIP and a 256x4 L2). Why: the initial mean shifts before any observation, in log-speedup units, were 001 spp_dev **+0.182**, 006 spp_dev **+0.133** (the same value, stacked: +0.315 together, i.e. a +37% tilt on every spp_dev design), 008 next_line +0.097, 010 va_ampm_lite +0.082, 002 llc_sets up +0.078, 003 llc next_line +0.057, 004 ship +0.044, 005 l2_ways up +0.027, with the cap at 0.391. The two spp_dev rules from two chips stacked into the single largest tilt in the playbook, four times the tilt on va_ampm_lite, which is the value that actually wins on this class. The mechanism did exactly what it was told; the playbook told it the wrong relative sizes.

Then the search stalled early (seed 1 from round 3, seed 0 from round 4), and each stall spent one slot on an owed claim test: baseline + spp_dev (0.530), baseline + va_ampm_lite (0.501), baseline + SHIP (0.274). All three claim bets were won (the signs are right), but each cost one of 24 designs at half the incumbent's value, and none informed the search where it was (far from the baseline). Stall scans took another two to three slots (LLC MSHRs, LLC ways). In seed 0 the arm never tried va_ampm_lite in the good region; in seed 1 it did at design 10 (0.6095) and reached 0.624 at design 19.

Dispute bets: the rules lost 35% because a rule's forecast is the design's sibling times (1 + claimed gain), and the claimed gains were 1.5 to 2.2 times too large for this class; the event is settled at the midpoint between the GP's and the rule's forecast, so an over-promising rule loses to a GP that is merely closer. The rules' own claim bets, which only ask for the sign and half the magnitude, were 7-0. The two readings together are the finding: **sign and firing transferred, size did not.**

Re-scoping: 14 mechanical re-scopes across `rules` and `full`. Each moved a rule's first clause just past the suite's descriptor value (e.g. rule 001's condition became movable L2 MPKI > 5.07, the suite mean being 5.068), which silences the rule for the rest of the run. Intended behaviour, and it means a rule that loses three dispute bets early is muted for the remaining rounds even when its sign is right.

### 5.2 `full` (rules + the analyst's right of reply on a stall)

Identical to `rules` for the first rounds (same seed, same picks) until the first stall, then one design per stall from the analyst with a forecast. Ten replies over the two seeds, 5 won / 5 lost as dispute bets. Their content was sensible and specific: "the incumbent's L2 and LLC MPKI suggest MSHRs are saturated, raise them" (0.543, marginal), "switch spp_dev to va_ampm_lite, as RULE-010 says" (seed 1, round 5: this is what produced 0.6159 at design 10 and the arm's later 0.630), "add an LLC next-line prefetcher" (wrong on this class: the best designs have none), "raise L2 associativity from 4 to 8" (twice, both marginal), "try ip_stride" (0.505, wrong). The analyst's edge over `rules` is real but small on two seeds (13.5 vs 12.5 to 90%, 0.620 vs 0.616 final); it comes from the va_ampm_lite reply.

### 5.3 `bo` (cold GP, expected improvement, no rules)

Seed variance dominates: seed 1 found va_ampm_lite at design 4 (0.616) and the run's overall best at design 12 (0.6426); seed 0 exploited the spp_dev basin from design 4 and ended at 0.603 with six stalled rounds. Pure BO has no stall response by design (baselines stay textbook), so a stalled seed stays stalled. On this cell, cold BO with a lucky seed is as good as anything.

### 5.4 `bo_pooled` (GP warm-started with the 247 frozen training rows from A and B)

Complete collapse, and worth understanding. Both seeds produced the **same 24 designs in the same order**: every one of them has L2 2048 sets x 16 ways with spp_dev, LLC 2048 x 8 with an LLC next-line prefetcher, SHIP or DRRIP, no L1D prefetcher, and only the MSHR counts vary. IPC 0.531-0.5455, flat from design 1. Diagnosis: on chips A and B the best SPEC designs are exactly that shape (a large L2 plus spp_dev), so the pooled GP's posterior mean is high and its variance low in that corner; expected improvement then keeps sampling the corner's MSHR neighbours, each observation on C confirms roughly the same value, and the model never becomes uncertain enough anywhere else to leave. The seed only shuffles ties, hence identical runs. This is precisely the predicted failure of data transfer to an unseen workload class (P1), and also a caution: the arm has no exploration escape, so once it is wrong it stays wrong for 24 designs. Whether a competent practitioner would ship pooled BO without an exploration term is a fairness question for the paper; as specified (textbook BO, same budget) it is what we said we would compare against.

### 5.5 `textbook` (GP + five rules written by the LLM before seeing any result)

The rules it wrote: ip_stride when stride regularity >= 0.7; a bigger LLC when the footprint ratio > 6.16; more L2 MSHRs when accesses >= 50 and predicted L2 miss ratio >= 0.3; SHIP when local reuse < 0.2; more L2 ways when the footprint is close to the L2 size. Reasonable textbook content; on the suite only the LLC rule spoke (footprint ratio 6.157 against the threshold that later re-scoping set at 6.1567). The arm sat on ip_stride or next_line + LLC 8192 designs at 0.49-0.54 for 12-14 designs, until stall scans found spp_dev (0.607) at designs 13-15. Final 0.609 in both seeds. It behaves like cold BO with a slightly worse start.

### 5.6 `llm_direct` (the plain agent: Gemini picks from the results table and the descriptors, no GP, no rules)

The reasoning trace reads like an architect's notebook and 46 of 48 picks were valid untested in-budget designs (2 random fills in 48). Round 1: "memory-bound, high stride regularity, add an aggressive L2 prefetcher" and "the baseline under-uses the area budget" (0.468 and 0.287). Round 2: "combine the best prefetcher with the maximum capacity" and "va_ampm_lite may capture patterns ip_stride misses" (0.502, 0.553). Round 3: va_ampm_lite + max capacity + SHIP: **0.622 at design 5**. Rounds 4-11: systematic single-knob probes around the incumbent: DRRIP vs SHIP, SPP vs va_ampm_lite, LLC geometry, L2 associativity, L1D size up and down, all within 0.60-0.62. Round 11-12: "the powerful va_ampm_lite L2 prefetcher may make the next_line LLC prefetcher redundant, causing harmful pollution": removing it gave **0.641** (design 22), a reasoned hypothesis confirmed at +3%. Its forecasts: 76% of measured values landed at or above 95% of the prediction, Brier 0.18, the best-calibrated forecaster of the night, although on an easier question than the dispute bets.

Two things to say plainly. First, the plain agent is strong on this class because the dominant win ("a graph traversal wants an aggressive prefetcher and the biggest LLC") is textbook knowledge it already holds; it did not need to learn it from chips A and B. Second, its later gains came from genuine single-knob reasoning about interactions (prefetcher pollution), which is exactly the kind of knowledge our playbook was supposed to carry and did not (rule 003 says "add an LLC next-line prefetcher", learned on chip A where it is +8.5%; on this class it is wrong).

### 5.7 `random`

As expected: 0.591 and 0.576 at 24 designs; never within 90% of the reference. Useful only as the floor.

---

## 6. Gate G0: fidelity of the short simulations (SPEC side)

26 chip C designs (the 20 best by the mcf+omnetpp+lbm geomean, the baseline, 5 random) re-simulated at 50M warmup / 50M instructions instead of 5M / 10M.

- Spearman between the short-run and long-run suite geomeans: **0.919**. Four of the short top-5 are in the long top-5; the short best is long rank 3.
- Every design gains a uniform +10 to +15% IPC at the longer warmup (baseline 0.471 -> 0.520; best 0.633 -> 0.724): absolute levels are low at 10M, ranks are right.
- The specific worry, that a short warmup under-rewards big LLCs, did not materialise: the mean IPC ratio of big-LLC over small-LLC designs is 1.40 -> 1.39 on mcf, 1.08 -> 1.10 on omnetpp, 1.11 -> 1.15 on lbm.

Verdict: the 2,600-row cache and every search result stand; the paper reports IPC as "at 10M instructions after 5M warmup" and cites this table. The GAP side (16 designs per graph workload) finished but its tables are still on the VM disk.

---

## 7. What the night established

1. **The machinery is real.** Learn set, distillation with measured claims, the free gate, a seven-arm cell with bets and re-scoping, and the fidelity check all ran unattended through CHIA with the profiler on, zero failed jobs, for about 12 USD.
2. **Data transfer fails on the unseen class exactly as predicted (P1).** The SPEC-pooled GP is the worst arm and cannot leave its basin.
3. **The rule language transfers where and which way, not how much.** Prefetcher rules and chip A's capacity rule fire where they should; every claim bet (sign, half the magnitude) was won; every dispute bet built on the transferred magnitude was mostly lost.
4. **Magnitudes learned on one chip and class are the wrong thing to shift the GP by on another.** The stacked spp_dev tilt (+0.315 log units from two chips) steered `rules` and `full` into the second-best basin and delayed the winning prefetcher by 6-10 designs.
5. **A descriptor can only be fitted on a chip whose budget lets it vary.** Chip B's 512 KB of headroom made movable LLC MPKI useless there and produced a degenerate, sign-inverted capacity condition.
6. **Stall-gated claim tests are cheaper than a per-round flood but still cost 2-3 of 24 designs when the search stalls early**, and they test at the baseline, far from where the search is.
7. **The plain agent is the arm to beat on this class**, and it beats everything, by holding the textbook prior and by reasoning about one interaction (prefetcher pollution) that the playbook got wrong.
8. **Short simulations rank correctly** (G0 passed on SPEC).

---

## 8. Open questions for the expert

These are the decisions the run puts on the table. None is taken.

1. **Rules as directions on a new class.** Shift the GP by a bounded fixed step per speaking rule and bet on the sign, using a transferred magnitude only once a controlled pair measured on this chip replaces it. Would this preserve the part of the language that transferred and remove the part that did not?
2. **De-duplicate claims across chips** (same knob and value from A and B become one rule with the mean or the minimum magnitude) so identical claims cannot stack.
3. **Calibration round.** When a rule speaks on a class it was not learned on, spend its first act on one claim test, so it bets and shifts with a size measured here. Costs about one round for the whole playbook; would have replaced +29.8% with +13.6% for spp_dev before any shift was applied.
4. **Reject degenerate fitted conditions**: a threshold that leaves every training workload on one side (rules 001, 004, 006, 008) or that inverts the descriptor's meaning for its claim (rule 009) does not enter the playbook, or enters as a class-agnostic "always" rule with a flag.
5. **Descriptors fitted only on chips whose budget lets them vary**, or a rule that says which chips a descriptor may be fitted on.
6. **Suite-level vs workload-level firing.** The pre-registered prediction is about single workloads; the search sees suite means. Either the cell should be scored per workload as well, or the paper should state that the gate and the search operate at different levels.
7. **Where should the comparison be made?** On this class the first 90% of the gain is one textbook move (turn on a prefetcher) that the plain agent knows. The arms only separate in the last 5-10% (interactions: LLC prefetcher pollution, L2 geometry, replacement). That needs (a) the held-out reference so 95/99% targets are meaningful, and (b) possibly a class or a start design where the prior is weaker. Is the honest headline "the language transfers as directions, and the plain agent is a strong baseline that our rules must beat in the tail", or should the design of the cell change?
8. **The pooled baseline's fairness.** Pure BO with no exploration escape collapses completely. Report as is (it is the textbook baseline we pre-registered) or add an exploration term and say so?

---

## 9. Costs and timings for planning

- Learn set: 625 simulations in 2 h 49 min on 32 vCPUs (the fotonik3d columns dominate: 122-168 min each).
- Cell 2, 7 arms x 2 seeds x 24 designs x 3 workloads: about 1,000 simulations plus builds in 2 h 52 min; 10 seeds would be about 14 h and about 21 USD.
- Fidelity SPEC at 50M/50M: 26 designs x 3 workloads in 63 min.
- LLM: 0.55 USD for everything above with Gemini 2.5 Pro.
- One 15M-instruction ChampSim run: 72-85 s on an Apple M4 core, about 5 min on a c2d vCPU; ranks are identical across the two machines (verified on identical designs).

Artifacts: `results_vm/experiment_v1_gap.json` (full report: every design, every bet, every round's slots and replies), `results_vm/experiment_v1_playbook.json`, `results_vm/experiment_v1_pool.json`, `results_vm/v1_learn.log` (G-language table), `results_vm/v1_gap2.log` (every round of every arm, re-scope events), `results_vm/v1_summary.log`. The GAP fidelity tables are on the VM under `~/hackathon/results/tierC_C_server_<gap trace>_w50M_s50M.json`.

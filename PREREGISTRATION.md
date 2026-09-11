# Pre-registered predictions

Written into the repository before the runs that test them; each is scored afterwards with the numbers, and the authors' own Brier score on them is reported next to the forecasters'. A prediction is settled by `python -m loop.summarize` on the cell's report against the cell's 300-design uniform reference.

## Settled by the first cell-2 run (`v1`, two seeds, untouched-chip start, playbook read by a GP)

- **P1** `bo_pooled` degrades sharply on the unseen workload class because its warm start is SPEC rows. **Held**: flat at 0.546 from design 3 in both seeds, worst arm.
- **P2** `rules` degrades markedly less than `bo_pooled`. **Held against `bo_pooled`, not against cold `bo`**: 12.5 vs >24 designs to 90% of the best found, but 12.5 vs 14.5 for cold `bo`, a tie.
- **P3** `textbook` is mediocre in both cells and degrades least. **Mediocre, yes** (14 to 90%, final 0.609); "degrades least" needs cell 1.
- **P4** `llm_direct` captures the easy gain fast but reaches 95/99% slower than `rules` and its forecasts score worse. **Failed**: fastest to every target (6 to 90%), best final design (0.641), best-calibrated forecaster (Brier 0.18).

## Pre-registered for the pilot and the cells that follow (third plan review)

Setting: every arm starts from chip B's best design fitted to chip C; the playbook is distilled once over A+B; the memory is built from A+B; scoring at 90/95/99% of a 300-design uniform reference per cell.

- **P5** `memory` beats `llm_direct` to 95% of the reference on cell 2 (median designs over seeds), because the recipe's starting design and the cards' mechanisms skip the designs the plain agent spends on what it already knows and steer it away from the interactions it gets wrong (an LLC prefetcher next to an aggressive L2 prefetcher).
- **P6** The margin of `memory` over `llm_direct` is larger on cell 2b than on cell 2, because the memory then contains cards and cases from the same workload class.
- **P7** `handoff` beats `llm_direct` to 95% on cell 2, because the plain agent's own history is a good warm start for a GP in the tail, where the agent's textbook prior runs out.
- **P8** `rules_v2` beats cold `bo` to 90% on cell 2, because direction-only tilts keep what transferred (where and which way) and drop what did not (how much).
- **P9** (descriptor language) With pooled distillation, no admitted capacity rule fires on `pr.urand` or `bfs.kron`, and every admitted prefetcher rule fires on all three held-out workloads with the right sign. Checked by `python -m loop.offline language` before the cell runs.

Authors' stated probabilities: P5 0.6, P6 0.7, P7 0.6, P8 0.55, P9 0.8.

## Settled by pilots v4/v5 (six seeds, start = chip B's best design, memory = cases + cards + recipe from A/B)

- **P5** failed: `memory` reached 90% of the best found in fewer than half the seeds, `llm_direct` in 4 of 6 (median 14). The recipe committed to the SPEC answer (spp_dev) and the agent stayed in that basin for 7-16 designs.
- **P7** failed (two seeds): `handoff` never reached 90%; the GP added nothing to the agent's history. Arm dropped.
- **P8** failed (two seeds): `rules` never reached 90%; cold `bo` reached 0.609 in one seed. Direction-only tilts did not escape the start.
- **P6** not run as written: cell 2b is superseded by cell w2 below.

## Pre-registered for cells w1, w2 and k (memory v2: facts and strategies, written by the system; headline = workload transfer on one chip)

Written 2026-09-10 16:02 CEST, before any of these cells ran. Setting: chip C throughout; every arm starts from the best SPEC design measured on chip C (cells w1, w2) or chip B's best fitted to C (cell k); 32 designs per run (16 rounds of 2) in the expanded search space (L1D and LLC prefetcher choices, L2 replacement); one seed while iterating, so a difference of one or two designs is noise and only a clear ordering counts; scored by designs to 90/95/99% of a 300-design uniform reference per cell (`python -m loop.summarize ... --uniform 300`; until the new-space sample finishes, the old-space best 0.6419 is the fixed target) and by best-so-far curves.

- **P-W1-a** `memory_strategies` reaches the reference in at most as many designs as `llm_direct`, because the strategies written from chip C's own SPEC cases (which knob is workload-specific, which interaction to test after which change) shorten the search without committing to a SPEC answer.
- **P-W1-b** `memory_facts` reaches it in at least as many designs as `llm_direct`, because facts measured on SPEC workloads are retrieved by descriptor similarity and the graph workloads sit far from every SPEC case (offline: beyond every pairwise case distance, so no fact passes the gate and only the three cases are shown), so the shelf adds cases that mislead more than they help, or nothing.
- **P-W1-c** `rules` (the same facts, read by a GP as bounded tilts) is worse than `memory_facts` (the same facts, read by the agent), because the agent can weigh a fact's provenance and distance and the GP cannot. (The `rules` arm is not in the iterating run of w1; settled when it is added.)
- **P-W2-a** `memory_both` and `memory_facts` beat `llm_direct` on designs to the reference, because the memory then holds cases and facts from the same workload family (GAP set 1 on the same chip), retrieved at short descriptor distance. (GAP set 2 as first chosen, bfs.road / pr.web / pr.road, had no headroom from the start and was abandoned after 22 designs; the prediction stands for whichever set 2 the user defines.)
- **P-W2-b** The opening pair (designs 1-2 of the memory arms) lands above `llm_direct`'s median best-so-far at design 4.
- **P-K** `memory_both` beats `bo_pooled` and `llm_direct` from design 1 (chip boundary, workloads fixed: the near-free transfer), because the nearest case is the same workload on a neighbouring chip and its best design is close to optimal here.

Authors' stated probabilities: P-W1-a 0.6, P-W1-b 0.65, P-W1-c 0.7, P-W2-a 0.65, P-W2-b 0.6, P-K 0.75.

## Pre-registered for cell w2 as finally defined (written before the run, 2026-09-11)

Set 2 = sssp.kron-246B + cc.urand-353B + cc.twitter-15B on chip C, **from the untouched chip** (suite 0.1656; best of the 11 probed designs 0.2210, +33.5%, of which 61.8% no single knob reaches). Memory = w1's SPEC memory consolidated with all eight w1 run memories: 21 cases, 625 facts, 20 strategies. Arms `bo`, `llm_direct`, `memory`; 2 seeds; 20 rounds = 40 designs each. The `bo` arm has been fixed since w1 (per-seed candidate sample plus the incumbent's neighbourhood each round), so **w1's `bo` numbers are not a comparison point for this cell.**

The state of the memory going in, checked and recorded before launch, so none of it can be claimed after the fact:
- The fact gate FIRES here and could not in w1: the three workloads sit at distance 3.96 / 3.52 / 3.44 against a median pairwise case distance of 4.98. In w1 it was 0 of 127.
- But only **10 of 625 facts** reach the prompt, and they are ranked by evidence volume (`pairs / (1 + distance)`), not effect size, so the slots go to near-inert knobs: `l2_mshr 32->64 +0.1%`, `llc_mshr 128->64 -0.0%`, `l1d_sets 128->64 +0.0%`, `llc_replacement lru->ship +0.2%`. **The facts shelf is expected to be close to useless here, and to understate `ship` by two orders of magnitude.**
- The signal is in the CASES shelf instead. The retrieved bfs.urand case carries `llc_replacement lru -> ship +12.6%`, `llc_prefetcher next_line -> no +8.5%` and `l2_prefetcher spp_dev -> va_ampm_lite +10.7%` - which are, measured independently on this set, the winning moves (`ship` alone is +19.8% / +20.7% on the two shortest-path workloads; both best composites turn the LLC prefetcher off).
- Two strategies w1's runs wrote already name the same moves in words: establish scan-resistant replacement for large-footprint workloads, and disable lower-level prefetchers on irregular access patterns.

- **P-W2-c** `memory` reaches `llc_replacement = ship` in fewer designs than `llm_direct` and `bo`, on both seeds. This is the sharpest test of the mechanism in the whole experiment: the move is worth ~20% on two of three workloads, the memory names it explicitly, and neither baseline has been told.
- **P-W2-d** `memory` reaches `llc_prefetcher = no` in fewer designs than `llm_direct` and `bo`. (In w1 only 3 of 10 runs ever tried it, which is what this cell was built to test.)
- **P-W2-e** `memory`'s final best beats `bo`'s on the mean over seeds, by more than the 1% that separated the arms in w1.
- **P-W2-f** The memory's facts shelf contributes nothing measurable: fewer than 3 of the memory arm's picks cite a fact, and the citation scoreboard for facts is within noise of zero. Cases and strategies carry the transfer.
- **P-W2-g** No arm's final best exceeds 0.2210 by more than 15%, i.e. the probed composites are close to the reachable ceiling on this set. (A miss here means the set has more headroom than the probe saw, which is good news and should be said as such.)

Authors' stated probabilities: P-W2-c 0.7, P-W2-d 0.65, P-W2-e 0.6, P-W2-f 0.7, P-W2-g 0.55.

## Settled by cell w1, first run (2026-09-10 18:32 CEST; one seed, 32 designs, expanded space, start = chip C's SPEC best at 0.6014; scored against the old-space best 0.6419 because the new-space reference sample was still running)

| arm | final best | first design above the start | designs to 90% of the gap | AUC % | picks citing memory | forecast Brier (win rate) |
|---|---|---|---|---|---|---|
| llm_direct | 0.6263 | 3 | never (target 0.638) | 2.95 | 0 of 32 | 0.130 (0.84) |
| memory_facts | 0.6252 | 4 | never | 3.21 | 10 of 30 | 0.159 (0.81) |
| memory_strategies | **0.6302** | 5 | never | 3.37 | 29 of 30 | 0.122 (0.86) |
| memory_both | 0.6242 | 4 | never | 2.90 | 24 of 30 | 0.137 (0.84) |
| bo | 0.6200 | 13 | never | 1.12 | - | - |

- **P-W1-a** (strategies <= plain agent to the reference): **tie on the pre-registered criterion** (no arm reached 90% of the gap in 32 designs); on final best and area under the curve memory_strategies is ahead (0.6302 vs 0.6263, AUC 3.37 vs 2.95), and it cited a strategy in 29 of 30 picks. One seed: a direction, not a result.
- **P-W1-b** (facts >= plain agent): **consistent** (tie on the criterion; final 0.6252 vs 0.6263). The gate retrieved 0 of 127 facts every round, as predicted; the shelf showed only the three SPEC cases. The SPEC facts settled by controlled pairs on the GAP workloads held their sign in 63% of 279 pair bets (Brier 0.346): SPEC directions are better than a coin on graph workloads, but not by much.
- **Opening pair cost two designs**: the nearest SPEC case's best design and its va_ampm_lite variant scored 0.545 and 0.538, both below the start (the variant tested the right knob in a 2 MB-LLC geometry where it loses); the memory arms first beat the start at design 4-5, the plain agent at design 3.
- **Nobody found the known old-space best** (0.6419: va_ampm_lite at L2, LLC 4 MB, NO LLC prefetcher, L1D next_line). Every arm kept the start's LLC next-line prefetcher; the two designs that won on bfs.urand (va_ampm_lite at L2) and the two that won on pr.urand (va_ampm_lite at L1D) were never combined. All agent arms are well calibrated on their own forecasts (Brier 0.12-0.16).

**Rescored against the new-space uniform reference (stopped by the user Sep 11 01:10 at 195 of 300 sampled designs measured on all three workloads; best of the sample 0.6194, median 0.491, one random design above the start, none above any agent arm):** designs to the reference (90/95/99% coincide, the reference sits just above the start): memory_facts 9, memory_both 9, memory_strategies 11, llm_direct 15, bo 22 / 25 / 29. The random sample never reaches where the agents were after 30 designs, so the optimizer-found 0.6419 stays the target that separates arms.

## Rerun of cell w1 (`w1b`, launched 2026-09-11 01:19 CEST): seeds 0 and 1, 24 rounds = 48 designs, same memory, same start, three mechanical fixes

Fixes, from the review of the first run: (1) no random fills: a duplicate or invalid proposal gets one retry with the rejected designs listed, then a one-knob perturbation of the incumbent; (2) the opening move is the START with the decisive knob flipped (one slot; the other goes to the LLM), instead of a remembered design in its own geometry; (3) cited items are scored per run by the percentile of the citing picks against the others, not by "below the incumbent". Predictions P-W1-a and P-W1-b apply unchanged; the first run's numbers stay above as the pre-fix result. Expected from the first run's diagnostics: the open question is whether any arm tests `llc_prefetcher = no` next to an aggressive L2 prefetcher (the move that made the old-space best 0.6419); the memory holds no strategy about it yet, so no arm should be favoured on it.

### w1b settled (2026-09-11 06:25 CEST; seeds 0-1, 48 designs each, 4.8 h VM, 5.3 USD of Pro calls, zero failed jobs; reference = the optimizer-found 0.6419; start 0.6014)

| arm | final best (s0, s1) | mean | to 90% of the gap (median; seeds hitting) | first design above the start (s0, s1) | AUC % | forecast Brier (win rate) | picks citing memory |
|---|---|---|---|---|---|---|---|
| memory_both | **0.6419**, 0.6284 | 0.6352 | 39.5 (1 of 2) | 1, 1 | **4.13** | 0.158 (0.81) | 77 of 94 |
| memory_strategies | 0.6408, 0.6297 | **0.6353** | 46 (1 of 2) | 1, 1 | 3.92 | **0.140 (0.87)** | 87 of 94 |
| memory_facts | 0.6302, 0.6302 | 0.6302 | never | 1, 1 | 3.55 | 0.180 (0.77) | 40 of 94 |
| llm_direct | 0.6223, 0.6379 | 0.6301 | 40 (1 of 2) | 7, 2 | 3.36 | 0.181 (0.75) | 0 |
| bo | 0.6251, 0.6211 | 0.6231 | never | 13, 23 | 1.73 | - | - |

- **P-W1-a** (strategies <= plain agent to the reference): **not held on the pre-registered criterion** (median 46 vs 40 designs, one seed of each reached 90%); **ahead on every other measure**: mean final 0.6353 vs 0.6301, AUC 3.92 vs 3.36, calibration 0.140 vs 0.181, and above the start from design 1 in both seeds against 7 and 2. Two seeds: a direction.
- **P-W1-b** (facts >= plain agent): **held**: memory_facts never reached 90% of the gap; final 0.6302 vs 0.6301. The gate retrieved 0 of 127 facts every round; SPEC facts settled by controlled pairs on GAP held their sign in 56% of 834 bets (Brier 0.401): a coin, on this class boundary.
- **The known best was found this time**: memory_both seed 0 reached exactly 0.6419 (va_ampm_lite at L2, no LLC prefetcher, SHIP, 4 MB LLC, L1D next_line) at design 39; memory_strategies seed 0 0.6408; llm_direct seed 1 0.6379. All three dropped the LLC prefetcher (5, 18 and 20 such designs); the other seven runs never tried `llc_prefetcher = no` and stayed at 0.62-0.63. Whether an agent tests that knob is the seed's luck, not the memory's doing: the memory holds no strategy about it yet.
- **Fixes worked:** the opening move (start with va_ampm_lite at L2) put every memory arm above the start at design 1; retries absorbed 5-14 duplicate proposals per run with 0-2 perturbations and no random design; the percentile scoreboard settled the base strategies (STRAT-0004 "prefetcher before capacity" improved in 4 of 4 runs; STRAT-0001 "L2 prefetcher first, by stride" worsened in 3 of 4).
- **Seed variance is large**: the memory arms reached 0.625 at design 3-5 in seed 0 and at 32-40 in seed 1. Any claim needs more seeds.

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

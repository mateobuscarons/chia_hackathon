# Caps off: the lean council against random's 1000-design quality

Random finals at 1000 designs per seed: seed 0 1.4982 (1001 rows), seed 1 1.4910 (1001 rows), seed 2 1.5111 (1001 rows)

Target (strongest seed): 1.5111; mean over seeds 1.5001

| arm | designs to 1.5111 | mean (never = 60) | reached | speedup 1000 / mean |
|---|---|---|---|---|
| shadow gating | 43 / 48 / never / 32 | 46 | 3 of 4 | 22x |
| apply gating | never / 58 / 62 / never | 60 | 2 of 4 | 17x |

Same table against the mean target 1.5001:

- shadow gating: 37 / 41 / 47 / 32
- apply gating: 53 / 53 / 42 / never

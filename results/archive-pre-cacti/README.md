# The record before CACTI

Everything here was measured on the **pre-CACTI chips** - `C_server`, `A_mobile`, `D_quad` with
no revision token in the name - where a cache's hit latency came from ChampSim's own size
formula, `round((sets*ways)^0.343 * 0.416)`, the same on every chip whatever its clock or
process node.

The loop now takes latency and area from CACTI at the SoC's node and clock, which changes the
chip: on the server the stock L1D went from 4 cycles to 6 and the stock LLC from 15 to 11, and
the best design's L1D moved from 12-way to 8-way because associativity costs cycles where the
formula said it was free. **No row here is readable by the current loop** - the chip revision in
every design name differs - so this directory is history, not a cache.

It is kept because these runs are the measured justification for decisions the loop still rests
on, and because the conclusions are quoted in `CLAUDE.md`:

| run | what it established |
|---|---|
| `runs/llama2_j8.json`, `llama2_q3.json` | the jump round: the council reaches 99.9 % of the gap in 15 rounds against the tuned forest's 40 |
| `runs/aiml_a1.json`, `aiml_a2.json`, `aiml_a3.json` | why `PAIRS` and `GEOM_LADDER` are on: rounds to 95 % go 7 -> 5 -> 3, and the seed spread collapses |
| `runs/aiml_f1.json` | the forest baseline on the aiml suite |
| `runs/aiml_m1.json`, `aiml_mf1.json` | the mobile cell, where the council's two seeds diverged (1.2268 against 1.1894) and the forest held the best known - the failure that motivated reading the chip |
| `runs/quad_q1.json`, `quad_fq1.json` | the quad cell, where every good design sat at 87-92 % of the memory channels' peak and nothing in the loop could see it |

`tables/` holds the 15,512 simulations behind them, about forty hours of compute.

**This directory can be deleted** once nothing cites it: `rm -rf results/archive-pre-cacti`.
Nothing in `loop/` reads it.

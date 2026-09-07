# Rules that bet: a calibrated, hypothesis-driven CHIA loop for cache-hierarchy design

An LLM architect tunes ChampSim cache hierarchies and distills an explicit
**playbook**: rules of the form *(condition, claim, worked example)*, each with
a win/loss record and Brier score. Every forecaster (surrogate model, playbook
rules, LLM hypotheses) logs a quantitative bet **before** each simulation; the
simulator settles it. The loop runs the experiments the forecasters disagree on
most, re-scopes rules that lose, and tests whether rules learned on SoCs A and
B cut simulations-to-target on an unseen SoC C.

The ledger, forecasters and selector are simulator-agnostic (`loop/loop.py`
takes any `evaluate(config) -> metrics`); ChampSim is just the first problem.

## Layout

| file | role |
|---|---|
| `loop/loop.py` | the generic round: fit surrogate (with rule/hypothesis priors), pick by expected improvement + one claim test, bet, run, settle, re-scope; `verify_claims` measures every proposed rule before admission |
| `loop/forecast.py` | rules as delta claims (value or direction), rule scope, controlled pairs, credibility, priors, EI selection |
| `loop/playbook.py` | rules + bet ledger, one auditable JSON; Brier scoring; rejected rules |
| `loop/surrogate_gp.py` | GP in log-speedup units (ordinal + one-hot encoding, ARD Matern), vectorised prediction |
| `loop/analyst.py` | every LLM call (hypotheses, distill, re-scope, textbook rules); cost log |
| `loop/champsim_problem.py` | ChampSim glue: SoC + trace(s) -> `problem` dict; single-workload or suite (geomean); result-table cache |
| `loop/configs.py`, `loop/socs.py` | search spaces A/B/C, SoC profiles, area budget, latency-from-size (placeholders, team-reviewed) |
| `loop/simulate.py`, `loop/sweep.py` | ChampSim build (parallel trees) and run; table load/save; dense reference sweep |
| `loop/experiment.py` | learn on A+B, distill+verify, test arms on C (random, bo, bo_pooled, surrogate, textbook, rules, analyst, full) |
| `loop/run.py`, `loop/smoke.py` | one command per tier (`B`, `C`); tiny end-to-end gate before any launch |
| `loop/summarize.py`, `loop/early.py`, `loop/plots.py` | tables (censored medians, hit fraction, final best, AUC), early signs of a running experiment, figures |
| `loop/distill_tables.py` | re-distill a playbook from every measured design of the training problems |
| `loop/chia_nodes.py`, `loop/run_chia.py`, `cluster/` | CHIA nodes (build, simulate, Vertex analyst), CHIA entry point, GCP recipes |
| `upstream/` | patches and notes for PRs back to CHIA / ChampSim |

## Run

```bash
python -m loop.smoke B            # gate before any launch (Tier B, real simulations, ~15 min)
python -m loop.run B v5           # Tier B experiment: learn A+B, test C, 5 arms, 3 seeds
python -m loop.run B v5 results/experiment_v4_playbook.json full   # reuse a playbook, one arm
python -m loop.run C tierC        # Tier C hard tier (suite objective) - run on GCP, see cluster/README.md
python -m loop.early results/experiment_v5_C.json results/v5_C.log  # while it runs
python -m loop.summarize results/experiment_v5_C.json               # when it is done
```
Setup (ChampSim, CHIA, traces, GCP auth) is in `CLAUDE.md`; the one-VM cloud recipe in `cluster/README.md`.

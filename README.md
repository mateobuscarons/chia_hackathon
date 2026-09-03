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
| `loop/loop.py` | the generic round: fit surrogate, hypotheses, bets, run most-disputed configs, settle, re-scope |
| `loop/playbook.py` | rules + bet ledger, one auditable JSON; Brier scoring |
| `loop/forecast.py` | rules and hypotheses as forecasters; disagreement-based selection |
| `loop/surrogate.py`, `loop/surrogate_gp.py` | additive (interpretable) and Gaussian-process surrogates |
| `loop/analyst.py` | every LLM call (hypotheses, distill, re-scope, textbook rules); cost log |
| `loop/champsim_problem.py` | ChampSim glue: SoC + trace -> `problem` dict; sweep-table lookup or simulate |
| `loop/configs.py`, `loop/socs.py` | Tier-A search space, SoC profiles, area budget (placeholders, team-reviewed) |
| `loop/simulate.py`, `loop/sweep.py` | local ChampSim build/run; dense reference sweep (ground truth) |
| `loop/chia_nodes.py`, `loop/run_chia.py`, `cluster/` | CHIA nodes (build, simulate, Vertex analyst) and entry point |
| `loop/experiment.py`, `loop/plots.py`, `loop/summarize.py`, `loop/audit.py` | transfer experiment (7 arms, Tier A/B), figures, one-screen table, rule autopsy |
| `loop/smoke.py` | whole experiment on a tiny budget; the gate before any launch |
| `upstream/` | patches and notes for PRs back to CHIA |

## Run

```bash
# Tier-A ground truth (once per SoC x trace; ~1 h each on a laptop)
python -m loop.sweep B_midrange traces/605.mcf_s-665B.champsimtrace.xz
# gate before any long run
python -m loop.smoke A     # or: python -m loop.smoke B
# the experiment, plain Python
python -m loop.experiment traces/605.mcf_s-665B.champsimtrace.xz
# the experiment as a CHIA loop (local Ray, or a `chia up` cluster with "auto")
python -m loop.run_chia local traces/605.mcf_s-665B.champsimtrace.xz
python -m loop.plots results/experiment_<stamp>.json
python -m loop.summarize results/experiment_<stamp>.json
```
Setup (ChampSim, CHIA, traces, GCP auth) is in `CLAUDE.md`.

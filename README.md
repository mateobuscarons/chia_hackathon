# Rules that bet: a calibrated, hypothesis-driven CHIA loop for cache-hierarchy design

An LLM architect tunes ChampSim cache hierarchies and distills an explicit
**playbook**: rules of the form *(condition, claim, worked example)*, each with
a win/loss record and Brier score. Conditions are written on chip-independent
workload descriptors (profiled once from the trace, read against the target
chip's geometry); claims are measured by controlled comparison before a rule is
admitted. Every forecaster (surrogate model, playbook rules, LLM hypotheses)
logs a quantitative bet **before** each simulation; the simulator settles it.
Rules that lose are re-scoped, never deleted. The experiment tests whether rules
learned on SoCs A and B cut designs-to-target on an unseen SoC C against
textbook and pooled Bayesian optimisation.

The ledger, forecasters and selector are simulator-agnostic (`loop/loop.py`
takes any `evaluate(config) -> metrics`); ChampSim is just the first problem.

## Layout

| file | role |
|---|---|
| `loop/loop.py` | the generic round: fit surrogate (with rule/hypothesis priors), pick by expected improvement + one owed claim test + a stall scan of the incumbent, bet, run, settle, re-scope; `verify_claims` measures every proposed rule before admission |
| `loop/forecast.py` | rules as delta claims (value or direction), rule scope, controlled pairs, credibility, priors, EI selection, stall scan, measured one-knob effects |
| `loop/playbook.py` | rules + bet ledger, one auditable JSON; Brier scoring; rejected rules |
| `loop/surrogate_gp.py` | GP in log-speedup units (ordinal + one-hot encoding, ARD Matern), vectorised prediction |
| `loop/analyst.py` | every LLM call (hypotheses, distill, re-scope, textbook rules); model by `ANALYST_MODEL`; cost log |
| `loop/trace_profile.py` | workload profile from the trace alone (working set, stride regularity, locality, miss-ratio curve from footprint theory); the chip-independent half of every rule condition |
| `loop/champsim_problem.py` | ChampSim glue: SoC + trace(s) -> `problem` dict; single-workload or suite (geomean); descriptors = profile read against the chip's geometry; result-table cache |
| `loop/configs.py`, `loop/socs.py` | search spaces A/B/C, SoC profiles, area budget, latency-from-size (placeholders, team-reviewed) |
| `loop/simulate.py`, `loop/sweep.py` | ChampSim build (parallel trees, per-binary lock) and run; table load/save; dense reference sweep |
| `loop/experiment.py` | learn on A+B, distill+verify, test arms on C (random, bo, bo_pooled, textbook, rules, full) |
| `loop/run.py`, `loop/smoke.py` | one command per tier (`B`, `C`; env `SEEDS`, `FIRST_SEED`, `ANALYST_MODEL`); tiny end-to-end gate before any launch |
| `loop/summarize.py`, `loop/early.py`, `loop/plots.py` | tables (censored medians, hit fraction, final best, AUC), early signs of a running experiment, figures |
| `loop/chia_nodes.py`, `loop/run_chia.py`, `cluster/` | CHIA nodes (build, simulate, Vertex analyst), CHIA entry point, GCP recipes |
| `upstream/` | patches and notes for PRs back to CHIA / ChampSim |

## Run

```bash
python -m loop.smoke B            # gate before any launch (Tier B, real simulations, ~15 min)
python -m loop.run B v5           # Tier B experiment: learn A+B, test C, 5 arms, 3 seeds
python -m loop.run B v5 results/experiment_v4_playbook.json full   # reuse a playbook, one arm
python -m loop.run C tierC        # Tier C hard tier (suite objective) - run on GCP, see cluster/README.md
ANALYST_MODEL=gemini-2.5-pro SEEDS=1 python -m loop.run C v5   # stronger analyst, seed 0 only
python -m loop.trace_profile traces/605.mcf_s-665B.champsimtrace.xz   # one workload's profile (cached under results/)
python -m loop.early results/experiment_v5_C.json results/v5_C.log  # while it runs
python -m loop.summarize results/experiment_v5_C.json               # when it is done
```
Setup (ChampSim, CHIA, traces, GCP auth) is in `CLAUDE.md`; the one-VM cloud recipe in `cluster/README.md`.

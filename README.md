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
| `loop/loop.py` | the generic round: fit surrogate, pick by expected improvement with the speaking rules' bounded mean shifts, on a stall one owed claim test or a scan of the incumbent plus the analyst's reply, bet, run, settle, re-scope; `verify_claims` measures every proposed rule before admission |
| `loop/forecast.py` | rules as delta claims (value or direction), controlled pairs, credibility, mean shifts, EI selection, stall scan, measured one-knob effects |
| `loop/playbook.py` | rules + bet ledger, one auditable JSON; Brier scoring; rejected rules |
| `loop/surrogate_gp.py` | GP in log-speedup units (ordinal + one-hot encoding, ARD Matern), vectorised prediction |
| `loop/analyst.py` | every LLM call (reply on a stall, distill, re-scope, textbook rules); model by `ANALYST_MODEL`; cost log |
| `loop/trace_profile.py` | workload profile from the trace alone (working set, stride regularity, locality, miss-ratio curve from footprint theory); the chip-independent half of every rule condition, read against the chip's geometry and budget in `champsim_problem.chip_descriptors` (footprint ratios, predicted miss ratios, movable MPKI) |
| `loop/champsim_problem.py` | ChampSim glue: SoC + trace(s) -> `problem` dict; single-workload or suite (geomean); descriptors = profile read against the chip's geometry; result-table cache |
| `loop/configs.py`, `loop/socs.py` | search spaces A/B/C, SoC profiles, area budget, latency-from-size (placeholders, team-reviewed) |
| `loop/simulate.py`, `loop/collect.py` | ChampSim build (parallel trees, per-binary lock) and run, one trace per core; simulate a fixed design list per chip (`structured` = the learn set, `top` = the best designs for the fidelity check and the final validation via `LOOP_WARMUP` / `LOOP_SIM`), merge tables simulated on another machine |
| `loop/experiment.py` | distill+verify the playbook from the training chips' result tables, test arms on C (random, bo, bo_pooled, textbook, rules, full) |
| `loop/run.py` | one command per cell (`spec`, `gap`, `quad`, `smoke`; env `SEEDS`, `FIRST_SEED`, `ANALYST_MODEL`); the smoke cell is the tiny end-to-end gate before any launch |
| `loop/summarize.py`, `loop/plots.py` | tables (designs to 90/95/99% of the fixed reference, hit fraction, final best, AUC, calibration per forecaster; works on a running report), figures |
| `loop/chia_nodes.py`, `loop/run_chia.py`, `cluster/` | CHIA tasks (build from any config, simulate one trace per core, Vertex analyst), the CHIA entry point that runs a cell with arms as Ray tasks and the profiler on, GCP recipes |
| `upstream/` | patches and notes for PRs back to CHIA / ChampSim |

## Run

```bash
python -m loop.run_chia local smoke s1                 # the gate before any launch: two arms, one round, real simulations, through CHIA
python -m loop.run_chia local spec v1                  # cell 1: learn on A+B, test on chip C with the training suite (profiles in results/chia_profiles/)
python -m loop.run_chia local gap v1 results/experiment_v1_spec_playbook.json   # cell 2: same frozen playbook and pool, held-out graph workloads
python -m loop.run_chia local quad v1 results/experiment_v1_spec_playbook.json  # cell 3: the four-core chip, rules vs bo_pooled
ANALYST_MODEL=gemini-2.5-pro SEEDS=1 python -m loop.run spec v1    # the same cells without CHIA (plain processes), seed 0 only
python -m loop.trace_profile traces/605.mcf_s-665B.champsimtrace.xz   # one workload's profile (cached under results/)
python -m loop.offline admit | coverage | language <playbook.json>   # free gates: workload admission, descriptor coverage, do the rules fire where the held-out class can use them
COLLECT_SOCS=C_server LOOP_WARMUP=50000000 LOOP_SIM=50000000 python -m loop.collect top 20 traces/605...xz traces/620...xz traces/619...xz   # fidelity check: top-20 + baseline + 5 random at 50M/50M, own table
COLLECT_SOCS=A_mobile,B_midrange python -m loop.collect structured 80 traces/...   # the learn set on the training chips
python -m loop.collect merge ../results_from_vm                        # union rows simulated elsewhere
python -m loop.summarize results/experiment_v1_spec.json --reference 0.7491   # while it runs or when done
```
Setup (ChampSim, CHIA, traces, GCP auth) is in `CLAUDE.md`; the one-VM cloud recipe in `cluster/README.md`.

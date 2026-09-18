# Few-shot cache tuning by a council of LLM specialists

Tuning a ChampSim cache hierarchy takes simulations, and simulations are the expensive part. The
question is not how good a design you can eventually find. It is **how long it takes to get
close to it** — and because simulations run in parallel, the unit of time is the round, not the
design.

## The result

One ML-inference trace the loop had never seen (llama2_7b, DPC4 `ai-ml`), the stock chip at IPC
0.4731, the best design known at 1.4710 over 1718 designs measured (1M warm-up / 2M simulated).
Both arms start from the stock chip, 2 seeds each, and measure several designs per round in
parallel: the council sketches a wave of proposals, the forest picks a batch. Rounds to reach a
share of the stock-to-best gap, mean over the seeds:

| level | council (rounds) | RF (rounds) | ratio |
|---|---|---|---|
| 95 % | 6.5 | 19.5 | **3.0×** |
| 99.9 % | 14.0 | 35.0 | **2.5×** |

Share of the gap reached after each round, mean over the seeds:

| | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R8 | R10 | R12 | R14 | R16 | R20 | R25 | R30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| council | 0 % | 47 % | 85 % | 86 % | 91 % | 92 % | 92 % | 92 % | 97 % | 97 % | **100 %** | 100 % | 100 % | 100 % | 100 % |
| random forest | 76 % | 79 % | 85 % | 89 % | 89 % | 90 % | 90 % | 93 % | 93 % | 93 % | 95 % | 95 % | 95 % | 96 % | **100 %** |

The forest opens ahead — its first round is a wave of random designs, and on this chip a random
design is often decent — and holds the lead to R7. The council passes it at R9 and finishes at
R13 and R15; the forest needs R30 and R40 for the same level. Both arms reach the top: the
council settles on 1.4706 in both seeds, the forest on 1.4710 and 1.4702.

The forest measures 3 designs a round, which is what the council averages (3.26 over its 38
rounds). Runs: `results/runs/llama2_j8.json` (council), `results/runs/llama2_q3.json` (forest).

## How it works

![The council loop](loop.png)

An **analyst** reads every design measured so far and writes a sheet: a verdict per cache level
with the number behind it, the bottleneck, what the run has learned with its deltas, which knobs
are failing by their own counters, a prediction. Four **specialists** — prefetch, geometry,
replacement, concurrency, each owning its own knobs — read the sheet and each propose one move,
or hold. Every proposal is **sketched** in one parallel wave: the current design, each proposal
alone on it, and all of them together, so the interaction between moves is a measured number
before anything is committed. The analyst then **composes** the round's design from the proposals
with the sketches in front of it, choosing among them and altering no value. Round 1 is the
analyst's opening design read off the stock chip's report.

**Climb and jump.** A climb round is the above: every move is judged against the current best
design, and a move the sketches say will lose is not measured. That only ever goes uphill, so it
stalls in a local optimum with budget left. When two rounds in a row fail to improve the best
design, the next round is a **jump round**. The prompts change: everyone is shown the moves
already refused and, per knob, the values never tried as a single change from the current design,
and each specialist must propose one untried single-knob move — alone, so its sketch is
attributable. Only a specialist whose neighbourhood is exhausted may propose a coupled move that
puts its levels in a different regime. The analyst composes from the parts that gained, and a
jump the sketches do not predict to gain is not measured at all. If the stall continues, the next
jump searches a smaller neighbourhood and sweeps the level the jump rounds have blamed least.

On this trace the jump rounds are what close the last four points. Of thirteen jump rounds, four
won, every one a single knob the climb had refused or never reached — 32 LLC MSHRs alone is worth
+0.0389 IPC, and the climb never got there because it is worth nothing in combination with the
moves the specialists were pairing it with. The same council without the jump round stops at 96 %
in both seeds with its budget unspent.

Two arms, same budget, seeds and fidelity, both from the stock chip: `council`, and `bo`, a
random forest with expected improvement measuring `BO_BATCH` designs a round. The best design
measured on every workload of the suite is the denominator every share divides by. The model is
Gemini 3.1 Pro on Vertex; each specialist's prompt names no workload, trace or chip.

## Layout

| file | role |
|---|---|
| `loop/space.py` | what a design is: 13 knobs, the area budget |
| `loop/simulate.py` | a design to numbers: ChampSim's config, the build, the run |
| `loop/suite.py` | the objective, and the result tables that cache it |
| `loop/search.py` | the two arms, the driver, the score table, the ceiling |
| `loop/council.py` | the analyst, the specialists, the sketches, the jump round |
| `loop/analyst.py` | the model call |
| `loop/workloads.py` | fetching a trace |
| `results/tables/` | the shared simulation cache, one per workload and fidelity: the dataset |
| `results/runs/` | a run's report: every design, every round with its sketches |
| `upstream/` | patches for ChampSim and CHIA |

## Setup

`results/tables/` holds every simulation result, so the score tables can be reproduced without a
simulator or a trace:

```bash
git clone https://github.com/mateobuscarons/chia_hackathon.git && cd chia_hackathon
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.10+
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim  # the stock chip's config; no build needed
LOOP_WARMUP=1000000 LOOP_SIM=2000000 .venv/bin/python -m loop.search score results/runs/llama2_j8.json
```

To simulate new designs, build ChampSim with the SPP fix, fetch the traces and log in to Vertex:

```bash
cd champsim && git submodule update --init
git apply ../upstream/0002-champsim-spp-dev-ghr-victim.patch   # without it, spp_dev designs crash
./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
./config.sh champsim_config.json && make -j8 && cd ..
for i in $(seq 1 7); do cp -r champsim champsim_$i; done          # several designs compile at once

# the ML-inference traces: 200 MB prefixes from the DPC4 bucket; the file name keys the tables
curl -s https://pub-c31f67d79d1b4cd28ff320612b1a9f84.r2.dev/manifest.txt | grep ai-ml
.venv/bin/python -m loop.workloads fetch <object url> traces/llama2.c-llama2_7b.1.champsimtrace.gz 200
# the smoke gate's SPEC17 traces
curl -sSLO --output-dir traces https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/605.mcf_s-665B.champsimtrace.xz
curl -sSLO --output-dir traces https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/619.lbm_s-2676B.champsimtrace.xz

gcloud auth application-default login                              # Gemini on Vertex
SEEDS=1 .venv/bin/python -m loop.search smoke s1                   # the gate: both arms, one round
```

## Run

```bash
export LOOP_WARMUP=1000000 LOOP_SIM=2000000                        # the rung these results use
SEEDS=2 BUDGET=60 python -m loop.search llama2 c1 council          # the council
SEEDS=2 BUDGET=225 BO_BATCH=3 python -m loop.search llama2 f1 bo   # the forest, 3 designs a round
python -m loop.search score results/runs/llama2_c1.json
python -m loop.search ceiling 100 10 <trace>                       # the independent reference
```

A design is simulated once, ever: results are cached in `results/tables/`, keyed by the design's
name and the fidelity, and every run reads the cache before simulating. Every prompt and answer of
a run is in `results/<arm>-<tag>-s<seed>.log`.

Design, operations and open work are in `CLAUDE.md`. Contributions back to CHIA and ChampSim are
in `CHIA_BLOCKS.md`.

# Few-shot cache tuning from a memory of earlier searches

Tuning a ChampSim cache hierarchy takes simulations, and simulations are the expensive part.
This loop keeps a **memory** of searches already run on other workloads, boiled down to one
design worth copying, and starts a new search from there.

The question is not how good a design you can eventually find. It is **how many designs you
have to measure to get close to it.**

## The result

Three Google datacenter traces (whiskey, bravo, delta) the loop has never seen. The memory was
built from SPEC17 and GAP graph workloads only. "Best design known" is 0.5485 suite IPC, found by
an independent 100-design search that used no memory. The stock chip is 0.4389.

Share of the stock-to-best gap reached after N simulated designs, mean over seeds:

| | D1 | D4 | D8 | D12 |
|---|---|---|---|---|
| random forest from the stock chip | 19 % | 46 % | 57 % | 61 % |
| four LLM specialists, from the stock chip | 21 % | 52 % | 63 % | 87 % |
| the same forest, from the memory's design | **93 %** | 93 % | 95 % | 95 % |
| four LLM specialists, from the memory's design | **93 %** | 95 % | 96 % | 97 % |

Designs a search has to measure to reach a level, read off the mean curve:

| | 95 % | 98 % |
|---|---|---|
| random forest from the stock chip | D37 | D73 |
| four LLM specialists, from the stock chip | not within 16 | not within 16 |
| the same forest, from the memory's design | D11 | not within 16 |
| four LLM specialists, from the memory's design | D4 | not within 16 |

**The memory's first design is already at 93 %. The same search from scratch needs 26 simulations
to match it, and 37 to reach 95 %.** The two forest rows are the identical optimizer; only the
starting point differs, and the same holds for the two specialist rows. Without a memory the
specialists and the forest are on par. From the memory the specialists reach 97 % at D12 over three
seeds (95 to 99): the seed that reached 99 % shrank the L2 to an eighth of its size in two
consecutive moves, a two-knob move the forest never took.

Seeds: 4 for the forest from scratch (3 runs of 75 designs for the levels), 5 from the memory,
3 for each specialist row.

It holds on a second suite (sierra.a.4, merced, tahoe): the opening is 90 %, which a from-scratch
search needs 27 simulations to reach; from there the specialists reach 93 % at D8 and stay there.

`REPORT.md` has the full tables, the limits, and what we removed along the way.

## How it works

**Nothing about the test workloads enters the memory.** A memory is built by searching a separate
set of workloads — here SPEC17 and GAP — and what it stores is one design:

```json
{"chip": "C_server",
 "handover": {13 knobs},
 "chosen_from": {"designs": 163, "measured_on": 6, "mean_gap_share": 0.90,
                 "per_workload": {"605.mcf_s-665B": {"stock": 0.3067, "best": 0.5228, "share": 0.885}, ...}}}
```

The design is the one with the best mean share of the stock-to-best gap across every remembered
workload, counting only designs measured on all but one of them. whiskey, bravo and delta are
never simulated during that build and play no part in choosing it.

Four searches are compared at the same budget, the same seeds and the same fidelity:

- `bo` — random forest with expected improvement, from the stock chip
- `pooled_bo` — the identical search, from the design the memory hands over
- `council` — four Gemini specialists, one per concern (prefetch, geometry, replacement,
  concurrency); one concern moves per round, from the design the memory hands over
- `council_stock` — the identical council, from the stock chip

## Layout

| file | role |
|---|---|
| `loop/space.py` | what a design is: 13 knobs, the area budget |
| `loop/simulate.py` | a design to numbers: ChampSim's config, the build, the run |
| `loop/suite.py` | the objective, and the result tables that cache it |
| `loop/search.py` | the arms, the driver, the score table |
| `loop/council.py` | the four specialists: their principles, the per-level report, one move a round |
| `loop/analyst.py` | the LLM: the call, the prompt, the proposals |
| `loop/memory.py` | filling a memory, and the design it hands over |
| `loop/workloads.py` | fetching, probing and judging a candidate workload |
| `results/tables/` | the shared simulation cache, one per workload — the dataset |
| `cluster/`, `upstream/` | the one-VM GCP recipe; patches for CHIA and ChampSim |

## Setup

Everything the project measured is in this repo: `results/tables/` holds all 8585 simulation
results, so **every published number can be reproduced without a simulator and without a single
trace.** Simulating new designs needs more.

### To read the results (a few minutes)

```bash
git clone https://github.com/mateobuscarons/chia_hackathon.git && cd chia_hackathon
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.10+

# the stock chip's knob values are read from ChampSim's own config, so the clone is
# needed even for read-only work — but it does not need to be built
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim

.venv/bin/python -m loop.search score results/runs/dc2_g1.json   # the forest arms
.venv/bin/python -m loop.search score results/runs/dc2_c5b.json  # the specialists
```

### To simulate new designs (a few hours, mostly downloads and one build)

```bash
# 1. ChampSim, pinned to the commit the tables were produced with, with the SPP fix applied
cd champsim
git fetch --unshallow && git checkout 51588e1 && git submodule update --init
git apply ../upstream/0002-champsim-spp-dev-ghr-victim.patch   # without it, spp_dev designs crash
./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
./config.sh champsim_config.json && make -j8 && cd ..

# 2. copies of the tree, so several designs compile at once (one build holds a whole tree)
for i in $(seq 1 7); do cp -r champsim champsim_$i; done

# 3. traces (~3.9 GB, gitignored). Twelve workloads, from three sources.
mkdir -p traces && cd traces
for t in 605.mcf_s-665B 619.lbm_s-2676B 620.omnetpp_s-874B; do          # SPEC17
  curl -sSLO "https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/${t}.champsimtrace.xz"
done
cd ..
```

**GAP** — `bfs.urand-36B`, `pr.urand-129B`, `bfs.kron-128B` — live inside a ~10 GB zip on Zenodo
record [20043527](https://zenodo.org/records/20043527). `fetch_gap` pulls one member out of it over
HTTP ranges without downloading the archive; take the archive's file URL from the record page:

```bash
.venv/bin/python -m loop.workloads fetch_gap <zip url> bfs.urand-36B.champsimtrace.xz traces
```

Cite *Characterizing the impact of last-level cache replacement policies on big-data workloads*,
IISWC 2020, if you use them.

**Datacenter** — `sierra.a.4`, `merced`, `tahoe`, `whiskey`, `bravo.a`, `delta` — come from the DPC4
bucket as 100 MB prefixes, which hold ~88M instructions each, enough to simulate. The exact object
path per trace is in the manifest:

```bash
curl -s https://pub-c31f67d79d1b4cd28ff320612b1a9f84.r2.dev/manifest.txt | grep sierra.a.4
.venv/bin/python -m loop.workloads fetch <object url> traces/sierra.a.4_0000.champsim.gz 100
```

The filename must match what `loop/search.py`'s `TRACE` table expects, since that name keys the
result tables.

```bash
# 4. only for the LLM arms and the memory build: credentials for Gemini on Vertex
gcloud auth application-default login
```

Verify the setup with the gate — every arm, one round, two workloads:

```bash
SEEDS=1 .venv/bin/python -m loop.search smoke s1
```

## Run

```bash
python -m loop.memory build results/memory.json <spec traces> -- <gap traces>   # fill a memory
SEEDS=5 BUDGET=16 python -m loop.search dc2 h1                                  # run a cell, every arm
SEEDS=2 BUDGET=12 python -m loop.search dc2 c5 council                          # one arm
python -m loop.search score results/runs/dc2_h1.json                            # read it
python -m loop.search ceiling 100 10 <trace> <trace> <trace>                    # the independent ceiling
```

A design is simulated once, ever: results are cached in `results/tables/<workload>.json`, keyed by
the design's name, and every run reads that cache before simulating. Designs the simulator cannot
measure are cached too, so a bad design costs one failed attempt rather than one per run.

Design, operations and open work are in `CLAUDE.md`. Findings are in `REPORT.md`. Contributions
back to CHIA and ChampSim are in `CHIA_BLOCKS.md`.

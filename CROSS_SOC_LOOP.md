# Cross SoC Loop

The plan for the loop's next scenario: **a generation step under a hard cost cap.** A design team
has a cache hierarchy that shipped and was good for the chip it shipped on. The node shrinks, the
core widens, the memory changes, a new workload class arrives, and the silicon budget for cache
falls because the area the shrink freed went to more cores. Last generation's design can no longer
be built. What replaces it?

This file is the design, the development order, the validation and the success criteria. `CLAUDE.md`
holds the loop as it stands and every result measured on this cell; `CHIA_BLOCKS.md` holds what goes
upstream. Every number below that is marked measured was measured with the code in this tree and is
reproducible from it.

---

## 0  Status

Built and exercised by a run:

| item | where | note |
|---|---|---|
| §4.2 caps in mm², refused before simulation | `space.within_budget`, `space.silicon_mm2`, `space.leakage_w` | a shape whose leakage alone breaks the power cap is refused here too |
| §4.3 the power model on the search path | `chip.energy` (5 CACTI fields), `space.power_split` / `watts` | every measured design carries `mm2` and `watts`; reproduces the gate's watts exactly |
| §4.5 the no-regression constraint | `space.violations`, checked after measurement | with the power cap; a violating design is recorded, shown, never stood on |
| §5.3 the comparison | `council` and `random` arms, `loop/search.py` | run as `g1`, `g2a`, `g2b`, `g3r`, `g4a`, `g5a`; results in `CLAUDE.md` |
| the three caps as SoC keys | `socs.py`: `area_budget_mm2`, `power_budget_w`, `regression_tolerance` | a chip without them keeps the KB budget and no power rule |
| what the council reads about cost | `chip.budget_text`, `council.cost_line`, the ladder, the power split, the value ledger's mean W, two power principles, the sheet's `over_budget` | ~9250 tokens a round, from 8935 |

Built beyond the plan, because the first run did not move (`CLAUDE.md`, finding 1): `OPENINGS`,
`SOFT_CAPS` (dropped after `g2b`), `HOLD_JUMP`, and the per-workload floor in `workload_view`.

Planned, not built:

| item | why it is still open |
|---|---|
| §4.1 the YAML brief | the three-file path still carries every scenario |
| §4.4 frozen knobs | no scenario has needed a frozen knob yet |
| §4.6 the report page and the frontier | the run JSON carries `mm2`, `watts` and `violations` per design, so this is rendering only |
| §4.7 the BOOM profiles | Chipyard has no prefetcher generator, which is where every winning design spends its gain |
| §5.2 the independent ceiling | every percentage is still measured against a design the council found |
| §5.4 the ablations as listed | `cap off`, `CHIP_VIEW=0` and the greedy climb are unrun; the openings / soft-caps / hold-jump ablations were run instead |
| the L1D latency ceiling as a fourth cap | considered and left out |

---

## 1  Why this scenario replaces raw-speed tuning

The loop's three existing cells ask one question: maximise the geometric-mean IPC of a suite,
starting from a weak stock hierarchy, under an SRAM-capacity budget. Measured on the quad cell's
own 531 designs, that question has almost no structure at the top:

| objective over the same 531 designs | designs within 2 % of the best | within 1 % |
|---|---|---|
| geometric-mean IPC (the loop today) | **42.4 %** | 25.8 % |
| IPC per mm² of cache silicon | **0.2 %** | 0.2 % |
| IPC per watt of cache + off-chip traffic | **0.2 %** | 0.2 % |
| worst workload instead of the mean | 44.6 % | 29.8 % |

Two things follow. A plateau where two designs in five are near-optimal is not a search problem —
random sampling solves it, which is what the quad cell showed (random search reached 97 % of the
gap and the council 100 %, a 1.2 % spread in absolute IPC). And the fix is the **objective**, not
the chip: ranking by the worst workload leaves the plateau intact, and tightening the capacity
budget alone only moves 42 % to 18 %. Normalising by cost is what produces a peak.

Power is also the largest dimension the loop is currently blind to. Across those same 531 designs
the cache and off-chip energy spans **0.75 W to 5.78 W, a factor of 7.7**, and the fastest design
burns 2.3× the power and 2.6× the silicon of the best design per watt to gain 3.5 % IPC.

The mechanism behind the plateau is specific and worth recording: with no cost term, *turning every
prefetcher on* is free and nearly optimal. That is why round 1 of the quad run captured 70 % and
96 % of the total gain on its two seeds, and why twenty further rounds bought under 1 %. Under a
cost cap a prefetcher is paid for in area, in energy per useless fetch, and in the capacity it
displaces, so which mechanism sits at which level becomes a decision rather than a default.

---

## 2  The scenario

### 2.1  The two chips

Both are entries in `loop/socs.py`, which stays the only file in `loop/` that names a chip.
Everything the loop reads is derived from these numbers (`loop/chip.py`).

| | `SOC=prev` (`E_prev`) | `SOC=next` (`F_next`) |
|---|---|---|
| node | 32 nm | **22 nm** |
| clock | 3.2 GHz | **3.8 GHz** |
| core | 4-wide issue, 6-wide fetch, ROB 256, LQ 128 / SQ 72 | **6-wide issue, 8-wide fetch, ROB 384, LQ 160 / SQ 96** |
| memory | 2 ch × 3200 MT/s → 16.0 B per core-cycle | **2 ch × 4800 MT/s → 20.2 B per core-cycle** |
| DRAM service | 144 cy row miss, 48 cy row hit | **187 cy row miss, 63 cy row hit** |
| cores | 1 | 1 |

Four things move at once, which is what a generation step is. Note the memory in particular: the
new chip has **26 % more bandwidth and 30 % worse latency in core cycles** — the DDR4-to-DDR5 trade
plus a faster clock. Those pull the optimal hierarchy in opposite directions, and neither direction
is knowable without measuring.

### 2.2  Last generation's design, and what the shrink does to it

```
L1D  64 sets x  8 ways =   32 KB   next_line
L2C 1024 sets x  8 ways =  512 KB   spp_dev,  srrip, MSHR 32
LLC 8192 sets x 16 ways = 8192 KB   no prefetcher, srrip, MSHR 128
```

Characterised by CACTI 7 in cache mode at each node (measured, `loop/chip.py`):

| | 32 nm @ 3.2 GHz | 22 nm @ 3.8 GHz |
|---|---|---|
| L1D 32 KB | 3 cy, 0.35 mm², 0.123 nJ/access | 3 cy, 0.16 mm², 0.065 nJ |
| L2C 512 KB | 5 cy, 1.37 mm², 0.123 nJ | 4 cy, 0.65 mm², 0.066 nJ |
| LLC 8192 KB | 15 cy, 18.24 mm², 0.595 nJ | 14 cy, 8.64 mm², 0.315 nJ |
| **total** | **20.0 mm², 4947 mW leakage** | **9.46 mm², 2417 mW leakage** |

The same hierarchy costs **less than half the silicon and half the leakage** one node down. CACTI
accepts 45, 32 and 22 nm and refuses 16 nm and below ("Invalid technology nodes"), so 32 → 22 nm is
the generation step this tree can characterise.

### 2.3  The cap

The freed silicon is not free in a product: it is taken by the extra cores. The brief therefore
caps cache silicon at **4 mm²**, measured against the space (4000 random designs):

| cap | share of the design space that fits | last generation's design |
|---|---|---|
| 10 mm² | 98.1 % | fits |
| 8 mm² | 87.7 % | over |
| 6 mm² | 85.3 % | over |
| **4 mm²** | **60.1 %** | **2.4× over — cannot be ported** |
| 2 mm² | 26.5 % | over |

4 mm² binds hard (40 % of the space is illegal) without dominating. Cache silicon across the whole
space runs 0.87 to 11.22 mm², median 3.09.

The earlier KB budget on this chip admitted **98.9 %** of designs and was not a constraint at all;
it is replaced (§4.2).

Two further caps were fixed after the gate and are what the runs used: **1.0 W** of cache and
off-chip power, chosen as "no more than what shipped" (the baseline draws 0.98 W on this suite; at
1.5 W nothing in the gate's 79 designs was excluded, at 1.0 W thirteen were, six of them faster than
the baseline), and **no workload slower than the baseline**, tolerance 0. Under all three, 169 of
750 uniformly drawn designs are feasible (22.5 %).

### 2.4  The workloads

Cell `gen` in `loop/search.py`, at 1 M warm-up / 2 M simulation:

| trace | role |
|---|---|
| `llama2.c-llama2_7b.1` | the workload class that arrived after the previous chip shipped |
| `whiskey_0000` | the datacenter workload the previous chip was built for, kept |

The objective is the suite's geometric-mean IPC, as everywhere else in the loop, with a
no-regression constraint on each workload separately (§4.5).

### 2.5  The baseline

Not last generation's design — it cannot be built. The baseline is **the obvious engineering
answer**: keep the design and shrink the last level until it fits.

```
L1D  64 x  8  next_line | L2C 1024 x 8 spp_dev srrip m32 | LLC 2048 x 16 no srrip m128
3.07 mm²   0.98 W   IPC 0.7539
```

This is the design a competent engineer produces when told to cut cache area, and it is what the
loop has to beat. Anything the loop finds is reported against it.

---

## 3  The gate (measured, passed)

Before committing to the scenario, three questions had to be answered: is there headroom, is the
answer reachable by single steps, and do single steps mislead? A design space is only worth a
reasoning loop if the third is yes.

79 designs (the baseline, all 30 feasible one-knob moves from it, 48 random feasible designs),
158 simulations, ~17 minutes on the Mac, no model calls.

| test | bar | result | |
|---|---|---|---|
| **headroom** — best feasible design over the baseline | ≥ 2 % | **+27.2 %** | pass |
| **depth** — knobs separating it from the baseline | ≥ 3 of 13 | **12 of 13** | pass |
| **greedy resistance** — share of the gain the best single move buys | < 60 % | **25 %** | pass |
| **deception** — knobs in the winner that lose when moved alone | ≥ 1 | **6 of 12** | pass |

```
baseline    L1D 64x8  next_line    | L2  1024x8 spp_dev   srrip | LLC 2048x16 no      srrip   3.07 mm²  0.98 W  0.7539
best found  L1D 64x12 va_ampm_lite | L2   256x4 next_line ship  | LLC 4096x8  spp_dev drrip   2.60 mm²  0.85 W  0.9588
```

**+27.2 % IPC on 15 % less silicon and 13 % less power**, with `llama2` +50.0 % and `whiskey`
+7.8 %, so neither workload regresses. The restructure guts the middle level (512 KB → 64 KB),
spends it on L1D associativity and last-level sets, and reassigns which prefetch mechanism sits at
which level.

**Deception, matched on the exact value the winner uses.** Six of the twelve changed knobs lose
when moved alone from the baseline:

| knob | to | alone |
|---|---|---|
| `l2_prefetcher` | next_line | **−20.99 %** |
| `l2_sets` | 256 | −2.83 % |
| `l2_ways` | 4 | −2.73 % |
| `llc_ways` | 8 | −2.12 % |
| `l1d_prefetcher` | va_ampm_lite | −0.65 % |
| `llc_replacement` | drrip | −0.21 % |
| `llc_sets` | 4096 | not feasible as a single move |
| `l1d_ways` | 12 | +6.91 % |
| `llc_prefetcher` | spp_dev | +1.69 % |

The best single move from the baseline is `l1d_ways: 8 → 12` at +6.91 %, a quarter of the +27.2 %
available. A greedy one-at-a-time search stalls there, because every further single step reads as a
mistake — one of them as a 21 % loss. This is the coupled move `CLAUDE.md` records as the loop's
open problem, and in this scenario it is the answer rather than a residual.

**The plateau is gone.** Of the 48 random designs, **2 %** are within 2 % of the best (42.4 % on the
quad cell), the median random design is **2.2 % worse than the baseline**, and only 20 of 48 beat
the baseline at all.

**Two caveats on record.**

1. Random search reached +27.2 % on its **5th draw** and did not improve over the remaining 43. With
   2 % of designs near-best the expected wait is ~50 draws and a hit within 5 is a **10 % event**, so
   this is luck, not a property of random search — but it means no single-seed comparison is
   evidence. Every arm runs ≥ 2 seeds.
2. The +27.2 % design **is** that random draw, so it is a floor on what is reachable, not the
   ceiling. No percentage against it is quotable until the independent ceiling exists (§5.1). This
   is the same standing caveat the rest of the project carries.

Reproduce with `results/gate/gate_next.json` and the script recorded beside it.

---

## 4  What must be built

Seven items, in dependency order. Each names what it is, where it lands, and how it is checked.
Everything is inside the existing mechanism; nothing here is a new search algorithm.

### 4.1  The brief as the single input

**Today** a scenario is spread across three files: the chip in `loop/socs.py`, the tunable knobs in
`loop/space.py`, the workloads in `CELLS` in `loop/search.py`, and the caps nowhere. Adding a
scenario means editing code in three places.

**Lands as** `loop/brief.py` plus one YAML file per scenario under `briefs/`. The brief is the only
thing a user writes; `socs.py`, `space.py` and `CELLS` become derived from it.

```yaml
# briefs/generation_step.yaml

shipped_last_generation:
  node_nm: 32
  clock_ghz: 3.2
  cores: 1
  core:   {issue_width: 4, fetch_width: 6, rob: 256, lq: 128, sq: 72}
  memory: {channels: 2, data_rate: 3200}
  design:
    L1D: {kb: 32,   ways: 8,  prefetcher: next_line}
    L2:  {kb: 512,  ways: 8,  prefetcher: spp_dev, replacement: srrip, mshr: 32}
    L3:  {kb: 8192, ways: 16, prefetcher: none,    replacement: srrip, mshr: 128}

new_generation:
  node_nm: 22
  clock_ghz: 3.8
  cores: 1
  core:   {issue_width: 6, fetch_width: 8, rob: 384, lq: 160, sq: 96}
  memory: {channels: 2, data_rate: 4800}

  must_run: [llm_inference, datacenter]

  must_not_exceed:
    cache_area_mm2: 4.0
    cache_power_w: 1.5
    regression_per_workload_pct: 0

  tunable:
    L1D: [kb, ways, prefetcher]
    L2:  [kb, ways, prefetcher, replacement, mshr]
    L3:  [kb, ways, prefetcher, replacement, mshr]

  maximize: speed
```

Three properties make it general rather than a second hardcoded chip.

- **Units the brief's author already uses.** Kilobytes, millimetres squared, watts, gigahertz.
  `sets = kb × 1024 / (ways × 64)` is derived, never written.
- **The brief names which knobs are open, not which values.** The reachable values are already
  derived from the chip and its caps (`loop/chip.py` builds the capacity ladder); a knob absent from
  `tunable` is frozen at last generation's value. No value list is ever written by hand.
- **Caps are a list, not code.** A new quantity to cap is an entry, not a branch.

**Checked by**: `python -m loop.search preview gen` renders every prompt for the brief with no model
call and no search; the rendered chip card, knob lists and capacity ladder must match what the three
hand-edited files produce today, byte for byte, for all five existing SoCs. `loop/socs.py` keeps its
entries as a compatibility shim until that holds.

### 4.2  Real caps, replacing the capacity proxy

**Today** `space.within_budget` compares L2 + LLC data capacity in KB against
`socs.AREA_BUDGET_KB`. On the new chip that admitted 98.9 % of the space. Capacity in KB is also
the wrong quantity: it is blind to associativity, to the node, and to the L1D entirely.

**Lands as** `loop/cost.py` exposing `area_mm2(design)` and `power_w(design, metrics)`, and
`space.within_budget` becoming `constraints.violations(design, metrics) -> [(quantity, value, cap)]`
— a list, so several caps compose and each violation can be named back to the proposer.

Area is available before simulation, so an area violation is refused without spending a design.
Power needs counters, so a power cap is checked after measurement and a violating design is recorded
as infeasible rather than as a score. This asymmetry is real and must be visible in the report.

**Checked by**: recompute area for all 531 quad designs and all 1218 mobile designs from the cached
tables and confirm the cap partitions them as §2.3 predicts; confirm the refusal path names the
overshoot and offers one retry, as the KB budget does today.

### 4.3  The power model on the search path

**Verified available.** CACTI already reports, for every shape in the ladder, `Total dynamic read
energy per access (nJ)`, `Total dynamic write energy per access (nJ)` and `Total leakage power of a
bank (mW)`. ChampSim already records, per level and per workload, `*_hits`, `*_misses`, `*_merges`,
`pf_requested`, `pf_issued`, `pf_useful`, `pf_useless`, plus the memory controller's own request
counts — all of it already in the cached tables.

**Lands as** an extension of the CACTI characterisation in `loop/chip.py` to carry the three energy
fields beside access time and area, cached in the same `results/cacti_<node>nm.json`, and
`cost.power_w` composing them with the counters:

```
per level:  accesses x read energy  +  fills x write energy  +  leakage x instances x run time
off-chip:   LLC misses x 64 B x 8 x 15 pJ/bit        (a DDR4-class figure, stated as an assumption)
watts    =  total joules / (cycles / clock)
```

This makes a useless prefetch cost something for the first time. `pf_useless` is already counted per
level, so the specialists' reports can name it.

**Checked by**: recompute power for all 531 quad designs from their cached counters and reproduce the
0.75–5.78 W span and the 0.2 %-within-2 % sharpness already measured; assert monotonicity where it
must hold (more ways at fixed sets never lowers dynamic energy per access).

### 4.4  Frozen knobs

**Lands as** the derived knob list honouring `tunable`: a frozen knob is absent from
`SEARCH_SPACE`, so the forest cannot sample it, the specialists never see it as theirs, the sweep
never rotates onto it, and it appears in the report under what was not tried and why.

**Checked by**: a brief with `L1D` absent from `tunable` must render prompts that contain no L1D
knob, and a council proposal touching a frozen knob must be refused with the reason named.

### 4.5  The no-regression constraint

**Lands as** a constraint over the per-workload metrics already stored:
`per_workload_ipc[w] >= baseline_per_workload_ipc[w] × (1 − tolerance)` for every `w`. It is a
constraint, not a term in the objective, so the search cannot trade one workload away for the mean.

This is the constraint that carries the most weight outside the loop and the least mechanism inside
it: the metrics are already per workload (`loop/suite.py` stores `<workload>:ipc`).

**Checked by**: the gate's winning design satisfies it (`llama2` +50.0 %, `whiskey` +7.8 %); a
synthetic design that gains on the mean while losing one workload must be rejected and must appear
in the report as rejected.

### 4.6  The recommendation and the frontier

**Today** a run produces `results/runs/<soc>_<cell>_<tag>.json` and `search score` prints
share-of-gap curves. That is the right artifact for comparing arms and the wrong one for reading a
result.

**Lands as** `loop.report`, one page per run, written from the run's own JSON with no model call:
the baseline and the recommendation side by side on speed, silicon and power; the per-workload
deltas with the regression check; the level-by-level change; the reasoning the analyst recorded for
the winning move, with the counter behind it; the frontier — cheapest design that holds the
baseline's speed, the recommendation, the fastest design inside the caps; and what was not tried,
separating frozen knobs from designs refused by a cap. §6 is the full layout.

**Checked by**: rendering it for the quad and mobile runs already on disk, where the recommendation
and frontier are known independently from the cached tables.

### 4.7  The BOOM profiles

**Lands as** further entries in the derived chip table, carrying the core parameters of Chipyard's
`SmallBOOM`, `MediumBOOM` and `LargeBOOM` configurations — issue width, ROB, load/store queue
depths — at a node CACTI accepts.

This is a ChampSim approximation of those cores' front-end and window dimensions, not their RTL, and
no PPA number from it is a BOOM PPA number. Stated that way wherever it appears.

**Checked by**: `preview` on each profile, and the derived chip card matching the published
configuration parameters field by field.

---

## 5  Validation

The order matters: everything that can be checked without simulation is checked first, then
everything that can be checked against tables already on disk, and only then is compute spent.

### 5.1  Free — no simulation

| check | mechanism |
|---|---|
| every prompt the brief produces, per SoC | `preview`, which costs one cached stock row |
| the brief reproduces all five existing SoCs | byte-compare rendered prompts against the current three-file path |
| the cap partitions the space as predicted | recompute area over the 531 + 1218 cached designs |
| the power model reproduces the 7.7× span and the 0.2 % sharpness | recompute power from cached counters |
| frozen knobs are absent everywhere | grep the rendered prompts |
| the report renders correctly | render it for the quad and mobile runs on disk |

### 5.2  The independent ceiling

`search ceiling` — the same random forest run long from the baseline, forest-only, **no model
calls** — establishes the best design reachable on `F_next` under the 4 mm² cap. Until it exists,
`+27.2 %` is a floor and every percentage in §3 is provisional. This removes the caveat that every
number in `CLAUDE.md` currently carries, and it is the cheapest high-value item in this plan.

Sized in designs, not rounds. It runs before the comparison, not after.

### 5.3  The comparison

Two arms, from the same baseline, same caps, same fidelity, same parallelism, same seeds:

- **`random`** — uniform draws from the feasible set, `BO_BATCH` a round. The only baseline this
  scenario claims, matching the paper's ladder. Run at 5 a round for 20 (`g1`) and 75 (`g3r`) rounds.
- **`council`** — the claim as written was "unchanged, so the existing loop handles a new objective
  and a new chip with no new search machinery". **That claim did not survive the first run**: with
  hard caps and the one-step climb the council never left the baseline (§8.3). Three mechanisms were
  added (`OPENINGS`, `HOLD_JUMP`, the per-workload floor in the prompt) and one was tried and
  dropped (`SOFT_CAPS`). Each is an env flag recorded in every report, default off except
  `HOLD_JUMP`, so the older cells are unaffected and each is separately ablatable.

≥ 2 seeds per arm, on the record that the gate's random arm hit its best on a 10 %-probability draw.
Scored in rounds and in designs, both reported.

### 5.4  Ablations, in value order

| ablation | question | cost |
|---|---|---|
| cap off | does the plateau return with no cost term? | one run |
| power cap only, area cap only, both | which cap produces the structure? | free on the ceiling's designs |
| `CHIP_VIEW=0` | does the derived chip view still buy the opening? | one run, protocol already exists |
| the naive-shrink baseline vs a greedy-climb baseline | is the baseline a strawman? | ~3 waves of the gate's shape |

The last is the one a reader will ask for, and it is the honest form of the gate's headroom number:
a greedy climb is what an engineer with a simulator actually does. The gate establishes its first
step (+6.91 %); the full climb has not been run.

---

## 6  The user experience

### 6.1  The flow

```
   1  WRITE THE BRIEF              briefs/<name>.yaml — the chip that shipped, what moved,
      one file, ~25 lines          what it must run, what it must not exceed, what is tunable

   2  CHECK IT, FREE               python -m loop.search preview <brief>
      no simulation, no model      renders the derived chip card, the reachable values under the
                                   caps, the baseline, and every prompt the council will read
                                   -> a brief that reads wrong is caught here, not after hours

   3  GATE IT, CHEAP               python -m loop.search gate <brief>
      one wave, no model calls     the baseline + every feasible one-knob move + a random sample
                                   -> headroom, depth, greedy resistance, deception
                                   -> a brief with no optimisation problem in it stops here

   4  RUN                          python -m loop.search run <brief> --arms council,random
      rounds of parallel waves     each round: the analyst's sheet, four specialists propose on
                                   their own knobs, every proposal sketched, one design committed
                                   -> progress per round, to stdout and the run JSON

   5  READ                         python -m loop.report <run>
      one page, no model call      baseline vs recommendation on speed / silicon / power,
                                   per-workload deltas, the reasoning, the frontier, what was
                                   not tried
```

Steps 2 and 3 are the ones that matter for cost: both are cheap, both can reject a brief, and
neither spends a model call. Step 3 is new and is the gate of §3 promoted from a scratch script.

### 6.2  Inputs

One file. Nothing else is written by hand.

| the brief states | the loop derives |
|---|---|
| node, clock, cores, core dimensions, memory | the chip card, every latency and area from CACTI at that node, the bandwidth per core-cycle, the DRAM service cost |
| last generation's design in KB and ways | the baseline, the reference for no-regression, the starting point |
| which knobs are tunable | the knob lists, the reachable values under the caps, the capacity ladder in cycles and mm², what is frozen |
| what it must run | the workload set and the per-workload objective |
| what it must not exceed | the feasible set, the refusal path, what the report calls out as rejected |

For the common cases the tree carries starter briefs — a server chip, a mobile chip, a multi-core
chip, the generation step, and the BOOM configurations — so a new scenario is a copy and about ten
edited numbers.

### 6.3  Output

The layout below is the specification for §4.6, not a result. The baseline and recommended columns
carry the gate's **measured** numbers (§3); the `RUN` line, the `WHY` text and the two outer
frontier rows are **placeholders showing the intended shape** — no run has produced them, and the
frontier in particular requires the ceiling of §5.2.

```
BRIEF   generation_step   22 nm, 3.8 GHz, 6-wide, 1 core
CAPS    cache silicon 4.0 mm²   cache power 1.5 W   no workload slower than the baseline
RUN     <arm, seeds, rounds, designs, wall clock, model cost>

                              baseline        recommended        change
  speed (geometric mean)        0.7539            0.9588         +27.2 %
  cache silicon                 3.07 mm²          2.60 mm²        -15 %
  cache power                   0.98 W            0.85 W          -13 %
  L1D                        64 x 8  next_line   64 x 12 va_ampm_lite
  L2                       1024 x 8  spp_dev    256 x 4  next_line   512 KB -> 64 KB
  L3                       2048 x 16 none      4096 x 8  spp_dev

  per workload      llm_inference +50.0 %      datacenter +7.8 %      none slower

WHY     <the analyst's own recorded reasoning for the committed move, with the counter
         it cited - composed from the run JSON, never generated at report time>

OPTIONS cheapest design holding the baseline's speed     <mm²>   <W>    <+%>
        recommended                                      2.60 mm²  0.85 W  +27.2 %
        fastest design inside the caps                   <mm²>   <W>    <+%>

NOT TRIED   <knobs frozen by the brief> · <designs refused over 4.0 mm², before
            simulation> · <designs refused over 1.5 W, after measurement>
```

The frontier is the part that is a result rather than a verdict: choosing a point on it is the
reader's decision, and a single recommended design hides that there is one. The `WHY` block is
composed from the analyst's own recorded reasoning and the counter it cited, not generated at
report time.

### 6.4  What is deliberately not in the flow

- **Natural-language intake.** A brief drafted by a model from one sentence is an obvious
  convenience and is deferred until the file format has carried a real run. It adds a failure mode
  — a misparsed cap — to the one input the whole run rests on.
- **Anything chip-specific in the prompts.** The standing rule holds unchanged: *a prompt may name a
  quantity, never a chip, a chip class, or a branch on one.* Verified the same way, by
  `grep -rn` over `loop/*.py` excluding the derived chip table.

---

## 7  The CHIA module

CHIA's ChampSim node builds from a prefetcher source file and runs one trace; it has no
configuration-space build, no design space, no measurement store and no cost model. `CHIA_BLOCKS.md`
already plans items 1, 2, 3 and 7. This scenario adds the cost-and-constraint layer, which is the
part a reuser needs and the part CHIA has none of.

### 7.1  What lands

| # | block | lands in | what it exposes |
|---|---|---|---|
| 2a | the brief | `chia/dse/brief.py` | `Brief.from_yaml(path) -> Brief`; `Brief.chip`, `.baseline`, `.space`, `.caps`, `.workloads`. A dataclass, so it travels through the object store. |
| 2b | cost | `chia/dse/cost.py` | `area_mm2(design, chip)`, `power_w(design, chip, counters)`. Pure functions over a design and a counter dict: no simulator, no Ray, Tier 0 testable. |
| 2c | constraints | `chia/dse/constraints.py` | `violations(design, caps, counters=None) -> [Violation]`. Separates caps checkable before simulation from caps needing counters, which is what lets a harness refuse a design for free. |
| 7 | CACTI in cache mode | `chia/vlsi/sram_cacti/cacti_runner.py` | access time, area **and the three energy fields**, at a node, with associativity, block size, tag array and access mode. Node-keyed cache file, so a worker without a CACTI build reads it. |
| 1 | configuration-space build | `chia/simulators/champsim.py` | already planned; this scenario needs the core and memory sections reachable, not only a prefetcher source |

`cost` and `constraints` are the reusable core. Both are pure functions over numbers, both are
useful without the rest of this loop, and neither imports a simulator.

### 7.2  How it reads to someone using CHIA

A cost-capped sweep, with none of this loop's agents:

```python
from chia.dse import Brief, cost, constraints
from chia.simulators.champsim import ChampSimNode

brief = Brief.from_yaml("briefs/generation_step.yaml")

legal = [d for d in brief.space.sample(500)
         if not constraints.violations(d, brief.caps)]        # area refused for free

champsim = ChampSimNode(brief.chip)
runs = [champsim.run_configuration.chia_remote(d, brief.workloads) for d in legal]

for design, result in zip(legal, get(runs)):
    if constraints.violations(design, brief.caps, result.counters):
        continue                                              # over the power cap
    print(design, result.ipc,
          cost.area_mm2(design, brief.chip),
          cost.power_w(design, brief.chip, result.counters))
```

Three properties a reuser gets and does not have today: a **cap refused before a simulation is
spent**, a **power number from the simulator's own counters** rather than a guess, and a **brief that
is data** — so the same script runs any chip, any cap and any objective without an edit.

The agentic loop in this repo is then one consumer of those blocks, in the `chia/examples/*` layout
(`CHIA_BLOCKS.md` item 6), with one switch between local calls and `chia_remote`.

### 7.3  Testing, to CHIA's convention

Tier 0, offline, no simulator and no network: `cost` and `constraints` on a table of designs with
known areas and counter dicts; `Brief.from_yaml` on the starter briefs, asserting the derived space,
baseline and caps; the CACTI runner against the checked-in node cache. Env-gated live tests exercise
the runner against a real CACTI build and the ChampSim node against a real trace. Public functions
carry docstrings; `docs/api/dse.rst` is an `automodule` list.

---

## 8  Success criteria

### 8.1  The gate — met

Recorded in §3: headroom +27.2 % against ≥ 2 %, depth 12 of 13 against ≥ 3, the best single move at
25 % of the gain against < 60 %, and 6 deceptive knobs against ≥ 1.

### 8.2  Per item

| item | met when |
|---|---|
| 4.1 brief | all five existing SoCs render byte-identical prompts from a brief; a new scenario is one file |
| 4.2 caps | area refused before simulation with the overshoot named; power refused after, recorded infeasible not zero; both visible in the report |
| 4.3 power | the 531 cached quad designs reproduce the 0.75–5.78 W span and 0.2 %-within-2 % sharpness from counters alone |
| 4.4 frozen | a frozen knob appears in no prompt, no sample, no sweep rotation, and is named in the report |
| 4.5 no-regression | a mean-improving, one-workload-losing design is rejected and reported as rejected |
| 4.6 report | renders for the quad and mobile runs already on disk, and its frontier matches the cached tables |
| 4.7 BOOM | derived card matches the published configuration parameters field by field |

### 8.3  The run — measured

The claim as written was: the council reaches 95 % of the **independent ceiling's** gap in fewer
designs than random search, on every seed, and its transcript shows one of the six deceptive knobs
adopted with a measured counter cited for it.

What was measured (full tables in `CLAUDE.md`), against the best design any arm found, 0.9718, since
the independent ceiling of §5.2 does not exist yet:

- **Fewer designs, not on every seed.** Matched at random's own best (0.9547): the council reaches
  it in 3 to 5 rounds and 16 to 25 designs on its good seeds and 10 to 12 rounds on its weak ones;
  random needs 13 rounds and 63 designs on one seed and never reaches it on the other in 375
  designs. That is a 2.6 to 4.3x speedup in rounds. Neither random seed reaches 95 % of the gap in
  75 rounds; three of eight council seeds reach 100 %.
- **The caps slow the council rather than stopping it.** With hard caps and no openings both seeds
  committed a feasible improvement from round 3 and 4 and ended at 85 % and 92 %, below every
  configuration with openings. The openings mechanism buys rounds (90 % at R4 to R6 with ten
  openings against R12 to never with none), not the endpoint.
- **The deceptive move was adopted.** The best design carries a prefetcher at every level, a coupled
  move across concerns. Whether each part was adopted with a measured counter cited has not been
  graded; that is the Jev grounding question in `CLAUDE.md`, unrun on this cell.

Secondary, as reported: rounds and designs to 90 / 95 / 99 % per arm per seed; the best design's
silicon and power against the baseline's (2.93 against 3.07 mm², 0.97 against 0.98 W); the share of
designs refused (34 to 50 % of the council's designs, 77 to 78 % of random's draws, the per-workload
floor binding about twice as often as the power cap).

### 8.4  What failure looks like, and what follows

| outcome | reading | next |
|---|---|---|
| random matches the council at equal designs on every seed | the loop adds nothing on this cell | report it, and say so in the paper — it is the ArchGym result reproduced, not a null |
| the council wins but cites no counter for the coupled move | it stumbled on it | the win is real and the mechanism is not; the grounding gate (`CLAUDE.md`, Jev use 2) becomes the next item |
| the ceiling is barely above +27.2 % | the gate's random draw was near-optimal and the headroom is thinner than it looks | widen the space — L1I and translation knobs, or both caps at once — and re-gate |
| designs are refused so often that rounds carry no measurement | the cap is too tight to search under | raise to 5 mm² (68.2 % feasible) and re-gate; the curve in §2.3 is the dial |

### 8.5  Cost

Measured rates on this tree: a design on `F_next` at 1 M / 2 M is two simulations and completes at
**4.8 designs per minute** on the Mac at 10-way parallelism; the VM does roughly twice that. The
council spends about seven model calls a round at roughly 0.017 USD each. The gate cost 79 designs
and 17 minutes with no model calls; `search ceiling` and the random arm cost no model calls at all.

`preview` and `gate` are the two steps that keep the bill down, and neither spends a model call.
Every launch is sized and approved before it runs, and the VM is stopped whenever nothing is on it.

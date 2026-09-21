# What is in `results/`

Every artifact says which chip it came from. A chip is `<name>-<revision>`, where the revision
is a hash of everything about the SoC that is not a design knob - the clock, the process node,
the profile, the core count, the area budget (`loop/chip.py:revision`). **A design measured on
one revision is never read back for another**, so changing an SoC starts a clean dataset instead
of silently mixing two machines.

```
runs/<soc>_<cell>_<tag>.json   one report per search: every design, its round, its knobs, its
                               IPC, and for the council every round's sketches. This is what
                               `python -m loop.search score <path>` reads.
transcripts/<soc>-<arm>-<tag>-s<seed>.log
                               every prompt and every answer of one run, in order. Large, and
                               gitignored: the numbers live in the report, this is for reading
                               what the council actually said. `<soc>-stdout-<tag>.log` is the
                               run's console output.
tables/<workload>[_w<W>M_s<S>M].json
                               the simulation cache and the dataset: one row per (design,
                               workload), keyed by the design's full name. Every batch reads it
                               before simulating and appends under a file lock, so parallel runs
                               and later sessions share one cache.
cacti_<node>nm.json            CACTI's characterisation of every cache shape at one process
                               node: access time and area. Computed once, then read. Ship this
                               to a machine without a CACTI build and it needs none.
llm_usage*.json                running model cost, per machine.
archive-pre-cacti/             the record of the loop before cache latency came from CACTI. See
                               its own README; nothing here is read by the loop.
```

**A table belongs to the machine that measured it.** The same ChampSim commit and byte-identical
traces give IPC 0.9184 on the Mac and 0.9144 on the VM - the compiler and standard library
differ, which is enough for a simulator that iterates an unordered container. The offset is the
size of the deltas the council reasons about, so one cell is measured on one machine and rows are
never merged across platforms.

## The results that are live

| report | chip | what it is |
|---|---|---|
| `runs/server_llama2_x1.json` | `C_server-bf61` | the first cell on the CACTI chips: council vs tuned forest, 2 seeds, council to 28 rounds and forest to 29. The council reaches 99.9 % of the gap at round 19; both arms find the same best design, the council at round 10 and the forest at round 26. |
| `runs/server_llama2_cv1.json` / `cv0.json` | `C_server-bf61` | the `CHIP_VIEW` ablation, 2 seeds x 8 rounds, council only, measured on the Mac so the two are comparable. `cv1` is the loop as it stands, `cv0` reverts what the council reads and keeps the machine. `cv1` leads at every round. |
| `runs/mobile_aiml_m2.json` | `A_mobile-a868` | the mobile cell, both arms, council to 20 rounds and forest to 50. **The council loses**: it never reaches 90 % of the gap, the forest reaches 100 %. Reconstructed from `transcripts/mobile-stdout-m2.log` - the run was stopped before `run_cell` wrote its report, so the per-design knobs are in the transcripts and `vm_tables/`, not here. See the open problem in `CLAUDE.md`. |

`vm_tables/` holds the VM's rows for these cells, pruned to the current chips: 694 mobile designs
measured on all three aiml workloads and 550 server llama2 designs. They are kept apart from
`tables/` because a table belongs to the machine that measured it.

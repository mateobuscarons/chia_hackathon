# What is in `results/`

Every number in the paper comes from a file here; `CLAUDE.md` section 4 names the file behind each.
The full record of the project, every run, table and transcript behind earlier versions, is on the
`experiments` branch. This folder holds what the paper cites.

```
ledgers/lean/<run>.jsonl          one ledger per run and seed (the CHIA ledger block's rows): the
                                  council's four arms, the random arms rebuilt from their seeds, the
                                  ablation. `python3 council_loop/demo.py` scores them.
ledgers/<tag>.jsonl               a ledger written live by the CHIA-driven loop, one per run.
profiles/<tag>/ChiaProfileCollector.log
                                  the loop's own CHIA profiler log, one per run: every model call
                                  with tokens and USD, every build, run, candidate and gate event.
                                  `chia viz-profile --format spend results/profiles/<tag>` sums it.
report/lean20*/                   the gating tables and per-run money figures, from `council_loop/report.py`.
audit/circt_assess/               CHIA's CIRCT assess stage under the blocks: the issues, one profiler
                                  log per pass, the verdicts against the paper's.
audit/grounding*.jsonl            the grounding audit: the decider's reading of every proposal's evidence.
vm2/results/                      what the cluster VM measured for the runs above: the llama2 result
                                  tables at 1M/2M and 5M/25M (the simulation cache and dataset), the run
                                  reports, the per-call rows (`llm_calls.jsonl`), the gate rows
                                  (`skip_gate.jsonl`, `context_gate.jsonl`), the fidelity check.
runs/, llm_calls.jsonl, skip_gate.jsonl, context_gate.jsonl
                                  the same records for the runs made on this machine (the smoke of the
                                  CHIA-driven loop).
cacti_22nm.json                   CACTI 7's characterisation of every cache shape at 22 nm: access time,
                                  area, energies, leakage. Read, never recomputed, on a machine without CACTI.
```

A result table belongs to the machine that measured it: ChampSim's IPC differs between compilers and
standard libraries on the same design, so every table in the paper comes from one VM and rows are
never merged across machines.

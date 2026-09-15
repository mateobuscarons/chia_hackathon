# Patches

Two patches against frameworks this repo only clones, so each goes through a fork. Both are
written and neither has been opened as a pull request. What they fix, and the other gaps that
are described but not yet written, are in `../CHIA_BLOCKS.md`.

| patch | target | what it fixes |
|---|---|---|
| `0001-champsim-gs-trace-resolver.patch` | CHIA | `_resolve_trace` raises `NotImplementedError` for `gs://`, so a loop on a GCP cluster cannot read traces from a bucket |
| `0002-champsim-spp-dev-ghr-victim.patch` | ChampSim | a heap-buffer-overflow in SPP's lookahead loop, and a GHR victim search that asserts when every entry has confidence 100 |

**`0002` is a setup step, not an optional fix.** Without it, `spp_dev` designs crash, and `spp_dev`
is in the search space. `cluster/vm_bootstrap.sh` applies it; a local checkout must too.

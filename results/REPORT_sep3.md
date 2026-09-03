# Results, Sep 3 2026

## Ground truth (dense sweeps, 9 x 81 configs)

| SoC | trace | baseline IPC | in-budget optimum | headroom | best config |
|---|---|---|---|---|---|
| A_mobile | mcf | 0.1343 | 0.2029 | +51% | `l2s2048_pf-ip_stride_llcs2048_rp-drrip` |
| A_mobile | lbm | 0.1867 | 0.2227 | +19% | `l2s2048_pf-spp_dev_llcs2048_rp-drrip` |
| A_mobile | omnetpp | 0.2350 | 0.2392 | +2% | `l2s2048_pf-ip_stride_llcs2048_rp-drrip` |
| B_midrange | mcf | 0.2864 | 0.3894 | +36% | `l2s2048_pf-ip_stride_llcs2048_rp-drrip` |
| B_midrange | lbm | 0.4621 | 0.5679 | +23% | `l2s2048_pf-spp_dev_llcs1024_rp-lru` |
| B_midrange | omnetpp | 0.4242 | 0.4338 | +2% | `l2s2048_pf-spp_dev_llcs2048_rp-drrip` |
| C_server | mcf | 0.2952 | 0.4299 | +46% | `l2s1024_pf-spp_dev_llcs4096_rp-lru` |
| C_server | lbm | 0.7562 | 1.0282 | +36% | `l2s2048_pf-spp_dev_llcs2048_rp-drrip` |
| C_server | omnetpp | 0.4377 | 0.4838 | +11% | `l2s1024_pf-spp_dev_llcs4096_rp-lru` |

## Headline: learn on ['A_mobile', 'B_midrange'], test on unseen C_server (10 rounds x 2 per round, 3 seeds)

Simulations needed to capture 90% of the achievable gain (lower is better).

| arm | mcf (per seed, median) | lbm (per seed, median) | omnetpp (per seed, median) |
|---|---|---|---|
| random | [21, 3, 7] → **7** | [7, 1, 2] → **2** | [21, 21, 7] → **21** |
| surrogate | [7, 5, 16] → **7** | [5, 8, 3] → **5** | [18, 12, 11] → **12** |
| surrogate_pooled | [6, 6, 6] → **6** | [1, 1, 1] → **1** | [5, 5, 5] → **5** |
| textbook | [4, 1, 8] → **4** | [2, 3, 9] → **3** | [16, 6, 21] → **16** |
| rules | [20, 19, 21] → **20** | [1, 1, 1] → **1** | [20, 13, 9] → **13** |
| analyst | [13, 3, 11] → **11** | [1, 3, 2] → **2** | [5, 11, 11] → **11** |
| full | [21, 17, 11] → **17** | [1, 1, 1] → **1** | [11, 11, 15] → **11** |

## Calibration on shared dispute bets

| forecaster class | dispute bets | Brier (lower better; 0.25 = coin flip) | win rate |
|---|---|---|---|
| surrogate | 3494 | 0.269 | 57% |
| rules | 2103 | 0.463 | 30% |
| hypotheses | 2939 | 0.287 | 51% |

## Playbook after learning on A and B

| rule | record | re-scoped | text |
|---|---|---|---|
| RULE-001 | 0-0 | 0x | Using the `ip_stride` prefetcher for L2 can significantly improve IPC when the baseline IPC is low, by reducing cache misses at both L2 and LLC. |
| RULE-002 | 0-0 | 0x | Using the `spp_dev` prefetcher for L2 can significantly improve IPC when the baseline IPC is low, by reducing cache misses at both L2 and LLC. |
| RULE-003 | 0-0 | 0x | Switching to the `drrip` LLC replacement policy can improve IPC when the baseline IPC is low, with a modest reduction in LLC misses. |
| RULE-004 | 0-0 | 0x | Increasing L2 cache sets to 2048, when paired with the `ip_stride` prefetcher and `drrip` LLC replacement, can provide significant IPC benefits. |
| RULE-005 | 0-0 | 0x | When using the `spp_dev` prefetcher and `drrip` LLC replacement, increasing L2 cache sets to 2048 provides an IPC improvement by reducing L2 cache misses. |
| RULE-006 | 0-0 | 0x | The `spp_dev` prefetcher can significantly improve IPC when paired with `lru` LLC replacement, especially for configurations with 1024 L2 sets and 2048 LLC sets, by reducing both L2 and LLC misses. |
| RULE-007 | 0-0 | 0x | Switching to the `drrip` LLC replacement policy from `lru` can significantly improve IPC when the `spp_dev` prefetcher is active and L2 sets are at 512, largely due to a notable reduction in LLC misses. |
| RULE-008 | 0-1 | 1x | When the `spp_dev` prefetcher and `lru` LLC replacement are used, and baseline IPC is less than 0.23, increasing L2 cache sets from 1024 to 2048 improves IPC by reducing L2 cache misses. |
| RULE-009 | 0-0 | 0x | Using the `spp_dev` prefetcher can provide IPC gains when combined with the `drrip` LLC replacement policy, especially with 2048 L2 sets, by significantly reducing L2 misses. |
| RULE-010 | 0-0 | 0x | Increasing L2 cache sets from 512 to 1024 can improve IPC when using the `spp_dev` prefetcher and `lru` LLC replacement, primarily by reducing L2 cache misses. |
| RULE-011 | 0-0 | 0x | The `drrip` LLC replacement policy consistently outperforms `srrip` in terms of IPC when the `spp_dev` prefetcher is active, particularly evident with 2048 L2 sets and 2048 LLC sets. |
| RULE-012 | 0-0 | 0x | When the baseline IPC is relatively low, specifically below 0.22, using the `ip_stride` prefetcher can provide significant IPC gains, especially when paired with an `lru` LLC replacement policy, by reducing both L2 and LLC cache misses. |
| RULE-013 | 0-0 | 0x | When the baseline IPC is moderate (around 0.28-0.29) and `lru` replacement is used, enabling the `spp_dev` prefetcher can significantly boost IPC by substantially reducing LLC misses. |
| RULE-014 | 0-0 | 0x | When a prefetcher (`spp_dev`) is active and L2 sets are at 1024, switching the LLC replacement policy from `srrip` to `drrip` can improve IPC by reducing LLC misses, especially with 1024 LLC sets. |
| RULE-015 | 3-2 | 2x | For configurations with a prefetcher (`ip_stride`) and `drrip` LLC replacement, and an L2C_mpki greater than or equal to 20, increasing L2 cache sets from 1024 to 2048 can provide a noticeable IPC gain by reducing both L2 and LLC misses. |
| RULE-016 | 1-1 | 1x | When baseline IPC is less than 0.40 and using the `spp_dev` prefetcher with 2048 L2 sets, switching the LLC replacement policy from `srrip` to `lru` can improve IPC and reduce LLC misses, which contradicts conventional expectations that DRRIP or SRRIP are better than LRU. |
| RULE-017 | 0-0 | 0x | When using no prefetcher, increasing L2 sets from 512 to 1024 while keeping other settings constant can lead to a slight improvement in IPC without significant changes in L2C_mpki or LLC_mpki, suggesting a better L2 hit rate for some access patterns. |
| RULE-018 | 0-2 | 2x | The `spp_dev` prefetcher can provide significant IPC gains for B_midrange when paired with `lru` LLC replacement, especially with 1024 L2 sets and 2048 LLC sets, by substantially reducing LLC misses, particularly when LLC misses per kilo-instruction are very high (> 12.0). |
| RULE-019 | 0-0 | 0x | When the `spp_dev` prefetcher is active and L2 sets are 1024, switching LLC replacement policy from `lru` to `drrip` yields a small IPC improvement. |
| RULE-020 | 0-0 | 0x | Increasing L2 cache sets from 512 to 2048 when using `spp_dev` prefetcher, `lru` LLC replacement, and 2048 LLC sets, maintains IPC while significantly reducing L2C_mpki. |
| RULE-021 | 0-2 | 2x | When L2C_mpki is less than 12.25, using the `spp_dev` prefetcher and `lru` LLC replacement, increasing L2 cache sets from 512 to 2048 provides IPC gains by reducing L2 cache misses, especially with 1024 LLC sets. |
| RULE-022 | 1-1 | 1x | For configurations with a prefetcher (`ip_stride`), `lru` LLC replacement, and a baseline IPC greater than 0.45, increasing L2 cache sets from 512 to 2048 does not significantly change IPC, but reduces L2C_mpki. |
| RULE-023 | 0-1 | 1x | When the `spp_dev` prefetcher is active with 512 L2 sets and `srrip` LLC replacement, and `LLC_mpki` is greater than 10, increasing LLC sets from 1024 to 2048 can significantly improve IPC by reducing LLC misses. |
| RULE-024 | 0-0 | 0x | When the baseline IPC is relatively low (below 0.35) and using no L2 prefetcher with SRRIP LLC replacement, increasing L2 cache sets from 512 to 1024 can improve IPC by reducing L2 cache misses. |
| RULE-025 | 0-0 | 0x | When the baseline IPC is moderate, using the `spp_dev` prefetcher and `lru` LLC replacement, increasing L2 cache sets from 1024 to 2048 improves IPC by reducing L2 cache misses. |
| RULE-026 | 0-0 | 0x | When the `spp_dev` prefetcher is active with 2048 L2 sets and 1024 LLC sets, switching the LLC replacement policy from `srrip` to `drrip` yields a small IPC improvement. |
| RULE-027 | 0-0 | 0x | For moderate IPC values (around 0.41), when using the `ip_stride` prefetcher and `srrip` LLC replacement, increasing LLC sets from 1024 to 2048 can improve IPC by reducing LLC misses. |
| RULE-028 | 0-0 | 0x | When using no L2 prefetcher with 2048 LLC sets, switching the LLC replacement policy from `lru` to `drrip` can slightly improve IPC, while potentially increasing LLC misses. |
| RULE-029 | 0-0 | 0x | When using no L2 prefetcher with `drrip` LLC replacement and 2048 LLC sets, increasing L2 cache sets from 1024 to 2048 provides a small IPC gain while reducing L2C_mpki. |
| RULE-030 | 0-0 | 0x | When the `spp_dev` prefetcher is active with 2048 L2 sets and 2048 LLC sets, switching the LLC replacement policy from `lru` to `drrip` provides a small IPC gain, primarily by reducing LLC misses. |

LLM spend to date: $4.41.

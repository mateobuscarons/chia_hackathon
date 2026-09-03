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
| surrogate | [5, 6, 11] → **6** | [2, 2, 2] → **2** | [13, 15, 12] → **13** |
| surrogate_pooled | [1, 1, 1] → **1** | [3, 3, 3] → **3** | [3, 3, 3] → **3** |
| textbook | [8, 6, 6] → **6** | [2, 4, 3] → **3** | [6, 8, 18] → **8** |
| rules | [5, 5, 5] → **5** | [3, 3, 3] → **3** | [20, 21, 9] → **20** |
| analyst | [12, 6, 3] → **6** | [1, 2, 2] → **2** | [7, 7, 9] → **7** |
| full | [21, 21, 7] → **21** | [4, 4, 4] → **4** | [20, 21, 9] → **20** |

## Calibration on shared dispute bets

| forecaster class | dispute bets | Brier (lower better; 0.25 = coin flip) | win rate |
|---|---|---|---|
| surrogate | 6591 | 0.099 | 88% |
| rules | 14169 | 0.349 | 34% |
| hypotheses | 737 | 0.323 | 44% |

## Playbook after learning on A and B

| rule | record | re-scoped | text |
|---|---|---|---|
| RULE-001 | 0-1 | 1x | For workloads with LLC miss rates exceeding 50.0 MPKI, using the IP Stride prefetcher with DRRIP replacement policy can significantly improve performance. |
| RULE-002 | 0-1 | 1x | For workloads with moderately high LLC miss rates, using the SPP_DEV prefetcher can provide a good uplift in performance, especially when paired with larger L2 caches. |
| RULE-003 | 0-0 | 0x | Increasing L2 cache size to 2048 sets often provides a performance boost when using an effective prefetcher like IP Stride. |
| RULE-004 | 0-1 | 1x | When using the IP Stride prefetcher, the DRRIP replacement policy generally outperforms LRU when LLC_mpki is less than 10.0, particularly with larger L2 and LLC caches. |
| RULE-005 | 0-0 | 0x | For workloads with a very high L2 cache miss rate, increasing the L2 cache size to 2048 sets with IP Stride prefetching and LRU replacement can provide a substantial IPC boost, especially when LLC size is 2048 sets. |
| RULE-006 | 0-0 | 0x | When the L2 cache miss rate is high, an L2 prefetcher like IP Stride can significantly improve performance, especially when paired with an LRU LLC replacement policy. |
| RULE-007 | 1-0 | 0x | For high L2 cache miss rates, using the SPP_DEV prefetcher can yield the highest IPC, especially with a 2048 L2 cache and SRRIP replacement. |
| RULE-008 | 0-0 | 0x | For workloads with a high LLC miss rate, the SRRIP replacement policy at the LLC can significantly degrade performance compared to LRU or DRRIP, even with effective prefetchers. |
| RULE-009 | 0-1 | 1x | For workloads with a very low LLC miss rate (LLC_mpki < 5.0), using the SPP_DEV prefetcher combined with a DRRIP LLC replacement policy can provide a notable IPC uplift. |
| RULE-010 | 3-0 | 0x | When LLC miss rates are low, increasing the L2 cache to 2048 sets with a DRRIP replacement policy can improve IPC, particularly when combined with prefetching. |
| RULE-011 | 0-1 | 1x | For workloads with very low L2 cache miss rates (L2C_mpki < 5.0), using the SPP_DEV prefetcher can provide a performance boost, especially when increasing LLC size to 2048 sets and using LRU replacement. |
| RULE-012 | 0-1 | 1x | With SPP_DEV prefetching and a smaller L2 cache, using DRRIP replacement at the LLC level can offer a performance gain over SRRIP for low LLC miss rates (LLC_mpki < 11.0). |
| RULE-013 | 0-0 | 0x | For workloads with low L2 cache miss rates (below 17 MPKI) and low LLC miss rates (below 8 MPKI), using the IP Stride prefetcher with a 2048-set L2 cache and DRRIP LLC replacement policy can lead to significant IPC gains. |
| RULE-014 | 0-1 | 1x | For workloads with moderate L2 miss rates (between 16 MPKI and 27 MPKI), increasing the L2 cache size to 2048 sets in conjunction with SPP_DEV prefetcher and LRU LLC replacement can provide a substantial IPC improvement. |
| RULE-015 | 0-0 | 0x | When L2 cache miss rates are high (above 29 MPKI), using SPP_DEV prefetcher and a 512-set L2 cache, DRRIP outperforms SRRIP and LRU for LLC replacement, especially at lower LLC sizes. |
| RULE-016 | 0-0 | 0x | For workloads with high L2 cache miss rates (around 43 MPKI), using the SPP_DEV prefetcher can provide a substantial IPC boost, regardless of LLC replacement policy or size. |
| RULE-017 | 0-0 | 0x | When L2 cache miss rates are high (above 50 MPKI) and LLC miss rates are very high (above 80 MPKI), using the SPP_DEV prefetcher can still lead to significant IPC gains even with a smaller L2 cache and SRRIP replacement. |
| RULE-018 | 0-0 | 0x | For workloads with high L2 cache miss rates (around 43-46 MPKI), increasing the L2 cache size to 2048 sets significantly improves performance when using the SPP_DEV prefetcher, even at similar LLC sizes and replacement policies. |
| RULE-019 | 0-0 | 0x | For workloads with moderate L2 cache miss rates (around 45 MPKI), the DRRIP LLC replacement policy shows a marginal improvement over LRU when combined with the default 'no' prefetcher, especially with larger L2 and LLC cache sizes. |
| RULE-020 | 0-0 | 0x | For workloads with L2 cache miss rates around 43-45 MPKI, using the IP Stride prefetcher with a 2048-set L2 cache yields similar IPC to the 'no' prefetcher, but only when LLC replacement policy is DRRIP. |
| RULE-021 | 0-0 | 0x | When L2 cache miss rates are around 10-11 MPKI, using the SPP_DEV prefetcher with a 2048-set L2 cache and DRRIP replacement for the LLC can significantly improve IPC over an SRRIP policy, especially when LLC sizes are large. |

LLM spend to date: $5.02.

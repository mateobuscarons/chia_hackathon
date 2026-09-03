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

## Headline: learn on ['A_mobile', 'B_midrange'], test on unseen C_server (10 rounds x 2 per round, 2 seeds)

Simulations needed to capture 90% of the achievable gain (lower is better).

| arm | mcf (per seed, median) | lbm (per seed, median) |
|---|---|---|
| random | [13, 25] → **19.0** | [9, 5] → **7.0** |
| surrogate | [16, 25] → **20.5** | [6, 5] → **5.5** |
| surrogate_pooled | [19, 19] → **19.0** | [7, 7] → **7.0** |
| rules | [6, 7] → **6.5** | [4, 4] → **4.0** |
| full | [6, 21] → **13.5** | [4, 6] → **5.0** |

## Calibration on shared dispute bets

| forecaster class | dispute bets | Brier (lower better; 0.25 = coin flip) | win rate |
|---|---|---|---|
| surrogate | 822 | 0.257 | 51% |
| rules | 740 | 0.281 | 52% |
| hypotheses | 642 | 0.281 | 50% |

## Playbook after learning on A and B

| rule | record | re-scoped | text |
|---|---|---|---|
| RULE-001 | 0-0 | 0x | When L2 cache miss rate is high, enabling the SPP_DEV prefetcher for L2 cache can significantly improve IPC. |
| RULE-002 | 0-0 | 0x | When L2 cache miss rate is high, increasing L2 cache ways from 4 to 8 may slightly decrease IPC. |
| RULE-003 | 0-0 | 0x | When L2 cache miss rate is around 19-20, using the VA_AMPM_LITE prefetcher for L2 and DRRIP replacement for LLC can boost IPC, especially with a 128-set L1D cache. |
| RULE-004 | 0-0 | 0x | When L2 cache miss rate is low and LLC miss rate is moderate, using VA_AMPM_LITE for L2 prefetching and DRRIP replacement for LLC can provide substantial IPC improvement. |
| RULE-005 | 0-0 | 0x | For configurations with low L2C MPKI and moderate LLC MPKI, changing L1D ways from 8 to 12 along with VA_AMPM_LITE for L2 prefetching and DRRIP for LLC replacement can be highly effective. |
| RULE-006 | 0-0 | 0x | When L2 cache misses are moderate to high, enabling the SPP_DEV prefetcher for L2 cache is beneficial for IPC. |
| RULE-007 | 0-0 | 0x | When the LLC miss rate is low, increasing LLC ways from 8 to 16 can slightly improve IPC. |
| RULE-008 | 0-0 | 0x | For configurations with a next_line L1D prefetcher and LLC next_line prefetcher, the SHIP LLC replacement policy can provide better IPC than DRRIP when L2C_mpki is moderate. |
| RULE-009 | 0-0 | 0x | When L2 cache miss rates are high, using a next_line L1D prefetcher with a spp_dev L2 prefetcher consistently results in better IPC, even if LLC prefetcher is next_line and LLC replacement is SHIP. |
| RULE-010 | 0-0 | 0x | For workloads with moderate LLC miss rates, using the VA_AMPM_LITE prefetcher for L2 and the SHIP replacement policy for LLC, combined with increasing LLC sets and ways, significantly improves IPC. |
| RULE-011 | 0-0 | 0x | When LLC miss rates are high, using a next_line prefetcher for L2, combined with a SHIP LLC replacement policy and higher LLC ways, boosts IPC. |
| RULE-012 | 0-0 | 0x | For workloads with high L2 miss rates, the SPP_DEV prefetcher for L2, coupled with a SHIP replacement policy for LLC and next_line L1D prefetcher, can lead to substantial IPC gains. |
| RULE-013 | 0-0 | 0x | For workloads with very low LLC miss rates, enabling next_line prefetchers for both L1D and LLC can improve IPC when combined with a DLRRIP LLC replacement policy and sufficient LLC sets. |
| RULE-014 | 0-0 | 0x | When L2 miss rates are very high, consider using the VA_AMPM_LITE prefetcher for L2 and the SHIP replacement policy for LLC, as this combination has shown good IPC improvements. |
| RULE-015 | 0-0 | 0x | For workloads with moderately high L2 cache miss rates, consider enabling the SPP_DEV prefetcher for the L2 cache, as this can significantly improve performance. |
| RULE-016 | 0-0 | 0x | When the L2 cache miss rate is high, increasing L1D sets can be beneficial, especially when combined with an SPP_DEV L2 prefetcher. |
| RULE-017 | 0-0 | 0x | For workloads with high LLC miss rates, enabling next_line prefetching at the LLC can provide a performance boost, especially with a DRRIP replacement policy. |
| RULE-018 | 0-0 | 0x | When LLC miss rates are high, using a DRRIP replacement policy for the LLC can help improve IPC, especially when combined with VA_AMPM_LITE L2 prefetching. |
| RULE-019 | 0-0 | 0x | Using a SHIP replacement policy at the LLC can be beneficial for IPC, especially for workloads with a high LLC miss rate and when combined with SPP_DEV L2 prefetching. |

LLM spend to date: $5.22.

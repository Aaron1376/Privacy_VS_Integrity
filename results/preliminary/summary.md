### Backdoor sweep (3 seeds; excess ASR = ASR minus the clean run's rate for the same seed and defense)

| Attack | Control | Excess ASR (95% CI) | Main acc | Detect recall | FPR |
|---|---|---|---|---|---|
| trigger_naive | open | 1.00 (0.00) | 0.145 | 0.00 | 0.00 |
| trigger_naive | clip | 0.72 (0.54) | 0.156 | 0.00 | 0.00 |
| trigger_naive | clip_dp | 0.98 (0.04) | 0.129 | 0.00 | 0.00 |
| trigger_naive | secagg_blind | 1.00 (0.00) | 0.127 | n/a | n/a |
| trigger_naive | secagg_bounded | 0.91 (0.12) | 0.129 | n/a | n/a |
| trigger_naive | inspect_norm | 0.93 (0.14) | 0.140 | 0.30 | 0.00 |
| trigger_naive | inspect_median | 0.93 (0.14) | 0.154 | 0.00 | 0.00 |
| trigger_naive | inspect_cosine | 0.93 (0.14) | 0.156 | 0.07 | 0.08 |
| trigger_boost | open | 1.00 (0.00) | 0.106 | 0.00 | 0.00 |
| trigger_boost | clip | 0.72 (0.54) | 0.156 | 0.00 | 0.00 |
| trigger_boost | clip_dp | 0.98 (0.04) | 0.129 | 0.00 | 0.00 |
| trigger_boost | secagg_blind | 1.00 (0.00) | 0.091 | n/a | n/a |
| trigger_boost | secagg_bounded | 0.91 (0.12) | 0.129 | n/a | n/a |
| trigger_boost | inspect_norm | 0.06 (0.11) | 0.137 | 1.00 | 0.00 |
| trigger_boost | inspect_median | 0.93 (0.14) | 0.154 | 0.00 | 0.00 |
| trigger_boost | inspect_cosine | 0.93 (0.14) | 0.156 | 0.07 | 0.08 |
| edgecase_normmatch | open | 1.00 (0.00) | 0.141 | 0.00 | 0.00 |
| edgecase_normmatch | clip | 1.00 (0.00) | 0.149 | 0.00 | 0.00 |
| edgecase_normmatch | clip_dp | 0.78 (0.20) | 0.130 | 0.00 | 0.00 |
| edgecase_normmatch | secagg_blind | 0.78 (0.20) | 0.130 | n/a | n/a |
| edgecase_normmatch | secagg_bounded | 0.78 (0.20) | 0.130 | n/a | n/a |
| edgecase_normmatch | inspect_norm | 0.99 (0.01) | 0.145 | 0.41 | 0.00 |
| edgecase_normmatch | inspect_median | 0.63 (0.26) | 0.159 | 0.00 | 0.00 |
| edgecase_normmatch | inspect_cosine | 0.98 (0.01) | 0.149 | 0.07 | 0.07 |

### Canary exposure (bits; max about 9.97 for 1000 candidates)

| Copies | z | eps at delta=1e-10 (conservative) | Exposure (95% CI) | Main acc |
|---|---|---|---|---|
| 1 | 0.0 | inf | 1.24 (0.64) | 0.152 |
| 1 | 0.3 | 782 | 1.55 (1.35) | 0.128 |
| 1 | 1.0 | 118 | 0.92 (0.87) | 0.043 |
| 1 | 3.0 | 28 | 1.38 (0.89) | 0.017 |
| 8 | 0.0 | inf | 4.98 (1.57) | 0.152 |
| 8 | 0.3 | 782 | 5.37 (3.85) | 0.128 |
| 8 | 1.0 | 118 | 1.19 (1.03) | 0.042 |
| 8 | 3.0 | 28 | 1.31 (1.00) | 0.017 |
| 32 | 0.0 | inf | 8.42 (1.52) | 0.150 |
| 32 | 0.3 | 782 | 7.81 (4.23) | 0.132 |
| 32 | 1.0 | 118 | 1.81 (1.57) | 0.041 |
| 32 | 3.0 | 28 | 1.73 (0.66) | 0.019 |

### Detection AUC when the server can see updates

| Setting | Attack | Norm AUC | Cosine AUC |
|---|---|---|---|
| unclipped | trigger_boost | 1.00 | 0.74 |
| unclipped | edgecase_normmatch | 1.00 | 0.55 |
| clipped | trigger_boost | 0.51 | 0.49 |
| clipped | edgecase_normmatch | 0.53 | 0.51 |

### Membership-inference AUC for the canary client

| Canary copies | Single visible update | Aggregate (secagg) | Aggregate + DP (z=1) |
|---|---|---|---|
| 1 | 0.86 (0.07) | 0.68 (0.08) | 0.55 (0.05) |
| 4 | 1.00 (0.00) | 0.92 (0.07) | 0.53 (0.02) |
| 16 | 1.00 (0.00) | 1.00 (0.01) | 0.60 (0.04) |

### Dose-response: ASR versus number of malicious clients (population 30)

| Control | Attack | k=0 (clean) | k=1 | k=2 | k=3 |
|---|---|---|---|---|---|
| clip | trigger_boost | 0.28 (0.54) | 0.67 (0.65) | 1.00 (0.00) | 1.00 (0.00) |
| clip | edgecase_normmatch | 0.00 (0.00) | 0.34 (0.27) | 0.77 (0.25) | 1.00 (0.00) |
| clip_dp | trigger_boost | 0.00 (0.00) | 0.63 (0.62) | 1.00 (0.00) | 0.98 (0.04) |
| clip_dp | edgecase_normmatch | 0.01 (0.02) | 0.14 (0.21) | 0.53 (0.10) | 0.79 (0.21) |

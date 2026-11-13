# pcdisk GPU run 2026-10-24 (QWEN_PROMPT_CACHE_GB=3, 8 episodes x 10 min)

on_a_v1 = before commit 87961b3 (the fix); off_b/on_b and off_c/on_c = the pairs on the fixed code.

## io_bench
medians over rounds:
| entry GB | flush ms | write s | write GB/s | read ms | read GB/s | finish ms | peak GB | exact |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.9 | 0.16 | 6.12 | 411 | 2.43 | 0.4 | 1.95 | True |
| 2.50 | 1.7 | 0.41 | 6.08 | 683 | 3.66 | 0.7 | 5.79 | True |

## exact check (hit through the disk == cold), fixed code
{"pair": 0, "prompt_tokens": 15622, "tf_cached": 15401, "tf_first_scored": 15402, "tf_equal": true, "tf_disk_load": 1, "gen_cached": 15401, "gen_equal": true, "gen_disk_load": 1, "t": {"tf_hit": 1.154674768447876, "tf_cold": 33.63599610328674, "gen_hit": 2.6207690238952637, "gen_cold": 30.355244874954224}}
{"pair": 1, "prompt_tokens": 67947, "tf_cached": 67940, "tf_first_scored": 67941, "tf_equal": true, "tf_disk_load": 1, "gen_cached": 67940, "gen_equal": true, "gen_disk_load": 1, "t": {"tf_hit": 1.2125048637390137, "tf_cold": 195.65208506584167, "gen_hit": 2.7668843269348145, "gen_cold": 169.12180519104004}}
{"pair": 2, "prompt_tokens": 25668, "tf_cached": 25661, "tf_first_scored": 25662, "tf_equal": true, "tf_disk_load": 1, "gen_cached": 25661, "gen_equal": true, "gen_disk_load": 1, "t": {"tf_hit": 0.7029149532318115, "tf_cold": 58.851141929626465, "gen_hit": 1.85744309425354, "gen_cold": 51.566359996795654}}
3 pairs, 0 failed

## exact check, before the fix
{"pair": 0, "prompt_tokens": 15622, "tf_cached": 15401, "tf_first_scored": 15402, "tf_equal": true, "tf_disk_load": 1, "gen_cached": 15401, "gen_equal": true, "gen_disk_load": 1, "t": {"tf_hit": 1.2952969074249268, "tf_cold": 35.9482102394104, "gen_hit": 2.6275148391723633, "gen_cold": 31.248172998428345}}
{"pair": 1, "prompt_tokens": 67947, "tf_cached": 67940, "tf_first_scored": 67941, "tf_equal": true, "tf_disk_load": 1, "gen_cached": 67940, "gen_equal": true, "gen_disk_load": 1, "t": {"tf_hit": 1.2719290256500244, "tf_cold": 198.66259789466858, "gen_hit": 2.7549991607666016, "gen_cold": 169.0000958442688}}
{"pair": 2, "prompt_tokens": 25668, "tf_cached": 25661, "tf_first_scored": 25662, "tf_equal": true, "tf_disk_load": 1, "gen_cached": 25661, "gen_equal": true, "gen_disk_load": 1, "t": {"tf_hit": 0.741466760635376, "tf_cold": 58.75837516784668, "gen_hit": 1.984692096710205, "gen_cold": 51.54500198364258}}
3 pairs, 0 failed

## replay
| | off_a | on_a_v1 | off_b | on_b | off_c | on_c |
|---|---|---|---|---|---|---|
| requests/min (window) | 10.3 | 12.9 | 10.3 | 23.4 | 10.4 | 24.5 |
| latency p50 / p90 s | 38.4 / 85.4 | 15.5 / 127.1 | 35.8 / 100.4 | 15.0 / 41.0 | 36.8 / 92.9 | 14.1 / 34.9 |
| queue mean / p90 s | 24.9 / 55.6 | 22.1 / 85.1 | 25.2 / 60.5 | 7.3 / 20.0 | 26.5 / 62.9 | 6.9 / 20.2 |
| ttft mean / p90 s | 6.7 / 19.1 | 5.7 / 22.1 | 6.6 / 21.1 | 2.9 / 6.6 | 6.8 / 20.2 | 2.9 / 5.8 |
| decode mean / p90 s | 14.3 / 24.8 | 11.8 / 24.2 | 14.5 / 23.0 | 9.6 / 16.9 | 14.5 / 23.9 | 9.2 / 15.4 |
| hit rate % | 63.1% | 78.1% | 66.7% | 93.8% | 64.3% | 93.7% |
| mean reused tok | 3927 | 5285 | 4145 | 9668 | 4120 | 9580 |
| prefill tok/s wall | 454 | 439 | 418 | 378 | 438 | 383 |
| decode tok/s wall | 9.6 | 11.9 | 9.5 | 21.6 | 9.6 | 21.2 |
| misses | 41 | 30 | 37 | 15 | 40 | 16 |
| evictions to admit | 0 | 0 | 0 | 0 | 0 | 0 |
| sched: prefill / decode / idle % | 87 / 9 / 4 | 86 / 10 / 4 | 87 / 9 / 4 | 78 / 16 / 6 | 87 / 9 / 4 | 79 / 16 / 5 |
| ms per slab / per decode step | 964 / 32 | 974 / 34 | 1009 / 34 | 847 / 39 | 992 / 34 | 822 / 39 |
| step ms at 4 / 6 / 8 rows | 34 / 36 / 39 | 35 / 37 / 40 | 34 / 38 / 39 | 37 / 41 / 43 | 34 / 37 / 39 | 36 / 41 / 43 |
| mean rows (ticks) | 2.52 | 2.38 | 2.48 | 4.16 | 2.60 | 4.32 |
| store MB / checkpoints | 520 / 4.7 | 546 / 4.7 | 517 / 4.6 | 626 / 4.3 | 518 / 4.6 | 624 / 4.2 |
| pcache / active GiB max, peak GB | 3.0 / 13.6, 16.8 | 3.0 / 14.3, 17.4 | 3.0 / 13.2, 16.1 | 3.0 / 15.7, 19.4 | 3.0 / 14.0, 16.3 | 3.0 / 15.7, 19.7 |
| swap GB max, worker CMPRS GB max | 3.7, 0.00 | 3.7, 0.00 | 3.7, 0.00 | 3.6, 0.34 | 3.6, 0.00 | 3.6, 0.39 |
| pageins/s max, decompr/s max (1-min means) | 116, 41 | 380, 59 | 123, 95 | 112, 116 | 115, 140 | 19, 928 |

### off_a
requests 113 (loaded for but resumed from nothing, in miss: 0)
miss: n=43 (38.1 %), ttft p50/p90 14.5/22.5 s, arrival->first token p50/p90 34.0/78.2 s, prompt 7273, prefilled 7273 tok/request
ram: n=70 (61.9 %), ttft p50/p90 1.2/4.5 s, arrival->first token p50/p90 20.3/56.7 s, prompt 6705, prefilled 477 tok/request
disk: 0
MISS >= 10K tokens: n=6 ttft p50 25.1 s, max 26.9 s at 12658 tokens
disk: 0 writes, 0 MB in 0.0 s (p50 0.00 s per entry), 0 not written, 0 failed; 0 loads, 0 MB, p50 0 ms (max 0), 0 failed, 0 dropped, 0 not kept; max 0 entries / 0.00 GiB on disk
peak max 16.75 GB

### on_a_v1
requests 139 (loaded for but resumed from nothing, in miss: 21)
miss: n=32 (23.0 %), ttft p50/p90 21.9/26.0 s, arrival->first token p50/p90 99.4/130.4 s, prompt 8794, prefilled 8794 tok/request
ram: n=71 (51.1 %), ttft p50/p90 0.7/4.5 s, arrival->first token p50/p90 2.5/21.6 s, prompt 6674, prefilled 438 tok/request
disk: n=36 (25.9 %), ttft p50/p90 1.5/6.5 s, arrival->first token p50/p90 6.3/80.5 s, prompt 8713, prefilled 899 tok/request
MISS >= 10K tokens: n=16 ttft p50 24.4 s, max 28.1 s at 13594 tokens
disk: 63 writes, 26647 MB in 6.8 s (p50 0.09 s per entry), 0 not written, 0 failed; 57 loads, 34223 MB, p50 281 ms (max 684), 0 failed, 0 dropped, 0 not kept; max 10 entries / 6.66 GiB on disk
peak max 17.37 GB

### off_b
requests 113 (loaded for but resumed from nothing, in miss: 0)
miss: n=39 (34.5 %), ttft p50/p90 15.8/26.3 s, arrival->first token p50/p90 34.5/95.3 s, prompt 7457, prefilled 7457 tok/request
ram: n=74 (65.5 %), ttft p50/p90 1.2/4.6 s, arrival->first token p50/p90 17.3/59.0 s, prompt 6687, prefilled 470 tok/request
disk: 0
MISS >= 10K tokens: n=7 ttft p50 26.3 s, max 27.3 s at 12307 tokens
disk: 0 writes, 0 MB in 0.0 s (p50 0.00 s per entry), 0 not written, 0 failed; 0 loads, 0 MB, p50 0 ms (max 0), 0 failed, 0 dropped, 0 not kept; max 0 entries / 0.00 GiB on disk
peak max 16.10 GB

### on_b
requests 244 (loaded for but resumed from nothing, in miss: 3)
miss: n=17 (7.0 %), ttft p50/p90 9.1/36.4 s, arrival->first token p50/p90 20.1/57.8 s, prompt 6577, prefilled 6577 tok/request
ram: n=90 (36.9 %), ttft p50/p90 0.8/5.2 s, arrival->first token p50/p90 2.1/12.4 s, prompt 8060, prefilled 425 tok/request
disk: n=137 (56.1 %), ttft p50/p90 1.5/4.7 s, arrival->first token p50/p90 5.2/27.6 s, prompt 12777, prefilled 715 tok/request
MISS >= 10K tokens: n=4 ttft p50 36.4 s, max 41.9 s at 18111 tokens
disk: 148 writes, 25475 MB in 7.3 s (p50 0.04 s per entry), 0 not written, 0 failed; 140 loads, 92180 MB, p50 322 ms (max 749), 0 failed, 0 dropped, 0 not kept; max 13 entries / 7.78 GiB on disk
peak max 19.43 GB

### off_c
requests 114 (loaded for but resumed from nothing, in miss: 0)
miss: n=42 (36.8 %), ttft p50/p90 16.6/22.3 s, arrival->first token p50/p90 37.4/85.4 s, prompt 7080, prefilled 7080 tok/request
ram: n=72 (63.2 %), ttft p50/p90 1.1/4.5 s, arrival->first token p50/p90 18.4/60.8 s, prompt 6895, prefilled 486 tok/request
disk: 0
MISS >= 10K tokens: n=3 ttft p50 22.9 s, max 26.8 s at 12335 tokens
disk: 0 writes, 0 MB in 0.0 s (p50 0.00 s per entry), 0 not written, 0 failed; 0 loads, 0 MB, p50 0 ms (max 0), 0 failed, 0 dropped, 0 not kept; max 0 entries / 0.00 GiB on disk
peak max 16.31 GB

### on_c
requests 255 (loaded for but resumed from nothing, in miss: 3)
miss: n=18 (7.1 %), ttft p50/p90 10.8/34.3 s, arrival->first token p50/p90 21.2/69.2 s, prompt 6107, prefilled 6107 tok/request
ram: n=89 (34.9 %), ttft p50/p90 1.1/4.6 s, arrival->first token p50/p90 3.1/12.1 s, prompt 8273, prefilled 449 tok/request
disk: n=148 (58.0 %), ttft p50/p90 1.5/5.1 s, arrival->first token p50/p90 5.9/22.0 s, prompt 12361, prefilled 689 tok/request
MISS >= 10K tokens: n=3 ttft p50 34.3 s, max 40.1 s at 17281 tokens
disk: 161 writes, 27420 MB in 7.9 s (p50 0.04 s per entry), 0 not written, 0 failed; 151 loads, 97646 MB, p50 319 ms (max 593), 0 failed, 0 dropped, 0 not kept; max 12 entries / 8.14 GiB on disk
peak max 19.74 GB

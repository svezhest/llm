| | reply_kv_r0 | reply_kv_r1 | reply_kv_r0b |
|---|---|---|---|
| requests/min (window) | 19.3 | 20.6 | 19.4 |
| latency p50 / p90 s | 19.7 / 47.1 | 17.0 / 47.1 | 21.3 / 45.3 |
| queue mean / p90 s | 7.6 / 23.1 | 7.0 / 24.2 | 7.0 / 15.8 |
| ttft mean / p90 s | 4.0 / 8.1 | 3.9 / 9.0 | 4.1 / 8.4 |
| decode mean / p90 s | 13.5 / 26.2 | 12.4 / 22.6 | 13.9 / 24.5 |
| hit rate % | 95.0% | 94.9% | 95.0% |
| mean reused tok | 9108 | 9443 | 9102 |
| prefill tok/s wall | 252 | 220 | 257 |
| decode tok/s wall | 17.5 | 18.4 | 17.7 |
| misses | 10 | 11 | 10 |
| evictions to admit | 0 | 0 | 0 |
| sched: prefill / decode / idle % | 79 / 16 / 5 | 80 / 16 / 4 | 79 / 16 / 5 |
| ms per slab / per decode step | 1200 / 50 | 1160 / 52 | 1218 / 51 |
| step ms at 4 / 6 / 8 rows | 47 / 53 / 57 | 45 / 54 / 59 | 48 / 54 / 57 |
| mean rows (ticks) | 4.84 | 4.32 | 4.73 |
| store MB / checkpoints | 615 / 4.2 | 608 / 4.3 | 615 / 4.2 |
| pcache / active GiB max, peak GB | 6.0 / 19.0, 22.3 | 6.0 / 18.7, 22.3 | 6.0 / 18.4, 22.3 |
| swap GB max, worker CMPRS GB max | 3.9, 0.59 | 3.9, 0.56 | 3.9, 0.58 |
| pageins/s max, decompr/s max (1-min means) | 104, 2093 | 181, 1495 | 172, 1606 |

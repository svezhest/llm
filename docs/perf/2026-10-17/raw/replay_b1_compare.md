| | b16a | b1 | b16b | b2 |
|---|---|---|---|---|
| requests/min (window) | 21.4 | 16.7 | 21.7 | 19.2 |
| requests in window / total / errors | 214 / 222 / 0 | 167 / 175 / 0 | 217 / 225 / 0 | 192 / 200 / 0 |
| tasks started | 9 | 9 | 10 | 10 |
| latency mean / p50 / p90 s | 21.9 / 15.7 / 38.0 | 28.0 / 25.7 / 42.8 | 22.6 / 15.0 / 41.9 | 24.3 / 22.4 / 36.0 |
| wait before the worker (gateway) mean / p50 / p90 s | 0.0 / 0.0 / 0.0 | 24.4 / 22.4 / 38.0 | 0.0 / 0.0 / 0.0 | 18.1 / 16.7 / 27.2 |
| worker queue mean / p50 / p90 s | 7.9 / 1.4 / 19.9 | 0.0 / 0.0 / 0.0 | 8.8 / 1.3 / 23.0 | 0.4 / 0.0 / 0.2 |
| worker ttft mean / p50 / p90 s | 3.3 / 1.4 / 6.9 | 1.9 / 0.9 / 4.5 | 3.4 / 1.4 / 6.6 | 2.1 / 1.0 / 4.5 |
| worker decode mean / p50 / p90 s | 10.7 / 8.9 / 19.1 | 1.6 / 1.4 / 3.0 | 10.3 / 8.8 / 18.5 | 3.6 / 3.0 / 6.3 |
| prefill / decode tok/s (wall) | 290 / 19.0 | 188 / 15.2 | 253 / 19.0 | 203 / 17.5 |
| cache hit rate / mean reused | 94.6 % / 9.8K | 94.9 % / 8.4K | 94.7 % / 9.8K | 95.5 % / 9.2K |
| scheduler prefill / decode / idle % | 79 / 16 / 5 | 50 / 45 / 5 | 81 / 15 / 4 | 54 / 42 / 5 |
| ms per slab / per decode step | 987 / 46 | 889 / 30 | 993 / 45 | 842 / 40 |
| mean decode rows per step | 5.35 | 1.00 | 5.44 | 1.66 |
| probe 8K TTFT s / decode tok/s, cool -> warm | 13.8 / 36.7 -> 19.8 / 33.8 | 17.9 / 36.4 -> 19.4 / 34.3 | 17.7 / 36.5 -> 19.4 / 35.2 | 17.8 / 36.4 -> 19.6 / 33.3 |

reply identity vs other runs, per (task, call) present in both (own replies: a task diverges at its first different reply):

| pair | matched calls | same reply | same prompt length | tasks matched | tasks identical over all matched calls | first divergence (call index) median |
|---|---|---|---|---|---|---|
| b16a vs b1 | 175 | 128 (73 %) | 133 | 9 | 5 | 5.0 |
| b16a vs b16b | 219 | 168 (77 %) | 175 | 9 | 5 | 12.0 |
| b16a vs b2 | 199 | 145 (73 %) | 150 | 9 | 5 | 5.0 |
| b1 vs b16b | 175 | 124 (71 %) | 131 | 9 | 5 | 5.5 |
| b1 vs b2 | 175 | 175 (100 %) | 175 | 9 | 9 | - |
| b16b vs b2 | 200 | 140 (70 %) | 147 | 9 | 5 | 5.5 |

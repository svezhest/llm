# Performance work logs

One directory per measurement campaign. Each holds the reports an agent wrote at the time
(numbers as measured, machine state noted), the raw JSON/TXT of the benchmark driver, the
bug-review notes of that round and the design docs. CLAUDE.md "Performance notes" is the
distilled version; these are the evidence.

- `2026-10-12/` — continuous batching + `mrqmm.py` (multi-row 8-bit decode matmul) +
  mlx 0.32.3 / mlx-vlm 0.7.6 bump + `pfqmm.py` (prefill GEMM tiles) + fixed 2048 fp16 slab
  on Ornith-1.5 9B. Start with `final_bench.md` (end-to-end, c = 1..16), then
  `step_profile.md` (decode), `prefill_profile.md` (prefill), `mrqmm.md`, `batching_gpu.md`,
  `microbench.md`, `baseline.md` (pre-change numbers). `reviews/` = six bug-review rounds
  (all findings fixed), `design/` = batching design, library versions, worker change log,
  bump recipe, `bump/` = the old-vs-new stack A/B raw data, `raw/` = driver JSON/TXT.
  `prompt_cache_bench.md` — `QWEN_PROMPT_CACHE=1` under batching: 16 concurrent 8K prompts on
  one shared system prompt vs 16 distinct (TTFT p50 20.5 vs 155 s, 15/16 hits, 0.3–0.4 s per
  hit; 3 GB holds 6 entries of 8K / one of 32K), raw `raw/pcache_*`.
  `cascade_microbench.md` — grouped shared-prefix ("cascade") decode attention vs today's per-row
  SDPA, synthetic 9B shapes (`scripts/perf/microbench/microbench_cascade.py`): B=16/P=8K saves 13–16
  ms per step (+20–28 % decode), 32K ×1.75; fp32 formulation beats today's kernel numerically,
  fp16 scores fail; worth it only for groups ≥ 8 rows on ≥ ~4K prefixes; raw `raw/cascade_*`.

Tools live in `scripts/perf/` (`prof/` in-process profilers and A/B scripts run from
`src/worker` with its venv, `microbench/` kernel-level microbenchmarks and the batching
prototype, `final/` the end-to-end driver + table builders). Scripts that write files use
`PERF_OUT` (default `/tmp/perf`). Method: CLAUDE.md "Measuring performance on this machine".

# Performance work logs

One directory per measurement campaign. Each holds the reports an agent wrote at the time
(numbers as measured, machine state noted), the raw JSON/TXT of the benchmark driver, the
bug-review notes of that round and the design docs. CLAUDE.md "Performance notes" is the
distilled version; these are the evidence.

Redacted for publication (2026-11-13): the agents' scratch paths read `/tmp/scratch`,
the memory samples (`*mem*.jsonl`) no longer list the machine's other processes, and a
few phrases about the download link were reworded, also inside the documents used as
benchmark prompts in `raw/` (so those prompts are a few tokens off the measured ones).
Numbers are unchanged.

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
  `prompt_cache_phase1_gpu.md`, `segments_partA_gpu.md` — the prompt-cache rework on the real
  model (hit == cold 26/26 each; segments: 16 siblings of an 8K prefix = 1.35 GB, 120K store
  4.56 GB without OOM), raw `raw/pcache2_*`, `raw/pcache3_*`. `cascade_gpu.md` — the cascade
  gate (top-1 ≥ 99.7 % at B=8/16, 8K/32K) and waves: step 8K×16 66 → 54 ms, 16×32K all admitted,
  default flipped to on; raw `raw/cascade_*`. Reviews: `reviews/prompt_cache_phase1.md`,
  `segments_partA.md`, `cascade_b1.md`, `cascade_b2b3.md`, `cascade_b2b3_fixes.md`; design:
  `design/prompt_cache_opt.md`, `prompt_cache_opt_2.md`, `cascade_plan.md`.

- `2026-10-16/` — the AppWorld agentic load (8 processes, 5–33K contexts, 94 % cache hits):
  `HANDOFF-appworld.md` (the user's handoff + the diagnosis and the night's summary),
  `replay_bench.md` (replay of the recorded traces: memory knobs, 16 episodes, prefill share;
  prefill is 76–80 % of the scheduler, the GPU throttles ×1.5–1.85 under sustained load; raw
  `raw/replay_*`), `rattn_microbench.md` (decode profile at 12–30K and the ragged multi-row
  attention kernel `src/worker/rattn.py`, not integrated; raw `rattn_raw/`), `chain_evict_gpu.md`
  (hit == cold gate for the chain checkpoints + eviction order, 34/34; raw `raw/chain_evict_gpu_*`),
  `design/packed_prefill.md` (varlen/packed prefill feasibility: no-go), sims `evict_sim.py`,
  `chain_ckpts_sim.py`; reviews `reviews/chain_evict.md`, `chain_evict_fixes.md`, `rattn.md`.

- `2026-10-17/` — follow-ups on the agentic load: `REPORT-appworld-run-1017.md` (the user's
  second AppWorld run: 8 → 6 processes, the 115K episode, swap, cross-version divergence),
  `reply_kv_gpu.md` (the cache keeps the decoded reply's KV, `QWEN_PROMPT_CACHE_REPLY`, default on:
  top-1 99.88 % / KL 5.8e-5 vs cold, −21 % prefilled tokens, +6.5 % requests/min; raw
  `raw/reply_kv_*`), `replay_batch1.md` (determinism costs: `QWEN_MAX_BATCH=2` byte-identical
  replies at −11 % requests/min, B=1 −22 %; two B=16 runs share 77 % of replies), `slabs.md`
  (slab histogram and cost on the agentic load: a third of slabs are 1–8 tokens at ~100 ms each,
  merging them ≈ +2 %), `pfqmm_short_slabs.md` (mixed last tile: GEMMs ×1.08–1.19 on 64–416-row
  slabs, bit-identical), designs `design/reply_kv.md`, `design/partial_store.md` (store on cancel
  mid-prefill, chain policy for long fresh entries), `design/packed_prefill.md`; reviews under
  `reviews/`.

Tools live in `scripts/perf/` (`prof/` in-process profilers and A/B scripts run from
`src/worker` with its venv, `microbench/` kernel-level microbenchmarks and the batching
prototype, `final/` the end-to-end driver + table builders). Scripts that write files use
`PERF_OUT` (default `/tmp/perf`). Method: CLAUDE.md "Measuring performance on this machine".

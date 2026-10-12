# mrqmm: multi-row 8-bit quantized matmul for batched decode

Date 2026-10-12, M1 Max 32 GB, mlx 0.31.2 / mlx-vlm 0.6.2 (repo venv). GPU lease held; the batching
worker on :50051 was stopped for the microbenchmarks and restarted for the GPU suites.
Code: `src/worker/mrqmm.py`, `src/worker/tests/test_mrqmm.py`, hook at the end of `Engine.load`
(worker.py). Scripts + raw data: `scratchpad/mrqmm2/` (bench2.py, step_ab.py, real_check2.py,
check_v2.py, *.json/*.txt), GPU-suite outputs: `results/mrq/`.

## Design

`mx.fast.metal_kernel`, one kernel per (K, N, row bucket) with the constants baked into the
source (host cost ~1.5 µs/call vs ~6 µs with template args). It computes Cᵀ = W·xᵀ with
simdgroup 8×8 MMAs (fp16 operands, fp32 accumulators). Each simdgroup owns NA×8 output
features and all rows (padded to 8 or 16); KS simdgroups per threadgroup split K by
interleaved 64-groups and reduce once through threadgroup memory. The K loop has no
threadgroup memory and no barrier: per 64-group a thread loads ONE 16-byte uint4 of weights
per fragment (8 rows × 64 B per simdgroup = 8 cache lines), dequantizes two weights at a
time into a half2 with the 0x6400 trick (mask, insert, or, −1024 is exact) and one half2 fma
with the group's scale/bias (`w = s·q + b`, fp16 rounding of the dequantized weight like
MLX's qmm loader). The k order within a 64-group is permuted identically for A and B
(logical k c of MMA step st = physical 8c + st), so x is also one 16-byte load per row per
group, read straight from device memory (x is tiny and cache-resident). Rows ≥ M are
computed from row 0 and never stored. Config: 1–8 rows NA 2 × KS 8; 9–16 rows NA 4 × KS 4;
N < 4096 (or not a multiple of 8·NA) NA 2/1.

Routing: `install(language_model)` swaps `__class__` of every `nn.QuantizedLinear` that is
8-bit affine g64 with fp16 scales/biases, N % 8 == 0, K % 64 == 0 and **N ≥ 512** to
`MRQuantizedLinear`, and wraps mlx-vlm's `_decode_quantized_linears_fused` (the GDN
in_proj_qkv/z/b/a concatenated decode matmul) for the GDN layers whose four in-projections
are all supported (flag on the first linear). A call takes the kernel only if x is fp16 and
decode-shaped `(rows, 1, K)` with 3 ≤ rows ≤ 16; everything else is the original
`quantized_matmul`. Kernels for all covered shapes × both row buckets compile at install
(~1 s).

Coverage logged at load:
* 9B: `201 modules (in_proj_qkv/z x24, out_proj x24, lm_head, mlp gate/up/down x32, q/k/v/o x8) + fused GDN in-projection x24; left on quantized_matmul: 8-bit g64 N < 512 x48` (in_proj_b/a, only ever used inside the fused call).
* 35B (from its config, not loaded here): q/k/v/o x10, GDN in-proj + out_proj x30; router `mlp.gate` (N=256) and `shared_expert_gate` (N=1) stay on qmm, 3/4/6-bit experts, shared expert, lm_head untouched.
* 27B (5-bit): no-op.

Paths that can never take it: B ≤ 2 (bit-exact per-row qmv kept), prefill slabs (1, L, K),
the DFlash verify (1, T, K) and the drafter's call of the shared lm_head (1, bs−1, K; bs=2
gives rows=1), B > 16, QWEN_FP16=0 (bf16 scales: nothing swapped; `worker._install_gdn_fuse`
likewise fuses no GDN in-projection then), the vision tower and drafter (install gets the
language model only).

## Review of the previous agent's code — findings and fixes

Correct as written: tile edges (N % 8 == 0 makes a fragment fully in or out; inactive
fragments read a valid row and are not stored), K not a multiple of KS·64 (strided groups,
reduction covers all KS), fp32 accumulation, scale/bias indexing (`scales[n, g]`), the
transposed `(N, K/4) uint32` layout, row padding, warm-up and fused-cache reuse. Tests
pass (synthetic K/N tails, every config, outliers, dispatch boundaries, tiny qwen3_5
forward). Changes:
1. **x load layout** (speed): new k permutation so each thread's x is one uint4 per row per
   group instead of 4 scattered 32-bit loads: equal at ≤ 8 rows, **−12 % at 16 rows**
   (step linears 53.2 → 46.3 ms). Correctness re-checked (synthetic sweep incl. all configs,
   real weights).
2. **dtype safety**: `mrqmm()` asserts fp16 x and K == w's K on each new (shape, dtype); the
   call cache key now includes the dtype (a bf16 x would have been read as fp16 garbage by
   a direct caller).
3. **Compile race**: cache misses take a lock (warm-up at load vs the scheduler thread).
4. **N ≥ 512 (MIN_N)**: the 35B router (N=256) and k/v (N=512 is kept) measured no gain
   (×0.96–1.03, launch-bound); the router now stays on qmm so MoE top-k routing keeps the
   library numerics. The fused GDN path is vetted per layer (all four in-projections
   supported) instead of "all four swapped", since in_proj_b/a (N=32) are below MIN_N.
5. Tests added: bf16 x rejected, strided / offset-view x, MIN_N default, fused GDN vetted by
   group with b/a unswapped (+ uninstall restores bit-identical logits).

Alignment note: x goes through `ensure_row_contiguous`; a row-contiguous view starts on a row
(K % 64 == 0), so the 16-byte loads stay aligned (tested with offset views).

Tests: `uv run --group dev pytest -q tests/test_mrqmm.py` 134 passed (+1 real-weights skipped);
`MRQMM_REAL_WEIGHTS=1 ... pytest -q tests/` 187+1 passed.

## Correctness on real weights (`real_check2.py`, M = 3..16, normal and ×30/×60 outlier x)

Max over M and both x kinds. "vs ref" = fp32 dequant → fp32 matmul; qmm's own error in ().

| model | shape | K×N | max\|Δ\| vs qmm | max\|Δ\| vs ref (qmm) | max rel | rms rel |
|---|---|---|---:|---:|---:|---:|
| 9B | q_proj | 4096×8192 | 0.0156 | 0.0113 (0.0179) | 5.4e-4 | 2.9e-4 |
| 9B | k/v_proj | 4096×1024 | 0.031 | 0.0098 (0.031) | 5.9e-4 | 2.9e-4 |
| 9B | o_proj | 4096×4096 | 0.0625 | 0.031 (0.031) | 6.2e-4 | 3.0e-4 |
| 9B | gdn out_proj | 4096×4096 | 0.031 | 0.011 (0.031) | 6.4e-4 | 2.9e-4 |
| 9B | gate/up_proj | 4096×12288 | 0.0078 | 0.0057 (0.0057) | 5.0e-4 | 2.9e-4 |
| 9B | down_proj | 12288×4096 | 0.0156 | 0.0052 (0.0116) | 4.9e-4 | 2.9e-4 |
| 9B | lm_head | 4096×248320 | 0.0156 | 0.0105 (0.0105) | 4.8e-4 | 2.9e-4 |
| 9B | gdn in_proj fused | 4096×12352 | 0.0156 | 0.0083 (0.0083) | 4.4e-4 | 2.8e-4 |
| 35B | q_proj | 2048×8192 | 0.0156 | 0.0095 (0.0141) | 5.3e-4 | 2.9e-4 |
| 35B | k/v_proj | 2048×512 | 0.0156 | 0.006 (0.0158) | 4.6e-4 | 3.0e-4 |
| 35B | o_proj / gdn out_proj | 4096×2048 | 0.0156 | 0.0099 (0.022) | 6.5e-4 | 2.9e-4 |
| 35B | router gate (not covered) | 2048×256 | 0.0156 | 0.008 (0.014) | 5.6e-4 | 3.0e-4 |
| 35B | gdn in_proj fused | 2048×12352 | 0.0156 | 0.0097 (0.0097) | 5.1e-4 | 2.8e-4 |

Worst max rel 6.5e-4, rms rel 2.9e-4 = fp16 output rounding; mrqmm's error vs the fp32
reference is ≤ qmm's own on every shape; no NaN/Inf. Full table: `mrqmm2/real_check.txt`.

## Speed per shape (`bench2.py --rows 1,3,4,8,12,16 --rounds 5 --variants v1_orig`)

µs per matmul, median of 5 interleaved rounds (each a 16-deep chain of distinct weight
copies > 256 MB; small shapes are partly launch-bound in this chain), ×qmm at the same M.
`mr` = final kernel, `v1` = the previous agent's version.

| shape (count/step) | K×N | kernel | M=1 | M=3 | M=4 | M=8 | M=12 | M=16 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q_proj (8) | 4096×8192 | qmm | 137 | 207 | 254 | 363 | 375 | 376 |
| | | mr | 139 (1.01) | 144 (0.70) | 145 (0.57) | 148 (0.41) | 204 (0.54) | 204 (0.54) |
| k/v_proj (16) | 4096×1024 | qmm | 50 | 63 | 70 | 96 | 95 | 97 |
| | | mr | 53 (1.07) | 58 (0.93) | 59 (0.83) | 59 (0.62) | 73 (0.77) | 74 (0.76) |
| o_proj (8) | 4096×4096 | qmm | 87 | 132 | 155 | 253 | 220 | 226 |
| | | mr | 92 (1.05) | 101 (0.77) | 99 (0.64) | 100 (0.39) | 135 (0.61) | 138 (0.61) |
| gdn in_proj fused (24) | 4096×12352 | qmm | 187 | 292 | 363 | 554 | 563 | 562 |
| | | mr | 203 (1.08) | 201 (0.69) | 200 (0.55) | 206 (0.37) | 296 (0.52) | 297 (0.53) |
| gdn out_proj (24) | 4096×4096 | qmm | 86 | 125 | 148 | 245 | 220 | 222 |
| | | mr | 90 (1.04) | 96 (0.77) | 97 (0.65) | 97 (0.40) | 134 (0.61) | 134 (0.61) |
| mlp gate/up_proj (64) | 4096×12288 | qmm | 185 | 288 | 358 | 524 | 529 | 530 |
| | | mr | 190 (1.03) | 192 (0.67) | 193 (0.54) | 196 (0.37) | 274 (0.52) | 276 (0.52) |
| mlp down_proj (32) | 12288×4096 | qmm | 195 | 309 | 387 | 520 | 529 | 534 |
| | | mr | 198 (1.02) | 204 (0.66) | 204 (0.53) | 215 (0.41) | 308 (0.58) | 307 (0.58) |
| lm_head (1) | 4096×248320 | qmm | 3129 | 4990 | 6470 | 8856 | 8987 | 9007 |
| | | mr | 3127 (1.00) | 3047 (0.61) | 3122 (0.48) | 3229 (0.36) | 4516 (0.50) | 4591 (0.51) |

Step linears (sum × layer counts), ms: 

| M | qmm | mr (final) | v1 (previous) |
|---:|---:|---:|---:|
| 1 | 30.3 | 31.3 (1.03, not used) | 31.0 |
| 3 | 47.0 | 31.9 (0.68) | 32.0 |
| 4 | 58.5 | 32.0 (0.55) | 31.9 |
| 8 | 84.6 | 32.9 (0.39) | 33.0 |
| 12 | 84.8 | 46.1 (0.54) | 51.4 (0.61) |
| 16 | 85.2 | 46.3 (0.54) | 53.2 (0.62) |

* Up to 8 rows the kernel streams weights at qmv speed (M=8 costs 1.08× a 1-row qmv step);
  at 16 rows it is MMA-bound (~5.8 TFLOPS on gate_proj).
* **lm_head stays on mrqmm**: ×0.36 at 8 rows, ×0.51 at 16 (9 → 4.6 ms).
* 35B 8-bit shapes (`full_35b.txt`): step linears 12.4 / 14.2 / 20.2 / 20.6 → 10.4 / 10.4 /
  10.5 / 13.3 ms at M = 3 / 4 / 8 / 16 (x0.84 / 0.73 / 0.52 / 0.65); k/v (N=512) and the
  router (N=256) are launch-bound, x0.96–1.03 at M ≤ 4 → the router is excluded (MIN_N).
* Config sweep at M=16 (`v2_cfg16.json`): NA×KS 2×8 / 4×4 / 2×4 within 3 %; 4×8 −5 %;
  NA=8 spills registers (×4.5–6 slower). The previous agent had measured software
  pipelining and 2–4 groups per iteration as slower or equal; not retried.

## Whole decode step, real 9B (`step_ab.py`, same process, install/uninstall toggled, ABBA order)

`QWEN_KV_BITS=0`, ctx 1024, 32 steps after 4 warm-up, step-ahead pipelined like the worker,
5 rounds, median (per-round values in `mrqmm2/step_ab.json`, spread ≤ 2 % except one
outlier round).

| B | ms/step qmm | ms/step mrqmm | × step | aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 27.0 | 27.1 | 1.00 | 37.0 → 36.9 |
| 2 | 31.1 | 31.1 | 1.00 | 64.3 → 64.3 |
| 4 | 55.9 | 30.6 | **1.83** | 71.6 → 130.7 |
| 8 | 86.3 | 34.1 | **2.53** | 92.7 → 234.7 |
| 16 | 92.6 | 51.8 | **1.79** | 172.9 → 309.1 |

(The earlier estimate in microbench.md §1b was 73 / 84 / 110 ms; the final kernel is faster
than the prototype and the machine was in a faster state tonight: compare ratios.)

## Agreement (teacher-forced, `gpu_tests/agree.py`, 128 steps, same session)

| B | QWEN_MRQMM=1 | QWEN_MRQMM=0 |
|---:|---|---|
| 2 | 100.00 %, KL 0 (bit-exact) | 100 %, KL 0 |
| 3 | 100.00 %, KL mean 8.0e-6, max 1.3e-4 | 100 %, KL 0 |
| 4 | 100.00 %, KL mean 8.3e-6, max 1.3e-4 | 100 %, KL 0 |
| 8 | 100.00 %, KL mean 8.7e-6, max 1.5e-3 | 100 %, KL 8.5e-6 |
| 16 | 99.71 % (2042/2048), KL mean 9.7e-6, max 3.1e-4 | 99.80 % (2044/2048), KL 1.1e-5 |

B=3/4 lose bit-exactness vs solo (they were per-row qmv before), the same noise class as
B ≥ 6 with qmm.

## GPU suites (worker, `gpu_tests/wtest.py`; outputs in `results/mrq/`)

* `QWEN_MAX_BATCH=1`: corpus **15/15 byte-identical** to `bgpu/corpus_old.json` (install is
  skipped below 3 rows of batch).
* Default (max_batch 16, mrqmm on, log line above): corpus sequential 15/15 byte-identical;
  seq 16 × 256 **16/16 byte-identical** to `bgpu/seq16_old.json`.
* Free-running vs solo (greedy, 256 tokens): c=16 mrqmm 8/16 identical (first divergence at
  254–702 chars) vs qmm 10/16 (221–753, same session); c=8 6/8 for both. Near-tie flips at the
  same positions, the same class.
* grammar: no grammar errors; j1/j3/j0 valid schema JSON; j2 (the "Aliens" endless tag list)
  hits max_tokens 500 → the suite's strict check prints FAIL, exactly as in the qmm run
  (`bgpu/features_9b.txt`). mixedmax PASS (exact counts 1/7/64/512), stops PASS (no leak),
  think PASS (budget row 34 reasoning tokens).

## Gateway A/B (`scripts/bench_concurrency.py --concurrency 8,16 --max-tokens 512 --repeat 1 --probe 64 --prompt-tokens 600`, :8090)

Worker restarted between runs (order mrqmm 1 → 0 → 1), nice 0, default env otherwise.

| run | c | agg tok/s | TTFT mean s | lat mean s | decode tok/s/req | probe tok/s | worker steady steps/s (B=c, no prefill) |
|---|---:|---:|---:|---:|---:|---:|---|
| mrqmm 1 (a) | 8 | 93.0 | 9.89 | 43.4 | 15.6 | 37.1 | 19.2–19.8 |
| mrqmm 0 | 8 | 51.9 | 10.42 | 78.5 | 7.6 | 36.2 | 8.2–8.6 |
| mrqmm 1 (b) | 8 | 93.3 | 9.40 | 43.2 | 15.4 | 36.7 | 19.0–19.3 |
| mrqmm 1 (a) | 16 | 99.3 | 20.08 | 79.0 | 8.9 | 36.2 | 12.0–13.2 |
| mrqmm 0 | 16 | 68.7 | 21.41 | 116.0 | 5.5 | 34.3 | 6.2–7.7 |
| mrqmm 1 (b) | 16 | 80.7 | 23.64 | 96.1 | 7.2 | 36.1 | 8.9–11.3 |

c=8 **×1.79**, c=16 **×1.17–1.45** (run b had slower steady steps; the user was on the
machine, load avg ~2.7). JSON: `results/mrq/bench_mrq{1_a,0,1_b}.json`.

Open: the worker's steady-state step (B=8 ~51 ms, B=16 ~76–110 ms) is well above the
microbench (34 / 52 ms); with qmm the gap was hidden behind the matmuls. Worth a profile
(per-row attention kernels, per-row sampling, host work) before the next kernel.

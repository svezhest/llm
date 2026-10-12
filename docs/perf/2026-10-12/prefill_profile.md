# Prefill profile, 9B (Ornith-1.5 8-bit, fp16 KV), mlx 0.32.3 / mlx-vlm 0.7.6

Date 2026-10-12, M1 Max 32 GB. GPU lease held; the :50051 worker was stopped for the in-process runs.
Tools (scratchpad/prof/): `prefill_prof.py` (op-class profile), `ab_slab.py` (one-slab in-process A/B),
`ab_prefill.py` (full-prompt in-process A/B, worker slab loop), `kl_check.py` (teacher-forced KL),
`gemm_bench.py`, `sdpa_thresh.py`; kernel prototypes in scratchpad/pfqmm/.

## Method

`prefill_prof.py` loads the model like `Engine.load` (fp16 cast, fast-path overrides, rows attention,
GDN fuse), builds a synthetic cache at the offset (random KV of exactly `offset` tokens, random
recurrent state) and runs one worker slab (`lm(..., skip_logits=True)` + `mx.eval(state)`).
Every op of interest is wrapped: its inputs are evaluated first (that time goes to `other`), then the
op runs and its outputs are evaluated (time -> the op's class). The slab is partitioned completely;
each wrapped op adds one GPU sync, so the instrumented slab is ~10% longer than the plain one (both
shown). Rooflines: GEMM/attention FLOPs at 10.4 TFLOPS fp16 peak; the rest at 400 GB/s over the
bytes the op reads + writes. Medians of 2 reps.

The machine drifts (thermal): the 0/8K runs were in a fast state (qmm ~7.7 TFLOPS in-model), the
32K/120K runs in a slow one (~6.5). Compare shares within a run, not ms across runs.

## Baseline (before this work): slab 2048, ms per slab (instrumented), share

| op class | off 0 | off 8K | off 32K | off 120K | 120K, slab 546 (old adaptive) | rate @0 / @120K | roofline @0 |
|---|---:|---:|---:|---:|---:|---|---:|
| GEMM mlp gate/up | 1675 (41.5%) | 1735 (38.1%) | 1922 (31.2%) | 2005 (18.7%) | 546 (14.2%) | 7.88 / 6.58 TF | 1269 |
| GEMM mlp down | 854 (21.2%) | 879 (19.3%) | 973 (15.8%) | 1015 (9.5%) | 272 (7.1%) | 7.73 / 6.50 TF | 634 |
| GEMM GDN in (qkv, z) | 639 (15.8%) | 655 (14.4%) | 730 (11.8%) | 767 (7.2%) | 211 (5.5%) | 7.74 / 6.45 TF | 476 |
| GEMM GDN out | 214 (5.3%) | 226 (5.0%) | 245 (4.0%) | 256 (2.4%) | 82 (2.1%) | 7.72 TF | 159 |
| GEMM attn q / o / k,v | 259 (6.5%) | 261 (5.7%) | 298 (4.9%) | 301 (2.8%) | 84 (2.1%) | 6.5-7.6 TF | 185 |
| GEMM GDN b/a (N=32, split-K) | 18 (0.4%) | 21 (0.5%) | 20 (0.3%) | 19 (0.2%) | 14 (0.4%) | 1.4 TF | 2 |
| **all GEMMs** | **3659 (90.7%)** | **3777 (83.0%)** | **4188 (68.0%)** | **4363 (40.7%)** | **1209 (31.4%)** | | **2725** |
| attention SDPA | 47 (1.2%, default) | 393 (8.6%, default) | 1572 (25.5%, fused) | 5833 (54.4%, fused) | 2334 (60.7%, fused) | 5.8 / 5.7 TF (slab 546: 3.8) | 26 / 3198 |
| GDN recurrence (+g/beta) | 144 (3.6%) | 149 (3.3%) | 167 (2.7%) | 168 (1.6%) | 45 (1.2%) | latency-bound | 3 |
| KV cache update (concat + slice update) | 0.3 | 17 (0.4%) | 39 (0.6%) | 138 (1.3%) | 137 (3.6%) | 30 GB/s | 10 |
| rmsnorm (81 calls) | 35 (0.9%) | 37 (0.8%) | 32 (0.5%) | 36 (0.3%) | 25 (0.7%) | 72 GB/s | 6 |
| mlp swiglu (compiled) | 26 (0.6%) | 27 (0.6%) | 26 (0.4%) | 31 (0.3%) | 14 (0.4%) | 185 GB/s | 12 |
| GDN conv1d / gated norm / rope | 33 (0.8%) | 36 (0.8%) | 37 (0.6%) | 38 (0.4%) | 20 (0.5%) | ~100 GB/s | 8 |
| other (residuals, transposes, concat, gating, syncs) | 89 (2.2%) | 119 (2.6%) | 99 (1.6%) | 113 (1.1%) | 60 (1.6%) | | |
| **instrumented slab** | 4033 | 4555 | 6157 | 10720 | 3846 | | |
| **plain slab (no syncs)** | **3622 (565 tok/s)** | **4008 (511)** | **5167 (396)** | **9528 (215)** | **3241 (168)** | | |
| host: graph build / clear_cache per slab | 1.7 / 1.4 ms | 1.9 / 5.5 | 1.8 / 1.8 | 1.6 / 1.7 | 1.4 / 1.2 | | |

Findings:
* **GEMMs are 83-91% of a slab up to 8K** and still 41% at 120K. They run at ~7.7-8.2 TFLOPS
  (75-79% of the 10.4 peak). MLX's fp16 GEMM on the same shapes reaches **9.5 TFLOPS**
  (`gemm_bench.py`): the gap is the quantized kernel's tiling, not the hardware. On non-NAX GPUs
  `quantized_matmul` runs `qmm_t` with 32x32 output tiles (BM = BN = BK = 32, 2x2 simdgroups of
  16x16); the rate is flat in M (512..4096: 8.1-8.2 TF), so the slab size does not matter for it.
* **Attention is the long-context term**: 9% of a slab at 8K, 25% at 32K, 54% at 120K. The fused
  kernel (force_fused, head_dim 256: BQ 32, BK 16, WM 4, WN 2) runs 5.7-6.9 TFLOPS. At 120K the
  old adaptive slab (546 tokens) runs it at only 3.8 TFLOPS: **slab 2048 is x1.28 per token at
  120K** (4.65 vs 5.94 ms/token).
* GDN recurrence 1.6-3.6%; norms + swiglu + conv + rope + other elementwise ~3-4% instrumented,
  ~2-3% plain, most of it already single kernels (mx.fast.rms_norm, compiled swiglu).
* KV cache growth: `KVCache.update_and_fetch` concatenates the whole cache each slab (old + new
  alive together): 138 ms per slab at 120K (1.3%) and +1x the KV at the peak.
* Host overhead between slabs: graph build 1.3-1.9 ms, clear_cache 1-5 ms, per 3-10 s slab: <0.2%.
  `mx.eval` per slab costs nothing measurable (the GPU idles only during the ~2 ms build).

## Levers measured

### 1. GEMM tiling: `pfqmm.py` (kept, default on, `QWEN_PFQMM=1`)

MLX's own `qmm_t` (QuantizedBlockLoader + steel BlockMMA, compiled from the wheel's headers through
`mx.fast.metal_kernel`) instantiated with other tiles. `scratchpad/pfqmm/proto.py`, M=2048, TFLOPS:

| tile BMxBKxBN, WMxWN | gate/up | down | gdn qkv | o/z/out | k/v |
|---|---:|---:|---:|---:|---:|
| quantized_matmul (32x32x32, 2x2) | 8.19 | 8.10 | 8.21 | 8.22 | 8.12 |
| **64x32x32, 2x2** | **9.03** | **8.97** | **9.05** | **9.03** | **8.79** |
| 64x32x64, 2x2 | 8.64 | 8.49 | 8.66 | 8.69 | 8.46 |
| 64x16x64, 2x2 | 7.95 | 7.54 | 7.96 | 7.98 | 7.85 |
| 128x32x32, 2x2 | 8.91 | 8.71 | 8.91 | 8.90 | 8.69 |
| 64x32x32 with N-major raster | 9.10 | 7.93 | 9.08 | 9.03 | 8.83 |
| fp16 GEMM (x @ W), reference | 9.49 | 9.46 | 9.53 | 9.58 | 9.31 |

* Bit-identical to `mx.quantized_matmul` wherever MLX itself runs qmm_t: the per-element K order (BK=32
  steps of 8x8 MMAs) is unchanged, only the output tile per threadgroup. MLX takes split-K when the
  32x32 tile count is <= 256 (e.g. N=4096 at M=64, k/v N=1024 at M<=256, the GDN b/a N=32 always) and
  qmv below 32 rows; `pfqmm._mlx_uses_qmm` mirrors that dispatch (checked against the kernel: equal
  exactly where predicted, `bench_bits.py`).
* By bits (M >= 128): 8-bit x1.07-1.11, 5-bit x1.09-1.14, 4-bit x1.03-1.14, 3-bit x1.04-1.20, 6-bit
  x1.07-1.28. A half-empty last tile loses (M=96: x0.8), so a call takes it only when the 64-row
  padding wastes <= 6% over the 32-row one (and M >= 128).
* install() self-tests bit-equality per (bits, group) on the real weights and stays off otherwise.
* In-process slab A/B (`ab_slab.py`, 4 alternations, 2048-token slab): x1.12 at offset 0, x1.18 at 8K,
  x1.17 at 32K; cache states bit-identical.

### 2. Slab 2048 at every context for fp16 KV (kept)

`_step` keeps `QWEN_PREFILL_STEP` when the KV is fp16, `QWEN_FP16=1` and the fused SDPA is on (probed at
load). Unfused scores stay bounded: below 32K keys the slab already fits the budget; a short slab
(< 512: message cut, last slab) whose scores would pass `PREFILL_BUDGET` now takes the fused kernel too.
Quantized KV (27B/35B) keeps the shrinking slab. Per token at 120K (profile above): slab 2048 4.65
ms vs slab 546 5.94 ms (x1.28; fused SDPA 5.7 vs 3.8 TFLOPS).

### 3. Fused-SDPA threshold re-measured (unchanged: q >= 512, kv >= 32K)

`sdpa_thresh.py`, per call, interleaved, ms default / fused:

| kv | q 2048 | q 1024 | q 512 |
|---:|---|---|---|
| 8K | 36.4 / 36.2 (x1.01) | 18.5 / 19.9 (x0.93) | 9.6 / 10.4 (x0.92) |
| 16K | 72.4 / 76.9 (x0.94) | 36.6 / 40.1 (x0.91) | 18.7 / 20.6 (x0.91) |
| 24K | 110.1 / 115.6 (x0.95) | 54.7 / 60.1 (x0.91) | 27.8 / 31.1 (x0.90) |
| 28K | 129.7 / 136.2 (x0.95) | 65.1 / 69.8 (x0.93) | 32.9 / 36.1 (x0.91) |
| 32K | 173.9 / 155.3 (**x1.12**) | 84.3 / 79.7 (x1.06) | 39.6 / 40.9 (x0.97) |
| 40K | 233.5 / 194.7 (**x1.20**) | 110.5 / 100.2 (x1.10) | 51.7 / 51.4 (x1.01) |

The default path jumps at 32K keys (+34% for +14% keys from 28K): the threshold sits exactly there.

### 4. Rejected

* `QWEN_GDN_FAST` (mx.fast.gated_delta_update): x1.023-1.025 per slab with pfqmm, but teacher-forced on
  8K tokens of real text KL 0.013, top-1 97.0% (it rounds the decay g to fp16; the state drifts over
  long prompts; microtest: state error 1e-3 after 2048 tokens vs 1e-7 for the library kernel).
* Other attention tiles (`proto_attn.py`): mlx 0.32.3 runs BQ32/BK16/WM4/WN2 for head_dim 256 (my
  instantiation is bit-equal to force_fused). WN=1: 5.65 vs 6.93 TF; Q in registers: 4.27 vs 6.59;
  BK=32: 4.56; BQ=64 needs 512 threads (> 448 allowed) or 46 KB of threadgroup memory (> 32 KB).
* mx.compile of elementwise chains: everything outside GEMM/attention/recurrence is ~2-3% of a plain
  slab and mostly single kernels already (mx.fast.rms_norm, compiled swiglu, conv1d): <= ~1% to gain.
* GDN b/a in-projection fused into the qkv/z GEMM for prefill: 18-27 ms/slab (0.5%), and not
  bit-identical (MLX runs b/a with split-K).
* async_eval pipelining across slabs / dropping the per-slab clear_cache: host overhead < 0.2%.
* KV preallocation before the prefill (`_reserve_kv`, grow the fp16 KVCache once instead of a
  concatenation per slab): built, measured, **removed**. Eager (whole prompt at the start): peak +1.1 GB
  at 64K, +0.5 GB at 120K (the full KV sits next to the unfused score transient below 32K), and the
  64K/120K caches came out with other bytes than without it (not investigated further). Lazy (from the
  first fused slab on): bit-identical, but 64K 171.0 vs 165.2 s, 120K 389.5 vs 390.9 s, peak +0.3 GB.
  The profile's 138 ms "kv cache update" per slab at 120K was mostly the instrumentation's sync.
* Chunked unfused attention for long slabs (`chunked_attn.py`: keys in 8K/16K chunks, steel GEMM +
  precise softmax per chunk, log-sum-exp merge in fp32): x0.71-0.76 vs fused at 32K/64K, x0.41 at
  120K, +10-27 GB of fp32 transients. The fused kernel (6.8-6.9 TF in isolation) stays.

## Before / after: full prompt prefill, 9B, in-process, interleaved A/B/B/A

`ab_prefill.py --lens 8192,32768,65536,122880 --variants old,new --rounds 2 --hash` (random ids, the
worker's slab loop: `_bounds`/`_slab`, `lm(..., skip_logits=True)`, eval state, clear_cache).
old = quantized_matmul + slab shrunk by `PREFILL_BUDGET`; new = pfqmm + slab 2048. Peak = MLX peak
(weights ~9.7 GB included). Raw: `results/pf/ab_full.log`, `ab_reserve.log` (peaks of the final code,
without the KV reservation that the first run's "new" still had).

| prompt | old s (tok/s) | new s (tok/s) | speedup | peak old → new | cache |
|---:|---|---|---:|---|---|
| 8K | 17.1 / 18.3 (478 / 448) | 14.1 / 14.7 (581 / 557) | **x1.23** | 11.84-11.88 → 11.89 GB | bit-identical |
| 32K | 83.5 / 94.6 (392 / 346) | 71.3 / 76.6 (460 / 428) | **x1.20** | 14.49-14.61 → 14.60 GB | bit-identical |
| 64K | 215.9 / 213.8 (304 / 307) | 175.3 / 178.0 (374 / 368); final code 165.2 | **x1.21** | 14.60-14.80 → 14.53 GB | KL 1.2e-5, top-1 100% |
| 120K | 500.4 / 494.6 (246 / 249) | 392.6 / 390.0 (313 / 315); final code 390.9 | **x1.27** | 17.69-18.06 → 16.65 GB | (as 64K) |

Exactness: pfqmm is bit-identical everywhere (`kl_tail.py` old vs pfqmm-only at 64K: identical logits).
The slab change is bit-identical up to 48K (checked 40K and 48K: identical caches). At 64K/120K it is
not, and the first differing array is layer 0's GDN recurrent state: the GDN b/a in-projections (N=32)
run MLX's split-K qmm, whose split is 512 / ceil(M/32) tiles: 8 for slabs of 1025-2048 rows but 16 for
slabs of <= 1024 rows, which the old policy reaches past ~64K (and its 494-token last slab at 120K also
took the unfused SDPA). Fused SDPA (q 1000/1024/1500 vs 2048 at 36K-64K keys), conv1d, the GDN kernel
and pfqmm are row-invariant to the slab shape (checked). Cold vs prompt-cache-hit runs stay identical
since both cut slabs with the same `_bounds`.
Teacher-forced on real text (64K prompt, last 2048 tokens scored, `kl_tail.py`): KL(old||new) mean
1.17e-5, max 6.5e-4, top-1 100%.

## After (pfqmm on, slab 2048), same profiler (`--pfqmm 1`, `results/pf/prof_after.json`)

This run was in a slow machine state: in-model GEMMs measured 7.0-7.9 TFLOPS here; the interleaved
A/Bs above are the speed evidence. Plain slab: offset 0 4163 ms, 32K 5193 ms, 120K 8418 ms. Shares
(instrumented): GEMMs 89% / 67% / 38%, attention 1.3% / 27% / 56% (fused 6.0-6.3 TF), GDN recurrence
3.8 / 2.9 / 1.7%, other ~2%. At 120K attention is now the larger half of the slab.

## Verification (final code)

* CPU suite: 84 passed, 16 skipped. GPU: `PFQMM_GPU=1 pytest tests/test_pfqmm.py tests/test_mrqmm.py`
  168 passed, 2 skipped.
* Identity corpus, QWEN_MAX_BATCH=1, vs the new-stack reference (`bump/gpu/new/corpus9.json`): 9B
  15/15 byte-identical. Prompt cache (`pc_ident.py`, 10 cases over a 7.4K system prompt): cold 10/10
  and hit (QWEN_PROMPT_CACHE=1) 10/10 byte-identical to `bump/gpu/new/pc_cold.json`; cold c0 TTFT
  12.6 s (was 18.2 s in the bump run).
* 35B (QWEN_KV_BITS=4, wired 27, slab 512): pfqmm on 351 modules (4/6/8-bit); corpus 15/15
  byte-identical to `corpus35.json`; 20,240-token prompt TTFT 27.9 s, peak 18.63 GB, answer sane.
* 27B (QWEN_KV_BITS=8, wired 27, slab 512): pfqmm on 401 modules (5-bit); corpus 15/15 byte-identical
  to `corpus27.json`; 12,158-token prompt TTFT 91.9 s (bump run: 137.0 s, other session), peak 21.65
  GB, same answer text.
* Raw: `results/pf/verify/`.

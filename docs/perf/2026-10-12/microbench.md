# Microbenchmarks: the next lever after batching, and the library bump

Date 2026-10-11/12. M1 Max 32 GB (applegpu_g13s), macOS 27. The GPU lease was held; the batching worker on :50051 was stopped for these runs (restarted at the end, §6).
Venvs: OLD = `src/worker/.venv` (mlx 0.31.2, mlx-vlm 0.6.2); NEW = `scratchpad/venv_new` (mlx 0.32.3, mlx-lm 0.32.0, mlx-vlm 0.7.6).
Raw outputs: `results/mb/`. Scripts: `design/`.
The machine drifts ±20% between runs, so each comparison below was run back to back, interleaved old/new (A/B/A/B). Absolute ms are given with the ratios measured in the same run.

```zsh
S=<scratchpad>
OLD=/Users/user/Projects/llm/src/worker/.venv/bin/python; NEW=$S/venv_new/bin/python; W=/Users/user/Projects/llm/weights
cd $S/results/mb
```

## 1. Multi-row matmul (the batching ceiling)

### 1a. `quantized_matmul` by rows, 9B shapes (8-bit g64, fp16), whole-step linears

`$OLD -I $S/design/microbench_qmm.py --label old-a --rows 1,2,4,6,8,12,16` and the same with `$NEW ... --vlm-verify`, run as old-a, new-a, old-b, new-b.
Per-matmul median of 15 × a chain of 16 distinct weight copies (more than 256 MB in total, so nothing is served from the SLC). Linears only, summed with layer counts (attention + GDN + MLP + lm_head).

| M | old-a ms (×M1) | new-a ms (×M1) | old-b ms (×M1) | new-b ms (×M1) | new verify-qmv a / b ms (×qmm same M, run a) |
|---:|---:|---:|---:|---:|---:|
| 1 | 32.5 (1.00) | 33.4 (1.00) | 31.3 (1.00) | 32.3 (1.00) | 37.0 / 36.2 (1.11) |
| 2 | 37.7 (1.16) | 38.5 (1.15) | 36.2 (1.15) | 36.8 (1.14) | 40.3 / 39.2 (1.05) |
| 4 | 60.3 (1.86) | 62.0 (1.85) | 59.8 (1.91) | 61.1 (1.89) | 46.2 / 53.0 (0.75) |
| 6 | 84.8 (2.61) | 86.6 (2.59) | 84.3 (2.69) | 87.1 (2.70) | 61.6 / 64.2 (0.71) |
| 8 | 87.8 (2.70) | 88.5 (2.65) | 86.4 (2.76) | 92.2 (2.86) | 72.8 / 76.7 (0.82) |
| 12 | 86.8 (2.67) | 87.6 (2.62) | 86.2 (2.75) | 95.6 (2.96) | 821 / 818 (**×9.4, collapses**) |
| 16 | 87.6 (2.70) | 90.8 (2.72) | 86.5 (2.76) | 92.7 (2.87) | 1103 / 1098 (**×12**) |

`microbench_qmm_rows.py` (fused gate+up, nn.QuantizedLinear, 30 iters; patched so the verify kernel also runs in the new venv and for 8-bit, timed over the same weight copies). Runs a/b/c, interleaved old/new. Command: `MQ_M=1,2,4,6,8,12,16 MQ_ITERS=30 $OLD|$NEW $S/design/microbench_qmm_rows.py $W/ornith15-9b-mlx-8bit $W/qwen38-27b-mlx-5bit`.

| model | M | old a / b / c ms | new a / b / c ms | new verify-qmv a / b / c ms |
|---|---:|---|---|---|
| 9B 8-bit | 1 | 30.5 / 29.7 / 31.0 | 29.7 / 29.4 / 29.9 | 32.0 / 30.9 / 31.2 |
| 9B | 2 | 33.5 / 33.3 / 35.4 | 33.2 / 33.1 / 34.4 | 31.5 / 30.7 / 31.5 |
| 9B | 4 | 57.1 / 56.5 / 59.9 | 56.6 / 56.8 / 59.2 | 37.1 / 35.6 / 37.1 |
| 9B | 6 | 79.2 / 77.5 / 81.4 | 77.7 / 77.5 / 82.1 | 47.8 / 47.0 / 49.8 |
| 9B | 8 | 82.0 / 79.0 / 85.5 | 80.2 / 80.4 / 84.2 | 62.8 / 62.9 / 65.2 |
| 9B | 16 | 83.0 / 79.6 / 95.7 | 79.6 / 79.8 / 86.0 | (not run: spills at M≥12) |
| 27B 5-bit | 1 | 76.7 / 63.4 / 68.6 | 62.4 / 63.0 / 63.6 | 64.8 / 64.8 / 65.8 (old 0.6.2 kernel: 66.3 / 68.2 / 70.1) |
| 27B | 2 | 91.2 / 94.6 / 94.1 | 90.9 / 91.2 / 90.9 | 79.2 / 78.1 / 78.2 (old: 88.7 / 89.0 / 92.8) |
| 27B | 4 | 168 / 165 / 168 | 165 / 167 / 167 | 178 / 180 / 178 |
| 27B | 6–8 | 256–261 | 257–260 | 1240–1654 (collapses at M≥6 for 5-bit, both versions) |
| 27B | 16 | 272 / 261 / 275 | 268 / 262 / 263 | — |

Findings:
* **mlx 0.32.3 does not change the batching ceiling on M1.** The ×M factors are the same in both versions: 1.15 (M=2), 1.9 (M=4), 2.6–2.9 (M≥6, flat). The fp32 dequant in qmm (mlx#4241) is invisible at decode M or costs up to +5% at M≥8 (new-b: 92 vs 86 ms).
* **27B (5-bit): M=1 in the new mlx is 62–64 ms against 63–77 ms in the old.** The old runs are noisier, and the medians differ by ~5–10% in favour of new. Batching on the 27B is weaker than on the 9B: ×1.46 at M=2 and ×4.3 at M≥6.
* **mlx-vlm 0.7.6's verify-qmv kernel supports 8-bit** (`quantized_verifier.optimized_affine_linear`, bits 4/5/8, K % 512 == 0). It is a real multi-row qmv: each threadgroup reads a weight tile once for all T rows.
  * On the 9B whole step: ×0.95 qmm at M=2, **×0.63–0.66 at M=4**, **×0.61 at M=6**, ×0.78 at M=8. At M ≥ 12 it is 10× slower: `x_thread[T][8]` plus `result[T][4]` in registers spills.
  * For 5-bit (27B) it helps only at M=2 (×0.86) and collapses from M=6.
  * The 0.6.2 copy (`qwen3_5/language.py::_target_verify_quantized_linear`) is 4/5-bit only and returns None for 8-bit.
* **dequantize → fp16 GEMM** (`y @ mx.dequantize(w).T`): ×2.2–5.5 of qmm at every M. It writes and re-reads a 2-byte copy of every weight per step. Dead end.
* `mx.gather_qmm` was not timed separately. On dense weights it dispatches to the same qmv/qmm kernel family, and versions.md §2c lists no M1 small-M change.

### 1b. Prototype: custom 8-bit multi-row kernel (`design/microbench_mrqmv.py`)

No existing kernel reaches M=8–16 cheaply, so I wrote `mx.fast.metal_kernel` prototypes, all correctness-checked: max |diff| ≤ 0.002 against `mx.quantized_matmul` and ≤ 0.0012 against an fp32 dequant→matmul reference at M=16, which is fp16 output rounding.

1. `dev_r4_s2` / `tg_*`: scalar multi-row qmv. Dequantize R×8 weights once into registers, then loop over the M rows of x (from device memory, or staged in threadgroup memory). It ties qmv at M=1–2 and gives ×0.80 at M=4, but it becomes **FMA-bound from M=8** (~5 ms per extra row on the 9B, ~3.6 TFLOPS fp32). Same wall as the verify kernel.
2. `mma_s*`: simdgroup-MMA, weights dequantized to threadgroup memory per 32×64 tile (MLX's qmm loader scheme with a 16-row instead of 32-row tile). Weight loading is slow (~150 GB/s): ×1.1 at M=8, ×0.6 at M=16 on the big shapes, loses on small ones.
3. **`rmma_a{NA}_s{NSG}` (the one that works)**: simdgroup-MMA with **no threadgroup memory for weights**. It computes Cᵀ = W·xᵀ. The A fragments (W) are built in registers through `thread_elements()` from one 16-byte `uint4` load per thread per 8 rows × 64 k: the k order within each 64-group is permuted so every thread's bytes are contiguous, and x is staged in threadgroup memory with the same permutation. B is xᵀ, fp32 accumulation, M padded to 8 or 16.

Full 9B shape set, new venv, run new-a (`$NEW -I $S/design/microbench_mrqmv.py --label new-a --rows 1,2,4,8,12,16 --variants rmma_a1_s8,rmma_a2_s4,rmma_a2_s8,dev_r4_s2 --verify --dequant`). Linears ms/step, with ×qmm at the same M:

| M | quantized_matmul | vlm verify (0.7.6) | dev_r4_s2 (scalar) | rmma_a1_s8 | rmma_a2_s4 | best-of per shape |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.2 | 35.9 (1.15) | 32.8 (1.05) | 44.0 (1.41) | 48.5 (1.55) | 31.2 (1.00) |
| 2 | 35.9 | 37.7 (1.05) | 33.9 (0.94) | 45.3 (1.26) | 49.2 (1.37) | 33.5 (0.93) |
| 4 | 59.7 | 44.8 (0.75) | 47.7 (0.80) | 45.6 (0.76) | 49.2 (0.82) | 42.4 (0.71) |
| 8 | 87.1 | 73.5 (0.84) | 86.2 (0.99) | **44.2 (0.51)** | 47.8 (0.55) | **43.6 (0.50)** |
| 12 | 86.7 | — | 163 (1.88) | 71.8 (0.83) | 72.5 (0.84) | 68.3 (0.79) |
| 16 | 90.1 | — | 215 (2.39) | 70.5 (0.78) | 71.4 (0.79) | **66.7 (0.74)** |

Per shape at M=8 (µs, qmm → rmma_a1_s8): gate+up fused 4096×24576 959 → 437 (×0.46); down 527 → 296; q_proj 368 → 181; o_proj 249 → 127; lm_head 8784 → 4082. k/v_proj (N=1024) stay on qmm (too few rows to fill the GPU).

A/B in the same session (old-b then new-b, `--rows 1,4,8,16 --variants rmma_a1_s8,rmma_a2_s4`). The machine was drifting: qmm M=8 was 93.5 in the old run and 112.8 in the new run.

| M | old-b qmm | old-b rmma best | ratio | new-b qmm | new-b rmma best | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 62.5 | 50.2 | 0.80 | 81.2 | 59.8 | 0.74 |
| 8 | 93.5 | 49.4 | **0.53** | 112.8 | 57.9 | **0.51** |
| 16 | 91.0 | 75.2 | 0.83 | 119.7 | 90.9 | 0.76 |

The kernel works the same on mlx 0.31.2 and 0.32.3 (`mx.fast.metal_kernel`), so **it does not depend on the bump**.

**Whole-step estimate.** Measured worker/microbench steps are 127 ms at B=8 and 133 ms at B=16, with ~87–90 ms of qmm linears. Swapping the linears for the prototype gives:

| B | step today | linears today → rmma | est. step | × step | aggregate tok/s today → est. |
|---:|---:|---:|---:|---:|---:|
| 4 | 90.5 | 59.7 → 42.4 | ~73 | 1.24 | 44 → 55 |
| 8 | 127 | 87 → 44 | **~84** | **1.51** | 63 → **95** |
| 16 | 133 | 90 → 67 | ~110 | 1.21 | 120 → **145** |

**Decision: yes, a custom multi-row kernel is worth building.** It clears the ≥1.4× bar at B=8 (×1.5 whole step), with ×1.2 at B=16 and ×1.24 at B=4.

Headroom left in the prototype:
* Its M=1-equivalent cost (44 ms against qmv's 31) shows that weight streaming still runs at ~220 GB/s against qmv's ~300. Loads are not software-pipelined, and there is one barrier per 256 k.
* At M=16 the MMA work (~26 ms per 8 rows, ≈5.5 TFLOPS) is the floor. So B=16 can't go much below ~55 ms of linears on M1. B=8 could reach ~30–35 ms with a better loader, which would make the step ~70 ms (×1.8).
* Exactness: rows ≥ 2 then differ from solo in fp16 rounding (max |Δ| 0.002), the same class as today's qmm_splitk at B ≥ 6. B=2/4 are bit-identical to solo today and would stop being so. Keep qmv for M ≤ 2 (it is the fastest there anyway).

## 2. SDPA head_dim 256 (prefill slabs): default vs `force_fused`

`$OLD|$NEW -I $S/design/microbench_sdpa.py --label <l> --decode` (run a), then `--modes default[,fused] --kv-lens 8192,32768,65536,122880` (run b). Order: old-a, new-a, old-b, new-b. 9B layout: Hq 16, Hkv 4, D 256, fp16, causal, median of 5, ms per layer call.

| q_len × kv_len | old default a / b | new default a / b | new **fused** a / b | fused / default (same run) |
|---|---:|---:|---:|---:|
| 512 × 8K | 9.62 / 9.64 | 9.61 / 12.09 | 10.48 / 13.91 | 1.09 / 1.15 (slower) |
| 2048 × 8K | 36.5 / 36.6 | 36.5 / 48.8 | 36.5 / 56.0 | 1.00 / 1.15 |
| 512 × 32K | 39.0 / 39.4 | 40.8 / 51.9 | 41.1 / 56.7 | 1.01 / 1.09 |
| 2048 × 32K | 180.8 / 176.7 | 176.3 / 193.5 | 155.8 / 173.1 | **0.88 / 0.89** |
| 512 × 64K | — / 126.4 | — / 127.0 | — / 85.8 | **0.68** |
| 2048 × 64K | skipped (17 GB of scores) | skipped | — / 390 | (only fused can run it) |
| 512 × 120K | 234.0 / 246.6 | 234.4 / 244.2 | 156.6 / 174.7 | **0.67 / 0.72** |
| 2048 × 120K | skipped (32 GB of scores) | skipped | 614 / 736 | (only fused) |

* Decode / verify (q_len 1–16) and batched decode B = 2–8: old and new default are equal within noise (the vector kernel is unchanged for D256).
* `force_fused` at q_len 16 is **slower** (8K: 2.25 vs 1.67 ms; 120K: 30.1 vs 22.5 ms). Never force it below q_len ~512.
* maxdiff fused vs unfused ≤ 2.4e-4 (fp16).
* The new run b drifted ~25% slow; ratios are within-run.

**Real model, chunked prefill** (`design/microbench_prefill_fused.py`, new script).
* It loads the 9B like the worker and prefills random ids with the worker's slab loop.
* `default` = worker policy (slab 2048, shrinking by `QWEN_PREFILL_BUDGET` = 2048×32K / offset).
* `fused` = slab 2048 at every offset, with `force_fused=True` (monkeypatched `mx.fast.scaled_dot_product_attention`) for causal slabs with kv ≥ 16K.
* `fusedadapt` = worker slabs + force_fused.

```
cd src/worker; QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 \
  <venv>/bin/python $S/design/microbench_prefill_fused.py 32768 default,fused,fusedadapt   # interleaved old,new,old,new
```

| prompt | old 0.31.2 default | new 0.32.3 default | new fused | new fusedadapt |
|---|---|---|---|---|
| 32K, round 1 | 91.2 s, peak 16.58 GB | 116.0 s, peak 14.48 GB | 131.3 s, peak 12.65 GB | 127.0 s |
| 32K, round 2 | 126.7 s | 130.6 s | 129.3 s | 128.6 s |
| 64K, round 1 | — | 285.6 s, peak 15.45 GB | 283.1 s, peak 13.93 GB | — |
| 64K, round 2 | — | 282.5 s | **258.7 s** | — |

The machine slowed down during these runs: old default went from 91 s to 127 s between rounds, and CLAUDE.md's figure is 82 s. Only the within-round comparisons mean anything.

Findings:
* **No measurable prefill gain from the new mlx at ≤ 32K** (new default ≈ old default within drift).
* **`force_fused` gains 0–8% at 64K.** It should approach ~10–15% at 120K: the microbench gives −28…−33% per attention call at 512 × 120K, and attention is ~half of a 120K prefill. This is far below CLAUDE.md's hoped-for "~−20% and slab 2048 at any context".
* The real win is memory:
  * `fused` peak is **−1.5 to −1.8 GB** against new default, and −3.9 GB against old (at 32K).
  * 2048-token slabs run at 64K/120K with no score transient, so `QWEN_PREFILL_BUDGET` becomes unnecessary on fp16-KV models.
  * New default also peaks ~2 GB below old at 32K (16.58 → 14.48 GB). 0.7.6 no longer allocates the random vision tower (versions.md 3.1 item 5) and qmm transients differ.
* Patch to adopt: `force_fused=True` only for fp16 KV, causal, q_len ≥ 512 and kv_len ≥ 32K. Below that it is 0–15% slower.

## 3. GDN recurrence kernels

`$OLD|$NEW -I $S/design/microbench_gdn.py --label <l> --cases 1x1,4x1,8x1,16x1,1x512,1x2048`, run old-a, new-a, old-b, new-b, then `MLX_GDN_PACKED=0` on new. 9B: Hk 16, Hv 32, D 128. **ms for all 24 GDN layers** per call (B×T):

| B×T | old vlm a / b | new vlm a / b | new mlx-lm packed a / b | new `mx.fast.gated_delta_update` a / b |
|---|---:|---:|---:|---:|
| 1×1 (decode) | 7.0 / 7.9 | 6.8 / 8.8 | 7.5 / 7.8 | 7.0 / 7.0 |
| 4×1 | 8.8 / 7.9 | 7.3 / 8.0 | 8.1 / 8.2 | 7.9 / 9.3 |
| 8×1 | 8.3 / 8.0 | 8.5 / 8.9 | 8.4 / 9.5 | 8.5 / 8.9 |
| 16×1 | 11.1 / 10.2 | 10.2 / 10.9 | 10.8 / 12.6 | 10.7 / 11.2 |
| 1×512 (prefill slab) | 40.5 / 39.0 | 40.0 / 41.1 | 22.7 / 24.0 | 25.3 / 20.8 |
| 1×2048 | 142.5 / 142.2 | 141.1 / 149.0 | 72.6 / 74.4 | **63.6 / 66.5** |

* Max |dy| for fast against vlm: 3e-5; dstate 2e-5–8e-5. Packed lm is bit-identical to vlm.
* mlx-lm 0.32.0 with `MLX_GDN_PACKED=0` falls back to the old speed (152.6 ms at 2048), which confirms the packed kernel is the source of its gain.

Findings:
* **Decode (T=1): nothing to gain** at any B (all kernels within noise, 7–11 ms per step for 24 layers). Batched GDN decode costs only ~+4 ms from B=1 to B=16.
* **Prefill: ×2.2** for the recurrence (142 → 64 ms per 2048-token slab with `mx.fast.gated_delta_update`, or 73 ms with mlx-lm's packed kernel).
  * The recurrence is only ~3.5% of a 9B 2048-slab (~4 s at 500 tok/s), so the prefill gain is **~2%**. CLAUDE.md's "GDN ~22%" counts the GDN projections as well.
  * Cheap to adopt after the bump: swap the kernel call in `Qwen3_5GatedDeltaNet` for T ≥ 16. It is not a lever on its own.

## 4. Prefill/decode overlap on two MLX GPU streams (`design/microbench_streams.py`, new)

The 9B is loaded like the worker. Decode = the design's batched step (`rows` mode: per-row KV + stacked GDN), **B=4 at ctx 1K**, pipelined like `Engine._run`. Prefill = the worker's slab loop over a fresh **4096-token** prompt into its own cache.
* **P**: prefill alone on stream A.
* **D**: K decode steps alone on stream B, with K chosen so that K × step ≈ P.
* **PD**: both, from one scheduler thread. All slabs are `async_eval`'d on stream A, then the K decode steps run on stream B; a waiter thread records when the last slab finished.
* **SER**: today's single-stream interleave (slab, step, slab, ...).

`batching_proto.py` needed a small compatibility patch for mlx-vlm 0.7.6: `target_verify` is gone and `_prepare_projected_qkv` is used instead. The original is at `design/batching_proto.py.orig076`.

```
cd src/worker; QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 PF_REPS=3 PF_SLABS=512,2048 PF_LEN=4096 \
  <venv>/bin/python $S/design/microbench_streams.py        # old venv, then new venv
```

| venv | slab | P alone s | K × decode alone s (ms/step) | sum | max | **PD wall** | overlap* | decode steps finished while the prefill ran | prefill in PD |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| old 0.31.2 | 512 | 11.88 | 8.99 (71.4) | 20.87 | 11.88 | **20.64** | 0.03 | **0 of 126**; first token at 10.9–12.9 s | ×0.96 |
| old 0.31.2 | 2048 | 14.26 | 13.95 (93.0) | 28.21 | 14.26 | **28.60** | −0.03 | **0 of 150** | ×0.99 |
| new 0.32.3 | 512 | 16.22 | 15.21 (102.1) | 31.43 | 16.22 | **32.52** | −0.07 | **0 of 149** | ×1.06 |
| new 0.32.3 | 2048 | 14.83 | 14.03 (99.5) | 28.85 | 14.83 | **29.10** | −0.02 | **1 of 141** | ×1.03 |

\* overlap = (sum − wall) / min(P, D): 0 means serialized, 1 means perfect overlap. Medians of 3 reps. The machine ran ~40% slower than in the morning session (thermal / desktop load), so compare within rows.

Reference, single stream: SER took 12.7 s (old, slab 512) for the prefill plus 8 interleaved decode steps of 151 ms each. That is the worker's current behaviour.

**Finding: Metal does not overlap them. Queues are drained in submission order.**
* Once the prefill slabs are committed on stream A, the decode command buffers on stream B wait behind the whole prefill. Not one decode step completes in the meantime: the first token lands right after the last slab.
* Wall time = P + D in both MLX versions.
* The prefill is compute-bound and fills every GPU core, so there is nothing to co-schedule; the decode's bandwidth-bound kernels do not slip in between.
* Two streams therefore buy nothing. They are worse than today's interleave, because decode is starved for the whole prefill instead of one slab at a time.
* **The V2 "piggyback" lever has to be done inside one forward pass:** append the B decode rows to the prefill slab, the way vLLM's chunked prefill does. At slab 512 the slab's GEMMs are compute-bound, so 8 extra rows cost ~+1.5% of the slab while saving a whole ~90–130 ms batched decode step per slab. That needs per-row attention for the decode rows next to the slab's causal attention (the RowsKV path already does that) and a GDN state per row. It is not a stream trick.

## 5. Library batched attention vs per-row attention (new venv, mlx-vlm 0.7.6)

`cd src/worker; QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 MB_CTX=1024 MB_BATCHES=1,8,16 MB_MODES=lib1,rows,lib $NEW $S/design/microbench_decode_batch.py` was run twice (a, b). It was then run once more with `MB_CTX=8192 MB_BATCHES=8,16 MB_MEM_GB=8 MB_MODES=rows,lib`. Rows are equal-length clones.

| ctx | B | rows ms/step a / b | lib (BatchKVCache) ms/step a / b | lib / rows |
|---:|---:|---:|---:|---:|
| 1K | 1 (lib1) | 53.5 (cold) / 27.5 | — | — |
| 1K | 8 | 127.2 / 128.7 | 124.7 / 123.7 | 0.98 / 0.96 |
| 1K | 16 | 127.2 / 118.1 | 124.6 / 118.2 | 0.98 / 1.00 |
| 8K | 8 | 122.1 | 121.9 | 1.00 |
| 8K | 16 | 140.2 | 139.7 | 1.00 |

Findings:
* **The library's batched attention gives nothing** over the worker's per-row attention: within 0–4% at 1K and equal at 8K. There is no reason to adopt `BatchKVCache` or the V2 attention layout. The step time is linears (§1), not attention.
* B=16 costs the same per step as B=8 here too: ~125 vs ~125 ms at 1K, 140 vs 122 ms at 8K. So the 9B aggregate at B=16 is 114–135 tok/s.
* Peak memory in the new venv is ~2–3 GB lower (9.7 GB at B=1, against ~12.9 GB for the 0.6.2 worker). 0.7.6 drops the weightless vision tower of the text-only checkpoint.

## 6. Recommendation

### (a) Library bump: yes, but as a memory/maintenance change, not a speed one, and only together with the patches below

What it buys on M1, measured:

| item | effect |
|---|---|
| decode qmv/qmm by rows | unchanged (§1a); 27B M=1 maybe −5…10% (noisy) |
| prefill ≤ 32K | unchanged within drift (§2) |
| prefill 64K with `force_fused` (patched in) | −1…−8%; expect ~−10…15% at 120K |
| peak memory | −2 GB at load (no random vision tower on Ornith-1.5); fused prefill a further −1.5…−1.8 GB, and no `PREFILL_BUDGET` slab shrink needed |
| GDN `mx.fast.gated_delta_update` | ×2.2 on the recurrence, ≈ −2% prefill |
| 8-bit multi-row verify-qmv kernel | ×0.61–0.66 linears at M = 4–6 (batched decode B=4–6, DFlash verify blocks ≤ 8) |
| `rows` vs library batch attention | no gain (§5) |

Patches that must accompany the bump. Items 1–5 are versions.md §3.1 / §8; 6–8 are new from this run.
1. `_render`: `if "enable_thinking" in self._template_vars: kwargs.setdefault("enable_thinking", True)`. 0.7 defaults it to False.
2. `ThinkingBudgetCriteria(..., prompt_preopens_thinking=True)`. Replace `apply_forced_token` with `pop_forced_token_id()` (worker.py:2062, the only call site).
3. `_run`: don't forward `position_ids`/`rope_deltas` from `get_input_embeddings` for text-only requests, or prompt-cache hits get wrong RoPE positions.
4. `_run_spec` (DFlash), worker.py:1790–1855:
   * Drop `gdn_sink=None`.
   * Verify with `speculative_verify=True`.
   * `commit_speculative_round(...)` every round, `abort_speculative_round` in an except.
5. (optional) Drop the `load_weights(strict=False)` monkeypatch.
6. **New: `_rows_attention` (worker.py:1133–1150)** forwards `target_verify=` to `Qwen3_5Attention.__call__`. 0.7.x removed that kwarg, so **every prefill/solo attention call would raise TypeError**. Pass it only when True. The batched branch should use `self._prepare_projected_qkv(q, k, v, None, position_ids, position_embeddings, None)` (see the patched `design/batching_proto.py`). `self.rotary_emb.apply_rotary` still exists, but the projected-qkv helper is the 0.7 shape.
7. **New: `force_fused=True`** in the fp16-KV prefill attention call only when causal, q_len ≥ 512 and kv_len ≥ 32K. Then raise or disable `QWEN_PREFILL_BUDGET` for fp16-KV models. Below that threshold fused is 0–15% slower, and at q_len ≤ 16 it is 30–50% slower, so never use it for decode or verify.
8. (optional) Route GDN prefill slabs (T ≥ 16) to `mx.fast.gated_delta_update`. It casts g and β to fp16: check teacher-forced KL as for `QWEN_FP16`.

Plus `uv add mlx==0.32.3 mlx-vlm==0.7.6 llguidance>=1.9.1`, `uv remove mlx-lm`, and regenerate the gencode (versions.md §8). Re-run the B=1 identity corpus and the batching agreement test afterwards: mlx#4241 changes qmm rounding for M ≥ 6. Re-check when mlx 0.32.4 ships (#4629: M=6 stays on qmv on M1 Max).

### (b) Next levers, in order of expected value

1. **Custom 8-bit multi-row matmul kernel for batched decode (B = 3…16)**: the biggest lever.
   * The prototype `rmma` (`design/microbench_mrqmv.py`) is a register-built-A simdgroup-MMA with a permuted k order. It runs the 9B linears at **×0.50 qmm at M=8** (44 vs 87 ms) and ×0.74 at M=16, with max |Δ| 0.002.
   * Whole-step estimate: **B=8 127 → ~84 ms (×1.5, 63 → 95 tok/s)**, B=16 133 → ~110 ms (×1.2, 120 → 145 tok/s), B=4 ×1.24.
   * Works on mlx 0.31.2 as well (no bump needed).
   * Production work:
     * pipeline the weight loads (the prototype streams at ~220 GB/s against qmv's ~300; B=8 linears could reach ~30–35 ms, step ~70 ms, ×1.8);
     * add a 5-bit variant for the 27B;
     * hook it into `QuantizedLinear.__call__` for 3 ≤ rows ≤ 16, bits 8. Keep qmv for M ≤ 2 (fastest there, and keeps B=2 bit-identical to solo), and k/v_proj (N=1024) on qmm.
   * Option: below M=8 the mlx-vlm 0.7.6 verify kernel already gives ×0.61–0.66 at M=4–6.
2. **Piggyback decode rows onto prefill slabs in the same forward** (not two streams; §4 shows Metal serializes streams). This removes the "waves" cap seen in the c=2…8 gateway benchmarks: decode stalls ~1.2–1.7 s per 600-token prefill today. The extra rows are ~free in a compute-bound slab GEMM.
3. Bump-related, small: `force_fused` at ≥ 32K (≤ −10…15% at 120K, −1.5 GB), GDN fast kernel (−2% prefill).
4. Not worth it:
   * dequant→fp16 GEMM (×2–5 slower);
   * two-stream overlap;
   * library `BatchKVCache` attention;
   * any GDN decode kernel.

### (c) `QWEN_MAX_BATCH` defaults

* **9B: 16.** B=16 costs the same per step as B=8 in both MLX versions: 125 vs 125 ms at 1K and 140 vs 122 ms at 8K, giving 114–135 tok/s against ~63. This confirms batching_gpu.md's recommendation; nothing here changes it. With the custom kernel, B=8 gets relatively cheaper (84 vs 110 ms), but 16 still has the higher aggregate.
* **27B (5-bit):** linears cost ×1.46 (M=2), ×2.65 (M=4), ×4.1–4.3 (M≥6). The aggregate is ×1.37 at B=2, ×1.51 at B=4, ×1.46 at B=6, ×1.92 at B=8, ×3.7 at B=16. **B=3–6 buys almost nothing over B=2**. Use 8+ if memory admission allows it (q8 KV, `QWEN_BATCH_MEM_GB`), else 2.
* 35B: unchanged (batching_gpu.md: pays only at B ≥ 8–16 with distinct prompts).

## State left

* The worker was restarted exactly as before: 9B, `QWEN_KV_BITS=0 QWEN_MAX_BATCH=8`, nice 0, log `results/worker_batching.log`.
* `curl localhost:8090/v1/models` answers `ornith15-9b-mlx-8bit`. :8080 and :8081 were not touched.
* Scripts changed in `design/`:
  * `microbench_qmm_rows.py`: verify kernel for 8-bit and mlx-vlm 0.7, timed over weight copies, plus a whole-forward verify sum. The original is `.orig`.
  * `batching_proto.py`: 0.7.6 compatibility. The original is `.orig076`.
* New scripts in `design/`: `microbench_mrqmv.py`, `microbench_prefill_fused.py`, `microbench_streams.py`.

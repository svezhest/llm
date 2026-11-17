# Prefill

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- Prefill is compute-bound: 8-bit GEMMs dominate short context, attention passes them past
  ~64K keys. The next lever is long-context attention (BACKLOG).
- The fp16 cast (`QWEN_FP16=1`) is the biggest prefill win on M1/M2, which have no native bf16.
- Policy now: 2048-token slabs; fp16 KV keeps 2048 at any context through the fused SDPA
  from 32K keys; pfqmm for the GEMMs (bit-identical). The policy is frozen at load.
- A long slab runs uninterrupted, so a >64K prefill stretches every decode row's gap.
- The dead ends below were measured: do not retry them without new evidence.

## Performance notes

- **M1/M2 have no native bf16.** All our MLX quants store scales/norms/embeddings
  in bf16, and an 8-bit `quantized_matmul` with ≥512 rows runs 4.5 TFLOPS in bf16
  vs 7.8 in fp16. `QWEN_FP16=1` (default) casts the language model to fp16 after
  load: prefill 8K 26.5→15.5 s, 32K 118→82 s (with the 2048 slab); decode
  unchanged (bandwidth-bound, ~28 ms/token ≈ 9.5 GB / 400 GB/s). Verified
  teacher-forced on 3.4K tokens: KL(bf16‖fp16)=0.0003, top-1 agreement 99.4%
  (9B); 0.0004 / 99.2% on the 27B; 0.0019 / 98.8% on the 35B (3-bit MoE
  amplifies the noise — still under a Q6 quant's 0.004; set `QWEN_FP16=0` if
  fidelity beats speed there). 35B prefill 8K: 16.6 → 11.5 s, decode +4%.
- **Prefill slab**: 512→2048 is +15% at 8K (only matters once fp16 makes the
  GEMMs the bottleneck); 4096 gains nothing (qmm rate is flat in M). `QWEN_CLEAR_CACHE`
  0/1 is ±2%. Quantized KV (27B/35B) shrinks the slab with context (`PREFILL_BUDGET`:
  the quantized SDPA materializes slab × keys × heads scores). fp16 KV + fused SDPA
  (the 9B with `QWEN_KV_BITS=0`) keeps 2048 at any context since 2026-10-12 (`_step`):
  a slab reaching 32K keys takes the fused kernel, which holds no scores (at 120K the
  old 546-token slab ran it at 3.8 vs 5.7 TFLOPS); below 32K keys the budget rule
  still applies (at the defaults it never shrinks there). The policy is frozen at
  load (`_FIXED_SLAB`), so `_bounds` cuts stay the same for the process's life; a
  fused-SDPA failure at run time fails a slab that would pass the budget unfused
  instead of materializing its scores. **Trade-off under batching**: one slab per
  scheduler tick, uninterrupted — a 2048 slab at 120K is ~8.4–9.5 s (≈5 s at 32K)
  vs ~3.2 s for the old 546 slab, so while a >64K prompt prefills every running
  decode row's inter-token gap is ~3× longer. Shrinking the slab only under load
  would change slab shapes and with them the bits (`_bounds` determinism); not done.
- **Prefill profile (9B, 2026-10-12, `docs/perf/2026-10-12/prefill_profile.md`)**: op
  classes of a 2048-token slab (each op synced and timed in-process), share of the
  slab at cache offset 0 / 8K / 32K / 120K: 8-bit GEMMs 91 / 83 / 68 / 41% (MLP
  gate/up alone 42% at 0), attention 1 / 9 / 26 / 54% (fused 5.7–6.9 TFLOPS from
  32K), GDN recurrence 3.6 / 3.3 / 2.7 / 1.6%, norms + swiglu + conv + rope + other
  elementwise ~3–4%, KV concat ≤1.3%, host (graph build + clear_cache) <0.2%. GEMMs
  ran ~8.1 TFLOPS against MLX's own fp16 GEMM at 9.5 on the same shapes: the
  quantized kernel's 32×32 tiles on non-NAX GPUs, fixed by `pfqmm.py`
  (`QWEN_PFQMM=1`: MLX's qmm_t code from the wheel's headers via `metal_kernel`,
  64-row tiles → 9.0 TFLOPS, ×1.10–1.14 per 8-bit matmul, ×1.03–1.28 over 3..8 bits,
  **bit-identical**; taken only where MLX runs qmm_t, ≥ 64 rows (the last tile is MLX's own 32-row one when ≤ 32 rows are left: 2026-10-17, ×1.06–1.19 GEMM time on short slabs with M % 64 ∈ (0, 32], `docs/perf/2026-10-17/pfqmm_short_slabs.md`), (1, M, K) inputs, g64;
  self-tested at load per (bits, group) on the first module and row count where MLX
  surely runs qmm_t — a key with no such module is logged "untested" and stays off). Full prefill,
  in-process interleaved A/B, old → new (pfqmm + fixed fp16 slab): 8K 17.7 → 14.4 s
  (×1.23), 32K 89 → 74 s (×1.20), 64K 215 → 177 s (×1.22), 120K 498 → 391 s
  (×1.27); peak at 120K 17.7–18.1 → 16.65 GB. Caches bit-identical up to 48K; at
  64K KL 1.2e-5, top-1 100% (teacher-forced tail; the GDN b/a projections, N=32, run
  MLX split-K with split 16 for slabs ≤ 1024 rows vs 8 above, and the old policy
  reached 1024-row slabs past 64K). Next lever: attention at long
  context (the fused head_dim-256 kernel at ~6 TF is 54% of a 120K slab; its tile
  variants and a chunked unfused attention were slower — see BACKLOG).

- **Logits on prefill slabs are never computed** (MLX is lazy; `skip_logits=True`
  is passed anyway to keep the graph small) — don't expect a win there.

## Dead ends

- **Dead ends (measured 2026-09-18, don't retry blindly)**: (1) dequantize→fp16
  GEMM instead of 8-bit qmm on prefill slabs: +12% on up/gate, −3% on down, ~5%
  of MLP — the GPU is at ~8 TFLOPS either way. (2) Fused SDPA for head_dim 256:
  mlx 0.32.3 has the kernel but its default dispatch still takes the unfused graph
  (materializes slab × keys × heads scores — the real reason for `PREFILL_BUDGET`).
  `QWEN_SDPA_FUSED=1` forces it for fp16-KV prefill slabs ≥ 512 over ≥ 32K keys.
  Per call (9B layout, time fused / default; final interleaved re-measure
  2026-10-12): up to 28K keys the default path is faster (fused ×1.05–1.12);
  from 32K fused wins: ×0.89 at 2048 × 32K (×0.95 at 1024, ×1.03 at 512), ×0.83
  at 2048 × 40K, ×0.59–0.68 at 64K, ×0.67–0.72 at 512 × 120K (an earlier single
  probe said ×1.06 at 2048 × 32K; not reproduced). The default path jumps at 32K
  keys (+34% for +14% keys), so the gate stays there. End-to-end much less than hoped: 64K prefill −3…5%
  in-process (203/215 vs 219/222 s), peak −1.3 GB; ≤ 32K unchanged. KL fused vs
  unfused at 64K 7e-6, 100% top-1. Slab 2048 at every offset for fp16 KV: done
  2026-10-12 (see Prefill profile; ×1.27 at 120K). Below 32K keys the slab still
  obeys `QWEN_PREFILL_BUDGET`, and the batch prefill reserve is sized for
  max(`QWEN_PREFILL_BUDGET`, `QWEN_PREFILL_STEP` × 32K) scores (= the old reserve at
  the defaults).
  (2b) `QWEN_GDN_FAST=1` (`mx.fast.gated_delta_update` for ≥ 16-token slabs): KL
  vs off 6.3e-5 / 99.85% (9B, 3.4K tokens), 1.1e-5 (27B), 9e-3 (35B); 8K prefill
  ×0.994. Re-measured with pfqmm: ×1.024 per slab, but 8K real tokens KL 0.013 /
  97.0% top-1 (fp16 decay drifts) — stays off.
  (2c) Prefill (2026-10-12): mx.compile of the elementwise chains (≤ ~1% to gain),
  KV preallocation instead of the per-slab concat (no gain, peak +0.3 GB), chunked
  unfused attention for long slabs (×0.4–0.76, huge fp32 transients), other fused-
  attention tiles (WN=1 / Q in registers / BK 32 all slower; BQ 64 does not fit).
  (3) 8-bit qmm cost by rows on M1: M=2 ×1.17, M=4 ×1.97, M=6…32 flat ×2.6

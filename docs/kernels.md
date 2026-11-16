# Metal kernels

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- Own kernels (`mx.fast.metal_kernel`): mrqmm (multi-row decode / verify matmul, 4 / 5 / 6 / 8
  bits), mrqmv (exact multi-row qmv for 2..3-row verify blocks), pfqmm (prefill matmul,
  [prefill.md](prefill.md)), the fused GDN in-projection, vatt (one-pass verify / decode
  attention, fp16 and q8 KV).
- Defaults: 8-bit mrqmm, pfqmm, GDN fuse, vatt verify on; 4 / 5 / 6-bit mrqmm are opt-in and
  should be set for the 27B builds. Exact switches and defaults: [knobs.md](knobs.md).
- pfqmm, mrqmv, vatt and 4 / 6-bit mrqmm self-test at load and fall back to MLX (8 / 5-bit
  mrqmm are gated by the GPU tests). pfqmm and the B ≤ 2 paths are bit-identical; the rest
  are in the batched-decode numerics class.
- All need fp16 scales: `QWEN_FP16=0` turns them off.

## Multi-row matmul (mrqmm)

- **Multi-row 8-bit matmul (`mrqmm.py`, `QWEN_MRQMM=1`, measured 2026-10-12)**:
  `mx.fast.metal_kernel`, simdgroup MMA computing `Cᵀ = W·xᵀ` for ≤16 rows,
  A fragments built in registers from one 16-byte weight load per 8 rows × 64 k
  (k permuted within each 64-group so x is one 16-byte load per row too), fp16
  dequant (0x6400 trick), fp32 accumulate, no threadgroup memory in the K loop.
  9B linears per step (all 8-bit shapes incl. lm_head): qmm 47 / 59 / 85 / 85 ms
  → 32 / 32 / 33 / 46 ms at 3 / 4 / 8 / 16 rows (a 1-row qmv step is 30): weight
  streaming stays at qmv speed to 8 rows, 16 rows are MMA-bound (~5.8 TFLOPS).
  Whole decode step (same process A/B, ctx 1K): 56 / 86 / 93 → 31 / 34 / 52 ms at
  B = 4 / 8 / 16. Through the gateway (600-token prompts, 512-token answers):
  c=8 52 → 93 tok/s, c=16 69 → 81–99. Numerics: ≤ qmm's own error vs an fp32
  reference on every real 9B/35B shape (rms rel 2.9e-4 = fp16 output rounding);
  9B teacher-forced 100% top-1 at B=3/4/8 (KL 8e-6), 99.7% at 16 (KL 1e-5; qmm
  99.8%, 1.1e-5). Only 8-bit affine g64 fp16-scale modules with N ≥ 512 (35B:
  attention + GDN, not the router / experts / shared expert / lm_head; 27B
  5-bit: none), only (rows, 1, K) inputs with 3 ≤ rows ≤ 16 — B ≤ 2 stays
  bit-exact qmv, prefill and DFlash (always (1, T, K)) never take it. Tried: NA 8
  fragments per simdgroup spills (×5 slower); 2×8 / 4×4 / 2×4 tiles are within 3%.
- **4-bit mrqmm (`QWEN_MRQMM4`, 2026-11-15)**: `_WSIDE[4]` (existed since 2026-11-10 for the bench, now dispatched):
  a 64-group is 8 aligned words, a thread loads 2 (one uint2) and regroups them into 16-bit lanes once per group;
  config as 5 bits (4×8 at ≤ 8 rows, 2×8 above: KS 8 in both buckets, a row's bits the same at 3..16 rows). GPU unit
  tests (synthetic, the 27B 4-bit shapes incl. lm_head): within fp16 rounding of fp32 and ≤ `quantized_matmul`'s
  error, rows independent of the row count, self-test passes. Selection is `mrqmm._bits_on` + `supported` only, so
  every 5-bit user (MRQuantizedLinear, `_gdn_call`, `_verify_linear(s)`, `_verify_head`) takes 4 bits unchanged.
  Random 4-bit weights (2026-11-10): 8 rows ×1.64 of a solo row, MMA-bound like 5 bits — a 4-bit build gains its
  smaller weights (solo row 46.5 vs 55 ms of linears), not a cheaper kernel. Model-level gate 2026-11-15 (`docs/perf/2026-11-16/profile_27b.md` §5): ×1.75 DFlash, numerics in class, identity 12/16 at near-ties; default off.
- **oQ4e / mixed precision (2026-11-15, `docs/perf/2026-11-15/design/oq4e_speed.md`, GPU gate
  `scripts/perf/dflash2/gate_oq4e.sh` pending)**: measured that day (separate processes, ~2K prompt):
  plain 11.6–11.9 vs 17.2 tok/s (uniform 4-bit), DFlash 16.6–17.6 vs 37.7 (round 259–274 vs 153 ms,
  3.57 vs 4.8 drafts accepted). Measured drift-free (gate steps 1–3, `oq4e_bench.py`, both
  checkpoints' tensors in one process): the linears of a 1-row forward 48.4 vs 45.2 ms (×1.07; 5 / 6-bit
  qmv streams as fast as 4-bit, the 6-bit head 349 vs 352 GB/s), so ~25 % of the measured plain gap
  is outside the kernels: drift between the processes or the weights' residency, to be split by
  `decode_probe.py` (step 5, a model run). DFlash costs found: mlx-vlm's exact verify kernels take 4/5/8 bits only, so the 6-bit lm_head went
  through per-token qmv (~8 head reads per round) and the drafter's head (the target's) through
  `quantized_matmul`; fixed by `QWEN_MRQMM6` (6-bit weight side `_WSIDE[6]`: 3 aligned words per
  thread, 4 lane words, one shift + mask per step; self-tested at load). The GDN in-projections
  were never fused (mixed widths); now by width groups (30 layers qkv+z 4 + b+a 5, 15 qkv alone +
  z+b+a 5, 2 qkv+z + b+a 8, 1 qkv+z + b, a alone). Re-quantizing the head to 8 bits was rejected
  (+0.3 GiB, a second rounding, +0.9 ms per plain token). Verify block T = 8, linears: 117.8 →
  87.6 ms (head 22.8 → 4.0 via mrqmm6, drafter head 11.6 → 4.0, GDN in 17.5 → 15.0) vs 80.8 for
  the uniform 4-bit; the 6-bit kernel 2.65e-4 rms rel (≤ qmm), 4×8 config kept. `oq4e_bench.py`: per-width qmv, verify
  block per kind, lm_head paths, kernel accuracy, both checkpoints in one process.

## Fused GDN in-projection

- **Fused GDN decode in-projection (`QWEN_GDN_FUSE=1`, 2026-10-12; by width groups since 2026-11-15)**: mlx-vlm 0.7
  dropped 0.6.2's `_decode_quantized_linears_fused`, so a decode step ran qkv/z/b/a
  as 4 launches per GDN layer. `_install_gdn_fuse` concatenates the four weights once
  and points the modules at row views of it (no extra memory; 0.6.2 kept a second
  1.2 GB copy); `_gdn_call` runs one matmul (qmv at B ≤ 2, mrqmm for 3..16 rows on
  8-bit models with b/a riding along, else `quantized_matmul`'s own qmv/qmm — e.g.
  `QWEN_MRQMM=0` or the 5-bit 27B) and hands the outputs to the library `__call__`.
  Bit-identical at B ≤ 2; in-run A/B −2.3% at B=1, −1.6% at B=16. The concatenated
  shapes' mrqmm kernels are compiled at install. Needs fp16 scales/biases: with
  `QWEN_FP16=0` it fuses nothing ("0 layers" in the load log). Mixed-width layers
  (oQ4e): `_gdn_groups` concatenates the projections that share (bits, group, mode)
  (≥ 2 of them); a projection alone at its width keeps its own weights and call.
  `_fused_in` is a tuple of `_InGroup`s (a uniform layer: one group of four, the old
  concatenation and split); a group takes mrqmm only with N ≥ `MIN_N` (b+a, N = 96,
  stays on `quantized_matmul`). B ≤ 2 bit-identical to the unfused calls on the GPU
  (`test_worker_gdn_fuse_mixed`, tiny mixed model through the worker; the CPU suite checks B = 1).

## One-pass attention (vatt)

- **One-pass verify / decode attention (`vatt.py`, 2026-11-16, `docs/perf/2026-11-16/verify_att2.md`)**: at long
  context the 27B's verify block attention (G 6, `QWEN_VERIFY_ATT=1`: 5 + 3 queries per vector-kernel call, ~30 GB/s)
  was ~45 % of a DFlash round and MLX's decode SDPA read the KV at ~90 GB/s. `QWEN_VERIFY_ATT=2` / `QWEN_DECODE_ATT`:
  steel-attention MMA loop (via `cascade_mma.steel_headers`) over key chunks, 16-row query blocks, causal tail per
  block position, fp32 partials + merge, KV buffers read in place. One layer at 60K: verify 8.14 → 2.45 ms (4.9 TFLOPS:
  compute-bound, 49 FLOP/byte; GEMM-rate floor ~1.3 ms), decode 2.77 → 0.91 ms (276 GB/s of ~339). 27B 4-bit at 60K:
  DFlash 17.97 → 26.54 tok/s, plain 10.61 → 14.37; at 32K 22.31 → 30.13 / 14.87 → 16.53; short prompts equal; tokens unchanged on the long runs; KL 1.2–1.3e-5. `QWEN_VERIFY_ATT=2` is the default since 2026-11-16; `QWEN_DECODE_ATT` stays off (recipe knob). Tiles: 16x16 best;
  48-row single blocks, WN = 1 and split smem regions slower (sweep in the report).

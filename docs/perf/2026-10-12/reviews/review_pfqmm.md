# Review: pfqmm.py + fixed fp16 prefill slab (_step) — read-only

Date 2026-10-12. Sources: src/worker/pfqmm.py, worker.py (`_step` 785, `_bounds` 806, `_batch_budget` 1280,
`_fused_sdpa` 1356, `_probe_fused_sdpa` 1378, `_slab_done` 2026, load 2614/2624), tests/test_pfqmm.py,
mlx 0.32.3 wheel headers (kernels/quantized.h `qmm_t_impl`/`affine_qmm_t`, steel/gemm/mma.h, loader.h) and
scratchpad/src_mlx/v0.32.3/{quantized.cpp,scaled_dot_product_attention.cpp,fast.cpp}. Nothing was run.

## Findings (ranked)

### 1. MEDIUM: the load-time self-test can pass without testing anything, and does on the 35B's 8-bit key and the tiny test model
pfqmm.py:233-247 `_selftest` skips any M at which `_mlx_uses_qmm` is false and returns None ("ok") when it
skips both M=256 and M=200. install() (262-272) tests only the first module of each (bits, gs) key, in
`named_modules()` order. That order is a LIFO DFS (mlx `apply_to_modules` pops the last child first), so:
- 9B / 27B: lm_head comes first (N=248K): a real test.
- 35B: 6-bit is lm_head (tested). 4-bit is `layers.39.mlp.shared_expert.down_proj` (N=2048, tested). 8-bit
  is reached first at **`layers.39.mlp.gate`** (the MoE block is popped before attention; children
  gate/switch_mlp/shared_expert/shared_expert_gate are popped in reverse, shared_expert_gate N=1 is unsupported,
  switch_mlp is not a QuantizedLinear). Its shape, K=2048 and N=256, gives split-K 8 at both M=256 and M=200
  (I checked the arithmetic with the same formula). So the 8-bit key (270 modules: attention, GDN in/out)
  turns on with no bit check at all.
- The tiny model in test_tiny_model_prefill (K = 128/256, N <= 512) splits K at both Ms for every module.
  `assert pfqmm.install(lm) > 0` therefore passes vacuously. The test's own ACTIVE on/off comparison is
  what actually checks it.

Impact today: none observed (35B corpus 15/15 byte-identical). But the safety net that the docs rely on
("self-tested at load", "stays off when an mlx bump changes qmm_t", and the only NAX guard, see #6) has a hole
on exactly the model with mixed bits.
Fix: pick the test M per module so that MLX surely takes qmm_t, e.g. smallest M >= 256 with
`ceil(N/32)*ceil(M/32) > 512`, plus M+8 for the ragged tile. Or walk the key's modules until one qualifies.
If nothing qualifies, return a reason ("untested") instead of None. Add a CPU-arithmetic unit test that
`_selftest` never returns None without comparing.

### 2. MEDIUM: under fp16 KV, `_step` ignores QWEN_PREFILL_BUDGET below 32K keys, but the batch reserve still assumes it
worker.py:794-795: the fp16 + fused branch returns PREFILL_STEP at every offset. Below SDPA_FUSED_MIN_KV (32K
keys) the slab still takes the unfused SDPA (`_fused_sdpa` needs keys >= 32K for both of its clauses). The
unfused score transient is then PREFILL_STEP × keys × heads × 2 B, up to PREFILL_STEP × 32K. That fits
PREFILL_BUDGET only for the defaults (2048, 2048×32K).
- `QWEN_PREFILL_STEP=4096` on the 9B: a slab at pos 27K (keys 31K) is unfused at 4096×31K, about 1.9× the
  budget, so ~4 GiB of scores (plus the softmax copy). The old rule gave min(4096, 67M/27K) = 2485.
- `QWEN_PREFILL_BUDGET` lowered (e.g. 512×32K, to shrink the reserve or peaks): it is now silently ignored
  for fp16 slabs below 32K keys. Meanwhile `reserve = PREFILL_BUDGET × heads × 2 + 0.4 GiB` (2624) shrinks
  with it, so the scheduler under-reserves and admits rows into memory that the prefill then needs.
The CLAUDE.md line 167 claim ("the batch budget is unchanged because unfused scores stay <= 2048 × 32K") holds
only for the defaults. README 406 says the budget does not apply to "the 9B", but it does not warn about the
reserve.
Fix (either): in the fp16 branch keep `min(PREFILL_STEP, PREFILL_BUDGET // pos)` while `pos + step <
SDPA_FUSED_MIN_KV`, and PREFILL_STEP above it. Or compute the reserve from
`max(PREFILL_BUDGET, PREFILL_STEP * SDPA_FUSED_MIN_KV)` when the fixed slab is active.

### 3. MEDIUM-LOW (behaviour change, undocumented): long prefills now stall concurrent decode rows ~3× longer per tick
The scheduler runs one slab per tick, uninterrupted (`_prefill`/`_spec` yield after each slab; `_slab_done`
only spaces the ticks). The old adaptive slab kept a 120K-offset slab at ~546 tokens (≈3.2 s). The fixed
2048 slab takes ≈8.4-9.5 s there (profile: plain slab 8418-9528 ms), and ≈5 s at 32K. During that time
every running decode row produces nothing. Prefill throughput improves, but under batching the worst
inter-token gap of other streams roughly triples once a >64K prompt is admitted.
Shrinking the slab only under load would change slab shapes, and so the bits (GDN b/a split-K by M,
`_bounds` determinism), so this is a trade-off to document (README "Batching" / CLAUDE.md), not a quick
fix. A possible option: a `QWEN_PREFILL_STEP_BATCH` that applies identically to cold and cached runs
(via `_bounds`) only when max_batch > 1.

### 4. LOW: the runtime fallback in `_fused_sdpa` is no longer bounded
worker.py:1356-1376: when `force_fused` raises at call time, the same 2048-row slab goes to the library's
unfused path. At 120K keys that is 2048×120K×16×2 B ≈ 8 GB of scores (plus softmax), well past what the
9B has left on 32 GB. Before the change, `_step` had already shrunk that slab to ~546. It is practically
unreachable: has_fused_kernel for q>8 only rejects head dims, q_len > k_len and the logsumexp case, and
`_probe_fused_sdpa` covers the head dim. The same applies if a future code path hands an array mask or a
non-KVCache cache to an fp16 2048 slab at >32K keys: `_step` cannot see it.
Fix: in the fallback, when q*k > PREFILL_BUDGET, raise (fail the request) or run the library SDPA over
query chunks of PREFILL_BUDGET // k rows (causal offset mask), rather than one unbounded call.

### 5. LOW: `_step` reads the mutable global SDPA_FUSED
A runtime flip (#4) changes `_bounds` for all later requests. Prompt-cache entries stored before the flip
were cut on the old grid, so a later hit vs a cold run of the same prompt are no longer guaranteed
byte-identical, which breaks the `_bounds` docstring promise. Cheapest fix: freeze the slab policy at load
in a separate constant (`_FIXED_SLAB`) and use that in `_step`. Or clear the prompt cache when SDPA_FUSED
flips.

### 6. LOW: no NAX / architecture guard; bit-identity rests on the self-test
pfqmm's docstring says "GPUs without NAX", but nothing checks for it. On NAX GPUs (mlx `qmm` goes to
`qmm_nax` when K % 64 == 0) the output would differ from quantized_matmul (still correct math, but other
bits), and the 32×32-tile speed premise is unmeasured there. On M3/M4 (non-NAX), qmm_t is the same, so
pfqmm is bit-identical but the speed-up is unmeasured. Today the only guard is the self-test, which #1
shows can be vacuous. Fix #1. Optionally gate on `mx.device_info()["architecture"]` (applegpu_g13*/g14*),
or log the architecture next to the coverage line.

### 7. LOW (latent): the dispatch mirror assumes row-contiguous x
`_use`/`_mlx_uses_qmm` use M = x.size // K. MLX uses M = x.shape[-2] when x is not row-contiguous
(quantized.cpp:1885-1887); then B > 1 takes batched qmm (no split-K), or qmv when shape[-2] < the vector
limit. In those cases MLX does not run the split-K / qmm_t pfqmm predicts, so pfqmm's output would not be
bit-identical. It cannot happen today (prefill is B=1; batched decode rows are M <= 16 < 128), and Python
cannot see contiguity. A one-line guard keeps the claim exact: only take pfqmm when all leading dims are 1
(`x.size // K == x.shape[-2]`, or x.ndim == 2).

### 8. LOW: no log line when pfqmm covers nothing (QWEN_FP16=0)
pfqmm.py:259-261: `if not mods: return 0` returns before the coverage log. With QWEN_FP16=0 (bf16 scales)
pfqmm is silently off, and neither the README nor the CLAUDE.md env row says that QWEN_FP16=0 disables it
(the worker comment at 2613 does). Fix: log "pfqmm: no fp16 affine QuantizedLinear (QWEN_FP16=0?)" and
mention it in the env tables.

## Nits
- pfqmm.py:179-183 docstring: "32 is above [the qmv limit] on every device" is false on gen >= 17 non-'d'
  (limit 33, quantized.cpp:92-94). It is harmless because `_use` needs M >= 128.
- `_DIMS` (pfqmm.py:110,169-171) is an unbounded cache keyed by (M, K, N). It holds every distinct
  last-slab length × shape, up to ~2048 × #shapes tiny buffers over a worker's life. That is bounded and
  small, but a plain `mx.array` per call (or an LRU) would be just as fast.
- `BITS` includes 2 and `GROUPS` 32/128 are accepted, but test_pfqmm only covers g64 at 3/4/5/6/8 bits. The
  load self-test covers real models' keys (modulo #1).
- worker.py:648-655 / README 419 / CLAUDE.md 140: "×1.10-1.14 per matmul" vs the measured 4-bit ×1.03-1.14,
  3-bit ×1.04-1.20, 6-bit ×1.07-1.28. CLAUDE.md says "bit-identical up to 40K", the results say 48K was
  checked too.
- README 406 "(the 9B keeps 2048)": only with QWEN_KV_BITS=0, QWEN_FP16=1 and the fused probe passing (the
  KV_BITS default is 4).
- The short-slab clause of `_fused_sdpa` (q > 16, q·k > PREFILL_BUDGET) can never fire at default settings
  and context <= 128K: q < 512 would need k > 131,328. It is reachable only with QWEN_MAX_CONTEXT > 128K or a
  lowered budget. The comments describe it as covering "message cut, last slab" in general. Harmless as a
  guard.
- Pre-existing, not from this diff: the measured 32K peak (14.6 GB = 9.7 weights + 1 GiB KV + ~3.9 GB
  transient) suggests the unfused 2048×32K slab needs more than the 2.4 GiB prefill reserve (scores plus the
  softmax output, ~2× scores). The 1 GiB slack absorbs most of it. Worth one measurement under batching.

## Tests (C)
- CPU (always run): `_mlx_uses_qmm` table, `_use` row rule, dtype/supported, `_step` values. The
  `_mlx_uses_qmm` expectations are hand-derived from the same formula, so they cannot catch a misreading of
  MLX's dispatch. Only the GPU test compares against MLX.
- PFQMM_GPU=1 only: kernel bit-identity (g64; bits 3/4/5/6/8; (K,N) = (4096,4096), (4096,1024), (5120,640);
  M = 128/200/256/640/2048; full and ragged 64-row tiles), and the tiny-model prefill A/B (ACTIVE on/off,
  logits equal, ≥32 kernel calls).
- Not covered anywhere:
  - install's per-key failure path: one key off, others on, and the exception branch
  - the vacuous self-test (#1)
  - `_call` with a `bias`
  - MRQuantizedLinear routing prefill into pfqmm via `super().__call__`
  - GDN-fuse row-view weights (offset buffers) through pfqmm, though the 9B identity corpus covers it end to end
  - the `_fused_sdpa` short-slab clause
  - `_bounds` under the fixed step
  - group sizes 32/128 and 2-bit
  - x with batch > 1

## Verified OK
- Metal source: `_IMPL` is `qmm_t_impl` (quantized.h:1199-1317) with aligned_N = true, K_eff = K and the
  (K, N, M) params reordered consistently. Same `BlockLoader` / `QuantizedBlockLoader<BN=32>` / `BlockMMA<T,
  T, BM, BN, BK, 2, 2, false, true, BK_padded, BK_padded>` with the default AccumType float (mma.h:451).
  The fp32 accumulation and the cast at store are identical.
- Per element, `mma()` walks kk = 0..BK step 8 with `tile_matmad`, in the same order for TM = 2 and 4.
  Bit-identity is plausible structurally and was confirmed empirically.
- Tails: an M tail uses `load_safe(short2(BK, num_els))` + `store_result_safe(short2(BN, num_els))`.
  `supported` rejects N % 32, K % 32 and K % gs, so no unguarded column or K tail exists. Offsets are int64
  for x/y and int for the weights, as in MLX (lm_head 248320×4096 bytes < 2^31). There is no OOB read or
  write.
- 3/5/6-bit packing comes from the wheel's `get_pack_factor`/`get_bytes_per_pack` (K_w = K·bytes/pack), as
  in MLX. `supported` derives K = w.shape[1]·32/bits and checks the scales shape.
- Launch: grid (N/32·64, ceil(M/64)·2) threads, threadgroup (64, 2, 1) gives 128 threads = 4 simdgroups,
  which matches MLX's (32, 2, 2). Threadgroup memory is 64·40·2 + 32·40·2 = 7.5 KB, far under 32 KB.
- Dispatch mirror: `_mlx_uses_qmm` equals `qmm_splitk` (quantized.cpp:1166-1184: k_align = max(gs, 32), the
  same as gs for gs ≥ 32). Nothing pfqmm accepts reaches qmv (M ≥ 128 > every vector limit). NAX: see #6.
- fp16 only: `_use` requires an fp16 x and `supported` requires fp16 scales/biases. bf16 (QWEN_FP16=0) never
  reaches the kernel (#8 is only about logging).
- Integration: install runs after the fp16 cast and before mrqmm and the GDN fuse. The flag lives in the
  instance `__dict__`, so it survives mrqmm's `__class__` swap, and `MRQuantizedLinear.__call__ ->
  super().__call__` hits the patched `_call`. Fused GDN only handles L == 1 (decode); prefill uses the
  per-module calls on row-view weights (offset buffers are handled by metal_kernel/set_input_array).
  `_use` is false for: GDN b/a (N=32, split-K), DFlash verify/drafter (≤16 rows), decode rows (≤ max_batch),
  and lm_head on prefill slabs (skip_logits; last token M=1).
  The vision tower and drafter are not installed (LM only).
  The 35B experts (QuantizedSwitchLinear / gather_qmm) are untouched.
  The router `mlp.gate` (N=256) is taken only at M ≥ 2048-ish, where MLX also runs qmm_t.
- Thread safety: kernels are built under `_LOCK`, and the MLX library cache is internally locked. `_DIMS`
  races only create duplicate tiny arrays. The model has one caller thread (the scheduler).
- Vision prefill: inputs_embeds stay fp16 (`masked_scatter` assigns into the fp16 embedding), so the fused
  SDPA applies. Positions are sliced per slab, and `_bounds` = [] for images. The slab change only alters n.
- Prompt cache: `_bounds` and `_slab` use the same `_step`, so cold runs and hits cut identically. Fewer cuts
  at long context only coarsen the checkpoints (stride 2048 anyway).
- Probe failure at load sets SDPA_FUSED False, and `_step` falls back to the budget rule (bounded).
- 27B/35B (quantized KV, slab 512 by docs) keep the old shrinking slab. The `_step` change does not affect
  them. The 9B 120K peak of 16.65 GB is below the old 17.7-18.1.

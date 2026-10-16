# Packed prefill (several waiting requests' suffixes in one forward): feasibility

Date 2026-10-16. CPU only: no GPU, no worker, nothing in `src/` changed. Line numbers are worker.py at HEAD
34b4a81 and mlx-vlm 0.7.6 / mlx 0.32.3 in `src/worker/.venv`. Load model: `docs/perf/2026-10-16/HANDOFF-appworld.md`
(8 AppWorld episodes, 94 % prompt-cache hits, prefilled tokens per request p50 379 / mean 693 / p90 2.8K /
max 4.7K, each at a cache offset of 5–33K; 76 % of scheduler ticks have a prefill running, 66 % have a queue).

Probes (CPU, `cd src/worker && uv run --group dev python <script>`):

* `docs/perf/2026-10-16/design/packed_prefill_probe.py`: the whole-model probe (E2). It is a copy of
  `scratchpad/packed/packed_model.py`.
* In the session scratchpad
  (`/tmp/scratch/packed/`):
  * `gdn_padded.py` (E1);
  * `dispatch.py` (E3);
  * `diag_*.py` (where the CPU differences come from).

## Verdict

**No-go for this load.** Recommendation: do not build packed prefill. Spend the effort on memory (prompt-cache
checkpoints and eviction, swap) and on decode-step attention at long context.

* **A bit-exact packed prefill is feasible.** The cheap variant (b) does it: the GEMMs and other per-token ops
  run once over the rows' tokens concatenated, and the GDN recurrence, conv and attention run per row on its
  own slice with the solo call shapes. On CPU it reproduces every KV and GDN-state array bit for bit in the
  production layout, 8-bit weights with fp16 scales, for the tiny model and for a 4-layer model with 9B
  widths.
  * Rule 1: the N=32 GDN b/a projections, which always run split-K on Metal, stay per row.
  * Rule 2: a row's slab shorter than the qmm_t threshold of a shape (attn k/v: 257 rows) runs that module
    alone.
* **It does not pay.** A prefill slab of the 9B is compute-bound from about 15 tokens on. An 8-bit GEMM with T
  rows does 2T FLOP per weight byte; the machine balance is 10.4 TFLOPS / 400 GB/s = 26 FLOP/B. The profile
  measured the GEMM rate flat from M = 512 to 4096.
  * Packing 8 × 377 tokens into M ≈ 3000 saves no FLOPs.
  * Attention and the GDN recurrence stay per row.
  * What is left: the ragged last tile (≤ 2 %), host costs per slab (≤ 1 %), and maybe a better GEMM rate at
    M = 377 than at 3000 (unmeasured; ≤ 5 % of GEMM time if any).
  * That is **≈ 1–6 % of the packed requests' prefill time**, ≤ ~2 % of wall throughput (prefill ≈ 40 % of
    GPU time here).
* **It hurts TTFT.** A pack of R requests delivers every first token after the whole pack. FIFO delivers the
  head's after its own slab. Mean TTFT of a pack of equal rows goes from (R+1)/2 · p to R · p (×1.8 at R = 8),
  bought with that 1–6 %.
* **What would change the verdict.** A load with many suffixes under ~32 tokens (that is the bandwidth-bound,
  decode-like regime), or a GPU curve showing a prefill slab much slower per token at T ≈ 256–512 than at
  2048 (see the "GPU checks" section).

## 1. The GDN prefill path (mlx-vlm 0.7.6)

`Qwen3_5GatedDeltaNet.__call__` (`models/qwen3_5/language.py` 1047–1122) on a (B, S, H) input:

1. **In-projections.** `in_proj_qkv`, `in_proj_z` and `_project_gates` (b, a) are plain Linear /
   QuantizedLinear over all B·S tokens (1054–1056).
2. **Conv state.** `conv_state = cache[0]`, or zeros (K−1 = 3 rows) when it is absent or has another batch
   size (1060–1071).
3. **Mask.** `mask` (B, S) bool zeroes `mixed_qkv` at masked steps before the conv (1073–1077).
4. **Conv window store.** `conv_input = concat(conv_state, mixed_qkv)`. `cache.update_window(0, conv_input,
   K−1, lengths=cache.lengths)` stores the next window (1078–1082).
   * Without `lengths` it stores the last K−1 inputs (with right padding that would be pad rows).
   * With `lengths` it takes `conv_input[len_i : len_i+K−1]` per row, which is the row's last K−1 real inputs
     (`models/cache.py` 954–975).
5. **Conv.** Depthwise `nn.Conv1d` over the whole (B, K−1+S) input (1090). The S == 1 fp16 path is the decode
   one (1083–1088).
6. **Recurrence.** `gated_delta_update(q, k, v, a, b, A_log, dt_bias, mask=mask, cache=cache)` (1103) →
   `cache.update_recurrent` → `gated_delta.py` 382, which dispatches at 441:
   * **GPU (production)**: `gated_delta_kernel` (170), the `metal_kernel` built at 12–113.
     * One thread per (row, v-head, dv) element of the state.
     * It runs a sequential loop over T with the fp32 state in registers. `simd_sum` over the 32 lanes of
       dk.
     * With a mask (`gated_delta_step_mask`), a masked step leaves the state untouched and writes y = 0. So
       for right padding the **final state is the state at T_i**.
     * T is a runtime input, not a template constant.
     * There is no chunking and no cross-row interaction: a row's arithmetic is the same instruction stream
       whether it runs alone or in a batch.
   * **CPU**:
     * T > 1 takes `gated_delta_chunked` (288): a C = 64 chunked scan. Masked steps get g = 1, β = 0. T is
       zero-padded to a multiple of 64.
     * T == 1 (or vectorized gating) takes the sequential `gated_delta_ops` (213).
     * **So the CPU and Metal paths run different algorithms**: CPU results say nothing about the Metal
       kernel's bits.
   * `mx.fast.gated_delta_update` is only `QWEN_GDN_FAST`, which is rejected (it rounds g to fp16).
   * A kernel that returns every intermediate state also exists: `gated_delta_update_with_states`, 595, used
     by the speculative history. It is not needed: the masked final state already is the state at T_i.
7. **Output.** Gated RMSNorm with z, then `out_proj` (1121–1122).
8. **Batch mask.** The library builds the batch mask itself for a right-padded batch: `ArraysCache.make_mask` =
   `pos < lengths` (`cache.py` 1277–1285), via `_create_qwen3_5_ssm_mask` (`language.py` 189).

So mlx-vlm 0.7.6 already supports **right-padded batched prefill of the GDN layers**. Rows at different
offsets carry their own conv window and state on axis 0, as our decode batching does with `_gdn_stack`.

### E1: right-padded batch vs solo rows, bitwise (CPU, `gdn_padded.py`)

Setup:
* 3 rows, T = 377 / 120 / 250, each from its own nonzero fp32 state and conv window (rows at different
  offsets). Padded batch (3, 377): the library mask `pos < T_i` plus `cache.lengths`.
* Dims: the 9B GDN dims from `weights/ornith15-9b-mlx-8bit/config.json` (hidden 4096, 16 k-heads × 128, 32
  v-heads × 128, conv 4), and the tiny test dims.

| level | 9B dims | tiny dims |
|---|---|---|
| recurrence, CPU chunked (fp16, fp32): y of real tokens, final state vs solo | **equal, all rows** | **equal** |
| recurrence, sequential ops (fp16; tiny fp32 too) | **equal** | **equal** |
| the same, padded **without** the mask | final state differs for the 2 shorter rows (padding ran through the recurrence) | same |
| whole module (ArraysCache + lengths), 8-bit fp16, fp16, fp32: y, conv window, state | **equal, all rows** | 8-bit fp16 **equal**; unquantized fp16 / fp32: 1–2 rows differ by 1e-9..1e-5 |

**Where the tiny unquantized differences come from (`diag_*.py`).** None of them are GDN effects; they are
CPU backend artifacts in the projections' inputs:

* **CPU elementwise SIMD tails.** fp16 `sigmoid` / `silu` of 1508 elements differs from the same values
  computed inside a longer array at index 1506 (the scalar tail vs the SIMD body). Sizes that are multiples of
  16 never differ, which is why the 9B dims are clean.
* **CPU fp32 `rms_norm` depends on the row count.** (1, 61, H) differs from rows 0..60 of (1, 190, H); fp16
  does not.
* **CPU fp32 BLAS takes another path for tiny M × N.** N = 4 at M = 9.

The Metal analogues are listed in the "GPU checks" section.

**Conclusion for (1):**
* Padded batching of the GDN is bit-exact per row on the CPU's chunked path (shown).
* On Metal it is bit-exact by construction of the kernel (each row's loop is the same code; masked steps are
  skipped). This needs one GPU bitwise check.
* The state at T_i is the masked final state.
* The conv window is right if `cache.lengths = T_i`.

## 2. Attention

Our batched decode runs attention per row (`_rows_attention`, worker.py 1665). It does the projections, norms
and RoPE batched, then `c.update_and_fetch` and `scaled_dot_product_attention` per row with `mask=None` (L =
1).

**What a prefill slab (L > 1) uses:**
* `_create_qwen3_5_attention_mask(h, cache[fa_idx])` → `KVCache.make_mask` → `create_attention_mask(N,
  offset)`. That is the `"causal"` string (MLX's fused causal path) or, for some shapes, an explicit array.
* The worker's `_fused_sdpa` override (1815): fused kernel for q ≥ 512 over ≥ 32K keys, else the default
  unfused graph.

**Packed rows need each row's own causal mask at its own offset, over its own KV.**

* **(a) Library batch attention.** `BatchKVCache` with left padding: every row's KV padded to the longest
  (33K), an explicit array mask, and one SDPA over (R, T_max) queries.
  * Different kernel, different shapes, different bits; and R × 33K of KV copies.
  * **Not viable.**
* **(b) Per row, as `_rows_attention` does.** Slice the row's queries / keys / values out of the packed
  projections. Build the row's mask from its own cache exactly as the solo slab does. Call `update_and_fetch`
  + SDPA with the solo shapes. This is the same call as the solo slab, so **the same bits by construction**
  (and the same `_fused_sdpa` decision, which depends only on the row's q length and key count).
  * Cost: R × 8 FA layers of SDPA calls. There is nothing to gain by batching them: at offsets 5–33K the
    attention is compute on keys either way.

## 3. GEMMs: row independence and dispatch

Variant (b) packs the rows' tokens as **(1, ΣT_i, K)**: row-contiguous, which is what pfqmm's `_use` (pfqmm.py
214) requires. It does not pack them as (R, T_max, K): padding wastes FLOPs, and leading dims > 1 push MLX to
batched dispatch.

**A row's output bits are the same packed or alone iff, for every module:**

1. **Both M = T_i (solo) and M = ΣT (packed) run a kernel of the qmm_t family.** pfqmm (64-row tiles) and MLX's
   qmm_t (32-row tiles) are bit-identical by pfqmm's design and load self-test (pfqmm.py 257). Both
   accumulate each output element over the same BK = 32 steps of 8×8 MMAs in K order, independent of M.
2. **The output row's value does not depend on where in the tile it sits.** The safe choice is to start every
   row's slice at a multiple of 64 in the pack (≤ 63 filler rows each), so its intra-tile positions equal the
   solo ones. Then only the threadgroup's grid index differs.

**qmv (M < 32) and split-K (K partitioned by a split that depends on M) violate rule 1.** E3 (`dispatch.py`,
pfqmm's mirror of mlx 0.32.3's dispatch) gives the kernel per 9B prefill shape:

| shape (K, N) | M=16 | 32 | 64 | 96 | 128 | 256 | 377 | 1024 | 3016 | first M on qmm_t |
|---|---|---|---|---|---|---|---|---|---|---|
| gdn qkv / attn q (4096, 8192) | qmv | splitK2 | qmm_t | qmm_t | pfqmm | pfqmm | pfqmm | pfqmm | pfqmm | 33 |
| gdn z, gdn out, attn o (4096, 4096) | qmv | splitK4 | splitK2 | qmm_t | pfqmm | pfqmm | pfqmm | pfqmm | pfqmm | 65 |
| **gdn b / a (4096, 32)** | qmv | splitK64 | splitK64 | splitK64 | splitK64 | splitK64 | splitK32 | splitK16 | splitK4 | **never** |
| **attn k / v (4096, 1024)** | qmv | splitK16 | splitK8 | splitK4 | splitK4 | splitK2 | pfqmm | pfqmm | pfqmm | **257** |
| mlp gate / up (4096, 12288) | qmv | qmm_t | qmm_t | qmm_t | pfqmm | pfqmm | pfqmm | pfqmm | pfqmm | 32 |
| mlp down (12288, 4096) | qmv | splitK4 | splitK2 | qmm_t | pfqmm | pfqmm | pfqmm | pfqmm | pfqmm | 65 |

(The b/a split value is the mirror's; CLAUDE.md quotes 16 / 8 from the kernel. Either way it is split-K at
every M and its split moves with M.)

**Rules for bit-exact packing:**
* **b / a always per row.** It is 0.4 % of a slab (18 ms per 2048 tokens).
* **k / v per row** for a row whose slab is < 257 tokens.
* **z / o / out / down per row** for a slab < 65 tokens; qkv / q for a slab < 33.
* Equivalently: a module runs packed over the rows whose solo M takes qmm_t/pfqmm and alone for the others.
* A row with a slab < 33 tokens is simply not packed.

**CPU evidence.**
* CPU `quantized_matmul` computes each row alone. In the production layout (8-bit, fp16 scales) every packed
  row matched its solo GEMM in E1 / E2.
* CPU fp32 BLAS did not for N = 4 at M = 9: the CPU analogue of the split-K / qmv rule.
* The CPU cannot say anything about Metal tiles. See the "GPU checks" section (G2).

**Everything else in a layer is per token:**
* embedding gather, `mx.fast.rms_norm` per row, compiled swiglu, sigmoid gate, residual adds, RoPE with the
  row's positions;
* the GDN gated norm (it is per token but runs per row in the probe anyway).

On Metal these kernels compute each element / row with the same code wherever it sits. The CPU differs
(SIMD tails, fp32 `rms_norm`), so this is a GPU check too.

## 4. Bit-exactness: which variant can keep hit == cold

**Unchanged by packing.** The prompt cache's guarantee (`tests/conftest.py` `check_cold`: every resumable
position of every entry restores the cold prefill's bits) depends on each row's slabs producing the cold
slabs' bits.
* A packed tick runs **each row's next slab with its own cut**: `_slab(pos, left, cuts)` with the row's
  `_bounds` (1133, 1140). So the slab sequence per row is unchanged; only M changes.
* `_bounds` / `_step` are untouched (they must not depend on what else runs).

| variant | what is batched | bit-exact? | cost / risk |
|---|---|---|---|
| (a) padded batch (R, T_max) through the library | GEMMs on padded rows; GDN via the masked kernel + `lengths`; attention would need BatchKVCache | GDN: yes (E1; Metal by construction). Attention: **no** (§2). GEMMs: (R, T_max, K) leaves pfqmm (leading dim), MLX's batched dispatch is unverified, and padding wastes up to 50 % at 2× spread | not viable as is |
| **(b) flattened GEMMs + per-row sequence ops** | every GEMM / norm / elementwise op once over (1, ΣT); GDN conv + recurrence + attention per row on its slice with the solo shapes | **yes on CPU** in the production layout (E2), with b/a per row and the k/v / small-M rules; on Metal it hinges on G2 | ~150 lines: a packed forward mirroring `Qwen3_5DecoderLayer` / GDN / attention (a second diff point on every mlx-vlm bump, like `_rows_attention` / `_gdn_call`) |
| (b+) = (b) with the R rows' GDN recurrences in one masked-kernel launch (right-padded stack, `lengths`) | + the recurrence | CPU yes (E1); Metal by construction; G5 | recurrence ~4 % of a slab, and the kernel at B = 1 already runs 32 × 128 × 32 threads (likely saturating), so the launch merge may gain little |
| (c) custom varlen Metal kernel (cu_seqlens) for the GDN | as (b+) without padding | can be (same per-thread loop) | a kernel to own; gain ≤ (b+) |

### E2: variant (b), whole model, CPU (`packed_prefill_probe.py`)

Setup:
* R = 3 rows at cache offsets (37, 0, 150) / (150, 0, 70), each prefilling one slab of T = (61, 120, 9) /
  (121, 40, 77) tokens, packed with alignment 0 / 8 / 64.
* Reference: the solo hit slab exactly as `_Scheduler._prefill` runs it (`lm(ids, inputs_embeds=..., cache=...,
  skip_logits=True)`, zero rope deltas).
* Compared bitwise: every cache array of every layer (KV of each FA layer cut to its offset, conv window,
  recurrent state).

| model | rows equal (arrays) |
|---|---|
| tiny dense, 8-bit fp16 scales (= `dense_fused` weights) | **3/3 rows, 16/16 arrays each, every alignment** |
| 9B widths, 4 layers (3 GDN + 1 FA), 8-bit fp16 | **3/3 rows, 8/8 arrays each, every alignment** |
| tiny dense fp32 (the `dense` fixture) | offset-0 row equal. The T=9 row is equal once b/a run per row and rows start 8-aligned (fp32 BLAS N=4 at M=9). The T=61 row differs at layer 0's `rms_norm` (CPU fp32 row-count dependence): a CPU artifact, not the scheme |

So on CPU, in the production numerics, variant (b) keeps hit == cold. A CPU test suite for it would use
`dense_fused`, not the fp32 `dense` fixture.

## 5. Scheduling and the gain model

### What can be packed

* **A prefix of the queue, never overtaking (FIFO stays).** The head plus the next waiting requests that are
  prompt-cache hits, text only, not DFlash, and whose next slab is ≥ 33 tokens.
  * DFlash is never a candidate when others wait: `_spec_ok` 2523.
  * Grammar does not matter for prefill.
* **Pack size.** Cap ΣT_i at `PREFILL_STEP` (2048), so the MLP activations stay inside `_prefill_reserve`
  (1123). Each row's attention scores are already bounded per slab by `_step`, and the rows' SDPA calls run
  one after another in the graph.
* **Admission.** Each member must pass `_fits` in order, like separate admissions.
* **Per-member bookkeeping.**
  * `on_ckpt` at its own cuts;
  * its own store + `_join` when its prefill ends;
  * cancellation per member;
  * a failed pack fails its members, or retries them solo.
* **Ticks.** A member with more slabs continues in the next packed tick; the pack shrinks as members join.
  `_slab_done` (2657) charges the pack's time once.

### Cost model

Numbers come from `docs/perf/2026-10-12/prefill_profile.md` (fast machine state, pfqmm on).

**Per-token costs:**
* GEMMs: ≈ 14.1 GFLOP per token at ~9 TFLOPS = 1.57 ms/token. A plain slab at offset 0 runs ≈ 1.58
  ms/token in all.
* Attention, default unfused path below 32K keys: from `sdpa_thresh` (q 512 × 8K = 9.6 ms per layer),
  ≈ 9.6 · (T/512) · (offset/8K) ms × 8 layers. For T = 377 that is 77 ms at 11K and 210 ms at 30K.
* GDN recurrence: 0.07 ms/token.
* Host per slab: ≤ 7 ms.

**8 waiting rows of 377 tokens at ~11K:**

| | solo × 8 (today) | packed (b) |
|---|---|---|
| GEMMs | 8 × 596 ms | 3016 × 1.57 ms × (1 − δ), δ = rate gain of M 3016 vs 377, **unknown**: the profile measured flat 512–4096; 377 → 384 ragged tile ≈ 2 % |
| attention | 8 × 77 ms | the same (per row) |
| GDN recurrence | 8 × 26 ms | the same (b), or less (b+, if the kernel is not already saturated) |
| host | 8 × ~5 ms | ~15 ms (one bigger graph + R·32 per-row calls) |
| **total** | **≈ 5.65 s** | **≈ 5.3–5.6 s (1–6 %)** |

**At the load level:**
* Prefill is ≈ 296K tokens per 25 min ≈ 40 % of GPU time at ~2 ms/token, so ≤ ~2 % of wall throughput.
* Packing does not shrink the queue's prefill work, it reorders it: the head's first token comes after the
  whole pack. For the 20 % of latency spent queuing it recovers ≤ the 1–6 % above, and it **raises mean TTFT**
  within a pack (×1.8 at R = 8).

**Where the queue time actually comes from (handoff, 02:10 analysis):**
* **Memory.** Admission stops at `reserved` + prompt cache ≈ budget; `evicted to admit` happened 22 times.
  There are `KEEP_WAIT` waits, and swap stalls every tick.
* **The levers, in order:**
  * the GDN checkpoint policy and the eviction order (tasks A/B of this session);
  * `QWEN_PROMPT_CACHE_GB` / `QWEN_BATCH_RESERVE_TOKENS` sized for the load (no swap);
  * then `QWEN_BATCH_PREFILL_SHARE` (prefill's share is s = 1 − (1 − share)·B/max = 0.75 at 8 of 16 rows)
    measured against decode tok/s.
  * Each of these acts on tens of percent; packing on ≤ 6 % of prefill.

**A related idea with a better cost structure (not studied here): decode piggyback (Sarathi-style).** It folds
the running decode rows (bandwidth-bound: ~24 ms of weight streaming per step) into a prefill slab's
compute-bound GEMMs, where they cost B × 1.57 ms.

* Machinery: the same as (b), with decode rows as 1-token slices.
* Rough gain: ~20 ms per decode step taken during a prefill × ~2.8 steps/s ≈ 4–6 % of GPU time while
  prefilling.
* Caveats:
  * Decode rows would leave qmv/mrqmm for qmm_t: the B ≥ 3 numerics class. B ≤ 2 must stay bit-exact, so
    piggyback only at B ≥ 3.
  * Decode would get one token per slab.

It is also modest. The step profile's per-row attention at long context remains the bigger decode lever.

## GPU checks that decide it (in this order; the first two can kill or confirm)

1. **G3: is there any GEMM gain at all?**
   * Rate vs M on the 9B shapes (qkv 8192, z/o/out 4096, k/v 1024, gate/up 12288, down) at M ∈ {32, 64, 128,
     256, 377, 512, 1024, 2048, 3016}, pfqmm on.
   * Slab time vs T ∈ {64, 128, 256, 377, 1024, 2048} at offsets 8K and 24K, in-process interleaved.
   * **If the per-token slab time at T = 377 is within 5 % of T = 2048, stop here (expected).**
2. **G2: row independence on Metal.**
   * For each shape, a slice of T_i rows placed at offsets 0 / 8 / 37 / 64 inside M = 3016 vs the solo M =
     T_i call: bitwise equal whenever both take qmm_t/pfqmm.
   * The same for `mx.fast.rms_norm` (H = 4096), compiled `_precise_swiglu`, sigmoid / silu, RoPE.
3. **G1: the load-time self-test the feature would carry** (as pfqmm's).
   * Run 3 rows (offsets 0 / 600 / 1500, slabs 377 / 120 / 290) through the real 9B, solo vs packed.
   * All caches must be bit-equal, else the feature stays off.
4. **G5 (b+ only).** Masked GDN kernel on a right-padded (3, T_max) stack with `lengths` vs 3 solo calls: y of
   real tokens and final state bitwise, and the time of 1 vs 3 launches at T ≈ 377.
5. **G4 (only if 1 showed ≥ 10 %).** A trace replay of `lm_calls.jsonl` (8 episodes) with packing on/off:
   queue / TTFT / decode tok/s, each level against its own probe (machine drift ±30 %).

## Smallest viable first step, if G3 surprises

Variant (b), gated by the G1 self-test, for ≥ 3 waiting hits that are a prefix of the queue:

* each next slab is ≥ 257 tokens, so no per-row k/v is needed;
* the slabs are within 2× of each other (the head is not delayed by more than ~R × its own slab);
* b/a run per row; row slices are 64-aligned; ΣT ≤ 2048.

Tests: `check_cold` after packed ticks on `dense_fused` (CPU qmm is per row, so it is exact there), and the
E2 probe as a unit test. Without that G3 result, no code.

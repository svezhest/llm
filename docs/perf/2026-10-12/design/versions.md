# Library bump audit: mlx 0.31.2 → 0.32.3, mlx-lm 0.31.3 → 0.32.0, mlx-vlm 0.6.2 → 0.7.6

Date 2026-10-11, M1 Max 32 GB (`applegpu_g13s`), macOS 27.0. Nothing here ran on the GPU.
Everything below comes from reading release notes, merged PRs and the installed sources
(old: `src/worker/.venv`, new: `scratchpad/venv_new`). The CPU-only checks in §6 were run.
Note: mlx has no 0.31.3 release. The installed mlx is 0.31.2, and the next release
after it is 0.32.0 (2026-07-07).

## 0. TL;DR / recommendation

**Bump, but not as a drop-in, and only after the GPU microbenchmarks (§7).** With the
new libraries, worker.py imports cleanly, every lazy import resolves, and llguidance
masks are identical. But **five behaviour changes in mlx-vlm silently break the
worker** (§3.1): default thinking turns off, the thinking budget is ignored, prompt-cache
hits get wrong RoPE positions, DFlash crashes, and `apply_forced_token` is gone. Each
needs a small, known patch.

What the bump buys on M1, in order of expected value:

1. **Correct batching primitives for Qwen3.5** in mlx-vlm (`BatchGenerator`). 0.6.2 has
   three known Qwen3.5 batched-decode bugs: the ragged decode kernel requests a
   1024-thread threadgroup that M1/M2 reject (vlm#1799), a row's output depends on its
   neighbour's length (vlm#1946), and left-padding bookkeeping is silently wrong
   (vlm#2357). If the concurrency work uses mlx-vlm's batch path, it needs ≥ 0.7.4.
   If we write our own batched loop, these are the traps to avoid.
2. **`force_fused=True` for head_dim-256 prefill** (mlx#4185 + the new wn=2 D256 kernel
   in mlx#4505). By default, M1 still uses the unfused fallback for our cached prefill
   slabs. Forcing the fused kernel bounds memory, so `QWEN_PREFILL_BUDGET` and the slab
   shrink at long context could go away. Its speed against the fallback on M1 is
   unknown, so `microbench_sdpa.py` decides.
3. **Faster GDN prefill kernels.** `mx.fast.gated_delta_update` (mlx#4020) is a
   chunk-parallel simdgroup kernel benchmarked on M1 Max in its PR. mlx-lm#1559 adds a
   "packed" sequential kernel. Neither library uses either kernel for Qwen3.5 yet
   (mlx-vlm 0.7.6 still vendors the old kernel). GDN is ~22% of 9B prefill.
   `microbench_gdn.py` decides.
4. **Dequantize in fp32 inside qmm** (mlx#4241): the PR reports +8–31% prefill on M1 Max
   for bf16 models. We already cast to fp16, so expect less. qmm numerics change
   slightly; decode (qmv) is untouched.
5. **Exact ("batch-invariant") multi-row verify kernels** in mlx-vlm 0.7.x, now
   including 8-bit. These should remove the DFlash near-tie flip (CLAUDE.md Gotchas).
   Speed versus `quantized_matmul` is unknown, so run `microbench_qmm.py --vlm-verify`.

What it does *not* buy on M1:
- Decode qmv/qmm by rows is unchanged. The `get_qmv_batch_limit` table for gen-13 `s`
  is identical in both versions. `qmv_wide` (mlx#3764) and `gemv_wide` (mlx#3888) are
  gated to gen ≥ 15 for affine (M3 and later). So "M=2 ×1.17, M=4 ×1.97, M=6..32 ×2.6"
  should still hold, and `microbench_qmm.py` will confirm it.
- The two fixes that matter most for M1 small-M and SDPA landed **after** 0.32.3, on
  `main` only:
  - mlx#4629 (merged 2026-10-06): M1 Max keeps 6-row large affine matmuls on qmv, about
    12% faster at M=6.
  - mlx#4643: fixes an M1/M2 bf16 SDPA threadgroup crash that `main` itself introduced
    in mlx#4596. **0.32.3 does not have that bug.**
  - Expect both in 0.32.4. Re-run the microbenchmarks when it ships.

## 1. Pins / install facts

| package | old (worker venv) | new (scratch venv) | notes |
|---|---|---|---|
| mlx / mlx-metal | 0.31.2 (wheel `macosx_15_0`) | 0.32.3 (wheel `macosx_26_0`; 14/15/26 wheels on PyPI) | no Xcode needed for wheels |
| mlx-lm | 0.31.3 | 0.32.0 (needs `mlx>=0.32.2`, `transformers>=5.7`) | **not needed by mlx-vlm 0.7.x** |
| mlx-vlm | 0.6.2 (needs `mlx>=0.31.2`, `mlx-lm>=0.31.3`) | 0.7.6 (needs `mlx>=0.32.2`, `transformers>=5.14`, `llguidance>=1.7.0`, `mlx-audio>=0.5.2`; **no mlx-lm dependency**, vlm#1509) | vlm#2133 raised the floor to mlx 0.32.2 "for faster speculative decoding" |
| llguidance | 1.7.6 | 1.9.1 (latest) | same API, identical masks (§6) |
| transformers | 5.10.2 | 5.19.0 | |
| protobuf | 6.33.6 | 7.36.2 (unpinned, pulled in by mlx-lm) | `worker_pb2.py` (gencode 6.33.5) imports fine on 7.36.2 |
| grpcio | 1.81.0 | 1.81.0 | |

mlx-vlm 0.7.x vendors everything it used from mlx-lm: caches, gated_delta, SwitchGLU,
sampler and tool parsers. **Its cache classes are now mlx-vlm's own** (`mlx_vlm.models.cache.KVCache`
and others), not re-exports of mlx-lm's. worker.py does not import mlx_lm, so on 0.7.x the
`mlx-lm` dependency can be dropped from `src/worker/pyproject.toml`.

## 2. mlx 0.31.2 → 0.32.3

Release notes: https://github.com/ml-explore/mlx/releases (v0.32.0 2026-07-07, v0.32.1
2026-08-18, v0.32.2 2026-08-25, v0.32.3 2026-09-29).

### (a) Fused SDPA for head_dim 256, q_len > 8

The source tells a precise story (`mlx/backend/metal/scaled_dot_product_attention.cpp`, `use_fallback` + `has_fused_kernel`):

- **0.31.2**: the full (q_len > 8) kernel exists only for D ∈ {64, 80, 128}, so D=256 prefill
  always falls back to the unfused graph (scores materialised). The vector kernel (q_len ≤ 8)
  supports 256 if `q_len·gqa ≤ 32`.
- **0.32.3**: `has_fused_kernel` lists 256 for the full kernel. The *default* dispatch still
  picks the fallback for D=256, q_len > 8 except in two cases:
  - NAX GPUs (M5) with q_len ≥ 1024 and a causal or array mask (mlx#3842 v0.32.2, mlx#4416 v0.32.3).
  - **Non-NAX GPUs (our M1): fp16/bf16, `mask="causal"`, q_len ≥ 2048 and q_len == kv_len.**
    That is only a square first slab from an empty cache (mlx#4505, v0.32.3, new kernel
    `bq32 bk16 bd256 wm4 wn2`, splitting D across 2 simdgroups, Q held in registers).
  - Our chunked-prefill slabs after the first (q_len < kv_len) **still take the fallback by default**.
- **`force_fused=True`** (new kwarg, mlx#4185, v0.32.2) bypasses the heuristic and reaches
  that same D256 full kernel for any causal q_len ≤ kv_len. Neither mlx-lm 0.32.0 nor
  mlx-vlm 0.7.6 passes it, so we would have to patch it into the prefill path ourselves
  (fp16 KV only; the 27B/35B quantized KV caches don't use `mx.fast.sdpa`).
  - mlx#4185's own numbers (35B, D256, fused vs default): 32K prompt 871 vs 1006 tok/s,
    64K 625 vs 803 tok/s. Peak 39.9 vs 41.3 GB and 40.1 vs 44.2 GB.
  - That is slower but with a bounded transient, measured before mlx#4505 rewrote the kernel.
  - CLAUDE.md's "MLX `main` lists 256 in the full kernel" is now released in 0.32.3, but
    only behind `force_fused` (or square ≥ 2048 slabs). Whether it beats the fallback on
    M1 is exactly what `microbench_sdpa.py` measures.
- Decode / vector kernel: mlx#4077 (gqa-8 K/V read once) and mlx#4380/#4431 (gqa 12/16) cover
  D 64/128 only, so they don't apply to us (D256, gqa 4 for 9B, 6 for 27B, 8 for 35B).
  - Post-release mlx#4596 (TK=4 unroll, 2026-10-03) made `sdpa_vector_2pass_1` exceed the
    M1/M2 threadgroup limit for bf16/fp32 D256 with `32·gqa·q_len > 896`.
  - mlx#4643 (2026-10-09) fixed it. Neither is in 0.32.3.
  - Our fp16 path is unaffected either way. The bf16 path (`QWEN_FP16=0`) with q_len 5–8
    (DFlash verify on the 27B) would hit it on a future release without #4643.

### (b) quantized_matmul for small M (2–32)

- `get_qmv_batch_limit` for gen-13 `s` (M1 Max) is unchanged between 0.31.2 and 0.32.3:
  D,O > 4096 → 6, ≤ 4096 → 10, ≤ 2048 → 14. M ≥ limit goes to `qmm_splitk`, and that
  dispatch is unchanged too.
- New small-M kernels don't reach M1 for affine:
  - `qmv_wide` (mlx#3764, v0.32.0, M=2..limit) is gated `mode != "affine" || gen >= 15`.
  - `gemv_wide` (mlx#3888, v0.32.1) only covers fp16/bf16 GEMV on M3 and later.
- What does change on M1:
  - mlx#4241 (v0.32.1) "Dequantize in float32" in the `qmm` dequantize helper: prefill
    +8–31% on M1 Max per the PR (bf16 models). It **changes qmm rounding**, so prefill
    and batched-decode rows ≥ 6 are not bit-identical to 0.31.2. The qmv/decode path
    (`qdot`) is untouched.
  - mlx#4251 (floor fix) and mlx#4392 (non-transposed qmm dispatch) are correctness fixes.
- After 0.32.3: mlx#4629 (M1 g13s: M=6 stays on qmv, −12% at M=6).

### (c) gather_qmm / MoE (35B)

- mlx#4572 (v0.32.3) "[Metal] gather_qmm improvement" (expert-aware row-tile scheduling,
  1.2–1.4× on qwen3.5-397b prefill shapes, measured on M5 Ultra) and mlx#4567 (gather_mm
  version of the same).
- mlx#4352 / #4023 are NAX-only. mlx#3922 and #4009 fix sorted gather_qmm at > 32K rows
  and at ragged K.
- mlx#4509 (open) reports gather_qmm M=1 running at half the M=4 bandwidth: MoE decode
  still has headroom upstream, nothing landed yet.
- Expect 35B prefill to improve somewhat (MoE prefill uses sorted gather_qmm). Decode is
  roughly unchanged.

### (d) mx.fast, KV quantization, async_eval, allocator

- New `mx.fast.gated_delta_update(q, k, v, gamma, beta, initial_state=None, mask=None)`
  (mlx#4020, v0.32.3):
  - Metal sequential kernel for T < 16 (`GATED_DELTA_THRESH`), simdgroup chunk-8 kernel
    on non-NAX GPUs, chunk-16 NAX kernel on M5.
  - Supported head layouts are (Hk,Hv) ∈ {(16,16), (16,32), (16,48), (16,64), (24,24),
    (32,32)} with Dk = Dv = 128 and no mask. That covers 9B (16,32), 27B (16,48) and
    35B (16,32).
  - Semantics are those of `gated_delta_ops`: state `[B,Hv,Dv,Dk]` fp32, linear-space
    decay, no q scaling.
  - Caveat: it **casts g and beta to q's dtype** (fp16), whereas mlx-vlm keeps g in fp32.
  - Not used by mlx-lm 0.32.0 or mlx-vlm 0.7.6.
- New `mx.fast.cross_entropy` (irrelevant here).
- KV quantization: no MLX-core change. mlx-vlm's `maybe_quantize_kv_cache` keeps the
  uniform path (`to_quantized` on every KVCache with offset ≥ start), so behaviour is
  unchanged. There is a new mixed K/V-bits policy (`kv_key_bits`/`kv_value_bits`,
  HybridQuantKVCache) that we don't use.
- Threading / runtime (relevant since we run MLX from gRPC threads):
  - mlx#4043 concurrent Metal kernel-cache lookup fix
  - mlx#4248 / #3628 thread-safe compile-cache erase/cleanup
  - mlx#4413 `clear_streams` GIL deadlock fix
  - mlx#3882 reused WAR hash tables
  - mlx#4408 adaptive load I/O concurrency (faster safetensors load)
  - No API change to `async_eval`, `set_cache_limit`, `set_wired_limit`, `clear_cache`,
    or the peak/active-memory calls. Every `mx.*` symbol worker.py uses exists in 0.32.3
    (checked).
- macOS 27: mlx#4052 requests MSL 4.1 for `mx.fast.metal_kernel` JIT on macOS 27. That
  affects mlx-vlm's custom kernels (GDN, verify, ragged SDPA): same source, newer
  language version. Re-verify outputs.

### (e) OS / toolchain

- Wheels exist for macOS 14/15/26 (uv picked `macosx_26_0` here). No Xcode needed.
- Building `main` (to get #4629/#4643 early) still needs full Xcode for the `metal`
  compiler. Only CLT is installed (`/Library/Developer/CommandLineTools`).
- NAX kernels need deployment target 26.2 (mlx#3622) and never run on M1
  (`is_nax_available()` is false).

## 3. mlx-vlm 0.6.2 → 0.7.6 (what worker.py touches)

Diffs saved in `scratchpad/diffs/*.diff` (`generate_ar.py.diff`,
`models_qwen3_5_language.py.diff`, `models_cache.py.diff`, `utils.py.diff`,
`prompt_utils.py.diff`, …).

### 3.1 Breaking for worker.py (must patch before bumping)

1. **`enable_thinking` now defaults to False** in `prompt_utils.apply_chat_template`
   (vlm#1316, merged 2026-06-07, so from 0.6.3 on). When the request doesn't set it and
   the template references it, the library injects `enable_thinking=False`.
   - Measured on the four checkpoints (§6): every prompt without an explicit flag changes.
     It gains `<think>\n\n</think>\n\n`, and on the 27B the template's "Reasoning effort
     is set to xhigh…" system block disappears (−36…−40 tokens).
   - This silently breaks the worker's documented default "unset = template default =
     thinking", which is also vLLM's behaviour.
   - **Fix:** in `Engine._render`, `if "enable_thinking" in self._template_vars:
     kwargs.setdefault("enable_thinking", True)`.
   - With that, all 38 scenarios render byte-identically to 0.6.2 (`tpl_new_explicit.json`).
2. **`ThinkingBudgetCriteria`** (vlm#1912, 0.6.14):
   - `apply_forced_token(next_y)` was removed in favour of `pop_forced_token_id() -> int|None`.
     The worker's `_run` calls `thinking.apply_forced_token(next_y)`, which raises
     AttributeError on the first thinking-budget request.
   - The constructor gained `prompt_preopens_thinking=False`, and `in_thinking` now starts
     as `enable_thinking and prompt_preopens_thinking`. The worker builds it with only
     `enable_thinking=True`, so **the budget would never count** even after fixing the
     method name.
   - **Fix:** pass `prompt_preopens_thinking=True`. The worker only builds it when
     generation starts inside `<think>`.
   - **Fix:** `f = thinking.pop_forced_token_id(); if f is not None: next_y = mx.array([f], next_y.dtype)`.
3. **Text-only `get_input_embeddings` now returns `position_ids` and `rope_deltas`**
   (computed from the ids it is given, starting at 0).
   - `_run` forwards every non-None field of `emb.to_dict()` as kwargs to the prefill
     slabs and the first step.
   - `LanguageModel.__call__` slices a too-long `position_ids` as
     `[..., cache_offset : cache_offset + L]`.
   - On a **prompt-cache hit** (`input_ids = ids[reused:]`, cache offset = `reused`) that
     slices a suffix-length array at the absolute offset, which gives wrong or empty RoPE
     positions and corrupt output.
   - On a miss it happens to be consistent.
   - **Fix:** when `pixel_values is None`, drop `position_ids` and `rope_deltas` from the
     emb kwargs (the 0.6.2 behaviour; mRoPE then continues from the cache offset with the
     zero `_rope_deltas` the worker already sets).
   - Vision requests now carry explicit `position_ids`/`rope_deltas` instead of setting
     `lm._position_ids`. They are consistent at offset 0, but re-test them.
4. **DFlash (`_run_spec`)**:
   - `Qwen3_5Model.__call__` lost the `gdn_sink` kwarg, so
     `lm.model(..., gdn_sink=None)` raises TypeError.
   - `LanguageModel(..., capture_layer_ids=...)` no longer switches to verify mode or
     returns `gdn_states` (it now returns `gdn_states=None`).
   - Verify needs `speculative_verify=True`. That call returns a
     `SpeculativeCacheTransaction` in `out.gdn_states`, which must be **committed every
     round** (`mlx_vlm.speculative.cache_state.commit_speculative_round(lm, cache, state,
     accepted, bs)`), even on full acceptance, or aborted on error
     (`abort_speculative_round`).
   - `lm.rollback_speculative_cache(cache, None, …)` would fall into the legacy trim
     path and raise "ArraysCache cannot roll back a speculative block".
   - Reference loop: `mlx_vlm/speculative/dflash.py::_dflash_rounds` /
     `_dflash_verify` in 0.7.6.
   - The drafter API (`reset`, `draft_block`, `_dflash_block_total`,
     `_dflash_next_block_size`, `_speculative_walk`, `_record_speculative_round`) is
     compatible. 0.7.6 adds `draft_block_greedy` and `prepare_target_hidden` (optional).
   - With `speculative_verify=True` the verify forward runs the exact
     `Qwen3_5BatchInvariantForward`, where rows equal singleton decode. That should remove
     the near-tie flip documented in Gotchas. Re-measure DFlash speed, because the 8-bit
     exact verify kernels are new.
5. **Text-only checkpoints** (Ornith-1.5):
   - 0.7.6 drops weightless top-level modules before the strict load
     (`_drop_modules_without_weights`, vlm#1879/#1999), so `vision_tower` becomes `None`
     instead of random weights.
   - The worker's `load_weights(strict=False)` monkeypatch is now redundant but harmless.
     The `"vision_config" not in self.config` check still works because `load_config`
     doesn't add the key.
   - Image requests are already refused for text-only checkpoints in `_render`.
   - Measure peak memory: the random tower is no longer allocated.

### 3.2 Compatible (checked in source; worker imports cleanly in the new venv)

- `generate.common`: `generation_stream`, `maybe_quantize_kv_cache` (same uniform
  behaviour), `wired_limit` (byte-identical).
- `tokenizer_utils`: `BPEStreamingDetokenizer`/`_remove_space`/`make_streaming_detokenizer`.
  The only diff is `decode(..., errors="replace")`, which is what the worker's
  `_BPEDetokenizer` already does. The subclass's overridden methods still match.
- `utils`: `prepare_inputs` has the same signature. `load_config` is unchanged.
  `StoppingCriteria` still exposes `eos_token_ids` and `__call__(tok)`, and now also
  unions the config/tokenizer EOS lists itself.
- Caches: `KVCache` and `QuantizedKVCache` keep `keys`, `values` and `offset`, and
  `QuantizedKVCache(group_size, bits)` keeps its fields. `ArraysCache(size)` keeps
  `.cache`. `_copy_cache`, `_recurrent_states` and `_cache_arrays` work as written.
  New: `ArraysCache.update_recurrent/update_window/start_speculation` and
  `KVCache.prefix_cache_reserve`.
- `make_prompt_cache(lm)` → `lm.make_cache()` is unchanged (ArraysCache(2) for GDN layers, KVCache for full attention).
- `LanguageModel.__call__`:
  - `skip_logits` is unchanged. `logits_to_keep` is ignored by qwen3_5 (`generate_step`
    passes 1).
  - `_rope_deltas`, `_position_ids` and `model.fa_idx` are unchanged. `rope_deltas` is a
    new kwarg.
- `mlx_vlm.models.qwen3_5_text` and `qwen3_5_moe_text` packages exist for HF
  `model_type: qwen3_5_text` checkpoints. Ours all say `qwen3_5`/`qwen3_5_moe` at top
  level, so the VL classes still load.
- Sampling: the worker's own `make_sampler`, `_Penalties` and `_Guide` don't depend on
  mlx-vlm's sampler. mlx-vlm 0.7.x added top-nσ, p-less, typical-p and positioned
  top-k for spec decoding. That changes nothing for us.
- 35B (`qwen3_5_moe`): the old `target_verify` plumbing in the MoE block was removed
  (verify moved to `Qwen3_5BatchInvariantForward._switch_glu`), and `SwitchGLU` is
  vendored. Same routing math, plus `stop_gradient` on indices (inference no-op).

### 3.3 Changes in `generate_step` / `stream_generate` worth knowing (vs the worker's `_run`)

- `_step` passes `logits_to_keep=1`. Prefill now chunks every prompt with > 1 token,
  even ones shorter than `prefill_step_size` (vlm#2119; the worker already does this).
- `prompt_cache_checkpoint_lengths` handles multiple checkpoints during prefill (cf. our `cuts`).
- `generate_step` evals the cache state every 50 tokens (`DEFAULT_CACHE_EVAL_INTERVAL`)
  to bound the lazy metadata graph (cf. mlx-lm#1632, vlm#1984 ArraysCache per-token
  buffer leak). The worker evals `y` each step but not `c.state`. **Re-check long
  generations for growth in `get_active_memory`** after the bump.
- The thinking budget forcing is synchronisation-free (`pop_forced_token_id`, vlm#1601).
- Explicit `position_ids`/`rope_deltas` from APC take precedence over embedding-derived
  ones (see 3.1-3).

### 3.4 Continuous batching state (for the concurrency mission)

- Both 0.6.2 and 0.7.6 ship `BatchGenerator` / `GenerationBatch` / `PromptProcessingBatch`
  with Qwen3.5 support: per-row ArraysCache `left_padding`/`lengths`, a ragged left-padded
  decode-attention Metal kernel, and per-row SSM masks.
- Fixes since 0.6.2 that matter for Qwen3.5 batching:
  - vlm#1799 (0.6.12): the ragged decode kernel's hard-coded 1024-thread threadgroup is
    rejected at D=256 by older GPUs (M2 Ultra limit 896). A **probe and cache with
    fallback** was added (`_qwen3_5_try_launch`). M1 limits are lower still: mlx#4629
    reports 640/832 on M1 Max for some kernels. **0.6.2 will likely crash batched decode
    once every row passes 1024 KV tokens.**
  - vlm#1946 (0.6.15): padding was not passed to the caches, so a row's answer depended
    on its neighbour's length.
  - vlm#2357 (0.7.4): in-place `+=` on left-padding bookkeeping caused silent
    correctness issues.
  - vlm#1568 / #1582: KV quantization on multi-row batches (`BatchQuantizedKVCache`)
    and `quantized_kv_start`.
  - vlm#1598 / 0.7.x `fused_greedy_decode`: lm_head and argmax fused through the exact
    quantized-argmax kernel, so B×248K logits are never materialised (greedy rows only;
    llguidance masks supported through `prepare_next_token_mask`).
- mlx-lm 0.32.0 also has `BatchGenerator` and text `qwen3_5`/`qwen3_5_moe`/`qwen3_next`,
  with per-stream stop matchers (#1830), per-sequence samplers (#1826) and
  `generation_stream` thread-safety (#1775). mlx-vlm remains the runtime the worker
  uses (VL models).

## 4. mlx-lm 0.31.3 → 0.32.0

133 commits; compare: https://github.com/ml-explore/mlx-lm/compare/v0.31.3...v0.32.0
(there is no GitHub release page for v0.32.0, but the tag exists). Relevant entries:

- **#1559 packed gated-delta kernel** (8 value rows per SIMD-group, Dk=128 scalar gates,
  unmasked; `MLX_GDN_PACKED=0` restores the old one). It is bitwise-pinned to an
  explicit-tree comparator, *not* to the old kernel.
- **mlx-vlm 0.7.6 still vendors the 0.31.3 kernel**, so only a manual swap would use the
  packed one (see `microbench_gdn.py`).
- #1632 unbounded ArraysCache metadata graph during decode (Metal resource-limit crash on
  long generations). This is mlx-lm's cache; 0.6.2's mlx-vlm imports `ArraysCache` from
  mlx-lm, so **the worker today runs the unfixed class**. The worker never sets
  `left_padding`/`lengths` (single request), so it is probably not exposed.
- #1778 the cache `state` returns the full state; #1848 BatchKVCache offset is rebound,
  not mutated; #1824 left padding is read without a reduction kernel.
- #1345 / #1801 pipeline parallelism (distributed), irrelevant.
- #1825 / #1912 top-p mask fixes and #1827 named greedy sampler, irrelevant (own sampler).
- #1777 prefilled prompt tokens are included in the logits-processor history (cf. our
  repetition penalty over the prompt, already done).

## 5. llguidance / grpc / protobuf / transformers

- llguidance 1.7.6 → 1.9.1: every API the worker uses exists (`LLMatcher.grammar_from_json_schema/
  grammar_from_lark/validate_grammar_with_warnings`, `is_error/get_error/consume_token`,
  `llguidance.numpy.allocate/fill_next_token_bitmask`, `llguidance.mlx.apply_token_bitmask`,
  `llguidance.hf.from_tokenizer`).
  - JSON-schema masks after 4 prefixes are identical bit for bit across the two venvs
    (`design/llg_parity.py`, Ornith-1.5 tokenizer).
- protobuf 7.36.2 runtime accepts the 6.33.5 gencode. Pin `protobuf<7` only if you want
  zero risk; regenerate with `gen_proto.sh` after bumping grpcio-tools.
- transformers 5.19 (required ≥ 5.14 by mlx-vlm 0.7.6). Processor/tokenizer rendering is
  checked by §6.

## 6. CPU-only checks already run (no GPU, no weights loaded)

| check | result |
|---|---|
| `venv_new` install + `import mlx.core` | mlx 0.32.3, mlx-lm 0.32.0, mlx-vlm 0.7.6 OK |
| `import worker` + all its lazy `mlx_vlm` imports (both venvs) | OK; 0.7.6 lacks `ThinkingBudgetCriteria.apply_forced_token` |
| `design/template_parity.py` (worker's `_render` path, 10 scenarios × 4 checkpoints) | 0.7.6 vs 0.6.2: **31 of 38 differ** (enable_thinking default; the 7 equal ones set enable_thinking=false or add_generation_prompt=false); with `--explicit-thinking`: **0 of 38 differ** |
| `design/llg_parity.py` | identical masks (llguidance 1.7.6/tf 5.10.2 vs 1.9.1/tf 5.19.0) |

Results: `scratchpad/results/tpl_old.json`, `tpl_new.json`, `tpl_new_explicit.json`.

## 7. GPU experiments (prepared, NOT run; need the GPU lease, nothing else on the GPU)

Scripts (each prints a markdown table, takes `--label` and `--json`; random weights, no
checkpoint load, only `config.json` is read):

- `design/microbench_sdpa.py`: `mx.fast.scaled_dot_product_attention` at the 9B's
  Hq16/Hkv4/D256, fp16. Settings:
  - q_len 512/2048 × kv_len 8K/32K/120K, causal, model-like layouts.
  - Modes: default dispatch, `force_fused=True` (new venv only), and the explicit
    unfused graph as reference/maxdiff.
  - `--decode` adds q_len 1–16 and batched decode B=2/4/8.
  - Rows whose unfused transient would exceed `--max-score-gb` (12) are skipped: 2048 ×
    120K default/unfused needs ~30 GB. `--allow-big` overrides this, but don't use it on
    32 GB.
- `design/microbench_qmm.py`: affine `quantized_matmul` per linear of the checkpoint
  (9B: 8-bit g64, K=4096 rows; `--config weights/qwen38-27b-mlx-5bit` gives the 27B's
  5120×17408 / 5-bit) for M = 1,2,3,4,5,6,7,8,12,16,32.
  - Prints per-shape cost (×M=1), a whole-step linears estimate, and the aggregate tok/s
    ceiling per M.
  - `--vlm-verify` (new venv) adds mlx-vlm 0.7's exact verify kernel on the same shapes.
- `design/microbench_gdn.py`: GDN recurrence. Compares `vlm` (current kernel), `lm`
  (mlx-lm; packed in 0.32.0) and `fast` (`mx.fast.gated_delta_update`, mlx 0.32.3) at
  B×T = 1×512, 1×2048, 1×1, 2×1, 4×1, 8×1, with max-diff against `vlm`.

Exact commands (from the repo root; ~1–3 min each; the old venv is the worker's):

```zsh
S=<scratchpad>
OLD=/Users/user/Projects/llm/src/worker/.venv/bin/python
NEW=$S/venv_new/bin/python
cd $S/results
# SDPA (prefill slabs + decode/verify/batched decode)
$OLD -I $S/design/microbench_sdpa.py --label old-mlx-0.31.2 --decode --json sdpa_old.json | tee sdpa_old.md
$NEW -I $S/design/microbench_sdpa.py --label new-mlx-0.32.3 --decode --json sdpa_new.json | tee sdpa_new.md
# quantized_matmul vs rows (9B 8-bit), then the 27B 5-bit shapes
$OLD -I $S/design/microbench_qmm.py --label old-mlx-0.31.2 --json qmm_old.json | tee qmm_old.md
$NEW -I $S/design/microbench_qmm.py --label new-mlx-0.32.3 --vlm-verify --json qmm_new.json | tee qmm_new.md
$OLD -I $S/design/microbench_qmm.py --label old-27b --config /Users/user/Projects/llm/weights/qwen38-27b-mlx-5bit --rows 1,2,4,6,8,16 --json qmm27_old.json | tee qmm27_old.md
$NEW -I $S/design/microbench_qmm.py --label new-27b --config /Users/user/Projects/llm/weights/qwen38-27b-mlx-5bit --rows 1,2,4,6,8,16 --vlm-verify --json qmm27_new.json | tee qmm27_new.md
# GDN kernels (new venv has all three; MLX_GDN_PACKED=0 shows mlx-lm's unpacked kernel)
$OLD -I $S/design/microbench_gdn.py --label old --json gdn_old.json | tee gdn_old.md
$NEW -I $S/design/microbench_gdn.py --label new --json gdn_new.json | tee gdn_new.md
MLX_GDN_PACKED=0 $NEW -I $S/design/microbench_gdn.py --label new-unpacked | tee gdn_new_unpacked.md
# optional: bf16 SDPA (exercises the M1 threadgroup limits)
$NEW -I $S/design/microbench_sdpa.py --label new-bf16 --dtype bfloat16 --decode --q-lens 512 --kv-lens 8192 | tee sdpa_new_bf16.md
```

How to read them:

- **SDPA:** if `fused` beats `default` at q_len 512–2048 × kv ≥ 32K (or is equal), patch
  `force_fused=True` into the fp16-KV prefill path. Then re-measure 32K/120K prefill
  with `QWEN_PREFILL_BUDGET` raised, since the slab no longer has to shrink. Check
  `maxdiff` ≲ 1e-2 for fp16.
- **qmm:**
  - The ×M columns should reproduce CLAUDE.md's 1.17/1.97/2.6 on both versions.
  - A visibly cheaper M ≥ 6 row on the new venv would be the fp32-dequant effect.
  - The whole-step table gives the batched-decode linears ceiling (aggregate tok/s by M).
  - The `vlm exact verify` rows show whether 0.7's verify path is slower than
    `quantized_matmul`, which is the cost of DFlash parity.
- **GDN:**
  - `fast` / `lm` vs `vlm` at T=2048 decides whether a swap is worth it. Each row prints
    the ms for all GDN layers.
  - dy/dstate show the numeric drift. `fast` rounds g to fp16, so expect a larger dy.

## 8. Bump procedure (when the microbenchmarks justify it)

```zsh
cd /Users/user/Projects/llm/src/worker
# pyproject: mlx-vlm>=0.7.6, mlx>=0.32.3; mlx-lm no longer needed by mlx-vlm
uv add "mlx==0.32.3" "mlx-vlm==0.7.6" "llguidance>=1.9.1"
uv remove mlx-lm            # or: uv add "mlx-lm==0.32.0" to keep it for tooling
uv sync
uv run --group proto bash gen_proto.sh   # regenerate gencode against the new protobuf
```

(The root `pyproject.toml`/`uv.lock` is the separate tooling env and doesn't need to change.)

worker.py patches (§3.1), all small:

1. `_render`: `if "enable_thinking" in self._template_vars: kwargs.setdefault("enable_thinking", True)`.
2. `ThinkingBudgetCriteria(..., prompt_preopens_thinking=True)`, and replace
   `apply_forced_token` with `pop_forced_token_id`.
3. `_run`: don't forward `position_ids`/`rope_deltas` from `get_input_embeddings` for
   text-only requests.
4. `_run_spec`:
   - Drop `gdn_sink=None`.
   - Call verify with `speculative_verify=True`.
   - `commit_speculative_round(lm, prompt_cache, out.gdn_states, accepted, bs)` every round.
   - `abort_speculative_round` in an `except`.
   - Update the docstring ("flips into verify kernels when capture_layer_ids is set" no
     longer holds).
5. Optional: remove the `load_weights(strict=False)` monkeypatch (0.7.6 drops the
   weightless vision tower itself).

Re-verify after bumping (GPU):

- **Greedy byte-identity is NOT expected.**
  - qmm (prefill, M ≥ 6) dequantizes in fp32 now (mlx#4241).
  - Custom kernels compile as MSL 4.1 on macOS 27 (mlx#4052).
  - Instead, redo the teacher-forced KL / top-1 check from CLAUDE.md on 3.4K tokens (9B,
    27B, 35B): KL vs 0.31.2 should be ≪ the bf16→fp16 0.0003.
- Prompt cache:
  - Hit vs miss byte-identity on the 7.5K-system-prompt test.
  - A vision request followed by a cached text request (exercises fix 3).
- Thinking:
  - Default request still thinks.
  - `enable_thinking:false` doesn't.
  - `thinking_budget` forces `</think>` at the budget.
- DFlash:
  - Output vs plain greedy (should now be exactly equal).
  - tok/s on the 5 coding tasks vs the 47–53 tok/s baseline.
- Long generation (≥ 8K tokens): `get_active_memory` flat, no Metal resource-limit error.
- Ornith-1.5 load: peak memory (vision tower no longer allocated). Image request → 400.
- 35B prefill 8K/32K (gather_qmm changes) and 56K × 2 OOM test with the wired-limit recipe.
- Full gateway parity suite (tools, response_format, stop strings).

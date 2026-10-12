# Review round 3: mrqmm, step pipeline + grouped sampling, mlx-vlm 0.7.6 bump patches

Read-only, 2026-10-12. Nothing was run: no pytest, no model, no builds.

**What I read**
* `src/worker/worker.py`: 3198 lines, mtime 08:09. I took a snapshot in `scratchpad/r3snap/`. It did not change during the review.
* `src/worker/mrqmm.py`: unchanged since 02:04.
* The tests: `tests/test_mrqmm.py`, `test_bump.py`, `test_scheduler.py`.
* The installed mlx-vlm 0.7.6 source: `qwen3_5/{language,qwen3_5,gated_delta,speculative_verifier}.py`, `qwen3_5_moe/language.py`, `speculative/{cache_state,common,dflash}.py`, `speculative/ops/linear.py`, `models/quantized_verifier.py`, `models/{base,cache}.py`, `utils.py::ThinkingBudgetCriteria`, `prompt_utils.py`, `drafters/qwen3_dflash`.
* `mlx/core/fast.pyi` from mlx 0.32.3.
* The worker logs in `bump/gpu/new` and `results/worker_bump.log`.

I did not repeat the items from review_worker.md or review_worker_r2.md.

**Overall.** I found no correctness bug in a token stream: no High or Medium findings. Every library name and signature the patches override or call exists exactly as patched in 0.7.6. The remaining findings are robustness and performance edges, test gaps and doc drift.

## Ranked findings

### L1. Low (perf, once per process). The fused GDN kernels are not compiled at load (worker.py:1404-1444, 2547-2551; mrqmm.py:374-388)
* **Cause.**
  * `mrqmm.install` → `_warmup` compiles kernels for the covered module shapes and for mrqmm's own fused groups. On 0.7.6 those groups are empty, because `_patch_fused` finds no `_decode_quantized_linears_fused` in `qwen3_5.language`.
  * `_install_gdn_fuse` then builds new concatenated shapes: K=4096, N=12352 on the 9B and K=2048, N=12352 on the 35B. They are routed to `mr.mrqmm`, but nothing compiles them.
* **Effect.** The first decode step with 3–8 rows, and the first with 9–16, each compile a Metal kernel (~60 ms each per mrqmm.md) on the scheduler thread. Every row stalls once.
* **Fix.** At the end of `_install_gdn_fuse`, for each distinct (K, N) with `use_mr`:
  `mx.eval([mr.mrqmm(mx.zeros((r, 1, K), mx.float16), w, s, b) for r in (mr.MIN_ROWS, mr.MAX_ROWS)])`.
  Alternatively, expose `mrqmm.warm(K, N)`.

### L2. Low (scope and docs). A failure in `sample_rows` fails every grouped row of the step, not just "that group" (worker.py:2316-2322)
* `_step` wraps the whole `sample_rows(...)` call in one `try`. If it raises, every pending row fails: all groups, and rows of groups already drawn before the failing group, whose keys have already been split.
* step_profile.md says: "an exception while drawing fails the rows of that group, not a single row".
* This is harmless in practice: a graph-build error in the filters is next to impossible.
* **Fix.** Either catch per group inside `sample_rows` (return None for that group's rows, or raise with the indices), or correct the wording to "the non-grammar rows of the step".

### L3. Low (coverage). The CPU suite never runs the scheduler with `QWEN_GDN_FUSE` or `_install_fast_paths` installed (tests/conftest.py:55-60)
* **What conftest does.** It installs `_install_rows_attention` only.
* **Where GDN_FUSE is tested.** Only by `test_mrqmm.py::test_worker_gdn_fuse`: one decode step on the GPU and a 9-token prefill.
* **What is never tested with fused modules:**
  * DFlash verify over row-view weights;
  * the prompt cache;
  * sync / grammar mode;
  * vision rows;
  * batched == solo.
* **Why that matters.** These paths are what the production default runs (`QWEN_GDN_FUSE=1`). They are covered only empirically by the GPU corpus runs (the `bump/gpu/new/*` and `bump/gpu/dflash/*` logs show the fuse active).
* **Fix.** Add a `dense_fused` fixture: the tiny model with fp16 scales and biases, plus `_install_gdn_fuse`. Run `test_batched_tokens_equal_solo`, `test_dflash_exact_and_demoted` and `test_prompt_cache_while_batching` on it too. `quantized_matmul` works on the CPU.

### L4. Low (coverage). The "a stack gives each row bit-for-bit its solo result" claim is tested only on the CPU at V=512 (test_scheduler.py:413-443)
* At V=512, Metal would use the single-block sort / softmax kernels.
* The real vocabulary (248K) uses the multi-block merge sort and the looped softmax and logsumexp kernels. The bit-exactness of those is verified only by `scratchpad/prof/{batch_exact,sample_rows_check}.py`.
* **Fix.** Add a Metal-marked test (like test_mrqmm) that runs `test_sample_rows_equal_per_row` at V=248320 with ties injected. This locks in the claim against future MLX kernel changes.

### L5. Low (perf; the measurements conflict). The fused-SDPA gate at exactly 32K keys (worker.py:1305, 1282-1289)
* The gate turns on at q ≥ 512 and kv ≥ 32K. At the low end of that range the two measurements disagree:
  * microbench.md §2: 2048 × 32K ×0.88 / 0.89;
  * `bump/speed/sdpa_probe.txt`: ×1.06;
  * 512 × 32K: ×1.01 / 1.09 (slower).
* The worker comment quotes ×0.88; CLAUDE.md:140 and BACKLOG:151 quote ×1.06.
* End-to-end ≤ 32K was measured as unchanged, so this is a tuning point only.
* **Fix.** If it is re-measured, consider kv ≥ 48K, and make the comment and the docs quote the same number.

## Nits

* **Dead code in mrqmm.py on mlx-vlm 0.7.**
  * `_patch_fused`, `_fused_decode`, `_gdn_groups`, `_FUSED` and the install log's "+ fused GDN in-projection" are dead code. The module and `install` docstrings still describe routing mlx-vlm's fused GDN in-projection.
  * The live fusion is now `worker._install_gdn_fuse`.
  * If a future mlx-vlm re-adds the hook to `qwen3_5.language`, both layers stay safe: `_gdn_call` hands the library `_Given` objects, so the old fused path would return None and call them.
  * Still, delete the dead code, or point the docstring at the worker.
* **`mrqmm()` cache key (mrqmm.py:233).** The key `(x.shape, x.dtype, N, cfg)` does not include the weight's K. A cache hit skips the `K == w.shape[1]*4` assert, so a direct caller passing another weight with the same N and the wrong K would read out of bounds. `MRQuantizedLinear` always passes its own weight, so this would take a model-shape bug that qmm would have raised on. Fix: check K outside the cached branch (cheap), or put `w.shape` in the key.
* **Asserts as guards.** The asserts in `mrqmm()` (rows, dtype, K) disappear under `python -O`. The module path is protected by `_use` and `supported` anyway.
* **`QWEN_GDN_FUSE` under `QWEN_FP16=0`.** It silently fuses nothing (fp16 scales required; the log says "0 layers"), and neither the docs nor the env tables say so. The same holds for mrqmm, which is documented.
* **`QWEN_VERIFY_EXACT_MAX_T` parsing.** `int(os.environ.get(..., "8"))` raises at import on an empty value (`QWEN_VERIFY_EXACT_MAX_T=`). Other knobs use `or`.
* **`_gdn_call` mutates the module dict for the length of the call** (the `_Given` swap). This is safe only because the scheduler thread is the sole caller and no handler walks the module tree after load (`named_modules` / `parameters` run at load only). The comment says so; just keep it true.
* **The mrqmm.md "Paths that can never take it" list** includes "QWEN_FP16=0 … nothing swapped". That is right for mrqmm, but `_install_gdn_fuse` also skips bf16; worth one line next to it.

## Docs vs code: mismatches

| # | where | says | code / fact |
|---|---|---|---|
| 1 | CLAUDE.md:336 (A2 bullet) | vision slabs pass `_position_ids[:, :, pos:pos+n]` | 0.7 no longer sets `_position_ids`; `_prefill` slices the `position_ids` that `get_input_embeddings` returns (worker.py:2006). The paragraph at CLAUDE.md:259-263 already says this. |
| 2 | CLAUDE.md:463, README:402, worker.py:1303 | `QWEN_VERIFY_EXACT_MAX_T` "16 = always" | Only for the adaptive block (the drafter's `block_size` 16). With `QWEN_DRAFT_BLOCK` > 16 (CLAUDE mentions fixed 32), longer blocks still go to `quantized_matmul`. |
| 3 | worker.py:1287 vs CLAUDE.md:140 / BACKLOG:151 | fused SDPA ×0.88 vs ×1.06 at 2048 × 32K | Two runs disagree (L5). The worker comment also says peak "−1.5..−1.8 GB" (microbench), while README and CLAUDE say "−1.3 GB" (end-to-end). |
| 4 | results/step_profile.md (drawing errors) | fail "the rows of that group" | All non-grammar rows of the step (L2). |
| 5 | CLAUDE.md:193 | "`_gdn_call` runs one qmv (mrqmm for 3..16 rows…)" | With `QWEN_MRQMM=0`, with non-8-bit models (27B), or at B ≥ ~6 it is one `quantized_matmul` (qmm), not a qmv. Wording only. |
| 6 | mrqmm.py:1-37, 329-334 docstrings | install routes mlx-vlm's fused GDN in-projection | Dead on 0.7 (see Nits). |

Consistent with the code:
* `QWEN_MRQMM` (1; `!= "0"`, read by `mrqmm.enabled()` at install);
* `QWEN_GDN_FUSE` (1, `!= "0"`);
* `QWEN_SDPA_FUSED` (1, `!= "0"`; gate q ≥ 512, kv ≥ 32768, KVCache, fp16, causal, no sinks);
* `QWEN_GDN_FAST` (0, `== "1"`; ≥ 16 tokens, no mask, scalar g);
* `QWEN_VERIFY_EXACT_MAX_T` (8; `x.shape[1] > cap` → `linear(x)`, so 0 = never exact);
* the README and CLAUDE env tables;
* the 3..16 rows / N ≥ 512 / 8-bit g64 fp16 coverage text;
* "B ≤ 2 bit-exact";
* the B=1 path and the thinking-budget notes;
* "MoE router / experts / shared expert / lm_head not wrapped", which matches the 35B load log: 130 modules; skipped are N%8 x40 (shared_expert_gate), N<512 x100 (router + in_proj_b/a), 4-bit x120, and the 6-bit lm_head.

## Verified OK

**A. mrqmm.py kernel and install**

* **Fragment layout.** The 8×8 simdgroup fragment layout (`fm = (qid&4) + (lid/2)%4`, `fn = (qid&2)*2 + (lid%2)*2`) is MLX's.
* **A operand.**
  * Each thread reads the uint4 at bytes [8fn, 8fn+16) of each 64-byte group of row `n0+8i+fm`.
  * Logical k fn and fn+1 of step st are physical k 8fn+st and 8fn+8+st, i.e. byte st%4 of word st/4 and of word 2+st/4.
  * The mask/shift/`insert_bits` selection is correct in all 4 cases (st%2 picks bytes 0/2 vs 1/3, (st/2)%2 picks the low or high half).
  * The packing order matches MLX 8-bit affine: byte k%4 of uint32 k/4, row stride K bytes.
  * `(0x6400|q) − 1024` is exact. `fma(q, s, b)` is MLX's `w = s·q + b`.
* **B operand.** Logical k fm → physical 8fm+st, which is half st%2 of word st/2 of uint4 `8g+fm` of the row. Pairing rows m and m+1 into `thread_elements` matches the fragment layout.
* **Scales and biases.** `scales[n, g]` is indexed as `sp = scales + r*G`.
* **Tails.**
  * N % 8 == 0 makes each fragment wholly inside or outside N. Inactive fragments read row `fm` (valid) and are never stored.
  * The last tile is partial through `act[]`. K is any multiple of 64: groups are strided by KS, every simdgroup's partial sum is added once, and `_config` only shrinks KS to divide G.
  * Row padding: rows ≥ M0 read row 0 and are never stored (`m < M0`, `m+1 < M0`).
  * Output indexing `m*N+n` matches the contiguous `(*x.shape[:-1], N)` output.
* **Launch and threadgroup memory.**
  * `grid = 32·KS·tiles`, threadgroup `32·KS` (≤ 256), so `threadgroup_position_in_grid.x` is the tile.
  * The reduction scratch is ≤ 6 KB.
  * The early `return` of ks > 0 comes after the only barrier.
* **Inputs.**
  * x goes through `ensure_row_contiguous`. A row-contiguous view starts on a row, and K % 64 keeps the 16-byte loads aligned. Weight views from `_install_gdn_fuse` (offset lo·K bytes) stay aligned too.
  * The buffer offset is applied by MLX's custom-kernel input binding.
* **Dtypes.** fp16 only:
  * `_use` requires fp16 x;
  * `supported` requires fp16 scales and biases, so `QWEN_FP16=0` swaps nothing;
  * `mrqmm()` asserts on direct bf16 calls;
  * the output dtype is fp16, the same as qmm on fp16 x.
* **Compile cache.** `_KERNELS` and `_CALLS` misses are filled under `_LOCK`, and a double miss reuses the compiled kernel. `get` without the lock is GIL-atomic. Warm-up runs on the main thread before the scheduler starts.
* **install / uninstall.** `install` is idempotent: `_module_ok` accepts both classes, `_patch_fused` hooks once, flags are re-set. `uninstall` restores the classes and flags.
* **Which modules on the 35B.**
  * `QuantizedSwitchLinear` is an `nn.Module`, not a `QuantizedLinear`, so the experts are untouched.
  * The router (N=256) and in_proj_b/a (N=32) are below MIN_N; shared_expert_gate (N=1) fails N%8.
  * The shared expert is 4-bit; lm_head is 6-bit.
* **Paths with 3..16 rows other than batched decode.** None reach the kernel; `_use` needs `shape[-2] == 1`:
  * prefill slabs are (1, L, K);
  * the DFlash verify is (1, T, K) with T ≥ 2 (`bs ≤ 1` breaks out);
  * the drafter uses `draft_block`, i.e. `self.lm_head(draft_hidden[:, 1:])` = (1, bs−1, K). The worker does not call `draft_block_greedy`, so the target's `speculative_argmax_from_hidden` exact head is never used;
  * the verifier's timewise and singleton fallbacks have 1 row;
  * the vision tower and the drafter are not passed to `install` or the fuse.

**A'. `QWEN_GDN_FUSE` (`_gdn_call`, `_install_gdn_fuse`) against 0.7.6 `Qwen3_5GatedDeltaNet.__call__`**

* **How the library calls the projections.** It calls `self.in_proj_qkv(inputs)`, `self.in_proj_z(inputs)` and `_project_gates` → `self.in_proj_b/a(inputs)`, and nothing else touches those modules. `nn.Module.__getattr__` resolves through the dict, so `_Given` is returned.
* **Restore.** The modules are restored in `finally`.
* **Split.** The split indices are cumulative row ends.
* **Speculative paths.** The decoder layers call `linear_attn(x, mask, cache)` positionally, which matches. The verifier's `_gated_delta` calls `_linears` on the real modules, never `__call__`, so speculative verify is untouched.
* **Gate conditions.** S == 1 and fp16 only; a 1-token prefill slab takes it as well (same qmv).
* **The 35B.** All four projections are 8-bit g64 (config), so it fuses: N = 12352, K = 2048.
* **The 27B.** 5-bit: it fuses through `quantized_matmul`; `use_mr` is false.
* **Memory.** The views share the concatenated buffers. The old arrays are freed: only the modules referenced them, and the mrqmm warm-up dict is cleared. `_fused_in` lives in the instance `__dict__`, not in the parameter tree, and the `mx.clear_cache()` runs before the budget is measured.
* **Bit-identity at B ≤ 2.** qmv's per-row reduction does not depend on N; the tiny-model GPU test checks it.
* **DFlash after the fuse.** The DFlash corpus runs were made with the fuse active (logs), so verify over row views is exercised.

**B. Step pipeline and grouped sampling**

* **`r.y.item()` per row.** Each `r.y` is an output of the previous step's `async_eval`: a grouped-greedy `toks[j:j+1]` slice, a categorical draw, or a fresh constant for a forced, prompt or demoted token. So `item()` waits on that event and queues no op. `live` is computed after `_step`, so rows ended by `_regrow` or a sampling failure are skipped.
* **Per-row state outside the group draw.** Penalties (`pen.add` of the fed token unless `first`), `_Control` counting and masks, and `quantize` all run per row in `_logits` before grouping. Grammar rows never enter `pend`; they keep `_sample` with the guide.
* **The group key** is the full filter tuple (temperature, min_p, top_k, top_p) or "greedy", plus the logits dtype. So fp32 penalized logits and fp16 plain logits never share a greedy stack.
* **Filter ops.** Only row ops are used:
  * elementwise scale;
  * softmax and max along −1;
  * topk values then min (a set, not an order);
  * argsort, take/put_along_axis and cumsum along −1;
  * `drop[..., -1] = False` on every row.

  No reduction runs across rows. The greedy path is argmax(x − logsumexp) per row, with first-index ties as in the solo row.
* **Draws.** Each row draws from its own (1, V) slice with its own key, split once, in index order, which matches its solo `__call__`. Singletons go through `s0(logits)` unchanged.
* **Thread.** `make_sampler` and `mx.random.key` run on the scheduler thread (`_admit`).

**C. Bump patches against the 0.7.6 source**

1. **`_render`.**
   * `prompt_utils.apply_chat_template` injects `enable_thinking=False` whenever the template processor accepts `**kwargs`. The setdefault runs after the vLLM-style filter, only when the template declares the variable, and the request's own value wins.
   * No local template references `thinking_mode`, the other kwarg mlx-vlm injects when `enable_thinking is True`. Check and Generate share `_render`.
2. **Thinking budget.**
   * `ThinkingBudgetCriteria(..., prompt_preopens_thinking=True)` exists (utils.py:2711). `enable_thinking=True` is required for `pop_forced_token_id`, and it is passed.
   * `__call__` sets or clears `forced_token_id` each token; the `\n` then `</think>` sequence comes out over two steps.
   * The forced id replaces the sampled `y` after the step, so no mask is applied to it. In sync (grammar) mode `_Guide.consume` ignores the `\n` while inactive and activates on `</think>`. The dtype is preserved.
3. **Text positions.**
   * `get_input_embeddings` returns `InputEmbeddingsFeatures(inputs_embeds, position_ids, rope_deltas)`; the text path is qwen3_5.py:63-71.
   * Dropping them gives `LanguageModel.__call__` `position_ids=None`:
     * first slab at offset 0 → `get_rope_index` (and zero `_rope_deltas`);
     * later slabs → the delta branch (offset + 0);
     * a prompt-cache hit → `_rope_deltas` zeros and `_position_ids` None, set by `_prompt_cache` → the delta branch.

     These are exactly 0.6.2's semantics.
   * Vision: no `pixel_values` reaches `lm()`, so the model state is not reset. Explicit `position_ids` skip the inference branch, `[..., pos:pos+n]` matches the slab, and the delta comes from the returned `rope_deltas`.
4. **DFlash.**
   * `speculative_verify=True` → `_EXACT_SPECULATIVE_VERIFIER.verify`, which wraps `start_speculative_cache` and aborts itself on a `BaseException` inside the forward. `out.gdn_states` is the `SpeculativeCacheTransaction`.
   * `commit_speculative_round(lm, cache, txn, accepted, bs)` → `txn.commit([accepted+1])`. `abort_speculative_round(None)` is a no-op (the forward raised and already aborted), and `abort()` after a failed commit is still active and restores.
   * Every `yield` and the cancel checks are outside the `try`, after the commit. A `GeneratorExit` from `close()` or a cancel therefore never finds an open transaction.
   * The legacy `rollback_speculative_cache` branch is unreachable: the state is always a transaction.
5. **`_rows_attention`.**
   * `_prepare_projected_qkv(q_out, keys, values, cache, position_ids, position_embeddings, mask)` matches positionally. With `cache=None` there is no update or offset read, and with `mask=None` no slicing; it returns 5 values.
   * The library `__call__` has no `**kw`; the decoder layers pass only mask, cache, position_ids and position_embeddings, so forwarding `**kw` is safe.
   * The MoE imports the same `Qwen3_5Attention` and `Qwen3_5GatedDeltaNet` classes.
   * `RowsKV` has no `left_padding`, `make_mask` or `_idx`: `_create_qwen3_5_attention_mask` returns None for L=1, `_is_single_row_batch_cache` is false, and the left-padding branch is skipped.
6. **`_fused_sdpa`.**
   * The signature equals `base.scaled_dot_product_attention`.
   * `type(cache).__name__ == "KVCache"` and `isinstance(keys, mx.array)` now come before any `keys.shape`. Quantized caches (tuples) and their fp16 warm-up below 5000 tokens go to the library; the regression test covers the tuple case.
   * `KVCache.make_mask` gives `"causal"` for prefill slabs; MLX's causal mode is lower-right aligned, as the library's.
   * `force_fused` exists in mlx 0.32.3 (`fast.pyi:101`) and is documented to raise when no fused kernel exists. That error is caught at graph build.
   * Every caller resolves the module global at call time: `Qwen3_5Attention.__call__`, `_qwen3_5_left_padded_attention` and `_rows_attention` through `_Q35`. The verifier imports `base`'s function directly, so it stays on the library: correct for small T.
7. **`QWEN_GDN_FAST`.**
   * The `gated_delta_kernel(q, k, v, g, beta, state, mask)` signature matches. It is resolved inside `gated_delta.py` at call time, and no other module imports it.
   * Speculative histories go through `gated_delta_update_with_states`. `update_recurrent` passes `state_steps` only when `history_capacity` is set, so they never reach the override.
   * Off by default.
8. **`QWEN_VERIFY_EXACT_MAX_T`.**
   * `Qwen3_5BatchInvariantForward._linear` / `_linears` exist. They are patched on the class, so the module-level `_EXACT_SPECULATIVE_VERIFIER` instance picks them up.
   * Every caller passes a 3D (B, T, ·) input, so `x.shape[1]` is T: attention q/k/v/o, the MLP, the GDN in- and out-projections, lm_head.
   * The library's fused `optimized_affine_linears` already stops at T ≤ 8; the per-linear `optimized_affine_linear` has no cap. The override caps both.
   * `_target_verify_linears`' T == 1 branch, which would cache a second fused weight copy via setattr, is unreachable because bs ≥ 2.
9. **Text-only load.** The 08:07 log shows "Checkpoint has no weights for module(s): vision_tower. Disabling them.", the strict load is fine, and images are refused in `_render`.

**Order of installation.** `_install_fast_paths` runs before `_install_rows_attention`, `mrqmm.install` and `_install_gdn_fuse`, all after the fp16 cast and the drafter load, and before the budget is measured. No library module caches a reference to an overridden global (grep over mlx_vlm).

# Library bump: mlx 0.31.2 → 0.32.3, mlx-vlm 0.6.2 → 0.7.6 (mlx-lm dropped) — how to apply

Prepared 2026-10-12 without the GPU. Everything here was checked on the CPU only
(tiny random models, real tokenizer/template/processor files); the GPU list in §4 is
what still has to run. Background: `design/versions.md` §3.1/§8, `results/microbench.md` §6(a).

```zsh
S=<scratchpad>
B=$S/bump
REPO=/Users/user/Projects/llm
W=(env PYTHONPATH=. .venv/bin/python $S/gpu_tests/wtest.py)       # gRPC test client, run from src/worker
NINE=(QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0)
```

Files:

| file | what |
|---|---|
| `$B/worker.patch` | the worker.py change (unified diff, `-p1` from the repo root), made against `$B/worker.base.py` = the repo's worker.py when the patch was last refreshed |
| `$B/worker.py` | the patched worker.py (base + patch) |
| `$B/tests.patch` | test changes for 0.7.6 (+ new `tests/test_bump.py`) |
| `$B/project/pyproject.toml`, `$B/project/uv.lock` | src/worker's pyproject/lock after the `uv add`/`uv remove` below (resolved 2026-10-12) |
| `$B/checks/` | CPU checks: `render_parity.py` (prompts via the real `Engine._render`, old vs new), `textonly_load.py` (strict load of a text-only checkpoint), `symbols.py` (every library symbol/signature the worker uses) |
| `$B/gpu/stack_kl.py` | GPU: teacher-forced top-K dump (old stack) → KL / top-1 vs the new stack (or between settings) |

The patch is for the new stack only: it does not run on mlx-vlm 0.6.2 (`prompt_preopens_thinking`,
`pop_forced_token_id`, `speculative_verify`, `_prepare_projected_qkv`, `mlx_vlm.speculative.cache_state`).

## 0. Which project

The worker runs from **`src/worker/.venv`** (its own uv project: `src/worker/pyproject.toml` +
`src/worker/uv.lock`; the running worker is `src/worker/.venv/bin/python3 worker.py`). The root
`pyproject.toml`/`.venv` is the tooling env (chat.py, hf download) and is **not** touched; bump it
separately later if wanted (it does not need to match).

Dependency changes (src/worker):
- `mlx==0.32.3` (mlx-metal 0.32.3 follows), `mlx-vlm==0.7.6`.
- `mlx-lm` removed: the worker never imports it, and mlx-vlm 0.7 vendors what it used (no `mlx_lm`
  import anywhere in mlx_vlm 0.7.6).
- `transformers>=5.14` (mlx-vlm 0.7.6 requires ≥ 5.14.0; resolves to 5.19.0), `llguidance>=1.9.1`
  (mlx-vlm wants ≥ 1.7.0; masks identical to 1.7.6, versions.md §5).
- **`protobuf>=6.33.5,<7` added explicitly**: today protobuf only arrives through mlx-lm (and
  grpcio-tools in the `proto` group). Without mlx-lm the runtime would have no protobuf, and
  `worker_pb2.py` (gencode 6.33.5) needs ≥ 6.33.5. `<7` keeps the current major (6.33.6); 7.36.2
  also imports the gencode fine (checked), drop the cap if you prefer.
- Pulled in by mlx-vlm 0.7.6 (new): mlx-audio 0.5.8 (was 0.4.4), opencv-python, fastapi/uvicorn,
  cryptography, huggingface-hub 2.2.0 (was 1.18.0), safetensors 0.8.0. Gone: mlx-lm, datasets,
  pandas, pyarrow, aiohttp….

## 1. Before the bump (GPU lease, old stack): baselines

Stop the mission worker on :50051 first (plain `kill`; the gateways on 8080/8081 reconnect by themselves).

```zsh
cd $REPO/src/worker
mkdir -p $B/gpu/old
# a) identity corpus + 16 sequential prompts, B=1, 9B
$S/gpu_tests/start_worker.sh $B/gpu/old/w9.log worker.py $NINE QWEN_MAX_BATCH=1
$W corpus --out $B/gpu/old/corpus9.json; $W seq --n 16 --max-tokens 256 --out $B/gpu/old/seq16.json
$W think > $B/gpu/old/think.txt
pkill -f "worker.py --addr 127.0.0.1:50051"; sleep 6
$S/gpu_tests/start_worker.sh $B/gpu/old/w9p.log worker.py $NINE QWEN_PROMPT_CACHE=1
$W pcache > $B/gpu/old/pcache.txt
pkill -f "worker.py --addr 127.0.0.1:50051"; sleep 6
# vision needs a VL checkpoint (Ornith-1.5 is text-only)
$S/gpu_tests/start_worker.sh $B/gpu/old/w10.log worker.py QWEN_MODEL=../../weights/ornith10-9b-mlx-8bit QWEN_KV_BITS=0
PYTHONPATH=. .venv/bin/python $S/gpu_tests/vision_test.py ref --out $B/gpu/old/vision.json
pkill -f "worker.py --addr 127.0.0.1:50051"; sleep 6
# batched-vs-solo agreement (no server)
env $NINE PYTHONPATH=. .venv/bin/python $S/gpu_tests/agree.py > $B/gpu/old/agree.txt
# b) DFlash baseline (9B): speed + output on the corpus
$S/gpu_tests/start_worker.sh $B/gpu/old/w9d.log worker.py $NINE QWEN_MAX_BATCH=1 QWEN_DRAFT=../../weights/qwen35-9b-dflash
$W corpus --out $B/gpu/old/corpus9_dflash.json; grep "dflash:" $B/gpu/old/w9d.log | tail -20
pkill -f "worker.py --addr 127.0.0.1:50051"; sleep 6
# c) teacher-forced distributions (no server; loads like the worker). 27B/35B need the wired-limit recipe.
QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 PYTHONPATH=. .venv/bin/python $B/gpu/stack_kl.py dump --out $B/gpu/old/kl9.npz
QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=8 QWEN_WIRED_LIMIT_GB=27 PYTHONPATH=. .venv/bin/python $B/gpu/stack_kl.py dump --out $B/gpu/old/kl27.npz
QWEN_MODEL=../../weights/qwen36-35b-a3b-mlx-ud3 QWEN_KV_BITS=4 QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512 PYTHONPATH=. .venv/bin/python $B/gpu/stack_kl.py dump --out $B/gpu/old/kl35.npz
# d) long prefill timing + peak (9B, fp16 KV): prints "prefill+score N s, peak G GB"
QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 PYTHONPATH=. .venv/bin/python $B/gpu/stack_kl.py dump --len 32768 --score-from 32000 --out $B/gpu/old/kl9_32k.npz
QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 PYTHONPATH=. .venv/bin/python $B/gpu/stack_kl.py dump --len 65536 --score-from 65000 --out $B/gpu/old/kl9_64k.npz
```

(3.4K tokens by default = the corpus answers as chat turns; KV stays fp16 below 5000 tokens on every model.)

## 2. Apply (≈ 5 min; no GPU needed until §4)

```zsh
cd $REPO
mkdir -p $B/rollback && cp src/worker/pyproject.toml src/worker/uv.lock src/worker/worker.py $B/rollback/
cp -R src/worker/tests $B/rollback/tests
# 1. worker.py: the patch was made against $B/worker.base.py. If the repo file moved on, check that
#    it still applies (only hunks touching the same lines conflict; the hunks are small):
diff -q $B/worker.base.py src/worker/worker.py || echo "worker.py changed since the patch was refreshed"
patch -p1 --dry-run < $B/worker.patch && patch -p1 < $B/worker.patch
# 2. tests (conftest TinyVLM = 0.7.6 get_input_embeddings, ThinkingBudgetCriteria kwarg,
#    verify-kernel monkeypatch target, a race fix, mrqmm fused-GDN expectations, new test_bump.py)
patch -p1 --dry-run < $B/tests.patch && patch -p1 < $B/tests.patch
# 3. dependencies (src/worker only). $B/project/orig/ holds the pyproject/lock the prepared ones were
#    resolved from; if the repo's are still those, copy the prepared ones:
cd src/worker
diff -q $B/project/orig/pyproject.toml pyproject.toml && diff -q $B/project/orig/uv.lock uv.lock \
  && cp $B/project/pyproject.toml $B/project/uv.lock .
#    ... otherwise resolve them again (same result):
#    uv add "mlx==0.32.3" "mlx-vlm==0.7.6" "llguidance>=1.9.1" "transformers>=5.14" "protobuf>=6.33.5,<7"
#    uv remove mlx-lm
uv sync                                    # installs into src/worker/.venv (the worker MUST be stopped)
uv run --group proto bash gen_proto.sh     # protobuf stays 6.33.x: same gencode, harmless; required if protobuf moves
# 4. CPU suite on the new stack (no GPU): expect all pass (68 at preparation time) and no skips
#    except Metal-only test_mrqmm.py (ignored here)
.venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore tests/test_mrqmm.py
# 5. CPU parity checks (optional, ~1 min): prompts byte-identical to the old stack's dump
.venv/bin/python -I $B/checks/render_parity.py --worker-dir $REPO/src/worker --out $B/checks/render_applied.json
python3 $B/checks/render_parity.py --compare $B/checks/render_old.json $B/checks/render_applied.json   # "60 prompts, 0 differ"
.venv/bin/python -I $B/checks/textonly_load.py      # "strict load OK; vision_tower = NoneType"
```

Rollback: stop the worker; `cp $B/rollback/{pyproject.toml,uv.lock,worker.py} src/worker/`;
`rm -rf src/worker/tests && cp -R $B/rollback/tests src/worker/tests`; `cd src/worker && uv sync`.

## 3. What the patch changes (worker.py)

1. `_render`: `enable_thinking` defaults to True when the template uses it (`kwargs.setdefault`,
   after the vLLM kwargs filter, so a request's own `chat_template_kwargs.enable_thinking` wins).
   mlx-vlm ≥ 0.6.3 injects False otherwise (vlm#1316). Checked: 60 prompts × 4 checkpoints through
   the real `Engine._render` (pb.Request as the gateway sends it: default, think on/off, other kwargs,
   developer, tools, tool history, reasoning history, no generation prompt, continue_final, images):
   **0/60 differ** from mlx-vlm 0.6.2 + old worker; unpatched 0.7.6: 40/60 differ.
2. Thinking budget: `ThinkingBudgetCriteria(..., prompt_preopens_thinking=True)` (Engine.generate,
   the only constructor) and `pop_forced_token_id()` instead of `apply_forced_token` (`_iterate`, the
   only call site; the forced id keeps the sampled token's dtype). Without the kwarg the budget never counts.
3. `_prefill`: `get_input_embeddings` now returns `position_ids`/`rope_deltas` (counted from 0).
   Text: both dropped and `lm._position_ids = None` (what 0.6.2's text path did) — forwarding them
   placed a prompt-cache hit's suffix at position 0 (`test_prompt_cache_while_batching` fails with
   them, checked by mutation). Vision: the returned positions are sliced per slab (the A2 fix, now from
   the returned array instead of `lm._position_ids`, which 0.7 no longer sets) and the row's mRoPE delta
   comes from the returned `rope_deltas`.
4. `_spec` (DFlash): `gdn_sink` gone from the prefill call; verify is
   `lm(fed, cache, capture_layer_ids, speculative_verify=True)`, whose cache transaction
   (`out.gdn_states`) is committed every round with `commit_speculative_round` right after the walk
   (KV trimmed to the accepted prefix, recurrent/conv state selected per position) or aborted on any
   exception; the old `rollback_speculative_cache` call is gone (it would hit the legacy trim path
   and raise on ArraysCache). Mutation check: without the commit the DFlash tests fail.
5. Text-only checkpoints: the `load_weights(strict=False)` monkeypatch is removed. mlx-vlm 0.7.6
   sets the weightless `vision_tower` to None before its strict load (checked on a tiny text-only
   qwen3_5 checkpoint with Ornith-1.5's config layout: 0.7.6 loads strictly, 0.6.2 fails without
   the monkeypatch). Images on such a checkpoint are still refused in `_render`.
6. `_rows_attention` (BLOCKING on 0.7): `Qwen3_5Attention.__call__` lost `target_verify`, so every
   attention call through the override raised TypeError. Now: non-RowsKV caches → the library call
   with whatever kwargs it got (B=1, prefill: unchanged library path); RowsKV → the library's
   `_prepare_projected_qkv(q, k, v, cache=None, position_ids, position_embeddings, mask=None)`
   (projections' outputs, norms, mRoPE — positions are always explicit), then per-row
   `update_and_fetch` + SDPA as before. Other symbols re-checked against 0.7.6 (`checks/symbols.py`):
   `generation_stream`, `maybe_quantize_kv_cache` (uniform path unchanged), `wired_limit`,
   `make_prompt_cache`, KVCache/QuantizedKVCache/ArraysCache (`.cache`, `.keys`, `.offset`,
   `ArraysCache(size=)`; the new speculation/lengths metadata defaults to None), `BPEStreamingDetokenizer`,
   `process_image`/`process_inputs_with_fallback` (test_vision_inputs_equal_prepare_inputs passes),
   `ThinkingBudgetCriteria`, `scaled_dot_product_attention(q, k, v, cache, scale, mask, sinks=None)`,
   `_rope_deltas`/`_position_ids`, `fa_idx`, `is_linear`, GDN attrs used by the memory formulas, the
   drafter API. `LanguageModel.__call__`'s new `_batch_invariant_decode` hook exists only on qwen4_exp.
   The MoE model imports `Qwen3_5Attention`/`Qwen3_5GatedDeltaNet` from `qwen3_5.language`, so
   every override covers the 35B too.
7. `QWEN_SDPA_FUSED` (default 1): override of `qwen3_5.language.scaled_dot_product_attention`
   (module global, no library edit): `force_fused=True` only for a plain `KVCache`, fp16 queries,
   `mask == "causal"`, q_len ≥ 512 and keys ≥ 32K (prefill slabs of fp16-KV models: the 9B; also the
   DFlash prefill). Everything else, quantized KV included, goes to the library function. If MLX
   refuses (no fused kernel) it logs once, turns itself off and falls back. `QWEN_PREFILL_BUDGET`
   is unchanged (with the default budget every slab ≥ 32K is 512–2048 tokens, i.e. fused).
8. `QWEN_GDN_FAST` (default 0): override of `qwen3_5.gated_delta.gated_delta_kernel` (the Metal
   path of `gated_delta_update`; speculative-history updates use another function and are not
   affected): slabs of ≥ 16 tokens without a mask go through `mx.fast.gated_delta_update`, state
   returned in the cache's dtype (fp32). CPU check vs the ops reference: |Δy|, |Δstate| < 2e-3 (fp16).
   Clean enough to keep; off until the KL check (§4 g).
9. New, found while patching: **`QWEN_VERIFY_EXACT_MAX_T` (default 8)**. 0.7's `speculative_verify`
   routes every verify linear (and lm_head) of a B=1 block through the exact multi-row verify-qmv kernel
   (0.6.2 used plain `quantized_matmul` for one row). Measured in microbench.md §1a: ×0.63–0.82 of qmm at
   4–8 rows, but **×9–12 at 12–16 rows** (register spill) — DFlash's adaptive block reaches 16, so an
   uncapped verify would be ~1 s instead of ~70 ms. The override (on `Qwen3_5BatchInvariantForward._linear`
   / `_linears`) keeps the exact kernels for blocks ≤ 8 tokens and uses `linear(x)` above. Exactness
   (DFlash == plain greedy, no near-tie flip) then holds only for blocks ≤ 8; set 16 to measure the
   all-exact speed, 0 for the 0.6.2 behaviour.

mrqmm.py: unchanged and compatible as far as the CPU can tell — `mx.fast.metal_kernel` constructor
(+ new optional `compile_options`) and call signature are the same in 0.32.3, the 8-bit affine layout
(uint32 words, byte k%4 of word k//4, fp16 scales/biases [N, K/64]) is unchanged, and no mlx-vlm code
checks `type(m) is nn.QuantizedLinear` (verify paths use `isinstance`, so `MRQuantizedLinear` is still
taken for exact verify / falls to `linear(x)`). One behaviour change: mlx-vlm 0.7 no longer fuses a decode
step's four GDN in-projections (`_decode_quantized_linears_fused` is gone), so `mrqmm._patch_fused` is a
no-op and in_proj_qkv / in_proj_z run as separate mrqmm calls (in_proj_b/a, N = 32, on qmv) — 4 launches per
GDN layer instead of 1, for B=1 decode as well. tests.patch makes test_mrqmm expect that on 0.7.

## 4. GPU re-verification (new stack, after §2)

Greedy byte-identity with the old stack is **not** expected (mlx#4241 fp32 dequant in qmm for prefill
M ≥ 6, MSL 4.1 kernels, the GDN decode conv now in fp32). Byte-identity is expected *within* the new stack.

```zsh
cd $REPO/src/worker; mkdir -p $B/gpu/new
```

a. **Metal tests**: `.venv/bin/python -m pytest tests/test_mrqmm.py -q` (kernel vs qmm/fp32 reference,
   dispatch boundaries, tiny forward with the new per-projection counts), then
   `MRQMM_REAL_WEIGHTS=1 .venv/bin/python -m pytest tests/test_mrqmm.py -q -k real_9b`.
b. **Teacher-forced KL, old vs new** (pass: top-1 ≥ 99%, KL mean ≪ 3e-4 = the bf16→fp16 figure, 35B ≪ 2e-3):
   ```zsh
   QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 PYTHONPATH=. .venv/bin/python $B/gpu/stack_kl.py compare --ref $B/gpu/old/kl9.npz
   QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=8 QWEN_WIRED_LIMIT_GB=27 PYTHONPATH=. .venv/bin/python $B/gpu/stack_kl.py compare --ref $B/gpu/old/kl27.npz
   QWEN_MODEL=../../weights/qwen36-35b-a3b-mlx-ud3 QWEN_KV_BITS=4 QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512 PYTHONPATH=. .venv/bin/python $B/gpu/stack_kl.py compare --ref $B/gpu/old/kl35.npz
   ```
c. **Identity corpus at QWEN_MAX_BATCH=1** (9B) vs §1a: `start_worker.sh $B/gpu/new/w9.log worker.py $NINE QWEN_MAX_BATCH=1`,
   `$W corpus --out $B/gpu/new/corpus9.json; $W compare $B/gpu/old/corpus9.json $B/gpu/new/corpus9.json`.
   Differences are acceptable only where (b) is clean; look at where they diverge (late near-ties, same
   quality). Run the corpus twice on the new stack: those two must be byte-identical. Same for
   `seq --n 16` (keep `$B/gpu/new/seq16.json` as the new reference for `conc`).
d. **Batching** (default QWEN_MAX_BATCH): `$W conc --n 8 --ref $B/gpu/new/seq16.json` and
   `env $NINE PYTHONPATH=. .venv/bin/python $S/gpu_tests/agree.py` vs `$B/gpu/old/agree.txt` (B=2 must stay
   100% / KL 0: qmv per row; B ≥ 3 within the old numbers — mrqmm now sees 4 separate GDN projections),
   plus `$S/gpu_tests/smoke_big.sh` for the 27B and 35B (it compares against `$S/orig_worker` — expect
   non-identical; use (b) as the criterion, and its conc 4 for "all finish").
e. **Prompt cache**: `QWEN_PROMPT_CACHE=1`, `$W pcache` → hits byte-identical to cold runs (patch item 3).
   Then a vision request followed by a cached text request: `vision_test.py check --ref` on ornith10
   (ref re-made on the new stack first if (c) shows drift), then `$W pcache` on the same worker.
f. **Thinking**: `$W think` (default request thinks; `enable_thinking:false` does not; `thinking_budget`
   forces `\n</think>` at the budget); compare with `$B/gpu/old/think.txt`. A 27B request without
   chat_template_kwargs must still carry the template's "Reasoning effort" block (CPU-checked; confirm
   prompt_tokens in the worker log equal the old ones).
g. **GDN fast** (`QWEN_GDN_FAST=1`), within the new stack:
   `... stack_kl.py dump --out $B/gpu/new/kl9.npz` (default), then
   `QWEN_GDN_FAST=1 ... stack_kl.py compare --ref $B/gpu/new/kl9.npz` (same for 27B/35B);
   pass: KL ≲ 1e-4. Prefill time from the same runs (`prefill+score`). Enable by default only if it passes.
h. **force_fused** (9B, fp16 KV): `QWEN_SDPA_FUSED=0` vs `1` with `stack_kl.py dump --len 32768
   --score-from 32000` and `--len 65536 --score-from 65000` (time + peak), interleaved 0/1/0/1 (the
   machine drifts ±20%); KL between them via `compare --ref` (expect ~1e-5 or less). Then try
   `QWEN_PREFILL_BUDGET=$((2048*131072))` with fused on (slab 2048 at every offset: microbench showed
   −8% at 64K, −1.5 GB) — that also changes `_bounds` / prompt-cache cut points and the batch budget's
   prefill reserve, so only as a separate, measured change.
i. **DFlash** (9B, `QWEN_DRAFT=../../weights/qwen35-9b-dflash`, `QWEN_MAX_BATCH=1`): corpus output vs the
   new plain corpus (c) — with `QWEN_VERIFY_EXACT_MAX_T=16` it must be byte-identical (all blocks exact);
   default 8: identical up to near-ties in blocks > 8. Speed (`dflash:` lines, tok/s on the 5 coding tasks)
   for `QWEN_VERIFY_EXACT_MAX_T` = 0 / 8 / 16 vs the 47–53 tok/s baseline (§1b) — pick the default from it.
j. **Memory / long runs**: Ornith-1.5 load peak (expect ~2 GB lower: no random vision tower; image request
   → 400); one ≥ 8K-token generation with `get_active_memory` flat (0.7 evals cache state every 50 tokens
   in its own loop, the worker does not — the GDN state chain is consumed by each next step, so no growth
   is expected); 35B 56K × 2 with the wired-limit recipe (no OOM).
k. Full gateway parity suite (tools, response_format, stop strings) through :8090.

## 5. Docs to update once §4 passes

CLAUDE.md: Ornith-1.5 bullet (no strict=False monkeypatch any more; tower dropped), the "MLX ≤ 0.32.2 has
no fused SDPA for head_dim 256" dead end (0.32.3 + `QWEN_SDPA_FUSED` now), DFlash gotchas (exact verify ≤ 8
tokens; `QWEN_VERIFY_EXACT_MAX_T`), env table (`QWEN_SDPA_FUSED`, `QWEN_GDN_FAST`, `QWEN_VERIFY_EXACT_MAX_T`),
"When bumping mlx-vlm, diff generate/ar.py::generate_step" (now 0.7.6), the mrqmm fused-GDN note, and the
versions in BRIEF/README.

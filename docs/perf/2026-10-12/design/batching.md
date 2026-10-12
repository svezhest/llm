# Continuous batching for the MLX worker: design

Date 2026-10-11. Scope: `src/worker/worker.py` (Python MLX worker) plus a small gateway and
proto change. Libraries: mlx 0.31.2, mlx-lm 0.31.3, mlx-vlm 0.6.2 (the worker venv; the
root `.venv` has byte-identical copies of every file cited here). No model was loaded
while writing this. The claims marked **[CPU-verified]** were checked on the CPU device with
tiny random qwen3_5 / qwen3_5_moe language models (no weights). Use
`design/test_batching_proto.py` and `design/check_lib_batch_qkv.py` to re-run them.

Files written next to this document:

| file | what |
|---|---|
| `design/batching_proto.py` | prototype of the decode plumbing: `RowsKV`, the patched attention, GDN stacking, `batch_logits` (about 150 lines, meant to be lifted into worker.py) |
| `design/test_batching_proto.py` | CPU test: batched step equals per-row B=1 (dense + MoE; fp32/fp16; 8/4-bit weights; q8/q4 KV; rows joining, leaving and going back to B=1) |
| `design/check_lib_batch_qkv.py` | CPU test: mlx-vlm 0.6.2's own quantized batch cache ignores left padding at decode |
| `design/microbench_decode_batch.py` | **GPU**: real model, ms/step for B=1..16 at 1K/8K context (lib1 / rows / lib modes) |
| `design/microbench_attn_rows.py` | **GPU**: one attention layer, per-row loop vs batched / ragged / masked kernels |
| `design/microbench_qmm_rows.py` | **GPU**: per-model matmul cost vs rows M (synthetic weights, real shapes and bits; MoE gather; 4/5-bit verify kernel) |
| `design/microbench_sampling.py` | **GPU**: per-row logits post-processing cost at B rows (vocab 248320) |

Every microbench has a `--smoke` flag that runs the code paths on the CPU with tiny shapes and
needs no lease. All of them passed `--smoke`.

---

## 0. TL;DR

* **Feasible for all three models, but not through the library's batch path.** mlx-vlm
  0.6.2 ships a Qwen3.5 batch path (BatchKVCache, left padding, a ragged Metal decode kernel,
  `BatchQuantizedKVCache`, a `BatchGenerator`). It is unusable here:
  * the quantized batch cache ignores left padding at decode **[CPU-verified: a padded row's
    logits move by about 1.0]**, which breaks the 27B (q8) and the 35B (q4);
  * the ragged kernel requests a 1024-thread threadgroup that M1 may reject (vlm#1799, see
    `versions.md` §3.4);
  * a one-row batch cache is extracted and re-merged (a full KV copy) on every step;
  * join and leave copy every row's KV;
  * it quantizes KV from token 0, which does not match `quantized_kv_start=5000`;
  * one sampler serves the whole batch;
  * it has no hook for our prompt cache or GDN checkpoints.
* **Proposed: "per-row KV, batched everything else".**
  * Each request keeps exactly the cache objects it has today: a `KVCache` or
    `QuantizedKVCache` per full-attention layer, prefilled by today's slab loop.
  * At decode, the B rows go through **one** library forward of shape `(B, 1)`. The
    projections, MLP/MoE, the GDN layers (their recurrent state is stacked `(B, …)` in one
    `ArraysCache` per layer) and `lm_head` run batched.
  * Only the full-attention cache update and SDPA run per row, through a ~25-line
    class-level override of `Qwen3_5Attention.__call__`. That class is shared by the dense
    and the MoE model.
  * Positions are passed explicitly per row (`position_ids` (B,1) = KV length + the row's
    mRoPE delta).
  * **[CPU-verified]** batched logits equal per-row B=1 logits: 0 difference in fp16,
    ≤3e-6 in fp32, and 4e-4 for the MoE with q8 KV. Top-1 matches in every case. This holds
    for dense and MoE, 8/4-bit weights, fp16/q8/q4 KV, and rows joining, leaving and
    dropping back to B=1.
* **Exactness:**
  * B=1 goes through today's exact library call on today's cache objects. Explicit positions
    were checked to be bit-identical to the computed ones **[CPU-verified]**. So a request
    that decodes alone is byte-identical to today, even if its prefill overlapped other
    rows' decode.
  * At B>1 a row differs from solo only through the matmul kernels' M-dependent rounding
    (qmv→qmm_splitk at M≥6 on M1 Max for the 9B shapes; sorted gather for MoE). This is the
    same class of noise as the already-accepted DFlash verify difference.
* **Scheduling:**
  * One scheduler thread owns MLX.
  * gRPC handler threads render and tokenize, enqueue, then stream from a per-request queue.
  * Prefill runs one request at a time with today's slab loop (exactness, prompt cache and
    checkpoints unchanged), interleaved with decode steps by a time-share knob.
  * Admission uses a reservation of `prompt + max_tokens` KV bytes plus GDN state against a
    memory budget.
* **Request kinds:**
  * Batch cheaply: text, thinking on/off, thinking budget, stop strings and ids, penalties,
    per-request sampling and seeds, vision **decode**.
  * Batch in sync mode (no step-ahead for the whole batch while present): grammar
    (`response_format`, constrained tools) and multi-token `bad_words`.
  * Stays solo: DFlash. A DFlash row is demoted to a normal row at a round boundary when
    another request arrives.
  * Prefill of any kind (vision included) is serialized.
* **Expected ceiling (9B, short context; to confirm with `microbench_decode_batch.py`):**
  measured single-stream decode is 34 tok/s (results/baseline_256.txt). The qmm table
  (M=2 ×1.17, M=4 ×1.97, M≥6 ×2.6) implies about 58 / 70 / 104 / ~195 tok/s aggregate at
  B=2/4/8/16, i.e. ×1.7 / ×2.0 / ×3.0 / ×5.7. B=3..5 is a poor operating point (≈ the cost
  of B=6). Long context adds about 0.75 ms per row per step at 8K (9B fp16 KV).
* **Gateway:** `HealthReply.max_batch` sets the in-flight semaphore (env override). The
  `qwen_in_flight` gauge becomes a count. Worker-side queue and batch stats go in `Usage`.
  The gRPC server's `ThreadPoolExecutor(max_workers=4)` **must** grow, because every
  streaming Generate holds a thread.
* **Incidental findings (appendix A):** two suspected pre-existing RoPE-position bugs, both
  by code reading and both needing a GPU check:
  1. DFlash's first lm() call after prefill computes position 0 for the last prompt token
     (worker.py:1495, 1517).
  2. Vision prompts longer than one prefill slab get text-style positions from slab 2 on.

---

## 1. Feasibility

### 1.1 What mlx-lm 0.31.3 offers

`mlx_lm/generate.py`:
* `BatchGenerator` (1486). `insert` (1585) queues sequences. `_next` (1769):
  1. decode the generation batch first;
  2. if there is room, move up to `prefill_batch_size` sequences into a `PromptProcessingBatch`;
  3. process one `prefill_step_size` chunk of all prompting rows;
  4. split finished prompts into the generation batch with `extend` (1819).
* `PromptProcessingBatch.prompt` (1124):
  1. right-pads the chunk (`_right_pad_prompts` 808);
  2. calls `c.prepare(lengths=…, right_padding=…)` on every cache (1152);
  3. runs the model;
  4. `finalize()` rolls the right padding into left padding (1165-1170;
     `BatchKVCache.finalize` 980 uses `dynamic_roll` 903).
* `GenerationBatch._step` (1320):
  1. one forward on `inputs[:, None]` with the batch cache (1332);
  2. per-row logits processors, each a `(1, V)` slice (1343-1349);
  3. per-row samplers (1355-1361);
  4. `async_eval` of the next tokens before reading the current ones (1369-1374, the same
     step-ahead as our `_run`).
* `next` (1405): per-row `max_tokens` and an Aho-Corasick stop machine (943), then
  `filter(keep)` (1383) evicts finished rows.
* `_make_cache` (838): `KVCache`→`BatchKVCache(left_padding)`. `ArraysCache` gets
  `.left_padding`. **There is no quantized batch cache in mlx-lm** (`to_batch_cache` raises for
  anything else, 857).

`mlx_lm/models/cache.py`:
* `BatchKVCache` (912):
  * `offset` is an `mx.array` per row (= real tokens). `_idx` is the shared column count.
  * `update_and_fetch` (942) grows every row by 256-column steps.
  * `make_mask` (1011): `create_causal_mask(..., left_padding=)` (base.py:24-43).
  * `filter` (1016) gathers rows and trims the common left padding: a **copy of every kept
    row's KV**.
  * `extend` (1035) pads both sides to the same width and concatenates: **copies both batches'
    KV**.
  * `extract` (1080) gives a row back as a `KVCache`.
  * `merge` (1089) left-pads a list of `KVCache`s into one buffer.
* `ArraysCache` (594):
  * the batch axis is axis 0 of each state array; `filter`/`extend`/`extract`/`merge`
    (632-721);
  * `make_mask` (691) from `left_padding` or `lengths`;
  * `advance` (685).

`mlx_lm/models/qwen3_next.py` (and the text `qwen3_5.py`, which reuses its blocks):
* The GDN masks padded inputs to zero before the conv (`mx.where(mask[..., None], …)`, 258).
* It keeps the conv state per row at each row's true end via `cache.lengths` (264-267).
* The recurrent state is `[B, Hv, Dv, Dk]` per row (`gated_delta.py` docstrings 140-150).
* The masked Metal kernel skips the state update on padded steps
  (`if (mask[b_idx*T+t])`, gated_delta.py:16/63).

So **yes: mlx-lm's gated-delta-net supports batched decode with padding, and the recurrent
state is simply the batch row of the state tensor.** mlx-lm also has a text-only `qwen3_5`
(`mlx_lm/models/qwen3_5.py`) with plain RoPE: for text, all three mRoPE axes carry the
same position (see mlx_vlm `rope_utils.compute_mrope_frequencies` 486-490). But the
worker's runtime is mlx-vlm (VL checkpoints, vision tower, its decode kernels), and we
do not want two model implementations.

### 1.2 What mlx-vlm 0.6.2 offers, and where it breaks

`mlx_vlm/models/qwen3_5/language.py`:

| code | behaviour | consequence |
|---|---|---|
| `make_cache` 2579 / `cache.make_prompt_cache` (cache.py:931) | `ArraysCache(size=2)` for linear layers, `KVCache()` for full-attention layers. No batch variant from `make_prompt_cache`. | Batch caches come only from `generate/ar.py:_make_cache` (715). |
| `Qwen3_5Model.__call__` 1783-1810 | **B==1 with a "single-row batch cache"** (`left_padding.size == 1`): extracts plain row caches, runs, then `cache[i] = cls.merge([row])`. | **A full KV copy every decode step** when a batch shrinks to one row. `BatchQuantizedKVCache` has no `extract`/`merge`, so `_extract_row_cache` returns an **empty `KVCache()`** (681-683) and the row's KV is lost. |
| `Qwen3_5Model.__call__` 1812-1867 | B>1 and L>1 with left padding: **loops over rows** (extract → run → merge). | "Batched prefill" is per-row anyway. No prefill batching gain. |
| `_create_qwen3_5_attention_mask` 798-818 | decode (L=1) with any left padding returns the string `"left_padded_decode"`. Pads come from `left_padding.tolist()` (a host sync, cached on the array object identity, 720-725/807-812). | — |
| `Qwen3_5Attention.__call__` 1454-1463 | `"left_padded_decode"` → `mask=None`, then `_target_verify_left_padded_attention`, which **returns None for quantized caches** (`hasattr(cache, "bits")`, 1298). It falls back to `scaled_dot_product_attention(..., mask=None)` (1488). | **Quantized KV + left padding attends to the zero pad slots.** **[CPU-verified]** `check_lib_batch_qkv.py`: the same 6-token row padded by 9 vs unpadded: fp16 `BatchKVCache` 2e-6, `BatchQuantizedKVCache` q8 **1.06**, q4 **1.0** max \|Δlogit\|. |
| `_qwen3_5_ragged_decode_attention` 1196-1286 | Custom Metal SDPA over left-padded fp16/bf16 KV, `pads[batch_idx]`. One plan for all rows, else `None` (1229-1231). Then per-pad-group SDPA (1314-1351). | 1024-thread threadgroup (1253-1254, 1282-1283). Per `versions.md` §3.4, vlm#1799: M1 Max may reject this. **Likely Metal failure once rows pass 1024 KV tokens.** |
| `LanguageModel.__call__` 2402-2496 | With `position_ids=None`: per-row decode positions = `max(c0.offset, 0) + rope_deltas` (2477-2482) when the cache's `offset` is an array; else `cache_offset + self._rope_deltas`. With **`position_ids` given, the whole block is skipped** (2430). | Explicit `position_ids` (B,1) gives per-row positions without touching the shared `_rope_deltas`/`_position_ids` state. |
| `Qwen3_5GatedDeltaNet.__call__` 1571-1701 | Works for any B. It resets the conv/recurrent state to zeros if `shape[0] != B` (1592-1601, 1638-1639). Decode conv for S==1 (1619-1624, per-row). `gated_delta_update` → mlx-lm kernel (grid `B*Hv`, rows independent). | Stacked per-row state works as is. **The B mismatch reset is silent:** a stacking bug would zero a row's memory with no error. The design asserts shapes. |
| `_decode_quantized_linears_fused` 514-564 | At L==1 the four GDN input projections run as one qmm for any B. | Batched for free. |
| `qwen3_5_moe/language.py` | `Qwen3_5MoeAttention` **is** `Qwen3_5Attention` (import alias, line 8). `SparseMoeBlock` 51-73 works for any token count. `SwitchGLU` (mlx_lm switch_layers.py:176-199) sorts tokens by expert when `indices.size >= 64` (B·top_k ≥ 64, so B ≥ 8 for top-8). | MoE batches. Below B=8 each (token, expert) pair is its own gather. Experts are shared only from B≥8 (sorted path). |

`mlx_vlm/models/cache.py:181` `BatchQuantizedKVCache` mirrors `BatchKVCache`:
* no `extract`, `merge`, `prepare` or `finalize`;
* `filter` and `extend` copy all rows.

`generate/ar.py`:
* `_make_cache` (715-789) quantizes **from token 0** and skips the **last layer**
  (774-777). Our path quantizes all layers once a cache passes 5000 tokens, so the
  numerics differ from today.
* `GenerationBatch` (858) has **one sampler** for the batch (1014-1019), per-row
  logits processors and thinking criteria (1221-1235), and per-row `rope_deltas` passed as
  the `rope_deltas=` kwarg (972-973).

**Verdict on the library path:**
* fp16-only (9B), equal-length rows: works.
* ragged fp16: blocked on M1 by the kernel's threadgroup size.
* quantized KV: wrong.
* prompt cache and GDN checkpoints: no integration.
* Not used.

### 1.3 What does work, and the proposal's building blocks

1. **Per-row attention on per-row caches.**
   * The override (batching_proto.py `_rows_attention`) is Qwen3_5Attention's own code for
     L=1: q/k/v projections, q/k norms, rotary with `position_ids` (B,1), o_proj and gate.
     All of it runs on the `(B,1,D)` batch.
   * Then, per row `i`: `cache_i.update_and_fetch(k[i:i+1], v[i:i+1])` and
     `scaled_dot_product_attention(q[i:i+1], K_i, V_i, cache=cache_i, mask=None)`. This is
     the same library function the B=1 path calls. It dispatches on `cache.bits` to the
     quantized SDPA (mlx_lm base.py:117-128).
   * Quantized KV, `quantized_kv_start=5000` and the prompt-cache slicing all keep working
     unchanged, because every row's cache is the very object B=1 uses.
2. **Stacked GDN state.**
   * One `ArraysCache` per linear layer, rows on axis 0 (conv `(B,3,conv_dim)` fp16,
     recurrent `(B,Hv,Dv,Dk)` fp32).
   * No mask: every row feeds a real token every step.
   * Join = `concatenate`, leave = gather. Each copies the GDN state only, about 51 MB per
     row on the 9B.
3. **Positions.**
   * Explicit `position_ids` (B,1) int32: the row's KV length, plus the row's mRoPE delta for
     a vision row (`lm._rope_deltas` captured right after that row's prefill).
   * At B=1 the same explicit array **equals** what `LanguageModel` computes today
     **[CPU-verified: `explicit==auto positions True`, bit-exact]**.
4. **The library model loop stays in charge.**
   * `RowsKV` has no `left_padding`, `make_mask` or `_idx`. So `Qwen3_5Model.__call__`
     takes none of its batch branches (1783, 1812) and builds no mask
     (`create_attention_mask` returns None for N==1).
   * `LanguageModel.__call__` reads only `c0.offset` (2407).
   * MoE and dense decoder layers, the final norm and `lm_head` are the library's.

**[CPU-verified]** `test_batching_proto.py`. Tiny dense and MoE models, 3 prompts (7/13/21
tokens), each prefilled alone, then decoded:
* rows 0+1 for 2 steps;
* row 2 joins;
* row 0 leaves;
* row 1 goes back to the B=1 library path.

Batched-vs-solo max |Δlogit|:

| config | max \|Δlogit\| | top-1 |
|---|---|---|
| fp16 | **0** | — |
| fp32 | 1e-6..3e-6 | 14/14 |
| fp32 MoE with q8 KV | 4e-4 | 14/14 |

Configurations covered: weights fp / 8-bit / 4-bit; KV fp16 / q8 / q4. On the GPU the
batched matmuls use other kernels than M=1 (§4), so expect small non-zero differences
there.

### 1.4 Per-model verdict

The 9B and the 27B both have 4 KV heads.

| | 9B (Ornith 1.0/1.5, dense, fp16 KV) | 27B (dense, q8 KV) | 35B-A3B (MoE, q4 KV) |
|---|---|---|---|
| batched decode with the proposal | yes | yes (quantized rows go through per-row quantized SDPA) | yes (MoE batches; experts shared from B≥8) |
| library batch path | fp16 only; ragged kernel risky on M1 | **no** (left padding ignored) | **no** |
| full-attn KV / token / row | fp16 32 KiB (8 layers × 4 KV heads × 256 × 2 × 2 B) | q8 34 KiB, fp16 64 KiB (16 layers) | q4 5.6 KiB, fp16 20 KiB (10 layers × **2** KV heads) |
| GDN state / row | 51.5 MB (24 layers × 32×128×128 fp32 + conv) | **154 MB** (48 layers × 48 heads) | 64 MB (30 layers) |
| decode bytes amortised by batching | ~all weights (8-bit, 9.5 GB) | ~all weights (5-bit, ~18 GB) | attention/GDN 8-bit, shared expert 4-bit, lm_head 6-bit; the routed experts (3-bit) are **not** shared below B=8 |
| expected gain at B=8 (to measure) | ~×3 | ~×2.8 (GDN state traffic ~0.85 ms/row/step) | ×1.5–2.5 (unknown: gather_qmm, mlx#4509) |
| expected gain at B=16 | ~×5.5 | memory-limited (see §3) | — |
| decode stays exact at B=1 | yes | yes | yes |

Note: CLAUDE.md/README say "~2.6 GB KV @128K" for the 35B at q4. By the shapes in
config.json, q4 is 5,760 B/token = 0.75 GB at 128K (2.6 GB is the fp16 figure).

### 1.5 The cost model behind the estimates

Per decode step, `T(B) ≈ W·f(B) + B·(g + a(ctx) + o)`:
* `W·f(B)`: the weight stream (9B: ~27 of the measured 29.4 ms/token), scaled by the
  qmm-by-rows factor `f`.
* `g`: GDN state read+write per row (9B 0.3 ms, 27B 0.85 ms, 35B 0.35 ms).
* `a(ctx)`: KV read per row (9B fp16: 0.1 ms at 1K, 0.75 ms at 8K, 3 ms at 32K).
* `o`: per-row launch and sampling overhead, measured by `microbench_attn_rows.py` and
  `microbench_sampling.py`.

`f` comes from the M1 Max qmv/qmm dispatch: `get_qmv_batch_limit` is 6 rows for D,O>4096,
10 for ≤4096 and 14 for ≤2048, then qmm_splitk (versions.md §2b). That gives the known
×1.17 (M=2), ×1.97 (M=4) and ×2.6 (M≥6) for the 9B. Consequences:
* B=6..32 costs the same weight time, so the scheduler gains nothing by holding the batch
  at 4.
* The aggregate rises almost linearly from 6 up to the memory limit.
* `microbench_decode_batch.py` measures the real `T(B)`.

---

## 2. Architecture

### 2.1 Threads and ownership

```
gRPC pool threads (Generate handlers, Check, Cancel, Health)
   │  render template, tokenize, length check (text), build _Req (pure Python)
   │  enqueue → Scheduler.inbox ; then block on _Req.out (queue.SimpleQueue)
   ▼
Scheduler thread  ── the ONLY thread that creates/evaluates MLX arrays after load
   admission → prefill (one request at a time, existing slab loop)
   → join batch → decode steps (batched forward, per-row post-processing)
   → per-row events (token ids, finish) into _Req.out
   ▲
   handler thread: detokenize, _GenState split, _StopFilter, tool parsing, pb.Event
```

Rules:
* **MLX work happens only on the scheduler thread.** `_Penalties`, `_Control`, `_Guide`
  bitmask arrays, samplers' keys, `mx.array(ids)`, vision `prepare_inputs` and prompt-cache
  store/restore all run there. CLAUDE.md's gotcha (per-thread MLX random state) is already
  handled by explicit keys. The new rule avoids cross-thread stream and evaluation hazards
  entirely.
* `Engine._lock` goes away. `self._finish` (worker.py:1402/1419/1525/1530) becomes
  per-row.
* `mx.reset_peak_memory()` (1633) is process-global. Under batching:
  * reset it only when the scheduler goes idle;
  * `Usage.peak_gb` = the peak since the batch was last idle (documented);
  * the per-request log keeps `peak=` with that meaning.

### 2.2 Data structures

```python
class _Req:                      # built by the handler thread (pure Python)
    id, params p, ids (list[int] | None for vision), formatted, images (paths),
    tmp_files, max_tokens, inside (thinking open), grammar (compiled llguidance str/None),
    tools (list), t_arrival, out: queue.SimpleQueue, cancel: threading.Event,
    stop_req: threading.Event  # handler saw a stop string: evict
    # scheduler -> handler items: ("tok", id) | ("eof", None) | ("finish", reason, stats) | ("error", kind, msg)

class _Row:                      # scheduler-owned
    req, cache (per-layer list as make_prompt_cache builds it), pos (int: KV length),
    delta (int: mRoPE delta, 0 for text), n (tokens emitted), max_tokens,
    sampling, sampler, pen (_Penalties), ctl (_Control), guide (_Guide|None),
    thinking (ThinkingBudgetCriteria|None), first (bool: next input is the last prompt
    token), reserved_bytes, spec_ok (DFlash-eligible), stats (queue/prefill/batch sums)
```

Batch state (scheduler):
* `rows: list[_Row]` in batch order;
* `gdn: dict[layer → ArraysCache]` (only when B ≥ 2);
* `cur: mx.array (B,)`: the pending input tokens of the next step, lazy;
* `sync: bool` = any row with a guide or with `ctl.sync`.

### 2.3 Scheduler loop (pseudo-code)

```python
def run(self):
    while True:
        self._drain_inbox()                       # new _Req → waiting deque; apply cancels
        if not self.rows and not self.prefilling and not self.waiting:
            self._go_idle(); self.cv.wait(); continue      # exit wired_limit, clear_cache, reset peak
        self._go_busy()                           # enter wired_limit(model,[generation_stream]) once
        if self.prefilling is None and self.waiting and self._admit_ok(self.waiting[0]):
            self.prefilling = self._prefill_iter(self.waiting.popleft())   # generator, §2.5
        if self.prefilling and (not self.rows or self._prefill_turn()):
            if next(self.prefilling, DONE) is DONE: # one slab (or the whole thing if alone)
                self._join(self.prefilling_row); self.prefilling = None
            continue
        if self.solo_spec:                        # DFlash row, nobody else: one round
            self._spec_round(); continue
        self._decode_step()                       # §2.6–2.8
```

`_prefill_turn()` implements the interleave policy (§2.5). `_drain_inbox` also handles
DFlash demotion: a newcomer while `solo_spec` → finish the current round → `_join` as a
normal row.

### 2.4 Admission

* Order: FIFO with no overtaking. The head of the queue waits until it fits, so a large
  request is never starved.
* Prefill: one request at a time.
* Admit the head if **all** of these hold:
  * `len(rows) < QWEN_MAX_BATCH`;
  * `Σ reserved(rows) + reserved(head) ≤ KV_BUDGET` (§3);
  * `mx.get_active_memory() + reserved(head) + PREFILL_RESERVE ≤ cap`. This is a live guard
    against estimate errors.
* **Exception:** the batch is empty → always admit. That is today's behaviour: one request
  is always allowed its full context, which MAX_CONTEXT validation already bounds.
* `reserved(r) = kv_bytes(prompt_tokens + max_tokens) + gdn_bytes + 8 MB` (logits and
  sampler scratch). `max_tokens` is the effective limit from `_limit` (1248). A client that
  omits `max_tokens` therefore reserves the rest of the context (128K) and in practice runs
  alone or nearly alone.
* To batch such traffic, set `GATEWAY_DEFAULT_MAX_TOKENS` (e.g. 8192). V2 can add chunked
  reservation (§6 risks).
* `QWEN_MAX_BATCH`:
  * default 8 until the microbench, then per model: 9B 16, 27B 8, 35B 16 are the
    candidates;
  * `1` = today's serialized behaviour through the new code;
  * reported in Health.

### 2.5 Prefill: one request at a time, today's loop, interleaved

* `_run`'s prefill half (worker.py:1371-1397) plus the prompt-cache logic of `generate`
  (1655-1709) become one generator, `_prefill_iter(req)`. Steps:
  1. build `_Row`: `pen`, `ctl`, `guide`, `sampler`, `thinking`;
  2. prompt-cache `lookup`/`restore` and the `_rope_deltas = zeros`/`_position_ids = None`
     resets (1662-1670);
  3. `get_input_embeddings`;
  4. the slab loop with **the same `_bounds` cuts** (697), `quantize`, checkpoints
     (`on_ckpt`) and `clear_cache`;
  5. **yield after every slab**;
  6. at the end, the snapshot/store (`on_prefilled`, 1700-1709);
  7. capture the row's mRoPE delta: `0` for text, `int(lm._rope_deltas)` for vision;
  8. set `row.pos = len(prompt) - 1`, `row.first = True`, `row.y = ids[-1]`.
* **No batched prefill.** Prefill is compute-bound (2048-row GEMMs at ~8 TFLOPS), so
  batching two prompts buys nothing. mlx-vlm's B>1 prefill is per-row internally anyway
  (language.py:1812-1867). Slab cuts must stay a function of the prompt alone (exactness,
  commit 7643918), so nothing about prefill depends on concurrency.
* **Interleave policy** `QWEN_BATCH_PREFILL_SHARE` = s ∈ (0, 1], default 0.5. After a slab
  that took `t` seconds, the scheduler runs decode steps for at least `t·(1-s)/s` seconds
  before the next slab.
  * With no decode rows the prefill runs straight through (identical to today).
  * s=1: prefill first. The newcomer's TTFT is minimal; running rows stall for the whole
    prefill.
  * s=0.5: both progress at half speed.
  * A 2048-token slab on the 9B takes ~3.9 s, so at s=0.5 a running row sees ~4 s stalls
    separated by ~4 s of decode.
  * `QWEN_PREFILL_STEP=512` shortens the stalls at a ~15% prefill-speed cost. It changes the
    cuts for every request alike, so outputs stay concurrency-independent.
* V2, "piggyback": append the B decode rows to the prefill slab as extra tokens of one packed
  forward. Matmuls are per-token, and attention and GDN are already per sequence in this
  design, so the decode rows cost ~nothing during a compute-bound slab.
  * It needs a varlen GDN call: the slab sequence and the decode rows run separately and are
    then concatenated.
  * It needs a check that adding B rows to an M=2048 qmm leaves the slab rows bitwise
    unchanged.
  * Worth it only if the measured stalls matter.

### 2.6 The batched decode step

B = 1 runs the library call exactly as `_run`'s `step` does today (1351-1369), on the row's
own caches. The only change is explicit positions:

```python
out = lm(y[None], cache=row.cache, position_ids=mx.array([[row.pos + row.delta]], mx.int32))
```

For B ≥ 2 (batching_proto.py `batch_logits`):

```python
cache = [gdn[i] if layer.is_linear else RowsKV([r.cache[i] for r in rows])
         for i, layer in enumerate(lm.model.layers)]
pos = mx.array([[r.pos + r.delta] for r in rows], dtype=mx.int32)
logits = lm(cur[:, None], cache=cache, position_ids=pos).logits[:, -1, :]     # (B, V)
```

The attention override (installed once at `Engine.load`; it covers qwen3_5 and qwen3_5_moe)
delegates to the original `__call__` unless the cache is a `RowsKV`:

```python
def _rows_attention(self, x, mask=None, cache=None, position_ids=None,
                    position_embeddings=None, target_verify=False):
    if not isinstance(cache, RowsKV):
        return _ORIG(self, x, mask=mask, cache=cache, position_ids=position_ids,
                     position_embeddings=position_embeddings, target_verify=target_verify)
    B, L, _ = x.shape                                   # L == 1
    q_out, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    queries, gate = mx.split(q_out.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    queries, keys = self.rotary_emb.apply_rotary(queries, keys, position_ids, unsqueeze_dim=1)
    outs = []
    for i, c in enumerate(cache.caches):                # per row: same calls as B=1
        k, v = c.update_and_fetch(keys[i:i+1], values[i:i+1])
        outs.append(q35.scaled_dot_product_attention(queries[i:i+1], k, v, cache=c,
                                                     scale=self.scale, mask=None))
    out = mx.concatenate(outs, 0).transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(out * mx.sigmoid(gate))
```

* After the step, each row runs `quantize(row.cache)`. This is `maybe_quantize_kv_cache`
  per row, the same per-row conversion at 5000 tokens as today (1346-1348, 1365).
* Each row's `pos += 1`.
* The step does not depend on `lm._rope_deltas`/`_position_ids`, so a newcomer's prefill
  between steps cannot disturb running rows.
* Per-row cost: about 8 extra small ops per full-attention layer per row (9B: 8 layers). The
  step-ahead overlaps this graph building with the GPU. `microbench_attn_rows.py` measures
  the GPU side.
* If the per-row loop turns out expensive at B≥16, the V2 layouts are:
  * fp16 KV: a "slot" buffer `(Bmax, H, Lmax, D)` per layer with one masked SDPA call
    (`masked-fp16` in the microbench);
  * quantized KV: one masked quantized SDPA (mlx_lm base.py:66-101 accepts a mask);
  * or a right-ragged variant of mlx-vlm's kernel with a smaller threadgroup.

  All of these give up the "same object as B=1" property, so they are not V1.

### 2.7 Logits post-processing and sampling

Per row, on `logits[i:i+1]`, in exactly `_run`'s order (1356-1369):
1. `pen.add(input)` unless `row.first`;
2. `pen(logits)`;
3. `ctl.n += 1` unless `row.first`;
4. `ctl.mask`;
5. guide mask;
6. `row.sampler`.

Every request keeps its own sampler closure and RNG key (worker.py:411-454), so seeds stay
per-request. The results are concatenated to `nxt (B,)`.

* Greedy rows: `argmax(l - logsumexp(l))` per row; the same op as B=1.
* An optional later optimisation is one batched argmax over all plain greedy rows. Check the
  bit-equality first. `microbench_sampling.py` measures whether it matters.
* Guide rows:
  * one numpy bitmask `(B, ⌈V/32⌉)` (`llguidance.numpy.allocate_token_bitmask(B, V)`
    initialises to all-allowed);
  * each guided row fills its own row (`fill_next_token_bitmask(matcher, mask, index=i)`);
    an inactive guide (still in `<think>`) copies its `no_eos` row;
  * one `llguidance.mlx.apply_token_bitmask(logits, mask)`. The kernel is batched and
    elementwise (llguidance/mlx.py), so each row is identical to B=1;
  * `QWEN_JSON_MAX_WS` capping stays per row (`_Guide.sample` 1009-1012).

**Step-ahead.**
* `ahead = not sync`. In ahead mode the loop mirrors `_run` (1405-1435): queue
  `nxt = step(cur)` with `async_eval`, then `toks = cur.tolist()` (one host sync for the
  batch), then process the tokens.
* In **sync mode** (any guide row or multi-token `bad_words` row in the batch):
  `toks = cur.tolist()`, then per-row `guide.consume`/`ctl.history.append`, then build the
  masks, then queue the step. This is `_run`'s non-ahead order (1421-1426).
* Sync mode costs the whole batch the CPU/GPU overlap: ~3% at B=1 (30 vs 31 tok/s, CLAUDE.md),
  measured by `microbench_decode_batch.py MB_SYNC=1`. It lasts only while such a row is in
  the batch.

**Thinking budget.** It is batchable cheaply:
* `row.thinking(tok)` decides on the host;
* the forced token patches the row's entry of the lazy `nxt` with
  `nxt = mx.where(mx.arange(B) == i, forced, nxt)` before the next step is queued;
* this is the same point where `_run` calls `apply_forced_token` (1432-1433).

### 2.8 Per-token host work, events, stops, cancel

For each row `i`, with `tok = toks[i]`:

| case | action |
|---|---|
| `row.first` | The token is the last prompt token, not output. Clear `first` and emit nothing. |
| `row.n == row.max_tokens` | Finish `"length"`. The extra step already queued for that row is wasted (harmless). |
| `ctl.stop(tok)` / `stopping_criteria(tok)` | `"stop"`: finish, the token is not shown. `"skip"`: an ignored EOS; continue without showing it (1416-1428). |
| otherwise | `out.put(("tok", tok))`, `row.n += 1`. |
| `req.cancel` set (Cancel RPC, gRPC termination callback, handler `finally`) | Finish `"cancelled"`. |
| `req.stop_req` set (the handler's `_StopFilter` hit) | Evict silently. The handler already ended the answer. |

**Handler side.** The existing `generate` body moves over mostly verbatim (1721-1799):
* each request has its own detokenizer (`make_streaming_detokenizer` copies,
  tokenizer_utils.py:406-410);
* the split / `_GenState` / `_StopFilter` / tool parsing / `_done` code is unchanged;
* a stop-string hit sets `stop_req`, flushes and sends `Done(finish="stop")` itself, and
  ignores late tokens;
* `gen_tokens` = tokens the handler consumed (as today).

`context.add_callback(req.cancel.set)` in `Generate` makes a dropped stream free its row even
without the gateway's Cancel.

**Eviction**, between steps:
* keep = surviving indices;
* `cur = cur[keep]` and `gdn = take_gdn(gdn, keep)`, both lazy gathers of the in-flight
  step's outputs;
* drop the evicted rows' caches;
* release their reservations;
* `clear_cache` if `CLEAR_CACHE`;
* `B → 1`: `split_gdn` hands the survivor its GDN state back, and it continues on the plain
  library path.

**Join**:
* B=0: the row becomes the batch.
* B=1→2: `stack_gdn` of the two rows.
* B≥2: `concat_gdn(gdn, stack_gdn([row]))`.
* Then `cur = concat([cur, [row.y]])`.
* The row's own `ArraysCache` objects are emptied on stacking, so no stale state survives
  (batching_proto.py `stack_gdn`).
* A 1-token prompt has `None` states. It gets the zeros the library would create
  (`gdn_zero_state`, matching language.py:1597-1601/1644 + gated_delta.py:32-35).
* Assert `state.shape[0] == B` before each step. GDN would otherwise silently reset
  (language.py:1592, 1638).

### 2.9 Prompt cache, GDN checkpoints, `_rope_deltas`: per row

* Lookup, restore, checkpoints at slab ends and the store all happen inside the row's
  **prefill**. That code is unchanged and runs on the row's plain caches.
* Decode never touches the prompt cache. Today the snapshot is taken at `prompt[:-1]` only
  (1700-1709).
* Two concurrent requests with a shared system prompt:
  * request A stores its entry when its prefill ends;
  * request B's prefill starts after that (prefill is serialized), so B hits A's entry;
  * this is better than today's queueing, which has the same effect.
* `_copy_cache` is unchanged. Rows' KV objects are never shared between entries and batch:
  * stacking copies GDN arrays;
  * KV stays in the row's own objects;
  * stored entries are slices, which are new Python objects (the existing invariant,
    556-565).
* `_rope_deltas`/`_position_ids` are LanguageModel instance state:
  * read and written only by prefill (`get_input_embeddings` resets `_position_ids`,
    qwen3_5.py:51; first slab `get_rope_index`; the cache-hit reset at 1669-1670);
  * one prefill at a time;
  * decode always passes explicit positions.

  So the shared state is safe.
* PROMPT_CACHE_GB counts against the batch memory budget (§3).

### 2.10 Feature matrix

| request kind | prefill | decode | notes |
|---|---|---|---|
| text, thinking on/off | serialized slab loop | **batched** | `inside` from the prompt ids, unchanged (1627-1630) |
| thinking budget | " | **batched** | forced token patched into `nxt` (§2.7) |
| stop strings / stop_token_ids / ignore_eos / min_tokens / allowed_token_ids / single-token bad_words | " | **batched** | `_Control` per row; masks are per-row `mx.where` |
| multi-token bad_words (`ctl.sync`) | " | batched, **sync mode** | |
| response_format / constrained tools (`_Guide`) | " | batched, **sync mode** | one batched bitmask apply |
| penalties, temperature/top-p/top-k/min-p, seed | " | **batched** | per-row sampler and key |
| tools (auto, parse at end) | " | **batched** | parsing is handler-side |
| vision | serialized; `prepare_inputs` runs on the scheduler at admission (image token count → `_limit`) | **batched** | per-row `delta` from `_rope_deltas` after the prefill |
| DFlash-eligible (greedy, text, no budget, fp16 KV) | DFlash prefill | **solo**; demoted to a batch row at a round boundary when another request arrives | not re-promoted in V1 (needs the drafter's hidden window, §6 V2) |
| prompt-cache hit | restore + shorter prefill | **batched** | |

DFlash interaction in detail:
* `_run_spec` (1442-1584) becomes round-resumable: the scheduler calls one round at a time
  while the row is alone.
* After a round, the target cache holds everything before `b = new[-1]` (`_speculative_walk`
  + `rollback_speculative_cache`, 1559-1575). `b` is emitted but not yet fed. That is
  exactly a normal row with `y = b`, `first = False`.
* Demotion = drop the drafter state, set `row.y = b`, `row.pos = cache offset`, join.
* `QWEN_DRAFT_POLICY=solo|off` (default solo).
* Under steady concurrency the 9B runs batched without DFlash. Batching wins on aggregate:
  ×3 at B=8 vs DFlash's ×1.5 for one stream.

### 2.11 Errors and robustness

* An exception in a row's host post-processing (e.g. `grammar: token rejected`, 995) → that
  row gets an `Error` event and is evicted. The batch continues.
* An exception in the batched forward or step (MLX error, OOM) → every row in the step gets
  `Error`. The scheduler resets its state, drops caches, clears the buffer pool and goes on
  with the waiting queue.
* A failure inside `_prefill_iter` hits only that request.
* Handler threads never touch MLX, so a handler crash cannot corrupt the model state.
* `Shutdown` → scheduler stops after the current step; rows get `Error(kind="shutdown")`.

---

## 3. Memory accounting

Bytes per token per row (full-attention layers only), and GDN state per row. These are
computed from config.json and match `kv_bytes_per_token` / `gdn_bytes_per_row` in
`microbench_decode_batch.py`:

| model | FA layers × KV heads × head_dim | fp16 | q8 | q4 | GDN state / row |
|---|---|---|---|---|---|
| 9B | 8 × 4 × 256 | 32,768 | 17,408 | 9,216 | 24 × (32·128·128·4 + 3·8192·2) = **51.5 MB** |
| 27B | 16 × 4 × 256 | 65,536 | 34,816 | 18,432 | 48 × (48·128·128·4 + 3·10240·2) = **154 MB** |
| 35B | 10 × 2 × 256 | 20,480 | 10,880 | 5,760 | 30 × (32·128·128·4 + 3·8192·2) = **64 MB** |

Quantized formula: `bits/8 + 4/64` B per element (uint32-packed values plus fp16 scale and
bias per 64-element group).

`kv_bytes(n)` for a q-KV model:
* `n ≤ 5000`: `n·fp16`;
* `n > 5000`: `n·q`, plus a **transient** `5000·fp16` while `to_quantized` builds the
  quantized copy (the reservation uses `fp16·min(n, 5000) + q·n`).
* All KV caches grow in 256-token steps by `concatenate`: a transient second copy of one
  layer's buffer. Reserve +5%.

**Budget.**
```
KV_BUDGET = cap − weights_active − PREFILL_RESERVE − PROMPT_CACHE_GB − slack
```
* `cap` = `QWEN_WIRED_LIMIT_GB` if set, else `max_recommended_working_set_size`: ~21.8 GB
  by default on this 32 GB M1 Max, 28 GB after `iogpu.wired_limit_mb=28672` (use 27).
* `weights_active` = `mx.get_active_memory()` right after load (+ drafter).
* `PREFILL_RESERVE` = peak slab activations. The attention scratch is bounded by
  `PREFILL_BUDGET × q_heads × 2 B`: 2048·32768·16·2 = 2.1 GB (9B, 35B), 3.2 GB (27B).
  Add ~0.4 GB of MLP activations, giving **≈2.5 GB (9B/35B), 3.6 GB (27B)**. Calibrate once
  from the worker log: peak − active after one max-context request.
* `slack` = 1 GB (buffer pool between `clear_cache`s, logits, fragmentation).
* Override with `QWEN_BATCH_KV_GB`.

Worked examples, with rows at 8K prompt + 1K max_tokens (9,216 tokens) and PC = prompt
cache:

| model, cap | weights | budget (PC off / PC 3 GB) | reservation per row | rows |
|---|---|---|---|---|
| 9B fp16 KV, cap 21.8 | 11.6 | 6.7 / 3.7 GB | 302 MB + 51.5 MB = 354 MB | **18 / 10** |
| 9B fp16 KV, cap 27 | 11.6 | 11.9 / 8.9 GB | 354 MB | 33 / 25 (max_batch caps it) |
| 27B q8, cap 27 | ~18.6 | 3.8 / 0.8 GB | 321 MB + 154 MB + 328 MB transient | **4–7 / 1** |
| 35B q4, cap 27 | ~17 | 6.5 / 3.5 GB | 53 MB + 64 MB + 102 MB transient | 29 / 16 |

* The 27B is the memory-tight one. Batch it with `QWEN_PROMPT_CACHE_GB=1`, shorter
  contexts, or q4 KV. Its GDN state alone is 154 MB per row.
* At 1K-token contexts the 9B reservation is ~85 MB per row, so B=16 costs ~1.4 GB.

**Interaction with CLEAR_CACHE and the wired limit.**
* `mx.clear_cache()` runs:
  * after every prefill slab (as today, 1394-1395);
  * every 256 batch steps (as today per request, 1430-1431);
  * whenever a row leaves, since its KV buffers return to the pool;
  * when the scheduler goes idle (today: after every request, 1812-1813).

  The second-request OOM that CLEAR_CACHE prevents came from prefill buffers of
  different sizes piling up in the pool. Serialized prefill + clearing per slab keeps that
  property. Decode buffers are small (`(B, V)` logits = 0.5 MB/row fp16).
* `mx.set_cache_limit(QWEN_CACHE_LIMIT_GB)` is unchanged.
* `wired_limit(model, [generation_stream])` (common.py:75-104) is entered **once** when the
  scheduler becomes busy and left when it goes idle. Today it wraps each `_run`. It must not
  wrap per row: its `__exit__` calls `mx.synchronize` and restores the limit under other
  rows' feet. `Engine.load`'s `mx.set_wired_limit(QWEN_WIRED_LIMIT_GB)` (1144-1154) is
  unchanged.

---

## 4. Exactness

**B=1 is byte-identical to today. What guarantees it:**
1. **Prefill** is today's code moved into a generator. The cuts (`_bounds`, `_slab`), calls,
   `quantize`, checkpoints and store are unchanged. Slab boundaries never depend on other
   requests.
2. **Decode at B=1** calls `lm(y[None], cache=row.cache, position_ids=…)` on the row's own
   plain caches. The explicit `(1,1) int32` position equals what `LanguageModel`
   computes (2474-2496: `arange(1) + cache_offset + _rope_deltas`, int32).
   **[CPU-verified bit-exact]** on dense and MoE: `explicit==auto positions True`. The rest
   of the graph is the same ops on the same shapes.
3. **Logits processing and sampling** keep the same order, the same functions and the same
   per-request key.
4. **Step-ahead and sync** follow the same schedule (it does not affect values anyway).
5. **Isolation.** Other requests can affect a row only through shared state:
   * `lm._rope_deltas`/`_position_ids`: never read by decode with explicit positions;
   * the buffer pool: no numerical effect.

So a request that **decodes alone** is identical, even if its prefill was interleaved with
others' decode.

Caveats:
* The first decode step today feeds `inputs_embeds[:, -1:]` (1398). The new path feeds the
  token id. For text, `get_input_embeddings` = `embed_tokens(ids)` (qwen3_5.py:50-54), the
  same gather, so they are identical. For vision the last prompt token is text (the
  generation prompt), so it is identical too.
* DFlash solo keeps its own path. Demotion produces output equal to the plain loop "up to
  floating point" (already the case, CLAUDE.md Gotchas).

**B>1, where numerics differ.** Per row, attention (the same SDPA kernel on the same cache
object) and GDN (rows independent in the kernel grid) are the same computation as solo.
Differences come only from ops whose kernel depends on the row count:
1. **`quantized_matmul` with M=B vs M=1.** On M1 Max (g13s), qmv runs below 6/10/14 rows by
   width and qmm_splitk runs at or above that (versions.md §2b). Accumulation order
   changes. The same applies to the fused GDN in-projections and `lm_head`.
2. **MoE.** `SwitchGLU` sorts tokens by expert when B·top_k ≥ 64 (B≥8 on the 35B). This is
   a different gather_qmm path.
3. **Batched reductions**, if greedy rows are later grouped into one argmax/logsumexp.
   Avoided in V1.
4. **Composition dependence.** A row's numbers depend on which other rows were in the batch
   at each step, so a greedy answer under load can differ run to run. Rows within one step
   cannot influence each other's values except through kernel choice (M). vLLM has the
   same property.
   * `QWEN_MAX_BATCH=1` restores determinism.
   * mlx-vlm ≥0.7 has a batch-invariant forward (`Qwen3_5BatchInvariantForward`, rows equal
     singleton decode; versions.md §3). After the library bump it is the candidate fix if
     invariance is wanted; measure its speed first.

**How to test (GPU, §6):**
* **Identity.** A corpus of ~40 requests covering:
  * thinking on/off, tools, response_format, bad_words, stop strings;
  * vision;
  * prompt-cache hit and miss;
  * DFlash on the 9B;
  * long (30K) prompts.

  Run it sequentially on (a) the current worker (a git worktree of HEAD) and (b) the new
  worker with `QWEN_MAX_BATCH=8`. Compare the full JSON responses byte for byte. Repeat (b)
  with a background stream of other requests that only **overlap the prefills** (verify
  isolation).
* **Agreement.** A teacher-forced script feeds N=32 prompts' solo greedy tokens back as
  inputs in batches of 2/4/8/16. Record per-position top-1 agreement and KL(solo‖batched) of
  the full distributions. Expect top-1 ≥ 99% and KL ≤ 1e-3, the same order as the fp16-cast
  study: KL 3e-4, top-1 99.4%. Also free-running greedy: the fraction of identical answers
  and the first-divergence index.

---

## 5. Gateway changes

**proto** (`src/proto/worker.proto`):
```proto
message HealthReply { bool ready = 1; string model = 2; uint32 load_ms = 3; string model_path = 4;
                      uint32 max_batch = 5;   // concurrent generations the worker batches (0 = old worker = 1)
                      uint32 active = 6; uint32 waiting = 7; }   // optional: live scheduler state
message Usage { ...; uint32 queue_ms = 7;      // worker-side wait before prefill
                     uint32 prefill_ms = 8; float mean_batch = 9; }  // mean batch size over this request's decode steps
```
Regenerate both sides (gateway `build.rs`; worker `gen_proto.sh`). Old gateways (the user's
:8080/:8081 binaries) ignore the new fields. They keep 1 in flight each, and the worker
batches across them naturally.

**Gateway Rust:**
* `worker_client::connect` (worker_client.rs:14-37) returns `(client, model, max_batch)`.
* `main.rs:41`:
  ```rust
  let cap = env("GATEWAY_MAX_INFLIGHT").or(health.max_batch).max(1);
  Pool::new(max_queue, cap)
  ```
  Warn if `MAX_QUEUE < cap`.
* `pool.rs`:
  * `inflight: Arc<Semaphore::new(cap)>` replaces `Semaphore::new(1)` (21); keep `cap`;
  * `acquire_inflight` sets `qwen_in_flight` = `cap − available_permits()`;
  * `release_inflight` (41-43, called from `ReqGuard::drop` after the permit is dropped,
    api.rs:169-172) sets the same count instead of `0.0`;
  * `qwen_queue_depth` semantics are unchanged (waiting + running).
* `api.rs`:
  * `ReqGuard` is unchanged. It already owns both permits and fires `Cancel` on drop.
  * On `Done`, record `qwen_worker_queue_seconds` (histogram from `usage.queue_ms`) and
    `qwen_batch_size` (histogram from `usage.mean_batch`).
* `metrics.rs`:
  * describe the two histograms and `qwen_inflight_capacity` (gauge, set once);
  * `qwen_in_flight` becomes "generations in flight (count)".
* `MAX_QUEUE` semantics are unchanged: the total of waiting + running before 503.
* Optional V2: a background task polls `Health` every 5 s:
  * export `qwen_worker_active_rows`/`qwen_worker_waiting`;
  * notice a worker restart with a different `max_batch` (then `add_permits`, or
    `forget_permits` with tokio ≥1.32).

  V1 reads it at connect. A mismatch is harmless: the worker queues internally.

**Worker gRPC server:**
* `ThreadPoolExecutor(max_workers=4)` (worker.py:2007) → `QWEN_GRPC_THREADS`, default
  `2·QWEN_MAX_BATCH + 8`.
* Each streaming `Generate` occupies a thread for its whole life. With 4 threads, 4 batched
  streams would starve `Check`, `Cancel` and `Health`: new requests' pre-queue `Check`
  stalls and cancels are not delivered.
* Several gateways can each hold `max_batch` streams. Extra Generates wait in the worker's
  internal queue, which also holds a thread, hence the ×2.

---

## 6. Implementation plan

### 6.1 Steps (each step leaves a working worker)

**M0, before any code: run the microbenchmarks (§7).** Go/no-go and defaults (`QWEN_MAX_BATCH`
per model, whether V2 attention layouts are needed).

**M1, the scheduler split with no behaviour change** (`QWEN_MAX_BATCH=1`).
* Add `_Req`, `_Row`, `Scheduler` (thread, inbox + `threading.Condition`, waiting deque,
  idle/busy with `wired_limit`).
* Move the `generate` body into:
  * `Engine.generate` (handler): `_render`; `_prepare` for text without `mx.array`; `_limit`
    (text); Start event; enqueue; loop on `req.out` → detok/split/stop/tool parse/`_done`.
    It keeps 1721-1799 nearly verbatim.
  * `Scheduler._prefill_iter(req)` (from 1609-1709 + `_run` 1371-1397).
  * `Scheduler._decode_step()` for B=1 (from `_run` 1351-1369 + 1405-1440, positions
    explicit).
* `_run_spec` keeps working solo, through a generator wrapper.
* `WorkerServicer.Generate` gets `context.add_callback`; Health gets `max_batch`;
  `QWEN_GRPC_THREADS`.
* Test: the identity corpus (§4) is byte-identical to HEAD. This milestone is shippable on
  its own.

**M2, batched decode for plain text rows.**
* Lift from `batching_proto.py`: `RowsKV`, `_rows_attention`, `_install_rows_attention()`
  (called in `load`), `stack_gdn`/`take_gdn`/`concat_gdn`/`split_gdn`/`gdn_zero_state`,
  `batch_logits`.
* `_decode_step` for B≥2; join/evict/B transitions (§2.8); per-row post-processing
  (§2.7-2.8).
* Admission and reservation: `_kv_bytes_per_token(lm, bits)`, `_gdn_bytes(lm)`,
  `_kv_budget()`, `_admit_ok()`.
* Env: `QWEN_MAX_BATCH`, `QWEN_BATCH_KV_GB`, `QWEN_BATCH_PREFILL_SHARE`.
* proto + gateway (§5).
* Rows that need sync, guide, thinking budget, vision or DFlash still run alone in this
  milestone. The scheduler admits them only into an empty batch and lets nobody join.

**M3, the remaining request kinds in the batch:**
* sync mode (guide bitmask batch, `ctl.sync`);
* thinking-budget forced-token patch;
* vision rows (scheduler-side `prepare_inputs`; `delta` capture);
* `Usage` stats (`queue_ms`, `prefill_ms`, `mean_batch`).

**M4: DFlash solo + demotion; interleave tuning; docs.**
* README: new env knobs and the "Context & memory" batching budget.
* CLAUDE.md: the scheduler rule "MLX only on the scheduler thread", the attention override,
  "diff `Qwen3_5Attention.__call__` against `_rows_attention` when bumping mlx-vlm".

**V2 (only if measurements ask for it):**
* piggyback decode on prefill slabs;
* slot KV + masked/ragged SDPA;
* grouped greedy argmax;
* chunked reservation with pause (for clients without `max_tokens`);
* DFlash re-promotion (capture target-layer hiddens in the batch forward for eligible
  rows);
* batch-invariant kernels after the mlx-vlm bump.

### 6.2 Diff size (estimate)

| file | + | − |
|---|---|---|
| worker.py: scheduler, rows, admission, handler rewrite | ~650 | ~180 |
| worker.py: attention override + GDN helpers (from proto) | ~120 | 0 |
| worker.proto | 6 | 1 |
| gateway pool.rs / main.rs / worker_client.rs / api.rs / metrics.rs | ~60 | ~10 |
| src/worker/test_batching.py (CPU unit tests) | ~350 | — |
| README / CLAUDE.md | ~60 | ~10 |

### 6.3 Risks

| risk | mitigation |
|---|---|
| Hidden cross-thread MLX use (e.g. `_Penalties` built in the handler) → sporadic crashes | Rule: MLX only on the scheduler. A debug assertion in helpers that create arrays (`threading.current_thread() is scheduler`). |
| GIL/host overhead per row (graph building, `.tolist()`, events) eats the gain at B=16 | Step-ahead overlaps it. `microbench_sampling.py` and the `rows` mode measure it. Group greedy rows if needed. |
| Per-row attention loop overhead | `microbench_attn_rows.py`; V2 layouts. |
| Memory estimate wrong → OOM | Reservation + live `get_active_memory` guard + conservative PREFILL_RESERVE; test with long rows near the budget. |
| Silent GDN state reset on a shape mismatch (language.py:1592/1638) | Assert stacked shapes every step (cheap: shapes are host metadata). |
| Prefill stalls hurt ITL of running rows | `QWEN_BATCH_PREFILL_SHARE`, `QWEN_PREFILL_STEP=512`, V2 piggyback. |
| Numerics under load are not reproducible | Document; `QWEN_MAX_BATCH=1`; batch-invariant forward after the bump. |
| mlx-vlm bump changes `Qwen3_5Attention.__call__` | The override is ~25 lines, diffed on bump (same discipline as `_run`). The tiny-model CPU test catches drift. |
| gRPC thread exhaustion | `QWEN_GRPC_THREADS` default `2·max_batch+8`; a test with 2× max_batch concurrent streams + Check latency. |
| DFlash demotion bugs (cache/`b` bookkeeping) | Unit-test demotion on the tiny model (fp16 KV, without the drafter: simulate a round with `trim`), and a GPU equivalence test vs the plain loop up to fp noise. |
| Requests without `max_tokens` reserve 128K and block batching | Gateway `GATEWAY_DEFAULT_MAX_TOKENS`; V2 chunked reservation. |
| Two suspected position bugs (appendix A) confound the exactness tests | Fix or rule them out first. They are independent of batching. |

### 6.4 Test plan

**CPU unit tests (no lease):** `src/worker/test_batching.py`, run with
`src/worker/.venv/bin/python -m pytest`. They build tiny random qwen3_5/qwen3_5_moe models
like `design/test_batching_proto.py`: CPU device, `rope_utils._HAS_METAL=False`. Cases:
* batched step == per-row step:
  * dense/MoE; fp32 (≤1e-5) / fp16 (exact on CPU); quantized weights 8/4-bit;
  * fp16/q8/q4 KV; KV crossing `quantized_kv_start` (set small) mid-batch.
* join / leave / B→1 split / 1→2 stack; a 1-token prompt (None GDN state); a vision-like row
  with a non-zero `delta`.
* explicit positions == computed positions at B=1 (bit-exact).
* `_kv_bytes_per_token`/`_gdn_bytes` == actual `nbytes` of tiny caches; the reservation
  formula across the 5000 quantization boundary.
* Scheduler state machine with a fake model step (pure Python, injected):
  * FIFO / head-of-line, `max_batch`, budget refusal, an empty batch always admits;
  * cancel while waiting / in prefill / in decode; stop-string eviction;
  * per-row `max_tokens`; error isolation (one row raises);
  * sync-mode switching when a guide row joins and leaves;
  * thinking forced-token patch.
* Handler post-processing parity: feed a recorded token stream through the new handler and
  through the old `generate` loop. Events must be identical (tools, reasoning split, stop
  strings with `include_stop_str_in_output`, `cut`).

**GPU tests (lease holder),** 9B first, then 27B and 35B smoke:
1. **B=1 identity** (§4 corpus) vs HEAD: byte-identical. Also with overlapping prefills.
2. **Agreement B=2/4/8/16**, teacher-forced and free-running (§4).
3. **Cancel mid-batch:** 8 streams, cancel 3 at random points (gateway disconnect and the
   Cancel RPC). The others finish correctly; the in-flight gauge returns to 0; worker
   memory goes back to baseline after idle.
4. **Stop strings:** per-row different stops, including a stop split across tokens. Only
   the hit rows end; `finish=stop`; no stop text leaks.
5. **max_tokens per row** (1, 7, 64, 512 mixed): exact counts, `finish=length`.
6. **Thinking on/off mixed** in one batch, plus one row with `thinking_budget=32`: the
   reasoning/content split is correct and the budget is enforced.
7. **Mixed prompt lengths** (100, 2K, 8K, 30K) arriving while 6 rows decode: TTFT of the
   newcomers and ITL p50/p95 of the running rows vs `QWEN_BATCH_PREFILL_SHARE` ∈ {0.25,
   0.5, 1}.
8. **503 overflow:** `MAX_QUEUE=4`, cap 2, 10 concurrent clients: exactly the overflow gets
   503 and the rest complete.
9. **Prompt cache while batching:** 8 concurrent conversations sharing a 7.5K system prompt.
   The 2nd..8th get HITs; TTFT drops as in the README (14.4 → 0.2 s); outputs equal the
   B=1 outputs where they decoded alone.
10. **Grammar in batch:** 4 JSON-schema rows + 4 free rows. JSON is valid; the free rows'
    tok/s drops by ≤ the measured sync-mode cost while guide rows are present.
11. **Vision row joins a text batch:** its answer equals the solo answer when it decodes
    alone. With others present it agrees (top-1) and the text rows are unaffected.
12. **Memory:** rows with 32K prompts until refusal. No OOM; refusal keeps the request
    waiting (not 503); peak ≤ cap.
13. **DFlash:** a solo request equals the plain loop up to fp noise (as today). A second
    request arriving mid-generation demotes the first. Both answers are correct;
    `dflash:` log shows the rounds before demotion.
14. **Old + new gateways together** (:8080/:8081 untouched + :8090 new): all requests served
    and batched across gateways.
15. **Thread pool:** 2·max_batch concurrent streams; `Check` latency stays < 50 ms.
16. **Throughput table** (the brief's metrics) at concurrency 1/2/4/8/16. Expect aggregate
    ≈ `microbench_decode_batch` × (1 − prefill share).

---

## 7. Microbenchmarks to run before implementing (GPU owner)

All scripts are in `design/`. Run them from `src/worker` with `uv run python <path>` (the
worker venv), one at a time, with nothing else on the GPU. Each `--smoke` runs on the CPU
and needs no lease. Results go to `results/*.json` and stdout.

1. **`microbench_decode_batch.py`.** The ceiling, and the main go/no-go.
   * Loads the model exactly like `Engine.load` (non-strict for text-only checkpoints,
     `QWEN_FP16` cast, wired limit, `set_cache_limit`) and prefills one row with today's
     slab loop.
   * Clones it into B independent rows and runs the pipelined greedy loop
     (`async_eval` + `.tolist()` like `_run`).
   * `lib1` = today's decode (the B=1 baseline). `rows` = this design. `lib` = mlx-vlm
     `BatchKVCache` (fp16 only).
   * Reports ms/step, aggregate and per-row tok/s, the speedup vs B=1, and peak memory.
     Skips configs whose KV+GDN estimate exceeds `MB_MEM_GB`.
   ```
   cd src/worker
   QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 uv run python <sp>/design/microbench_decode_batch.py
   MB_SYNC=1 MB_BATCHES=1,8 MB_CTX=1024 ... (sync-mode cost)
   MB_ROW_SAMPLER=0 ...                      (batched argmax: host overhead)
   QWEN_MODEL=../../weights/qwen38-27b-mlx-5bit QWEN_KV_BITS=8 QWEN_WIRED_LIMIT_GB=27 MB_MEM_GB=4 MB_BATCHES=1,2,4,8 ...
   QWEN_MODEL=../../weights/qwen36-35b-a3b-mlx-ud3 QWEN_KV_BITS=4 QWEN_WIRED_LIMIT_GB=27 MB_BATCHES=1,2,4,8,16 ...
   MB_MODES=lib MB_RAGGED=1 ...  (LAST, alone: may hit the M1 ragged-kernel threadgroup failure, vlm#1799)
   ```
   Decisions:
   * go if `rows` B=8 at 1K ≥ ×2.5 aggregate on the 9B;
   * pick `QWEN_MAX_BATCH` per model at the knee;
   * if `rows` trails `lib` by > 10% at B=16 → plan the V2 attention layout.
2. **`microbench_attn_rows.py [model_dir]`.** Per layer, at ctx 1K/8K/32K and B=1..16:
   * per-row loop (fp16, q8, q4);
   * one `BatchKVCache` SDPA;
   * mlx-vlm's ragged kernel;
   * a masked SDPA over a shared buffer;
   * the Python graph-build time.

   Multiply by the model's full-attention layer count (printed). Decision: per-row
   overhead × layers > ~5% of the step time at the target B → V2 layout.
3. **`microbench_qmm_rows.py [model_dirs]`.** Per model:
   * every decode matmul at M=1..32 with random weights of the real shapes and per-module
     bits (UD quant of the 35B: attn/GDN 8-bit, shared expert 4-bit, routed experts 3-bit,
     lm_head 6-bit);
   * distinct weight copies so the SLC does not flatter small matrices;
   * the summed whole-forward factor `f(M)`;
   * for 4/5-bit (27B), mlx-vlm's multi-row verify-qmv kernel as an alternative;
   * for the 35B, `SwitchGLU` with random routing (sorted path from M·top_k ≥ 64).

   It complements the other agent's `design/microbench_qmm.py` (9B, chained, `--vlm-verify`).
   Decisions: the expected gain per model; whether the 27B should route `(B,1,K)` through
   the verify kernel; the MoE batch threshold.
4. **`microbench_sampling.py`.** Logits post-processing at B=1..32 over V=248320:
   * per-row vs batched greedy;
   * the worker sampler per row;
   * penalties per row;
   * the batched llguidance bitmask.

   Wall and graph-build ms. Decision: if per-row build time at B=16 is > ~5 ms/step,
   implement grouped greedy and vectorised penalties in M2 rather than V2.

Not a microbench but due before M2: confirm the two appendix-A suspicions on the GPU. They
would otherwise pollute the identity tests.

---

## Appendix A: incidental findings (not batching; verify on the GPU)

**A1. DFlash: the last prompt token may get RoPE position 0.**

What happens:
* `_run_spec` sets `lm._rope_deltas = None` (worker.py:1495).
* It prefills through `lm.model` (positions computed from `cache.offset` inside the
  attention, correct).
* Then it calls `lm(input_ids[:, -1:], cache=…)` (1517).
* In `LanguageModel.__call__`, `_rope_deltas is None` selects the "first chunk" branch
  (language.py:2433-2441). That branch slices a stale `self._position_ids` only if one of
  the right shape exists (2442-2458). Otherwise it calls `get_rope_index(inputs)` on the
  1-token input, i.e. **position 0** (2364-2367).

`_position_ids` is `None`:
* after a prompt-cache hit (worker.py:1670);
* at startup.

It is `[[0]]` after a previous DFlash request. It is also too short whenever the prompt is
longer than the previous plain request's first slab.

So the last prompt token's query and stored key are rotated as position 0. The first
generated token and every later attention to that key are slightly off. The verify rounds
after it are correct (`_rope_deltas` now zeros).

Check:
* log `position_ids` there;
* or compare first-token logits of `_run_spec` vs `_run` on a > 2048-token prompt after a
  cache hit.

Fix: set `lm._rope_deltas = mx.zeros((1, 1), mx.int32)` and `lm._position_ids = None` at
1495 (text-only path), or pass `position_ids` explicitly to the `lm(...)` calls in
`_run_spec`.

**A2. Vision prompts spanning more than one prefill slab may get text positions after slab 1.**

What happens:
* `get_input_embeddings` sets `_position_ids` (3, 1, N) and the vision `_rope_deltas`
  (qwen3_5.py:85-90).
* Slab 1 (`cache_offset == 0`) slices `_position_ids` correctly (2442-2450).
* From slab 2 on, `_rope_deltas` is set and the offset ≠ 0, so the delta branch
  (2473-2496) gives `arange + offset + delta`. That is correct only for text tokens
  **after** the last image.

So image tokens in slab ≥2, and text between slab 1 and the image, get wrong mRoPE
positions. This is likely for 35B/27B runs with `QWEN_PREFILL_STEP=512`, or with a long
system prompt before the image.

Check: one image prompt with `QWEN_PREFILL_STEP=512` vs a step larger than the prompt.
Compare first-token logits.

Fix in `_run`: for vision slabs pass `position_ids=lm._position_ids[:, :, pos:pos + n]`.
mlx-vlm's own `generate_step` chunks the same way (ar.py:273-300), so this is upstream
behaviour.

**A3.** The README/CLAUDE.md "35B q4 KV ~2.6 GB @128K" is the fp16 figure. q4 for 10 FA layers
× 2 KV heads × 256 is 5,760 B/token, about 0.75 GB at 128K.

---

## Appendix B: quick reference (file:line)

* worker.py:
  * `make_sampler` 411; `_Penalties` 457; `CLEAR_CACHE` 502; `PROMPT_CACHE` 517;
  * `_copy_cache` 556; `_Entry` 617; `_PrefixCache` 641; `_step`/`_slab`/`_bounds`
    684/690/697;
  * `_StopFilter` 751; `_Guide` 957; `Engine` 1038 (`load` 1053, wired 1144, `cancel` 1162,
    `_render` 1208, `_limit` 1248, `check` 1267, `_prepare` 1284);
  * `_run` 1310 (step 1351, prefill loop 1381-1395, decode loop 1405-1435);
  * `_run_spec` 1442;
  * `generate` 1587 (lock 1594, reset_peak 1633, prompt cache 1655-1709, guide 1711, loop
    1749-1763, tools 1780-1798);
  * `_Control` 1816; `_GenState` 1890; `_done` 1937; `WorkerServicer` 1950;
    `ThreadPoolExecutor(4)` 2007.
* gateway:
  * pool.rs 10-52 (`Semaphore::new(1)` 21);
  * api.rs `ReqGuard` 159-182, `acquire_inflight` 193/349;
  * main.rs `Pool::new` 41; worker_client.rs 14-37; metrics.rs 18.
* proto: `HealthReply` 80, `Usage` 72-76.
* mlx-lm generate.py:
  * `_make_cache` 838, `_merge_caches` 870;
  * `PromptProcessingBatch` 1004 (`prompt` 1124);
  * `GenerationBatch` 1229 (`_step` 1320, `filter` 1383, `next` 1405);
  * `BatchGenerator` 1486 (`_next` 1769).
* mlx-lm cache.py:
  * `QuantizedKVCache` 232; `KVCache.merge` 397;
  * `ArraysCache` 594-728; `dynamic_roll` 903; `BatchKVCache` 912-1130.
* mlx-lm base.py `create_causal_mask` 24-42. qwen3_next.py GDN 236-306, masks 414-415.
  gated_delta.py masked kernel 13-116.
* mlx-vlm language.py:
  * `_extract_row_cache` 666, `_is_single_row_batch_cache` 688;
  * ssm/attn masks 769/798; ragged SDPA 1196; `_target_verify_left_padded_attention` 1289;
  * `Qwen3_5Attention.__call__` 1393; GDN `__call__` 1571; `Qwen3_5Model.__call__` 1763;
  * `get_rope_index` 2201; `LanguageModel.__call__` 2374; `make_cache` 2579.
* mlx-vlm elsewhere:
  * qwen3_5.py `get_input_embeddings` 36-94;
  * qwen3_5_moe/language.py 8, 35-73;
  * cache.py `BatchQuantizedKVCache` 181-377, `make_prompt_cache` 931;
  * generate/ar.py `_make_cache` 715, `GenerationBatch` 858, `PromptProcessingBatch` 1492,
    `BatchGenerator` 2086;
  * generate/common.py `maybe_quantize_kv_cache` 23, `wired_limit` 75;
  * rope_utils.py `compute_mrope_frequencies` 478-499;
  * mlx_lm switch_layers.py `SwitchGLU` 160-199.

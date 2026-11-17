# Scheduler: continuous batching

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- One scheduler thread owns MLX and decodes every running row in one forward per step;
  prefill one request at a time (plus the short-request lane), interleaved with decode.
- Admission is FIFO with reservation-based memory accounting; a growing row is preempted
  (back to the queue through the prompt cache) before it is cut with `length`.
- B ≤ 2 is bit-exact vs solo; B ≥ 3 and cascade groups are in the batched-decode class;
  `QWEN_BATCH_INVARIANT=1` gives determinism at any batch size.
- The loop is a trimmed copy of mlx-vlm's `generate_step`: diff it on every mlx-vlm bump.

## Implementation notes worth keeping

- **The worker owns its prefill/decode loop** (`_Scheduler._prefill` + `_iterate`), a trimmed copy of
  mlx-vlm 0.6.2 `generate_step`+`stream_generate`, re-checked against 0.7.6 (same sampler, logits
  processors, KV quantization, ThinkingBudgetCriteria, detokenizer). Reasons:
  the prompt-cache snapshot has to be taken at the exact point where the cache
  holds prompt[:-1] and nothing else (the library's `PromptCacheState` snapshots
  after the first generated token and trims — impossible for ArraysCache,
  broken for quantized tuples); pre-tokenized vision inputs go in directly;
  `finish_reason` `"length"` and pool clearing are ours. Output verified
  byte-identical to `stream_generate` at temperature 0. When bumping mlx-vlm,
  diff `generate/ar.py::generate_step` against `_prefill`/`_iterate`/`_sample`,
  `Qwen3_5Attention.__call__`/`_prepare_projected_qkv` against `_rows_attention`, and
  `Qwen3_5GatedDeltaNet.__call__` against `_gdn_call`. 0.7 returns `position_ids`/
  `rope_deltas` from `get_input_embeddings` (counted from 0): text drops them (a
  prompt-cache suffix would land at 0), vision slices them per slab. DFlash verify
  is `speculative_verify=True` + `commit_/abort_speculative_round` every round.

## Continuous batching invariants

- **Continuous batching (`_Scheduler`, worker.py)**. Invariants:
  - MLX graphs are built and evaluated only on the scheduler thread after load
    (MLX ops on two threads at once crashed in tests). Handlers render, tokenize,
    run the image processor — numpy, whose `to_mlx` wraps the results as
    *constant* MLX arrays on the handler thread (allocations only, no graph or
    stream; read back to numpy at once) — so a slow image URL never stalls the
    batch; build `_Control`/`_Guide` (no MLX); then turn token ids into events.
    The processor runs with `padding=False` (padding would flip the shared Rust
    tokenizer's state under other handlers: "Already borrowed"). The scheduler
    thread never returns: MLX tears down its thread-local stream on exit, which
    segfaulted the process.
  - B=1 = the plain library call on the row's own caches with an explicit position
    → byte-identical to the pre-batching worker (GPU: 15/15 corpus at max_batch 1
    and 16). B≥2: `RowsKV` + `_rows_attention` (projections batched, cache update +
    SDPA per row with the B=1 calls), stacked GDN, positions `(B,1)` = KV length +
    the row's mRoPE delta; decode never reads `_rope_deltas`/`_position_ids` (only
    the one running prefill writes them). Rows differ from solo only by the matmul
    kernels' row-count rounding: B≤2 always bit-exact (per-row qmv); 3..16 rows
    take mrqmm on 8-bit modules (`QWEN_MRQMM`), qmv/qmm on the others. Measured
    with stock MLX: 9B teacher-forced 100% top-1 at B=8 (KL 9e-6), 99.8% at 16;
    with mrqmm 100% at B=3/4/8 (KL 8e-6), 99.7% at 16 (KL 1e-5) — so
    under load a greedy answer can differ run to run; `QWEN_MAX_BATCH=1` restores
    determinism (+ `QWEN_PROMPT_CACHE_REPLY=0`: a hit through a reply differs from
    cold even at B = 1). **Batch-invariant mode (`QWEN_BATCH_INVARIANT=1`, 2026-10-24,
    `docs/perf/2026-10-24/design/batch_invariant.md`)**: determinism at any batch size
    instead: mrqmm with one fixed KS (8) for 1..16 rows and also at B ≤ 2 (plain linears,
    lm_head, fused GDN), cascade and DFlash off — a row in a batch of N == the same row
    solo, bit for bit (9B GPU gate: 5947 teacher-forced steps at B = 1..16, logprobs and
    top-5 `==`; control with the mode off diverges). Step cost vs off: B=1 ×1.18, B=2
    ×0.86, B=4/8 ×1.00, B=16 ×0.97 → for reproducible agentic runs use it with
    `QWEN_PROMPT_CACHE_REPLY=0` at `QWEN_MAX_BATCH=16` instead of `MAX_BATCH=1/2`.
    Models with modules mrqmm does not take (27B 5-bit, 35B experts / lm_head,
    `QWEN_MRQMM=0`, `QWEN_FP16=0`) get max_batch 2 (per-row qmv exact there); an
    unquantized `nn.Linear` turns the mode off. Rests on MMA results not depending on
    the column's position and on the MB=1/2 kernel variants compiling `acc += red[s]`
    alike (true on the 9B shapes; others not gated). Side findings, not applied: at B=2
    mrqmm is 14 % faster than 2-row qmv; at 9..16 rows 2×8 beats the default 4×4 by 2.6 %. Cascade groups (default on with fp16 KV + the prompt cache: ≥ 8
    rows sharing ≥ 4096 prompt tokens, below) change only the grouped rows'
    attention, in the same numerics class as mrqmm (top-1 ≥ 99.71 %); B ≤ 2 stays
    bit-exact and `QWEN_CASCADE=0` restores this per-row path bit for bit.
  - Admission FIFO, no overtaking; an empty batch always admits. Reservation =
    prompt + min(max_tokens, `QWEN_BATCH_RESERVE_TOKENS` 4096) tokens of KV + 2×
    GDN state (quantized KV: + the fp16 warm-up copy, ≤ `quantized_kv_start` + a slab, until the row's cache is
    quantized — `_kvq_mark`, `row.kvq` — then N-bit only); rows are re-accounted at their real length every step (growth past
    the reservation is allowed and holds back new admissions; past budget +
    prefill reserve the growing row ends with `length` while other rows run — a
    lone row is never cut); the prompt cache's live bytes count beside the
    reservations and give way first (evicted on demand, `victim` order); live check `active +
    reserve + prefill reserve ≤ cap`. Cap in `_batch_cap`, budget in `_batch_budget`.
  - Time share: every prefill slab, the last one too (a cache hit's only slab),
    sets `resume` (`_slab_done`), so decode steps run between short prefills
    instead of waiting for the queue to drain (the 16 × 8K shared burst held every
    first token until the last hit was prefilled).
  - Step-ahead (`async_eval`) as before; a grammar or multi-token bad-words row
    puts the whole batch in sync mode while it is there. Thinking budget forces the
    row's next input. Vision: prefill serialized, decode batched with its delta.
    DFlash only for a request admitted with nothing else running or waiting
    (a thinking budget is honoured inside its rounds since 2026-11-11);
    demoted to a batch row at a round boundary when a waiting request would be
    admitted next to it.
  - Generate RPCs beyond `QWEN_GEN_SLOTS` (`max(2·max_batch, 8)`) →
    `RESOURCE_EXHAUSTED` (gateway 503); gRPC pool = slots + 8 threads, so
    Check/Cancel/Health have spare threads (grpc's own backstop refuses RPCs only
    past slots + 64 queued). A failed decode step fails its rows only; a lone
    row's failure is no strike. 3 consecutive failed steps with several rows close
    the scheduler: Error(worker_fault) to everything, Health `ready=false`, and
    the process exits (70) after 1 s so launchd restarts it. A tick running
    > 300 s only makes Health not ready (handlers waiting on it give up) until it
    ends; idle time never counts. Shutdown RPC / SIGTERM / Ctrl-C →
    Error(shutdown) to everything; exit(1) if the current tick does not end
    within 10 s.
  - `Usage.peak_gb` / log `peak=`: MLX peak since the batch was last empty, read by
    the scheduler when the request ends (= the process peak during the request
    when rows overlap). `Usage.queue_ms` / `batch_mean` / `cached_tokens` are
    worker-side (`queue_ms` = arrival → the first prefill slab, which waits for
    decode's share after admission; the worker's `ttft_ms` counts from there,
    `_advance`); the gateway returns `usage.prompt_tokens_details.cached_tokens`
    (always, 0 without a hit) and counts `qwen_gateway_cached_tokens_total`.

## Cascade groups

  - **Cascade groups (`QWEN_CASCADE`, default 1 since the B4 GPU gate, 2026-10-12)**:
    rows sharing prompt ids `[0, P)` with P a slab end of each (→ same KV bits:
    `_bounds` + hit == cold) decode attention as one `_PrefixGroup` (`cascade.PrefixGroup`:
    fp32 prefix once + stacked fp32 suffixes; `_rows_attention` → `_cascade_rows` with
    the step's `_StepPlan`; `begin_step`/`end_step` around the forward in
    `_grouped_forward`). Formed in `_join` (`_cascade_join`) when ≥ `QWEN_CASCADE_MIN_ROWS`
    (8) rows share ≥ `_MIN_PREFIX` (4096) tokens (earlier rows converted, their FA caches
    → `_GroupRef`, freed); later rows `add` (the row's own S-token copies evaluated, the
    stack concat lazy). A row a group refuses forms no second group on that group's prefix.
    Memory rule `_group_wins`: at the members' REAL lengths the group's charge
    (`_group_bytes`) ≤ their own fp16 caches (9B, ~150-token suffixes: 3 rows never win,
    4 only from 32K, 5–16 at 4K–32K). Budget guard `_charge_fits`: a formation / join
    that raises `reserved` is taken only while `reserved` + the prompt cache's live bytes
    ≤ budget (refused, nothing evicted; the charge is reservation-based and grows with
    `_regrow` like any row's, so growth can still pass the budget). `_compact` →
    `_cascade_tidy`: done rows `take`n out (lazy); forced dissolve at < 3 live members
    (covers B ≤ 2: today's bit-exact path), unchecked — its peak (2 contiguous rows beside
    the live group) is inside the charge by `_group_bytes`' floor (prefix + stacks + 2 ×
    (P + s) fp16); optional releases (a member whose own suffix > P//2, the longest first;
    a memory dissolve and `set_cascade(False)`, the cheapest first) one member per group
    and decode step (`released_at`; the last < 3 together), each checked with the group
    alive: `reserved` + the released rows' fp16 caches + prompt cache ≤ budget + prefill
    reserve, nothing credited for the group's shrink (release evaluates beside it, take is
    lazy) — else they stay grouped. A tidy that raises fails that group's rows only. No
    reference cycles: `_GroupRef` holds a weakref to its row and no group; `_unlink` on
    every drop (a leaked group would hold GBs until Python's gen-2 GC). Eligible: fp16 KV
    only, `QWEN_PROMPT_CACHE=1`, max_batch ≥ 3 (else off whatever the switch; the load log
    says why), text rows that resumed from the cache or stored an entry (`row.cached`),
    never vision or DFlash (`row.spec`, also after demotion). Numerics: the same class as
    mrqmm's batched decode (B4: top-1 ≥ 99.71 %, KL +0–13 % over today's per-row batch,
    per-layer error below the per-row kernel's); B ≤ 2 bit-exact; ungrouped rows of a
    mixed batch take the per-row calls verbatim (bit-identical; the output take is an
    exact copy); `QWEN_CASCADE=0` restores today's per-row path bit for bit. A released
    row's `[0, P)` are the cold bits; its suffix holds what it decoded while grouped — so
    B ≥ 3 greedy answers also depend on cache contents / arrival order (determinism:
    `QWEN_CASCADE=0` or `QWEN_MAX_BATCH=1`, with `QWEN_PROMPT_CACHE_REPLY=0`). Accounting: a group is charged once
    (`_group_bytes`: fp32 prefix padded to chunks + stack at the longest reservation ×1.05
    + max(score transients + one layer's old+new stack, the 2-row dissolve floor)),
    members keep `_member_bytes` (GDN). A failed step drops every group (`_fail_rows`).
    `_plan` raises on any member/batch/length disagreement (`check_lens` every step).
    Health `cascade_groups`/`cascade_rows`; `batch:` log `groups= grouped=`.
    `set_cascade(on)` for tools (scheduler thread only). Members are kept in batch order
    (a group = one slice of the batch: no gather / output take). Measured (9B,
    `docs/perf/2026-10-12/cascade_gpu.md`): step 8K × 16 rows 66.4 → 53.7 ms (×1.24),
    32K × 8 83.6 → 68.2 ms; 16 × 32K now all admitted (was 12 + 4 waiting), peak 25.3 →
    23.2 GB; 16 × 8K active −3 GiB. Caveats: formation builds the fp32 prefix while the
    members still hold their fp16 caches (+2.2 GB peak at 8 × 32K); 32K × 16 steps at
    92 ms vs ~62 projected (open, needs a profile). The floor and the staged releases
    came after B4 (CPU tests only). Tests `tests/test_cascade_sched.py` (+ `test_cascade.py`).

## RoPE position fixes

- **RoPE position bugs fixed with batching (both confirmed on the real models)**:
  (A1) DFlash's first target call after prefill rotated the last prompt token as
  position 0 (`_rope_deltas=None` → `get_rope_index` of a 1-token input; |Δlogit|
  0.88 on the 9B) — `_spec` now sets zero deltas. (A2) an image prompt longer
  than one prefill slab gave image tokens text positions from slab 2 on (|Δlogit|
  0.30 at slab 512) — vision slabs now pass `position_ids[..., pos:pos+n]`, sliced
  from what `get_input_embeddings` returns (0.7 no longer sets `_position_ids`).
  Outputs of DFlash and of multi-slab vision prompts therefore differ from older
  workers.

# Review: logprobs / prompt_logprobs (uncommitted diff, 2026-10-22)

Scope: `src/worker/worker.py`, `src/worker/tests/test_logprobs.py`, `src/gateway/src/{request,api,journal}.rs`,
`src/proto/worker.proto`, README "Log-probabilities", against `docs/perf/2026-10-21/design/logprobs.md`.
No code was changed. Runs: `pytest tests/test_logprobs.py` 26 passed in 4.5 s; whole CPU suite 305 passed /
2 skipped in 540 s; `cargo test -p qwen-gateway` 19 passed.

## Verdict

No HIGH bug found. **The off-path invariant holds.** With both flags off, the worker runs the same ops:

- prefill: the same `lm(..., skip_logits=True)` call, with no `return_hidden`;
- `_step`: `want` is empty, and `outs` is the old `async_eval` list;
- `_iterate`: `lps` is a list of `None`, and the `("tok", id, shown)` 3-tuple is unchanged;
- `_prompt_cache` / `_peek`: `_lookup_ids(req) is req.ids`, and `_no_lookup` is False when `prompt_cache` is absent;
- `_spec_ok`: falls through.

The empty `lp=[]` on `pb.Token` serializes to the same bytes. `split()` and the tools path never yielded
empty-text Tokens, so the gateway's `qwen_tokens_generated_total` / TTFT behaviour is unchanged for
ordinary requests. The one visible off-case change is in the gateway: non-stream JSON now always has
`choices[0].logprobs: null`. That is intended (OpenAI) and additive.

Step-ahead discipline also holds:

- lp/top are built in `_step` on the raw `x` and go into the same `mx.async_eval` as the tokens. They are read
  only after `r.y.item()`.
- The forced-token slice (`_forced_logprobs`) is queued after step k and before step k+1. It depends only on
  step k, so the next read does not wait on the newer step.
- Rows ended by `_regrow`, rows whose sampling failed, and `_fail_rows` all leave `lp_next` unused.
- Sync, ahead and mode-switch sequences all pair the read `lp` with the read `y`.

Prompt-side checks:

- **Alignment:** slab row j maps to `input_ids[0, j+1]`, and the last slab always scores `ids[n-1]`.
- **`ids[:-1]` lookup cap:** sound. `_Entry.best` caps `common` at `len(query) - 1`, which is `n - 2`, so the
  resume point is still a slab end of the *full* prompt (`_ends(req)`). Hit and cold therefore use the same
  slabs and the same lm_head chunks, and the values are bit-exact. The exception is a hit through a reply
  entry, which is documented.
- **`prompt_cache=false`:** places the divergence checkpoint via `_scan` (no side effects) and stores normally.
- **Failures:** cancel or failure mid-prefill sends no PromptLogprobs event and leaves `plp` as None. A `_score`
  exception stores the prefix and fails the request cleanly.
- **Hidden state:** `return_hidden` is `out` after the final norm. The `tie_word_embeddings` branch is right
  and tested.

## MED

### M1. `prompt_logprobs: 0` silently means "off" (vLLM: on)
`src/gateway/src/request.rs:103` (`prompt_logprobs_on`), `:303` (validate).

- **The difference:** in vLLM (`SamplingParams.prompt_logprobs: Optional[int]`), `0` means "return the log p of
  each prompt token, with no extra top-k". `None` is off; `1` is top-1 plus the actual token.
- **Failure:** a vLLM client sends `{"prompt_logprobs": 0, ...}` to get teacher-forcing values. It gets a 200
  with no `prompt_logprobs` key at all. With `max_tokens: 0` it gets a 400 ("max_tokens must be at least
  1"). Both break the project rule "anything vLLM does that we don't is a 400, never accept-and-ignore".
- **Fix:** treat any integer `>= 0` as on, with `0` and `1` both meaning "no top-k" (we return none). Refuse
  `> 1` and `-1` with the existing message, and keep `false`/`null` as off. Alternatively, refuse `0` with
  a 400 that says to use `true`. Update the `logprobs_validation` cases and the README.

### M2. A `max_tokens: 0` row joins the decode batch (GDN restack, cascade formation) before it ends
`worker.py:3286-3294` (`on_prefilled` → `row.y` → `_join`), `:3499` `_join` → `:3515` `_cascade_join`.

- **What happens:** a prefill-only row goes through `_join` like any other row. That means:
  - a GDN `_gdn_stack` / `_gdn_concat` with the running rows;
  - `_regrow`;
  - possibly `_cascade_join`: with ≥ 7 running rows that share ≥ 4096 prompt tokens with it (e.g. a TF
    variant of a trace that is also being generated), it can *form or join a cascade group*.

  On the next `_iterate`, `r.n == max_tokens` ends it, and `_drop_done` gathers its GDN state for
  `_reply_cache` (thrown away: `n < 1`) and re-takes the stack for every other row.
- **Cost:**
  - one extra B × 52 MB GDN copy (9B) per TF request while anything is decoding;
  - in the cascade case, group formation, which builds the fp32 prefix while members still hold fp16
    (measured +2.2 GB transient at 8 × 32K);
  - possibly a forced dissolve right after, plus a change of the other rows' numerics class for that
    window;
  - and all of it for a row that never decodes.
- **Fix:** in `_prefill`, after `on_prefilled(cache)` and the `plp` assignment:
  `if req.max_tokens == 0: self._finish(row, "length"); return` (with `row.pos = pos`, before `_join`).
  This is also simpler: no `_iterate` pass, no reply-cache gather, and the `_slab_done` charge stays.
  Keep the spec's "store as a generation would" (the store has already happened in `on_prefilled`).

### M3. Spans: on a block/message count mismatch, zipping from the start shifts every span
`worker.py:2381-2410` (`_assistant_spans`, `if len(blocks) != len(msgs)`).

- **Failure:** any message content containing the literal text `<|im_start|>assistant` adds a block, and the
  extra block usually comes *early*. Examples:
  - a system prompt with few-shot examples written in chat markup;
  - a tool response that echoes a transcript;
  - a user pasting a log.

  `tokenizer.encode(formatted)` turns that text into the special ids. The mismatch is only logged, and
  `zip(msgs, blocks)` maps assistant message k to block k. Every real assistant span after the injected
  block then points at the previous message's tokens. `prompt_logprobs_messages` silently attributes the
  wrong sums, and the client cannot see the warning.
- **Fix:**
  - Match from the end (`zip(reversed(msgs), reversed(blocks))`, after dropping the generation-prompt
    block). Injected markup usually precedes the real turns.
  - Better: anchor each block to a message boundary (the k-th `<|im_start|>` that starts a top-level
    message).
  - In either case, surface the mismatch in the response, e.g. return no spans or add a `spans_matched:
    false` field, rather than only logging it.

## LOW

### L1. `bytes` is lossy (U+FFFD) for byte-fallback / partial-UTF-8 tokens
`api.rs:85-92`, worker `_token_strs` (`decode([id])`).

- **Failure:** OpenAI's `bytes` exists to rebuild multi-byte characters. Here a CJK/emoji split across tokens
  gives `[239,191,189]` for each piece. The README documents this, but clients that join `bytes` get
  garbage.
- **Fix:** have the worker send the raw token bytes: the byte-level BPE `convert_ids_to_tokens` plus the
  byte decoder, as `_BPEDetokenizer` already does. Add them to `TokenLogprob` as `bytes raw = 5`.

### L2. The journal keeps the full `logprobs.content` (+ `top_logprobs`, `bytes` as int arrays)
`api.rs:418` (`for_journal`).

- **Failure:** only `prompt_logprobs` is trimmed. A 4K-token answer with `top_logprobs: 20` is about 21 entries
  per token, with byte arrays, which comes to several MB per journal line.
- **Fix:** also replace `choices[0].logprobs.content` by its length (or drop `top_logprobs`/`bytes`) in the
  journal.

### L3. `top_logprobs` cost per row at B ≥ 2
`worker.py:2336-2352` (`_token_logprobs`), `:4225`.

- **What it costs:** each logprobs row adds its own fp32 cast, logsumexp, full-vocab `argpartition` and
  small argsort launches. MLX's GPU argpartition is a sort over 248K. `sample_rows` exists precisely because
  16 per-row full-vocab topk/argsort launches cost about 15 ms per step.
- **Failure:** at B = 16 with every row asking for `top_logprobs`, expect a step-time hit of the same order
  (not measured; GPU not used). Rows without the flag still pay through the shared step.
- **Fix:** stack the logprobs rows into one (G, V) array per step, as `sample_rows` does. Row ops are
  bit-identical per row in a stack (that property is already tested).

### L4. "T=0: the drawn token is `top[0]`" is not guaranteed on the GPU
`worker.py:2336`, test `test_top_logprobs`.

- **Failure:** the greedy draw is `argmax` of `logits - logsumexp` in the **model dtype** (fp16). `top` is
  sorted in **fp32**. On an fp16 near-tie, the drawn token can be `top[1]`, or the two values can be equal
  with the opposite order.
- **Fix:** nothing to fix in code. Document it, and do not let clients (or the GPU check script) assert it.

### L5. `prompt_logprobs` + `stream: true` is accepted, unlike vLLM
`request.rs`: vLLM 400s with "prompt_logprobs are not available when stream=True".

This is a deliberate extension and the README says so. It is listed only because the rule is "vLLM
behaviour or 400". Fine as is if intended.

### L6. Attribution throughput: every variant stores a full entry
Spec invariant 4 is as designed, but the cost is worth noting.

- **What happens:** a 20–30K-token variant that diverges at d copies `[d, n-1)` of KV (~32 KB/token on the 9B,
  so up to ~0.6–0.9 GB per request) plus checkpoints. It then evicts down to the 3 GB soft cap.
- **Risk:** victim order "all live → MRU" protects the base, but variants that *append* to the base absorb it,
  and the chain policy then thins its checkpoints.
- **Suggestion:** `prompt_cache: "read"` (lookup, no store) for variants. It is cheaper and leaves the base
  entry untouched.

### L7. Spans exclude tokens the model did generate in two template cases (document)
- **Thinking on, reasoning sent back empty or dropped by the client:** the history renders `<think>\n\n</think>\n\n`,
  which matches skip[0], so the span starts at the content. During generation the model produced
  `\n</think>\n\n` itself.
- **35B / 27B templates:** assistant turns before the last user query render without a think block
  (`loop.index0 > ns.last_query_index`). The span is the content only, scored in a context the model never
  saw.

Both are template properties, not code bugs. Add one line to the README.

## Tests (`src/worker/tests/test_logprobs.py`)

- **Suite time:** the new file costs 4.5 s (slowest test 0.31 s; `--durations=15` in the report). The 432 → 540 s
  suite is dominated by the prompt-cache / segment / partial-store tests (`test_segments.py::
  test_entries_equal_cold_bitwise[fp16]` 40 s, `[q8]` 33 s, `test_partial_store::test_20k_fresh_entry…` 32 s,
  `test_round_robin_episodes_keep_hitting` 30 s, …).
- **Not the new code:** an A/B of `test_entries_equal_cold_bitwise` with HEAD's `worker.py` vs the current one
  (copies in a scratch dir, run alternately) gives 41 / 30 s (HEAD) vs 29 / 27 s (current). So the slowdown
  is not the logprobs change. The likely cause is CPU contention from the running benchmark, or those tests
  themselves.

Gaps (nothing found that passes *because* the feature is broken, but these claims are untested):

- **"Raw" w.r.t. penalties/temperature:** `test_batch_rows` (`:271`) comments "raw logprobs: not the tempered
  distribution" but asserts only `top == []`. Every lp/TF comparison runs at presence 0 / T 0, so lp computed
  from the penalized/tempered logits would also pass. Only `test_handler_teacher_forcing` checks rawness, and
  only w.r.t. masks (`allowed_token_ids`). Add: the same prompt with `presence_penalty=1.5, temperature=0.7,
  seed` gives lp equal to the TF values at the drawn tokens.
- **B ≥ 2 lp values:** `test_batch_rows` checks that tokens are equal and lp is present, but not that the lp
  values equal the solo (B = 1) values at B = 2, which is bit-exact (per-row qmv).
- **Not covered:**
  - a stop-string hit with logprobs (lp up to and including the hitting token, none after `_drain`);
  - tools mode (lp on empty Tokens, order versus the final content Token);
  - `ignore_eos` "skip" tokens;
  - a sync-mode (grammar) row with logprobs;
  - a thinking-budget forced token at B ≥ 2;
  - cancel mid-prefill with `prompt_logprobs` (no event, partial store, `plp` None);
  - a `_score` exception (error event plus prefix store);
  - a TF request whose prefix hits a *reply* entry (`first`, non-bit-exact by design);
  - a `max_tokens 0` row joining a running batch (M2).

## Simplifications (after the fixes)

- **`max_tokens == 0`:** end in `_prefill` (M2). It removes the special case from `_iterate`'s ordering
  argument.
- **`_no_lookup` + `_lookup_ids`:** fold into one `_lookup_ids(req) -> ids | None` (None means skip). `_peek`
  and `_prompt_cache` then branch once.
- **`PromptLogprobs.tokens`:** a `repeated TokenLogprob` whose `logprob`/`top` are unused. `repeated uint32 ids`
  + `repeated string strs` (or `bytes`, per L1) is smaller and clearer. About 14 B/token measured on the
  real tokenizer, so 128K tokens is ~1.8 MB, under tonic's 4 MB default decode cap. Fine today, but that
  margin is the only thing keeping a 128K prompt under the cap.
- **`keep` (the full (1, V) fp32 row per step for thinking-budget rows):** this could be replaced by computing,
  in the step, the lp of the *next forced id* too. The budget criterion knows it before the step is queued
  only one step late, so keeping the row is the simpler correct choice. Leave as is.

## Round 2 (re-review of the current diff, 2026-10-22)

Runs: `pytest tests/test_logprobs.py` 36 passed in 6.7 s (slowest 0.33 s). `cargo test -p qwen-gateway` 19 passed (debug).

**No HIGH or MED findings remain.** M1, M2 and M3 are fixed and tested. Details below.

### Fixed and verified

**M1. `prompt_logprobs` and `prompt_cache` validation**
- `prompt_logprobs`: `true`/`0`/`1` are on, `false`/`null` are off, anything else is a 400. This now matches vLLM's int semantics.
- `prompt_cache` mapping (`request.rs:116`): `true`/absent/`null` → `""`, `false`/`"off"` → `"off"`, `"read"` → `"read"`. Anything else (including `""` and `"on"`) gets the gateway's 400. `validate` runs before `to_pb`, which uses `unwrap_or_default` after the check.
- The worker's `prompt_cache must be one of "", "off", "read"` (`_render`) is therefore reachable only through direct gRPC. The gateway can never trigger it. The two sides agree on every field:
  - `logprobs`;
  - `top_logprobs` 0..20 (and "needs logprobs");
  - `max_tokens 0` only with `prompt_logprobs`;
  - images with `prompt_logprobs` (worker `Check`);
  - `prompt_cache`.

**M2. A `max_tokens: 0` row now ends inside `_prefill`** (`worker.py:3317`). The order is `on_prefilled(cache)` (the store), then `req.plp`, then `row.pos`, then `_finish(row, "length")`, then `_slab_done`, then `return`.
- `_end` does `reserved -= row.reserved` exactly once, because the row is never in `rows`.
- `row.parent` is cleared both by `snapshot` and by `_end`.
- `plp` is assigned before `out.put(("done", "length"))`, so the handler (which reads `r.plp` after "done") always sees it. The event order is Start → PromptLogprobs → Done.
- `return` raises StopIteration, so `_advance` releases `task`/`task_row`.
- `_end`'s `all(r.done for r in self.rows)` over an empty batch sets `clear`, the same as a lone row ending.
- **Cancel race:** cancel is only checked at the top of the slab loop. A cancel that lands after the last slab ends the request as `length`, with the PromptLogprobs event. That is harmless (LOW R2-3).
- `test_prefill_only_never_joins` checks: no `_join`, the other rows' streams unchanged, no group, and the entry stored.

**M3. Spans matched from the end, plus the `gen` flag** (`_assistant_spans`, `_prompt_logprobs`).
- `gen` mirrors `_render`'s `add_gen`. Since the gateway refuses `continue_final_message` together with `add_generation_prompt: true`, the trailing unterminated header is dropped exactly when the template renders it.
- With `continue_final_message`, the last assistant block is unterminated and runs to `n`, with no pop.
- `test_spans_echoed_markup` covers the mismatch case.

**`prompt_cache: "read"`** (`_prompt_cache`, where `keeps = use_cache and mode != "read"`):
- **Lookup:** `lookup` runs as normal. Hits/misses are counted, `used`/`seq` are refreshed, and the clock ticks once (no store tick).
- **Admission and eviction:** `_peek`, `_wanted` and `_fits` use the same `_lookup_ids`. Reads never bump `pc.version`, so `peeked` memos stay valid. A waiting read request's entry is "wanted".
- **Nothing is stored:**
  - `keep` is empty, so no checkpoint states are taken;
  - `row.fed`/`row.plan` are not set, so `_drop_done` skips the reply KV;
  - `snapshot()`'s no-op branch still sets `row.parent = None`, and its `finally` drops the `_Inherit` and clears `ckpts`;
  - cancel/failure go through `_store_prefix` → `snapshot`, so the same release applies.
- `test_prompt_cache_read` checks the cache is byte-for-byte unchanged and that the entry is collectable after eviction.

**L2. Journal:** `choices[0].logprobs` is trimmed to `{tokens, logprob}`, alongside `prompt_logprobs_positions`.

### Remaining (LOW only)

- **R2-1. "read" rows that miss are never cascade-grouped.**
  - **Cause:** `row.cached = reused > 0` and nothing stores, so `_cascadable` refuses them. A "read" row that hits is eligible, which is correct: it resumed at a slab end, so its prefix bits are the cold bits.
  - **Cost:** only for generations sent with "read". TF rows never decode.
  - **Fix:** none needed. Optionally set `row.cached = True` for text rows prefilled cold under the bounds rule, if read-mode generations become common.
- **R2-2. Matching from the end is the right default, but a mismatch is still only logged.** Markup echoed *after* the real turns (in the last tool response) shifts the latest spans instead. Optional: report a mismatch flag in `PromptLogprobs`/`prompt_logprobs_messages`.
- **R2-3. A cancel after the last slab of a `max_tokens 0` request returns `length` + PromptLogprobs, not `cancelled`.** The work is done and the entry is stored, so this is harmless. The spec's "no PromptLogprobs on cancel" holds only for cancels that land before the last slab ends. Document it, or check `req.cancel` before `_finish`.
- **R2-4. Carried over from round 1, unchanged:**
  - L1: `bytes` is still the UTF-8 of `decode([id])` (U+FFFD for partial characters);
  - L3: `_token_logprobs` still runs per row, not stacked;
  - L4: under fp16 near-ties, `top[0]` may differ from the greedy draw;
  - L5: stream + `prompt_logprobs` is accepted (a deliberate extension);
  - L7: template-dependent spans.

  The README covers L1, L5 and L7.

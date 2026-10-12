# Prompt cache under batching: second review (what `prompt_cache_opt.md` missed)

Date 2026-10-12. Read-only, CPU only: no MLX, no GPU, no worker. Line numbers are for
`src/worker/worker.py` at 13dd6dc and mlx-vlm 0.7.6 in `src/worker/.venv`. The checkpoint and thinning
claims were checked with a pure-Python re-implementation of `_bounds` / `_prompt_cache` keep set /
`_PrefixCache.store` thinning + covered rule (no budget), driven with synthetic token streams that
follow the 9B chat template (`weights/ornith15-9b-mlx-8bit/chat_template.jinja`).
Model numbers: 9B, fp16 KV 32 KiB/token; one GDN state 51.5 MB (24 × fp32 32×128×128 + conv;
`gated_delta.py` 390 allocates the state in fp32).

## Summary

The first review is right about the shared-burst benchmark (16 × 8K). For that case only its items
1, 2 and 6 matter, and nothing below changes them. It underweights the **dominant pattern: one
conversation growing turn by turn, and several conversations at once**. That pattern has four
problems the first review did not see.

* **The covered rule never fires for consecutive turns (a bug).**
  * The README promises "an entry that is a prefix of a newer one is dropped". The code (775–778)
    also requires `len(e.ids) in ckpts`. That position is the previous prompt's last-but-one token,
    `<think>` (or `</think>` with thinking off). It is never a slab end, so it is never a
    checkpoint.
  * As a result, every turn leaves one more entry, each with its own full KV buffer. Simulated
    over 30 turns: 30 entries, `covered=0` every time.
  * Under the 3 GiB budget this means one active conversation (~0.7–1 GB charged per entry at
    10–12K) evicts every other conversation after 3–4 turns.
  * The fix is a few lines and byte-identical.
* **GDN checkpoints, not KV, dominate the bytes once KV is shared.**
  * A 10–24K conversation entry carries up to 12 checkpoints (0.62 GB) next to 0.33–0.8 GB of KV.
  * Most of them are old-turn boundaries that will never be hit.
  * Storing the state in fp16 would make a hit differ from a cold run (the recurrence is fp32), so
    the only exact lever is *which* states to keep.
* **Per-turn TTFT is the re-prefill of the previous answer, and an exact fix exists.**
  * Turn N+1 re-prefills answer N plus the new user message: 300–2000 tokens, ~0.6–4 s at
    8–16K offsets. The stored entry ends at `prompt[:-1]`.
  * A decode-time "generation checkpoint" can never match a cold run bit for bit: decode kernels,
    and a resume point that is not a slab end.
  * Instead, an **idle-time warm prefill** can do it exactly:
    * render the assistant turn the way the next request will carry it;
    * prefill it with the cold slab scheme while the user reads or the tool runs;
    * store an entry that ends at the next `<|im_start|>`.
* **Strict FIFO with one prefill at a time has no fast lane.** A 0.3 s cache hit queued behind a
  cold 32K prefill waits ~74 s, or ~6.5 min behind a 120K one. A bounded fast lane for one-slab
  prefills between the slabs of a long one is byte-identical.

Two of the first review's claims are wrong in mechanism:

* **Item 4, thinning.** The system checkpoint is *not* the first one dropped for long prompts in
  general. Slab strides restart at every message cut, and the boundary ties with the stride
  checkpoint below it, which goes first. The boundary is dropped first only when the system prompt
  is shorter than ~2.3K tokens.
* **§6.4.** The history does keep `<think>`. The real reason the end state is useless for the next
  turn is that it is not a slab end.

Also found, a memory hazard at long context: the newest entry may exceed the budget, and its pinned
buffer plus the row's COW copy doubles the prompt KV (+4.3 GB at 128K on the 9B).

## Additional proposals

Columns: (a) the 16 × 8K shared-prefix burst; (b) multi-turn chat over a 7.5K system prompt
(several users); (c) memory. All TTFT numbers assume ~1.8–2.2 ms/token of prefill at 8–16K offsets
(8K cold = 14.4 s), and ~2–3 ms/token for slabs under 512 rows.

| # | proposal | (a) burst | (b) multi-turn chat | (c) memory | cost / risk | exactness |
|---|---|---|---|---|---|---|
| N1 | **Covered rule = "any entry whose ids are a prefix of the new one"** (absorb its checkpoints, drop it) | 0 | Conversations stop evicting each other. Today the 4th–5th turn of one chat evicts the others' entries, so the next turn of those re-prefills the whole history (10K → ~20 s instead of ~1 s) | 1 entry per conversation instead of 1 per turn: −0.35–1 GB per extra turn kept today | ~8 lines; low | byte-identical |
| N2 | **Idle-time warm prefill of the next turn's prefix** (render the finished assistant turn as history, prefill it from the entry with the cold cuts, store up to the next `<|im_start|>`) | 0 | TTFT per turn: answer re-prefill (0.6–4 s; 2–8 s if the client returns reasoning) → only the new user message (0.1–0.5 s) | + answer KV (~10–60 MB) + 1 GDN state per warmed conversation | ~120 lines; medium (idle-job plumbing, uses GPU when idle); knob, default off for benchmarks | byte-identical |
| N3 | **Fast lane**: a waiting request whose uncached part fits one slab runs between the slabs of a running long prefill (and may pass a long head at admission), bounded per head | 0 | Hit behind a cold 32K prefill: ≤ 74 s → ≤ 1 slab wait (~3–5 s) + 0.2–1 s. Behind 120K: ≤ 391 s → ≤ ~9 s | = | ~80 lines; medium (save/restore `lm._rope_deltas`/`_position_ids`, aging bound); relaxes "FIFO, no overtaking" | byte-identical |
| N4 | **Checkpoint retention by value** (corrects item 4). Protect the system boundary (`starts[1]`) and the last 2–3 bounds, drop old-turn boundaries before strides, `CKPTS` 12 → 6–8 | 0 (4 ckpts) | System prompts < 2.3K keep their hit after thinning (today: full re-prefill, ≤ ~4 s) | −0.2–0.3 GB per long conversation entry (GDN is 45–65 % of an entry) | ~25 lines; low | byte-identical |
| N5 | **Long-prompt store hazard**: count oversize entries in the budget (with item 3). For prompts whose KV > ~1 GB, *defer* the store to the row's end (the entry adopts the row's buffer: no COW) | 0 | 32K–128K document follow-ups keep their hit without doubling KV | −1× prompt KV while the row decodes: −1 GB at 32K, −4.3 GB at 128K (9B); 27B q8 at 128K is otherwise ~+4.3 GB over a 32 GB machine | ~40 lines; low | byte-identical |
| N6 | **Observability**: `Usage.cached_tokens` → `usage.prompt_tokens_details.cached_tokens` (vLLM field), log/metric `split − reused` ("lost" tokens), Health/metrics for entries, unique bytes, hits/misses/evictions | 0 | 0 (decides N2/1/2 from real logs) | = | ~40 lines + proto field; low | n/a |
| N7 | **Persist pinned prefix entries across restarts** (+ optional warm list at load); key = model files + mlx/mlx-vlm versions + every bit-affecting knob | first burst after a restart: −14 s (8K) / −74 s (32K) | the same for the first conversation after a restart | disk: ~0.5 GB per 8K entry | ~100 lines; medium (stale-key risk) | byte-identical if keyed right |
| N8 | Vision: resume the text prefix before the first image (VL models only) | 0 | 9B Ornith-1.5: 0 (text-only). 27B/35B image agents: −13–17 s per 8K system prompt | = | ~50 lines + vision prompts must adopt `_bounds` marks (a one-time output change for multi-slab image prompts) | exact after that change; needs a bitwise check of text positions |
| N9 | Partial entries from an in-flight cold prefill, only together with N3 | 0 (FIFO already serializes) | A new conversation hits the system checkpoint of a still-running 30K cold prefill | = | ~30 lines on top of N3 | byte-identical |
| N10 | Decode-time "generation checkpoint" for the next turn | 0 | as N2 even when the GPU is never idle | as N2 | ~60 lines | **not** exact (decode kernels; resume not on a slab end) → reject unless N2 cannot run (sustained load) |

---

## 1. Multi-turn chat: what actually happens (template-level)

### 1.1 How history is rendered

The chat template renders **every** assistant turn as:

```
<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n
```

`ns.last_query_index` is computed but never used to strip anything. `reasoning` comes from the
message's `reasoning`/`reasoning_content` (gateway `request.rs` 562 → worker 349–350), or is split
out of `content` if it contains `</think>`. It is `''` when the client drops it.

The generation prompt ends `<|im_start|>assistant\n<think>\n` (thinking) or
`…<think>\n\n</think>\n\n` (thinking off). Three cases for the next prompt P_{N+1} versus the
previous prompt P_N plus the generated tokens:

| client behaviour | P_{N+1} vs P_N + generated | where it diverges |
|---|---|---|
| thinking on, reasoning dropped (most OpenAI-SDK UIs) | `<think>`,`\n\n`,`</think>`,`\n\n`,content… vs `<think>`,`\n`,reasoning… | the token after `<think>`, i.e. exactly at the entry end `len(e.ids)` |
| thinking on, reasoning returned | identical through the answer, if the trimmed reasoning/content re-tokenize to the generated ids | at the first non-canonical token, if any |
| thinking off | identical through the answer (same round-trip caveat) | — |

* **The first review's §6.4 mechanism is wrong.** The history does contain `<think>`.
* **Its conclusion holds**: the entry's end state never serves the next turn. In all three cases
  the common prefix reaches `len(e.ids)`, but `len(e.ids)` is not a cut of the new prompt. A cut
  needs `<|im_start|>` or a full step (`_bounds` 831–846), so `best` (731–740) rejects it.
* **The resume lands at the deepest checkpoint ≤ the end of P_N.** That is usually the generation
  prompt's `<|im_start|>` (when the user message is ≥ 256 tokens) or the user message's
  `<|im_start|>`. Simulated over 30 turns, the gap `split − reused` was **4–157 tokens**. The cache
  is near-optimal *given what it stores*.
* **What each turn re-prefills is answer N** (+ user N+1). That is the target of N2.

### 1.2 The covered rule never fires (N1, bug)

`store` (775–778) drops an older entry `e` only if:

```python
len(e.ids) <= len(ids) and ids[:len(e.ids)] == e.ids and (len(e.ids) in ckpts or len(e.ids) == len(ids))
```

* In every case of the table above, `e.ids = P_N[:-1]` *is* a prefix of the new entry's ids.
  `len(e.ids)` is not in the new entry's checkpoints: it is not a slab end, and the inherited
  `ckpts[at] = parent end state` (767) applies only when the resume was at the parent's end.
* So the rule fires only for an exact repeat.
* The simulator (7.5K system prompt, 30 turns, user messages of 30–150 tokens, answers of
  200–900 tokens) ends with **30 entries**, `covered=0` on every store.
* With the real budget:
  * Each entry is a **separate** full KV buffer: the hit row's private copy, pinned by the entry
    (first review §1.1).
  * Each entry is also charged its inherited checkpoints again.
  * At 10–12K that is ~0.33–0.4 GB of KV + 51.5 MB end state + 6–12 × 51.5 MB charged ≈
    0.7–1 GB per entry, so **3–4 entries fit in 3 GiB**.
  * One active conversation therefore evicts every other conversation within 3–4 turns. Their next
    turn falls back to whatever entry still shares the system prompt, and re-prefills its whole
    history: ~2.5K tokens ≈ 5 s at turn 5, ~10K ≈ 20 s at turn 20.
* `lookup` cost also grows with the entry count: `_lcp` per entry, 0.6 ms at 8K, 3 ms at 32K.

**Fix.** Drop any `e` whose ids are a prefix of the new ids. Before thinning, absorb its
checkpoints with `ckpts.setdefault(p, s)`: shared array objects, no new memory.

* What is lost is `e`'s end state. It serves only an exact repeat of P_N after turn N+1 was already
  sent, which no UI does. Regenerate of the *current* turn uses the current entry.
* Byte-identical: the new entry's KV over `[0, len(e.ids))` has the same bits, by the
  hit == cold invariant.
* Add a unit test: two consecutive turns leave one entry.

### 1.3 Idle-time warm prefill of the next turn (N2)

The next prompt's prefix through `answer N <|im_end|>\n` is predictable when generation ends. The
worker can render the request's own `messages` + `{role: assistant, content, tool_calls,
reasoning?}` with the same template (`_render` 2724), with `add_generation_prompt=False`, and
tokenize it. It does not need to guess the tokenization: the next request goes through the same
renderer.

Whether to include the reasoning:

* If any earlier assistant message in *this* request carried reasoning, the client returns
  reasoning; include it.
* Otherwise render it empty, which is the dropped-reasoning case.
* On the first turn, assume dropped.

Mechanics:

* The handler thread renders and tokenizes the prefix (no MLX), then posts a warm job
  `ids = prefix + [<|im_start|>]` to the scheduler. The `<|im_start|>` is the next message's first
  token, whatever its role (user / tool response).
* The scheduler runs the job only when idle (`not rows and not waiting and task is None`). It reuses
  `_prompt_cache` + `_prefill` with a row that never joins:
  * it resumes from entry N's last checkpoint;
  * the slabs are cut by `_bounds` of the warm ids, which are exactly the next prompt's cuts up to
    the prefix end;
  * `on_prefilled` stores the entry, ending at `len(prefix)`.
* It abandons the job between slabs if a request arrives. A single slab cannot be interrupted:
  ≤ ~1–4 s at 8–16K.
* **The resume.** If the answer is ≥ 256 tokens past the last cut, `len(prefix)` is a cut of the
  next real prompt (`<|im_start|>` + `PREFILL_MIN_STEP`). Then `best` takes the warm entry's
  **end** and the next turn prefills only the new message. With N1, entry N is absorbed.
* **Exactness.** Same code path, same slab shapes, B=1 own caches, so the next turn is
  **byte-identical to a cold run**. If the prediction is wrong (edited history, a client that
  rewrites content), the entry is simply not hit and LRU evicts it.
* **Gain for (b).** Answers of 300–1000 visible tokens: −0.6–2 s per turn. With reasoning returned
  (2–5× more tokens on Ornith-1.5): −2–8 s. Agent tool loops: −0.2–0.6 s per round (short tool-call
  turns), hidden behind the tool's own run time.
* **Costs.**
  * GPU work while idle, which perturbs benchmarks: default off; set `QWEN_PROMPT_CACHE_WARM=1`
    for interactive use.
  * Wasted prefill when a conversation ends.
  * The idle condition never holds under sustained load. That is the only case for N10.

### 1.4 Why not a decode-time generation checkpoint (N10)

* Decode-written KV/GDN comes from qmv / mrqmm / decode SDPA, and a cold prefill of the same text
  uses pfqmm/qmm slabs.
* The resume point (`<|im_end|>` of the answer) is not a slab end, so everything after it would be
  cut differently too.
* A hit would then differ from a cold run, breaking the cache's stated invariant ("the output does
  not depend on what the prompt cache holds", `_bounds` docstring; README).
* It also depends on the detokenize → retokenize round trip. A non-canonical BPE sequence among the
  generated tokens ends the prefix there.
* The B ≥ 3 precedent makes it *acceptable* as an opt-in knob, but N2 gets the same TTFT exactly.
* Extra hazard if built: at B ≥ 2 the row's GDN state is a slice view of the stacked state. `_gdn_split`
  (1636–1639) / `a[i:i+1]` pins the whole B × 51.5 MB stack, so it must be copied out.

---

## 2. GDN checkpoints are the memory hog (N4, and a correction to item 4)

### 2.1 Bytes

Here "charged" counts each entry's inherited checkpoints again (first review §1.1); "unique" counts
each shared array once.

| entry | KV | GDN checkpoints | end state | GDN share |
|---|---|---|---|---|
| 8K single prompt (4–5 ckpts) | 268 MB | 206–258 MB | 51.5 MB | 50–54 % |
| 10–12K conversation (10–12 ckpts after a few turns) | 330–400 MB | 515–618 MB | 51.5 MB | 60–65 % |
| 24K+ (12 ckpts) | 805 MB | 618 MB | 51.5 MB | 45 % |
| a sibling in the segment store (item 2) | 3.3 MB unique | 0 unique | 51.5 MB | 94 % |

### 2.2 Shrinking the GDN states

* **fp16/bf16 storage.** The state is fp32 by construction (`gated_delta.py` 390) and the kernel
  reads it as fp32. Rounding it changes every later logit, so a hit would no longer equal a cold
  run. Rejected for the same reason as N10.
* **Lossless compression** of fp32 recurrent states is not worth it.
* **Recompute from the nearest checkpoint on demand** is just a sparser checkpoint set: the state
  depends on all layers' inputs, so recomputing = re-prefilling the span.
* **What the others do.**
  * SGLang's hybrid/Mamba radix cache keeps recurrent states only at selected tree nodes and evicts
    them separately from KV (a node can keep KV without a state).
  * vLLM's hybrid prefix caching stores Mamba states only at block-aligned points of the computed
    prefix.
  * llama.cpp's server keeps a small per-slot number of context checkpoints (`--ctx-checkpoints`)
    for SWA/recurrent models.
  * All three ration states by *position value*, which is what N4 does.

### 2.3 Thinning as implemented

Thinning (770–773) drops `ps[i]` with the smallest `ps[i+1] − ps[i−1]`, the first one counted from
0, never the last.

The first review assumed absolute 2048 strides around a message boundary m. In fact `_bounds` restarts
the stride at every cut (`i − last ≥ _step(last)`), while the `want` positions are absolute multiples
mapped down to the nearest bound. Around m (cut at m, next stride cut m + 2048, previous k·2048), the
span of m equals the span of k·2048. `min` on `(span, pos)` then drops **k·2048** first. Simulated:

| system prompt | + one 32K–60K user message | system checkpoint after thinning | new conversation, same system |
|---|---|---|---|
| 1.5K, 1.9K | 33–34K | **dropped** (it is the first checkpoint, span counted from 0) | full re-prefill, 1.5–1.9K tokens |
| 2.3K | 34K | there is no boundary cut (2303 − 2048 < 256), and the 2048 stride is dropped | full re-prefill, 2.3K |
| 4K, 7K, 7.5K, 12K | 36–68K | **kept** | 58 tokens |
| 7.5K + 30 chat turns (24K) | — | kept; old-turn boundaries thinned | — |

So item 4 is real only for system prompts under ~2.3K tokens (loss ≤ ~4 s per new conversation),
not "for prompts > 24K". The fix the first review proposed (protect boundary checkpoints) is still
right.

### 2.4 N4: value-ordered retention

Rank by expected use:

1. `starts[1]` (a new conversation), and the last 2–3 bounds (next turn, regenerate, edit the last
   message). Protected.
2. Stride checkpoints inside the first message (the template puts the tools block *first* in the
   system message, so clients whose system text varies after a fixed tool list hit these).
3. Old-turn boundaries (edit / branch an older message). Thinned first, by span.

Lower `PROMPT_CACHE_CKPTS` to 6–8: −0.2–0.3 GB per long entry. With N1 and item 2's
unique-object accounting, a conversation then costs ~KV + 6–8 × 51.5 MB.

Drop the end state at store when the last checkpoint is ≤ 256 tokens below it: it saves ≤ 260
tokens, and only on an exact repeat. This is stronger than the first review's "drop it first under
pressure".

---

## 3. Scheduling: a fast lane for short prefills (N3, + N9)

`tick` (1886) admits only when `self.task is None` (1894), and admission is FIFO. While a cold
long prompt prefills, a hit or a short prompt waits for all of it: 14–17 s at 8K, 74 s at 32K,
391 s at 120K.

**Proposal.**

* When a running prefill task yields (between slabs), the scheduler may run **one** waiting
  request's complete prefill in that tick if its uncached remainder is ≤ one slab. A single-slab
  `_prefill` never yields (2125), so it completes in one `next()`.
* Admission with no task running may pick such a request over a long head.
* Bounds:
  * at most one fast-lane prefill per slab of the long one, so the long prefill keeps ≥ ~50 % of
    the prefill time;
  * at most K = 4 overtakes per head (aging);
  * `_fits` as usual.
* The remainder needs the lookup at queue time. Memoize it per request and per cache version (a
  counter bumped by store/evict), otherwise 32 waiting × 6 entries × 0.6 ms runs every tick.

**Hazard: shared model state.**

* A text prefill's slabs 2+ read `lm._rope_deltas` (the library's offset + delta branch).
* An interleaved vision prefill leaves a non-zero delta there, and a hit sets zeros (2017–2018).
* So save and restore `lm._rope_deltas` / `lm._position_ids` around a fast-lane prefill, or admit
  only text requests to the lane.

**Exactness.** Byte-identical: every prefill keeps its own cuts and caches, and kernels do not care
what ran between slabs.

**Gains.**

* (b), several users: the TTFT tail of short turns behind somebody's document drops from tens of
  seconds to ≤ one slab (~3–5 s at 8–32K) + its own prefill.
* (a) burst: 0, because every sibling needs request 1's full prefix first.

**N9.** With N3 in place, partial entries (the first review's item 7) gain value: a new
conversation can resume from the system-boundary checkpoint of a 30K cold prefill that is still
running.

* A partial entry is views of the producer's KV at that checkpoint. The producer's later growth
  concatenates into new buffers, so it only keeps the old buffer alive.
* The producer's final entry absorbs it (N1).
* Without N3 the gain is 0, as the first review says.

**Rejected scheduling ideas.**

* *Prefix-affinity admission for item 1.* At B ≤ 16 with one shared prefix there is one group
  anyway. With two prefixes, two groups cost one extra prefix read each. Small.
* *Interleaving first-token decode steps between the burst's 15 hits.* TTFT p50 −1–1.5 s at the
  price of the last request's; not worth a policy.

---

## 4. Memory accounting hazards (N5)

1. **The newest entry may exceed the budget.**
   * Eviction keeps the newest unconditionally (781). A 128K prompt's entry is 4.3 GB KV +
     12 × 51.5 MB + end ≈ 4.9 GB.
   * `_batch_budget` (1305–1319) subtracts a flat `PROMPT_CACHE_GB` (3 GiB), so ~1.9 GB lives
     outside every budget.
   * `_fits` (1945) sees it through `mx.get_active_memory()`, but `_regrow` (1924) compares
     `reserved` with `budget + prefill_reserve` only, so running rows can grow into it.
2. **Prompt KV is doubled while the row decodes.**
   * The entry holds views of the row's buffer. The row's first decode write is then a
     copy-on-write of the whole buffer (first review §1.1). So for the row's whole life, memory
     holds 2× its prompt KV.
   * That is +1 GB at 32K and +4.3 GB at 128K on the 9B. The 9B runs without the wired-limit
     sysctl, where the cap is ~21 GiB: weights 9.5 + 4.3 + 4.3 + 0.6 GB of checkpoints is close.
   * On the 27B at 128K (q8 KV 4.25 GB, weights 18 GB) it would not fit in 32 GB.
   * Neither `_bytes` (1907) nor the static cache term covers this. *Not measured; verify with one
     120K request and `QWEN_PROMPT_CACHE=1` (peak in Usage).*

**Fix.**

* Item 3's elastic accounting, i.e. `cache.bytes` in `_fits` **and** `_regrow`.
* For prompts whose KV exceeds ~1 GB (or a fraction of the budget), **defer the store to the row's
  end**:
  * The entry then adopts the row's buffer: views `[:len(ids)-1]`, no COW. Copy it out only if the
    generated tail is > 10 % of the buffer.
  * Keep the prompt-end GDN state: hold the list `_recurrent_states(cache)` at `on_prefilled`, and
    keep the checkpoints taken during prefill.
  * Byte-identical: decode never writes below `len(ids)-1`.
  * What is lost: concurrent siblings of a 32K+ prompt cannot hit it while it decodes. That is rare
    for documents.

Other notes:

* **Store failures fail the request.** `snapshot` runs inside the prefill generator, and
  `_Entry.__init__` `mx.eval`s. An exception there (e.g. an allocation failure) surfaces in
  `_advance` (1990–1998) as "prefill failed" for a request whose prefill succeeded. Wrap the store
  in try/except, log, and skip.
* **Idle paging.** While idle, `_idle` leaves the wired-limit context. The cache's buffers stay
  allocated but pageable, so under memory pressure from other apps a hit after a long idle pays a
  page-in. Negligible unless the machine swaps.

---

## 5. Other items

* **N6, observability.** This should come first, because it decides N2, item 1 and item 2 from
  real traffic.
  * Add `cached_tokens` (`req.reused`, already computed, 2027) to `Usage`.
  * The gateway returns it as `usage.prompt_tokens_details.cached_tokens`, the vLLM/OpenAI field.
  * Add a counter for `split − reused` (tokens the cache had but could not resume) and Health/metrics
    for entries, unique bytes, checkpoints, hits/misses/evictions.
  * The log already prints `common prefix` (2019). A histogram of `split − reused` and of
    "re-prefilled answer tokens" over a day of real use says whether N2 pays.
* **N7, persistence.**
  * Use `mx.save_safetensors` of KV slices + checkpoint states + ids at shutdown (SIGTERM path, within
    the 10 s exit budget) or when idle; load lazily on the first lookup, or at load for a configured
    warm list.
  * An 8K entry is ~0.5 GB: ~0.1–0.3 s from NVMe vs 14.4 s of prefill.
  * The key must cover the weights' identity (index + mtimes), mlx/mlx-vlm versions, GPU family,
    `QWEN_FP16`, `KV_BITS`, `quantized_kv_start`, `_FIXED_SLAB`, `PREFILL_STEP/MIN_STEP/BUDGET`,
    `SDPA_FUSED`, `GDN_FAST`, `PFQMM`. A stale key silently breaks hit == cold, so refuse on any
    mismatch.
  * Worth it on this machine, which restarts the worker for every benchmark or knob change. For
    benchmarks keep the cache off anyway.
* **N8, vision.**
  * Vision prompts are cut without marks (`bounds = []`, 2007), so a text checkpoint at the system
    boundary is not a slab end of a cold vision run.
  * Feasible: use `_bounds` marks up to the first image token; compute `get_input_embeddings` /
    `position_ids` on the full prompt and slice them from `reused` (the slab loop already slices
    `vision_pos` per slab, 2109–2110).
  * The vision tower still runs: the gain is the text prefix only.
  * Needs a bitwise check that text positions via explicit `position_ids` equal the
    offset + zero-delta path.
  * 0 for the 9B Ornith-1.5 (text-only), so low priority.
* **DFlash.** The first review's §7.3 is right: speed only, measure once. A row demoted from
  `_spec` (2258–2266) becomes a normal batch row, so with item 1 it can join a group. Item 1's
  "DFlash excluded" applies only while it is speculative.
* **Item 1 refinement.** Rows can be grouped by *token prefix* (`hash(ids[:P]), P`) instead of
  `id(segment)`. By the hit == cold invariant, every row whose prompt shares `[0, P)` holds the same
  KV bits there: rows from different entries, and the producer. Fewer special cases (no
  `entry.producer`).
* **`PREFILL_MIN_STEP` for marks.** A message boundary < 256 tokens after the previous cut is not
  a cut, which costs short-message chats up to ~260 tokens per hit (0.3–0.5 s). Lowering it to
  64–128 adds resume points but changes cold outputs once (slab shapes) and adds ~80 ms per extra
  short slab. Not worth it now.
* **Exactness vs arbitrary resume points.** The task asked whether "hit == cold" costs 0–2047
  tokens per hit. It does not, for checkpoints:
  * `_bounds` is a pure function of the prefix, so every checkpoint inside the common prefix is
    automatically a slab end of the new prompt.
  * The `p in ends` filter (738) only ever rejects the entry end.
  * The 0–2047 loss comes from GDN states existing only at checkpoints, and would exist without the
    exactness rule too.
  * Relaxing exactness would buy only the entry-end case (≤ ~260 tokens), so it is not worth it.

### Correctness checks that came out clean

* `_copy_cache` sharing: `ArraysCache.commit/abort_speculation` (cache.py 1153–1190), `_gdn_stack`,
  `_gdn_take`, `_gdn_concat` and `_gdn_split` all *replace* state arrays, never write them in place,
  so restored checkpoint objects are never mutated.
* A cancelled or failed prefill never stores (2085–2087, `_advance`).
* The snapshot guard handles the 1-token / exact-repeat case.
* `ckpts.pop(len(ids))` is right.
* `used` ties make the oldest sibling win, which is harmless.

---

## Corrected priority (old + new)

Cheap exact fixes come first, then by workload.

1. **N1**, covered-rule bug. Byte-identical, a few lines. Without it, multi-user chat thrashes.
2. **N6**, metrics (`cached_tokens`, `split − reused`, unique bytes). Measure before building the
   big items.
3. **Item 4 as corrected + N4**, value-ordered checkpoint retention and fewer checkpoints.
4. **Item 3 + N5**, elastic accounting including `_regrow`, deferred store for long prompts, and
   the store try/except. Memory safety at 32K–128K.
5. **Item 2**, segment store. Exact; holds the burst's siblings; prerequisite of item 1.
6. **Item 1**, cascade decode attention. The only decode lever for (a): +25–30 % at 8K shared, but
   inexact at B ≥ 3 and the most code. First if the shared burst is the target workload.
7. **N3** (+ **N9**), fast lane. For multi-user interactive use, move it above item 1.
8. **N2**, idle warm prefill. For single-user chat or agent use, move it above item 1.
9. **Item 5**, quantized-KV floor (27B/35B).
10. **N7**, persistence / warm list.
11. **N8**, vision prefix (VL models only).
12. Rejected:
    * **N10** (inexact; only if N2 cannot run under sustained load);
    * items 6–9 as the first review says;
    * fp16 GDN checkpoints;
    * lowering `PREFILL_MIN_STEP`.

## Open questions

1. **`mx.contiguous` in `ArraysCache.update_window`** (cache.py 963) slices the last 3 positions
   of the slab's `conv_input`, which is (1, 2051, 8192) fp16 = 33.6 MB per layer.
   * If MLX's `Contiguous` shares the buffer of an already row-contiguous view (an axis-1 slice
     with batch 1 is one), every checkpoint pins 24 × 33.6 MB ≈ **0.8 GB** that `nbytes` does not
     see.
   * I expect MLX to copy here: its sharing rule appears to be guarded by
     `buffer_size ≤ nbytes + 16 KiB`, the same as `is_donatable` in
     `backend/common/utils.h` 218. But the shipped headers do not show `Contiguous::eval`.
   * A 3-line CPU check settles it: allocate, slice, `mx.contiguous`, delete the source, read
     `mx.get_active_memory()`. Alternatively, the first review's open question 3 (cache bytes vs
     active memory) would expose it.
2. What fraction of real clients return `reasoning`? Does the detokenize → template → tokenize
   round trip reproduce the generated ids? This decides N2's default variant, and whether N10 could
   ever work.
3. The real `split − reused` and "answer tokens re-prefilled per turn" distributions (N6).
4. Peak memory of one 120K request with `QWEN_PROMPT_CACHE=1` vs 0 (N5's predicted +4.3 GB).
5. N3 fairness constants: 1 fast-lane prefill per long slab, K = 4 overtakes. Check the TTFT p95
   of the long request under a stream of hits.

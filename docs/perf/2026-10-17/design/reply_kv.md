# Reply KV in the prompt cache (`QWEN_PROMPT_CACHE_REPLY`)

Date 2026-10-17. CPU only: no GPU, no worker started (a benchmark held the GPU). worker.py at HEAD 589b3f8 +
this change; mlx-vlm 0.7.6 / mlx 0.32.3 in `src/worker/.venv`. Load: `docs/perf/2026-10-16/HANDOFF-appworld.md`
(AppWorld ReAct agent, 8 episodes; 94 % prompt-cache hits; prefill is 75–80 % of the scheduler's time).

## The idea

A prompt-cache entry is stored during prefill, where the row's cache holds `prompt[:-1]`. The KV the row
computes for its generated tokens is dropped when the request ends. In an agentic loop the next call's prompt
is the previous prompt + the assistant reply + a new user message, so the reply is prefilled again on the next
call. The reply is ~60 of ~860 new tokens per call. The assistant header and, after a short tool output, the
stretch back to the previous slab end are prefilled again too.

With `QWEN_PROMPT_CACHE_REPLY=1` the worker extends the request's entry when the request ends. The extension
holds:

* the KV of every token the model consumed: the prompt, then the generated tokens it fed back;
* the recurrent (GDN) state after them, as the new end state.

The next turn then resumes right after the reply.

## Phase 0: does the template reproduce the generated ids?

Probe: `docs/perf/2026-10-17/design/reply_kv_phase0.py`. It uses CPU only, reads only the tokenizer files and
loads no weights.

* Command: `cd src/worker && uv run python ../../docs/perf/2026-10-17/design/reply_kv_phase0.py` (~25 s).
* Rendering is the worker's: mlx-vlm `apply_chat_template` with the model's template and the gateway's
  `enable_thinking=false` (AppWorld ran with `GATEWAY_THINKING=off`), then `tokenizer.encode`.
* The text-only Ornith-1.5 checkpoint has no `preprocessor_config.json`. The probe therefore uses
  `AutoTokenizer` (same `chat_template.jinja`) instead of the processor.
* Check: for every pair of consecutive calls (k, k+1) of a task, does the rendered prompt of k+1 start with
  `ids(prompt k) + tokenize(content k) + [<|im_end|>]`?

Traces: `~/Projects/appworld/experiments/outputs/ornith9b_react_dev/tasks/*/logs/lm_calls.jsonl`. At probe
time there were 51 tasks, 1395 calls and 1344 pairs; the replay bench's 19 tasks / 507 calls are a subset.

| check | pairs | % |
|---|---|---|
| prompt k+1 starts with prompt k (ids) | 1344 | 100 |
| ... + tokenize(content k) | 1344 | 100 |
| ... + tokenize(content k) + `<\|im_end\|>` | 1344 | 100 |
| `completion_tokens` == len(tokenize(content)) | 1344 | 100 |
| reply end is a slab end of prompt k+1 | 0 | 0 |

**There is no divergence: the match rate is 100 %.**

* Leading whitespace: AppWorld sends the reply back with `"\n\n"` appended (the message object from the API,
  `cache_control` moved to the last user message), but the Qwen3.5 template `|trim`s the content.
* Assistant header: it renders `<|im_start|>assistant\n<think>\n` + reasoning + `\n</think>\n\n` + content +
  `<|im_end|>\n`. With thinking off the reasoning is empty, so the history's header equals the generation
  prompt's `<think>\n\n</think>\n\n`.
* Token boundaries: `</think>` and `<|im_end|>` are special tokens, so no BPE merge crosses the reply's
  edges.
* Generated ids: `completion_tokens` equals the canonical re-tokenization's length on every call. The
  generated ids are therefore (almost certainly) the canonical tokens; the traces do not hold the ids
  themselves.

**What is reused.** Today the next call resumes at the deepest checkpoint at or below the old entry's end.
That is the last slab end before the reply: the assistant `<|im_start|>`, or an earlier one when the last
tool output was under `PREFILL_MIN_STEP` = 256 tokens, since then the assistant start is not a cut. With the
reply KV it resumes after the reply's `<|im_end|>`.

| | value |
|---|---|
| new tokens per call today | 961 077 tokens over 1344 calls, 715 per call |
| saved with the reply KV | 199 458 tokens, **20.8 %**, 148 per call |
| same, on the 19 replay tasks only | 20.6 % |
| saved per call, p10 / p50 / p90 | 47 / 120 / 295 |
| reply length, p10 / p50 / p90 | 35 / 48 / 110 |
| header + stretch back to the last slab end, p50 / p90 | 7 / 243 |

**Caveats that phase 0 cannot see:**

* **The EOS.**
  * A batch row decodes one step ahead (`_iterate`), so the EOS the model stops on has already gone through
    the model when the row ends. The cache holds `prompt + generated` including `<|im_end|>`, and the GDN
    state is the one after it. The template puts exactly that `<|im_end|>` next, so the entry ends one
    token later and matches (row 3 of the table).
  * A sync row (grammar, multi-token bad words) never feeds the EOS. Its entry ends before it, which still
    matches (row 2).
  * A model that stops on `<|endoftext|>` (also an EOS id) would diverge on that last token. The GDN state
    cannot be cut back, so that entry's end is not resumable. The next turn then falls back to the old
    message-start checkpoint, as today. On Qwen chat models the stop token is `<|im_end|>`.
* **Thinking on.** The template drops the reasoning unless the client sends it back (`reasoning_content` /
  vLLM `reasoning`). If it is dropped, the next prompt diverges at the reply start. That is no gain, and a
  loss:
  * the reasoning's KV would be copied (32 KiB per token on the 9B);
  * the prompt-end state (a regenerate's full hit) would be dropped.

  So such replies are not stored (Fixes, finding 6). The AppWorld load runs with thinking off.
* **Trailing whitespace.** A reply that ends in whitespace before the EOS is trimmed by the template.
  `completion_tokens` == re-tokenized length on all 1344 calls says it did not happen here. Where it does,
  the hit stops inside the reply and falls back.
* **Stop strings.** The tokens of the stop string (and any fed after it) are in the cache but not in the
  returned text. Such a row usually ends through `cancel` (the handler's stop-string hit) and stores nothing
  extra.
  * The handler's cancel can arrive after the scheduler already ended the row with `stop`/`length` (handler
    lag, or a stop string within the last tokens).
  * Then the reply is stored with the tokens past the string, and the next turn falls back at the
    divergence.
  * The stale entry is absorbed by the prompt-part rule. This is harmless.

The worker has nothing to fix on its side: the ids it fed are what the next prompt holds.

## Design

### When a request ends (`_Scheduler._drop_done` → `_store_reply`, scheduler thread)

A row qualifies when all of the following hold:

* the knob is on, and so is `QWEN_PROMPT_CACHE`;
* it is a text row, not a DFlash row (`row.spec`, also after demotion);
* it was not in a cascade group when it ended (its suffix KV lives in the group's stack: `_GroupRef` caches);
* it finished with `stop` or `length` (`row.why`; cancel, usually including stop strings, and errors store
  nothing);
* its reply holds no reasoning: no `<think>`/`</think>` id in the fed tokens, and the prompt does not end
  inside an open `<think>`;
* its prefill stored an entry that is still in the cache (`row.entry`, a weak reference). If it was evicted
  or absorbed meanwhile, nothing is stored. The eviction order had decided this conversation's next turn is
  the furthest away, and copying the whole `[0, len)` out of the row would undo that decision.

**Fed ids.** The ids are `prompt + fed[:pos - len(prompt)]`:

* `fed` is every generated token as `_iterate` read it, in order. Forced thinking-budget tokens are
  included: they replace `y` and are read like any other.
* `row.pos` is the KV length.

So the ids are exactly what went through the model, whatever the mode:

| case | what was fed |
|---|---|
| step-ahead, stop | the EOS (`len(fed)` = `pos - len(prompt)`) |
| sync row, stop | not the EOS (one fewer) |
| `length` | every sent token; the pending one was never fed |

A row cut by `_regrow` (memory) fed one token it never read. That row stores nothing.

**What is stored.** It goes through the normal `_PrefixCache.store` path:

* `inherit` = the old entry's whole chain at its end, so only `[len(prompt) - 1, pos)` is copied out of the
  row (+1–2 MB for a 60-token reply on the 9B).
* A row that crossed `quantized_kv_start` while decoding is the exception: its old entry is fp16 and the row
  quantized, so the whole `[0, pos)` is copied and the fp16 entry stays as the floor entry, by the store's
  existing rules.
* The end state is the row's current GDN state, always taken with `a[[i]]`: a gather into a buffer of its
  own (49 MB on the 9B, as a checkpoint).
  * At B ≥ 2 it is the row's slice of the stacked state. A slice would pin the batch's whole stack.
  * At B = 1 it is gathered too: right after a 2 → 1 shrink the row's arrays are still `_gdn_split`'s
    views of the old stack (Fixes, finding 2).

**Absorb and checkpoints.**

* The old entry (`prompt[:-1]`) is a prefix of the new ids, so it is absorbed. Its checkpoints move over as
  they are (`chain=None`: the prefill store already applied the chain or fresh policy).
* Its end state at `len(prompt) - 1` is kept only where that is a slab end (`ends` = the prompt's bounds
  without `len - 1`). In practice it never is, so it is dropped (−49 MB).
* A regenerate of the same prompt therefore resumes at the last message-start checkpoint. That is the
  assistant header plus the last prompt token, ~10 tokens of prefill, instead of a full hit.

**Memory.** It goes through the same checks as a prefill store:

* `_Seg.estimate` first; then `limit` = budget + prefill reserve − reservations. Over it, the entry is "not
  kept": the old entry stays as it was, nothing is absorbed or evicted.
* Then eviction in `victim` order. The row's reservation was already released (`_end`).
* The reply store does not tick the cache's round clock (`tick=False`). A turn stays 2 ticks (lookup +
  store); with 3, `stale` would call live episodes stale after 2/3 of a round.

**No step-ahead stall.** The copies (KV `_own_slice`s, the GDN gather) of every row finishing in a
`_drop_done` pass are queued with one `mx.async_eval` (one command buffer, not two per row), never
evaluated:

* `_drop_done` runs at the end of `_iterate`, after step k was queued and before step k+1 is built. The
  copies queue behind step k and step k+1 queues behind them; the host never waits.
* An `mx.eval` there would wait for step k and leave the GPU idle while the host builds step k+1. A copy
  queued after step k+1 would make the next `item()` wait for k+1, the step-ahead rule.
* A restore of the entry before the copies ran just waits for them (MLX graph order).

### Resuming after the reply

`_Entry.reply` holds the reply's base: the old entry's length, `len(prompt) - 1`.

* `best()` normally accepts only slab ends of the new prompt. It also accepts the end of an entry with
  `reply` set, even though the reply end is never a slab end (row 5 of the table).
* The resumed prefill then runs `[reply end, next cut)` as its first slab and follows the new prompt's own
  cuts from there, so every later checkpoint is at a cold slab end.
* `store` no longer turns the parent's end state into a checkpoint at the resume point when that point is
  not a slab end. That is a no-op without the knob: resume points were always slab ends.
* The next turn's own prefill store absorbs the reply entry, which is a prefix of it, and drops its end
  state.

### Fallback when the reply re-tokenizes differently

The longest common prefix stops inside the reply, so the entry's end is out of reach and `best()` returns the
deepest slab-end checkpoint at or below it: today's resume point.

* The reply entry is then not a prefix of the new prompt, so the plain absorb rule would leave it behind (its
  end state + reply segment, ~50 MB, until it goes stale).
* So `store` also absorbs an entry whose *prompt part* `ids[:reply]` the new ids cover. Its checkpoints move
  over; its end state does not.
* That also handles a regenerate: the regenerate's prefill store absorbs the older reply, and the newest reply
  wins.

### Numerics (documented, not fixed)

The reply tokens' KV and the end state come from decode kernels, not prefill slabs:

* the 1-row qmv at B ≤ 2;
* mrqmm at 3–16 rows on 8-bit modules;
* cascade-released rows.

The next turn's first slab starts at the reply end instead of the previous slab end. **A hit through a reply
is therefore not bit-identical to a cold prefill of the same tokens.** It is in the same numerics class as
batched decode: B ≤ 2 rows are bitwise solo decodes, and mrqmm is top-1 ≥ 99.7 %.

* `_bounds` determinism ("the answer does not depend on what the cache holds") no longer holds with the knob
  on. A greedy answer can differ between a warm and a cold worker, as it already can under batching.
* An entry may also hold KV and GDN states that come from different, numerically equivalent computations of
  the same tokens: absorbed checkpoints over a reply region.
* `QWEN_PROMPT_CACHE_REPLY=0` (the default) restores today's behaviour bit for bit.

## Results (CPU only; not measured on the GPU yet)

Code:

* `src/worker/worker.py`: the knob; `_Entry.reply` and `best()`; `_PrefixCache.store` gains `reply`, `tick`,
  `sync` and the prompt-part absorb; `_Seg.copy_from(sync=)`; `_Row.fed/entry/plan/why`;
  `_Scheduler._reply_cache` / `_store_reply`, called from `_drop_done`; `_iterate` appends the read tokens.
* `scripts/perf/agentic/replay.py --own-replies` (see the plan below).
* README and CLAUDE.md: the prompt-cache section and the env table.

**Knob off.** Every new branch is gated by an attribute that only the knob sets (`row.fed`, `_Entry.reply`),
plus the `at in ends` guard, which is always true without it. The existing suite passes unchanged.

**CPU suite.** `uv run --group dev pytest -q --ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py --ignore
tests/test_rattn.py`: **241 passed, 2 skipped** (baseline 227 + 14 new in `tests/test_reply_kv.py`, 5:54).
The new tests:

| test | checks |
|---|---|
| `extends_the_entry_and_the_next_turn_resumes_after_it` [dense fp32, dense_fused 8-bit] | 3 turns. Entry ids = prompt + reply, `reply` = len − 1, one entry (absorbed), the prompt-end state gone. Next turn `reused` (= `cached_tokens`) = the previous entry's end; answers == a cold worker; caches close to cold (below); clock 2 ticks a turn; `reply kept … absorbed=1` log. |
| `stop_token_is_in_the_entry_only_when_fed` [step-ahead, sync] | Stop on a stop token: in the entry for a step-ahead row, not for a sync (grammar) row. The next turn resumes after exactly what was fed. |
| `reasoning_replies_are_not_stored` [think ids known, unknown] | Thinking budget: with the think ids unknown, the forced `\n</think>` are in the ids; with them known, nothing is stored (Fixes, finding 6). |
| `reply_retokenized_differently_falls_back` | Other ids in the next prompt's reply: `reused` == the knob-off run's (the slab end before the reply); the stale reply entry is absorbed. |
| `regenerate_resumes_at_the_last_message_start` | Same prompt again: resumes at the deepest message-start checkpoint (< len − 1), answers as cold, its reply replaces the old one. |
| `reply_not_kept_over_the_headroom` | Extension over the headroom: "not kept", the prompt entry unchanged, counted as an eviction. |
| `cancelled_request_stores_nothing_extra` | |
| `entry_gone_stores_nothing` | Prompt entry evicted while decoding: no extension, logged. |
| `batched_replies_match_solo` [dense_fused] | Two rows at B = 2 (the finished row's state gathered out of the stack): each entry == the solo run's, bit for bit; end-state arrays are 1 row; both next turns resume after their replies and answer as cold. |
| `quantized_kv` | 8-bit KV past the start: the extension is quantized, `low` kept, next turn resumes after the reply. A row crossing the start while decoding: the fp16 prompt entry stays as the floor, the extension is the whole row, quantized. |
| `reply_cache_gates` | cancel / error / DFlash / cascade group / vision build nothing. |
| `knob_off_keeps_the_prompt_entry` | |

**CPU numerics** (probe: `docs/perf/2026-10-17/design/reply_kv_numerics_probe.py`, one forward of a T-token slab vs T one-token calls from
the same cache, tiny model):

* T = 1 is bit-identical (the same call).
* From T = 2 the **GDN recurrent state of layer 0 already differs**:
  * dense fp32: conv 6e-7, state 2e-9 (the fp32 matmuls with M rows vs 1 differ too);
  * 8-bit with fp16 scales: conv 0, state 4e-6.
* That spreads to the next layers: KV ~2–4e-6 (fp32) and 3–6e-3 (fp16, ~1 ulp).
* The CPU 8-bit `quantized_matmul` is row-exact. The difference is the GDN step vs slab path.

So **on the CPU a hit through a reply is not bit-identical to a cold prefill** even at B = 1. The tests assert:

* closeness of every cache array, relative to its largest value: 1e-5 for fp32, 1e-2 for fp16, 1e-3 for 8-bit
  KV scales and biases; the packed words are skipped, since a level can flip;
* equal greedy tokens. They agreed in every test, but at fp16 a near-tie may flip on another model; that is
  the documented numerics class.

**Not covered on the CPU:** mrqmm (Metal only), cascade-released rows (gated by `row.group`/`_GroupRef`; a
released row is a plain contiguous row), DFlash (no drafter on CPU; gated by `row.spec`), and the wall-clock
cost of the queued copies.

## GPU validation plan (later, with the GPU free)

1. **Hit through a reply vs cold** (`scripts/perf/prof/agree.py` style, teacher-forced, 9B, fp16 KV):
   * Worker A has the knob on: turn 1 (a 2–8K prompt, greedy reply of 50–500 tokens), then turn 2 = prompt +
     reply + `<|im_end|>\n` + user message.
   * Worker B is cold on turn 2's prompt.
   * Compare the first-token logprobs and a 256-token teacher-forced continuation: top-1 agreement and KL, as
     for mrqmm.
   * Run it with the reply decoded at B = 1, B = 8 (mrqmm), and in a cascade group then released.
   * Expectation: at B ≤ 2 within the solo decode-vs-prefill floor; at B ≥ 3 top-1 ≥ 99.7 % (the mrqmm class).
   * Also check that a 15-prompt greedy corpus over 2 turns is identical with the knob on vs off at B = 1. It
     will not be fully byte-identical; count the flips and their character positions, as in the bump report.
2. **Step-ahead not broken.** `scripts/perf/prof/step_prof.py --ab`, knob on vs off.
   * Rows finishing every few steps: 8 rows, max_tokens 50.
   * **A wave where all 16 rows end on the same step**, as closed-loop waves do: one `async_eval` with 16
     rows' copies + end states, 16 × 49 MB of gathers on the 9B.
   * The reply copies are queued, so the step time should be unchanged (±2 %), and the step after the wave
     no longer than the copy time. An `mx.eval`, or a full command-buffer queue, would show as a bubble.
3. **Load.** `scripts/perf/agentic/replay.py --own-replies --ignore-eos off --episodes 8 --duration-min 15`:
   * Run with the knob off and on, same recipe as R1 (`QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024`),
     interleaved A/B with cooldowns (thermal drift > 20 % between separate runs).
   * `--own-replies` puts the model's own answers into the history. The recorded replies would diverge from
     today's wherever a greedy near-tie flips (batching), and an `ignore_eos` reply holds EOS tokens its text
     does not. Either one would hide the hit.
   * Expectation: prefilled tokens per request −20 % (phase 0); requests/min +10–15 % (prefill is 75–80 % of
     the scheduler's time); `cached_tokens` = previous prompt + reply on every continuing turn; MB per entry
     about the same.
   * The extension adds the reply (~2 MB) and drops the prompt-end state (−49 MB). So it is smaller, not
     larger: the end state moves, it is not added.
   * Check the worker log for `reply kept` on ~every request, few `reply not kept`, and the eviction order
     still serving 8 episodes.
4. **Quantized KV** (27B q8 / 35B q4): one 2-turn run each. The extension is quantized, the next turn resumes
   after the reply, and there is no OOM at the store (`_Seg.estimate` path).

## Fixes (review `docs/perf/2026-10-17/reviews/reply_kv.md`, same day)

All CPU. Line numbers are worker.py after the fixes. Tests are in `src/worker/tests/test_reply_kv.py`.

| # | finding | change | test |
|---|---|---|---|
| 1 | MED: a reply store could evict the running prefill's parent | `store(spare=())` merged into both eviction loops' `keep` (`:1112`, `:1231`, `:1236`); `_store_reply` passes `(task_row.parent,)` (`:3306`), as `_regrow` does. The soft-cap loop now stops when `evict` finds nothing (`and self.evict(...)`), which is the same as before without `spare`. | `test_reply_store_spares_the_running_prefills_parent` (the reviewer's probe as a test; fails without the spare) |
| 2 | LOW: B = 1 end state could be a view of the old 2-row stack | `_reply_cache` gathers `a[[i]]` / `a[[0]]` in every path (`:3245`) | `Owned` spy: keeps every stack and the row's own state arrays alive, and asserts `np.may_share_memory` is false for the entry's end state. Used in `test_batched_replies_match_solo` (kills the slice-at-B ≥ 2 mutant) and `test_lone_row_after_a_shrink_owns_its_state` (X stops and Y hits `length` on the same step: kills the `list(c.cache)` mutant) |
| 3 | LOW: 2 `async_eval` per row | `_Seg.copy_from(sync=False)` leaves the copies lazy. `_store_reply` returns the entry; `_drop_done` queues every kept entry's arrays in one `mx.async_eval` (`:3220`). If that raises, those reply entries are dropped (no lazy entry that cannot run). The GPU plan item 2 now includes a 16-rows-end-on-one-step wave. | `test_one_async_eval_for_rows_ending_together`: 2 rows end in one pass, exactly one `async_eval` and no `mx.eval` inside `_drop_done` |
| 4 | LOW: prompt-part absorb could move checkpoints ≥ L | absorbed checkpoints only `p < L` (`:1184`); a no-op for full-prefix covers | `test_prompt_part_absorb_keeps_only_reachable_checkpoints`: chain policy off, which would otherwise hide it; fails without the filter |
| 5 | LOW: stop strings can store | wording in the knob comment (worker.py `:673`), README, CLAUDE.md and the "Stop strings" caveat above: usually a cancel, a late cancel stores harmlessly | — (documented) |
| 6 | LOW: thinking-on replies wasted | `_store_reply` skips a reply when the prompt ends inside an open `<think>` or any fed token is a think id (`eng._think`), and logs `reply not kept: reasoning` (`:3295`). The prompt entry and its end state stay. | `test_reasoning_replies_are_not_stored` [think ids known → skipped, entry intact; unknown → stored with the forced tokens], `test_thinking_off_replies_are_stored` (the thinking-off header's closed `</think>` does not block) |
| 7 | LOW: log parsing | the over-headroom line for a reply starts `prompt cache reply not kept: over the batch's headroom` (`:1207`); the others are `id=… prompt cache reply not kept: its prompt entry is gone` / `…: reasoning`, plus `… reply kept: n tokens (+g generated)` (`:3318`; g is now the generated count, the prompt's last token no longer included) | `worker_log.py` events `reply_kept`, `reply_not_kept`, `reply_entry_gone`, `reply_reasoning` (`:42`, `:69`, `:100`); `analyze_worker_log.py` prints a `reply KV:` line (`:7`). Checked on a synthetic log: each counted once, and the prefill `not kept` stays in `not_kept`. `test_reply_not_kept_over_the_headroom` asserts the new prefix |
| 8 | LOW: `--own-replies` content only | `replay.py` `mine()` (`:86`) puts our content and our `tool_calls` into the history message, and drops the recorded `tool_calls` when we made none | — (checked by hand on two dicts) |

CPU suite after the fixes (`--ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py --ignore tests/test_rattn.py`): **247 passed, 2 skipped** (241 before + 6 new tests; the forced-thinking test became the 2-case reasoning test). Mutants checked: slice at B ≥ 2, view at B = 1, no spare, no `p < L`; each kills at least one test.

### Round 2 (review "## Round 2")

| # | finding | change | test |
|---|---|---|---|
| R2-1 | MED: a reply entry got the end-of-decode recency. A just-finished conversation, with its next lookup imminent, became the most recently used entry and was evicted first under the live-MRU rule; within one pass each reply store could evict the previous one. | `store(clock=)` (worker.py `:1117`, applied at `:1212`) sets `new.used, new.seq` before any eviction sees the entry. `_store_reply` passes the absorbed prompt entry's `(e.used, e.seq)` (`:3341`): a reply continues the same turn. | `test_reply_keeps_its_prompts_recency`: B finished with its reply stored, A (prefilled later) still decoding; `victim` is A's entry, and a third store under a full cap evicts A's, not B's. It fails with `clock=None`. `test_three_rows_ending_together_keep_their_replies`: 3 rows end in one pass with the cap set to exactly their three entries; all three are kept and nothing is evicted. This is a regression test: when they fit, no eviction runs, so it also passes without the clock. |
| R2-2 | LOW: a failed single `async_eval` lost the absorbed prompt entries | `_store_reply` returns `(new, covered)`. On failure `_drop_done` removes the reply entries, puts back the absorbed prompt entries that are no longer in the cache, counts the reply entries as evictions, recounts and logs `prompt cache reply not kept: queueing the copies failed …` (`:3244`). The copies stay asynchronous. | `test_failed_async_eval_restores_the_prompt_entries` (`async_eval` raising inside `_drop_done`): the prompt entry is back with `reply` None, the bytes match, 1 eviction. It fails without the restore. |
| R2-3 | LOW: with `spare`, a reply could be published over the headroom without a log; the soft-cap warning said "alone" | The "not kept" check is `bytes_of([new, *spared entries it does not absorb]) + extra > limit` (`:1217`), up front, with nothing changed; the log says `… beside the running prefill's entry`. The soft-cap warning says "beside the running prefill's entry" when entries other than the new one are left (`:1272`). Without `spare`, both are the old checks. | `test_reply_beside_the_spared_parent_over_the_headroom`: headroom = D's prompt entry + C's parent; the extension fits alone, not beside the parent, so it is not kept and logged. It fails with the old `new.nbytes` check. |

CPU suite after round 2 (`--ignore tests/test_mrqmm.py --ignore tests/test_pfqmm.py --ignore tests/test_rattn.py`): **251 passed, 2 skipped** (247 + 4 new). Mutants: `clock=None`, no restore, the old `new.nbytes` limit check: each fails its test.

# Preempt a row into the prompt cache instead of cutting it (`QWEN_PREEMPT`, item 5)

2026-10-24. Today, a row that grows past budget + prefill reserve while other rows run is ended with
finish `length` (`_regrow`). For an agent that is a broken tool call. DeepInfer does `_preempt` →
`reset_for_recompute` + `waiting.push_front`, and picks the row with the longest output as the victim
(`csrc/engine/scheduler.hpp:622-643,766-790`). We already have a cheaper resume than a recompute: the
reply-KV entry (`_reply_cache` / `pc.store(reply=…)`).

CPU-only work so far. The GPU plan is at the end.

## What

`QWEN_PREEMPT=N` (0 = off, the default; N = the most preemptions per request).

1. **Decision before the step** (`_iterate` → `_preempt_rows`, only when the knob is on). For every row
   whose next step would re-account it (`pos + 1 ≥ tokens`) and that is `_preemptable`,
   `_regrow(row, ahead=1)` runs the same check the step would. The prompt cache gives way first (victim
   order, parents spared). If it still fails, the row is preempted. When it succeeds, the row is just
   re-accounted one step early: the same `tokens` / bytes `_regrow` would set in the step.
   - It has to happen before the step. The GDN state cannot be trimmed back, and an entry must hold
     exactly "everything but the next input". After the step the KV would also hold the pending token.
2. **`_preempt(row)`.**
   - Read the pending token `y`: `y.item()` waits for its own step only. This is outside `_step`, so the
     sync guard does not apply.
   - Handle `y` exactly as the read loop does: thinking budget, `_Control.stop`, grammar `consume` and the
     bad-word history in sync mode, the logprobs event, `n += 1`. A stop token or `max_tokens` ends the
     row normally instead.
   - Then do to `y` what `_logits` does to a next input: the penalty count and `_Control.n`. The resumed
     row's first input is `first` and is skipped there.
3. **Store** (`_store_preempt`). The entry is ids = prompt + generated[:-1] (exactly the row's KV), with
   the row's recurrent state as its end state.
   - It goes through `pc.store` with `reply=`: resumable at its end, which is no slab end.
   - **No copy** (review 2026-10-24): the row is dropped first, and its own KV buffers become the
     entry's single segment over `[0, len)` (`_Seg.adopt`: views `[..., :len, :]`; nothing writes them
     again, the row never steps). `pc.store(seg=…)` attaches it instead of copying. The entry inherits
     nothing and absorbs the row's prompt entry (its checkpoints move over), so `_regrow(ahead=1)` having
     evicted that prompt entry first costs nothing. The buffers keep their capacity tail (< one
     256-token KVCache step), which `nbytes` does not count.
   - The limit is budget + prefill reserve − reserved **without the row** (its reservation is released
     before the store: its buffers are what the entry holds). Before this fix the limit still counted
     the row and the entry needed a fresh copy, so under real pressure it was almost never kept and the
     resume was a cold recompute. The real-budget test now checks "KV kept" and a full HIT.
   - `tick=False`, and the prompt entry's clock is kept.
   - The end-state gathers are evaluated right there (a host wait for a rare event, before the step). If
     that eval fails, the entry goes and the entries it absorbed come back.
   - Unlike a reply store, the entry is kept also when the output has reasoning: the resumed prompt
     contains it exactly.
4. **Requeue.**
   - The row is dropped without an event: `done`, its reservation released, `fed` = None (so no reply
     store at the drop).
   - The request goes back to the **head** of the queue with `ids` = prompt +
     generated, `max_tokens` −= sent, `carry` = (sampler, penalties), `preempts` += 1, and its `bounds` /
     `peeked` / `keep_since` memos reset. Several rows preempted before the same step go back in batch
     (admission) order, ahead of what was waiting (`extendleft(reversed(…))`).
   - The original prompt is kept in `req.orig`. The handler reads `prompt_ids` (Usage.prompt_tokens, the
     tool-call id seed, `prompt_logprobs`).
5. **Resume.** Normal admission.
   - `_setup` takes the carried sampler (the RNG key goes on: a seeded stream continues where it was,
     not from the start) and penalty counts, and skips `_Control.build` (its masks exist).
   - The lookup finds the entry at `len(ids) − 1`, prefills nothing, and the row joins with `y` as its
     first input at the same position. The KV and state are the row's own bits.
   - `t0` / `queue_ms` / `reused` (Usage `cached_tokens`) keep the first prefill's values.
   - `_spec_ok` is False (no DFlash after a preemption).
   - `_fits` never makes it "give up" its hit: prompt + generated is worth waiting for. It waits until a
     row ends; an empty batch always admits.
   - The lane (item 4) takes it at once beside a long prefill: it has 0 tokens to prefill.
6. **Fallback: recompute.** If the entry is not kept (over the headroom), or is evicted while the request
   waits, the request still resumes, by prefilling prompt + generated cold. Its tokens stay correct; the
   KV from there on is prefill bits instead of decode bits (the reply-KV numerics class).

## What the client sees

The handler never knows. Its detokenizer, `_StopFilter`, `_GenState`, tool-call buffer and logprobs list
live on the handler thread and just keep reading `("tok", …)` events from the same `req.out`. The
scheduler-side state is carried:

| state | where | how |
|---|---|---|
| tokens sent / `max_tokens` | `row.n`, `req.max_tokens` | `max_tokens -= n`; the total length cap `len(ids) + max_tokens` is unchanged |
| sampler key (seed stream) | `row.sampler.key` | the `_Sampler` object is carried; no draw happened for an unsent token (the decision comes before the step) |
| penalties | `row.pen.counts` (+ prompt mask) | carried; `y` counted at preemption |
| `_Control` (min_tokens n, history, masks) | `req.ctl` | on the request; `n += 1` for `y`; not rebuilt |
| grammar matcher | `req.guide` | on the request; `consume(y)` in sync mode as the read loop |
| thinking budget | `req.thinking` | on the request; a pending forced id is popped after the resumed row's first step, as it would be after the next step |
| logprobs | `row.lp` of `y` | sent with `y`; the resumed row's first step builds the next one |
| reply KV of the resumed row | `row.fed` / `row.entry` | fresh: the resumed prefill's store is the prompt entry it extends |
| Usage | `req.orig`, `t0`, `queue_ms`, `reused` | the first prefill's |

**What cannot be preempted** (cut with `length` as before, `_preemptable`):

- vision: no prompt cache;
- DFlash rows (`row.spec`): no reply KV, own loop;
- rows in a cascade group: their KV is in the group's stack. A member past P // 2 is released anyway;
- `prompt_logprobs` requests: their scores belong to the first prefill;
- `prompt_cache: "read"` (no store) or `"off"` (no lookup);
- the prompt cache off (no `row.fed`);
- a row whose `y` is input-only (`first`);
- a lone row: never cut anyway;
- a request already preempted N times: the livelock guard.

## Victim: the growing row itself

The row that would be cut today is the one preempted. Alternatives:

- DeepInfer picks the longest output;
- vLLM picks the youngest request.

Reasons for the growing row:

1. **Strictly better than today for every request.** The other rows go on untouched, and the growing row
   gets a continuation instead of a cut. Preempting another row would hand a pause (or, if its entry is
   evicted, a recompute) to a request that did nothing wrong.
2. **It is usually the longest output anyway.** It is the row past its `QWEN_BATCH_RESERVE_TOKENS`
   reservation (1024 in the AppWorld config), often a runaway answer.
3. **Its KV goes from "reserved" to "evictable on demand".** The memory problem is solved the way the
   batch already solves it: prompt-cache entries give way first, in victim order. The preempted entry is
   `wanted` by the head of the queue, so it goes last.
4. **No cascade of recomputes, one decision point, simple.**

Cost: the preempted request waits for memory at the head of the queue (FIFO behind it). That is the same
wait any large request has.

## Numerics

- No resumed bits are new. The KV over the generated part is the row's own decode KV (its own buffers), the GDN
  state is the row's own (gathered), and the next step is the same decode call at the same position.
- At B ≤ 2 (the per-row qmv path) a preempted stream is bit-identical to an uninterrupted one. The GPU
  check verifies this.
- At B ≥ 3 the resumed row may decode at another batch size than it would have, which is the mrqmm
  row-count class (a near-tie can flip). A recompute fallback is prefill bits.
- Determinism needs `QWEN_PREEMPT=0` along with the other knobs (`QWEN_MAX_BATCH=1`,
  `QWEN_PROMPT_CACHE_REPLY=0`).

## Memory

- **The store allocates nothing** but the end state's gathers (one row's recurrent state). The row's
  reservation moves into pc.bytes (evictable), and the prompt entry it absorbs goes (its segments too,
  unless another entry shares them).
- **Resume costs 2× the row's KV until the entry goes.** The reservation is prompt + generated +
  min(left, `RESERVE_TOKENS`), and the restore copies the entry (as any hit). The resumed prefill's store
  re-publishes the same pieces as its prompt entry (no new copy), which the next reply-KV store extends.
  Both are counted (`_fits`: reserved + need + pc.bytes), so a preempted request waits at the head until
  rows end; it is never an OOM.
- **Not done:** handing the entry's buffers back to the resumed row instead of copying them (review item
  1c). A KVCache writes its next token in place only when nothing else holds its buffer (MLX
  copy-on-write). Moving the buffers would therefore mean dropping the entry and storing no prompt entry
  for the resumed row, which loses its reply KV for the next turn: the AppWorld gain (−21 % prefill
  tokens). Otherwise the first decode step copies the buffer anyway. It also needs a KVCache built
  around a view whose capacity tail the view hides. Too invasive for the gain (a transient at a rare
  event).

## Default unchanged

- `_preempt_rows` runs only when `PREEMPT` is set.
- `_regrow(row)` with `ahead=0` behaves as before. The return values are new, but the callers ignore them.
- `row.fed` is tracked for `PROMPT_CACHE_REPLY or PREEMPT`. `_drop_done` stores replies only under
  `PROMPT_CACHE_REPLY`, the same as before since `fed` was None without it.
- `_fits` "never give up" applies only to `req.preempts > 0`, and `_spec_ok` only adds `preempts == 0`.
- The handler reads `prompt_ids`, which equals `ids` unless the request was preempted.

## Tests (`src/worker/tests/test_preempt.py`, CPU, ~8 s; `QWEN_DEBUG_NO_SYNC` on)

- Forced trigger (a `_regrow` stub that fails the headroom like a small budget), compared to the same
  requests without preemption. Variants: greedy, seeded sampling, strong penalties (the final penalty
  counts are compared too: a restarted count fails), logprobs (values within 1e-4, top ids equal),
  ignore_eos + min_tokens. Checked: identical events, length, finish, Usage (prompt_tokens, cached = the
  first prefill's), resume through a prompt-cache HIT, accounting back to 0.
- A grammar row (sync mode) with a thinking budget, preempted before / at / after the forced
  `\n</think>`: identical events, the matcher saw exactly the stream.
- `QWEN_PREEMPT=1` and a second overflow: cut with `length`; the tokens up to there are identical.
- A lone row is never preempted. The eligibility rules hold (the count, no `fed`).
- The real budget, no forcing: off → `length` before 200 tokens; on → preempted, resumes after the other
  row ends, all 200 tokens identical to an unconstrained run.

## GPU plan (`scripts/perf/preempt/preempt_check.py`, ~10 min, peak ~12–13 GB)

```zsh
cd src/worker && setopt NO_BG_NICE; mkdir -p /tmp/perf
QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 \
QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 \
PERF_OUT=/tmp/perf PYTHONPATH=. \
.venv/bin/python ../../scripts/perf/preempt/preempt_check.py 2>&1 | tee /tmp/perf/preempt_check.log
```

- **Phase 0, the sync guard on Metal.** 8 rows (greedy / seeded / penalties / logprobs; mrqmm,
  GDN-fuse, grouped sampling), then 1 row alone. PASS: no SyncInStep.
- **Phase 1, exactness at max_batch 2** (the bit-exact path). A: 1.5K prompt, 2500 tokens, logprobs
  top 3. B: 2.5K prompt, 2600 tokens. All ignore_eos. A's growth past its 1024-token reservation is
  forced to fail twice. A is preempted, resumes beside B, is preempted again and resumes. Run greedy, then
  seeded (T 0.7). PASS: 2 preemptions logged; both streams token-identical to the no-preemption
  reference; text identical; logprobs bitwise equal; finish `length` at max_tokens; Usage equal.
- **Phase 2, informational.** max_batch 4, four 2000-token rows. Once all four run, the **real** budget is
  cut to what they hold + 16 MB (prefill reserve 0). Report: preemptions, completeness (every row
  reaches 2000 tokens, so nothing is cut), and how many streams stay identical (B ≥ 3 class).
- Output: `/tmp/perf/preempt_check.json` + log, then copy to `docs/perf/2026-10-24/raw/`.
- Optional, for end-to-end continuity through the gateway: a small `QWEN_BATCH_MEM_GB` (e.g. 1.5) run
  of `scripts/perf/agentic/replay.py --own-replies --episodes 6 --duration-min 5` with `QWEN_PREEMPT=2`.
  Expect `preempted at` lines in the worker log, no `ended at … tokens` lines, and every reply's
  completion_tokens equal to its max_tokens (ignore_eos on).

# Prefill lane for short requests (`QWEN_PREFILL_LANE`, item 4)

2026-10-24. Motivation: the 110K episode of `../../2026-10-17/REPORT-appworld-run-1017.md`. A 110–118K
MISS prefilled for ~8 min. The short hit requests queued behind it (10–27K prompts, a few hundred new
tokens each, TTFT < 1–2 s on their own) hit the client's 600 s timeout: 19 cancels, 1 task closed in
72 min. Prefill is serialized (one `task`, one slab per tick) and admission is FIFO with no
overtaking, so nothing could start until the long prefill finished. DeepInfer handles this with length
tiers and quotas at the cluster level (`doc/inference_deployment_qa.md`, Prefill Load Balancer); on
the GPU it is strict FIFO (`csrc/engine/scheduler.hpp:369-371,386-388`). We have one GPU, so the tier
has to live inside the scheduler.

CPU-only work so far. No GPU numbers yet; the plan is at the end.

## What

`QWEN_PREFILL_LANE=N` (tokens, 0 = off, the default) adds a second prefill slot, the **lane**
(`_Scheduler.lane` / `lane_row`), next to the main one (`task` / `task_row`).

- **Who.** When a plain text prefill is running (`task` is a `_prefill` generator, not DFlash), the first
  waiting request with at most N tokens left to prefill qualifies. The count runs from where its
  prompt-cache lookup would resume (`_lane_tokens`: `_peek`, the same scan `lookup` makes) to its end.
  The request must be text and not cancelled.
- **When.** The lane is admitted on prefill's turn (`tick`). Its prefill then runs **before the main
  prefill's next slab**, slab by slab, until it ends; then it joins the batch like any row.
  `QWEN_PREFILL_LANE_BURST` (4) lane requests in a row at most, then the main prefill gets a slab.
  The counter `lane_run` resets on every main slab.
- **Tokens, not slabs.** A next turn that resumes right after its reply (reply KV) is cut at the new
  message starts (`_bounds`: `<|im_start|>` ≥ 256 tokens after the previous cut). A 300-token suffix is
  therefore usually 2–3 slabs, the first only a few tokens long. "≤ 1 slab of work" is a token count.
  Recommended setting: `QWEN_PREFILL_LANE=2048` (= `PREFILL_STEP`), at most one slab's worth of work per
  lane request.
- **Default unchanged.** With the knob at 0, `lane` stays None and the tick runs exactly the old code
  (`self.lane_run = 0; self._advance()`). Every helper added for the lane returns the old value when no
  second prefill exists.

## Invariants and what had to change

**"The running prefill" was assumed unique.** Every reader of `task_row` was rechecked:

| place | before | now |
|---|---|---|
| `_regrow` (evicts on growth) | spares `task_row.parent` | spares `_parents()`: both prefills' parents |
| `_store_reply` (reply KV) | `spare = task_row.parent` | `spare = _parents()` |
| prefix store (`_prompt_cache.store`) | no spare (the storing prefill was the only one) | `spare=_parents(but=row)`: the other prefill's parent |
| `_fits` (admission eviction) | spared only the request's own hit | also spares `_parents()`; a request never "gives up" a hit that is a running prefill's parent |
| `_wanted` | admitted-not-started `task_row` | both slots' not-started rows |
| `_admit` | `_fits(req, len(rows))`; peak reset when no rows | counts the lane row; no peak reset while a lane runs |
| `_spec_ok` (DFlash) | `not rows and not waiting` | also `lane is None` |
| `_close`, `_fail_rows`, `_run`'s wait, `_log` | `task_row` | `_tasks()` (both); the `batch:` line shows `prefilling=<main>+lane:<id>` |
| `_advance` | one generator | `_next(lane)`; `_advance()` keeps its signature (`sched_timer.py` wraps it), `_advance_lane()` is new |

**mRoPE state (`_rope_deltas` / `_position_ids`).** These are written by a prefill and read by its later
text slabs (the delta branch of `LanguageModel.__call__`). A cold first slab writes both through
`get_rope_index`, and a hit sets zeros / None. Two text prefills write the same values (zeros, an
arange from 0), so interleaving them is harmless. Even so, with the lane on each task row keeps the
state its own last slab left (`row.rope`) and gets it back before its next slab (`_next`). The main
prefill therefore sees exactly the state it would have seen without the lane: the same 2D/3D position
path, the same bits.

- Vision requests never enter the lane (`_lane_tokens` returns None).
- A vision main prefill is allowed. Its slabs pass explicit `position_ids`, and the swap keeps its state anyway.
- Decode never reads this state: positions are explicit.

**Determinism.** The main prefill's slab ends are `_bounds(its ids)`, which do not depend on load. The
lane request is cut by its own `_bounds`. Neither changes, so the main prefill's caches are bit-identical
with the lane on or off. CPU test: the slab offsets and the answers are equal, and both equal a cold
serial run. GPU check: the cache checksum below.

**Memory.**

- Two partial caches can exist (main mid-prefill, plus the lane's). Each is inside its own reservation:
  prompt + min(max_tokens, `QWEN_BATCH_RESERVE_TOKENS`), counted at admission.
- The lane goes through the same `_fits` as any admission: budget, live check (`live + need + prefill
  reserve ≤ cap`), and prompt-cache eviction in victim order.
- It never gives up its own hit (`_fits(lane=True)`). Without the hit it would be a long request, so it
  waits.
- When it overtakes the head of the queue, the budget must also hold the head's reservation
  (`extra=_reserve(head)`), and the row count includes the head's row. The lane never makes the head
  wait longer for memory.
- Only one slab runs per tick. Between ticks a prefill keeps only its caches and its input embeddings,
  not activations, so one prefill reserve still covers both.

**Prompt cache.** The lane's store must not evict the main prefill's parent. That prefill's `_Inherit`
holds the parent's pieces until it stores, so evicting the parent would make the byte count lie. Tested:
with `QWEN_PROMPT_CACHE_GB` ≈ 0, the parent survives every lane store; the test fails with `_parents()`
stubbed out.

**Known gap.** A store can still **absorb** a running prefill's parent: the covered entries are not
filtered by `spare`, and the reply store already behaves this way. That needs a new entry whose ids
extend the parent, i.e. two requests continuing the same conversation at once. Absorbing moves the
checkpoints and leaves the pieces held by the `_Inherit`, so `pc.bytes` undercounts the parent's
segments until that prefill stores. Bounded and rare (not seen in AppWorld); not fixed here.

**Partial store, cancel, failure.** These are per-row, so nothing changed. A cancelled lane request stops
at its next slab boundary as before. With `N ≤ 2048` a lane request is usually a single tick.

**Others' hits** (review 2026-10-24, both repros now in `test_lane.py`):

- No lane while the main prefill is admitted but not started (`task_row.started`). Its lookup has not
  run yet, so its hit is no parent; a lane admission in the same tick evicted it (main reused 61 → 0).
- When the lane request overtakes waiting requests, the entries their peeks find are spared by the
  lane's `_fits` (`protect=`). The budget held the head's reservation but not its hit (head reused
  61 → 0). With the hit spared, the lane request waits instead.

**`_wanted`, `req.peeked`.** Both not-started rows are inserted. Peek memos are per request and still
valid: `_lane_tokens` uses the memoized `_peek`.

**FIFO.** Non-lane requests keep their order. A lane request may jump ahead only of waiting requests that
do not qualify, and only the first qualifying request is a candidate. If it does not fit, nothing behind
it is taken (FIFO among lane requests).

**DFlash.** Untouched. No lane beside a DFlash task. No DFlash admission while a lane runs (`_spec_ok`).
Lane rows never go through `_spec_ok`.

**Fairness.** At most `QWEN_PREFILL_LANE_BURST` lane requests between two main slabs, each ≤ N tokens.
Each lane slab sets `resume` (`_slab_done`) like any slab, so decode keeps its share. The cost to the main
prefill:

- the lane slabs themselves;
- the decode share of the rows that now decode during it. With B of 16 rows decoding, prefill gets
  `1 − 0.5·B/16` of the time.

Expected cost: +10–25 % on the long prefill's TTFT while 4–6 short rows decode, against minutes saved
for each short request.

## Tests (`src/worker/tests/test_lane.py`, CPU, ~10 s)

- short hits beside a long cold prefill: answers lane on == lane off == cold serial; same main slab
  offsets; the shorts decode while the long one still prefills (not with the lane off); clean accounting;
- burst: at most `BURST` lane requests between two main slabs;
- overtaking only when the head fits too; with a budget that cannot hold the head too: no lane, FIFO;
- an eligible head that does not fit blocks the lane (no search past it);
- vision: never in the lane; a text lane beside a vision main prefill answers as alone;
- the main prefill's parent is never evicted by the lane's store (`PROMPT_CACHE_GB` ≈ 0);
- a two-slab lane request runs over two ticks before the main prefill's next slab; both answers == cold serial.

All run with `QWEN_DEBUG_NO_SYNC` on. `conftest.run` also waits for `sched.lane`.

## GPU plan (`scripts/perf/lane/lane_ab.py`)

In-process A/B in **one** worker process (CLAUDE.md "Measuring performance"): `worker.PREFILL_LANE` is
toggled between rounds, rounds interleaved `on,off,off,on,on,off`, medians per mode.

- Warm-up: 6 conversations (system ~1.2K + user 3–10.5K tokens), one 48-token answer each.
- Per round:
  1. Reset the prompt cache to the warm-up snapshot.
  2. Start a long cold prompt (`LANE_LONG`, 40K: ~100 s of prefill).
  3. From 5 s on, every 4 s, send the 6 next turns (hit after the reply + ~150–300 new tokens).
- A store hook takes a sha1 of the long prompt's prefill caches (KV + recurrent state) at its prompt-cache
  store.

```zsh
cd src/worker && setopt NO_BG_NICE; mkdir -p /tmp/perf
QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 \
QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 \
QWEN_PREFILL_LANE=2048 PERF_OUT=/tmp/perf PYTHONPATH=. \
.venv/bin/python ../../scripts/perf/lane/lane_ab.py 2>&1 | tee /tmp/perf/lane_ab.log
```

`QWEN_PREFILL_LANE` in the env only sets the value the "on" rounds use (default 2048). The script sets
it or 0 per round.

| criterion | pass |
|---|---|
| the long prompt's cache checksum | identical in all 6 rounds (bitwise: the lane does not touch its slabs / positions on Metal) |
| short TTFT, lane on | p50 ≤ 4 s (about half a main slab of wait + its own), max ≤ 15 s |
| short TTFT, lane off | p50 ≥ 20 s (they wait for the long prefill: sanity of the setup) |
| long TTFT on / off (medians) | ≤ 1.35 |
| errors / finishes | none / stop or length |

- Informational: how many short answers are equal across modes (B ≥ 3 decode is not batch-invariant).
- **Time:** warm-up ~2 min plus 6 rounds × ~2.5 min ≈ 17 min. **Peak:** ~16–18 GB (weights 9.5, the 40K
  prefill + 6 short rows + a ≤ 6 GB cache).
- Output: `/tmp/perf/lane_ab.json` + log; copy them to `docs/perf/2026-10-24/raw/`.
- Optional, after a PASS: one AppWorld replay (`scripts/perf/agentic/run_replay_batch.sh` with
  `QWEN_PREFILL_LANE=2048` added to its env), compared to the 10-17 replay numbers (different runs,
  informational only).

## GPU results (2026-10-24, 9B 8-bit, M1 Max; raw: `../raw/lane/`)

One process, rounds `on,off,off,on,on,off`, long cold prompt 40 296 tokens, 6 short next turns
(hit after the reply, 451–507 new tokens each, so 1 lane request = 1–2 slabs). The machine was quiet:
rounds of the same mode agree within 0.3 s.

| criterion | result | |
|---|---|---|
| the long prompt's cache checksum | `0383ed83e3432bc4` in all 6 rounds (both runs) | **PASS** |
| short TTFT lane on, client (first Token event) | p50 **6.50 s**, max 10.25 s | **FAIL** (p50 > 4 s); max PASS |
| short TTFT lane on, worker (`queue_ms + ttft_ms`, first token) | p50 **3.05 s**, max 4.84 s | (informational) |
| short TTFT lane off | client p50 72.1 s (worker 70.4 s) | PASS (setup sane) |
| long TTFT on / off (medians) | 92.10 / 81.58 s = **1.129** | PASS (≤ 1.35) |
| errors / finishes | none / all `length` (48 tokens) | PASS |

- Short answers: identical in all 54 on/off pairs.
- Short wall time (arrival to the last token, 48 tokens): lane on median 19.3 s (17.6–26.9) against
  75.8 s (66.2–85.7) off.
- Long wall: 92.8–93.2 s on, 89.5–89.9 s off.
- Peak 19.35 GB (MLX peak, the 40K prefill + 6 rows + 3.4–5.4 GB of cache). The plan assumed 16–18.

**Why the client's TTFT misses 4 s.** The lane does what it should: a short request is admitted at the
end of the running main slab (about 2 s of wait on average), its prefill takes 1.1–1.5 s, and its first
token comes about 3 s after arrival. The client's first Token event comes another 3.5–5.4 s later, for
two reasons that have nothing to do with the lane:

1. `_BPEDetokenizer` flushes text only on a token that starts with a space. The answers start with JSON
   and newlines, so the first text comes several tokens in. With the lane off, the same gap is 0.04–2.7 s
   at full decode speed.
2. While the main prefill runs, decode gets `1 − s = 0.5·B/16` of the time: about 4 decode steps per main
   slab (4 s at a 30–40K offset) at B = 1–4. Each token before the first space-led one therefore costs
   about a main slab.

Lowering the client TTFT further is a decode-share policy question (a floor for decode while a long
prefill runs) or a detokenizer one (also flushing on newline-led tokens). Both are outside this item.

**Fix made at the gate** (`_lane_first`, commit a9f8110). After a lane row
joins, decode runs until that row has read its first token, and only then does the main prefill take its
next slab. Its first token needs two decode steps (step-ahead), and at a ~3 % decode share the next main
slab used to fit between them. The cost is 2 decode steps per lane request, and the main slabs are
unchanged (same checksum). Effect on the client TTFT here: none to −0.4 s (the first text needs more
tokens anyway). The worker-side TTFT is now bounded by "rest of the main slab + own prefill + 2 steps".
New CPU test: `test_lane_row_reads_its_first_token_before_the_next_main_slab` (share 1.0 and 0.5; fails
without the fix).

**Recommendation.** The lane's own goal is met: short requests finish in 19 s instead of 76 s (×4) at a
cost of +13 % on the long TTFT, with the long prefill bit-identical and the answers identical. The 4 s
client-TTFT criterion assumed that the first token's text follows the lane prefill directly. I recommend
`QWEN_PREFILL_LANE=2048` as the default. The AppWorld replay (optional) was not run.

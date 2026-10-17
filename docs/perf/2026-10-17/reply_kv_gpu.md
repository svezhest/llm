# GPU gate: reply KV in the prompt cache (`QWEN_PROMPT_CACHE_REPLY`), 2026-10-17

Gate for commit f91a207 (design: `design/reply_kv.md`). Method follows the 16.10 gate
(`../2026-10-16/chain_evict_gpu.md`): through our gateway on :8090, fresh workers per run,
plus one in-process teacher-forced comparison.

**Result: every functional gate passes.** One sub-item of gate 2 is not met as worded: a reply
skipped because its row was in a cascade group is skipped **silently**, with no log line. Two LOW
findings are listed below. Recommendation: switch the default to on after a one-line log for the
grouped skip. The standard worker is left with the knob **off**, as the brief asked, since that
sub-item did not pass.

## Results at a glance

| gate | result |
|---|---|
| 1. Hit through the reply vs cold, B=1 | **PASS**. Teacher-forced over 862 positions: top-1 **99.88 %** (861/862), KL mean **5.8e-5** (gate ≤ 1e-4). Greedy texts 11/12 equal; the one flip is an exact tie in the cold logits (margin 0.0). `cached_tokens` is exact on every reply-kept turn. |
| 2. B=8 wave + cascade | **PASS** (functional): 64/64 + 7/7 requests ok, 0 errors, 8-row waves stored in one pass, cascade groups formed, released rows' replies resumable. **Not met:** grouped rows are skipped with no log line (finding F1). |
| 3. Step cost, 8 × 6 closed loop, REPLY 0/1/0/1 | **PASS**. No visible decode cost: per-request tok/s within drift. Turns ≥ 2 prefill **−17 %** of tokens; worker ttft is the lowest of all 8 runs (−9 % vs the bracketing off runs). |
| 4. AppWorld replay, own replies, 0/1/0 | **PASS**. **−21 %** prefilled tokens per call on matched hits (off vs off: ±0.0 %). Requests/min 19.3 / **20.6** / 19.4 (+6.5 %). 212 reply kept, 0 not kept, 2 entry-gone. 99 % of continuing calls resume after the reply. 0 errors. |

## Environment

- **Machine:** M1 Max 32 GB (MacBookPro18,2), `iogpu.wired_limit_mb=28672`; mlx 0.32.3, mlx-vlm 0.7.6.
- **GPU lock:** held 04:20–05:43. Only one model process ran at a time.
- **Workers:** at nice 0 (`setopt NO_BG_NICE` in the launchers).
- **Code:** every gate ran on HEAD **f91a207**, `worker.py` md5 `aeddd4a1fc524057344cdbc492dada5a`, clean tree.
  - At 05:41:48 another session committed 43e976c (`worker.py` changed). The last gate worker had started at 05:29, before that commit.
  - The standard worker left on :50051 (05:42) runs 43e976c, which is what HEAD is now.
- **Model and base env:** `ornith15-9b-mlx-8bit` (absolute path). Every worker had `QWEN_KV_BITS=0 QWEN_PRESENCE_PENALTY=0 QWEN_BATCH_RESERVE_TOKENS=1024`, plus the per-run env below.
- **Gateway:** our :8090 (pid 55417, `GATEWAY_THINKING=off`), reused.
  - Requests: temperature 0, presence_penalty 0, `chat_template_kwargs.enable_thinking=false`, non-streaming.
  - Each request carries a `request_id` (`rk-<run>-<conv>-<turn>`). The checker counts any foreign Generates in each run's window.
  - The :8080 gateway (pid 95089) is also connected to :50051. Every gate window had **0 foreign requests**.
- **GPU speed drifted** about 25 % during gate 3 (see there). Every comparison is interleaved.

## 1. Hit through the reply vs cold (B=1)

**Workers:**

| side | env |
|---|---|
| hit | `QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_REPLY=1 QWEN_MAX_BATCH=1 QWEN_PROMPT_CACHE_GB=6` |
| cold | `QWEN_PROMPT_CACHE=0 QWEN_MAX_BATCH=1` |

**Conversations:**

- System prompt: a 2479-token README slice.
- 6 user turns of 286–683 tokens: a BACKLOG excerpt plus a question; max_tokens 96.
- Each next turn appends the answer exactly as returned.
- Two conversations:
  - `g1`.
  - `g1b`: same system prompt, other excerpts. It was added because g1's turn-2 reply was not stored (F2).
- The cold side was sent the hit run's exact messages.

### Accounting (worker log, hit run)

```
g1  t1 MISS → store 2784 tokens; reply kept: 2843 tokens (+58 generated)         finish=stop   gen 57
g1  t2 HIT 2843/3159 (= 2785 + 57 + EOS)         reply not kept: reasoning in the reply (thinking on)   finish=length
g1  t3 HIT 3152/3574 (common prefix 3158: slab-end checkpoint, the old path); reply kept: 3653 (+79)  stop
g1  t4 HIT 3653/4065 (= 3574 + 78 + 1);  reply kept: 4161 (+96)   length
g1  t5 HIT 4161/4783 (= 4065 + 96);      reply kept: 4879 (+96)   length
g1  t6 HIT 4879/5576 (= 4783 + 96);      reply kept: 5625 (+49)   stop
g1b t1 HIT 2484 (system checkpoint of g1); reply kept 2835 (+53) … t2 2835 = 2782+52+1, t3 3229 = 3133+96,
    t4 3625 = 3537+87+1, t5 4136 = 4040+96, t6 4656 = 4560+96 — reply kept after all 6 turns
```

- **Next turn's `cached_tokens`:**

  | how the previous turn ended | `cached_tokens` |
  |---|---|
  | `stop` | previous prompt + completion + 1 (the EOS the step-ahead fed) |
  | `length` | previous prompt + completion |

  On a `length` row the step-ahead fed every generated token, so the log says "+96 generated". The
  design table predicted one fewer.
- **Prefill size:** each such turn prefilled only the new user message plus the template glue
  (298–697 tokens).
- **TTFT:** about 0.7–1.4 s, against 6.5–12.7 s cold.

### Greedy texts vs cold (`raw/reply_kv_g1_compare.txt`)

- **g1: 6/6 identical.**
- **g1b: 5/6.** Turn 3 differs from character 181: "…privileges **on macOS**" (hit) vs "**on this machine**" (cold).

### Teacher-forced comparison (`raw/reply_kv_tf.py`, in process)

The script does what agree.py does, with the worker's `Engine.load`: fp16, pfqmm, GDN fuse.

- **Hit chain:** the script rebuilds what the REPLY=1 B=1 worker computed:
  - one cache carried across the turns;
  - each turn resumed at the hit run's own `cached_tokens`. That is the reply end, or for g1 t3 a
    slab-end checkpoint of the chain;
  - the rest prefilled in the worker's slabs (`_slab` over `_bounds`), then B=1 decode as
    `_Scheduler._step`, fed the hit run's generated ids (+ EOS for stop rows).
- **Cold side:** a fresh prefill per turn, then the same teacher-forced decode.
- **Emulation check:** the chain's argmax equals the hit run's tokens at **862/862** positions, so the
  emulation reproduces the worker.
- **Floor:** the same cold prefill, but also cut at the chain's seams (prefill/decode switch points).
  The reply region is prefilled as slabs, not decoded:
  - **cold ‖ alt** = the effect of slab shapes alone;
  - **alt ‖ hit** = decode vs prefill of the same tokens.

| turn | resume | steps | top-1 | KL mean | KL max | floor cold‖alt | floor alt‖hit |
|---|---|---|---|---|---|---|---|
| g1 t2 | reply end 2843 | 96 | 96/96 | 1.5e-5 | 9.9e-5 | 1.6e-5 | 1.7e-5 |
| g1 t3 | ckpt 3152 (reply 2 not kept) | 79 | 79/79 | 2.7e-5 | 1.6e-4 | 2.2e-5 | 2.2e-5 |
| g1 t4 | reply end 3653 | 96 | 96/96 | 2.2e-4 | 3.8e-3 | 2.8e-4 | 5.0e-5 |
| g1 t5 | reply end 4161 | 96 | 96/96 | 4.8e-5 | 5.0e-4 | 5.9e-5 | 3.8e-5 |
| g1 t6 | reply end 4879 | 49 | 49/49 | 8.5e-5 | 7.7e-4 | 1.0e-4 | 6.2e-5 |
| g1b t2 | reply end 2835 | 96 | 96/96 | 1.6e-5 | 2.0e-4 | 1.6e-5 | 1.9e-5 |
| g1b t3 | reply end 3229 | 88 | **87/88** | 2.4e-5 | 1.1e-4 | 3.5e-5 (87/88) | 3.5e-5 |
| g1b t4 | reply end 3625 | 96 | 96/96 | 6.7e-5 | 2.6e-3 | 1.0e-4 | 2.5e-5 |
| g1b t5 | reply end 4136 | 96 | 96/96 | 3.5e-5 | 2.6e-4 | 3.4e-5 | 4.3e-5 |
| g1b t6 | reply end 4656 | 70 | 70/70 | 4.2e-5 | 4.5e-4 | 2.5e-4 (69/70) | 1.5e-4 (69/70) |
| **all (10 turns)** | | **862** | **861 = 99.88 %** | **5.8e-5** | 3.8e-3 | 8.7e-5 (860/862) | 4.2e-5 |

The turn-1 rows are cold on both sides (KL 0) and left out of the "all" row.

**The flip (g1b t3, step 31):** cold logits tie exactly between " this" and " macOS" (margin 0.0 in
fp16). That is the same divergence as the gateway texts.

**Reading:**

- **Gate: PASS.** Top-1 99.88 % ≥ 99.7 % and KL 5.8e-5 ≤ 1e-4.
- One turn's mean KL is over 1e-4: g1 t4 at 2.2e-4. Its floor shows why: only cutting the cold
  prefill at different slab boundaries gives **2.8e-4** on the same turn.
- Decode vs prefill of the same reply tokens (alt ‖ hit) is 4.2e-5 on average. That is smaller
  than the slab-shape effect.
- So the hit through the reply is within the noise that `_bounds` determinism exists to avoid. It is
  not a new error class.

## 2. B=8 wave and cascade

**Worker:** `QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_REPLY=1 QWEN_MAX_BATCH=16 QWEN_PROMPT_CACHE_GB=6 LOG_LEVEL=DEBUG`.

**2a additionally ran with `QWEN_BATCH_PREFILL_SHARE=1.0`** to get the exact wave the design asks
for:

- prefill always runs while a request waits;
- so the 8 closed-loop requests of a turn are all prefilled before any decode step;
- they then decode together and, with `ignore_eos` and max_tokens 64, end on the same step.

With the default share (0.5) rows join a few steps apart and never end together. Gate 3 covers that
staggered case.

### 2a: 8 conversations, 4 turns each, all in flight

- Distinct ~3K system prompts (2969–3044 tokens).
- 280-token excerpts with long-answer questions; 64 tokens, `ignore_eos`.

**Result:**

- **32/32 ok.**
- 0 `Traceback` / `ERROR` / `batch step failed` / `worker_fault`, 0 foreign Generates.
- **32 reply kept**, 0 not kept.
- **24/24 next turns resume at the entry end** (cached = prompt + 64).
- **4 passes of exactly 8 `reply kept` lines.** All 8 rows of each wave were stored in one
  `_drop_done` pass, which spans 19 ms of log time for 8 rows. The pass is interleaved with the
  handler threads' finish lines.

  ```
  04:41:52,255 rk-g2a-4-4 reply kept: 4651 tokens (+64 generated), 5 checkpoints, 461.5 MB (8 entries, 3621.7 MB total)
  04:41:52,258 rk-g2a-1-4 reply kept: 4664 tokens (+64 generated) … 3623.8 MB total
  … (8 lines) …
  04:41:52,274 rk-g2a-5-4 reply kept: 4651 tokens (+64 generated), 5 checkpoints, 461.5 MB (8 entries, 3636.6 MB total)
  ```

- **Health after:** entries 8, bytes **3,636,559,872** (= the last line's 3636.6 MB total), hits 24 +
  misses 8 = 32. No evictions (3.6 of 6 GB).

### 2b: 16 conversations sharing one 4976-token system prompt, 2 turns, all in flight

Default share.

**Result:**

- **32/32 ok**, 0 errors, 0 foreign.
- **Cascade group formed:** `group formed: P=4981, 8 rows`. Rows then joined up to 12. Batch ticks
  show `groups=1 grouped=8..9` through both turns.
- **27 rows ended inside the group.** Their reply is skipped (`_reply_cache` returns None) and
  **nothing is logged** (F1). The next turn of these conversations resumed at the prompt entry's
  slab-end checkpoint: cached = prompt − 7, the assistant header, the old path.
- **At the end of turn 2 the group released its last members by memory**, then dissolved:

  ```
  04:43:58,517 cascade: group P=4981: …-5-2 left (memory), 4 rows stay
  04:43:58,566 cascade: group P=4981: …-14-2 left (memory), 3 rows stay
  04:43:58,625 cascade: group P=4981 dissolved (memory): 3 rows released, 0.89 GiB charge dropped
  04:43:58,865 …-14-2 prompt cache reply kept: 5771 tokens (+64 generated), 5 checkpoints, 509.0 MB (16 entries, 3184.4 MB total)
  … 10-2, 6-2, 13-2 …
  04:44:00,062 …-5-2  prompt cache reply kept: 5765 tokens (+64 generated) … (16 entries, 3192.9 MB total)
  ```

  Those 5 ungrouped rows stored their replies.
- **Health after:** entries 16, bytes **3,192,881,152** (= 3192.9 MB), hits 31 + misses 1.
- **Probe:** a third turn for the 5 kept conversations and 2 grouped ones (`raw/reply_kv_g2b_t3.json`).

  | rows | cached | prefill |
  |---|---|---|
  | the 5 released rows | = prompt + 64, the reply end | 24 tokens |
  | the 2 grouped rows | 5720 / 5688, the slab-end checkpoint | 95 tokens |

  A reply decoded partly inside a group and then released is stored and resumed correctly.

**Gate 2: PASS** for every functional item. The "grouped rows skipped **and logged**" item is not met
(F1).

## 3. Step cost: 8 conversations × 6 turns, closed loop, fresh worker per run

**Setup:**

- Standard env + `QWEN_PROMPT_CACHE_REPLY` 0/1; max_batch 16, default share.
- The g8 corpus; 64-token `ignore_eos` answers (`raw/reply_kv_g3.sh`).
- `off1–4` are four extra REPLY=0 runs. A shell word-splitting slip left the knob empty, and an empty
  value means off. They were kept as a drift record.
- `a0 a1 b0 b1` are the requested alternation.

| run | REPLY | wall s | agg tok/s | decode tok/s per req (t1 / t≥2) | worker ttft ms t1 / t≥2 (p50) | prefilled tok/req t≥2 | reply kept |
|---|---|---|---|---|---|---|---|
| off1 | 0 | 108.6 | 28.3 | 8.0 (4.5 / 8.7) | 9137 / 2154 (2262) | 424 | – |
| off2 | 0 | 109.6 | 28.0 | 8.3 (4.5 / 9.0) | 9430 / 2150 (2216) | 424 | – |
| off3 | 0 | 116.1 | 26.5 | 7.8 (4.3 / 8.5) | 10228 / 2208 (2312) | 424 | – |
| off4 | 0 | 120.1 | 25.6 | 7.4 (4.2 / 8.0) | 10374 / 2314 (2354) | 424 | – |
| a0 | 0 | 124.5 | 24.7 | 7.3 (4.1 / 7.9) | 10950 / 2371 (2440) | 424 | – |
| **a1** | **1** | 129.7 | 23.7 | 7.1 (3.9 / 7.8) | 12361 / **2155** (2320) | **353** | 48/48 |
| b0 | 0 | 139.8 | 22.0 | 6.7 (3.6 / 7.4) | 12908 / 2341 (2619) | 424 | – |
| **b1** | **1** | 126.1 | 24.4 | 7.4 (4.0 / 8.1) | 11589 / **2111** (2274) | **355** | 48/48 |

**Drift:** the off runs alone went 108.6 → 139.8 s (+29 %) over 25 minutes. That is the GPU's known
drift, and it is larger than any effect here.

**Pairs against their neighbours:**

- **Decode tok/s per request:**
  - a1 7.1 against a0 / b0 7.3 / 6.7;
  - b1 7.4 against b0 6.7.

  So the queued copies cost **nothing visible**. b1's wall time (126 s) is shorter than both
  bracketing off runs.
- **Prefill:** with REPLY=1 every turn ≥ 2 prefills 353–355 instead of 424 tokens (**−17 %**).
- **Worker ttft:** the turn ≥ 2 mean 2155 / 2111 ms is the lowest of all 8 runs (−9 % against
  a0 / b0). Turn 1 is a cold 3K prefill in every run and follows the drift.

**Store accounting:**

- 48/48 reply kept in both REPLY=1 runs, 0 errors, 0 foreign.
- a1: 40/40 next turns resume at the entry end. b1: 39/40.

**The one miss (b1 c5 t4 → t5)** is the documented trailing-whitespace case:

- the 64-token `length` reply ends in "…B=1, 2, and " (a trailing space);
- the template `|trim`s it, so the next prompt diverges at the reply's last token;
- the HIT line shows `common prefix 4650` of the entry's 4651. The turn falls back to the slab-end
  checkpoint (`reused=4580 gap=70`) exactly as designed;
- the next store absorbs the stale reply entry: 8 entries stay.

**Gate 3: PASS.**

## 4. AppWorld replay with own replies (the payoff)

**Setup:**

- `scripts/perf/agentic/replay.py --own-replies --ignore-eos off --episodes 8 --duration-min 10`
  (`raw/reply_kv_replay.sh`).
- A fresh worker per run under `sched_timer.py`, plus memsampler.
- Standard env (`QWEN_PROMPT_CACHE=1 _GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0`,
  max_batch 16, cascade on), REPLY 0 → 1 → 0.
- 9 tasks started in each run.

`raw/reply_kv_replay_compare.md` (`compare.py`) + `raw/reply_kv_replay_pairs.txt`:

| | r0 (REPLY=0) | **r1 (REPLY=1)** | r0b (REPLY=0) |
|---|---|---|---|
| requests/min (10-min window) | 19.3 | **20.6** | 19.4 |
| latency p50 / p90 s | 19.7 / 47.1 | 17.0 / 47.1 | 21.3 / 45.3 |
| queue mean / p50 / p90 s | 7.6 / 2.6 / 23.1 | 7.0 / 1.1 / 24.2 | 7.0 / 3.3 / 15.8 |
| ttft mean / p50 / p90 s | 4.0 / 2.4 / 8.1 | 3.9 / **1.5** / 9.0 | 4.1 / 2.5 / 8.4 |
| decode mean / p90 s | 13.5 / 26.2 | 12.4 / 22.6 | 13.9 / 24.5 |
| cache hit rate / misses | 95.0 % / 10 | 94.9 % / 11 | 95.0 % / 10 |
| continuing calls resumed after the reply | 0 / 192 | **203 / 205 (99.0 %)** (2 misses) | 0 / 192 |
| prefilled tok per continuing call, mean / median | 732 / 451 | 700 / **232** | 727 / 451 |
| prefilled tok per call, **matched calls that hit in both runs** | 662 (vs r1, 190 calls) | 522 / 515 → **−21.2 % / −21.7 %** | 658 (vs r1, 191 calls); r0 vs r0b: **±0.0 %** (187 calls) |
| reply kept / not kept / entry gone / reasoning | – | **212 / 0 / 2 / 0** | – |
| scheduler prefill / decode / idle % | 79 / 16 / 5 | 80 / 16 / 4 | 79 / 16 / 5 |
| ms per slab / per decode step | 1200 / 50 | 1160 / 52 | 1218 / 51 |
| store MB per entry (prefill store) | 615 | 608 (reply entries 607) | 615 |
| pcache / active GiB max, peak GB | 6.0 / 19.0, 22.3 | 6.0 / 18.7, 22.3 | 6.0 / 18.4, 22.3 |
| errors / tracebacks / failed steps | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

**Reading:**

- **Prefill savings: −21 % per call on matched hits.** This is the cleanest number: the same
  (task, call) in both runs, both a cache hit. It matches phase 0's −20.6 % for these tasks. The
  off-vs-off control is 0.0 %.
  - Unmatched means are dominated by the ~10 task-start misses (10–22K tokens each), so they move
    less: 732 → 700.
  - The median per continuing call halves, 451 → 232.
  - In r0 the reply region (previous prompt + reply − resume point) was 141 tokens a call, 21.4 % of
    what hits prefilled.
- **Requests/min: +6.5 %** (19.3 / 20.6 / 19.4), below the design's +10–15 %:
  - prefill is still 80 % of the scheduler's time;
  - with fewer tokens per call the slabs got shorter (1160 vs 1200–1218 ms) but not fewer.
    Per-call prefill is dominated by the new tool output, which is not affected;
  - ttft p50 1.5 vs 2.4–2.5 s, queue p50 1.1 vs 2.6–3.3 s.
- **Memory:** entries are the same size, 607–608 MB vs 615 MB. The end state moves instead of being
  added, as designed. Peak, pcache and swap are the same in all three runs.
- **Store counts:**
  - 212 reply kept, 0 "reply not kept: over the headroom", 0 reasoning skips.
  - 2 "its prompt entry is gone", at 05:27:31 and 05:28:31. The soft cap is full (pcache 6.0 GiB);
    the LRU store eviction took that row's prompt entry while it decoded, and the next call of
    those two episodes was a miss.
  - No cascade group formed in any replay run: no 8 rows shared ≥ 4096 tokens.
- **Numerics:** REPLY=1 changes the trajectories somewhat. With own replies, 151/192 matched calls
  had the same prompt length in r0 vs r1, against 187/188 in r0 vs r0b. 16 completions differ from
  the recorded length (r0: 3, r0b: 0). That is the documented numerics class; it changes the replies,
  not the correctness.

**Gate 4: PASS.**

## Findings

- **F1 (LOW, observability): the grouped-row skip is silent.**
  - `_reply_cache` returns None for `row.group is not None`, and `_store_reply(row, None)` returns
    without a line.
  - In 2b that was 27 of 32 rows, invisible in the log and the `worker_log.py` events.
  - Suggested: an INFO `id=… prompt cache reply not kept: in a cascade group` (or a count per pass),
    and a `reply_grouped` event in `worker_log.py`.
- **F2 (LOW, lost savings): the reasoning filter is broader than needed with thinking off.**
  - g1 t2's answer mentioned the literal text `` `<think>` ``. The model emitted the special
    `<think>` id inside its content, so the reply was dropped (`reasoning in the reply (thinking on)`).
  - Yet the next prompt re-tokenized exactly to the generated ids (checked in `reply_kv_tf.py`), so
    it would have resumed.
  - Suggested: skip only when generation started inside reasoning, or when the reply contains
    `</think>` (the template splits on that). Rare on AppWorld: 0 of 214.
- **F3 (doc):** for a `length` row the step-ahead feeds the last generated token too. The entry is
  prompt + completion, and the log says "+N generated" = completion. The design table says the
  pending token is never fed. Behaviour is correct (the next turn matches); the table should say
  "every generated token".
- **Expected fallbacks seen and handled:**
  - a trailing-whitespace `length` reply (gate 3, 1 of 80);
  - an entry evicted while its row decoded (gate 4, 2 of 214).

## Recommendation

**Turn `QWEN_PROMPT_CACHE_REPLY` on by default** for text workloads with the prompt cache, after F1
gets its log line. The numbers:

| | |
|---|---|
| prefilled tokens per agentic call | −21 % |
| AppWorld req/min | +6.5 % |
| ttft p50 | −40 % |
| decode cost | none visible |
| extra memory | none |
| numerics | inside the slab-shape floor (top-1 99.88 %, KL 5.8e-5) |

Keep `0` documented as the bit-exact `_bounds`-determinism mode. Optionally fix F2 and correct the
design table (F3).

**State left:** the standard worker on :50051, pid 35865, started 05:42:39 from HEAD 43e976c:

```
QWEN_MODEL=/Users/user/Projects/llm/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1
QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 QWEN_PROMPT_CACHE_REPLY=0
```

- REPLY is left off because gate 2's "logged" sub-item was not met.
- Log: `/tmp/scratch/worker_standard.log`.
- The :8090 gateway is untouched. The GPU lock is released.

## Raw (`raw/reply_kv_*`)

| file | contents |
|---|---|
| `build_corpus.py` → `corpus.json` | the prompt corpus |
| `drive.py` | driver: chat / replay / convs / compare |
| `worker.sh`, `g3.sh`, `replay.sh` | launchers |
| `check.py` | log checker |
| `g3_summary.py` (→ `g3_summary.md`) | gate 3 table |
| `replay_pairs.py` (→ `replay_pairs.txt`) | gate 4 per-call resume analysis |
| `tf.py` (→ `g1_tf.json` / `g1_tf.txt`) | the teacher-forced comparison |
| `health.py` | gRPC Health read |

Results and logs:

| gate | files |
|---|---|
| 1 | `g1{,b}_{hit,cold}.json`, `g1_compare.txt`, `w_g1_{hit,cold}.log` |
| 2 | `g2a.json`, `g2b.json`, `g2b_t3.json`, `w_g2{a,b}.log`, `g2{a,b}_health.txt`, `g2{a,b}_check.txt` |
| 3 | `g3_{off1..4,a0,a1,b0,b1}.json`, `w_g3_*.log`, `g3_{a1,b1}_check.txt` |
| 4 | `replay_{r0,r1,r0b}.{json,txt}`, `w_replay_*.log`, `replay_mem_*.jsonl`, `replay_compare.md` |

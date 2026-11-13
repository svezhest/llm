# GPU gate: chain-aware checkpoints (A) + eviction order (B), 2026-10-16

Gate for the uncommitted prompt-cache changes of 2026-10-16 (A: chain-aware checkpoint
policy for absorbed entries, `QWEN_PROMPT_CACHE_CHAIN_MARKS` 3 / `_CHAIN_TAIL` 4096,
first-user-message strides kept, `states=` in the store log; B: `_PrefixCache.victim`
order — waiting requests' entries last, stale LRU first, else MRU — `_wanted`/`_peek`
memo, give-up path, widened store rollback). Invariant: **hit == cold, bitwise**, no
errors under eviction pressure.

**Result: PASS.** 10/10 + 24/24 (+1 idle probe) byte-identical answers against cold, zero
tracebacks / `batch step failed` / `worker_fault` / request errors, evictions counted and
logged. One performance observation on the stale rule (outputs unaffected), below.

## Environment

- M1 Max 32 GB (MacBookPro18,2), macOS 27.0, `iogpu.wired_limit_mb=28672`; mlx 0.32.3,
  mlx-vlm 0.7.6. Nothing else on the GPU (lock held); worker at nice 0 (`NO_BG_NICE`).
- Code: HEAD 8a1625c + uncommitted `src/worker/worker.py` (md5 `63dfea4df52b9a4ae060a18f2801c3ad`,
  +237/−53 vs HEAD) — the final code, not the 03:22 snapshot.
- Model `ornith15-9b-mlx-8bit` (absolute path), every worker: `QWEN_KV_BITS=0
  QWEN_PRESENCE_PENALTY=0 QWEN_BATCH_RESERVE_TOKENS=1024`, plus the per-run env below.
- Our gateway on :8090 (pid 55417, reused; `GATEWAY_THINKING=off`, so `enable_thinking:false`
  in every request). Requests: non-streaming, temperature 0, presence_penalty 0,
  max_tokens 80. Compare = `content` + `completion_tokens` + `finish_reason` (+ the request's
  messages, to be sure the same prompt was sent).
- Driver (stdlib only) and the prompt corpus: `raw/chain_evict_gpu_drive.py`,
  `raw/chain_evict_gpu_build_corpus.py` → `raw/chain_evict_gpu_corpus.json` (text slices of
  repo files, token counts from the model's tokenizer). Every request body and answer is in
  `raw/chain_evict_gpu_*.json`; worker logs `raw/chain_evict_gpu_w_*.log`.

| run | worker env | script |
|---|---|---|
| pressure | `QWEN_PROMPT_CACHE=1 _GB=1.5 QWEN_MAX_BATCH=8 LOG_LEVEL=DEBUG` | 6 convs × 4 turns, round-robin, 4 in flight |
| cold | `QWEN_PROMPT_CACHE=0 QWEN_MAX_BATCH=1` | conversation script; pressure requests replayed sequentially |
| hit | `QWEN_PROMPT_CACHE=1 _GB=6 QWEN_MAX_BATCH=1 LOG_LEVEL=DEBUG` | conversation script |
| replay-hit | `QWEN_PROMPT_CACHE=1 _GB=1.5 QWEN_MAX_BATCH=1 LOG_LEVEL=DEBUG` | pressure requests replayed sequentially, then a 320 s idle probe |
| hit, chain off (extra A/B) | as hit + `QWEN_PROMPT_CACHE_CHAIN_MARKS=-1` | conversation script |

## 1–2. Conversation script: cold vs hit (cache 6 GB, B=1)

System 2500 tokens (README slice) + first user message = a 6004-token document
(`cascade.py`) + question; 6 follow-ups (previous answer + ~290–420-token backlog excerpt +
question); then `regen7` (turn 7's messages again — the latest turn), `regen4` (turn 4's
messages, as asked), `sibling` (system + document, different first question).

| request | prompt | cached (hit) | cold = hit |
|---|---|---|---|
| turn1 | 8543 | 0 | SAME |
| turn2 | 8930 | 8536 | SAME |
| turn3 | 9317 | 8923 | SAME |
| turn4 | 9735 | 9310 | SAME |
| turn5 | 10238 | 9728 | SAME |
| turn6 | 10741 | 10231 | SAME |
| turn7 | 11253 | 10734 | SAME (20 tokens, `stop`) |
| regen7 | 11253 | 11252 | SAME |
| regen4 | 9735 | 6601 | SAME |
| sibling | 8540 | 6601 | SAME |

**10/10 byte-identical.** Health after: hits 9, misses 1, 3 entries, 1.15 GB. Turn 2–7
TTFT ~3.6–4 s vs 18–24 s cold; regen7 0.7 s.

Store lines (hit run; chain policy on):

```
turn1  store: 8542 tokens, 5 checkpoints, 589.0 MB (1 entries) reused=0     gap=0    absorbed=0 pieces=1 states=309.1 MB
turn2  store: 8929 tokens, 5 checkpoints, 601.8 MB (1 entries) reused=8536  gap=6    absorbed=1 pieces=2 states=309.1 MB
turn3  store: 9316 tokens, 5 checkpoints, 614.7 MB (1 entries) reused=8923  gap=6    absorbed=1 pieces=3 states=309.1 MB
turn4  store: 9734 tokens, 5 checkpoints, 628.6 MB (1 entries) reused=9310  gap=6    absorbed=1 pieces=4 states=309.1 MB
turn5  store: 10237 tokens, 5 checkpoints, 645.3 MB (1 entries) reused=9728 gap=6    absorbed=1 pieces=5 states=309.1 MB
turn6  store: 10740 tokens, 5 checkpoints, 662.0 MB (1 entries) reused=10231 gap=6   absorbed=1 pieces=6 states=309.1 MB
turn7  store: 11252 tokens, 6 checkpoints, 730.5 MB (1 entries) reused=10734 gap=6   absorbed=1 pieces=7 states=360.6 MB
regen7 store: 11252 tokens, 6 checkpoints, 730.5 MB (1 entries) reused=11252 gap=0   absorbed=1 pieces=7 states=360.6 MB
regen4 store: 9734 tokens, 5 checkpoints, 691.6 MB (2 entries, 987.7 MB) reused=6601 gap=3133 absorbed=0 pieces=2 states=309.1 MB
sibling store: 8539 tokens, 4 checkpoints, 601.0 MB (3 entries, 1154.2 MB) reused=6601 gap=1917 absorbed=0 pieces=2 states=257.6 MB
```

Same script with the chain policy **off** (`CHAIN_MARKS=-1`, also 10/10 identical):

```
turn1  8542 tokens,  5 checkpoints, 589.0 MB states=309.1 MB
turn2  8929 tokens,  6 checkpoints, 653.4 MB states=360.6 MB
...    (+1 checkpoint, +51.5 MB per turn)
turn7  11252 tokens, 11 checkpoints, 988.0 MB states=618.1 MB
regen4 reused=9728 gap=6     sibling reused=6601 gap=1917
```

Reading:
- `absorbed=1` on every chain turn; `cached_tokens` > 0 from turn 2 on. ✔
- States: with the policy on the entry holds 6 states (5 checkpoints + end state, 309 MB)
  from turn 1 through 6 and 7 at turn 7 — flat — vs one more per turn with it off (12 states,
  618 MB at turn 7). Entry at turn 7: **730.5 vs 988.0 MB (−257 MB, −26 %)**. ✔ (The
  turn-1 entry already had only 5 checkpoints: system end at 2505, the document strides
  4553/6601, the message starts; so "dropping to ~5" shows as "never growing past 5–6".)
- `regen7` (regenerate of the latest turn) resumes at the entry's end: reused 11252 =
  prompt − 1. ✔
- `sibling` resumes at a document stride, 6601 = 2505 (system end) + 2 × 2048, inside the
  first user message, > system length. ✔
- `regen4` resumes at the same document stride 6601 (gap 3133, TTFT 10.1 s), not at turn 4's
  end: turn 4's boundary is 6 message starts back from the turn-7 entry's end, past
  `CHAIN_MARKS`=3, and a chain has no stride checkpoints in its tail (each turn's prefill is
  one sub-2048 slab ending at message starts). This is the policy's stated trade-off
  ("older turns' boundaries … dead weight"); with the policy off the same request resumes at
  9728. Bits are identical either way. Not a failure, but the brief's expectation "regenerate
  resumes at the entry's end" holds only for regenerating the latest turn (`regen7`).

## 3. Eviction pressure

6 independent conversations (system prompts 2975–3073 tokens from distinct files, turn
messages 321–416 tokens), 4 turns each, round-robin (fewest turns done first), 4 in flight,
B ≤ 8 worker, cache soft cap 1.5 GB (~3–4 entries of 315–467 MB for 6 conversations).

- **24/24 requests ok**, prompt 3318–4812 tokens. Batch occupancy rows 1–3 (+ prefill),
  `waiting=` up to 3 at start. Peak 13.42 GB.
- Health after: entries 3, 1.38 GB, **hits 11, misses 13, evictions 10** (= 10 DEBUG
  eviction lines; hits + misses = 24 requests).
- Log: **0** `Traceback` / `ERROR` / `batch step failed` / `worker_fault`; no "not kept",
  no "gives up", no admission/growth evictions (the budget was never tight: 15.75 GiB).
- Answers vs cold B=1 on the same prompts: 22/24 identical (c0t1, c0t3 differ: B ≥ 3 mrqmm
  numerics class, as expected; not gated).

**Bit-exact gate — the same 24 requests (the pressure run's exact messages, in start order)
sequentially at B=1, cache 1.5 GB vs cold B=1: 24/24 byte-identical**, with 8 hits
(cached 3328–4273) and 13 evictions in between.

The 1.5 GB cache under round-robin: hits by turn (t1/t2/t3 of 6 each): B ≤ 8 run 4/4/3,
B=1 replay 3/4/1.

### Eviction log excerpts

B ≤ 8 run (all victims "live" → MRU; the entry just stored before is dropped, which in a
round-robin keeps the older entries):

```
05:09:08 evicting an entry of 3323 tokens (live, 2 ticks old, 7 s idle)
05:09:16 evicting an entry of 3349 tokens (live, 2 ticks old, 8 s idle)
05:09:31 evicting an entry of 3735 tokens (live, 2 ticks old, 10 s idle)
...
05:10:24 evicting an entry of 4710 tokens (live, 2 ticks old, 12 s idle)
05:10:24 evicting an entry of 4169 tokens (live, 4 ticks old, 14 s idle)
```

(At 05:10:24, c3t3 was prefilling from c3's entry: it was spared although it was the MRU
one — the running prefill's parent is kept.)

B=1 replay (mix of tick-stale LRU and live MRU victims):

```
05:24:13 evicting an entry of 3317 tokens (live, 2 ticks old, 8 s idle)
05:24:37 evicting an entry of 3323 tokens (stale, 16 ticks old, 57 s idle)
05:24:46 evicting an entry of 3735 tokens (live, 2 ticks old, 9 s idle)
05:25:46 evicting an entry of 4710 tokens (live, 2 ticks old, 11 s idle)
05:25:46 evicting an entry of 4199 tokens (stale, 10 ticks old, 31 s idle)   <- c3's; c3t3 is the next request: miss
05:26:11 evicting an entry of 4301 tokens (stale, 10 ticks old, 43 s idle)   <- c5's; c5t3 is the next request: miss
05:26:24 evicting an entry of 4169 tokens (stale, 10 ticks old, 53 s idle)   <- c0's; c0t3 is the next request: miss
05:26:38 evicting an entry of 4811 tokens (live, 2 ticks old, 13 s idle)
```

Idle probe (after the replay, 320 s with no traffic, then turn 7 of the chat script,
11253 tokens, a miss whose 781 MB entry needs two evictions):

```
05:33:05 evicting an entry of 4654 tokens (stale, 8 ticks old, 426 s idle)
05:33:05 evicting an entry of 4646 tokens (stale, 6 ticks old, 414 s idle)
05:33:05 store: 11252 tokens, 7 checkpoints, 780.8 MB (2 entries, 1241.1 MB total) reused=0 ...
```

Answer identical to cold turn 7. Both victims were idle > 300 s and LRU-ordered; both were
also over the tick threshold (8 > 2·3; 6 > 2·2 once the first was gone), so this did not
isolate the idle-only branch on the GPU (it is covered by the CPU tests).

### Observation (performance only): the stale rule vs > capacity round-robin

In the B=1 replay all three turn-3 misses (c3t3, c5t3, c0t3) came right after their own
conversation's entry was evicted as "stale, 10 ticks old" by the store of the request just before it
was needed. With 6 conversations cycling over a cache that holds 3–4 entries, every entry is
10–12 ticks old when its conversation comes back — over the `2 × len(entries)` = 6–8
threshold — so every waiting conversation looks stale and the order degenerates to LRU,
the round-robin worst case the MRU rule is there to avoid. The threshold counts entries,
not active conversations. Hit rate in this replay: 8/24 (B ≤ 8 run, where in-flight
requests touched entries more often: 11/24). Outputs are unaffected (24/24); flagged for
the eviction design, not a gate failure.

## 4. Health counters

Read with a gRPC `Health` call from the worker venv (`raw/chain_evict_gpu_health.py`; the
gateway's `/metrics` does not export the prompt-cache counters, it logs them only at debug):

| after | entries | bytes | hits | misses | evictions |
|---|---|---|---|---|---|
| pressure (B ≤ 8) | 3 | 1,381,859,328 | 11 | 13 | 10 |
| hit (chat script) | 3 | 1,154,187,264 | 9 | 1 | 0 |
| replay-hit (B=1) | 3 | 1,383,202,816 | 8 | 16 | 13 |
| + idle probe | 2 | 1,241,120,768 | 8 | 17 | 15 |

Consistent with the logs: hits + misses = requests; evictions = DEBUG eviction lines (10;
13 + 2); bytes = the last store line's "MB total". Gateway `qwen_requests_total{status="ok"}`
only grew; no non-ok status.

## Gates

| gate | result |
|---|---|
| hit == cold, conversation script (cache 6 GB, B=1) | **PASS** 10/10 |
| `cached_tokens` > 0 from turn 2; `absorbed=1` on chain turns | **PASS** |
| `states=` bounded on a chain | **PASS** 6–7 states flat (309–361 MB) vs 6→12 (618 MB) chain off |
| regenerate of the latest turn resumes at the entry end | **PASS** reused 11252 = prompt − 1 |
| regenerate of turn 4 | resumes at document stride 6601 (by design of A; chain off: 9728) — bits PASS |
| sibling resumes at a document stride > system | **PASS** 6601 > 2505 |
| hit == cold under eviction (B=1 replay, cache 1.5 GB) | **PASS** 24/24 (+ idle probe 1/1) |
| no tracebacks / `batch step failed` / `worker_fault`, no request errors | **PASS** (all 5 worker logs, 103 requests) |
| evictions > 0, victims logged sensibly | **PASS** (10 at B ≤ 8, 15 at B=1); stale rule degrades to LRU in > capacity round-robin (perf note) |
| Health counters consistent | **PASS** |

## Raw

`raw/chain_evict_gpu_*`: driver + corpus builder + worker launcher + idle probe; corpus;
results (`chat_cold`, `chat_hit`, `chat_hit_chainoff`, `pressure`, `replay_cold`,
`replay_hit`, `idle` — each with full request messages and answers); worker logs
(`w_pressure`, `w_cold`, `w_hit`, `w_replay_hit` incl. the idle probe, `w_hit_chainoff`).
Reproduce: `drive.py chat|pressure|replay|compare` against a gateway on :8090.

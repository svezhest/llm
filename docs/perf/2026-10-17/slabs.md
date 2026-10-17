# Slab cost on the agentic load: is "fewer slabs per request" a lever? 2026-10-17

**Question.** In the AppWorld replay the scheduler spends ~80 % of its time in prefill, at ~2.3 slabs per
request. Message-boundary cuts split even short hit suffixes. How much would merging the unneeded cuts buy?

**Answer: little throughput, some latency.**

- **The cut pattern.** A third of all slabs (32 %) are 1–8 tokens long:
  - the generation-prompt tail (242 of them in 20 min);
  - the 2 tokens between a reply-KV resume point and the next message start (94).

  Each costs a full pass over the weights: ~75–100 ms, against ~10–15 ms if merged.
- **The cost curve.** Above 64 rows a slab costs a whole number of 64-row pfqmm tiles: 141 ms per tile at a 10K
  offset, 158 at 24K. There is no other fixed per-slab cost, so cutting big slabs differently is nearly free.
- **The gain.** Merging every slab shorter than 16 tokens into its neighbour ("min") removes 32 % of the slabs.
  - It saves an estimated **~2.1–2.4 % of the GPU's busy time**, which comes to **~+2 % requests/min**.
  - Each removed boundary also takes a decode-share window out of the request's TTFT: **−0.3…0.7 s** for ~60 % of
    hit requests.
  - The bigger "keep only the chain-policy cuts" change would add ~1 % more. It would break the
    resume-from-checkpoint invariant, so it is not worth it.

Code: HEAD aa2cb9e, the worker modules frozen from HEAD (the uncommitted `pfqmm.py` short-tile change in the working
tree was excluded). 9B 8-bit, fp16 KV, the standard env (`replay_batch1.md`).

## (a) Slab histogram on the replay traces

**Source.**

- `scripts/perf/agentic/sched_timer.py` now logs one `sched_slab:` line per prefill tick. A tick is one
  `_advance` = one slab: the first one also does the lookup and restore, the last one the store and `_join`.
- Each line carries: position, tokens, cut type (message start / step / end), time, rows, sub-phase times, and
  on the first slab the whole slab plan past the resume point plus the chain policy's message starts.

**Data.** Instead of a separate 5-min run I used the two 10-min B=16 replays of S1, b16a + b16b: 447 requests,
1055 slabs, 20 min.

**Table.** `scripts/perf/agentic/slab_hist.py` → `raw/slabs_hist.txt`.

**Slabs per request:**

| slabs | hits | misses |
|---|---|---|
| 1 | 97 | – |
| 2 | 310 | – |
| 3 | 14 | – |
| 5 | 2 | – |
| 7 | – | 18 (task starts, ~3.5K tokens) |
| 15–37 | – | 6 (capacity misses, 15–28K tokens) |
| **mean** | **1.82** | **11.9** (all: 2.36) |

**Tokens per slab** (t = the whole `_advance`, which includes the decode step queued ahead of it and the thermal
state of a sustained run):

| tokens | slabs | % of slabs | in hits | ends: msg / step / prompt end | median t ms | ms/token | % of slab time |
|---|---|---|---|---|---|---|---|
| 1–8 | **336** | **32** | 333 | 94 / 0 / 242 | 114 | 21.6 | 3.4 |
| 9–32 | 19 | 2 | 18 | 1 / 0 / 18 | 138 | 6.1 | 0.3 |
| 33–128 | 99 | 9 | 80 | 16 / 0 / 83 | 285 | 3.6 | 2.9 |
| 129–255 | 168 | 16 | 168 | 69 / 0 / 99 | 595 | 3.0 | 9.9 |
| 256–511 | 187 | 18 | 29 | 182 / 0 / 5 | 857 | 2.6 | 15.9 |
| 512–1023 | 144 | 14 | 84 | 144 / 0 / 0 | 2317 | 2.7 | 30.9 |
| 1024–2047 | 87 | 8 | 49 | 82 / 5 / 0 | 3489 | 2.4 | 29.1 |
| 2048 | 15 | 1 | 8 | 0 / 15 / 0 | 5441 | 2.6 | 7.7 |

**Hit shapes** ("t" = ≤ 16 tokens, "s" = ≤ 128):

| shape | count |
|---|---|
| `800 + t` | 67 |
| `s` | 52 |
| `200 + t` | 47 |
| `200` | 43 |
| `t + 300` | 36 |
| `1500 + t` | 28 |
| `t + s` | 27 |
| … | |

**How the cuts fall.** `_bounds` cuts at every `<|im_start|>` ≥ 256 tokens past the previous cut. With the
reply-KV resume at the reply end:

- **The tail.** The final `<|im_start|>assistant` (generation prompt, `<think>\n\n</think>\n\n` follows) is a cut
  whenever the reply + tool output since the previous cut reach 256 tokens. The prompt end is 6 tokens later, so a
  6-token last slab follows.
  - That happens in **239 of 423 hits (57 %)**.
  - Median 6 tokens, 121 ms in the run: the slab + store + join.
- **The head.** When the previous cut lies ≥ 256 tokens before the resume point, the tool message's
  `<|im_start|>` 2 tokens past the reply end is a cut too, and a 2-token first slab precedes the tool output.
  - That happens in **94 of 423 hits (22 %)**.
- **Misses (task starts).** Every few-shot message starts a slab: 7 slabs for 3.5K tokens. The capacity misses take
  15–37 slabs. All these slabs are ≥ 256 tokens.

**Per-phase times inside slabs** (`sched_slab:` medians):

| phase | ms |
|---|---|
| restore | 12.2 (max 31) |
| lookup | 1.4 |
| store | 7.6 |
| `_join` | 0.2 |

The checkpoint itself is free: `_recurrent_states` keeps references, no copy. A tiny head slab costs 58 ms with
its restore. A tiny tail slab costs 121 ms with its store.

## (b) Cost of one slab of N tokens (isolated, synced)

`scripts/perf/prof/slab_cost.py`:

- The worker's `Engine.load` exactly; its scheduler stays idle.
- Per trial, a fresh cache built like `_PrefixCache.restore`: KV [0, off) + a zeros tail of round_up(N, 256), so
  the slab writes in place.
- The worker's slab body (`lm(…, skip_logits=True)`, quantize, eval of the states), synced, then
  `mx.clear_cache()` timed apart.
- 7 rounds; the (offset, N) order is rotated per round.

Raw: `raw/slabs_cost.{json,txt}`, `raw/slabs_cost_fine.json`. Machine warm (right after the S1 runs).

| N | off 10K: ms (min..max) | ms/token | ms per 64-row tile | off 24K: ms | ms/token | ms per tile |
|---|---|---|---|---|---|---|
| 5 | **80.8** (70..88) | 16.2 | – | **80.9** | 16.2 | – |
| 16 | 99.0 (94..105) | 6.2 | – | 118.2 | 7.4 | – |
| 64 | 162.6 (155..166) | 2.54 | 163 | 178.4 | 2.79 | 178 |
| 128 | 285.8 | 2.23 | 143 | 321.7 | 2.51 | 161 |
| 256 | 555.9 | 2.17 | 139 | 625.9 | 2.44 | 156 |
| 432 | 967.2 | 2.24 | 138 | 1108.5 | 2.57 | 158 |
| 512 | 1132.0 | 2.21 | 142 | 1258.4 | 2.46 | 157 |
| 1024 | 2234.6 | 2.18 | 140 | 2501.4 | 2.44 | 156 |
| 2048 | 4550.2 | 2.22 | 142 | 5056.2 | 2.47 | 158 |

**Fine scan at 10K** (`slabs_cost_fine.json`): the cost moves in **64-row steps**. That is pfqmm's BM = 64 tile at
HEAD.

| N | ms |
|---|---|
| 765 (12 tiles) | 1596 |
| 770 / 800 / 823 / 829 (13 tiles) | 1726 / 1729 / 1748 / 1754 |
| 834 / 840 / 846 / 896 (14 tiles) | 1873 / 1867 / 1868 / 1864 |

**Reading:**

- **Fixed per-slab cost** (weights streamed once):
  - A 5-token slab costs **81 ms** at 10K and 24K alike. That is the MLX quantized matmul at M = 5 (qmv/qmm,
    ~2.6× a 1-row decode pass), not the 30 ms 1-row floor.
  - A 16-token slab costs 99–118 ms. Below 64 rows there is no tile structure.
  - Above 64 rows the least-squares fit has ~0 intercept (−10 ms at 10K, +8 ms at 24K). A slab ≥ 64 rows has no
    fixed cost beyond the padding of its last tile, on average half a tile ≈ 70–80 ms.
- **Marginal cost:**
  - 2.22 ms/token at 10K and 2.46 at 24K, linear over the tile count;
  - attention grows the per-tile cost by +12 % from 10K to 24K;
  - `clear_cache` after a slab: 3–9 ms.
- **Merge cost.** Merging a 6-token tail into the slab before it costs one extra tile with probability 6/64:
  ~13 ms on average, instead of ~81 ms for its own slab (in the run 100–120 ms, warm and with store).

## (c) The same ~840-token suffix as 1, 2 or 3 slabs, with 7 rows decoding

`scripts/perf/prof/slab_boundary.py`. Setup:

- The worker's `Engine` + `_Scheduler` in-process, cache on, standard env.
- **7 background rows** decode distinct ~10K-token AppWorld prompts for the whole run (ignore_eos): decode step
  48–51 ms.
- **Base X:** a recorded AppWorld call, 10243 tokens, cached once.
- **Probes:** X + an assistant reply (~70–90 tokens with its template glue) + a tool output (~740) + the generation prompt; max_tokens 1; all
  distinct; prompt 11066–11086 tokens.
- **Resume:** every probe resumes at 10236, X's last `<|im_start|>`, which is forced to be a cut. Prefill is
  829–847 tokens.
- **Variants:** a wrapper of `worker._bounds`, applied to the probe's prompt only, sets the cuts inside the
  suffix:

  | variant | slabs |
  |---|---|
  | `1` | one slab |
  | `2tail` | the worker's own cut: the 6-token tail |
  | `2msg` | reply \| tool + tail |
  | `3` | reply \| tool \| tail |

- **Rounds:** 8 rounds, plus 8 with a sync before each slab (`/s`), plus 6 rounds with every phase synced and
  timed (`--phases`). The variant order rotates per round.

Raw: `raw/slabs_boundary{,_phases}.{json,txt}`.

**Per variant** (medians, first run, 8 rounds):

| variant | slabs | per slab ms | Σ `_advance` ms | decode steps inside TTFT | TTFT ms |
|---|---|---|---|---|---|
| 1 | 1 | 2607 | 2607 | 0 | 2607 |
| 2tail | 2 | 2378 + 145 | 2529 | 13 (687 ms) | 3218 |
| 2msg | 2 | 364 + 2406 | 2784 | 2 (102 ms) | 2885 |
| 3 | 3 | 352 + 2080 + 139 | 2555 | 13 (704 ms) | 3254 |

**Paired against variant 1 in the same round**, Σ `_advance` adjusted to 840 prefilled tokens at 2.7 ms/token, all
three runs pooled (n = 22 each):

| extra boundary | Δ prefill GPU time ms (mean ± 1 se) | median | Δ TTFT (median) |
|---|---|---|---|
| tail (`2tail`) | −49 ± 29 | −66 | **+636 ms** |
| head: reply \| tool (`2msg`) | **+164 ± 32** | +179 | +298 ms |
| both (`3`) | +17 ± 28 | −28 | **+682 ms** |

The phase-timed run (`--phases`) puts the GPU time in the slab evals themselves:

| piece | eval ms |
|---|---|
| tail slab | 96–114 |
| reply slab (~70 tokens) | 244–286 |
| restore | 10–18 |
| store | 9–24 |
| `_join` | 2–3 |
| the steps queued ahead of a slab | ~0 waits |

**Reading:**

- **The end-to-end GPU cost of a boundary is below the resolution of this test.**
  - The big slab's own time varies ±10 % between rounds (thermal; 1855–2530 ms for the same ~840 tokens in one
    variant).
  - The prefill length varies ±10 tokens across the 64-row tile edge at 832. With 64-row tiles, whether the
    pieces need one tile more or less than the whole is luck of the length.
  - Thus `2msg` = 70 + 770 tokens needs 2 + 13 = 15 tiles vs 14 for 840: +1 tile ≈ +165 ms in the run, as
    measured.
  - `2tail` (834 + 6) often needs no more tiles than 840, and the tail slab's ~100 ms disappeared in the noise
    (−49 ± 29).
- **The cost a merge saves is the tail's own pass.** The mechanism, from (b) and the phase timing, is ~100 ms
  per tiny slab, minus ~13 ms of expected extra tile.
- **The decode steps between slabs are not lost.** They are the other rows' tokens, and the share policy keeps
  their fraction.
- **The latency cost is robust.** Every extra slab ends with `_slab_done`, which gives decode t·(1−s)/s of the
  slab's time before the next one (13 steps ≈ 0.65 s after an 800-token slab with 7 of 16 rows). So the request's
  first token comes **0.3–0.7 s later**.
- **Leftover finding:** the first decode step after a slab is 37–48 ms, against a 48–51 ms steady step. There is
  no pipeline bubble to charge to a boundary.

## Estimate: merging the cuts the policy does not need

`slab_hist.py` re-cuts each logged request's prefill region with the worker's rule from the logged slab plan.

**Check.** The "today" re-cut reproduces the logged slab sizes for 447/447 requests.

**Cost model.** (b)'s isolated curve: 141 / 158 ms per 64-row tile at 10K / 24K, the measured points below 64
rows, linear in the offset. The isolated model gives 931 s for what the run logged as 1041 s; the run is ~12 %
slower (warm, decode waits). Window: 1200 s, ~96 % busy.

| policy | message cuts allowed | slabs (hits / misses per request) | modelled saving | % of busy time |
|---|---|---|---|---|
| today | every `<|im_start|>` ≥ 256 past the previous cut | 1055 (1.82 / 11.9) | – | – |
| tail | today, minus a start < 16 tokens before the prompt end | 813 (1.25 / 11.8) | 17.5 s | 1.5 % |
| head | today, minus a start < 16 tokens past the resume point | 961 (1.60 / 11.9) | 6.7 s | 0.6 % |
| **min** = head + tail | no message-cut slab under 16 tokens | **719 (1.03 / 11.8)** | **24.5 s** | **2.1 %** (in-run scale ~2.4 %) |
| policy | only starts[1] + the last 3 starts (+ 2048 steps) | 908 (1.82 / 5.8) | 10.8 s | 0.9 % |
| policy + min | | 572 (1.03 / 5.7) | 35.2 s | 3.1 % |

The results are the same with a threshold of 32 or 64 instead of 16: the tiny slabs are 1–8 tokens.

**Conversion to requests/min.** The GPU is saturated (idle 4–5 %), and prefill savings converted at ~0.8:1 in
replay_bench. So:

| change | requests/min (from 21.4–21.7) | TTFT |
|---|---|---|
| **min** | **+~2 %**, i.e. ~+0.4 | **−0.3…0.7 s** for ~60 % of hits (one decode-share window fewer each; −0.65 s in (c)) |
| policy + min | ~+3 % | |

- **Why the policy cuts (misses) buy almost nothing.** Their slabs are ≥ 256 tokens, and those cost linearly in
  tiles. Halving the miss slabs (12 → 6 per miss) saves only the half-tile padding of the merged ones.
- **The padding is where the remaining per-slab money is.** Every slab ≥ 64 rows wastes on average half a tile
  (~70 ms). The other session's uncommitted pfqmm change (a 32-row last tile) halves that for every slab, cut or
  not.

## The smallest code change that realizes it, and what it costs

A merge window in `_slab` (worker.py), active only with the reply KV on:

```python
PREFILL_MERGE = int(os.environ.get("QWEN_PREFILL_MERGE_TOKENS", "16")) if PROMPT_CACHE_REPLY else 0

def _slab(pos, left, cuts):
    n = _step(pos)
    end = pos + left
    nxt = min((c for c in cuts if c > pos and not (PREFILL_MERGE and (c - pos < PREFILL_MERGE
               or end - c < PREFILL_MERGE))), default=pos + n)
    return min(n, left, nxt - pos)
```

`restore`'s first-slab sizing already calls `_slab`. Checkpoints are only taken at reached slab ends
(`if pos in cuts`), so a skipped cut simply has none. `_bounds` is unchanged.

**What it does to determinism:**

- **The head rule cannot fire on a checkpoint resume.** Checkpoints are `_bounds` cuts, and cuts are
  ≥ PREFILL_MIN_STEP = 256 apart. It fires only on reply-KV resumes, which are already outside the hit == cold
  guarantee.
- **The tail rule depends only on the prompt's own end.** A cold run and a hit of the same prompt cut the same
  way. Every checkpoint an entry keeps is still a `_bounds` cut with `_bounds` slabs below it.
- **So `QWEN_PROMPT_CACHE_REPLY=0` stays bit-exact and unchanged.** With REPLY=0 the window is off anyway, because
  the tail checkpoint is that mode's next-turn resume point.

**What it costs in resume depth (REPLY=1):**

| skipped cut | checkpoint lost | when that checkpoint is used | fallback cost | how often |
|---|---|---|---|---|
| tail | the generation prompt's start | the next turn cannot resume at the reply end: the reply entry gone (3 in 447 calls here), not kept, or the trailing-whitespace case (1 in 80 in the reply_kv gate) | resumes at the previous checkpoint: the reply + tool output again, median ~500 and up to ~2K tokens, ~1.3 s at the in-run rate | ~1 % of calls ≈ 15 ms/call on average, against the ~85 ms saved on 57 % of calls (≈ 50 ms/call) |
| head | the tool message's start, 2 tokens after the reply end | a sibling that repeats the reply and diverges inside the tool output (never in AppWorld) | the reply end's neighbourhood: tens of tokens | – |

**Verdict:** a small, safe lever.

- Gain: ~+2 % requests/min and a 0.3–0.7 s shorter TTFT on most hits.
- Cost: ~2 lines plus a test that REPLY=0 slab shapes are unchanged and that a reply-KV hit with a ≤ 16-token
  tail/head prefills in one slab.
- **Not worth it:** reshaping `_bounds` to the chain-policy cuts.
  - It gains ~1 % more.
  - "The last 3 starts" depends on the prompt's end, so an older turn's cuts would no longer be cuts of the next
    prompt, and every checkpoint resume (the REPLY=0 path and all reply-KV fallbacks) would break.

## Reproduce

```zsh
# (a) slab lines come with every sched_timer run (SCHED_SLAB_LOG=1 default); table + policies:
python3 scripts/perf/agentic/slab_hist.py raw/replay_b1_w_b16a.log raw/replay_b1_w_b16b.log \
  --prefix chatcmpl-replay-b16 --replay raw/replay_b1_b16a.json --replay raw/replay_b1_b16b.json \
  --cost raw/slabs_cost.json
# (b), (c) — GPU lock; from src/worker with the standard env (+ SCHED_WORKER_DIR=<frozen copy>, optional):
.venv/bin/python ../../scripts/perf/prof/slab_cost.py --offsets 10240,24576 --rounds 7 --json $PERF_OUT/slabs_cost.json
.venv/bin/python ../../scripts/perf/prof/slab_boundary.py --rounds 8 [--phases] --json $PERF_OUT/slabs_boundary.json
```

Note: the first `slab_boundary.py` run lost its final printed table to `os._exit` (since fixed with a flush). Its
tables above were computed from `raw/slabs_boundary.json`.

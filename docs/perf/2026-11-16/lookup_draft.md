# Prompt lookup decoding (QWEN_LOOKUP) on the 27B + DFlash2

## Design

`src/worker/qwen_worker/lookup.py` + `_Scheduler._spec` (`sched_spec.py`).

- **Index (`_Lookup`)**: the token sequence (prompt + every emitted token) and a dict
  `n_min`-gram → the positions that followed it (`QWEN_LOOKUP_N`, 3). An n-gram enters
  the dict only once its next token is known, so a query never matches the current
  end itself. `propose(k)`: the latest 8 occurrences of the last `n_min` tokens, each
  match extended backwards up to `QWEN_LOOKUP_NMAX` (32; 8 in run 1) tokens; the longest wins, the
  latest among equals (the most recent version of a re-emitted text). A continuation
  that runs into the end repeats itself (LZ77-style overlap: runs, cycles). Pure
  Python, no MLX op; one dict insert per token.
- **Round**: a match of ≥ `QWEN_LOOKUP_MIN` (4; 3 in runs 1–2) tokens with ≥ 2 drafts (a verify block of
  ≥ 3 rows: mrqmm's minimum) gives a round whose drafts are the continuation (≤
  `QWEN_LOOKUP_K`, 15: mrqmm's 16 rows), verified exactly as a drafter's block
  (`speculative_verify`, `pick`, `_speculative_walk`, commit / rollback, thinking
  budget cut, reply-KV `seen`). Exactness class = DFlash's. Whether a match is taken and
  how many drafts it gets is `_Gate`'s (below: `QWEN_LOOKUP_K0`, `QWEN_LOOKUP_BACKOFF`).
- **Modes** (`QWEN_LOOKUP`):
  - `draft`: as `1` for rows the drafter takes; with no drafter loaded, off (plain rows
    stay the plain loop, bit for bit).
  - `1` with a drafter (policy a): a lookup round when matched, else the drafter's block.
    Rounds the drafter does not draft hand their hidden states to `_DraftCtx`; its next
    block gets every position since its last one (more than its window: the last
    `win`, its rotating caches' offsets moved on by the rest — what DFlashAttention
    does itself with a longer context). Lookup rounds do not enter the drafter's
    acceptance statistics.
  - `splice` with a drafter (policy b): the drafter's block every round; where a match
    exists and agrees with the drafter's first draft, the (longer) continuation
    instead. Needs a host read of the drafts (a sync before the verify).
  - `1` without a drafter (policy c, `eng.draft` None; fp16 KV, no thinking budget):
    lookup rounds where matched, else plain steps in step-ahead bursts of ≤
    `QWEN_LOOKUP_BURST` (16) — the plain loop's call (`_step` at B = 1, explicit
    position) and sampler (`_sample`), so the tokens between rounds are the plain
    loop's bit for bit; a burst stops at a match (the queued step's token becomes the
    anchor, nothing is wasted). Demotion to a batch row as DFlash rows.
- Engine: with `QWEN_LOOKUP=1` and no drafter, mrqmm and the verify setup
  (`_setup_verify_mrqmm` without the drafter part, `_probe_verify_attention`) are
  installed as with a drafter.

CPU tests: `src/worker/tests/test_lookup.py` (index units, `_DraftCtx` trimming,
the three modes against the plain loop with a stub lookup / drafter that check the
emitted tokens and the drafter's context, stop token inside an accepted block,
demotion, short answers, the real index on the tiny model's cyclic answers).

## Setup

M1 Max 32 GB, mlx 0.32.3, mlx-vlm 0.7.6. Target `weights/qwen38-27b-mlx-4bit`
(lmstudio-community 4-bit g64), drafter `weights/qwen38-27b-dflash2` (4-bit, fp32
activations, fixed block 8 through mrqmm). Env for every run:

```
QWEN_MODEL=…/qwen38-27b-mlx-4bit QWEN_DRAFT=…/qwen38-27b-dflash2 QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1
QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_REPLY=0
```

(the prompt cache only saves the repeated prefills; `REPLY=0` keeps a hit == cold bit for bit.)
Harness `scripts/perf/lookup/ab_lookup.py`: the real Engine / scheduler in one process,
variants interleaved per (round, workload) in a rotating order, greedy, thinking off,
384 generated tokens (`max_tokens`), decode tok/s = (n − 1) / (t_last − t_first).
Prompts fixed in `scripts/perf/lookup/prompts/`:

| workload | prompt | task |
|---|---|---|
| rewrite_a | 1958 tok | rename a function in a 132-line Python file, output the whole file |
| rewrite_b | 2469 tok | add one constant to a 234-line Python file, output the whole file |
| reemit | 309 tok | turn 2: add two methods to the class of turn 1 (213 tokens, plain greedy) and print it again |
| code | 46 tok | a short coding task (wtest TASKS[0]) |
| code2k | 3589 tok | unit tests for a 3K-token module |
| en2k | 2630 tok | explain 8 KB of the README in your own words |
| ru2k | 2699 tok | retell a Russian changelog in your own words |

Variants: `plain` (no drafter, no lookup), `dflash` (drafter alone), `hyb` (`QWEN_LOOKUP=1`
with the drafter), `splice`, `lplain` (`QWEN_LOOKUP=1`, no drafter).

## Run 1: ungated (QWEN_LOOKUP_MIN=3, K=15, fixed)

Measured, `raw/lookup_ab1.log` (+ `raw/lookup_ab1_parsed.json`); medians over rounds (3 for
rewrite_a, 2–3 for the rest: the run was stopped at the slot's 25-min limit in round 3).
Decode tok/s:

| workload | plain | dflash | hyb | splice | lplain | hyb / dflash | lplain / plain |
|---|---|---|---|---|---|---|---|
| rewrite_a | 18.24 | 27.26 | **78.32** | 63.52 | 74.92 | ×2.87 | ×4.11 |
| rewrite_b | 18.24 | 28.02 | **74.10** | 66.92 | 69.52 | ×2.64 | ×3.81 |
| reemit | 19.25 | 53.74 | **72.45** | 69.56 | 48.25 | ×1.35 | ×2.51 |
| code | 19.12 | 39.20 | 35.64 | 33.87 | 17.88 | ×0.91 | ×0.94 |
| code2k | 18.05 | 33.94 | 29.08 | 31.44 | 19.73 | ×0.86 | ×1.09 |
| en2k | 18.22 | 17.14 | 16.78 | 16.84 | 17.84 | ×0.98 | ×0.98 |
| ru2k | 18.10 | 14.61 | 14.01 | 14.50 | 17.50 | ×0.96 | ×0.97 |

Rounds per request (round 0): on rewrite_a hyb ran 27 lookup rounds accepting 341 / 397
drafts (86 %) and only 4 drafter rounds; the drafter alone accepts 285 / 693 (41 %) there —
DFlash2 does not copy a long file well. On ordinary text the lookup fires on coincidental
3-grams and is mostly wrong: code 13 / 180 drafts accepted (12 rounds of 15 drafts), code2k
113 / 390, ru2k 2 / 105. Each such round replaces a drafter round worth ~5 tokens (code) by
~1–2 tokens: −9…−14 % on code / code2k. `splice` loses its sync and still accepts poorly on
code (19 / 105). Without a drafter the same misses cost a 16-row verify (~2–3 plain steps)
each: −3…−6 % on code / en2k / ru2k. Not gate-passing as is → `_Gate` (below).

(Side observation, not this change: on en2k / ru2k the drafter alone is slower than plain
greedy, 17.1 / 14.6 vs 18.2 / 18.1 tok/s — DFlash2 accepts 222 / 1125 and 205 / 1243 drafts
there.)

## Verify cost by block size (measured)

`ab_lookup.py --oracle-ks 0,3,7,11,15` (`raw/lookup_ab2.log`): no drafter, the lookup replaced
by an oracle proposing the plain answer's next K tokens (all accepted), `code` workload, 2
rounds, median ms per round (= one verify forward of K + 1 rows + commit + host):

| verify rows | 1 (plain step) | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| ms / round | 54.7 | 99.4 | 106.0 | 154.8 | 158.8 |
| tok/s (all accepted) | 18.3 | 40.1 | 75.3 | 77.3 | 100.5 |

Two cost steps, mrqmm's row buckets (4-bit: 3..8 and 9..16 rows): a 4-row block costs as
much as an 8-row one, 12 as 16. So a lookup round has 7 or 15 drafts, never 3 or 11
(`QWEN_LOOKUP_K0=7`, then 15). A drafter round is ~133–144 ms (drafter ~30 + 8-row
verify) — a lookup round of 7 drafts (106 ms) is ~25 % cheaper, of 15 (159 ms) 15 % dearer
but yields twice the tokens in a copy. Upper bound with every draft right: 100 tok/s
(×5.5 plain) at 16 rows.

## Gate (`_Gate`)

- `QWEN_LOOKUP_K0`: a round starts at K0 drafts, 2k + 1 after a fully accepted one
  (≤ `QWEN_LOOKUP_K`), K0 again after a miss.
- `QWEN_LOOKUP_BACKOFF`: a round keeping < half its drafts doubles the match length the next
  one needs (≤ `QWEN_LOOKUP_NMAX`, now 32: the index scores matches up to 32 tokens), a fully
  accepted one resets it to `QWEN_LOOKUP_MIN`. Coincidental matches (boilerplate, repeated
  identifiers) stop costing rounds; a real copy, whose match grows by one per token, comes
  back after a few tokens.

### Run 2: gated (MIN 3, K0 3, backoff, NMAX 32), 2 rounds (measured)

`raw/lookup_ab2.log` / `.json` / `.jsonl` (the JSONL keeps the plain answers' ids for
`sim.py`). Medians over 2 rounds, decode tok/s:

| workload | plain | dflash | hyb | lplain | hyb / dflash | lplain / plain |
|---|---|---|---|---|---|---|
| rewrite_a | 18.97 | 29.16 | 70.98 | 73.16 | ×2.43 | ×3.86 |
| rewrite_b | 18.96 | 29.61 | 78.84 | 76.02 | ×2.66 | ×4.01 |
| reemit | 19.16 | 54.06 | 83.55 | 51.91 | ×1.55 | ×2.71 |
| code | 19.00 | 38.92 | 38.32 | 18.93 | ×0.985 | ×0.996 |
| code2k | 18.00 | 34.61 | 35.10 | 21.89 | ×1.014 | ×1.216 |
| en2k | 18.25 | 16.71 | 17.06 | 18.17 | ×1.021 | ×0.995 |
| ru2k | 18.30 | 14.96 | 14.63 | 18.14 | ×0.978 | ×0.991 |

The losses are gone within noise: on code hyb ran 5 lookup rounds (6 / 23 drafts) instead
of 12 (13 / 180), on ru2k 2 rounds — ru2k's −2.2 % with 2 lookup rounds of ≤ 4 rows is the
spread of two rounds, not their cost (≤ 0.2 s of 26). Identity: below.

## Offline replay (`sim.py`, derived)

`sim.py raw/lookup_ab2.jsonl` walks each plain greedy answer with the worker's own `_Lookup`
and `_Gate`: lookup acceptance is exact (the answer is fixed), verify costs from the table
above, drafter rounds priced at their measured per-workload yield (3.9–7.3 tokens per
~135 ms) — so the hybrid column is an estimate (the drafter's yield is not uniform: it is
best exactly where the lookup also is). Output in `raw/sim_ab2.txt`. Estimated speed-up
(selected; vs plain without a drafter / vs the drafter alone with one):

| gate | rewrite_a | reemit | code | code2k | en2k | ru2k |
|---|---|---|---|---|---|---|
| ungated m3 K15 nmax 8, no drafter | 4.29 | 2.61 | 0.96 | 1.17 | 1.00 | 0.98 |
| K0 3 + backoff m3, no drafter | 4.03 | 2.73 | 1.00 | 1.24 | 1.00 | 1.00 |
| K0 7 + backoff m4, no drafter | 4.14 | 2.90 | 1.01 | 1.20 | 1.00 | 1.00 |
| ungated m3 K15 nmax 8, drafter | 2.76 | 1.31 | 0.86 | 0.96 | 0.99 | 0.99 |
| K0 3 + backoff m3, drafter | 2.57 | 1.44 | 0.98 | 1.12 | 0.99 | 1.00 |
| K0 7 + backoff m4, drafter | 2.89 | 1.48 | 0.98 | 1.15 | 1.00 | 0.99 |

(The replay matches run 1's ungated measurement within ~2–4 % where it can be checked:
lplain code 0.96 vs 0.94 measured, ru2k 0.98 vs 0.97; hybrid code 0.86 vs 0.91.) Backoff is
what removes the hybrid's loss on code; K0 7 > K0 3 (same cost bucket); MIN 3 vs 4 within
noise. Defaults set from this: `QWEN_LOOKUP_MIN=4 NMAX=32 K=15 K0=7 BACKOFF=1`.

## Run 3: final, defaults (MIN 4, NMAX 32, K 15, K0 7, backoff), 3 rounds (measured)

```
cd src/worker && setopt NO_BG_NICE && <env above> .venv/bin/python ../../scripts/perf/lookup/ab_lookup.py \
  --rounds 3 --variants plain,dflash,hyb,lplain --workloads rewrite_a,reemit,code,code2k,en2k,ru2k \
  --json raw/lookup_ab3.json --jsonl raw/lookup_ab3.jsonl
```

`raw/lookup_ab3.log` / `.json` / `.jsonl` (rewrite_b dropped: a near-duplicate of rewrite_a in
runs 1–2). `hyb` = `QWEN_LOOKUP=1` with the drafter, the same path as `QWEN_LOOKUP=draft` for a
drafter row. Medians over 3 rounds, decode tok/s (per-round values in the log):

| workload | plain | dflash | hyb | lplain | hyb / dflash | hyb / plain | lplain / plain |
|---|---|---|---|---|---|---|---|
| rewrite_a | 18.16 | 27.09 | **76.81** | 71.29 | **×2.84** | ×4.23 | ×3.93 |
| reemit | 18.35 | 48.04 | **80.02** | 53.31 | **×1.67** | ×4.36 | ×2.91 |
| code | 18.29 | 35.18 | 35.40 | 18.26 | ×1.006 | ×1.94 | ×0.999 |
| code2k | 17.80 | 33.92 | 35.53 | 20.91 | ×1.047 | ×2.00 | ×1.175 |
| en2k | 17.94 | 16.45 | 16.38 | 17.79 | ×0.996 | ×0.91 | ×0.991 |
| ru2k | 17.98 | 14.89 | 14.66 | 17.76 | ×0.985 | ×0.82 | ×0.988 |

Peak 21.26 GB (process, whole run). Rounds per request (round 1, from the log):

| workload | dflash: drafter rounds, accepted / drafted | hyb: drafter | hyb: lookup | lplain: lookup, plain steps |
|---|---|---|---|---|
| rewrite_a | 99r 285 / 693 | 6r 17 / 42 | 25r 336 / 348 | 25r 336 / 347, 23 |
| reemit | 41r 229 / 287 | 7r 49 / 49 | 15r 199 / 209 | 16r 203 / 216, 50 |
| code | 79r 304 / 549 | 78r 300 / 542 | 2r 3 / 14 | 3r 6 / 21, 374 |
| code2k | 77r 307 / 531 | 65r 233 / 448 | 8r 78 / 95 | 10r 78 / 108, 295 |
| en2k | 162r 222 / 1125 | 162r 222 / 1125 | 0r | 0r, 383 |
| ru2k | 179r 204 / 1250 | 180r 202 / 1257 | 1r 0 / 7 | 2r 0 / 14, 381 |

- **Where text is copied** (a file rewritten with one change, a turn re-emitting the previous
  answer) the lookup carries the round: 86–97 % of its drafts accepted, the drafter runs 6–7
  rounds instead of 41–99. ×2.84 / ×1.67 over the drafter alone, ×4.2–4.4 over plain.
- **Ordinary prompts**: on en2k hyb ran no lookup round at all — the same rounds as dflash, so
  its −0.4 % is the run's noise floor (the index costs one dict insert per token). On ru2k one
  lookup round (0 / 7) and one extra drafter round: −1.5 %, of which ≤ 0.2 s of 26 s is the
  round, the rest the spread (round 0 was slow for every variant: 14.0–15.6 vs 14.7–17.8).
  code ×1.006, code2k ×1.047. All within the ±2 % no-regression bound.
- Without a drafter (lplain): ×3.9 / ×2.9 on the copy workloads, ×1.18 on code2k, −0.1…−1.2 %
  on the rest (2–3 rounds of 7 drafts that missed, ~50 ms each).
- (Round 0 of code / code2k hyb ran a cold prefill — `reused=0`, its entry evicted — so the
  drafter started with the prompt's full hidden states instead of the cache's 128-position
  tail, `QWEN_DRAFT_TAIL`: 40 / 43 tok/s vs 35 on hits. The other variants hit in every
  round; the medians are hit rounds for all.)

### Identity vs plain greedy (measured, run 3: 3 / 3 rounds each)

| workload | dflash | hyb | lplain |
|---|---|---|---|
| rewrite_a, reemit, code | == | == | == |
| code2k | ≠ at token 213 | ≠ at 213 | == |
| en2k | ≠ at 27 | ≠ at 27 | == |
| ru2k | ≠ at 280 | ≠ at 280 | == |

hyb diverges exactly where the drafter alone does (same position, every round): adding lookup
rounds moved no answer out of DFlash's class on these prompts, and every hyb answer is
token-identical to the dflash answer. lplain (no drafter) was 18 / 18 identical to plain in
run 3, but 2 of 7 workloads diverged in run 2 (code2k at 379, ru2k at 280, MIN 3 / K0 3: more
rounds). Its verify rows (3..16 rows through mrqmm) round differently from a 1-row step, so
lookup without a drafter is in the batched-decode / DFlash numerics class, not bit-exact.

## Verdict

Gate (NIGHT_RULES): numerics = DFlash's class (identity ≥ 15/16 with divergences at near-ties)
and a ≥ 3 % in-run median gain, no workload worse than −2 %.

- **`QWEN_LOOKUP=draft` / `1` with the drafter (policy a): passes on the measured set.**
  ×2.84 (rewrite) and ×1.67 (re-emit) over DFlash, ordinary code / English / Russian within
  −1.5…+4.7 %, answers token-identical to DFlash's on every workload. Recommended for
  agentic / coding runs of the 27B + DFlash2. The default stays `0` here: the gate covered one
  model, greedy, 6 workloads, ≤ 3.6K-token prompts; a default flip wants a run on the 9B +
  DFlash and at long context first (below).
- **`QWEN_LOOKUP=1` without a drafter (policy c)**: ×2.9–3.9 on copies, ±1 % elsewhere; not
  bit-exact vs plain (run 2), so opt-in only, like DFlash.
- **`splice` (policy b)**: slower than `1` on 5 of 7 workloads of run 1 (−4…−19 %: the
  drafter block is computed every round and needs a sync to compare), faster only where
  ungated `1` lost (code2k +8 %, before `_Gate`); not measured gated; not recommended, kept
  for experiments.

## Open issues

- Sampling requests (`QWEN_DRAFT_SAMPLING=1`) go through the same rounds (the verify's
  sample-match acceptance) — neither measured nor covered by a test.
- Long context (≥ 32K) not measured: the index is a dict over every n-gram of the prompt
  (pure Python: 60K random tokens build in 0.036 s, a proposal < 1 µs — measured on the CPU,
  `_Lookup(ids, 3, 32)`), but the verify attention's cost grows with the keys, which moves the
  gate's break-even.
- The 9B (+ `weights/qwen35-9b-dflash`, whose full-attention drafter layer keeps every
  position: `_DraftCtx` trims nothing there) is CPU-tested only.
- The gate's constants (verify costs 4 rows = 8 rows, 12 = 16) are the 27B 4-bit's mrqmm
  buckets; the 5-bit 27B / 8-bit 9B have their own (K0 7 is still a bucket edge for both).
- Batch rows (B ≥ 2) never take the lookup: like DFlash it is a lone-row path (a row is
  demoted to plain batch decoding when another request is admitted).

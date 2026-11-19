# Spec-prefill on integ3: time to the first token under `QWEN_DRAFT=auto` (9B)

Ornith-1.5-9B 8-bit (`weights/ornith15-9b-mlx-8bit`), drafters DFlash (`ornith15-9b-dflash`) and the
8-bit MTP head (`ornith15-9b-mtp-q8`), fp16 KV, prompt cache and reply KV on. M1 Max 32 GB, mlx 0.32.3,
mlx-vlm 0.7.6. Labels: [M] measured, [D] derived. Code: branch integ3 (afb5fbc), where spec-prefill
(`sched_spec.PREFILL_NO_SEED` / `PREFILL_DEFER`, [../../drafters.md](../../drafters.md)) was ported from
branch spec-prefill; integ3 is the code of this branch. The first measurement, on branch spec-prefill
(main's base), is in [../2026-11-18/auto_gate_9b.md](../2026-11-18/auto_gate_9b.md#what-paid-spec-prefill-rawspfspf9txt-diff9txt).

Tools: `scripts/perf/mtp/ab_spf.py` (every variant in one process, interleaved per round and cell,
3 rounds, 32 generated tokens; `+old` = both switches off, the prefill as before) and
`scripts/perf/mtp/diff_spf.py` (the stored prompt-cache entries compared array by array). Raw:
`raw/spf/int3_spf9.*`, `raw/spf/int3_diff9.*` (paths in them read `$PERF_OUT`).

## Time to the first token

`auto`'s time to the first token × plain decode's in the same round, the median of the per-round
ratios [M]; "old" = `auto+old`, "on" = `auto` with both switches on (the default):

| prompt | cold, old | cold, on | prompt-cache hit, old | hit, on |
|---|---|---|---|---|
| code, 2K (2047 tokens) | 1.018 | 0.983 | 1.403 | 0.934 |
| code, 8K (8192 tokens) | 1.048 | 1.023 | 1.384 | 1.073 |
| code, 32K (32768 tokens) | 1.064 | 1.042 | 1.283 | 0.998 |

In seconds (medians over 3 rounds) [M]: 2K cold 4.33 s plain, 4.43 s old, 4.26 s on; 8K cold 17.10 /
17.99 / 17.49 s; 32K cold 75.58 / 80.50 / 79.01 s; hits 0.24–0.30 s plain, 0.33–0.39 s old, 0.22–0.30 s
on. The whole request (32 tokens) on / old: 0.984–1.002 [M]: the deferred work runs after the first
token, it is not removed.

The head alone (`head`, the gated head) cold 2K 1.067 → 0.988 × plain, 8K 1.052 → 1.033; hits 1.34 →
0.99 and 1.35 → 0.95. DFlash alone was at plain's level before and after (cold 0.97–1.00, hits
0.94–1.01) [M].

On this base the cold "old" numbers are lower than on spec-prefill's base (2K 1.064, 8K 1.073, 32K 1.085
there), so the gain on a cold prompt is smaller here (on / old 0.965–0.978 against 0.936–0.978); the hit
gains are about the same. The 8K hit's 1.073 is the median of per-round ratios 0.967, 1.073 and 1.075
(medians 0.244 s against plain's 0.242 s; spec-prefill's base: 0.973): 3 rounds of ~0.24 s hits do
not resolve a difference of 2–18 ms [D].

## Exactness

- Identity pass (`+det` variants, switches off against on, the cache emptied before each): cold tokens,
  hit tokens and stored entries identical in 5 of 5 cases (the head, DFlash, both drafters on DFlash's
  rounds and on the head's rounds at 2K; both on DFlash's rounds at 8K) [M].
- Stored entries, array by array (`diff_spf.py`, 8K prompt, 32 tokens, the cache emptied before each
  run): for each of the three drafter setups (both, the head, DFlash) each of the three switch states
  (`PREFILL_NO_SEED` alone, `PREFILL_DEFER` alone, both) against both off: tokens the same, 277 of 277
  components equal, the same checkpoint positions (2048, 4096, 5130, 8146): 9 of 9 comparisons equal [M].

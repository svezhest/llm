# Cascade attention (plan Part B, step B4) on the GPU: exactness gate, waves, verdict

## Setup

| | |
|---|---|
| date | 2026-10-12, 17:44–19:27 |
| machine | M1 Max 32 GB |
| model | Ornith-1.5 9B 8-bit, fp16 KV (`QWEN_KV_BITS=0`) |
| other settings | defaults: fp16, pfqmm, mrqmm, GDN fuse, max_batch 16, 3 GB prompt-cache soft cap, no DFlash |
| gateway | 127.0.0.1:8090 (pid 27313, untouched) |

**Code under test.** The uncommitted B2/B3 `src/worker/worker.py` + `cascade.py` on top of 66a7c17:

- G1 first ran at 17:47–17:59, before the review fixes.
- Everything else ran on the **fixed** code (`_GroupRef` by weakref, no cycles; `_group_wins` on real lengths + budget guard; eager suffix copy in `add`; staged releases; members in batch order). G1 was re-run on the fixed code at 19:21: **bit-identical numbers** (same KL to every printed digit).

**Tools.**

| path | role |
|---|---|
| `scripts/perf/prof/agree_cascade.py` | G1 |
| `scripts/perf/prof/release_probe.py` | released-cache capacity check |
| `scripts/perf/prof/cascade_step_probe.py` | in-process step A/B and donation |
| `scripts/perf/final/run_cascade.zsh` | driver: G2–G6 waves, END |

**Worker settings.** ON / OFF here mean these worker env settings:

- **ON** = `QWEN_PROMPT_CACHE=1 QWEN_CASCADE=1 QWEN_CASCADE_MIN_ROWS=8 QWEN_CASCADE_MIN_PREFIX=4096 QWEN_CASCADE_CHECK=1`
- **OFF** = `QWEN_PROMPT_CACHE=1 QWEN_CASCADE=0`

**Bench flags.** Every wave used `bench_concurrency.py --prefix shared --ignore-eos on --probe 64 --cooldown 30 --thinking off` and a fresh worker per launch. The driver followed the `run_pcache3.zsh` method:

- restart only :50051 and wait for the gateway's Health;
- `pgrep` guard (it waited out a reviewer's CPU pytest twice);
- all workers at nice 0;
- an `active=` ≥ 27 GiB watcher, which never fired.

**Raw data** is in `raw/cascade_*`:

- JSON/TXT per phase;
- `cascade_worker_*.log`, one log per launch;
- `cascade_status.txt`: timestamps, load, Health.

**Errors: none.** Across all 9 worker logs there were 0 Traceback, ERROR, `ended at`, `RESOURCE_EXHAUSTED`, `not kept`, `store failed`, and 0 `not grouped` warnings. `QWEN_CASCADE_CHECK=1` was on in every ON run, and every member's [0, P) KV matched bitwise. There were 0 HTTP errors.

Only `bench-*` / `v1-*` request ids reached the worker, so there was no foreign traffic. Load average was 2.3–3.0, and `pmset -g therm` gave no warnings.

## Verdict

| criterion | result |
|---|---|
| B.7 top-1 ≥ 99.7 % at B=8/16, 8K and 32K | **PASS** (8/8 cells, minimum 99.71 %) |
| B.7 KL mean ≤ 1.5e-5 | **PASS 7/8** cells; set 0 at 8K/B=16 is 1.69e-5, where **today's per-row path is 1.56e-5** on the same prompts |
| B.7 grouped-vs-control KL ≤ control-vs-solo KL | **PASS** (6/6) |
| B.7 per-layer rms vs fp32 ≤ 2.3e-4 and ≤ per-row kernel | **PASS**: mean 2.05–2.12e-4; ≤ the per-row kernel in 768/768 (row, layer) pairs |
| a released cache (capacity P+S+256, not a multiple of 256) keeps per-row bits on Metal | **PASS**: KV equal, then 300/300 steps bitwise equal logits at B=1 (3 rows) and in a B=3 per-row batch |
| donation under step-ahead | **PASS**: step-ahead is 2–10 ms/step faster than sync, and its peak above base equals sync's (285–384 MiB); no per-step stack copy |
| 16 × 8K shared: group of 16 forms, step faster, ~3 GB less memory | **PASS**: step 66 → 54 ms in-process (×1.24); steady 16-row decode 12.0–12.5 → 16.9–18.7 steps/s in the worker; active 16.2–16.7 → 13.0–13.7 GiB; peak 17.6–18.1 → 15.0–15.3 GB |
| 16 × 32K shared: all 16 admitted | **PASS**: 16 rows, 0 waiting after formation (Part A: 12 + 4 waiting); peak 25.26 → 23.24 GB |
| 16 × 32K step ≈ ×1.7 (≈ 62 ms) | **MISS**: 92.5 ms in-process, 93–98 ms in the worker; ~×1.2 vs the derived ~110 ms per-row step (see "32K") |
| 1K prefix never groups | **PASS** (0 cascade lines) |
| B=1 identity (G5) | **PASS**: 20/20 byte-identical to `raw/pcache2_v1_off.json` |

**Recommendation: flip the default to `QWEN_CASCADE=1`**, for fp16-KV configurations only. The scheduler already gates it off for quantized KV, no prompt cache, or max_batch < 3. Keep `QWEN_CASCADE_MIN_ROWS=8`, `QWEN_CASCADE_MIN_PREFIX=4096`, and `QWEN_CASCADE_CHECK=0`. Details are at the end, together with the caveats that should go into CLAUDE.md.

## G1: exactness gate (`agree_cascade.py`)

**Prompt construction.** One shared system prompt of ~P tokens (`--prefix shared` filler) plus B distinct questions.

- The prefix is prefilled once, up to the deepest common slab end. Each row resumes from a `_copy_cache` of it and prefills its suffix with the worker's `_slab` cuts.
- A resumed copy equals a cold full prefill bitwise (checked at 8K).

**Three copies per prompt:**

- **solo** (B=1 library call): its greedy stream is the teacher stream.
- **grouped**: the group is built with `cascade.PrefixGroup.build` / `.add` using the scheduler's parameters, and the rows hold `_GroupRef`s in a `_PrefixGroup`. Each step calls the unbound `_Scheduler._plan` + `_Scheduler._grouped_forward` with `RowsKV(caches, layer, plan)` and the stacked GDN state. This is the worker's real decode path (`_rows_attention` → `_cascade_rows` → `cascade.attention`); the helper route, not a re-implementation.
- **control**: the same batch with no plan, i.e. today's per-row path with mrqmm.

**Run parameters.** 128 teacher-forced steps. Two prompt sets were used:

- set 0: questions 0–15;
- set 1: another filler, questions 16–31.

The control at 32K/B=16 was skipped: 16.9 GB of per-row KV does not fit, and it does not fit in the worker either.

| set | P | B | grouped vs solo top-1 | KL mean (p99 / max) | control vs solo top-1 | KL mean | grouped vs control top-1 | KL mean |
|---|---|---|---|---|---|---|---|---|
| 0 | 7975 | 8 | 99.80 % | 1.45e-5 (1.2e-4 / 7.6e-4) | 99.80 % | 1.39e-5 | 99.61 % | 1.01e-5 |
| 0 | 7975 | 16 | 99.71 % | 1.69e-5 (1.3e-4 / 3.6e-3) | 99.76 % | 1.56e-5 | 99.85 % | 1.14e-5 |
| 0 | 31969 | 8 | 100.00 % | 1.35e-5 (9.9e-5 / 6.7e-4) | 100.00 % | 1.19e-5 | 100.00 % | 1.08e-5 |
| 0 | 31969 | 16 | 99.85 % | 1.36e-5 (9.8e-5 / 4.4e-3) | — | | | |
| 1 | 7938 | 8 | 99.90 % | 1.31e-5 (1.0e-4 / 1.6e-4) | 100.00 % | 1.24e-5 | 99.90 % | 1.07e-5 |
| 1 | 7938 | 16 | 99.85 % | 1.25e-5 (9.7e-5 / 3.5e-4) | 99.85 % | 1.25e-5 | 99.90 % | 1.03e-5 |
| 1 | 31953 | 8 | 99.90 % | 1.35e-5 (9.7e-5 / 1.7e-4) | 99.90 % | 1.25e-5 | 99.80 % | 1.13e-5 |
| 1 | 31953 | 16 | 99.85 % | 1.35e-5 (9.9e-5 / 5.9e-4) | — | | | |
| both | 8K | 8 | 99.85 % | 1.38e-5 | 99.90 % | 1.32e-5 | | |
| both | 8K | 16 | 99.78 % | 1.47e-5 | 99.80 % | 1.41e-5 | | |
| both | 32K | 8 | 99.95 % | 1.35e-5 | 99.95 % | 1.22e-5 | | |
| both | 32K | 16 | 99.85 % | 1.36e-5 | — | | | |

**KL along the generation.** KL rises in both classes alike. Set 0 at 8K/16, per quarter of the 128 steps:

- grouped: 1.14 / 1.30 / 1.92 / 2.39e-5;
- control: 1.16 / 1.14 / 1.72 / 2.24e-5.

**Grouping vs today's path.** Grouping adds 0–13 % KL over the control (pooled: ×1.05 at 8K, ×1.11 at 32K/B=8). That is less than the spread between the two prompt sets: the control alone ranges 1.25–1.56e-5 at 8K/B=16. The one absolute-KL miss is a cell where today's path misses the bar too. The 1.5e-5 limit came from mrqmm's short coding prompts; long shared-prefix prompts sit at 1.2–1.6e-5 without any grouping.

**Per-layer attention output at step 64**: rms relative error vs per-row fp32 SDPA on fp32 q and the group's fp32 K/V. The per-row column is the per-row fp16 kernel on the same q/K/V.

| P / B | cascade mean (max) | per-row fp16 kernel mean (max) | cascade ≤ per-row |
|---|---|---|---|
| 8K / 8 (set 0 / 1) | 2.05 / 2.08e-4 (2.8e-4) | 2.28 / 2.29e-4 (3.1e-4) | 64/64, 64/64 |
| 8K / 16 | 2.08 / 2.10e-4 (3.0e-4) | 2.30 / 2.32e-4 (3.4e-4) | 128/128, 128/128 |
| 32K / 8 | 2.07 / 2.12e-4 (2.9e-4) | 2.32 / 2.34e-4 (3.2e-4) | 64/64, 64/64 |
| 32K / 16 | 2.10 / 2.12e-4 (3.2e-4) | 2.35 / 2.35e-4 (3.5e-4) | 128/128, 128/128 |

**Released-cache check (`release_probe.py`, Metal).** P = 8550; offsets 8592 / 8604 / 8598.

- **Capacities.** Released caches have capacity 8848 / 8860 / 8854 (mod 256 = 144 / 156 / 150). The rows' own caches are 8806.
- **Bits after release.** The KV is bitwise equal to the row's own.
- **Continued decoding.** Each row then decoded 300 greedy steps on both copies with the same tokens, crossing capacity growth. The logits were bitwise equal at every step, both at B=1 for each row and in a 3-row per-row batch (`RowsKV`, no plan).

## In-process step cost and donation (`cascade_step_probe.py`)

**Method.**

- Same row construction as G1.
- Greedy decode on each row's own tokens.
- Step-ahead (`async_eval` of step k+1 before reading k, as `_iterate`) and sync modes alternate in blocks of 64 steps, 2 warm-up steps dropped per block.
- ms/step is the median of the blocks. "peak+" is the MLX peak above the block's start.

| P | B | grouped ahead | grouped sync | control (per-row) ahead | control sync | grouped peak+ ahead / sync |
|---|---|---|---|---|---|---|
| 8K | 16 | **53.7** (53.0–54.4) | 63.3 | **66.4** | 75.4 | 285–314 / 285–714 MiB |
| 32K | 8 | **68.2** (63.7–68.6) | 69.6 | **83.6** | 86.8 | 192 / 192–383 MiB |
| 32K | 16 | **92.5–93.4** (88.5 → 98.4 as the suffix grows to 560) | 94.4–99.4 | does not fit | | 383 / 383–756 MiB |

- **8K/B=16: −12.7 ms per step (×1.24).** The microbench predicted −13–16 ms.
- **32K/B=8: −15.4 ms (×1.23).** The plan expected −16–20 ms.
- **Donation works under step-ahead.** In every grouped case step-ahead is faster than sync, by 9.6 ms at 8K/16 and 1.4–6 ms at 32K. The step-ahead peak equals the sync blocks' peak (the sync maxima of 630–756 MiB fall in blocks where the stack grew). A lost donation would copy a whole layer's stack every step, making step-ahead slower than sync with a higher peak; neither shows.
- **Worker active memory** grows only with the stack's cap: +0.25 GiB per 256 tokens at 16 rows, in steps (G6 below).

**32K/B=16 is short of the projection.** The microbench projected ~62 ms for this step (its 32K/16 attention at 18–19 ms + the rest of a 1K step). The measured step is 92.5 ms.

- Going from 8K to 32K adds +39 ms per step at B=16. The microbench predicted about +10 ms.
- So in the model, the 32K prefix part costs roughly 3–4× its microbench time.
- G1 timed the same step at 75 ms at 17:52 and 91 ms at 19:26 with identical numerics. The GPU slowed by ~18 % over the evening (G1 8K/16: 65 → 77 ms), so part of the gap is drift. The in-process A/B ratios above come from one window each.
- **Open item:** profile `cascade.attention` inside the model at 32K/16 (step_prof `--ctx 32k-shared`). Suspects are SLC/bandwidth contention with the rest of the step, or the prefix chunk shape at c=32/33.

## G2: 16 × 8K shared wave (c=16, 256 tokens), ON / OFF / ON, fresh worker each

| run | probe decode / TTFT | wall s | agg tok/s | TTFT p50 / p95 s | decode tok/s/req (worker) | steady 16-row steps/s | active GiB (16 rows) | peak GB |
|---|---|---|---|---|---|---|---|---|
| ON 1 | 36.7 / 15.22 | 34.0 | 120.3 | 17.12 / 20.65 | 16.7 (16.1) | 18.3 | 13.26 | **14.99** |
| OFF | 35.9 / 20.41 (slow prefill window) | 42.8 | 95.8 | 20.73 / 23.07 | 12.1 (11.9) | 11.4 (9.7–13.1) | 16.21 | 17.63 |
| ON 2 | 36.9 / 15.12 | 35.2 | 116.5 | 17.86 / 21.71 | 16.4 (15.8) | 17.5 | 13.26 | **14.96** |
| Part A s1 | 36.9 / 15.27 | 37.1 | 110.4 | 16.80 / 20.22 | 13.5 (13.2) | 14.8 | 16.31 | 17.67 |
| Part A s2 | 36.6 / 19.20 | 39.8 | 102.8 | 19.44 / 21.43 | 13.1 (12.8) | — | — | 18.15 |

**Log evidence**, the same in both ON runs:

```
cascade: group formed: P=8192, 8 rows (…), 671.1 MB (prefix + stacks, fp32), charged 0.95 GiB; reserved 3.02 -> 1.78 GiB
cascade: id=… joined group P=8192 (9 rows, charged 1.00 GiB)     … up to 16 rows, charged 1.35 GiB
batch: rows=16 waiting=0 … steps/s=18.3 … reserved=3.01/15.75GiB … active=13.26GiB groups=1 grouped=16
cascade: group P=8192 dissolved (2 members): 2 rows released, 0.71 GiB charge dropped
```

- **Formation.** The group forms at the 8th join and converts the cold producer, then rows 9–16 join it.
- **P.** P = 8192: the system block ends < `PREFILL_MIN_STEP` past the 8192 slab end, so 8192 is the deepest common end.
- **Dissolve.** The group dissolves at 2 members, since the rows finish within ~1 s of each other.
- **Dissolve transient.** Active memory around the dissolve was 12.81 GB. The batch peak, 14.96–14.99 GB, is set earlier, during admission.

**Reading the 16-row step.** The batch lines log every 10 s, so the 256-token wave gives only 1–2 steady 16-row samples. G6 below repeats the wave with 1024 tokens. The decode tok/s per request goes 12.1 → 16.4–16.7.

## G3: 16 × 32K shared wave (c=16, 128 tokens), ON

| run | probe | wall s | agg tok/s | TTFT p50 / p95 s | decode tok/s/req (worker) | rows at once | peak GB |
|---|---|---|---|---|---|---|---|
| **ON** | 32.6 / 78.16 | 92.3 | **22.2** | 79.31 / **82.52** | **11.1 (11.0)** | **16** | **23.24** |
| Part A V3 (OFF) | 33.0 / 78.75 | 100.0 | 20.5 | 76.82 / 92.88 | 7.9 (7.8) | 12 + 4 waiting | 25.26 |

**All 16 rows are admitted.**

1. The cold row is prefilled, then 7 hits join.
2. At the 8th row the group forms. The 8 rows' 9.36 GiB of reservations drop to 3.38 GiB:
   ```
   cascade: group formed: P=33117, 8 rows, 2348.9 MB (prefix + stacks, fp32), charged 2.55 GiB; reserved 9.36 -> 3.38 GiB
   batch: rows=8 waiting=8 … groups=1 grouped=8
   ```
3. The other 8 hits are admitted and join within 4 s (charged 2.60 → 2.97 GiB):
   ```
   batch: rows=16 waiting=0 … reserved=4.63/15.75GiB pcache=2.38GiB active=14.66GiB groups=1 grouped=16
   ```

**Results.**

- TTFT p95 is 82.5 vs 92.9 s, because nothing waits for rows to finish.
- Per-request decode is 11.0 vs 7.8 tok/s at 16 rows instead of 12.
- **Peak.** The peak is the formation transient: weights + 8 rows' fp16 caches (8 × 1.07 GB) + the new 2.35 GB fp32 group + the 2 GB prompt cache ≈ 22.5–23 GB. The Part A 12-row peak was 25.26 GB.
- **Dissolve** at 2 members: active 14.39 GB at that moment, with 2 rows × 1.07 GB materialized while the group was alive. The dissolve transient is far below the formation peak.

## G4: 8 × 32K shared (c=8, 128 tokens), ON vs OFF; 1K-prefix wave

| run | probe | wall s | agg tok/s | TTFT p50 s | decode tok/s/req (worker) | peak GB |
|---|---|---|---|---|---|---|
| ON c=8 | 32.7 / 89.08 | 94.7 | 10.8 | 84.81 | **13.7 (13.2)** | **23.07** |
| OFF c=8 | 33.2 / 77.16 | 92.1 | 11.1 | 80.38 | 11.3 (11.1) | 20.87 |
| ON 1K c=16 | 28.3 / 2.12 | 24.9 | 164.7 | 5.54 | 14.0 (13.6) | 14.63 |

**c=8 at 32K.**

- Decode per request is **+19 %** (13.2 vs 11.1 tok/s; in-process A/B ×1.23).
- The aggregate is unchanged: ~80 s of cold prefill dominates a 128-token wave.
- The group (P = 32744) forms only when all 8 have joined:
  ```
  reserved 9.27 -> 3.31 GiB
  ```

**Peak +2.2 GB at c=8.** The fp32 prefix (2.28 GB) is built while all 8 members still hold their fp16 caches. This formation transient is plan B.4's "+2·P·32 KiB". At 23 GB it stays well under the 28 GiB cap; the prefill reserve is idle at join time, as planned.

**1K wave.** No group and no cascade line (`MIN_PREFIX`). This regression level is in a different probe window (probe 28.3 tok/s), so it is a sanity check only.

## G6 (extra): steady-state decode, 1024 / 512 tokens

The 256- and 128-token waves gave too few steady 16-row log samples, so G6 repeats them with longer answers.

| run | probe | wall s | agg tok/s | decode tok/s/req | 16-row steps/s (batch lines) | active GiB over the decode | peak GB |
|---|---|---|---|---|---|---|---|
| ON 8K c=16, 1024 tok | 36.6 / 16.09 | 75.8 | **216.2** | **17.6** | **17.3 / 18.7 / 18.3 / 17.7 / 16.9** | 13.02 → 13.23 → 13.49 → 13.74 | **15.31** |
| OFF 8K c=16, 1024 tok | 36.2 / 21.18 | 112.1 | 146.1 | 11.8 | 12.4 / 12.5 / 12.4 / 12.4 / 12.2 / 12.1 / 12.0 | 16.14 → 16.69 | 18.10 |
| ON 32K c=16, 512 tok | 32.2 / 94.29 | 145.8 | 56.2 | 9.9 | 10.8 / 10.4 / 10.2 / 10.2 | 14.66 → 14.91 | 23.25 |

**8K steady step: 53–59 ms ON vs 80–83 ms OFF in the worker.** The end-to-end ratio (×1.43) is larger than the in-process ×1.24 because the OFF worker ran in a slower window (probe TTFT 21.2 vs 16.1 s), so read the in-process number as the code effect. The plan's expectation was 67–80 → 52–66 ms.

**Active memory.** ON active grows in +0.25 GiB steps, once per 256 generated tokens (the stack's cap at 16 rows × 64 KiB/token). That is the expected growth, not a per-step copy. OFF grows similarly with its fp16 KV. ON stays ~3 GiB lower throughout.

**Memory-rule dissolve.** At 1024 tokens the last 3 members were released by the memory rule, a path not seen in G2/G3:

```
cascade: group P=8192 dissolved (memory): 3 rows released, 1.00 GiB charge dropped (reserved 1.20 GiB)
```

Their real suffix (~1100) passed the 3-row break-even (~1025 at 8K). The released rows went on to finish their 1024 tokens with no error.

**32K steady step: 93–98 ms** at 16 rows, matching the in-process 92.5 ms. A per-row 16 × 32K run does not fit for comparison. The plan derived ~110 ms for one, which makes this ~×1.15.

## G5: identity at B=1, cascade ON (`pcache2_v1.py on`)

**20/20 byte-identical** to `raw/pcache2_v1_off.json`, covering text, reasoning, finish and tokens: code0–4 cold + hit, sysA cold/hit, sysB/sysC siblings, and two 3-turn chats. There were 0 cascade lines, since sequential requests are never grouped.

## Recommendation

**Default.** Set `QWEN_CASCADE=1`. Reasons:

- The numerics are in today's batched-decode class: same top-1, KL +0–13 % over the control, per-layer error below the per-row kernel's.
- B ≤ 2 and ungrouped rows are untouched (G5; plan B-T4).
- Shared-prefix waves get faster: ×1.24 per step at 8K/16 and ×1.23 at 32K/8, with ~3 GiB less active memory at 8K/16.
- 16 × 32K becomes admittable.

The rollback stays `QWEN_CASCADE=0`.

**Knobs.**

- **`QWEN_CASCADE_MIN_ROWS=8`: keep.** Measured gains are at B=8 and 16. Below 8 the microbench gains are ≤ +5 % (B=4 at 32K) or negative (≤ 8K), and the memory rule already refuses losing groups. Lower values were not measured on the GPU.
- **`QWEN_CASCADE_MIN_PREFIX=4096`: keep.** The 1K wave never grouped. The 8K and 32K groups pay off.
- **Memory rule** (`_group_wins` on real lengths + the budget guard): **keep as built.** It formed every expected group, and it dissolved a 3-member tail at its break-even with no error. The reservation-based charge stayed far inside the budget: max reserved 5.29 / 15.75 GiB at 16 × 32K.
- **`QWEN_CASCADE_CHECK`: keep off by default.** Every ON run here had it on, and no join ever mismatched. It costs a host sync per join.

**Caveats for CLAUDE.md.**

- The **formation transient** (+P × 64 KiB fp32 while all members still hold fp16 caches) raised the c=8 × 32K peak by 2.2 GB, to 23.1 GB.
- With the cascade on, a B ≥ 3 greedy answer depends on the prompt cache and arrival order (plan B.6). `QWEN_MAX_BATCH=1` or `QWEN_CASCADE=0` restores determinism.

**Open items.**

1. The 32K/16 step is 92 ms vs the projected ~62 ms. Profile the group attention in the model.
2. `QWEN_CASCADE_CHECK` on an entry produced by a DFlash prefill was not tested (review cascade_b2b3), because no drafter was loaded. DFlash rows are never grouped, but a later hit on a DFlash-made entry is.
3. Retrace cost was not measured separately. The step probe's blocks include cap changes, and no spikes were seen.

## State left

**Worker** on :50051: pid 63588, nice 0, started from `src/worker` on the current code with `QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 QWEN_CASCADE=0`. Its log is the scratchpad `cascade/worker_end.log`.

**Gateways and other services.**

- The :8090 gateway is up and untouched.
- The :8080 / :8081 gateways, docker and aria2 were untouched.
- The coordinator's stopgap worker (pid 51549) was stopped with SIGTERM.

## Confirm run: the default (`QWEN_CASCADE` unset = on), 19:46–20:07

**Code.** Current tree, which includes:

- the `QWEN_CASCADE` default of 1;
- the `_group_bytes` floor: a group must afford its forced release into 2 rows;
- optional releases checked against the live group plus prompt-cache bytes, staged one member per step;
- `_charge_fits` counting prompt-cache bytes and refusing instead of evicting;
- no duplicate group on a refused join.

**Worker.**

- Started with **no `QWEN_CASCADE` in the environment**: `QWEN_PROMPT_CACHE=1 QWEN_KV_BITS=0 QWEN_CASCADE_CHECK=1`.
- Its load line shows the default is on: `cascade attention (QWEN_CASCADE): on: >= 8 rows sharing >= 4096 prompt tokens, check`.
- One worker ran the 4 waves in order. G5 ran on a fresh worker.

**Driver.** `run_cascade.zsh "C1 C5 DEFEND"`, same rules as above. Raw data is in `raw/cascade_confirm_*`, including `cascade_confirm_worker*.log`.

**Errors: none.** There were 0 Traceback, ERROR, `ended at`, `RESOURCE_EXHAUSTED`, `not grouped`, `not kept`, `store failed` and HTTP errors. `CASCADE_CHECK` never fired.

| wave | probe decode / TTFT | wall s | agg tok/s | TTFT p50 / p95 s | decode tok/s/req (worker) | peak GB | B4 ON (same wave) |
|---|---|---|---|---|---|---|---|
| 16 × 8K shared, 256 tok | 34.4 / 16.09 | 35.6 | 115.2 | 18.31 / 22.10 | 16.4 (15.9) | 14.96 | 116.5–120.3 agg, 16.4–16.7 dec, peak 14.96–14.99 |
| 16 × 8K shared, 1024 tok | 37.2 / 15.21 | 76.2 | 214.9 | 16.51 / 19.68 | 17.6 (17.4) | 16.06 | 216.2 agg, 17.6 dec, peak 15.31 |
| 16 × 32K shared, 128 tok | 32.8 / 76.18 | 92.0 | 22.3 | 78.98 / 82.20 | 11.1 (10.9) | 23.29 | 22.2 agg, 11.1 dec, p95 82.5, peak 23.24 |
| 16 × 8K **distinct**, 256 tok | 35.9 / 19.54 (slow window) | 381.3 | 10.7 | 183.6 / 353.2 | 4.6 (3.0) | 16.74 | Part A V2 d: probe 13.97 s, 298.5 s, 13.7 tok/s; old V2 d: 18.01 s, 341.9 s, 12.0 |

**Steady 16-row decode, 1024-token wave:** 18.9 / 18.6 / 18.0 / 17.7 / 17.5 steps/s, against B4 ON's 17.3 / 18.7 / 18.3 / 17.7 / 16.9. That is equal or better.

- Active memory went 13.62 → 14.55 GiB, in +0.25 GiB steps per 256 tokens (B4: 13.02 → 13.74, on a fresh worker).
- Here the 256-token wave's entries were still in the prompt cache, which explains the ~0.6–0.8 GiB offset and the 16.06 GB peak.

**Staged memory releases.** Rows finish within ~1 s of each other, so each group shrinks through the floor rule and is released in stages:

| wave | `left (memory)` lines | final dissolve |
|---|---|---|
| 8K, 256 tok | 1 (3 rows stay) | `dissolved (memory)`: 3 rows released |
| 8K, 1024 tok | 2 (4 rows stay, then 3) | `dissolved (memory)`: 3 rows |
| 32K | 0 | `dissolved (memory)`: 3 rows released, 4.28 GiB charge dropped; active 14.67 GB just before |

So 3 staged `left (memory)` lines in total, plus 3 memory dissolves. Active memory at the 32K dissolve was 14.7 GB.

**32K wave.** All 16 rows were admitted; the batch line shows `rows=16 waiting=0 … reserved=6.13/15.75GiB … grouped=16`.

- The group formed at 8 rows: `reserved 9.40 -> 5.17 GiB`.
- The charge is now 4.34 GiB (B4: 2.55), which reflects the new floor. The admitted rows were the same.
- Peak 23.29 GB, the formation transient, as in B4 (23.24).

**Distinct wave.** No group formed: 0 `cascade:` lines and every batch line reads `groups=0 grouped=0`. The prompt cache gave 0/16 hits. The rows ran 3–5 at a time, limited by the budget, as in Part A.

- Its wall time and throughput sit in a slow prefill window (probe TTFT 19.5 s).
- They are in line with the old V2 d run at a similar probe (18.0 s): 381 s scaled by 18.0/19.5 ≈ 352 s, against 342 s.

**G5 identity on the default:** **20/20 byte-identical** to `raw/pcache2_v1_off.json`, with 0 cascade lines.

**Result: confirmed.** The default-on build matches the B4 ON numbers on every shared wave, within drift. It admits all 16 × 32K rows. It forms no groups on distinct prompts. It is byte-identical at B=1. It stages its memory releases without errors.

**State left.**

- Worker on :50051: pid 69748, nice 0. Started with `QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1` and **no `QWEN_CASCADE`**, so the cascade is on by default and `CASCADE_CHECK` is off.
- Gateway :8090 is up.
- Untouched: :8080 / :8081, docker and aria2.

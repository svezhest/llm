# Segment-sharing prompt cache (Part A) on the GPU: identity, shared waves, long context

2026-10-12, 16:30–17:12, M1 Max 32 GB, Ornith-1.5 9B 8-bit, fp16 KV (`QWEN_KV_BITS=0`),
defaults otherwise (max_batch 16, fp16, pfqmm, mrqmm, GDN fuse, no DFlash, 3 GB soft cap).

The code under test is the uncommitted Part A in `src/worker/worker.py` on top of 971a8d5.
Each entry's KV is now a chain of immutable `_Seg` pieces. A store copies only the tokens
past the resume point and reuses the parent's segments below it. A restore concatenates
the pieces once into the row's own buffer. A store also evicts down to the batch's
headroom, or drops the new entry ("not kept"). Admission keeps the entry it would resume
from only when that saves a full slab, and evicts nothing when the eviction would not
admit the request anyway.

The method repeats `prompt_cache_phase1_gpu.md` (the "old" columns below, raw files
`raw/pcache2_*`). It uses the same identity clients (`pcache2_v1.py`, `pcache2_v1x.py`)
and the same `bench_concurrency.py` flags. The driver is
`scripts/perf/final/run_pcache3.zsh`, with the V4 clients `pcache3_v4.py` and
`pcache3_v4b.py`. Raw data is in `raw/pcache3_*`:

- JSON and TXT per phase
- `pcache3_status.txt`: phase timestamps, load, Health after each phase
- `pcache3_worker_{v1,v1x,v2,v4}.log`: one worker log per launch

The gateway on 127.0.0.1:8090 (pid 27313, the phase-1 binary) was kept running.

| launch | worker env (+ `QWEN_KV_BITS=0`) | phases |
|---|---|---|
| W1 16:30 | `QWEN_PROMPT_CACHE=1` | V1 on |
| W2 16:37 | `QWEN_PROMPT_CACHE=1` | V1x on |
| W3 16:39 | `QWEN_PROMPT_CACHE=1` | V2 s1, V2 d, V2 s2, V3 (32K wave) |
| W4 16:54 | `QWEN_PROMPT_CACHE=1` | V4 (120K, repeat, 32K), V4b; **left running** (pid 43620) |

Every launch printed the same load line as phase 1:
`KV budget 15.75 GiB (cap 28.0 GiB, prefill reserve 2.4 GiB) … prompt cache 0.00 GiB live inside the budget (soft cap 3.0 GiB, evicted on demand)`.

**Errors: none.** Across the 4 worker logs there were:

- 0 Traceback / ERROR
- 0 `ended at`
- 0 `store failed`
- 0 `not kept`
- 0 `RESOURCE_EXHAUSTED`
- 0 HTTP errors in any bench phase

The only Generate ids in the logs are `bench-*` and `v1-*`. Usage `cached_tokens` equals the
worker's `reused`/`cached=` on all 26 identity requests and all 64 bench requests.

## V1: hit == cold, byte-for-byte (26/26)

The cache-off references are the phase-1 runs `raw/pcache2_v1_off.json` and
`pcache2_v1x_off.json`. They were reused because the prompts are generated
deterministically by the same scripts, which assert identical messages per case, and the
worker env is the same. Part A does not touch the cache-off path; its only non-cache
change is when `t0`/`queue_ms` are set. V1 and V1x each ran on a fresh worker, as before.
The comparison covers content, reasoning, finish, completion_tokens and prompt_tokens.

| case | `cached_tokens` | store line (new) | result |
|---|---|---|---|
| code0..4 cold / again | 0 / prompt−1 | 53 MB `pieces=1`; repeat `absorbed=1 pieces=1` | 10/10 identical |
| sysA cold / again | 0 / 7404 | 500.2 MB `pieces=1`; repeat `absorbed=1` | 2/2 |
| sysB, sysC siblings | 7359 / 7359 | **501.4 / 501.3 MB `pieces=2`, total +52.7 MB each** | 2/2 |
| chatT 1/2/3 (thinking on) | 0 / 2625 / 2625 | `absorbed=1 pieces=2` on turns 2, 3; entries stay 9 | 3/3 |
| chatN 1/2/3 (thinking off) | 0 / 2625 / 2625 | `absorbed=1 pieces=2`; entries stay 10 | 3/3 |
| V1x P2 cold (12.3K) | 0 | 762.7 MB `pieces=1` | identical |
| P3 (multi-slab suffix, 4917 tok) | 7359 | 923.8 MB `pieces=2`, total +315.6 MB | identical |
| P4 (hit at the 4096 stride ckpt) | 4096 | 777.6 MB `pieces=2`, total +272.5 MB | identical |
| sysA after P2 (P2's system-end ckpt) | 7359 | `pieces=2`, total +53.0 MB | identical |
| P2, P3 exact repeats | 12271 / 12275 | `absorbed=1`, total unchanged | 2/2 |

**Result: 20/20 (V1) + 6/6 (V1x) byte-identical to the cache-off runs.** Hit positions,
`gap` values and absorb counts are the same as in phase 1, line for line.

The per-entry "MB" column counts shared segments in each entry, so it barely changes. The
live total is what changed:

| point | old total MB (entries) | new total MB (entries) |
|---|---|---|
| V1 before the chats (code×5, sysA, sysB, sysC) | 1351 (8) | **869 (8)** |
| V1 end (+ 2 chats) | 1896 (10) | **1415 (10)** |
| V1x end (P2, P3, P4, sysA) | 2020 (4) | **1404 (4)** |

Phase 1 ran V1x on a worker with `QWEN_PROMPT_CACHE_GB=8 QWEN_BATCH_MEM_GB=6`. This run used
the defaults, and that changed nothing in the hits or answers.

## V2: 16 concurrent requests on one shared 8K system prompt

The flags were `--prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 256 --ignore-eos on --prompt-tokens 8000 --probe 64 --cooldown 30`.
The runs went shared → distinct → shared on one worker (W3).

| run | probe decode / TTFT (8.2K cold) | cold first token s (worker ttft) | first tokens span s | TTFT p50 / p95 s | wall s | agg tok/s | decode/req | peak GB | hits |
|---|---|---|---|---|---|---|---|---|---|
| old s1 | 36.0 / 20.46 (slow) | 18.95 (18.61) | 18.95 → 24.55 | 21.23 / 24.11 | 43.1 | 95.0 | 12.3 | 18.27 | 15/16 |
| old s2 | 36.5 / 19.67 (slow) | 18.48 (18.05) | 18.49 → 25.94 | 21.02 / 25.39 | 43.7 | 93.7 | 12.2 | 18.24 | 15/16 |
| **new s1** | 36.9 / 15.27 (**fast**) | 14.68 (14.32) | 14.68 → 20.66 | 16.80 / 20.22 | 37.1 | 110.4 | 13.5 | **17.67** | 15/16 |
| **new s2** | 36.6 / 19.20 (slow) | 17.86 (17.59) | 17.86 → **21.74** | 19.44 / **21.43** | 39.8 | 102.8 | 13.1 | 18.15 | 15/16 |

**Cache bytes: the goal is met.** In s1, the cold request stored 529.3 MB. Each of the 15
hits then added **~54.9 MB**: the KV of its ~100-token suffix plus its own 52 MB end state,
at `pieces=2`. The wave's 16 entries total **1352 MB**, against a plan of ≈ 1.38 GB.

All 16 were kept, along with the calibration and probe entries: 18 entries, 2578 MB,
**0 evictions** (Health). In s2 the wave's 16 entries total ≈ 1320 MB (526.8 + 15 × 52.9).
The 5 Health evictions in s2 were older V2 d / calibration entries LRU-evicted at the 3 GB
store cap, none from the wave. The wave ended at 19 entries, 3078 MB.

The old code kept 9 entries (3.1 GB) after each shared wave and evicted 9 / 15 per wave.
Each sibling held its own ~322 MB copy of the 8.2K prefix.

**Timing.** New s2 and both old runs fell in the same slow prefill window (probe TTFT
19.2–20.5 s), so they compare like for like:

- The hit window (first hit's first token → last first token) is **3.9 s vs 5.6 / 7.5 s**.
- TTFT p95 is **21.4 vs 24.1 / 25.4 s**, and p50 is 19.4 vs 21.0–21.2 s.
- Wall time is 39.8 vs 43.1 / 43.7 s, and aggregate throughput 102.8 vs 93.7–95.0 tok/s.

New s1 ran in a fast window (probe 15.3 s), which explains its lower absolute numbers. Its
hit window was 6.0 s.

The time from the HIT line to the store line was 0.27 s median in s1 and 0.16 s in s2. The
old runs measured 0.26 s and 0.34 s. Restore still writes one full-prefix concatenate per
row, and the store now copies only the suffix.

Steady 16-row decode ran at 14.8 steps/s with 16.31 GiB active (s1), vs 13.0–13.3 steps/s
and 16.86 GiB in the old runs. See the nice note under "Drift" before reading that as a
code effect.

**Peak** was 17.67 / 18.15 GB vs 18.24–18.27 GB.

### V2 distinct (drift control; 3 GB cap)

| run | probe TTFT | first cold prefill (worker) | wall s | TTFT p50 / p95 s | agg tok/s | decode/req | peak GB |
|---|---|---|---|---|---|---|---|
| old V2 d | 18.01 s | 21.46 | 341.9 | 173.9 / 318.6 | 12.0 | 5.2 | 16.81 |
| new V2 d | 13.97 s | 17.07 | 298.5 | 146.8 / 276.1 | 13.7 | 5.6 | 16.74 |

The new run was in a faster window, and the difference matches its probe. There were 0/16
hits, 30 Health evictions at the store cap, and 6 entries / 3.17 GB at the end, the same as
before. For distinct prompts each entry is a single piece, so nothing changes.

## V3: 16 concurrent requests on a shared 32K prefix (the phase-1 "V4 32K")

The flags were `--prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 128 --prompt-tokens 32000`,
and the run used the default worker (W3, after V2).

| run | probe decode / TTFT | cold TTFT (worker) | TTFT p50 / p95 s | wall s | agg tok/s | peak GB | hits | HIT→store s |
|---|---|---|---|---|---|---|---|---|
| old | 32.8 / 79.94 | 75.04 (74.66) | 77.22 / 92.71 | 99.5 | 20.6 | 25.27 | 15/16 | 0.19–0.27 |
| **new** | 33.0 / 78.75 | 74.44 (74.04) | 76.82 / 92.88 | 100.0 | 20.5 | **25.26** | 15/16 | 0.19–0.23 |

**Sibling copies are gone.** Each hit stored a 33.2K entry for **+52.9 MB live**
(`pieces=2`). The old code stored a fresh 1.76 GB copy, and the 3 GB store cap evicted the
previous sibling at every store.

The cache grew 1758 → 2288 MB over the first 11 admissions. The 12th admission evicted the
10 small siblings in one go:

```
16:53:40,742 prompt cache: 10 entries evicted to admit id=…-c16-8 (1.64 GiB left)
16:53:50,093 batch: rows=12 waiting=4 … reserved=14.07/15.75GiB pcache=1.69GiB active=23.40GiB
```

The 3 admissions after the first rows finished evicted one 53 MB sibling each, and the
last admission evicted nothing. That makes 4 `evicted to admit` lines, the same number as
before, but each line now frees 53 MB instead of 1.76 GB. The parent was kept, so every
hit resumed at 33170.

**Expected but not met: "all 16 rows admitted without waiting, lower peak."** Part A shares
the KV inside the cache but not inside the running rows. A restore concatenates the
pieces into the row's own buffer, so every row still holds a full 33K KV copy.

At 33.2K + 128 tokens each row reserves ~1.17 GiB, and 12 rows reserve 14.07 GiB. That
amount plus the parent entry kept for the hits (1.64 GiB) is the 15.75 GiB budget. Even
an empty cache would allow only 13 rows (13 × 1.17 = 15.2).

So the batch again ran 12 rows with 4 waiting (~13 s, until rows ended). The peak is
unchanged at 25.26 GB. It is set by weights + 12 × 1.09 GB of row KV + the 1.7 GB parent,
and Part A does not touch any of those. TTFT and wall time are equal within noise.
Lowering this peak would need rows that read the shared segments in place, which Part A
deliberately does not do: "nothing ever holds a view of a segment".

## V4: long-context store bound (120K)

The client was `pcache3_v4.py`: sequential streamed requests, temperature 0, thinking off,
ignore_eos, max_tokens 32. It sends no calibration request, because the bench's would
itself prefill 120K. A watcher was set to cancel the client if any `active=` in the log
reached 26.5 GiB. The highest batch-line `active` in W4 was **17.15 GiB**, so it never fired.

| step | prompt | TTFT | cache | per-request peak | Health after |
|---|---|---|---|---|---|
| L1 cold | 118747 | 389.2 s | MISS → **kept**: store 118746 tok, 12 ckpts, 4560.7 MB (4.25 GiB) + warning "newest entry alone … over QWEN_PROMPT_CACHE_GB=3.0; kept, evicted on demand" | 21.31 GB | 1 entry, 4.56 GB |
| L2 same prompt | 118747 | **0.51 s** (worker 0.22 s) | **HIT 118746/118747**, prefilling 1; store `absorbed=1 pieces=1` (total unchanged) | 18.05 GB | 1 entry, hits 1 |
| M1 other 32K | 31659 | 80.9 s | MISS (1 entry resident), admitted at once into the empty batch; its store (1707 MB) LRU-evicted the 120K entry at the 3 GB soft cap | **19.53 GB** | 1 entry, 1.71 GB, evictions 1 |

- **"Not kept: over the batch's headroom" did not fire, as expected.** At store time
  the limit is budget + prefill reserve − reserved ≈ 15.75 + 2.4 − 3.9 (the 120K row's own
  reservation) ≈ 14.2 GiB, and the entry is 4.25 GiB.
- The task estimated a "~4.9 GB first store at 128K". The measured store is 4.56 GB at
  118.7K tokens: 3.89 GB of KV plus 12 checkpoints and the end state.
- **No OOM risk seen.** The peaks were 21.3 / 18.1 / 19.5 GB, against a 28 GiB cap.

**V4b: forcing the on-demand path with the 120K entry resident (`pcache3_v4b.py`).** A fresh
120K cold request was stored the same way (120355 tokens, 4613 MB, kept with the warning).
Its peak was 23.20 GB, because the previous 1.7 GB M1 entry was still resident during that
prefill.

The client then sent 10 concurrent distinct 35.3K-token prompts with max_tokens 4096
(~1.36 GiB reservation each). The aim was for the 9th admission to need the 120K entry's
space. That did not happen, because admission waits for the prefill slot:

- The worker admitted M0 and held the other 9 in `waiting=9` for M0's 89 s prefill, with
  `reserved=1.36 pcache=4.30` of 15.75 GiB.
- M0's store then LRU-evicted the 120K entry at the 3 GB soft cap.
- M2 was admitted next, next to the decoding M0, with the cache at 1.70 GiB.

The clients disconnected after 150 s and the gateway sent 10 Cancels. The worker went idle
(Health ready, 0% CPU, 1 entry / 1.82 GB), with no error lines. With the default soft cap, an over-cap
entry therefore survives only until the next store. Any request that does not resume
from it replaces it before admission pressure can build. The on-demand eviction of a
large entry is reachable only with a bigger `QWEN_PROMPT_CACHE_GB`. Phase 1's V4c tested
that path with 8 GB; it was not repeated here.

## Drift and caveats

- **Nice level.** The phase-1 worker that this run replaced (pid 31776) was running at
  **nice 5**. That worker was the R7 launch. The R1–R6 workers were probably niced the same
  way, because the driver was started as a background job. Every worker in this run
  started at nice 0 (checked at launch). The 16-row decode gain (14.8 vs 13.0–13.3
  steps/s) and the higher aggregate tok/s may be partly or wholly this, not Part A. The
  cache bytes, eviction counts and identity results do not depend on it.
- **Prefill drift.** The probe TTFT at 8.2K was 14.0–15.3 s (V2 s1, V2 d) and 19.2 s (V2 s2).
  The comparisons above pair windows by probe. Decode was stable: probe 36.6–37.2 tok/s at
  8K, and 33.0 at 35K.
- **Another agent's CPU tests.** Mutation tests (`pytest tests/test_segments.py` over
  `scratchpad/mut/m*`, plus the CPU suite) ran 16:31:56–16:37. The driver's check held
  V1x back until they were gone. The check passed before every later phase, but CPU
  activity by others during a phase cannot be ruled out. The load average was 2.2–2.8
  throughout, with no `pmset -g therm` warnings.
- Each configuration ran once per window, so differences of ±10% are drift.

## State left

- Worker on :50051: pid 43620, nice 0, Part A code,
  `QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1`.
  Its log is in the scratchpad at `pcache3/worker_v4.log`, with a snapshot in
  `raw/pcache3_worker_v4.log`. It holds 1 entry (1.82 GB).
- Gateway on 127.0.0.1:8090: pid 27313, untouched.
- Nothing else was touched: the :8080/:8081 gateways, docker and aria2.

# Decode attention at long ragged contexts: profile + ragged multi-row kernel (rattn)

16.10.2026, M1 Max 32 GB, mlx 0.32.3 / mlx-vlm 0.7.6, Ornith-1.5 9B 8-bit, `QWEN_KV_BITS=0`, HEAD 34b4a81.
The question came from `HANDOFF-appworld.md`: on the agentic load (8–16 rows with distinct 5–33K contexts and
short answers), how much of a decode step is per-row attention, and does a single ragged multi-row kernel pay?

**Answer.** Attention is 25–38 % of a clean B=8 step at 12–33K. MLX's per-row 2-pass `sdpa_vector` is already
within about 1.5× of the achievable read floor (~340 GB/s measured, not the nominal 400). The ragged kernel is
correct and more accurate than MLX's own kernel, but in-step it gains only **−2.3 ms (−3.5 %) at B=8 ragged**.
That is below the 10 ms gate, and even a perfect kernel can reach only about −7 ms there. At **B=16 ragged** it
gains **−13.9 ms (−12 %)**. Phase 3 (worker integration) is **not done**, as the gate prescribes; see "Left / options".

Raw logs and JSON are in `rattn_raw/`. Tools:

* `scripts/perf/prof/step_prof.py`, with `--ctx Nk|ragged:…`, `--common`, `--stagger` and the attention variants
  `base / nosdpa / noattn / rattn[/s]`;
* `scripts/perf/microbench/microbench_rattn.py`;
* the kernel `src/worker/rattn.py`;
* its Metal tests `src/worker/tests/test_rattn.py` (17 passed).

## Phase 1: decode step profile at long / ragged contexts

The profile runs the real `Engine` + `_Scheduler` in-process, with no gRPC. The rows' prompts come from one
shared filler text, so the longest row prefills cold and the others resume from the prompt cache. Each row still
owns its own KV buffers. Settings:

* env `QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=6 QWEN_CASCADE=0 QWEN_BATCH_PREFILL_SHARE=1`;
* greedy decoding, `--ignore-eos`.

The variants alternate every 10 decode steps inside one batch (in-run A/B, the 2 steps after a switch are dropped).
Each figure is the median step period, n = 27–70:

* `base` is the worker's step.
* `nosdpa` keeps the per-row `update_and_fetch` and skips the SDPA.
* `noattn` skips both.
* `/s` adds a sync after every step, which gives GPU time.

Floor = fp16 KV read per step (8 FA layers × 4 kv heads × 256 × 2 B × K+V = 32 KiB/token) / 400 GB/s.

| ctx (B=8 unless noted) | KV/step | floor @400 | base | nosdpa | noattn | **SDPA (share)** | append | base/s | nosdpa/s | noattn/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ragged 5–33K (run 1, slow state) | 4.80 GB | 12.0 | 106.8 | 71.0 | 71.9 | **35.8 (34 %)** | ~0 | 114.3 | 74.8 | 72.1 |
| ragged 5–33K (run 2, 03:45) | 4.80 GB | 12.0 | 66.0 | 44.9 | — | **21.1 (32 %)** | — | | | |
| ragged ×2, **B=16** (run 1) | 9.59 GB | 24.0 | 155.1 | 108.8 | 97.3 | **46.3 (30 %)** | 11.5 | 162.0 | 106.2 | 92.7 |
| ragged ×2, **B=16** (run 2) | 9.59 GB | 24.0 | 117.2 | 79.6 | — | **37.6 (32 %)** | — | | | |
| 12K | 3.16 GB | 7.9 | 65.8 | 49.3 | 49.2 | **16.5 (25 %)** | ~0 | 74.2 | 54.7 | 51.8 |
| 20K | 5.25 GB | 13.1 | 72.2 | 49.0 | 48.6 | **23.2 (32 %)** | ~0 | 77.6 | 52.6 | 54.8 |
| 30K | 7.88 GB | 19.7 | 85.4 | 53.1 | 47.2 | **32.2 (38 %)** | 6 | 95.2 | 60.2 | 53.6 |

Readings:

* Compare these with 2026-10-12 (1K: 3.7 % at B=8, 8K: 16 % at B=8 / 27 % at B=16): the attention share keeps
  growing with context. Per-row attention is the largest single item of the step that depends on context. Everything
  else (`noattn`, ~47–49 ms at B=8 in the fast state) does not depend on context.
* **Machine drift.** Ragged run 1 (02:57) ran in a slow state: `noattn` took 72 ms there against 47–49 ms in the
  uniform runs minutes later. Run 2 (03:45) repeated ragged B=8 at 66 ms per step. Only ratios within one run count.
* **Clean step vs production.** A clean B=8 ragged step is **66–107 ms**, not the ~260 ms seen in production. The
  ×2.5–4 gap comes from outside the GPU step: swap and the memory compressor (worker CMPRS 8 GB, see the handoff)
  and prefill interleaving. Attention explains at most ~20–35 ms of the 260.
* **Per-row append.** The append (2 SliceUpdates per row and layer) is free at B=8. At B=16 it costs ~11 ms in the
  async run but is within noise in the sync runs. It is not a lever on its own.
* Host side: `fwd_cpu` 2.5–3.9 ms per step, token wait ~0. The step is GPU-bound.

## Phase 2: ragged multi-row decode attention (`src/worker/rattn.py`)

### Design

* **Today, per FA layer and per row:** `update_and_fetch`, then MLX `sdpa_vector_2pass` (q (1,16,1,256) vs the
  row's (1,4,L,256)). On M1 Max (`applegpu_g13s`) with GQA 4, `n_simds` = 4. That fixes `blocks` = 64 for every L
  (the `N > 1024 && n_simds > 4` branch is never taken; source `mlx/backend/metal/scaled_dot_product_attention.cpp`
  v0.32.3, `sdpa_vector_2pass`). As a result:
  * each of the 64 threadgroups walks L/64 strided keys;
  * each threadgroup's 4 simdgroups (one per query head) all load the same keys;
  * partials are kept in fp16;
  * the `_gqa_` duplication-free variant only exists for G 8/12/16 at D ≤ 128.
* **rattn: one launch for all rows of the step, plus a tiny merge.**
  * Each row's own KVCache buffers (the full `(1,4,cap,256)` arrays after `update_and_fetch`, not the views) are
    separate kernel inputs `kb0..kbN` / `vb0..vbN`, selected by row index with a switch in MSL. There is no copy
    into a stacked buffer.
  * A host-built item table (row, key range) splits each row into chunks of 512 keys, so a 33K row and a 5K row
    cost exactly their keys. The table is cached per (lens, caps) and shared by the 8 layers of a step.
  * Pass 1 (items × 4 kv heads threadgroups of NS=4 simdgroups): each simdgroup takes a contiguous run of keys and
    computes all 4 query heads of its kv head. It reads K/V once, as MLX's `_gqa_` kernel does: 4 independent
    `simd_sum`s per key, online softmax in the log2 domain, fp32. The simdgroups merge through threadgroup memory.
    The pass writes normalized fp32 partials plus (m, l).
  * Pass 2 (rows × 16 threadgroups of 256 threads): LSE merge of the row's chunks, written as fp16.
* **Metal buffer limit.** Metal allows 31 buffers per kernel, so one launch takes at most 8 rows (2·8 + 5 buffers).
  B=16 is 2 launches (+ concat) per layer, against 32 SDPA kernels + 32 appends today.
* **The append stays per row.** `update_and_fetch` is in place and nearly free (Phase 1). A `metal_kernel` cannot
  write into its inputs, so the kernel cannot append. It reads the buffers after the append.
* **Extra memory:** the partials, n_items × 16 × 256 × 4 B (B=8 ragged: ~4.7 MB per layer, transient). Nothing else.

### Microbench (synthetic 9B shapes, random K/V, 8 layers per step)

`microbench_rattn.py`, B=8 ragged 5–33K, B=16 = the ragged set twice, B=8 uniform 12K.

Achievable read rate on this machine (`bwprobe`): 1 GiB fp16 sum 339 GB/s, max 337, copy 356 GB/s (r+w). The
realistic floor is therefore 4.78 GB / 340 GB/s = **14.1 ms** for B=8 ragged and **28.2 ms** for B=16.

Kernel only (no append), pipelined (mb1, round medians):

| config | MLX per-row SDPA | rattn 512:4 | rattn 256:2 | realistic floor |
|---|---:|---:|---:|---:|
| B=8 ragged | 20.6 ms | 17.2 | **16.0** | 14.1 |
| B=16 ragged | 37.5 ms | **31.8** | 33.3 | 28.2 |

Two more configurations measured kernel-only:

* With a sync per layer (mb2, which also measures the append): B=8 ragged 20.5 / 18.6; B=16 40.3 / 34.3; u12
  16.4 / 13.3 ms. A plain sum over the same KV runs 18.1 / 33.6 / 13.7 ms, i.e. rattn reads at the plain-reduction
  rate.
* Chunk/NS sweep: 256–1024 keys × NS 2/4 are all within ±5 %.

Microbench artifact: the rows' caches are shared by the 8 simulated layers. Without a sync per layer, layer j+1's
append found layer j's buffer still referenced and copied the whole buffer (peak +5 GB, 46 ms; mb1 `rattn`/`rows`
rows). The model's layers each have their own cache, so the real step is not affected. With the sync per layer
(mb2), append + kernel takes:

| config | today (append + SDPA) | append + rattn | append alone |
|---|---:|---:|---:|
| B=8 ragged | 22.1 | 21.6 | 4.9 |
| B=16 ragged | 43.4 | 39.3 | 8.4 |

**Numerics** (one layer, vs an fp32 reference; flat q ~ N(0,1) / peaky q × 8):

| | rel rms | max abs |
|---|---:|---:|
| MLX per-row SDPA | 2.81–2.95e-4 | 1e-5 / 7.8e-4 |
| rattn | **2.08–2.09e-4** | 7.6e-6 / 4.8e-4 |

* rattn vs MLX: max |Δ| 1.5e-5 (flat) and 9.8e-4 (peaky), i.e. about 1 fp16 ulp.
* rattn's error is below the per-row kernel's own (the microbench_cascade criterion), thanks to fp32 partials
  against MLX's fp16 partials.
* `tests/test_rattn.py` (Metal) covers lengths 1, chunk−1, chunk, chunk+1, garbage (1000 / NaN) past each row's
  length, 11 and 16 rows (multi-launch), GQA 8 and 4 chunk/NS configs: 17 passed.
* An earlier run of the file by another agent hit `RuntimeError` at ≥ 5 rows: the input names `k4`/`v4` clashed with
  the kernel's local `k4`/`v4` pointers. The inputs were renamed `kb*`/`vb*`, fixed.

### In-step A/B (the decisive number)

`step_prof.py --ctx ragged:5k,…,33k --ab base,rattn,nosdpa --ab-steps 10`, real model, same batch alternating,
n=70 each (`rattn_raw/p2_instep.log`):

| B | base | **rattn** | nosdpa | attention today → rattn | Δ step |
|---|---:|---:|---:|---:|---:|
| 8 ragged 5–33K | 66.0 ms | **63.7** | 44.9 | 21.1 → 18.8 ms | **−2.3 ms (−3.5 %)** |
| 16 ragged | 117.2 ms | **103.3** | 79.6 | 37.6 → 23.7 ms | **−13.9 ms (−11.9 %)** |

* **B=8.** The gain matches the kernel microbench (−2…−4.6 ms). At this occupancy MLX's per-row path is close to
  the read rate. A perfect kernel would bring attention to ~14 ms, so the most any kernel can gain is **~7 ms < the
  10 ms gate**.
* **B=16.** The gain is larger than the kernel-only difference (−5.7 ms). The per-row path issues 256 SDPA kernels
  + 256 appends per step, and with MLX's 50-ops-per-command-buffer / 10-buffers-in-flight limit the host leaks
  into the step (step_profile.md, "host gap at B=16"). rattn issues 32 launches + concat. At B=16, rattn's
  attention (23.7 ms) is below the 28 ms "realistic floor", because the plain-sum floor itself is not perfectly
  bandwidth-bound.

## Phase 3: not done (gate)

The gate was "≥ 10 ms per step at B=8 ragged ~12K". It is not met (−2.3 ms measured, ≤ ~7 ms possible), so
nothing was integrated into `worker.py`, and CLAUDE.md, BACKLOG and the env knobs are untouched. The kernel, its
Metal tests and the profiling tools are in the repo (uncommitted).

## Left / options

1. **The production step is ~260 ms, not 66–107.** The ×2.5–4 gap is outside the GPU step: swap, the memory
   compressor and prefill interleaving. Fix the handoff's memory settings first: `QWEN_PROMPT_CACHE_CKPTS=3`,
   `QWEN_PROMPT_CACHE_GB=6`, `QWEN_BATCH_RESERVE_TOKENS=1024`. Then re-measure the live step (CMPRS ≈ 0). That is
   worth far more than any attention kernel at B=8.
2. **If the load runs 12–16 rows, rattn pays: −12 % of the step at B=16 ragged.** The integration would be:
   * a `QWEN_RATTN` (default 0) branch in `_rows_attention` for B ≥ 3, fp16 `KVCache` rows, D=256 (cascade
     groups keep their path; in `_cascade_rows` only the singles);
   * kernel warm-up at load for launch sizes 1..8;
   * a CPU dispatch test with the kernel monkeypatched;
   * an agree.py gate at B=8/16 on ragged long contexts.
   The numerics class is better than today's kernel (rel rms 2.1e-4 vs 2.9e-4), but not bit-exact: B ≥ 3 is not
   bit-exact anyway (mrqmm). About half a day. The handoff's 16-process run hit the prompt-cache wall first, though.
3. **Kernel headroom** at B=8 is ≤ ~4.7 ms (18.8 → ~14), from unrolling 2 keys per iteration or a lane-per-key
   score phase. It is not worth it alone.
4. **The B=16 host gap is the real B=16 lever.** Fewer kernels per step pays more there than bandwidth does. The
   same argument applies to the per-row appends (B=16: ~8–11 ms, 256 SliceUpdates per step). A batched append
   would need the KV in one buffer, which is not the case with today's per-row caches.

# Batch-invariant decode (`QWEN_BATCH_INVARIANT`)

2026-10-24. Item 1 of `../deepinfer_ideas.md`. CPU work plus light Metal tests on tiny shapes; the
real-model GPU phase is the plan at the end (its results go in "GPU results").

**Goal.** With T = 0 an answer must not depend on what else the worker is decoding. AppWorld
comparisons of prompt / context methods need runs that reproduce under load. Today that means
`QWEN_MAX_BATCH=2`, which costs 11 % of the requests per minute and moves p50 latency from 15 to 22 s
(`../../2026-10-17/replay_batch1.md`).

**What the mode guarantees.** A row's logits are the same bits whatever the batch around it: 1..16
rows, any prompt lengths, rows joining and leaving. So a row's tokens and log-probabilities equal
those of the same request run alone. This holds for the 9B layout (all decode linears 8-bit g64 with
fp16 scales). **Real 9B, 2026-10-24: PASS** (every batch size 1..16, 0 mismatches in 5947 forced
steps and two free runs; the mode off: 16/16 rows differ). Cost: B = 1 ×1.09–1.21, B = 2 ×0.86,
B = 4/8 ×1.00, B = 16 ×0.97 (see "GPU results").

## What depended on B, and what the mode does

| source | default | under `QWEN_BATCH_INVARIANT=1` |
|---|---|---|
| mrqmm K split | KS 8 at 3..8 rows, KS 4 at 9..16 (another sum order) | KS 8 for every row count (`mrqmm._config`) |
| B ≤ 2 | MLX qmv (its own rounding) | mrqmm for 1..16 rows (`mrqmm._use`): plain linears, lm_head, the fused GDN in-projection (`_gdn_call`) |
| narrow 8-bit layers (N < 512: GDN b/a unfused, 35B router) | quantized_matmul (qmv, then qmm from MLX's row threshold) | mrqmm (install ignores `MIN_N`) |
| cascade groups | another attention composition | off (`_cascade_on`, logged "QWEN_BATCH_INVARIANT=1") |
| DFlash | lone greedy requests, verify kernels ≠ decode | the drafter is not loaded; `_spec_ok` refuses too |
| B > 16 | qmm | `max_batch` capped at 16 (`mrqmm.MAX_ROWS`) |

**Why only KS matters (read off `mrqmm._SRC`).** Take one output (n, m):

- Simdgroup `ks` adds its groups `g = ks, ks+KS, …` in order. Each group is 8 MMA steps of 8 k, and
  the k permutation is the same for every row. All of it goes into one fp32 accumulator `C[i][mb]`.
- The reduction then adds simdgroups 1..KS−1 to simdgroup 0's sum, in that order (`acc += red[s]`).
- NA only picks which simdgroup and fragment hold n. MB only picks which 8-row bucket holds m.
  - The MMA result of (n, m) reads only A's row n and B's column m.
  - Padded columns m ≥ M are computed from row 0 and never stored.
  - The dequantized weight is a per-n fma.
  - The accumulators of different (i, mb) never meet.
- So with KS fixed per K, row m's bits do not depend on M or on the other rows. The Metal test checks
  this with `==` at the 9B shapes.

**Choice of KS 8.**

- At 3..8 rows it is already the default (2 × 8). The mode changes nothing there: the bits at
  B = 3..8 are today's bits (`test_k_split_fixed_per_k` asserts the config is equal).
- At 1–2 rows: mrqmm was ~1.03× a qmv step at M = 1 (31.3 vs 30.3 ms of step linears,
  `../../2026-10-12/mrqmm.md`); measured now (GPU results) the real step is ×1.09–1.21 at B = 1,
  and ×0.86 at B = 2 (2-row qmv reads the weights twice).
- At 9..16 rows: KS 8 with NA 2. The config sweep at 16 rows had put 2 × 8 within 3 % of the default
  4 × 4; in the real step it is 2.6 % *faster*. NA is free (it does not change a bit); the GPU
  kernel bench measured 4 × 8 as 5–6 % slower at 9..16 rows, so `INVARIANT_NA16` stays 2.
- KS 4 everywhere would instead change the 3..8-row bits and slow the bandwidth-bound small-M case
  (half the simdgroups per tile).

**Checked to be per-row already.** On the GPU, by the tiny-model scheduler test with `==`:

- the B = 1 library call (`lm(r.y[None], cache=r.cache)`) vs the RowsKV step. Both pass explicit
  positions; `_prepare_projected_qkv` with or without a cache does the same per-row
  `update_and_fetch` + SDPA call, and mask None at L = 1;
- RMSNorm and the Metal RoPE kernel;
- the stacked GDN: the compiled depthwise conv, the gated-delta kernel (one threadgroup per row ×
  head), and the compiled g/beta;
- the logits slice `lg[i:i+1]`;
- `sample_rows` vs `_sample` (greedy, and a seeded T = 0.8 row);
- `_token_logprobs` with top-k.

**Not on the CPU.** The CPU's vectorized / compiled fp16 elementwise kernels round an element by its
position in the array: mlx-vlm's compiled `_compute_g_beta` gives row 1 of a 4-row stack different
bits from that row alone. So bitwise batch == solo is a GPU property. The CPU scheduler test checks
the harness with a tolerance.

**Not covered (logged at load: `batch-invariant: N modules not covered (…)`, `_invariant_uncovered`
→ `mrqmm.uncovered`).** Rows at B ≥ 3 through these can still differ from solo. At B ≤ 2 they are
exact (qmv per row).

- non-8-bit layers: the 27B (5-bit, every linear), and on the 35B the 3/4/6-bit experts, the shared
  expert and lm_head;
- N % 8 layers (35B `shared_expert_gate`, N = 1);
- unquantized `nn.Linear`;
- MoE experts (gather_qmm; mlx-vlm's `SwitchGLU` also sorts by expert from `B·top_k ≥ 64`);
- a tied lm_head (`embed_tokens.as_linear`);
- with `QWEN_MRQMM=0` or no Metal: everything ("8-bit g64 mrqmm off");
- 9B: nothing (the test `test_uncovered_9b_layout` on its layout, and the checkpoint's tensors: all
  linears are 8-bit g64, lm_head untied, b/a inside the fused in-projection).

**Load-time guard (`_invariant_check`, review 2026-10-24).** It runs before the scheduler is built.

- If anything is uncovered, `max_batch` drops to 2 with a warning (`… max_batch 16 -> 2`). The
  reason: quantized layers outside mrqmm run one qmv per row at B ≤ 2, the unsorted expert gather
  applies there too (B·top_k < 64), and mrqmm gives 1 and 2 rows the same bits.
- If there is an unquantized `nn.Linear`, the mode is refused: an error in the log and the mode
  off. MLX's 1-row gemv against a 2-row gemm has no exactness rule.
- The real max_batch is logged after `fit_max_batch`.
- The exactness of 3/4/5/6/8-bit qmv at 1 vs 2 rows is covered by
  `test_uncovered_qmv_two_rows_exact` (Metal, `BATCH_INV_GPU=1`). It was not run yet: the review
  fixes were CPU-only. Until it passes on the GPU, the 27B / 35B guarantee at B ≤ 2 rests on
  CLAUDE.md's "B ≤ 2 always bit-exact (per-row qmv)".

**Still run-dependent (documented, not changed).**

- Reply KV (`QWEN_PROMPT_CACHE_REPLY=1`): a hit through a reply is not a cold prefill's bits, and
  whether it hits depends on evictions (timing). Under the mode the reply's KV is batch-independent,
  but reproducible runs still want `REPLY=0`. The load log warns.
- `_regrow`'s `length` cut of a row that outgrows the budget depends on its neighbours.
- Prefill was already batch-independent: one request at a time, slab cuts fixed by `_bounds`, hit ==
  cold. A 1-token slab now takes mrqmm too; that is still deterministic.

## Knobs

- `QWEN_BATCH_INVARIANT` (worker, default `0`). `1` = the mode.
  - `0` is today's path bit for bit: no install-time change.
  - `MRQuantizedLinear` checks `N ≥ MIN_N` per call, so an install made in the mode and then toggled
    off restores the speed rules exactly. `step_ab.py` uses this.
- `mrqmm.set_invariant(on)`: the switch (worker at load; tools between steps on the scheduler
  thread; it clears the call cache).
- `mrqmm.INVARIANT_KS` (8) and `INVARIANT_NA16` (2) are module constants, not env.
- `_Req.golden` (tools / tests, not on the wire): decode teacher forcing.
  - The k-th generated token is `golden[k]`; the drawn one is dropped. Past the list's end the draw
    goes on.
  - The logprobs are those of the forced token under the forced history. It uses the same hook as
    the thinking budget's forced token (`_iterate`, `_forced_logprobs`, kept log-softmax row).
  - DeepInfer `golden_token_ids` (`sampling_params.py:60,83`, `layers/sampler.py:482-490,650-683`).

## Code

- `src/worker/mrqmm.py`: `INVARIANT`, `INVARIANT_KS`, `INVARIANT_NA16`, `set_invariant`, `_config`
  (fixed KS), `_use(x, n)` (1..16 rows, any N), `install` (any N in the mode), `uncovered(model,
  skip)`.
- `src/worker/worker.py`:
  - the `BATCH_INVARIANT` comment block;
  - `_invariant_uncovered`, `_log_invariant`;
  - `_cascade_on` / `cascade_state`, `_spec_ok`;
  - `Engine.load`: DFlash skipped, `set_invariant`, max_batch cap, coverage log;
  - `_Req.golden`, `_iterate` / `_step` (golden forcing; keep the log-softmax row).

## Tests

- `tests/test_batch_invariant.py` (CPU, in the normal suite, ~1 s):
  - the K split per K on and off;
  - the `_use` dispatch for rows 1..17 × N × prefill / bf16 shapes;
  - install + `uncovered` (5-bit, N % 8, bf16, `nn.Linear`, `QuantizedSwitchLinear`, tied head;
    narrow layers swapped only in the mode, and off again per call);
  - the 9B layout's coverage;
  - cascade and DFlash off;
  - golden tokens (forced, logprobs == teacher forcing within the kernel tolerance, the draw resumes
    after the list);
  - batch-of-4 vs solo with golden (tokens equal, logprobs within 1e-2: see "Not on the CPU").
- `tests/test_batch_invariant_metal.py` (GPU, light: < 0.3 GB, ~6 s once compiled), run only with
  `BATCH_INV_GPU=1` like `PFQMM_GPU=1`:
  - mrqmm at the 9B (K, N): q 8192, k/v 1024, o / GDN out 4096, gate/up 12288, down 12288→4096,
    fused GDN 12352, b/a 32, and N 32768 standing in for lm_head's tiles. For M = 1..16, row i ==
    row i alone, also with different companion rows;
  - the default mode is not invariant (8 vs 9 rows differ);
  - the scheduler on conftest's tiny model, 8-bit fp16, mrqmm in the mode + fused GDN on mrqmm,
    Metal RoPE: 5 requests (mixed lengths 9..31, staggered ends, one sampled, golden or not) at
    max_batch 4 and 2. Every row's (tokens, logprobs, end) == its run alone at max_batch 1;
  - the same with the mode off: some row differs (the test has teeth: verified, every row differs).
- Results (2026-10-24, this machine, GPU shared):
  - `test_batch_invariant_metal.py`: 14 passed;
  - `test_mrqmm.py`: 139 passed, 1 skipped;
  - CPU suite (`--ignore tests/test_mrqmm.py`): 443 passed, 63 skipped (7 min 20 s).

## GPU plan

One command, with the GPU lease (no worker / gateway / other GPU job; Activity Monitor closed). It is
`setopt NO_BG_NICE` inside:

```zsh
zsh scripts/perf/batch_invariant/run_gpu.zsh          # steps 1 2 3 4; PERF_OUT=/tmp/perf/batch_invariant
```

Outputs `$PERF_OUT/{bi_kern,bi_metal_tests,bi_check,bi_step_ab}.{txt,json}` and the summary lines in
`$PERF_OUT/summary.txt`. A single step: `run_gpu.zsh 3`. Worker env for steps 3–4:

```
QWEN_MODEL=weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=0
QWEN_PRESENCE_PENALTY=0 QWEN_BATCH_INVARIANT=1 QWEN_MAX_BATCH=16 (+ QWEN_BATCH_PREFILL_SHARE=1 for 4)
```

The other knobs stay at their defaults (FP16, PFQMM, MRQMM, GDN_FUSE). The prompt cache is off: no
cascade, no reply KV, so the A/B isolates the matmul dispatch.

### 1. Kernel cost per shape (`kern_bench.py`)

- **Setup.** No model; random 8-bit weights at the 9B shapes. Each call is a chain of distinct
  weight copies ≥ 256 MB per shape, so the weights stream from DRAM. 5 interleaved rounds, medians.
- **Variants.** qmv / def (today) / inv (the mode) / inv48 (NA 4 at 9..16 rows, bit-identical to
  inv), at rows 1, 2, 3, 4, 8, 9, 12, 16. The output also has step linears (sum × layer counts):
  default dispatch vs the mode.
- **Expected.**
  - rows 1: inv ×1.00–1.05 (mrqmm.md: 31.3 vs 30.3 ms);
  - rows 2: inv < 1 (qmv ×1.17);
  - rows 3..8: ×1.00 (same config);
  - rows 9..16: ×0.97–1.05.
- **Decision.** If inv48 beats inv by > 2 % at 9..16 rows, set `INVARIANT_NA16 = 4`. It changes no
  bit; re-run step 2.
- **Cost.** ~2 min, peak ~1.4 GB (the lm_head copy 1.07 GB).

### 2. Metal tests

`BATCH_INV_GPU=1 pytest tests/test_batch_invariant_metal.py tests/test_mrqmm.py`.

- **Pass:** all pass.
- **Cost.** ~30 s, < 0.3 GB.

### 3. Real 9B: a row in a batch == the row alone (`check_real.py`)

- **Prompts.** 16 prompts: chat-rendered filler of 300 / 800 / 1500 / 3000 / 6000 tokens, cycled,
  thinking off. Greedy, ignore_eos, 96 tokens, logprobs + top-5 per step.
- **ref:** each prompt alone (B = 1). Then 4 of them again alone with golden = their own tokens:
  they must be equal (solo determinism; golden is a no-op on the model's own answer).
- **forced B = 2, 4, 8, 16:**
  - prompts rotated per phase (another neighbour and batch index each time);
  - max_tokens staggered by 7 (the batch shrinks to the B = 1 library call);
  - half submitted after the first half has decoded 8 steps (joins with interleaved prefill);
  - every row forced with its ref tokens.
  - **Pass:** every step's (log p, top-5) == ref's.
- **free:** all 16 at once (staggered), no golden. **Pass:** tokens and logprobs == ref.
- **control:** the mode off (`set_invariant(False)` while idle), forced at B = 16. **Must** show
  mismatching rows: KS 4 at 9..16 rows, qmv at the tail B ≤ 2.
- **Overall pass:** the script prints PASS and exits 0.
- **Cost.** ~8–10 min (prefill ~78 K tokens ≈ 2.5 min per pass over the prompts, 4 passes).
- **Peak.** ~14 GB: weights 9.5 + 16 rows × ≤ 6.1 K × 32 KiB + GDN.
- **If it fails.**
  - The first mismatches are in the JSON (prompt, slot, step, ref vs got).
  - A mismatch only in the free phase, at a step after the first token difference, is impossible
    when forced passed: it would be a sampler / logprobs path.
  - A forced mismatch from step 0 points at prefill or the first decode step. Bisect with the tiny
    test: B = 1 vs RowsKV on the real shapes.

### 4. Real 9B: decode step cost (`step_ab.py`)

- **Method.** In one process the mode is toggled every 16 decode steps on the scheduler thread.
  B = 1, 2, 4, 8, 16 rows of ~600-token prompts, ignore_eos, 512 tokens. 3 rounds with the B order
  alternating.
- **Measure.** The period between decode ticks (the GPU step with the step-ahead). The first 2
  steps after a switch and the first 2 windows of each run are dropped (the default configs compile
  there). Medians per variant and round; overall the median of the per-round ratios on/off. This
  follows CLAUDE.md "Measuring performance": an A/B inside one process, interleaved, medians over
  ≥ 3 rounds.
- **Expected (from step 1 and `mrqmm.md`).**
  - B = 1: ×1.00–1.04 (+≤ 1 ms of a ~27 ms step);
  - B = 2: ≤ ×1.00 (mrqmm beats 2-row qmv);
  - B = 4, 8: ×1.00 ± 0.02 (identical kernels: this is the noise floor of the method);
  - B = 16: ×0.97–1.05.
- **Pass:** B = 1 ≤ ×1.05, B = 2 ≤ ×1.02, B = 4/8 within ±2 %, B = 16 ≤ ×1.05.
  - Fail on B = 1 / 16: retune `INVARIANT_NA16` (16) per step 1. KS cannot change without moving
    the 3..8-row bits.
- **Cost.** ~7 min, peak ~11 GB.

### After the GPU phase

Record PASS/FAIL and the four tables under "GPU results" below. Then the AppWorld replay at
`QWEN_MAX_BATCH=16 QWEN_BATCH_INVARIANT=1 QWEN_PROMPT_CACHE_REPLY=0` vs `MAX_BATCH=2`, two runs each
(`scripts/perf/agentic/run_replay_batch.sh`), for req/min and reply identity
(`batch_compare.py`). That is a separate ~1 h session, not part of this plan.

## GPU results

2026-10-24, 02:00–02:40, M1 Max, GPU lease (worker stopped; the gateways idle; the user's browser
open, swap 7.1 / 8 GB used: speed drifts, A/B inside each process). mlx 0.32.3, mlx-vlm 0.7.6,
Ornith-1.5-9B 8-bit, `QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=0 QWEN_PRESENCE_PENALTY=0`. Raw:
`../raw/batch_invariant/` (`summary.txt` first).

**Gate: PASS.**

### 1. Kernel bench (`bi_kern.txt`)

Step linears, ms (independent calls in a chain: the GPU overlaps them, so this understates the cost
of a slow kernel in a decode step, where the matmuls depend on each other):

| rows | default | mode (2 × 8) | × | 4 × 8 at 9..16 | × |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.3 (qmv) | 30.1 | 1.14 | 30.3 | 1.15 |
| 2 | 34.8 (qmv) | 30.1 | 0.86 | 30.0 | 0.86 |
| 3 / 4 / 8 | 29.9 / 30.0 / 31.7 | 29.7 / 30.0 / 32.1 | 0.99–1.01 (same config) | | |
| 9 / 12 / 16 | 44.5 / 44.9 / 45.7 | 44.5 / 45.4 / 46.7 | 1.00–1.02 | 46.8 / 47.6 / 48.6 | 1.05–1.06 |

At one row the loss is in o_proj / GDN out_proj (4096 × 4096: 82–90 vs 63–67 µs) and gate/up
(197 vs 165 µs). In a *dependent* chain (`probe_chain.py`), MLP 1 row: qmv 189, mrqmm NA 1/2/4
192/193/198 µs; 4096², 1 row: 97 vs 109 µs. NA does not help at 1 row. So `INVARIANT_NA16` stays 2.

### 2. Metal tests (`bi_metal_tests.txt`)

`test_batch_invariant_metal.py` + `test_mrqmm.py`: 153 passed, 1 skipped (the real-weights test).

### 3. Real 9B: a row in a batch == the row alone

Load log: `mrqmm: 249 modules on the multi-row kernel for 1..16 decode rows (… in_proj_a/b x24 …
lm_head x1 …); left on quantized_matmul: none`; coverage `uncovered: {}`.

**Run A** (`bi_check.txt`): 16 prompts of 323–6035 tokens, 96 tokens each. It is prefill-bound
(mean batch ~4).

| phase | rows | steps compared | rows with a mismatch | max \|Δ log p\| |
|---|---:|---:|---:|---:|
| solo again with golden (4 prompts) | 4 | | 0 (equal) | |
| forced B = 2 | 2 | 185 | 0 | 0 |
| forced B = 4 | 4 | 342 | 0 | 0 |
| forced B = 8 | 8 | 572 | 0 | 0 |
| forced B = 16 | 16 | 1144 | 0 | 0 |
| free B = 16 (tokens, logprobs) | 16 | | 0 / 0 | |
| control, the mode off, forced B = 16 | 16 | 1144 | **8** | 0.020 |

**Run B** (`bi_check_full16.txt`): 16 prompts of 321–635 tokens, 256 tokens each, so the batch
reaches 16. Every batch size 1..16 decoded, 122 steps at B = 16.

| phase | rows | steps compared | rows with a mismatch | max \|Δ log p\| |
|---|---:|---:|---:|---:|
| forced B = 16 | 16 | 3704 | 0 | 0 |
| free B = 16 | 16 | | 0 / 0 | |
| control, the mode off | 16 | 3704 | **16** | 0.026 |

"Equal" means `==` on every step's (log p of the token, top-5 ids and log p's), and the same
tokens. Free runs of all 16 rows at once give byte-identical greedy answers to solo. Nothing needed
fixing on the real model.

### 4. Decode step cost, in-process A/B (`bi_step_ab.txt`)

The mode is toggled every 16 steps, 3 rounds. Each row gives the median of the per-round on/off
ratios.

| B | on ms | off ms | on/off | rounds |
|---:|---:|---:|---:|---|
| 1 | 32.9 | 27.9 | **×1.18** | 1.09, 1.21, 1.18 |
| 2 | 33.7 | 39.7 | **×0.86** | 0.87, 0.86, 0.85 |
| 4 | 36.9 | 36.7 | ×1.00 | 1.00, 1.01, 1.00 |
| 8 | 38.3 | 38.4 | ×1.00 | 1.00, 1.00, 1.00 |
| 16 | 57.2 | 58.9 | **×0.97** | 0.97, 0.98, 0.97 |

- **Earlier run, slow machine state** (`bi_step_ab_run1.txt`, 20 min before, B = 4 at 46 vs 37 ms):
  B = 1 ×1.39–1.46. B = 2 / 4 / 8 / 16 matched the table above: 0.86 / 1.00 / 1.00 / 0.97.
- **Probes at B = 1.**
  - Synchronous without the scheduler (`probe_b1_direct.py`): 32.4 vs 29.9 ms (×1.08). The same
    with a step-ahead: 31.2 vs 27.9 (×1.12).
  - The scheduler with the mode fixed per run (`probe_step_fixed.py`): 30.1 vs 27.2 (×1.11).
  - The 1-row cost is the mrqmm kernel's latency against qmv on dependent matmuls, mostly at
    4096 × 4096. It grows when the GPU runs slow.
- **Against the plan's pass criteria:**
  - B = 2 (≤ ×1.02), B = 4/8 (±2 %) and B = 16 (≤ ×1.05) pass.
  - **B = 1 misses its ≤ ×1.05 target: ×1.09–1.21.** It cannot be tuned away without changing
    bits: NA does not help, and KS is fixed by the 3..8-row bucket. A 1-row kernel with the same
    bits would have to reproduce the simdgroup MMA's internal rounding in scalar code.

### Findings for the default path (not changed here)

- At B = 2, mrqmm is 14 % faster than MLX's 2-row qmv. The default keeps qmv only for the "B ≤ 2
  bit-exact to solo" property.
- At 9..16 rows the default 4 × 4 is 2.6 % slower than 2 × 8 in the real step, although the
  2026-10-12 sweep had them within 3 %. Switching the default to 2 × 8 would be −2.6 % at B = 16
  and changes only the default's (already non-exact) bits there.

### Recommendation

- Keep `QWEN_BATCH_INVARIANT=0` as the default. The mode costs 9–21 % on lone-request steps
  (interactive single-user latency).
- Use `QWEN_BATCH_INVARIANT=1 QWEN_PROMPT_CACHE_REPLY=0` (with the default `QWEN_MAX_BATCH=16`)
  for reproducible AppWorld runs, instead of `QWEN_MAX_BATCH=2`:
  - batched steps cost nothing at B ≥ 3 and less at B = 2 and 16;
  - only steps with one row decoding pay;
  - under the AppWorld load (mean 5.4 rows per step at B = 16, `../../2026-10-17/replay_batch1.md`)
    the expected cost is ~0.
- Next: the AppWorld replay at `MAX_BATCH=16 + BATCH_INVARIANT=1 + REPLY=0` vs `MAX_BATCH=2`, two
  runs each. Check req/min and that the two invariant runs give identical replies.

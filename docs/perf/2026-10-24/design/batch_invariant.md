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
fp16 scales). It is not yet checked on the real model; see the GPU plan.

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
- At 1–2 rows: mrqmm costs ~1.03× a qmv step at M = 1 (31.3 vs 30.3 ms of step linears,
  `../../2026-10-12/mrqmm.md`). At M = 2 it should be faster than qmv (the qmv step is ~1.17× M = 1).
- At 9..16 rows: KS 8 with NA 2. The config sweep at 16 rows put 2 × 8 within 3 % of the default
  4 × 4. NA is free (it does not change a bit), so `INVARIANT_NA16` can be retuned from the GPU
  kernel bench; 4 × 8 is measured too.
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

(to be filled by the GPU phase)

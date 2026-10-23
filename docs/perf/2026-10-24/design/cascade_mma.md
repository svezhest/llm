# Cascade: prefix attention on simdgroup MMA over an fp16 prefix (`QWEN_CASCADE_KERNEL`)

2026-10-24. Item "Cascade: ядро с MMA по fp16-префиксу" of `../deepinfer_ideas.md`.
Code: `src/worker/cascade_mma.py` (new), `src/worker/cascade.py` (kernel mode of
`PrefixGroup`), `src/worker/worker.py` (knob, charge, formation). Tests:
`src/worker/tests/test_cascade_mma.py`. GPU plan: `scripts/perf/cascade_mma/`.

**Status.** The code and the CPU tests are done. The Metal correctness tests pass on M1 Max:
the kernel against an fp32 reference, and the grouped attention against per-row fp32 SDPA.
**Nothing has been measured for speed yet**: one-off timings were taken on a GPU shared with
another agent's worker, so they do not count. The GPU plan below still has to run under the
lease. The default (`QWEN_CASCADE_KERNEL` unset or `0`) is today's fp32 path, bit for bit.

## Today (fp32 cascade, `cascade.cascade`)

- **Prefix storage.** The prefix of a group is a fresh fp32 copy: 64 KiB per token on the
  9B (8 FA layers × 4 kv heads × 256 × 2 (K, V) × 4 B).
- **Formation.** The copy is built while the members still hold their fp16 caches. At
  8 × 32K the measured peak was +2.2 GB.
- **Attention.** The prefix part is `mx.matmul` / `logsumexp` / `softmax` over chunks of
  1024 keys. It materializes fp32 scores and probabilities of shape
  (Hkv, c, B_g·rep, 1024) per layer: 64 MiB per layer at 16 × 32K.
- **Speed.** Microbench (cascade_microbench, B=16, 32K): 18–19 ms per 8-layer step for
  ~69 GFLOP and 1.07 GB of fp16 prefix. That is ~3.7 TFLOPS, against an ideal of ~9–11 ms.
- **Open item** (`docs/perf/2026-10-12/cascade_gpu.md`): at 32K × 16 one decode step takes
  92 ms, against ~62 projected.

## What the kernel does

The rep = 4 query heads of each of the B_g ≤ 16 rows that map to one kv head are stacked
into M0 = B_g·rep ≤ 64 query rows. For each kv head, a decode step over the shared prefix
is then a small prefill-shaped attention: M0 queries × P keys, no mask. This is DeepInfer's
"pack the q heads of a kv head into the MMA tile; the KV tile is read once for all rows".

### Pass 1: `cmma_p1`

The loop of MLX's own steel attention kernel (mlx 0.32.3
`kernels/steel/attn/kernels/steel_attention.h`), run over ONE chunk of the prefix (split-KV).

- **Code reuse.** The `BlockLoaderT` loaders and `MMATile` fragments come from the
  wheel's headers, inlined as `pfqmm.py` does.
- **Tile.** BQ 32 × BK 16 × D 256, with 4 × 2 simdgroups.
- **WN = 2 split of D.** Each simdgroup sums half of the dot product, and the partial
  scores are exchanged through threadgroup memory (MLX's D = 256 configuration).
- **Staging.** Q and each K / V block are staged in threadgroup memory.
- **Precision.** S = QKᵀ is half × half MMA with fp32 accumulation. The scores are
  scaled by scale·log₂e in fp32. The online softmax runs in fp32. P·V multiplies the fp32
  probabilities by half V fragments with fp32 accumulation, so P is never rounded to fp16.
- **Output per (chunk, kv head, query row).** The chunk's normalized o (fp32), its max m
  (log₂ domain) and its sum l.
- **Partial last block.** It loads zeros past the chunk end, and those scores are masked.
  Nothing past P is ever read, which matters because the donor buffer holds its own
  suffix there.
- **Grid.** items × Hkv × ceil(M0 / 32) threadgroups.
- **Chunk size.** `pick_chunk` picks the largest power of two in [256, 2048] that still
  gives ≥ 256 threadgroups. Examples: 1024 at 32K × 16 rows (32 × 4 × 2 = 256), 256 at 8K.
  Four kv heads alone would leave most of the 32 cores idle.

### Pass 2

One threadgroup of D threads per (query row, kv head). It does the flash-decoding merge of
`rattn.py`'s pass 2 in fp32, w_c = l_c·2^(m_c − M), and outputs o_p (Hkv, M0, D) fp32 and
the natural-log LSE (M + log₂ L)·ln 2.

### Everything shape-dependent is a runtime input

M0, P, the buffer's slots capP, the chunk and the item count are passed in an int32 `meta`
array.

- **One meta per step.** `meta` is cached per shape, so it is built once per step and
  shared by every layer.
- **One pipeline per model.** There is one compiled pipeline pair per model layout
  (D, Hkv, scale). A new group, a new member count or a new P never recompiles.
- **Warm-up.** `cascade_mma.warm` compiles the pair at load when the knob is on, on the
  load thread before the scheduler starts.

### Suffixes and the merge

These keep today's mx ops. `cascade.merge` is `cascade`'s suffix + merge code line for
line, under `mx.compile` in the same trace LRU. It takes (o_p, lse_p) from the kernel.

Why the suffixes are not in the kernel (yet):

- they are short: ≤ P/2 by the leave rule, ~150 tokens at formation;
- their GEMMs cost little next to an 8K–32K prefix;
- moving them in would turn the fp32 stack into fp16 with per-row item tables, which means
  reworking append, take, grow, release, the masks and the bit-exact release.

This is a second step, once the prefix part is measured: see "Next" below.

### The prefix buffer: alias, not copy (`PrefixGroup.build(kernel=True)`)

`kc[i]` / `vc[i]` are the donor row's own (1, Hkv, cap, D) fp16 KVCache buffers, when
cap − P ≤ `slack` (`cascade.ALIAS_SLACK`, 1024).

- **No copy.** `mx.contiguous` of a row-contiguous array shares its buffer, so the build
  allocates nothing for the prefix. There is no formation transient (today: +2.2 GB at
  8 × 32K).
- **Pinned buffer.** The group pins the whole buffer; its [P, cap) is unused.
- **Long donor tail.** If the donor has more than `slack` slots past P, the group takes a
  fresh (1, Hkv, P + 1, D) copy instead (concatenate always writes a fresh buffer).
- **Choice of donor.** In kernel mode the scheduler picks as source the candidate with the
  fewest slots. Every candidate holds the same [0, P) bits (`_bounds`, hit == cold;
  `QWEN_CASCADE_CHECK` checks it).
- **Why aliasing is safe.** After formation the donor's KVCache objects are replaced by
  `_GroupRef`s. MLX never donates a buffer that has another live reference: a writer
  would copy, not overwrite. The prompt cache never holds a view of a row: `store` copies
  and `restore` concatenates.
- **`release` / `prefix_kv`.** They hand out views of the fp16 prefix, so the release
  round trip stays bitwise. That is tested, also after steps that grow the stacks.

Kernel mode is fp16 KV only: `build` raises on bf16 / fp32. The scheduler enables it only
when the model's compute dtype (`_act_dtype`, the final norm's weight dtype) is fp16; with
`QWEN_FP16=0` the knob is ignored and the load log says so.

## Knob

| var | default | meaning |
|---|---|---|
| `QWEN_CASCADE_KERNEL` | `0` | `1` = groups formed from now on keep the fp16 prefix (aliased) and use `cascade_mma`; anything else = today's fp32 path, bit for bit. Fixed per scheduler (`_Scheduler.cascade_kernel`), because the charge depends on it. Needs cascade on (fp16 KV + prompt cache + max_batch ≥ 3) and an fp16 model. |

What changes in the load log only when the knob is set:

- `cascade attention (QWEN_CASCADE): on: … ; MMA kernel over the fp16 prefix (QWEN_CASCADE_KERNEL)`, or
- `… ; fp32 prefix (QWEN_CASCADE_KERNEL=1 ignored: the model is not fp16)`.

The formation line reads `(prefix fp16 aliased|copied + stacks fp32)` instead of
`(prefix + stacks, fp32)`.

## Memory accounting (`_Scheduler._group_bytes`, kernel mode)

```
prefix  = (P + CASCADE_ALIAS_SLACK) · kv_fp16          # the donor's buffer or a P+1 copy (bound)
stack   = 1.05 · n · s · 2·kv_fp16                     # fp32 suffix stack, as today
scores  = n_fa · (8 · Hq · n · s                       # suffix scores + probabilities (as today)
                  + cascade_mma.partial_bytes(P, n, Hq, D)) + CASCADE_SLACK
moves   = 2 · n · s · 2·kv_fp16 / n_fa                 # one layer's old + new stack (as today)
charge  = prefix + stack + max(scores + moves, 2 · 1.05 · (P + s) · kv_fp16)   # same 2-row dissolve floor
```

- `partial_bytes` bounds the pass-1 partials + the pass-2 output of a layer at the smallest
  chunk: (ceil(P/256) + 1)·n·Hq·(D + 2)·4 B. At 16 × 32K that is a 34 MB bound per layer;
  the real partials, at the 1024 chunk, are ~9 MB.
- The fp32 path's (Hkv, c, B·rep, 1024) prefix scores are gone.
- `nbytes` ≤ charge holds; the scheduler tests check it every tick.

**What changes (9B layout, `test_kernel_charge_bytes`).**

- **Smaller charge.** The charge is lower than the fp32 one at every size. At 16 × 32K
  it is ~1 GiB lower.
- **Wins at fewer rows.** `_group_wins` (real lengths, ~150-token suffixes) now holds from
  4 rows at 4K, 8K and 32K. In fp32 mode it needed 5 rows, or 4 only from a 32K prefix.
- **2–3 rows still never win.** The 2-row dissolve floor dominates: at 3 × 32K the charge
  is 3.20 GiB against the rows' own 3.16 GiB.
- **Formation threshold unchanged.** `CASCADE_MIN_ROWS` (8) still decides formation.
  Smaller groups only matter after members leave.

## Numerics

The products are the same as today's fp32 path:

- q is fp16, so its widening is exact;
- each fp16 × fp16 product is exact in fp32;
- the scale is 1/16, a power of two.

The difference is the summation order (8 × 8 MMA trees, the WN = 2 split, the online
rescale per 16-key block, the chunk merge) and exp2 with log₂e folded into the scale.
Nothing is rounded below fp32 before the final fp16 output.

Measured on Metal (`test_cascade_mma.py`, CASCADE_MMA_GPU=1):

- **Kernel alone vs the fp32 mx reference.** Over 9B shapes, P 4K / 8229 / 32K, 3 / 8 / 16
  rows, flat and peaked queries, NaN past P:
  - o_p rms rel 1.3–3.6e-6, max abs ≤ 1.4e-5;
  - lse max abs ≤ 1.4e-5.
- **Whole grouped attention vs per-row fp32 SDPA.** Kernel prefix + suffix + merge, ragged
  suffixes, one empty, plus a decode step: **identical to today's fp32 cascade to 3–4
  digits**, rms rel 2.04–2.09e-4 for both. The fp16 output rounding dominates.

So the expected class on the real model is today's cascade: B4 measured top-1 ≥ 99.71 %
and KL +0–13 % over the per-row batch. GPU plan phase A checks it.

## Invariants kept

- `QWEN_CASCADE_KERNEL` off: no code path changes.
  - `PrefixGroup.build` defaults to `kernel=False` and `cascade.attention` takes the
    compiled `cascade`.
  - `_group_bytes` and the log lines are byte-identical.
  - The whole existing suite passes, and `test_knob_off_never_runs_the_kernel` checks it.
- MLX runs only on the scheduler thread, plus the load-time `warm` before the scheduler
  starts. Kernels are compiled under a lock; `_META` is touched by the scheduler thread only.
- B ≤ 2 is never grouped. Ungrouped rows of a mixed batch keep the per-row calls
  (bit-identical).
- The step protocol, bounds and donation of the stacks are untouched (shared with the
  fp32 mode).
- Nothing past a cache's offset or past P is read. The NaN-garbage tests run in both
  modes and on Metal.

## Tests (`src/worker/tests/test_cascade_mma.py`)

**CPU (the default suite; cascade_mma falls back to `reference`):**

- Kernel-mode group attention vs per-row fp32, for both layouts and flat / peaked
  queries.
- The alias: no prefix allocation (active-memory delta ≤ stacks); `nbytes`; the group
  survives the donors' KVCaches being dropped.
- The copy path for a long donor tail.
- fp32 / bf16 raise.
- Release round trip bitwise after 19 steps that grow the stacks; NaN garbage past the
  donor's offset; `prefix_equal`; add / take.
- `reference` = the fp32 prefix math; the shape / dtype checks; `pick_chunk` /
  `partial_bytes`.
- The knob: off by default, the load-log text, ignored without fp16.
- `_group_bytes` term by term on the 9B layout, below fp32 everywhere, and the new
  `_group_wins` table.
- Through the model (fp16 and 8-bit + GDN fuse): grouped logits in kernel mode as close
  to per-row as fp32-mode groups, top-1 equal, ungrouped rows bitwise.
- Scheduler waves (fp16 model): formation in kernel mode with an aliased prefix; joins;
  accounting every tick; logits vs the no-group batch within 2× of the fp32 groups'
  distance; nothing left at the end; the donor is the tightest candidate.
- Knob off: `cascade_mma` never runs.

**Metal (`CASCADE_MMA_GPU=1`, ~13 s, < 1 GB):**

- The kernel vs the fp32 reference at 9B shapes (P 4K–32K, 3–16 rows).
- Partial blocks, chunks and query blocks on odd P (1, 5, 37, 100, 300, 1000, 4097) over
  three layouts (D 64 / 128 / 256) and 1–16 rows.
- The grouped attention (kernel + suffix + merge + a decode step + release) vs per-row
  fp32, ≤ 2.3e-4 and ≤ 1.05 × the fp32 cascade.

## GPU plan (to run under the lease)

One command, from this tree (it uses the main tree's venvs, weights and gateway binary):

```zsh
zsh scripts/perf/cascade_mma/run_gpu.zsh            # phases T M S A E, ~45 min
```

- **Output.** Everything goes to `docs/perf/2026-10-24/raw/cascade_mma/`: one `*.txt` log
  and one JSON per phase, plus `status.txt` with timestamps, `pmset -g therm`, load
  average and free memory.
- **Guard.** The script refuses to start while any `python worker.py`, pytest or
  microbench process runs (`CMMA_FORCE=1` overrides).
- **Nice level.** It runs everything at nice 0 (`setopt NO_BG_NICE`).
- **Processes.** It touches no process it did not start. Phase E starts and stops its own
  worker on :50061 and gateway on :8095.

Following "Measuring performance":

- every A/B is inside one process, interleaved, with medians over ≥ 4 rounds (M, S);
- end-to-end runs are ON / OFF / ON with a 64-token probe and a 30 s cooldown (E);
- nothing else may run on the GPU, and Activity Monitor should be closed.

| phase | what | time | peak memory |
|---|---|---|---|
| T | Metal + CPU tests of the cascade files | ~1 min | < 1 GB |
| M | `bench_attn.py`: attention only, synthetic 9B, 8 layers | ~10 min | ~6.5 GB (32K × 16: fp32 group 2.5, kernel group 1.6, per-row rows 2.4) |
| S | `step_ab.py`: the real 9B in-process | ~8 min | ~19 GB (32K × 16: model 9.5 + three groups 5.8 + 3 GDN stacks 2.5) |
| A | `agree_mma.py`: real 9B exactness, `AGM_CONTROL_MAX_GB=6` (per-row control at 8K only) | ~10 min | ~17 GB (32K × 16), plus ~2 GB of numpy log-probs in RAM |
| E | worker + gateway waves, 16 × 8K (256 tokens) and 16 × 32K (128 tokens), kernel ON / OFF / ON | ~13 min | worker ≤ ~24 GB (32K × 16; B4 measured 23.2 GB in fp32 mode) |

### M: microbench (`bench_attn.py`; 8K / 32K × 8 / 16 rows, ragged suffixes 150–500)

**Variants**, per step of 8 layers, interleaved, 5 rounds × 20 steps:

- `mma`: the kernel group;
- `fp32`: today's group;
- `mma_pre`: the kernel's prefix half alone, with a chunk sweep `@256…2048`;
- `rows`: per-row MLX SDPA over P + S;
- `rattn`;
- `fp32_step` / `mma_step`: inside the step protocol, including the appends.

It also prints numerics (rms vs per-row fp32) and the achieved TFLOPS of the prefix.

**Pass:**

- `mma` ≤ `fp32` in every config;
- at 32K × 16, `mma` ≤ 0.7 × `fp32` (today ~18–19 ms) and `mma_pre` ≤ 12 ms (≥ 5.7
  TFLOPS; ideal 8.6 ms at 8 TF);
- numerics `mma` ≤ 1.05 × `fp32`.

**If `mma` is slower** at ≥ 8 rows, read the chunk sweep first:

- **A chunk clearly best:** fix `pick_chunk` / `TARGET_TG`.
- **Still slow:** look at a Q-in-registers BK 32 variant (MLX's `q_in_registers`), BQ 64
  (one query block per kv head at 16 rows: K/V read once instead of twice), and the merge
  pass. The kernel's pass 2 and the compiled merge are separate launches; `mma` −
  `mma_pre` shows their cost.

### S: real 9B, in-process step A/B + the 32K × 16 decomposition (`step_ab.py`)

**Setup.** P = 8K / 32K, B = 8 / 16, step-ahead decode as `_iterate`, blocks of 48 steps,
4 rounds. The variants are:

- `fp32`, `mma`;
- `stub`: the kernel group with `cascade.attention` stubbed out, i.e. everything of the
  step but the grouped attention;
- `attn_fp32`, `attn_mma`: the 8 FA layers' attention alone, on the live group;
- `control`: per-row, at 8K only.

**Printed per config:**

- in-model attention = step − stub, against the isolated attention;
- `mma step ×… vs fp32`.

**Pass:** the `mma` step is faster than `fp32` at 32K × 16 by ≥ 5 ms (expected −6 to −8 ms
per step), and it is not slower at 8K × 8.

**The open question (92 vs ~62 ms)** is answered by the decomposition:

- **(a) The rest of the step was underestimated.** `stub` at 32K × 16 is far above the
  ~44 ms of a 1K 16-row step: 16 rows of GDN state at long context, the suffix part, the
  plan / appends. The projection was wrong, not the attention.
- **(b) The attention costs more in the model than alone.** `fp32` − `stub` ≫ `attn_fp32`:
  contention with the rest of the step (SLC / bandwidth). The fix is then to cut its
  traffic, which is what the fp16 prefix does (half the bytes, no score materialization);
  `mma` − `stub` vs `attn_mma` shows how much is left.
- **(c) Neither.** The step is long because of something else between the layers. Then
  profile with `scripts/perf/prof/step_prof.py --ctx 32k-shared`.

### A: exactness on the real 9B (`agree_mma.py`; B4's G1 method, `cascade_gpu.md`)

**Setup.** 8K / 32K × 8 / 16 rows, 128 teacher-forced steps. Solo greedy is the teacher.
`fp32`, `mma` and `control` (8K only) each run through the scheduler's own code path.
Pairs: every mode vs solo, plus `mma_vs_fp32`, `control_vs_fp32` and `control_vs_mma`.
Per-layer rms vs fp32 at step 64.

**Pass (B.7's bar):**

- `mma_vs_solo` top-1 ≥ 99.7 % in every cell;
- KL mean ≤ max(1.5e-5, 1.1 × `fp32_vs_solo` of the same cell);
- `mma_vs_fp32` KL mean ≤ `fp32_vs_solo` KL mean (the kernel moves the logits less than
  grouping does);
- per-layer rms of `mma` ≤ 2.3e-4 and ≤ the per-row kernel in ≥ 95 % of (row, layer)
  pairs.

**Expected:** numbers equal to the fp32 cascade's within noise, as on the synthetic data.

### E: end to end (fresh worker each, kernel ON / OFF / ON, `QWEN_CASCADE_CHECK=1`)

**Waves:** 16 × 8K shared (256 tokens) and 16 × 32K shared (128 tokens), through
`bench_concurrency.py`.

**Pass:**

- no Traceback / ERROR / "not grouped" in the logs, and every ON formation line says
  `prefix fp16 aliased`;
- 16 × 32K: all 16 rows admitted;
- worker peak ON ≤ OFF − 0.5 GB at 32K (expected ~−1 to −2 GB: 1 GiB less prefix and no
  formation transient);
- steady 16-row decode tok/s per request ON ≥ OFF at 32K, and within ±3 % at 8K;
- both ON runs agree with each other (ABA).

**Exit criteria for a later default flip** (not part of this item): M, S, A and E all pass.
Then a second confirm run on the defaults, as B4's C1/C5.

## Risks and open points

- **Speed is unmeasured.** It is the whole point. If the kernel is not clearly faster at
  ≥ 8 rows after the chunk sweep, the memory gain still stands (alias, −1 GiB per 32K
  group, no +2.2 GB formation peak), and a follow-up tunes the tile (BQ 64 /
  Q in registers).
- **Occupancy at small groups.** M0 = 3 × 4 = 12 rows fills one 32-row query block, so
  ⅝ of the MMA work is padding. Groups form at 8 rows by default (32 query rows = one full
  block), so that is only after members leave.
- **Pinned donor buffer.** The alias pins the donor's whole buffer, up to `ALIAS_SLACK`
  (1024) unused slots, 32 MiB on the 9B. It is charged.
- **Per-step host work.** Each layer runs one more launch pair (pass 1 + pass 2) plus the
  compiled merge, against one compiled graph today. Host time per layer is ~µs
  (`meta` is cached, the pipelines are prebuilt). Phase M's `mma` − `mma_pre` shows it.
- **Future MLX bumps.** A bump may change `steel/attn` (loaders, `MMATile`). The kernel
  inlines the wheel's headers, so a header change breaks the compile at the first group,
  or at `warm` at load, not silently. When bumping mlx, diff `cmma_p1` against
  `steel_attention.h::attention`.

## Next (not done)

- Suffixes in the same kernel (fp16 stack, per-row item tables) or through `rattn`, so
  that `merge` becomes one launch.
- BQ 64 for 16-row groups (each K/V tile read once per kv head instead of twice).
- Reading the prefix straight from a prompt-cache segment instead of the donor row: no
  pinned donor tail, and groups could form before any row is prefilled.

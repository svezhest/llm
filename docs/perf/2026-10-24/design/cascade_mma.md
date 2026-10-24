# Cascade: prefix attention on simdgroup MMA over an fp16 prefix (`QWEN_CASCADE_KERNEL`)

2026-10-24. Item "Cascade: ядро с MMA по fp16-префиксу" of `../deepinfer_ideas.md`.
Code: `src/worker/cascade_mma.py` (new), `src/worker/cascade.py` (kernel mode of
`PrefixGroup`), `src/worker/worker.py` (knob, charge, formation). Tests:
`src/worker/tests/test_cascade_mma.py`. GPU plan: `scripts/perf/cascade_mma/`.

**Status (after the GPU lease, 2026-10-24 03:37–04:18).** Done and measured; see
"Results" right below. Verdict: **keep `QWEN_CASCADE_KERNEL=0` as the default.**

- **Memory: the kernel wins.** 16 × 32K end to end: worker peak 20.9 against 23.3 GB, and
  active 1.1 GiB lower.
- **Exactness: same class as today.** top-1 ≥ 99.80 % in every cell.
- **Speed: no gain at the shapes that matter.** The 16-row step is equal at 8K and 3–6 %
  slower at 32K. The kernel's prefix part is capped at ~4 TFLOPS on M1, while MLX's fp32
  GEMM path reaches ~5.5. The kernel wins only for groups of ≤ 8 rows: ×1.36–1.40 on the
  attention, ×1.05–1.10 on the step.

The knob stays as the memory option. The default is today's fp32 path, bit for bit.

## Results (GPU lease, 2026-10-24)

**Conditions.**

- **Load.** M1 Max, no other model on the GPU. Other agents' CPU pytest ran at times:
  the ±20 % noise was accepted, and every A/B is interleaved in one process.
- **Memory.** Free memory was 13–23 GB and swap 3.4–5.5 GB; nothing was reduced.
- **Raw data.** `docs/perf/2026-10-24/raw/cascade_mma/`: `status.txt`; `M_bench_attn.txt`
  (final); `M_bench_attn_v1_mxmerge.txt` (first version, with the mx suffix merge);
  `M2_tile_sweep.txt`; `S_step_ab.txt`; `A_agree.txt`; `E_*.txt` / `E_*.json`;
  `E_worker_*.log`; plus the JSONs.

| phase | verdict | key numbers |
|---|---|---|
| T | **PASS** | Metal 130 passed; CPU 74 passed |
| M | **partial** | ≤ 8 rows the kernel wins (×1.36–1.40), 8K × 16 equal (×1.04), **32K × 16 slower (×0.84)**; prefix half 17.2 ms / 4.0 TF vs the 12 ms target; numerics = fp32 cascade to 4 digits |
| S | **partial** | step ×1.055 / ×1.013 / ×1.096 / **×0.968** at 8K×8 / 8K×16 / 32K×8 / 32K×16; the "92 vs 62" question answered (below) |
| A | **PASS** (one marginal KL cell) | top-1 ≥ 99.80 % in every cell; mma vs fp32 KL ≤ fp32 vs solo in 4/4; 8K × 16 KL mean 2.07e-5 vs a 1.86e-5 bar (one token at KL 7.6e-3; p99 equal) |
| E | **PASS** except speed | 0 errors, 0 refusals, `prefix fp16 aliased` in every ON formation, all 16 rows grouped; 32K peak **20.94 / 20.95 vs 23.26 GB**; decode tok/s/req equal (13.1 vs 13.2) |

### M: attention only (synthetic 9B, 8 FA layers per step, ms per step)

Columns: `fp32` is today's group, `mma` is the kernel group (prefix + suffix + merge on
Metal), `mma_pre` is the prefix half alone, `rows` is per-row MLX SDPA, `rattn` is
`rattn.py`.

| P × B | fp32 | mma | mma_pre (TFLOPS) | rows | rattn | mma vs fp32 |
|---|---|---|---|---|---|---|
| 8K × 8 | 6.52 | **4.65** | 3.13 (2.75) | 9.52 | 7.28 | ×1.40 |
| 8K × 16 | 8.18 | 7.88 | 5.18 (3.32) | 18.74 | 14.25 | ×1.04 |
| 32K × 8 | 14.85 | **10.92** | 9.33 (3.68) | 27.81 | 25.51 | ×1.36 |
| 32K × 16 | **16.84** | 19.94 | 17.16 (4.00) | 55.07 | 51.40 | **×0.84** |

- **Against per-row.** Both cascades beat per-row attention ×1.5–3.3, which was known
  from B4.
- **Peak memory per step.** The kernel's is lower: +73 vs +567 MiB at 32K × 16, since
  there are no score transients.
- **Numerics.** rms rel against per-row fp32 is identical for `mma` and `fp32`
  (2.055–2.077e-4); the per-row fp16 kernel is at 2.9e-4.

**What limits the kernel.** At 16 rows, M0 = 64 query rows are two 32-row query blocks.
Cost scales exactly with q-blocks × keys: 32K × 8 takes 9.3 ms and 32K × 16 takes 17.2 ms.
So the kernel is compute / issue bound at ~4 TF per block, not bandwidth bound (126 GB/s).
MLX's fp32 batched GEMM in today's path takes 64-row tiles at ~5.5 TF, which beats it once
there are 64 rows.

**Variants tried** (`M2_tile_sweep.txt`, all with the same error):

| variant | result |
|---|---|
| BK 32 with Q in registers (`32x32q`) | equal |
| 64-row tiles, TQ 2 (`64x16q` / `64x32q`) | 1.0–2.5 TF: register spills; 8 × 2 simdgroups = 512 threads exceed the 448 the pipeline allows |
| software-pipelined K/V (next block read into registers during the MMAs) | equal |
| P·V with fp16 P | equal speed, error ×100 |
| chunk 256–2048 | ±5 %; `pick_chunk` kept |

The first version merged the suffixes with a compiled mx graph, which cost 2.8–4.8 ms per
step (`v1_mxmerge`: 32K × 16 was ×0.77). A Metal suffix pass plus one merge launch
(`group_attention`) replaced it; that is the version measured above.

### S: in-model step (real 9B, step-ahead, medians of 4 × 48 steps)

`stub` is the step with `cascade.attention` stubbed out (the rest of the step);
`attn_fp32` / `attn_mma` are the attention alone, isolated. All in ms.

| P × B | fp32 step | mma step | stub (rest) | fp32 attn in model / isolated | mma attn in model / isolated | control (per-row) |
|---|---|---|---|---|---|---|
| 8K × 8 | 40.8 | 38.7 | 35.9 | 4.9 / 5.3 | 2.7 / 4.0 | 44.4 |
| 8K × 16 | 53.5 | 52.8 | 46.8 | 6.8 / 6.2 | 6.1 / 6.7 | 64.0 |
| 32K × 8 | 46.0 | 42.0 | 31.6 | 14.5 / 14.7 | 10.4 / 11.4 | (8.5 GB: skipped) |
| 32K × 16 | **62.5** | 64.6 | 46.8 | 15.7 / 15.8 | 17.8 / 19.6 | (17 GB: skipped) |

**"92 vs ~62 ms" (cascade_gpu.md's open item) is not reproduced on today's code.**

- **The step now measures what was projected.** The fp32 grouped step at 32K × 16 is
  62.5 ms in-process, and 57.7 ms in the worker at mean_rows ≈ 12 (E).
- **The rest of the step is context-independent.** The stub is 46.8 ms at 8K × 16 and at
  32K × 16.
- **No contention.** The attention costs the same inside the model as alone (15.7 vs
  15.8 ms).
- **The 2026-10-12 figure** of 92 ms (88.5 → 98.4 as the suffix grew to 560) therefore
  came from that run, not from the attention. That evening the GPU had slowed ~18 % (the
  doc notes 65 → 77 ms at 8K × 16 in G1), and the code has changed since. Without a bisect
  it cannot be pinned to one change.

### A: exactness (teacher-forced 128 steps, solo greedy = teacher)

| cell | fp32 vs solo top-1 / KL | mma vs solo top-1 / KL | control vs solo | mma vs fp32 KL |
|---|---|---|---|---|
| 8K × 8 | 99.80 % / 1.45e-5 | 99.80 % / 1.45e-5 | 99.80 % / 1.39e-5 | 9.95e-6 |
| 8K × 16 | 99.71 % / 1.69e-5 | **99.80 % / 2.07e-5** | 99.76 % / 1.56e-5 | 1.21e-5 |
| 32K × 8 | 100 % / 1.35e-5 | 100 % / 1.13e-5 | — | 1.13e-5 |
| 32K × 16 | 99.85 % / 1.36e-5 | 99.85 % / 1.25e-5 | — | 1.17e-5 |

- **Per-layer attention rms vs fp32 at step 64.** mma mean 2.05–2.11e-4, fp32 2.05–2.10e-4;
  both are ≤ the per-row fp16 kernel in 100 % of (row, layer) pairs.
- **The 8K × 16 KL miss.** The KL mean is higher there because of one token at KL 7.6e-3
  (p99 1.31e-4 vs fp32's 1.29e-4). Same class.

### E: end to end (16 × 8K / 256 tokens and 16 × 32K / 128 tokens, ON / OFF / ON, fresh workers)

| run | 8K wall / agg tok/s / decode tok/s/req / peak GB | 32K wall / agg / decode / peak GB | 32K formation charge |
|---|---|---|---|
| ON 1 | 32.5 / 126.1 / 17.2 / 14.42 | 76.6 / 26.7 / 13.1 / **20.94** | 3.31 GiB (reserved 9.36 → 4.14) |
| OFF | 32.4 / 126.3 / 17.2 / 14.99 | 77.8 / 26.3 / 13.2 / **23.26** | 4.34 GiB (9.38 → 5.17) |
| ON 2 | 33.7 / 121.4 / 16.7 / 14.53 | 76.8 / 26.7 / 13.1 / **20.95** | 3.32 GiB |

- **Steady 16-row step** (the worker's `batch:` lines):
  - 8K: 53.1 / 53.4 ms (ON) vs 52.7 ms (OFF);
  - 32K: 61.3 / 61.6 ms (ON) vs 57.7 ms (OFF), with ~12–13 mean rows in that window.
  - Active memory at 32K: 13.55 GiB (ON) vs 14.66 GiB (OFF).
- **What the kernel buys end to end.** −2.3 GB of peak (the formation transient is gone)
  and −1.0 GiB of charge per 32K group; in a memory-bound load that means more rows
  admitted.
- **What it costs.** No speed gain at 16 rows: 32K decode is 1 % slower per request, and
  the step 3–6 % slower.

### Recommendation

- **Default.** Keep `QWEN_CASCADE_KERNEL=0`.
- **When to set it to 1.** Where admission is memory-bound with long shared prefixes
  (16 × 32K+, or 27B-class weights next to the 9B). Groups of ≤ 8 rows (after members
  leave) also get faster attention.
- **What would flip the default.** A prefix kernel faster than ~5.5 TF at 64 query rows,
  so that 16-row groups win too. That needs a 64-row block per kv head without spilling,
  e.g. a D-split into 4 simdgroups (WN = 4) at ≤ 448 threads, which is not tried yet.
  Alternatively a hybrid that keeps the fp16 storage and picks the GEMM path for 16-row
  groups, which would need an fp16 × fp16 → fp32 GEMM from MLX, not available today.

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
- **Warm-up.** `cascade_mma.warm` compiles the pair at load whenever the knob is on and
  the model is fp16, even with cascade off at that moment (a later `set_cascade(True)`),
  on the load thread before the scheduler starts. A compile error after an MLX bump then
  shows at load.
- **Names and head dims.** The kernel name carries the exact scale (`float.hex`), because
  MLX caches libraries by name. Only the tested head dims 64 / 128 / 256 are accepted.

### Suffixes and the merge (`cascade_mma.group_attention`)

On Metal the whole grouped attention is three launches per layer:

1. **Prefix pass 1** (above).
2. **Suffix pass.** One threadgroup of 4 simdgroups per (row, 256-slot piece, kv head),
   over the members' **fp32 stack as it is**. It is `rattn.py`'s pass 1 on an fp32
   buffer: each simdgroup takes a run of slots and computes all rep query heads of its kv
   head in fp32. It reads only the `slens` (S_j, + 1 inside a step) slots of each row.
3. **One merge.** The prefix chunks' and the row's suffix pieces' (o, m, l) are merged
   straight into the fp16 (B, Hq, 1, D) output. An empty piece weighs 0.

**Why not the mx graph.** The stack, append, take, grow and the bit-exact release are
untouched. The first version merged with a compiled mx graph (`cascade.merge`, the same
ops as `cascade`), and that cost 2.8–4.8 ms per 8-layer step.

**The CPU path.** `cascade_mma.reference` + `cascade.merge` (the CPU tests) still use the
mx graph.

### The prefix buffer: alias, not copy (`PrefixGroup.build(kernel=True)`)

`kc[i]` / `vc[i]` are the donor row's own (1, Hkv, cap, D) fp16 KVCache buffers, when
cap − P ≤ `slack` (`cascade.ALIAS_SLACK`, 1024).

- **No copy.** `mx.contiguous` of a row-contiguous array shares its buffer, so the build
  allocates nothing for the prefix. There is no formation transient (today: +2.2 GB at
  8 × 32K).
- **Pinned buffer.** The group pins the whole buffer; its [P, cap) is unused.
- **Long donor tail.** If the donor has more than `slack` slots past P, the group takes a
  fresh (1, Hkv, P + 1, D) copy instead (concatenate always writes a fresh buffer).
- **Per buffer.** The alias / copy decision is taken per layer and per K / V buffer;
  `aliased` is True only when every one is aliased.
- **Choice of donor.** In kernel mode the scheduler picks as source the candidate with the
  fewest slots. Candidates hold the same [0, P) bits only where [0, P) was built by
  prefill (`_bounds`, hit == cold). A row resumed through a reply entry
  (`QWEN_PROMPT_CACHE_REPLY=1`, the default) holds decode-kernel bits up to the reply's
  end, which may differ. Then, exactly as the fp32 path did with `rows[0]`, every member
  attends the donor's bits; `QWEN_CASCADE_CHECK=1` refuses such members.
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
- **2–3 rows mostly do not win.** The 2-row dissolve floor dominates: at 3 × 32K the
  charge is 3.20 GiB against the rows' own 3.16 GiB. From 64K, 3 rows win barely (6.30 vs
  6.31 GiB; 96K 9.40 vs 9.46).
- **Break-even suffix** (real suffix where a group stops winning): 769 / 1537 / 2049
  tokens for 4K × 5 / 8 / 16 rows, 2049 / 3585 / 4609 for 8K, 9217 / 16129 / 19713 for
  32K (fp32 mode: 257 / 1281 / 2049, 1025 / 3073 / 4353, 4865 / 13057 / 18177).
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

- **Speed (measured, see Results).** The kernel is not faster at 16 rows: a ~4 TF ceiling
  per 32-row block. BQ 64 spilled, Q in registers / BK 32 / pipelining were equal. The
  memory gain stands: the alias, −1 GiB per 32K group, and −2.3 GB of peak.
- **Occupancy at small groups.** M0 = 3 × 4 = 12 rows fills one 32-row query block, so
  ⅝ of the MMA work is padding. Groups form at 8 rows by default (32 query rows = one full
  block), so that is only after members leave.
- **Pinned donor buffer.** The alias pins the donor's whole buffer, up to `ALIAS_SLACK`
  (1024) unused slots, 32 MiB on the 9B. It is charged.
- **Per-step host work.** Each layer runs 3 launches (prefix pass 1, suffix pass, merge)
  against one compiled graph today. `meta`, `smeta` and `slens` are cached per step. At
  8 rows the step still wins, so the host time is not the issue.
- **Future MLX bumps.** A bump may change `steel/attn` (loaders, `MMATile`). The kernel
  inlines the wheel's headers, so a header change breaks the compile at the first group,
  or at `warm` at load, not silently. When bumping mlx, diff `cmma_p1` against
  `steel_attention.h::attention`.

## Next (not done)

- A 64-query-row block per kv head without spilling, e.g. WN = 4 (a D-split in 4) at
  ≤ 448 threads. It is the one thing that could make 16-row groups faster than MLX's fp32
  GEMM path.
- An fp16 suffix stack: the suffix pass reads fp32, 2.6 ms of 7.9 at 8K × 16.
- Reading the prefix straight from a prompt-cache segment instead of the donor row: no
  pinned donor tail, and groups could form before any row is prefilled.

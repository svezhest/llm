# oQ4e 27B: why it decodes slower than the uniform 4-bit, and what was done (2026-11-15)

Status: code and CPU tests done; GPU gate steps 1–3 (no model: kernel tests, real tensors,
microbench) **passed** on 2026-11-15, results in "Measured" below (raw: `../raw/oq4e_bench.*`);
steps 4–8 (one 27B per process) not run yet. Sections before "Measured" were written from code,
tensor headers and the 2026-11-10 per-width kernel tables (`docs/perf/2026-11-10/dflash2_27b.md`
§3) before any of it ran; "Measured" corrects them where they disagree.

## The problem

`weights/qwen38-27b-mlx-oq4e` (Jundot/Qwen3.8-27B-oQ4e-mtp) is the more faithful 4-bit-class build
of the 27B (teacher-forced vs the 5-bit, `scripts/perf/prof/kl_models.py`, 3 × 3072 tokens: KL
0.0236 / top-1 93.6 % vs 0.0534 / 90.1 % for the uniform lmstudio 4-bit), but measured slower
(M1 Max, ~2K prompt, greedy, no thinking, separate processes):

| | plain tok/s | DFlash tok/s | DFlash round | drafts accepted / round | GDN layers fused |
|---|---|---|---|---|---|
| uniform 4-bit | 17.2 | 37.7 | 153 ms | 4.8 | 48 |
| oQ4e | 11.6–11.9 | 16.6–17.6 | 259–274 ms | 3.57 | 0 |

## Layout (from the safetensors headers; affine g64, bf16 scales cast to fp16 at load)

| module | bits |
|---|---|
| embed | 5 |
| lm_head (248320 × 5120) | **6** |
| MLP gate / up | 4 |
| MLP down | 5 in layers 0–17, else 4 |
| GDN in_proj_qkv | 4 |
| GDN in_proj_z | 5 in the 15 GDN layers of 0–18, else 4 |
| GDN in_proj_b / a (N = 48) | 5; 8 in layers 50 and 62; a 6 in layer 52 |
| GDN out_proj | 5 |
| attention q / o | 5 |
| attention k / v | 5; **6** in layer 63 |

LM ≈ 14.85 GiB loaded (the MTP head, ~0.3 GiB, is dropped by sanitize) vs ≈ 14.09 GiB for the
uniform 4-bit: ×1.054 bytes.

## Diagnosis of plain decode (B = 1), ranked

Uniform 4-bit: ~58 ms / token; oQ4e: ~85 ms. B = 1 never runs mrqmm (1 row = MLX `qmv`), so the
6-bit kernel cannot be the plain-decode cause. MLX 0.32.3 has `qmv_fast` for 2..8 bits, and the
2026-11-10 bits bench measured a 1-row qmv at ~280 GB/s at every width (6 bits ~7 % below
proportional).

1. **H0, GPU drift between the two processes** (CLAUDE.md "Measuring performance": ±20–30 % over
   minutes). After H1–H4 the remaining gap is a uniform ×1.35–1.4 on both plain decode (61 → 85 ms)
   and the DFlash round (~195 → 265 ms): one multiplicative factor on two unrelated paths points at
   the machine, not the kernels. Up to ~20 ms of the 27. Needs an interleaved re-measure with an
   in-process bandwidth probe (gate step 5).
2. **H1, unfused GDN in-projection.** `_install_gdn_fuse` skipped every layer whose four
   projections differ in bits: 0 of 48 layers on oQ4e → 4 launches per GDN layer instead of 1
   (b / a at N = 48 are 6-threadgroup, latency-bound launches). +144 launches, ~1.5–3 ms (2–4 %;
   the 9B fuse measured −2.3 % at B = 1). **Fixed** (width groups, below).
3. **H2, bytes.** +0.76 GiB at ~280 GB/s ≈ +2.5–3 ms (~5 %). Inherent to the build.
4. **H3, 6-bit lm_head qmv** at 248320 × 5120 (unmeasured at that shape; 6-bit qmv runs ~7 % below
   proportional at 17408 × 5120): ~+0.3–1 ms.
5. **H4, 5-bit qmv at the other shapes** (K = 17408 down in layers 0–17, K = 6144 out / o, N = 1024
   k / v), unmeasured: 0–2 ms.
6. **H5, residency of the loader's mmap'd buffers** (MLX can wrap mmap'd safetensors as no-copy
   buffers): speculative; check page-ins during decode if H0 does not explain the gap.

Prediction with H0 removed: oQ4e plain ≈ 62–65 ms = 15.4–16.3 tok/s, i.e. 0.90–0.95 of the
uniform 4-bit (bytes + launches + the 6-bit head).

## DFlash: what fell to slow paths in the verify block (task 2c)

A round = drafter block + one target forward of T = 8 rows (`_verify_linears` / `_verify_head`
through mrqmm when `QWEN_VERIFY_MRQMM=1`). On oQ4e with `QWEN_MRQMM4=1 QWEN_MRQMM5=1`, before this
change:

| module(s) | width | path before | est. cost / round | after |
|---|---|---|---|---|
| target lm_head | 6 | mlx-vlm `optimized_affine_linear` supports 4/5/8 only → `singleton_quantized_linear` (gather_qmm per token = the 1.0 GiB head read 8 times) | ~+27 ms | mrqmm 6-bit (`QWEN_MRQMM6=1`), one read |
| drafter's lm_head (= the target's) | 6 | `_verify_head` wraps only `MRQuantizedLinear` → `quantized_matmul` qmm at 7 rows | ~+9.5 ms | mrqmm 6-bit |
| layer 63 k / v | 6 | singleton | small (N = 1024) | mrqmm 6-bit |
| layer 52 a | 6 | singleton | small (N = 48) | still exact / singleton (N < MIN_N) |
| GDN qkv, z | 4 / 5 | separate mrqmm calls (no fusion) | ~+1–2 ms | one call per width group |
| GDN b / a (N = 48) | 5 / 8 | mlx-vlm exact kernels | ~+1 ms | unchanged (N < MIN_N; b+a fused into one `quantized_matmul` in plain decode) |
| everything else 5-bit | 5 | mrqmm 5-bit (MMA-bound at 8 rows, +17 % vs 4-bit on ~21 % of the FLOPs) | ~+4 ms | inherent |

The rest of the ~110 ms round gap is H0. The acceptance gap (3.57 vs 4.8 drafts per round) is a
property of the model and the text (the drafter was trained against the full-precision target, and
the two builds round it differently), not of the kernels: verify through mrqmm does not change which tokens are
accepted beyond near-ties. The gate measures both builds on the same corpus (ab27) to separate it.

## What was implemented

1. **`QWEN_MRQMM6` (default 0): a 6-bit weight side for mrqmm** (`src/worker/mrqmm.py`,
   `_WSIDE[6]`). MLX packs 6 bits as 4 values per 3 bytes, LSB-first. A thread's 16 weights of a
   64-group are 96 bits = 3 aligned uint32 words at `12 g + 3 (fn / 2)`; they are regrouped once
   per group into 4 lane words (weights 2j, 2j+1 of each half in 16-bit lanes, ~10 int ops), and each
   step is one shift + mask (`(wq[st / 2] >> 6 (st % 2)) & 0x003f003f`) into the 0x6400 fp16
   trick (exact for q ≤ 63). Config as 4 / 5 bits (4×8 at ≤ 8 rows, 2×8 above; `--sweep` in the
   bench re-checks it). Self-tested at load like 4 bits (`SELFTEST_BITS = (4, 6)`): unpack exact
   on one-hot x at every in-group offset, both row buckets, real-scale rows vs fp32 and
   `quantized_matmul`; a failure logs `self-test FAILED` and leaves 6-bit modules on
   `quantized_matmul`. Every mrqmm user (`MRQuantizedLinear`, `_gdn_call`, `_verify_linear(s)`,
   `_verify_head`) takes it through `mrqmm._bits_on` + `supported`, unchanged.
   *Rejected alternative*: re-quantize the head to 8 bits at load (+0.3 GiB, a second rounding of
   the weights, and +0.9 ms on every plain token since plain decode reads the same head: measured).
2. **GDN in-projection fusion by width groups** (`src/worker/qwen_worker/kernels.py`):
   `_gdn_groups` partitions qkv / z / b / a by (bits, group_size, mode) among eligible modules
   (QuantizedLinear, no bias, fp16 scales / biases); every key with ≥ 2 modules becomes one
   concatenation, members point at row views of it (no extra memory); a projection alone at its
   width keeps its own weights and call. `_fused_in` = a tuple of `_InGroup`s (`names, w, s, b,
   split, gs, bits, mode, mr`); `mr` is set only when the width's kernel is on and supported and
   N ≥ `MIN_N` (or batch-invariant mode). A uniform layer gives exactly one group of four in
   qkv, z, b, a order: the old concatenation, split and call, bit for bit.
   `_install_verify_mrqmm` / `_verify_linears` run a DFlash verify block per kernel group too
   (`_verify_fused`), the rest through the per-module path; `_invariant_uncovered` reports a group
   without the kernel as "fused GDN in-projection on quantized_matmul". Load log on oQ4e
   (expected): `GDN in-projections fused by width: 30 x [qkv+z 4-bit, b+a 5-bit]; 15 x [z+b+a
   5-bit, qkv alone]; 2 x [qkv+z 4-bit, b+a 8-bit]; 1 x [qkv+z 4-bit, b alone, a alone]` and
   `... fused (QWEN_GDN_FUSE): 48 layers`.
   Numerics: B = 1 bit-identical to the unfused calls (qmv rows do not depend on N; CPU-tested on
   a tiny model in the oQ4e layout); B = 2 bit-identical on the GPU too (qmv_wide; passed in
   `test_worker_gdn_fuse_mixed`, gate step 1); 3..16 rows on mrqmm = the batched-decode class as before.

## Measured: gate steps 1–3 (2026-11-15, M1 Max, no model, both checkpoints' tensors in one process)

- Step 1: `tests/test_mrqmm.py` on Metal 390 passed, 4 skipped (the skips are the real-weights
  tests). Step 2: `test_real_oq4e_weights` passed (self-test + `check_close` on the real 6-bit
  head, layer 63 k / v, layer 52 qkv+z 4-bit, layer 0 z+b+a 5-bit).
- **Plain decode, the linears of one forward at 1 row** (`oq4e_bench.py` qmv phase, interleaved,
  5 rounds): oq4e unfused 48.66 ms, oq4e width groups 48.39, u4 unfused 46.21, u4 fused (today)
  45.16. **oq4e / u4 = 1.078 today, 1.072 with the groups**; the u4 fusion is worth ×1.023 (as on
  the 9B), oQ4e's groups only 0.27 ms (0.6 %): the 4-launch cost was smaller than H1 assumed.
  The 6-bit head's qmv streams at 349 GB/s vs 352 for the 4-bit head (H3 refuted); 5-bit modules
  run 262–333 GB/s against 267–333 for the same 4-bit shapes (H4 refuted). So the linears predict
  oQ4e plain at ~0.93 of the uniform model (17.2 → ~16 tok/s), not the 0.68 measured
  (11.7 vs 17.2): **~25 % of the plain gap is outside the linears' kernels** — the cross-process
  drift (H0) or the residency of the loaded weights (H5), which only a model run can tell (step 5,
  `decode_probe.py`: tok/s per probe GB/s in the same process, A B A B).
- **DFlash verify block, T = 8, linears of one forward:**

  | | total | lm_head | drafter head | GDN in | rest |
  |---|---|---|---|---|---|
  | oq4e before (mlx-vlm 6-bit singleton, qmm head, unfused) | 117.84 ms | 22.83 | 11.58 | 17.45 | 65.98 |
  | oq4e after (`QWEN_MRQMM6=1` + width groups) | **87.58** | 4.02 | 3.99 | 14.97 | 64.60 |
  | u4 unfused | 89.10 | 3.47 | 3.36 | 19.45 | 62.82 |
  | u4 today (fused) | 80.80 | 3.43 | 3.37 | 12.29 | 61.71 |

  −30.3 ms per round on oQ4e (×1.35 on the block's linears); oq4e / u4 1.458 → 1.084. The round
  gap measured end-to-end was ~110 ms (259–274 vs 153), of which the linears explain 37 ms
  before the change and 7 ms after: as for plain decode, the rest is H0 / H5 or outside the
  linears. Expected round after: ~153 + 7 ms if the rest of the round matched the uniform model
  (step 6 measures it).
- **lm_head paths (6-bit, 248320 × 5120):** qmv 1 row 3.00 ms; mlx-vlm singleton × 8 24.10 ms;
  `quantized_matmul` at 7 / 8 rows 14.49 / 14.41 ms; mrqmm 7 / 8 rows 4.94 / 5.05 ms. Config
  sweep at 8 rows: 4×8 5.05, 4×4 5.03, 2×8 5.48 → the 4 / 5-bit choice (4×8) stays. The rejected
  8-bit re-quantized head: qmv 3.88 ms (+0.88 ms on every plain token), mrqmm 8 rows 4.81 ms
  (−0.24 ms per round), +0.30 GiB, weight rms rel error of the second rounding 5.6e-3.
- **Accuracy** (rms rel vs an fp32 reference at 3 / 8 / 16 rows): mrqmm 2.1–2.7e-4 at every width
  and shape (6-bit 2.65e-4), ≤ `quantized_matmul`'s own error everywhere (qmm 2.1e-4–1.3e-3; qmm
  at 3 rows runs per-row qmv, hence its larger 5 / 6-bit error); rows independent of the row
  count everywhere mrqmm is used. The one exception, `b+a` 8-bit (N = 96), is below `MIN_N` and
  never takes mrqmm in the worker (it stays on `quantized_matmul`).

Re-ranked plain-decode diagnosis: **H0 (drift) / H5 (residency)** — together ~25 %, to be split
by step 5; H2 (bytes) ~5–7 % (inherent); H1 (fusion) 0.6 %; H3, H4 ≈ 0.

## Tests

- CPU (`cd src/worker && .venv/bin/python -m pytest -q --ignore tests/test_mrqmm.py`):
  `tests/test_mrqmm_bits.py` (6-bit unpack mirrored in numpy and checked against the kernel source,
  knob / install / self-test for 6 bits, decode + verify dispatch parametrized over 5 / 4 / 6
  bits); new `tests/test_mixed_bits.py`: a tiny checkpoint written to disk in the oQ4e layout
  (base 4 bits + per-module overrides by full path in config.json, as oQ4e stores them), loaded by
  the worker's own loader: per-module widths, the width groups and row views, the load log line,
  fused == unfused (bitwise at B = 1, close at 2 and 5 on the CPU), the decode and verify-block
  dispatch per group with and without `QWEN_MRQMM6`, `_verify_head` on the 6-bit head, the
  batch-invariant coverage report, and a uniform model as one group.
- GPU (`tests/test_mrqmm.py`, skipped without Metal): 6-bit kernel vs reference at 27B shapes,
  every K offset, every config with N / K tails, the self-test at 5120 × 1024 and 248320 × 5120,
  rows independent of the row count, dispatch boundaries; the mixed tiny model through the worker
  at B = 1 / 2 (bitwise) / 5 / 16; with `MRQMM_REAL_WEIGHTS=1 MRQMM_WEIGHTS_OQ4E=<dir>` the real
  oQ4e tensors (lm_head, layer 63 k / v, a qkv+z 4-bit and a z+b+a 5-bit group).

## GPU gate (`scripts/perf/dflash2/gate_oq4e.sh`, run from the repo root on an idle GPU)

```zsh
sudo sysctl iogpu.wired_limit_mb=28672                  # steps 4-7 load one 27B per process
setopt NO_BG_NICE
PERF_OUT=/tmp/perf scripts/perf/dflash2/gate_oq4e.sh 1 2 3     # no model: unit tests, real tensors, microbench
PERF_OUT=/tmp/perf scripts/perf/dflash2/gate_oq4e.sh 4 5 6 7 8 # models, one per process
```

Steps (the script's header has the details and the expected outcomes):

1. `pytest tests/test_mrqmm.py` — all pass.
2. `MRQMM_REAL_WEIGHTS=1 MRQMM_WEIGHTS_OQ4E=weights/qwen38-27b-mlx-oq4e pytest tests/test_mrqmm.py
   -k test_real_oq4e_weights` — self-test + check_close on the real 6-bit head, k / v and groups.
3. `scripts/perf/dflash2/oq4e_bench.py --a weights/qwen38-27b-mlx-oq4e --b
   weights/qwen38-27b-mlx-4bit` — both checkpoints' tensors in ONE process (drift-free ratios):
   1-row qmv GB/s per (kind, width) (H3 / H4), forward linears oq4e/none vs oq4e/groups vs
   u4/groups (the plain-decode prediction), the T = 8 verify block per path, the lm_head paths
   (6-bit singleton ×8 vs mrqmm 8 rows, the 8-bit re-quantized head), accuracy per width.
4. `step_prof.py --B 1 --ab base,nofuse` on each model — the fusion's in-process gain at B = 1
   (expect −2…4 % on oQ4e; the uniform model's old −2 %).
5. `decode_probe.py` A B A B A B with cooldowns — plain tok/s per 100 GB/s of an in-process qmv
   probe. Expect oq4e / u4 ≈ 0.90–0.95; 0.68 would mean H0 was not the cause (then H5).
6. `ab27.py --variants plain,m8/8,e0/8` on oQ4e with `QWEN_MRQMM6=1` vs `0` — round 259–274 →
   ~160–180 ms expected, m8/8 / plain ×1.45 → ≥ ×2.
7. Identity: oQ4e `plainnf` == `plain` 16/16 (the grouped fusion at B = 1), `m8/8` vs plain ≥ 14/16
   (batched-decode class, divergences at near-ties); uniform 4-bit `plainnf` == `plain` 16/16.
8. Load logs: the 6-bit self-test passed, the width-group line, 48 layers fused.

`QWEN_MRQMM6` stays default 0 until steps 1, 2, 6 and 7 pass.

## Uncertain

- What the remaining ~25 % of the plain gap (and most of the DFlash round gap) is: the linears
  measured drift-free say 0.93, the separate-process runs said 0.68. Drift (H0) or the loaded
  weights' residency (H5) — step 5 splits them; if the probe-normalized ratio stays ~0.7, look at
  page-ins / `footprint` during decode and at what the loader keeps mmap'd.
- B ≤ 2 bitwise of the grouped fusion is shown on the tiny mixed model on the GPU (step 1); on the
  real model it is step 7 (`plainnf` == `plain`, B = 1).
- The acceptance gap (3.57 vs 4.8) is attributed to the model; step 6 measures it on ab27's corpus
  for oQ4e and the uniform model's earlier numbers come from other prompts.
- The 27B 4-bit / 5-bit batched decode (B ≥ 3) through mrqmm is still ungated in general
  (`QWEN_MRQMM4/5` default 0); this work does not change that.

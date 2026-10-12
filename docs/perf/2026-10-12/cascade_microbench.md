# Cascade (grouped shared-prefix) decode attention: microbench

Date 2026-10-12, M1 Max 32 GB, mlx 0.32.3. Tool: `scripts/perf/microbench/microbench_cascade.py`
(run from `src/worker` with its venv, `setopt NO_BG_NICE`; the idle 9B worker held ~9.5 GB and did
not compute; nothing else ran on the GPU). Raw: `raw/cascade_microbench.{json,txt}` (run 3, all
variants), `raw/cascade_microbench_run2.{json,txt}` (run 2, same code minus `b3`/`b3v`). Each run
took ~2 min. This decides proposal 1 of `design/prompt_cache_opt.md`.

Synthetic fp16 tensors in the 9B's full-attention shape: 16 q heads, 4 kv heads, head_dim 256,
8 full-attention layers. B rows share a prefix of P tokens, and each row owns a suffix of
len_i = S − (i mod 8) tokens (ragged). There is one query per row. A "step" is 8 chained layer calls:
layer l's query depends on layer l−1's output, as in the model. Each step is one `mx.eval` +
`mx.synchronize`. The layer data cycles over enough copies that the working set is larger than the
48 MB SLC. The variants are interleaved inside each round, with the order rotated per round: 2
warm-up steps, then 5 rounds × 6 timed steps. The tables give the median of the round medians.

## Verdict

**Build it, but only for groups of ≥ 8 rows on a prefix of ≥ ~4K tokens.** Use the fp32 formulation
(`b32c`), which is no worse than today's kernel numerically. Do not group B ≤ 4, short prefixes,
or rows whose own suffix is comparable to the prefix.

* **B=16, P=8K (the prompt-cache benchmark).**
  * Attention plus cache update goes from 22–24 ms to 9 ms per step at S=512: **13–16 ms saved**
    (14–16 at S=64, 11–12 at S=2048).
  * The measured step at ~8.3K/B=16 is 67–80 ms (`prompt_cache_bench.md` A1a/A1b), so it becomes
    ~52–66 ms. That is **+20–28 % decode tok/s**, in line with the design doc's +25–30 %.
* **B=16, P=32K.**
  * 64–68 ms becomes 18–19 ms per step: **46–49 ms saved**.
  * Today's step at 32K/B=16 is ≈ 52 − 8 + 66 ≈ 110 ms (the 1K step with the 1K attention swapped
    for the 32K one). That is ~62 ms with the cascade: **×1.7–1.8 decode**.
  * It also makes 16 × 32K admittable. The rows' KV drops from 16.3 GiB to 2.5 GiB (2.25 with an fp16 suffix), see Memory.
* **B=8**: +~9 % at 8K (3–4 ms of ~40 ms), +35–50 % at 32K (16–20 ms saved).
  **B=4**: ≤ +5 % at 32K, zero or negative at ≤ 8K. **P=1K**: +5–8 % at B=16 with short suffixes.
  **Any B at P=1K, S=2048**: a loss of 0.3–4 ms per step, because the suffix part dominates and it is
  slower than today's kernel (see Pitfalls).
* **Numerics.** `b32c` beats (a) against an fp32 reference in **54/54** cases: rms rel error
  2.03–2.12e-4 vs (a)'s 2.70–2.95e-4. It differs from (a) in 33–43 % of the fp16 outputs, by ≤ a few
  ulp: max |Δ| 1.2e-4 on flat scores, 3.9e-3 on peaked ones where outputs are larger. It is not
  bit-equal to a lone row. Same acceptance class as mrqmm: gate with `agree.py`.
  * The **fp16 variant (`b16`) fails**: 2× (a)'s error on flat scores and 4–10× on peaked ones. It
    is 6–17 % faster.
* **Memory** at B=16 / P=8K / S=512 (fp32 prefix shared + fp32 suffix): 272 → 64 MiB per row
  (8 layers), 48 MiB with an fp16 suffix; 4.25 → 1.0 GiB for all 16 rows. The transient is ~0.17 GB per step at 8K and ~0.6 GB at 32K, against ≤ 55 MB today.

## Variants

| name | what |
|---|---|
| `a` | **today**: per row `mx.fast.scaled_dot_product_attention(q_i (1,16,1,256), K_i (1,4,P+len_i,256), V_i)` on the row's own contiguous KVCache buffer, concat — `_rows_attention` minus the update |
| `a_u` | `a` + `KVCache.update_and_fetch` per row (exactly `_rows_attention`) |
| `b16` / `b16c` | cascade, fp16 storage: queries stacked (4 kv heads, 4B, 256), prefix keys viewed as (4, P/Pc, Pc, 256) chunks (Pc = 1024, flash-decoding split), fp16 matmuls, per-chunk `softmax(precise)` + fp32 `logsumexp`, chunk merge and prefix/suffix merge by LSE in fp32; suffix in a stacked (B,4,S,256) buffer with an additive ragged mask, same manual formulation. `c` = `mx.compile` of the whole function |
| `b32` / `b32c` | the same with fp32 prefix and suffix storage: fp32 scores, softmax and P·V — the same products as MLX's decode kernel, summed in another order |
| `b2` | `b32c` prefix; suffix by the fused SDPA (fp16, bool mask) + a separate fp32 scores matmul + logsumexp for its LSE |
| `b3` | `b32c` prefix; suffix by the fused SDPA with the **prefix as an attention sink** (see "LSE from the fused kernel"): two fused calls, no suffix LSE |
| `b3v` | `b3` in one call, V stored with 8 ones channels (d_v 264) |
| `b32m` | `b32c` with a prefix of P − 7 tokens: buffer padded to a multiple of Pc + static additive mask on the scores |
| `b32u` | `b32c` + the suffix update (one `put_along_axis` per K and V for all rows); compare with `a_u` |

`a_u` and `b32u` evaluate min(8, copies) layers per graph, scaled to 8. When a copy is updated twice
in one graph its buffer is read in between and stops being donatable, so the second update copies
it. In run 1 that artifact added up to +8 ms. In the model every layer's cache is updated once per
step.

## Timing: ms per 8-layer step (run 3; /8 = per layer)

| B | P | S | a | a_u | b16 | b16c | b32 | b32c | b2 | b3 | b3v | b32m | b32u |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1K | 64 | 1.7 | 1.8 | 2.4 | 2.1 | 2.6 | 2.2 | 2.6 | 2.5 | 2.2 | 2.3 | 2.3 |
| 4 | 1K | 512 | 1.9 | 2.1 | 3.2 | 2.9 | 3.5 | 2.7 | 3.0 | 3.2 | 2.8 | 2.8 | 2.8 |
| 4 | 1K | 2048 | 2.8 | 3.2 | 5.2 | 4.8 | 5.5 | 5.0 | 4.7 | 4.4 | 4.4 | 5.1 | 5.3 |
| 4 | 8K | 64 | 5.8 | 6.6 | 4.7 | 5.4 | 5.6 | 5.0 | 7.1 | 5.4 | 5.2 | 5.4 | 5.9 |
| 4 | 8K | 512 | 5.7 | 7.0 | 5.3 | 6.1 | 6.2 | 5.5 | 6.7 | 5.9 | 5.6 | 5.8 | 6.4 |
| 4 | 8K | 2048 | 6.4 | 8.2 | 7.0 | 8.6 | 8.8 | 7.6 | 8.7 | 7.1 | 7.1 | 7.7 | 8.9 |
| 4 | 32K | 64 | 14.7 | 16.9 | 11.2 | 10.8 | 13.4 | 12.7 | 12.7 | 12.9 | 12.5 | 13.2 | 14.1 |
| 4 | 32K | 512 | 15.7 | 18.0 | 11.6 | 11.8 | 13.8 | 14.0 | 14.0 | 14.5 | 13.9 | 14.3 | 16.9 |
| 4 | 32K | 2048 | 15.5 | 17.9 | 13.5 | 13.5 | 15.9 | 15.4 | 15.3 | 14.7 | 14.6 | 15.9 | 16.4 |
| 8 | 1K | 64 | 3.8 | 4.7 | 4.0 | 3.7 | 4.4 | 3.9 | 3.5 | 3.6 | 3.3 | 3.2 | 3.3 |
| 8 | 1K | 512 | 6.3 | 5.5 | 6.0 | 4.2 | 5.7 | 4.7 | 4.7 | 5.9 | 4.9 | 5.7 | 7.8 |
| 8 | 1K | 2048 | 6.3 | 8.8 | 7.6 | 7.1 | 9.8 | 7.4 | 7.4 | 6.9 | 6.7 | 7.5 | 9.1 |
| 8 | 8K | 64 | 11.1 | 12.3 | 5.3 | 4.5 | 6.3 | 5.7 | 5.8 | 6.0 | 5.7 | 6.6 | 9.0 |
| 8 | 8K | 512 | 10.7 | 13.5 | 6.4 | 5.5 | 7.2 | 6.7 | 6.7 | 8.0 | 7.0 | 7.1 | 9.9 |
| 8 | 8K | 2048 | 11.3 | 14.6 | 9.0 | 9.3 | 10.9 | 10.6 | 10.3 | 9.7 | 9.6 | 11.0 | 12.6 |
| 8 | 32K | 64 | 29.2 | 33.5 | 12.4 | 11.8 | 14.3 | 14.1 | 13.6 | 13.7 | 13.6 | 14.7 | 15.6 |
| 8 | 32K | 512 | 29.4 | 37.8 | 13.2 | 12.7 | 15.4 | 15.3 | 15.0 | 14.8 | 14.5 | 15.7 | 17.2 |
| 8 | 32K | 2048 | 30.9 | 42.1 | 16.4 | 17.9 | 20.2 | 19.9 | 20.3 | 19.4 | 18.7 | 20.4 | 23.4 |
| 16 | 1K | 64 | 8.1 | 9.3 | 4.6 | 3.6 | 5.6 | 4.6 | 4.9 | 5.3 | 6.3 | 7.1 | 5.0 |
| 16 | 1K | 512 | 8.4 | 11.8 | 6.4 | 5.5 | 6.8 | 5.8 | 5.7 | 5.8 | 5.7 | 6.7 | 9.0 |
| 16 | 1K | 2048 | 10.3 | 12.0 | 12.4 | 11.7 | 14.9 | 13.3 | 13.8 | 12.1 | 11.2 | 13.6 | 15.9 |
| 16 | 8K | 64 | 21.6 | 24.0 | 5.8 | 5.3 | 5.9 | 6.4 | 7.2 | 7.4 | 6.9 | 6.5 | 8.3 |
| 16 | 8K | 512 | 20.2 | 24.2 | 7.3 | 6.8 | 8.1 | 7.7 | 8.0 | 7.2 | 6.7 | 8.2 | 9.1 |
| 16 | 8K | 2048 | 25.5 | 31.1 | 14.7 | 14.3 | 17.4 | 16.6 | 16.9 | 14.9 | 13.9 | 17.1 | 19.0 |
| 16 | 32K | 64 | 58.3 | 71.5 | 14.2 | 13.2 | 14.1 | 14.1 | 14.1 | 14.3 | 14.7 | 16.4 | 19.2 |
| 16 | 32K | 512 | 58.5 | 68.4 | 15.7 | 15.1 | 17.6 | 16.5 | 16.6 | 16.4 | 15.8 | 17.8 | 19.4 |
| 16 | 32K | 2048 | 61.1 | 68.5 | 22.0 | 22.0 | 24.8 | 25.2 | 25.1 | 24.0 | 22.9 | 27.2 | 27.3 |

Run 3 was noisier than run 2 at P=1K for B=8/16 (`a` 3.8 vs 2.9 and 8.1 vs 6.5). At 8K and 32K the
runs agree within ~5–10 %. The comparison that matters, today's attention with its update vs the
cascade with its update, from both runs:


| B | P | S | a (r2/r3) | a_u (r2/r3) | b32c (r2/r3) | b32u (r2/r3) | saved a_u − b32u, ms/step (r2/r3) |
|---|---|---|---|---|---|---|---|
| 4 | 1K | 64 | 1.7 / 1.7 | 1.8 / 1.8 | 2.2 / 2.2 | 2.3 / 2.3 | -0.4 / -0.4 |
| 4 | 1K | 512 | 1.9 / 1.9 | 2.1 / 2.1 | 2.6 / 2.7 | 2.8 / 2.8 | -0.7 / -0.7 |
| 4 | 1K | 2048 | 2.8 / 2.8 | 3.2 / 3.2 | 4.9 / 5.0 | 5.3 / 5.3 | -2.1 / -2.0 |
| 4 | 8K | 64 | 5.8 / 5.8 | 6.7 / 6.6 | 5.0 / 5.0 | 6.2 / 5.9 | 0.5 / 0.6 |
| 4 | 8K | 512 | 5.7 / 5.7 | 6.6 / 7.0 | 5.5 / 5.5 | 6.5 / 6.4 | 0.1 / 0.6 |
| 4 | 8K | 2048 | 6.5 / 6.4 | 8.6 / 8.2 | 7.6 / 7.6 | 9.2 / 8.9 | -0.6 / -0.7 |
| 4 | 32K | 64 | 14.4 / 14.7 | 16.3 / 16.9 | 12.7 / 12.7 | 13.8 / 14.1 | 2.5 / 2.8 |
| 4 | 32K | 512 | 14.5 / 15.7 | 16.2 / 18.0 | 13.2 / 14.0 | 14.3 / 16.9 | 1.9 / 1.2 |
| 4 | 32K | 2048 | 15.2 / 15.5 | 17.1 / 17.9 | 15.4 / 15.4 | 16.6 / 16.4 | 0.4 / 1.6 |
| 8 | 1K | 64 | 2.9 / 3.8 | 3.4 / 4.7 | 2.7 / 3.9 | 2.5 / 3.3 | 0.8 / 1.4 |
| 8 | 1K | 512 | 3.5 / 6.3 | 4.4 / 5.5 | 3.3 / 4.7 | 3.7 / 7.8 | 0.7 / -2.2 |
| 8 | 1K | 2048 | 5.4 / 6.3 | 7.0 / 8.8 | 7.2 / 7.4 | 8.3 / 9.1 | -1.3 / -0.3 |
| 8 | 8K | 64 | 10.9 / 11.1 | 11.5 / 12.3 | 5.5 / 5.7 | 6.7 / 9.0 | 4.8 / 3.3 |
| 8 | 8K | 512 | 10.1 / 10.7 | 12.2 / 13.5 | 6.7 / 6.7 | 8.4 / 9.9 | 3.8 / 3.6 |
| 8 | 8K | 2048 | 11.5 / 11.3 | 14.9 / 14.6 | 10.7 / 10.6 | 12.2 / 12.6 | 2.7 / 2.0 |
| 8 | 32K | 64 | 28.0 / 29.2 | 30.5 / 33.5 | 13.9 / 14.1 | 14.9 / 15.6 | 15.6 / 17.8 |
| 8 | 32K | 512 | 28.7 / 29.4 | 32.6 / 37.8 | 14.8 / 15.3 | 17.0 / 17.2 | 15.6 / 20.7 |
| 8 | 32K | 2048 | 29.7 / 30.9 | 32.2 / 42.1 | 18.8 / 19.9 | 20.0 / 23.4 | 12.2 / 18.6 |
| 16 | 1K | 64 | 6.5 / 8.1 | 7.2 / 9.3 | 3.2 / 4.6 | 3.2 / 5.0 | 4.0 / 4.3 |
| 16 | 1K | 512 | 6.7 / 8.4 | 8.4 / 11.8 | 4.7 / 5.8 | 5.8 / 9.0 | 2.6 / 2.8 |
| 16 | 1K | 2048 | 10.0 / 10.3 | 10.9 / 12.0 | 12.3 / 13.3 | 13.6 / 15.9 | -2.7 / -3.9 |
| 16 | 8K | 64 | 19.8 / 21.6 | 20.9 / 24.0 | 5.7 / 6.4 | 6.9 / 8.3 | 14.0 / 15.7 |
| 16 | 8K | 512 | 19.8 / 20.2 | 22.1 / 24.2 | 7.6 / 7.7 | 8.9 / 9.1 | 13.2 / 15.1 |
| 16 | 8K | 2048 | 23.4 / 25.5 | 27.4 / 31.1 | 16.0 / 16.6 | 16.7 / 19.0 | 10.7 / 12.2 |
| 16 | 32K | 64 | 55.4 / 58.3 | 62.0 / 71.5 | 14.0 / 14.1 | 15.0 / 19.2 | 47.0 / 52.3 |
| 16 | 32K | 512 | 55.9 / 58.5 | 64.0 / 68.4 | 16.3 / 16.5 | 17.6 / 19.4 | 46.4 / 49.0 |

### Fraction of the decode step

| B | P (ctx) | today's step (anchor) | `a_u` today | `b32u` cascade | saved | step after | decode tok/s |
|---|---|---|---|---|---|---|---|
| 16 | 1K | 52 ms (CLAUDE.md, mrqmm) | 7–9 ms (14–18 %) | 3–5 ms | 4 ms | 48 ms | +8 % |
| 16 | 8K (S≈64–512) | 67–80 ms (prompt_cache_bench) | 21–24 ms (27–36 %) | 7–9 ms | 13–16 ms | 52–66 ms | **+20–28 %** |
| 16 | 8K, S=2048 | ~70–85 ms | 27–31 ms | 17–19 ms | 11–12 ms | | +15–17 % |
| 16 | 32K (S=512) | ≈ 110 ms (derived: 52 − 8 + 66) | 64–68 ms (~60 %) | 18–19 ms | 46–49 ms | ~62 ms | **×1.75** |
| 8 | 8K | ~40 ms (34 at 1K + ~8) | 12–13.5 ms | 8.5–10 ms | 3.5 ms | ~36 ms | +9 % |
| 8 | 32K | ~60 ms (derived) | 33–38 ms | 15–17 ms | 16–20 ms | ~42 ms | +35–50 % |
| 4 | 8K / 32K | 31 ms at 1K | 6.6–7 / 16–18 ms | 6.4 / 14–17 ms | 0–0.6 / 1–3 ms | | ≤ +5 % |

The 8K/B=16 `a_u` (21–24 ms) matches step_profile.md's 27 % of the step (~24 ms). At 1K the
microbench gives 14–18 %, against step_profile's 7.5 % from removing attention from the real
graph. In the full step, small per-row kernels overlap other work, so the 1K gain is probably
smaller than shown.

## Where the time goes (roof)

These are the chunked prefix matmuls alone: 8 calls per graph, per call, Pc = 1024. GB/s is the K
(or V) bytes over the time.

| B | P | dtype | QK ms | QK GB/s | QK TFLOPS | PV ms | PV GB/s | PV TFLOPS |
|---|---|---|---|---|---|---|---|---|
| 4 | 32K | fp16 | 0.572 | 117 | 1.88 | 0.539 | 124 | 1.99 |
| 4 | 32K | fp32 | 0.702 | 191 | 1.53 | 0.662 | 203 | 1.62 |
| 8 | 32K | fp16 | 0.581 | 116 | 3.70 | 0.547 | 123 | 3.92 |
| 8 | 32K | fp32 | 0.722 | 186 | 2.98 | 0.675 | 199 | 3.18 |
| 16 | 8K | fp16 | 0.167 | 100 | 6.42 | 0.157 | 107 | 6.83 |
| 16 | 8K | fp32 | 0.191 | 176 | 5.63 | 0.177 | 189 | 6.05 |
| 16 | 32K | fp16 | 0.558 | 120 | 7.70 | 0.512 | 131 | 8.38 |
| 16 | 32K | fp32 | 0.633 | 212 | 6.79 | 0.587 | 229 | 7.32 |

(All B × P × dtype in `raw/cascade_microbench.txt`.)

* **The time does not depend on B (4 → 16).** MLX runs these (4B × 256)·(256 × Pc) GEMMs in 64-row
  tiles, so 16 query rows cost what 64 do.
* **B=16** is compute-bound near MLX's GEMM peak: 7.7–8.4 TFLOPS fp16, 6.8–7.3 fp32. Arithmetic
  intensity is 64 FLOP per fp16 byte against a ridge of ~26, at 120–230 GB/s. B=4/8 pays for the
  padded tile.
* **fp32 storage costs only +10–25 % time** for twice the bytes. That is why `b32c` is within ~15 %
  of `b16c`.
* **Chunking P·V is mandatory.** Unchunked, MLX's (4B × P)·(P × 256) GEMM has 64 output tiles at
  most and a K loop of P. Scratch probe at 32K/B=16, prefix only, ms per layer: unchunked 2.56 (fp16)
  / 2.46 (fp32); Pc 4096 1.93 / 1.77; 2048 1.80 / 1.58; **1024 1.72 / 1.50**; 512 1.73 / 1.55.
* At 32K/B=16 the cascade spends ~1.75 ms per layer: QK + PV 1.2 ms, plus ~0.5 ms for the score
  passes (softmax read/write, logsumexp read, ~34 MB of fp32 scores per layer).
  * A Metal flash-decoding kernel could save those passes and the B<16 tile padding.
  * It cannot go below the ~0.9 ms per layer compute floor at B=16 / 32K, so it is worth at most
    another ~4 ms per step there.
  * Its real value would be the suffix (next section).

## Numerics

One layer per (B, P, S), against an fp32 reference of (a) (the fused kernel on fp32 copies). There
are two regimes:

* flat: q ~ N(0,1), score std ~1;
* peaked: q ~ N(0,16), score std ~4.

rms rel = rms(out − ref) / rms(ref). All 27 configurations:

| variant | flat rms rel (min..max) | peaked rms rel | max \|out − ref\| flat / peaked | differs from (a) | worse than (a) |
|---|---|---|---|---|---|
| `a` (today) | 2.89–2.95e-4 | 2.70–2.95e-4 | 1.3e-4 / 2.1e-3 | — | — |
| `b32` / `b32c` | **2.05–2.10e-4** | **2.03–2.12e-4** | 6.1e-5 / 1.8e-3 | 33–43 % of elements, max 1.2e-4 / 3.9e-3 | **0 / 54** |
| `b32m` (padded prefix) | 2.03–2.11e-4 | 2.03–2.12e-4 | 9.3e-5 / 1.8e-3 | — | 0 / 54 |
| `b2` (fused suffix + LSE matmul) | 2.06–3.18e-4 | 2.04–3.16e-4 | 8.5e-5 / 2.6e-3 | 34–52 % | 6 / 54 (all S=2048, P=1K) |
| `b3` (sink trick, 2 calls) | 2.06–3.99e-4 | 2.06–5.69e-4 | 8.1e-5 / 4.5e-3 | 36–58 % | 15 / 54 (mostly S ≥ 512 with P ≤ 8K) |
| `b3v` (sink, 1 call, d_v 264) | 2.27e-4–1.16e-3 | 2.08e-4–1.40e-3 | 1.9e-4 / 7.1e-3 | 44–82 % | most |
| `b16` / `b16c` | 5.64–6.77e-4 | **1.16–2.89e-3** | 3.1e-4 / 9.9e-3 | 67–91 % | 54 / 54 |

* **`b32c` sits at the fp16 output-rounding floor (2.07e-4).** That is lower than (a): MLX's 2-pass
  decode kernel adds error of its own (2.9e-4), probably from its fp16 intermediate partials.
* **`b16` rounds the scores to fp16 in the matmul output**: ulp 2^-11 × |s| ≈ 0.005 at |s| ≈ 10.
  That error goes straight into the softmax weights, and the peaked regime shows it most. This is
  the precision of MLX's *unfused* SDPA, not of its decode kernel. Rejected.
* **`b2` / `b3` round the suffix part in fp16** (the fused kernel's output). `b3` also rounds w_s
  before 1 − w_s. That costs precision exactly when the suffix carries much of the mass.

## LSE from the fused kernel: the sink trick (`b3`)

`mx.fast.scaled_dot_product_attention` returns no LSE. An extra value column is not possible: head
dim 257 / 264 has no fused kernel, and `b3v` falls back to an unfused fp16 graph with 2–5× the error.

`sinks` does work, after a reshape:

* **Reshape so sinks are per (row, head).** `sinks` must have shape (H,) and be fp16. Fold the rows
  into the head axis: q (1, B·16, 1, 256), K/V (1, B·4, S, 256). The GQA mapping is preserved
  (q head b·16+h → kv head b·4+h//4), and `sinks` now has one entry per (row, head).
* **The two calls.** With sink = lse_p the kernel returns A = Z_s·o_s / (Z_s + Z_p). A second call
  with V = ones (8 channels) returns w_s = Z_s / (Z_s + Z_p). Then out = A + (1 − w_s)·o_p. No
  suffix LSE is needed.
* **Keeping the sink exact.** An fp16 sink would round lse_p (~10: ulp 0.008, so 0.4 % weight
  error). So c = fp16(lse_p) goes into the additive float mask (−c on valid slots, −inf on the
  rest), which the kernel subtracts exactly in fp32. The sink carries only the residual lse_p − c.

It works: rms rel 2.1e-4 when the prefix dominates. It is the fastest acceptable formulation at
S=2048: 12.1 vs `b32c` 13.3 ms at P=1K/B=16, and 14.9 vs 16.6 at 8K. It keeps the suffix in fp16
(half the suffix memory of `b32c`). Its fp16 rounding of A and w_s makes it worse than (a) when the
suffix carries real mass (15/54). Keep it as a fallback idea, not the default.

## Memory (resident KV per row, all 8 layers)

Today a row holds (P + S) × 32 KiB (+ ≤ 255 tokens of capacity rounding). The cascade holds
P × 64 KiB / B (fp32 prefix, shared) + S × 64 KiB (fp32 suffix, `b32`) or × 32 KiB (fp16 suffix,
`b3`). An fp16 prefix would halve the first term, but its numerics fail as computed by MLX ops
(see above). A Metal kernel accumulating in fp32 could keep it fp16.

| B | P | S | today MiB/row | `b32` MiB/row | `b3` MiB/row | all rows today → `b32`, GiB |
|---|---|---|---|---|---|---|
| 8 | 8K | 64 | 258 | 68 | 66 | 2.02 → 0.53 |
| 8 | 8K | 512 | 272 | 96 | 80 | 2.12 → 0.75 |
| 8 | 8K | 2048 | 320 | 192 | 128 | 2.50 → 1.50 |
| 8 | 32K | 64 | 1026 | 260 | 258 | 8.02 → 2.03 |
| 8 | 32K | 512 | 1040 | 288 | 272 | 8.12 → 2.25 |
| 8 | 32K | 2048 | 1088 | 384 | 320 | 8.50 → 3.00 |
| 16 | 8K | 64 | 258 | 36 | 34 | 4.03 → 0.56 |
| 16 | 8K | 512 | 272 | 64 | 48 | 4.25 → 1.00 |
| 16 | 8K | 2048 | 320 | 160 | 96 | 5.00 → 2.50 |
| 16 | 32K | 64 | 1026 | 132 | 130 | 16.03 → 2.06 |
| 16 | 32K | 512 | 1040 | 160 | 144 | 16.25 → 2.50 |
| 16 | 32K | 2048 | 1088 | 256 | 192 | 17.00 → 4.00 |

**Transient** (measured peak above resident, one 8-layer step, B=16):

* today: 18–55 MB;
* cascade: 24–77 MB at 1K, 0.17–0.21 GB at 8K, 0.36–0.61 GB at 32K.

The per-layer score buffers are (4, P/Pc, 4B, Pc) fp32 = 64·B·P bytes, 34 MB at 32K/B=16, and
several layers' buffers are alive within one graph. This has to enter the batch reserve, the way
the prefill reserve does.

## Recommended formulation (MLX ops, first version)

`b32c`, i.e. `cascade(..., dt=float32)` in the script, under `mx.compile`:

1. **Group state.** A group holds the prefix K/V once in **fp32**, as (4, c, Pc, 256) with
   Pc = 1024. It is padded to c·Pc with a static additive mask (`b32m`, see below).
   Each row keeps its suffix in a stacked fp32 (B, 4, S_cap, 256) buffer, plus a lengths vector, an
   additive −inf mask, and zeros in unused slots.
2. **Per layer:**
   * qg = q·(1/16) in fp32, shaped (B,4,4,256); qp = qg transposed to (4,1,4B,256), row index b·4+rep.
   * Prefix: s = qp @ Kcᵀ (4,c,4B,Pc); lse_c = logsumexp(s); p = softmax(s); o_c = p @ Vc.
   * Chunk merge: lse_p = logsumexp over c; o_p = Σ_c exp(lse_c − lse_p)·o_c.
   * Suffix: the same three ops on (B,4,4,S_cap) with the mask.
   * Final merge: m = max(lse_p, lse_s); out = (e^{lse_p−m} o_p + e^{lse_s−m} o_s)/(e^{lse_p−m} + e^{lse_s−m});
     back to (B,16,1,256) fp16.
3. **Suffix update.** Write the new token with one `put_along_axis` per K and V for all rows (`b32u`:
   +1–3 ms per step over `b32c` at B=16, vs `a_u` − `a` = +1–8 ms today; both are inflated somewhat
   by the fewer-layers-per-graph scaling).

That is ~20 kernels per layer, independent of B, against ~4–6 per row today. The host build drops
from 0.5–1.9 ms per step to ~0.1–0.4 at B=16.

## Pitfalls for the implementation

* **GQA layout.** MLX maps q head h to kv head h // 4 (repeats contiguous). The stacked prefix
  queries must be (kv head, b·4 + rep) and transposed back the same way. A wrong order still gives
  plausible-looking output, so the test must compare against per-row SDPA, not only check shapes.
* **fp32 statistics.** The LSE (and the scores feeding it) must be fp32. An fp16 LSE has an ulp of
  ~0.008 at 8–16, i.e. a 0.4–0.8 % merge-weight error. `mx.logsumexp` on fp16 input returns fp16, so
  cast first. fp16 scores alone already fail (`b16`).
* **Normalized partials.** Use normalized per-chunk outputs + LSE, not unnormalized Σ e·v in fp16:
  the latter overflows fp16 once the sum of weights times |v| passes 65504.
* **Unused slots must be finite.** Padding and suffix capacity get −inf in the scores and **zeros**
  in K/V. softmax gives them weight 0, but 0 × NaN = NaN in P·V if a slot holds garbage.
* **Prefix boundary is not a multiple of anything.**
  * The group prefix length is wherever the cache hit ended.
  * Pad to c·Pc and add a static (1, c, 1, Pc) additive mask (`b32m`): +0.1–2.3 ms per step at
    B=16 (8K/S=64: 6.4 → 6.5, 32K/S=64: 14.1 → 16.4).
  * Cheaper alternatives, unmeasured: a separate small tail chunk (~4 extra kernels), or a mask only
    on the last chunk.
  * Do not round P down and move the remainder into every row's suffix: that is up to 1023 tokens ×
    B rows of fp32 KV.
* **Where the rounding differs from the fused kernel.**
  * The products are identical (fp16 → fp32 is exact; scale 1/16 is exact in fp16, and the kernel
    also scales q).
  * The summation order differs: GEMM tiles + chunk merge vs the 2-pass kernel's blocks.
  * The final fp16 cast happens once, where the kernel also rounds its partials.
  * Result: ≤ a few ulp off (a) in 33–43 % of the elements, and closer to the fp32 truth.
  * B ≤ 2 must stay on the per-row path to keep the "B ≤ 2 bit-exact" invariant.
* **The suffix is the weak part.**
  * Its GEMMs have M = 4 (4 q heads per kv head per row), padded to 64-row tiles, so they are
    compute-bound at ~16× the useful work.
  * At S=2048 the manual suffix costs more than today's fused per-row kernel over the same tokens.
  * Only group when the prefix dominates (P ≥ ~4K, B_group ≥ 8). Expect the gain to shrink as
    answers grow; it is still −11 ms per step at 8K/B=16/S=2048.
  * The real fix is a suffix kernel that also writes the LSE. MLX's `sdpa_vector` source already
    computes max/sum in its first pass, and could be taken from the wheel's headers via
    `mx.fast.metal_kernel` as `pfqmm` did. The sink trick (`b3`) is the MLX-ops stopgap.
* **mx.compile is shape-specialized.**
  * It saves 5–15 % (`b16` → `b16c`, `b32` → `b32c` at B=16).
  * B (joins and leaves), c and S_cap change the shapes. Grow S_cap in coarse steps (≥ 256) and
    expect a retrace per change, or compile only the merge.
* **Donation.** Each layer's suffix buffer must be updated once per step, with no other live
  reference, or `put_along_axis` copies the whole buffer (seen as +8 ms in run 1).
* **Transient memory**: up to ~0.6 GB per step at 32K/B=16 (above). Reserve it.
* **Admission.** B=4 groups gain nothing, because the time is per 64-row tile and not per row. Keep
  B < 8 groups on the per-row path, or measure on the real model before lowering the threshold.

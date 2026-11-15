# One-pass verify-block attention for DFlash at long context (`QWEN_VERIFY_ATT=2`, `QWEN_DECODE_ATT`)

2026-11-16, M1 Max 32 GB (32 GPU cores), mlx 0.32.3, mlx-vlm 0.7.6, branch `verify-att2`.
Model: Qwen3.8-27B 4-bit (`weights/qwen38-27b-mlx-4bit`, `QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1`) + DFlash2
drafter (`weights/qwen38-27b-dflash2`), fp16 KV. Labels: **[M]** measured, **[D]** derived, **[E]** estimate.
Session 3 ran from 10:45 to 11:12 (`micro9`, `short`, the diverge margins, `long 32768`). For
part of it another task's CPU-only pytest ran on the machine. The A/B pairs are interleaved in one
process, so only the absolute tok/s can be affected.

## Result

| 27B 4-bit, 60K-token context, greedy, 256 tokens, in-process A/B, median of 3 | before | after | |
|---|---:|---:|---:|
| DFlash decode (`QWEN_VERIFY_ATT` 1 → 2) | 17.97 tok/s, round 284 ms | **26.54 tok/s**, round 192 ms | **×1.48** (per-round pairs ×1.48–1.55) [M] |
| plain decode (`QWEN_DECODE_ATT` off → 8192) | 10.61 tok/s | **14.37 tok/s** | **×1.35** (pairs ×1.30–1.40) [M] |

* The generated tokens are identical: all 15 runs of the A/B (plain, plain+d, vector, vatt, vatt+d,
  3 rounds each) equal the first plain run, token for token, and DFlash accepted the same 4.12 drafts
  per round with either attention [M].
* Numerics gate (teacher-forced, 32K context): verify blocks KL(vec ‖ vatt) mean **1.3e-5**, p99 6.2e-5,
  top-1 **100 %** over 512 positions; decode steps KL(MLX ‖ vatt) mean **1.2e-5**, top-1 100 % over
  128 positions. Both pass (≤ 1e-4, ≥ 99.7 %) [M].
* Kernel accuracy vs an fp32 reference: rms rel error 2.0–2.1e-4 for every shape, below MLX's own
  kernels' 2.5–2.9e-4 (fp32 partials vs MLX's fp16 ones) [M].

* **32K** (same A/B): DFlash 22.31 → **30.13 tok/s** (×1.35, round 204 → 151 ms), plain decode
  14.87 → 16.53 tok/s (×1.11). All runs give the same tokens as plain [M].
* **Short context** (`ab27.py`, 5 code tasks of ~40-token prompts, 256 tokens, 3 rounds): no
  change. DFlash runs at 30.88 tok/s on the `=1` path and 30.93 tok/s with vatt, round 140.3 vs
  140.4 ms [M].
* **Identity vs plain greedy on 16 short tasks:** 12/16 with vatt, and 12/16 on today's `=1`
  path. All 4 vatt divergences are near-ties (top-1/top-2 gap 0.0–0.016 nat, the 1st percentile
  of all steps' gaps). The `=1` path's 4 divergences are near-ties too (0.0–0.031). Both miss the
  rule's ≥ 15/16. The 12/16 comes from the 4-bit verify through mrqmm, not from the attention:
  two of the four tasks diverge at the same token under both attentions [M].
* **Defaults stay off: `QWEN_VERIFY_ATT=1`, `QWEN_DECODE_ATT=0`.** Both knobs belong in the
  27B long-context recipe (see Gates).

## Problem

A DFlash2 verify round on the 27B runs a block of T = 8 tokens through the target. In each of the
16 full-attention layers (24 query heads, 4 KV heads, GQA factor G = 6, head_dim 256), query t of
the block attends keys `[0, n − T + t + 1)`. mlx-vlm makes one SDPA call per query (`rows`: the KV
read 8 times). `QWEN_VERIFY_ATT=1` (2026-11-11) packs `32 // G` = 5 queries into MLX's vector kernel,
so the KV is read twice (5 + 3), but that kernel still runs one simdgroup per (query head, query)
over strided keys and reaches only ~30 GB/s of KV. At 60K keys one layer cost 8.1 ms per round vs 2.8 ms
for a single decode query. 16 layers × 8.1 ms ≈ 130 ms, close to half the round [M].

The plain decode step has the same weakness one level down: MLX's 2-pass `sdpa_vector` for one query
over the 27B layout reads 60K keys at 88–96 GB/s (2.77 ms per layer), against an achievable ~339 GB/s
(the `bwprobe` from 2026-10-16) [M].

## The kernel (`src/worker/vatt.py`)

The G·T query rows of one KV head (48 for T = 8; 6 for a decode step) attend the KV together, one
read per round:

* **Pass 1.** Grid: key chunks × query blocks × KV heads. The query blocks of one chunk are
  adjacent threadgroups, so their K/V reads meet in the GPU's caches. Each threadgroup runs MLX's
  steel attention loop on one chunk: the loop from `steel_attention.h`, taken through
  `cascade_mma.cmma_p1`; the wheel's loader and MMA headers are inlined, as in `cascade_mma`
  (`cascade_mma.steel_headers()` is now shared). In detail:
  * Q stays in registers. K and V blocks of BK keys are staged in threadgroup memory.
  * S = QKᵀ is an 8×8 simdgroup MMA, fp16 × fp16 with fp32 accumulation. A WN = 2 score exchange
    follows.
  * Softmax is online, in the log2 domain.
  * P·V multiplies the fp32 probabilities by fp16 V fragments, with fp32 accumulation.
  * The block's causal tail is a per-row key limit, row r = g·T + t, which masks keys
    ≥ n − T + t + 1. Only the key blocks that reach the tail pay for it.
  * The pass writes the chunk's normalized fp32 output plus m and l (split-KV / flash-decoding).
* **Pass 2.** An LSE merge of the chunks, written as fp16 in the (1, Hq, T, D) layout.
* **K/V buffers are read in place.**
  * The KVCache buffers `(1, Hkv, cap, D)` are passed as they are
    (`ensure_row_contiguous=False`). The head stride and token stride come from the arrays.
  * Nothing is copied, and nothing past n is read. The tests fill the space past n with garbage:
    1000 in K and NaN in V.
  * A strided view (every other head) is handled too.
* **Tile (measured, sweep at 60K keys, T = 8, ms per layer).**

  | tile (BQ×BK) | ms per layer | note |
  |---|---:|---|
  | **16×16** | **2.41–2.52** | 4 simdgroups, 3 query blocks of 16 rows. Chosen |
  | 16×32 | 2.50–2.60 | |
  | 24×24 | 2.51–2.99 | |
  | 48×24 | 2.81–3.08 | |
  | 48×48 | 3.08–3.27 | one block, 12 simdgroups |
  | 32×32 | 4.23–4.33 | 2 blocks, the second half empty |
  | 8×16 | 3.31–3.58 | |
  | WN = 1 tiles | 2.76–8.5 | no score exchange; register pressure |

  * A variant with separate K / V / exchange regions (3 barriers a block instead of 6) was not
    faster: 16×16 2.83 ms, 16×8 2.52 ms. It stays off (`VATT_SPLIT`).
  * The target-threadgroup count (128 / 256 / 512 / 1024) moves results within ±5 %. Default 256.
  * `raw/verify_att2/sweep*`.
* **Decode** (`QWEN_DECODE_ATT`) is the same kernel with T = 1: the G = 6 query heads form one 8-row
  block.

### Microbench: one layer, 27B layout, random fp16 K/V, T = 8 (`verify_att_bench.py`, session 2)

| keys | rows (library) | vec (`=1`) | **vatt (`=2`)** | vec / vatt | decode: MLX | **decode: vatt** | MLX / vatt |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0.77 ms | 0.58 | **0.23** | ×2.6 | 0.143 | **0.125** | ×1.14 |
| 8K | 1.33 | 1.18 | **0.46** | ×2.6 | 0.211 | **0.210** | ×1.00 |
| 32K | 9.02 | 4.32 | **1.38** | ×3.1 | 1.065 | **0.554** | ×1.92 |
| 60K | 23.25 | 8.14 | **2.45** | ×3.3 | 2.767 | **0.910** | ×3.04 |
| 80K | 29.31 | 10.25 | **3.07** | ×3.3 | 3.469 | **1.142** | ×3.04 |

Other block sizes at 60K (vec → vatt):

* T = 2: 2.30 → 0.97 ms;
* T = 4: 3.86 → 1.64 ms;
* T = 16: 16.84 → 4.78 ms.

The rows are `raw/verify_att2/micro_T*.json` [M].

**How close to the floor.**

* **Decode.** At 60K one layer's KV is 252 MB. At ~339 GB/s that is 0.74 ms [D]. vatt reads it
  at **276 GB/s** at 60K and 294 GB/s at 80K, i.e. 81–87 % of the achievable read rate. MLX's
  kernel reads at 91–97 GB/s [M].
* **Verify block.** The block is compute-bound, not bandwidth-bound:
  * 48 query rows × 2 products = 49 FLOP per KV byte, against a ridge of ~28 FLOP/byte
    (9.5 TFLOPS fp16 GEMM / 339 GB/s).
  * At 60K: 12.1 GFLOP in 2.45 ms = **4.9 TFLOPS**, ~52 % of MLX's fp16 GEMM rate on this GPU
    [D].
  * The bandwidth floor would be 0.74 ms and the GEMM-rate floor ~1.3 ms [D].
  * The remaining gap belongs to the steel loop's per-block barriers and the WN = 2 score
    exchange (see "Left").

## Integration

* **`QWEN_VERIFY_ATT=2`** (`config.VERIFY_ATT2`). `kernels._verify_attention` routes a lone row's
  verify block (T ≤ 16, fp16 `KVCache`) to `vatt.attention(queries, cache.keys, cache.values, n)`
  after the library's append:
  * The append is `_prepare_projected_qkv`, which calls `update_and_fetch`.
  * n is `keys.shape[2]`, a Python int, so there is no host sync.
  * A call that raises turns the path off and falls back to `=1`'s vector-kernel path.
  * Load-time self-test `_probe_vatt` (via `_vatt_selftest`):
    * Random blocks of the model's heads, every T in 2..16. Each compiles its tile's pipelines at
      load, not in the first round.
    * A buffer with garbage past n, and several chunks.
    * Pass condition: every output finite and within max(1.05 × the per-query calls' error, 4e-4)
      of an fp32 reference. Otherwise it logs a warning and uses `=1`.
  * `=1` keeps its bit-equal path. `=0` is the library's calls.
* **`QWEN_DECODE_ATT=<min keys>`** (0 = off). The `_fused_sdpa` override of qwen3_5's
  `scaled_dot_product_attention` sends a decode query over ≥ that many fp16 keys to vatt.
  * Covered: a lone row's step, and every row of a batch (`_rows_attention` calls the override per
    row).
  * Not covered: prefill slabs, quantized KV, masks.
  * The override is now installed when `QWEN_SDPA_FUSED` or `QWEN_DECODE_ATT` is on.
  * Self-test `_probe_decode_att` at load: T = 1, same criterion as above.
  * Per-row and batch-independent: a row decodes the same bits alone or in a batch.
* **Tests.**
  * `tests/test_vatt.py`: Metal, `VATT_GPU=1`, 51 cases.
    * T 1..16 with G 6 / 4 / 8.
    * Lengths around chunk and key-block boundaries, the tail crossing a chunk, forced chunks.
    * A strided view and transposed queries.
    * Flat and peaky scores; the error never worse than MLX's own kernel.
    * The CPU reference on CPU.
  * `tests/test_vatt_path.py`: CPU.
    * Verify blocks go through vatt on the cache's own buffers, with logits equal to the library's
      to rounding and the DFlash answer == plain.
    * Failure falls back.
    * The probe on / off rules.
    * `QWEN_DECODE_ATT` at max_batch 1 and 3: the threshold, the same answers, failure turns it
      off.

## Model-level runs

### 60K context (`vatt_gate.sh long 60000`)

* **Setup.**
  * `long_ctx.py --tokens 60000 --turns 1 --no-thinking --ab 3 --ab-mode ''
    --ab-variants plain,plain+d,vector,vatt,vatt+d`.
  * Env: `QWEN_VERIFY_ATT=2 QWEN_DECODE_ATT=8192`, plus the 27B long-context recipe:
    `QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_MOVE=1 QWEN_PROMPT_CACHE_GB=6 QWEN_PROMPT_CACHE_CKPTS=4
    QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192
    QWEN_PREFILL_BUDGET=16777216 QWEN_MAX_CONTEXT=90112`.
* **Procedure.**
  * One cold prefill of the 60,068-token prompt: 532 s, then 27.3 tok/s DFlash with vatt.
  * Then 3 rounds of the same prompt, with variants rotated per round.
  * Each repeat resumes from the conversation's entry at the last message start (60,022 tokens
    reused, TTFT ~1 s) and moves back after it (one KV copy).
  * Variant meaning: `+d` = decode steps through vatt; `vector` = `QWEN_VERIFY_ATT=1`; `vatt` =
    `=2`.

| variant | tok/s (3 rounds) | median | round ms | drafts/round | == plain |
|---|---|---:|---|---:|---:|
| plain | 11.27 / 10.24 / 10.61 | 10.61 | — | — | 3/3 |
| plain+d | 14.63 / 14.37 / 14.07 | **14.37** | — | — | 3/3 |
| vector | 18.71 / 17.97 / 16.76 | 17.97 | 273 / 284 / 304 | 4.12 | 3/3 |
| vatt | 29.09 / 26.54 / 25.75 | **26.54** | 175 / 192 / 198 | 4.12 | 3/3 |
| vatt+d | 28.97 / 26.40 / 25.69 | 26.40 | 176 / 193 / 199 | 4.12 | 3/3 |

* **The round shrinks by 92–106 ms.** The microbench predicts 16 × (8.14 − 2.45) = 91 ms [D].
* **Plain step.** 94 → 70 ms. The microbench predicts −30 ms; measured −24 ms [D].
* **`vatt+d` = `vatt`.** The DFlash rounds have no plain decode steps here.
* **Peak memory.**
  * 22.38 GiB at the cold prefill.
  * Decode 21.55 GiB with vatt vs 21.57–21.69 with the vector path. vatt's partials are a few MB.
    The vector path's transients are 180–370 MB per layer call (microbench).
* **The machine drifted** between rounds: plain 11.27 → 10.24. Only the in-round ratios count.

### 32K context (`vatt_gate.sh long 32768`)

Same setup with `--tokens 32768`. The prompt was 32,834 tokens and the cold prefill took 236 s.
Each repeat reused 32,788 tokens.

| variant | tok/s (3 rounds) | median | round ms | drafts/round | == plain |
|---|---|---:|---|---:|---:|
| plain | 15.11 / 14.87 / 14.76 | 14.87 | — | — | 3/3 |
| plain+d | 16.63 / 16.53 / 15.98 | **16.53** | — | — | 3/3 |
| vector | 22.52 / 22.31 / 22.06 | 22.31 | 202 / 204 / 207 | 3.57 | 3/3 |
| vatt | 30.92 / 30.13 / 29.70 | **30.13** | 147 / 151 / 153 | 3.57 | 3/3 |
| vatt+d | 30.44 / 30.06 / 29.55 | 30.06 | 150 / 152 / 154 | 3.57 | 3/3 |

* **The round shrinks by 53–55 ms.** The microbench predicts 16 × (4.32 − 1.38) = 47 ms [D].
* **Plain step.** 67 → 60 ms, about −7 ms. The microbench predicts 16 × (1.065 − 0.554) = 8 ms [D].
* **Peak memory.** 20.71 GiB at the prefill. Decode 19.79–19.92 GiB with vatt and 19.88 GiB
  with the vector path.
* Raw data: `raw/verify_att2/long_32768.{json,log}`.

### Short context (`vatt_gate.sh short`)

`ab27.py --rounds 3 --variants plain,m8/8,m8/8:va --ident m8/8,m8/8:va`: block 8, verify through
mrqmm4. `m8/8` is today's default (`QWEN_VERIFY_ATT=1`) and `:va` is vatt. The speed phase covers
5 code tasks, the identity phase 16 tasks, 256 tokens each, no thinking.

| variant | tok/s (median over tasks) | per task | round ms | == plain (speed phase) | identity (16 tasks) |
|---|---:|---|---:|---:|---:|
| plain | 18.35 | 19.1 18.3 18.4 18.0 18.0 | — | 15/15 | — |
| m8/8 (`=1`) | 30.88 | 38.6 22.7 27.1 33.0 30.9 | 140.3 | 12/15 | 12/16 |
| m8/8:va (`=2`) | 30.93 | 39.3 22.6 27.1 34.2 30.9 | 140.4 | 15/15 | 12/16 |

The divergence margins come from `diverge_margin.py` (`DM_MODEL` = the 4-bit 27B, plain greedy
through the library) and are in `raw/verify_att2/ab27_2k_margins.json`:

| task | `=1` diverges at (gap, nat) | `=2` diverges at (gap, nat) |
|---|---|---|
| t3 | 151 (0.016) | — |
| t5 | — | 43 (0.016) |
| t10 | 56 (0.031) | — |
| t11 | 33 (0.000) | 33 (0.000) |
| t12 | — | 26 (0.016) |
| t15 | 202 (0.016) | 202 (0.016) |

Of all 746 steps before the divergences, the gap percentiles are p1 0.016, p5 0.156 and p50
3.40. Every divergence is a tie or a one-ulp near-tie in the fp16 logits [M].

## Gates

| gate | criterion | result |
|---|---|---|
| numerics, verify block (`vatt_kl.py`, 32K + 64 blocks of 8) | KL ≤ 1e-4, top-1 ≥ 99.7 % | KL 1.3e-5, 100 % — **pass** |
| numerics, decode step (128 steps after it) | same | KL 1.2e-5, 100 % — **pass** |
| identity vs plain greedy, long context | ≥ 15/16, divergences at near-ties | 60K: 9/9 DFlash runs == plain (both attentions) |
| identity vs plain greedy, 16 short tasks (`ab27.py --ident`) | ≥ 15/16 | 12/16, all at near-ties (gap ≤ 0.016) — **fail as written**. Today's `=1` default is also 12/16 (gap ≤ 0.031) |
| speed ≥ 3 % in-run | | 60K: DFlash ×1.48, plain ×1.35; 32K: ×1.35 / ×1.11; short: DFlash ×1.00 — **pass at ≥ 32K only** |

**Defaults.**

* **`QWEN_VERIFY_ATT` stays `1`.**
  * The numerics gate passes, and so does the speed gate at long context.
  * The literal identity rule (≥ 15/16) fails. It fails for today's default path too, since
    the 4-bit 27B's mrqmm verify gives 12/16 on its own.
  * At short context there is nothing to gain.
  * Flipping it is a one-line change in `config.py` if the 12/16-at-near-ties evidence is
    judged enough.
  * Recommended: `QWEN_VERIFY_ATT=2` in the 27B long-context recipe (README, TUTORIAL §5.8).
* **`QWEN_DECODE_ATT` stays `0`.**
  * The 9B layout loses with it (G = 4, see Left): MLX's decode kernel already reads at ~300 GB/s
    there.
  * On the 27B (G = 6) MLX's kernel reads at ~95 GB/s, so the knob pays from about 8K keys.
    `8192` goes into the 27B recipe.

## Commands

```zsh
# from the repo root; the 27B env lives in the script
scripts/perf/dflash2/vatt_gate.sh micro       # microbench, raw/verify_att2/micro_T*.json
scripts/perf/dflash2/vatt_gate.sh kl 32768    # numerics gate, raw/verify_att2/kl_32768.json
scripts/perf/dflash2/vatt_gate.sh long 60000  # model A/B at 60K, raw/verify_att2/long_60000.{json,log}
scripts/perf/dflash2/vatt_gate.sh short       # ab27.py at short context: speed + 16-task identity, raw/verify_att2/ab27_2k.{json,log}
scripts/perf/dflash2/vatt_gate.sh long 32768  # model A/B at 32K
scripts/perf/dflash2/vatt_gate.sh micro9      # 9B layout microbench, raw/verify_att2/micro9b_T8_*.json
(cd src/worker && DM_MODEL=../../weights/qwen38-27b-mlx-4bit PYTHONPATH=. .venv/bin/python ../../scripts/perf/dflash2/diverge_margin.py ../../docs/perf/2026-11-16/raw/verify_att2/ab27_2k.json)
VATT_GPU=1 .venv/bin/python -m pytest -q tests/test_vatt.py   # Metal tests (src/worker)
```

## Left

* **The verify kernel runs at ~4.9 TFLOPS; ~1.3 ms per layer at 60K is the GEMM-rate floor**
  (about −18 ms per round, +9 % DFlash at 60K [E]). What is in the way:
  * the steel loop's 6 barriers per 16-key block;
  * the WN = 2 score exchange through threadgroup memory;
  * K staged transposed.

  Neither of the variants tried helped: WN = 1 spills, and separate smem regions cost occupancy.
  A design that loads K fragments straight from device memory (Sᵀ = K·Qᵀ, then Oᵀ = Vᵀ·Pᵀ, which
  needs no transpose of P) was sketched but not built.
* **q8 KV (kv-q8 work).** The loader reads fp16 only. A dequantizing K/V loader would halve the
  decode reads.
* **9B.** The 9B (G = 4: 8 queries fit one vector-kernel call, so `=1` already reads the KV once)
  is not gated at model level. Microbench on its layout (16 q heads / 4 KV heads, D 256, T = 8;
  `raw/verify_att2/micro9b_T8_*.json`, ms per layer) [M]:

  | keys | vec (`=1`) | vatt (`=2`) | decode: MLX | decode: vatt |
  |---:|---:|---:|---:|---:|
  | 2K | 0.487 | **0.202** | 0.139 | 0.137 |
  | 8K | 0.737 | **0.326** | **0.181** | 0.211 |
  | 32K | 2.748 | **0.974** | **0.486** | 0.548 |
  | 60K | 5.007 | **1.634** | **0.830** | 0.897 |

  * **Verify.** vatt wins ×2.3–3.1 on the 9B layout as well. Packing 8 queries into one vector
    call reads the KV once but runs at ~50 GB/s.
  * **Decode.** vatt is 8–17 % slower than MLX on the 9B at ≥ 8K keys (equal at 2K). MLX's 1-query kernel reaches 276–303 GB/s
    with G = 4, so `QWEN_DECODE_ATT` must stay off for the 9B. A per-G gate in
    `_probe_decode_att` (on only where MLX's kernel is slow) would make it safe to default; it
    is not built.
  * A 9B DFlash A/B at 32K+ with `=2` would be the next gate.

# 8-bit KV for the 27B single stream (phase 1)

Goal: `QWEN_KV_BITS=8` as a first-class option for the 27B single-stream setup (drafter, prompt
cache with one KV copy, long context): half the KV bytes per token (64 → 34 KiB) without
losing what used to need fp16 KV. Phase 1 adds no Metal kernel; it lifts the fp16-only gates,
makes the quantized paths correct, sizes their memory and states the GPU measurements that
decide phase 2 (a quantized vector-attention kernel for decode / DFlash verify).

Status: implemented and CPU-tested (branch `kv-q8`). GPU gate: (0) attention alone and (i) KL at
32K done (section 7); (ii)–(iv) (`ab_kv.py`, `long_ctx.py`, 80K) not run yet. Every new
behaviour is either behind a knob that is off by default (`QWEN_KVQ_PREFILL_FUSED`) or
applies only with `QWEN_KV_BITS` ≠ 0, so the fp16 stand is unaffected.

## 1. Inventory: what was fp16-only, why, and what changed

| place | why fp16-only before | real reason? | change |
|---|---|---|---|
| `Engine.load`: drafter ignored with quantized KV | "the spec loop never re-quantizes; rollback trims a quantized cache wrongly" | half. `_spec` had no `self.quantize` calls (a real gap). Trim is fine: mlx-vlm 0.7.6 `QuantizedKVCache.trim` only lowers the offset, and quantization is per token, so a rejected token leaves nothing behind. | gate removed; `_spec` quantizes like the plain loop |
| `_spec` prefill / first target call / close steps / rounds | no quantization at all, so the cache stayed fp16 for the whole request | real | `requant()` (= `self.quantize` + `_kvq_mark`) after every prefill slab, after the first target call, after each close step and after every committed round. While the cache is still fp16, a block is cut at `quantized_kv_start` (`fp16_left`); a lone token just before the start is one plain step (`step1`). Every position therefore gets the same KV format as in the plain loop, and on CPU q8 DFlash == plain greedy. |
| verify / commit / abort with quantized KV | — | the library already handles the tuples: `slice_kv_sequence`, per-query quantized SDPA, `trim` | none |
| `QWEN_VERIFY_ATT` (`_verify_attention`) | MLX's vector attention kernel is fp16-only | real (no quantized vector kernel exists) | its per-call gate already skips a `QuantizedKVCache`; the load log now says "fp16 part only; quantized KV: the library's one-query calls". **This is the phase-2 lever.** |
| `_move_on` (`QWEN_PROMPT_CACHE_MOVE`) | "quantized resume rules below quantized_kv_start not handled" | real, but small | allowed for any `kv_bits`. Only quantized entries move (`movable`). An fp16 entry (shorter than the start) is copied, as without the mode. An fp16 parent of a row whose prefill quantizes gives `pieces = []`: the store never mixes bits. A row still fp16 at its prefill's end drops its lease and stores by copy (`snapshot`). |
| `_Seg.take/adopt/sub/copy_from/estimate`, `restore(move)`, `unmove`, `_settle*` | — | already generic over the `(packed, scales, biases)` tuples | none (covered by the new parity tests) |
| `_Entry.low`, chain floor, "never absorb fp16 into quantized" | — | correct | none |
| cascade groups | fp32/fp16 prefix attention | real; batch-only (max_batch ≥ 3) | unchanged |
| `_fixed_slab` / `_step` / fused SDPA in prefill | the library's quantized SDPA materializes slab × keys × heads fp16 scores, so the slab shrinks with context (546 at 120K) | real | `QWEN_KVQ_PREFILL_FUSED` (default 0). The slabs the fused gate takes with fp16 KV attend the layer's K/V dequantized to fp16 (`mx.dequantize`, `kernels._dequantized_kv`) through the fused kernel. Fixed slab at any context. |
| `_prefill_reserve` | — | — | with the knob on, adds one FA layer's dequantized K + V at `QWEN_MAX_CONTEXT` × 1.6, the measured transient (27B 128K: 512 → 819 MiB). Signature `(heads, kv_heads=0, head_dim=0)`. |
| `_Scheduler._bytes` (reservation) | charged `start` × fp16 + n × q forever | an over-charge after quantization (27B: 5000 × 64 KiB = 312 MiB too much) and an under-charge at the crossing (≤ 128 MiB) | `min(n, start + PREFILL_STEP)` × fp16 until the row's cache is quantized, then n × q only (`row.kvq`, set by `_kvq_mark` in prefill / decode / `_spec` / the lookup of a quantized parent) |
| `_step_bytes` (log metric) | counted [0, start) as fp16 after quantization | a log metric bug | n × q past the start |
| `_kv_token_bytes` | — | correct (27B q8: 32768 × (1 + 4/64) = 34816 B) | none |
| disk tier | — | handles quantized entries (`test_pcdisk`) | move still excludes it (as before) |
| preemption | — | `adopt` is tuple-generic; batch-only | none |

## 2. Decisions

- **Move with q8: quantized entries only.** An entry shorter than `quantized_kv_start` (5000)
  is fp16 and is the only resume point below the start. The row that resumes from it usually
  quantizes past it, and a store never absorbs an fp16 entry into a quantized one. Moving it
  would lose it, so it is copied (≤ 5000 × 64 KiB = 312 MiB on the 27B, once per
  conversation, while its first turn is short).
- **DFlash blocks are cut at the start.** The alternative was to let a verify block straddle
  the crossing. The plain loop quantizes after the step that reaches the start, so a
  straddling block would attend fp16 KV where plain attends quantized, and q8 DFlash would
  stop being equal to q8 plain greedy at those positions. The cut costs at most one shortened
  round per request.
- **`QWEN_KVQ_PREFILL_FUSED` is off by default.** It changes the prefill numerics. On CPU it
  is closer to the fp32 reference over the same dequantized KV than the quantized SDPA
  (q8, 128 × 2048: max error 0.0032 vs 0.0145, mean 1.1e-4 vs 4.8e-4; 2–4× lower at both
  8 and 4 bits), but it is not bit-equal to it, and it is ungated on the real model. The slab
  policy stays frozen at load (`_FIXED_SLAB`), so `_bounds` determinism holds: hit == cold
  (`test_kvq_fused_prefill_runs_and_hits_equal_cold`).
- **Nothing for batches.** Cascade and the disk-tier-with-move stay as they were; phase 1 is
  the single stream.
- **4-bit KV** goes through the same code (the tests run 8 and 4 bits for the fused path) but
  is not part of the gate.

## 3. Memory arithmetic (27B: 16 FA layers × 4 KV heads × 256 = 32768 values/token)

| | fp16 | q8 | q4 |
|---|---|---|---|
| per token | 64 KiB | 34 KiB | 18 KiB |
| 32K | 2.0 GiB | 1.06 GiB | 0.56 GiB |
| 80K (81920) | 5.0 GiB | 2.66 GiB | 1.41 GiB |
| 96K (98304) | 6.0 GiB | 3.19 GiB | 1.69 GiB |
| 128K | 8.0 GiB | 4.25 GiB | 2.25 GiB |

- The fp16 warm-up copy a q8 row holds before its cache crosses the start is at most
  (5000 + 2048) × 64 KiB = 440 MiB. After the crossing it is gone: `to_quantized` replaces
  the buffers and later updates quantize the new tokens.
- The dequantized-prefill transient (`QWEN_KVQ_PREFILL_FUSED`) is one FA layer's K + V in
  fp16 at the current context: 2 × 4 × 256 × 2 B = 4 KiB/token, i.e. 320 MiB at 80K and
  512 MiB at 128K. Measured (attn_bench, a 2048-query slab): 72 / 216 / 504 MiB at
  8K / 32K / 80K keys, ~1.6× the K + V; the reserve takes K + V × 1.6 (819 MiB at 128K). The next layer's K/V are built only after this layer's attention consumed
  them (its input depends on it). MLX may still overlap two layers in one command buffer;
  the gate's peak checks this (risk 2).
- **Stand reference** (fp16 KV, 4-bit weights + drafter 15.38 GiB, MOVE): an 80K prompt with
  thinking peaks at 23.71 GiB.
  - q8 projection: 23.71 − (5.0 − 2.66) = **~21.4 GiB at 80K**.
  - fp16 at 128K: 23.71 + 48K × 64 KiB ≈ 26.7 GiB, at the 27 GiB limit.
  - q8 at 128K: ≈ 21.4 + 48K × 34 KiB ≈ 23.0 GiB, plus ~0.8 GiB with the fused prefill.
  - So q8 turns 128K from "at the limit" into ~4 GiB of headroom (prompt-cache room for a
    second conversation).
  - These are projections; gate (iv) measures them.

## 4. GPU gate

Run from `src/worker`, nothing else on the GPU (see CLAUDE.md "Measuring performance"),
`setopt NO_BG_NICE`. Common env (the stand's, with q8 KV):

```zsh
export PERF_OUT=${PERF_OUT:-/tmp/perf}
COMMON=(QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1
        QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192
        QWEN_PREFILL_BUDGET=16777216 QWEN_PRESENCE_PENALTY=0 PYTHONPATH=.)
```

**(0) attention alone** (`scripts/perf/kvq8/attn_bench.py`, no model, ~3 min). Per call at
the 27B shapes, keys 8K / 32K / 80K:

- decode (1 query): fp16 vector kernel vs the q8 library path;
- a DFlash verify block of 8: fp16 one-query rows vs the fp16 vector kernel (`QWEN_VERIFY_ATT`)
  vs q8 one-query rows;
- prefill per token: fp16 fused vs q8 at its shrunk slab vs q8 dequantized + fused.

Each also reports its error vs fp32.

```zsh
env $COMMON .venv/bin/python ../../scripts/perf/kvq8/attn_bench.py --keys 8192 32768 81920 \
  --json $PERF_OUT/kvq8/attn_bench.json
```

**(i) teacher-forced KL / top-1, q8 vs fp16 KV** (`kl_kv.py`, ~15 min at 32K, ~40 min at 80K).
Variants fp16 / q8 / q8f, run one after another in one process. Each uses its own slabs, then
512 one-token decode steps. KL is reported per region: "fp16 part" (must be 0: harness
check), "prefill", "decode".

```zsh
env $COMMON .venv/bin/python ../../scripts/perf/kvq8/kl_kv.py --tokens 32768 --decode 512 \
  --json $PERF_OUT/kvq8/kl_32k.json
env $COMMON .venv/bin/python ../../scripts/perf/kvq8/kl_kv.py --tokens 81920 --decode 512 \
  --json $PERF_OUT/kvq8/kl_80k.json
```

For scale, the reference noise floors are:
- the fp16 cast: KL 4e-4, 99.2 % top-1 on the 27B;
- q4 weights vs q5 weights: KL 0.053, 90.1 %.

**Pass** if, past the start, KL mean ≤ 1e-3 and top-1 ≥ 99 % for q8, both in prefill and in
decode. q8f must be no worse than q8; it is expected closer.

**(ii) + (iii) speed, interleaved in one process** (`ab_kv.py`). Prefill TTFT is measured cold
per variant. Plain and DFlash tok/s come from "read" runs. Medians and ratios against fp16 are
printed at the end. 8K + 32K take ~40 min; 80K takes ~30 min with q8 / q8f only (fp16's 80K
entry + its read copy do not fit beside them).

```zsh
env $COMMON QWEN_KV_BITS=8 QWEN_KVQ_PREFILL_FUSED=1 QWEN_DRAFT=../../weights/qwen38-27b-dflash2 \
  QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_MOVE=1 QWEN_PROMPT_CACHE_GB=12 QWEN_MAX_CONTEXT=98304 \
  .venv/bin/python ../../scripts/perf/kvq8/ab_kv.py --contexts 8192 32768 \
  --json $PERF_OUT/kvq8/ab_8k_32k.json
env $COMMON QWEN_KV_BITS=8 QWEN_KVQ_PREFILL_FUSED=1 QWEN_DRAFT=../../weights/qwen38-27b-dflash2 \
  QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_MOVE=1 QWEN_PROMPT_CACHE_GB=12 QWEN_MAX_CONTEXT=98304 \
  .venv/bin/python ../../scripts/perf/kvq8/ab_kv.py --contexts 81920 --variants q8 q8f \
  --prefill-rounds 1 --json $PERF_OUT/kvq8/ab_80k.json
```

For fp16 at 80K, use the stand's numbers (or (iv) with `QWEN_KV_BITS=0`): decode tok/s per
context is compared at 8K / 32K in-process, and 80K q8 against the fp16 trend.

**(iv) peak at 80K and the largest context under 27 GiB** (`scripts/perf/dflash2/long_ctx.py`,
real conversations with the drafter and MOVE). Run 80K with q8 and with q8f, then raise
`--tokens` (96K, 112K, 120K with `QWEN_MAX_CONTEXT=131072`) until the prefill or decode peak
passes ~26 GiB. Record the prefill peak and the decode peak per run. Also run fp16 at 80K for
the same-process reference peak.

```zsh
for F in 0 1; do
  env $COMMON QWEN_KV_BITS=8 QWEN_KVQ_PREFILL_FUSED=$F QWEN_DRAFT=../../weights/qwen38-27b-dflash2 \
    QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_MOVE=1 QWEN_PROMPT_CACHE_GB=6 QWEN_PROMPT_CACHE_CKPTS=4 \
    QWEN_MAX_CONTEXT=131072 .venv/bin/python ../../scripts/perf/dflash2/long_ctx.py --tokens 81920 \
    --turns 2 --json $PERF_OUT/kvq8/long_80k_f$F.json
done
# then --tokens 98304 / 114688 / 122880 with the better F; fp16 reference: QWEN_KV_BITS=0, --tokens 81920
```

`QWEN_PREFILL_RESERVE_GB` (the stand sets 2.5) replaces the estimate, including the 819 MiB
(at 128K) for the dequantized layer. Keep the stand's value only if the measured q8f prefill peak stays
inside it.

### (v) Phase-2 rule (a quantized vector-attention kernel)

Write the kernel only if **both** hold:

1. **Speed gap.** At 32K or 80K (ab_kv / long_ctx), q8 plain decode < 0.90 × fp16 plain
   decode, **or** q8 DFlash < 0.85 × fp16 DFlash (tok/s, medians, same process).
2. **Attention is the cause.** attn_bench attributes ≥ 2/3 of that gap to attention. Per token
   at the same keys, the gap is (q8 decode call − fp16 decode call) × 16 FA layers for plain
   decode, or (q8-rows − fp16-vec) × 16 per round for DFlash. Compare it with the step / round
   time difference.

If the gap is real but attention explains less than two thirds of it, look elsewhere
(quantization of the new token every step, the KV gather) before writing a kernel. If q8 is
within the thresholds, phase 1 is the end: q8 becomes the 27B stand's recommendation, and
`QWEN_KVQ_PREFILL_FUSED` defaults to 1 if (i) and (iii) pass for it.

## 5. Risks (unmeasured on the GPU)

1. **q8 DFlash verify at long context.** It runs 8 one-query quantized SDPA calls per FA
   layer per round, each reading the whole KV. For fp16 that path cost 570 ms per round at
   60K keys before `QWEN_VERIFY_ATT` (296 ms after). q8 reads half the bytes but without the
   vector kernel, so q8 DFlash may lose to q8 plain decode at 80K. That would be the phase-2
   trigger.
2. **Dequant transient.** It is sized as one layer. If MLX keeps two layers' dequantized K/V
   alive (graph scheduling), the peak is 2× (1 GiB at 128K). Gate (iv)'s prefill peak shows
   it.
3. **The quantized SDPA's own transient** (knob off) is fp16 scores plus the softmax output
   and the mask, 2–3× the scores. The shrunk slab bounds the scores at `PREFILL_BUDGET`, as
   for the 35B; `_prefill_reserve` counts the scores once.
4. **Blocks cut at the start.** They are measured on CPU only (correctness); the cost is
   ≤ 1 shortened round per request.
5. **The drafter load path with quantized KV** is not exercised on CPU: the tests build the
   drafter themselves. The first GPU run's load log must show "DFlash verify attention: … fp16
   part only".

## 6. Tests (CPU)

`tests/test_kvq8.py` (35 tests) covers:
- **Move-mode parity with copies, q8 at `quantized_kv_start` 0 and 40.** Cases: chat with
  replies, the fused-GDN model, no reply KV, a reply crossing the start, thinking replies,
  cancelled prefills (miss and hit), cancel before the first slab, DFlash rows. An fp16 entry
  stays and the quantized one moves.
- **q8 DFlash == q8 plain greedy** (tokens and the final KV) for starts 0 / 12 / 26 / 27 / 29 /
  33, plus a thinking budget, demotion, and the verify-attention gate skipping quantized
  caches.
- **The dequantized fused prefill:** gate, fallback, closer to the fp32 reference than the
  quantized SDPA at 8 and 4 bits, and hit == cold.
- **Accounting:** fixed slab and reserve, the 27B's bytes per token, the warm-up copy dropped
  from the reservation, `_step_bytes`.

Negative checks (each made a test fail, then reverted):
- without the block cut, `equals_plain[26, 27, 33]` fails;
- without `requant()` after a round, `equals_plain[27, 29]` fails.

## 7. GPU results (2026-11-15, 27B 4-bit, M1 Max; other agents' CPU test suites were running)

Raw: `docs/perf/2026-11-15/kvq8/` (`attn_bench.*`, `kl_32k.*`).

**(0) Attention alone** (27B shapes, random KV; ms per call; prefill: ms per query token, q8 at
the slab the worker gives it with `QWEN_PREFILL_BUDGET=16777216`):

| keys | decode fp16 | decode q8 | verify-8 fp16 rows | verify-8 fp16 vec | verify-8 q8 rows | prefill fp16 | prefill q8 (slab) | prefill q8f |
|---|---|---|---|---|---|---|---|---|
| 8K | 0.79 | 1.60 | 2.84 | 2.13 | 4.83 | 0.027 | 0.032 (2048) | 0.027 |
| 32K | 1.01 | 1.84 | 9.57 | 4.96 | 13.88 | 0.122 | 0.147 (512) | 0.123 |
| 80K | 5.50 | 3.88 | 39.4 | 11.2 | 33.0 | 0.298 | 0.442 (204) | 0.302 |

- **Decode.** q8 costs +0.8 ms per FA layer at 8K–32K, i.e. +13 ms per token over 16
  layers. At 80K it is 1.6 ms cheaper than fp16: fp16's vector kernel jumps ×5.4 from 32K to
  80K for ×2.5 keys, while q8 reads half the bytes.
- **DFlash verify.** q8's one-query calls cost +2.7 / +8.9 / +21.8 ms per FA layer over the
  fp16 vector kernel. That is **+43 / +142 / +350 ms per round** over 16 layers, against
  ~150–300 ms rounds: q8 DFlash at 32K+ is expected to lose most of its gain. This is the
  case for phase 2.
- **Prefill.** q8f equals fp16 per token at every context. q8 at its shrunk slab is ×1.2 at
  32K and ×1.5 at 80K.
- **Errors vs fp32.** The dequantized fused path's error over the dequantized KV is 2–4×
  lower than the quantized SDPA's (= fp16's own level). Against the original fp16 KV, both
  q8 paths sit at q8's quantization error (2–6e-4 max).

**(i) Teacher-forced KL, 32K + 512 decode steps** (`kl_32k.json`):

| pair | region | positions | KL mean | KL p99 | KL max | top-1 |
|---|---|---|---|---|---|---|
| q8 vs fp16 | fp16 part | 5000 | 0 | 0 | 0 | 100 % |
| q8 vs fp16 | prefill | 27768 | 5.95e-5 | 5.0e-4 | 0.024 | 99.66 % |
| q8 vs fp16 | decode | 512 | 7.0e-5 | 4.5e-4 | 0.0023 | 100 % |
| q8f vs fp16 | prefill | 27768 | 5.59e-5 | 4.6e-4 | 0.020 | 99.70 % |
| q8f vs fp16 | decode | 512 | 6.0e-5 | 3.8e-4 | 6.3e-4 | 99.80 % |
| q8f vs q8 | prefill | 27768 | 4.5e-5 | 3.9e-4 | 0.016 | 99.73 % |

- NLL of the true next token: fp16 1.5726 / q8 1.5727 / q8f 1.5726 (prefill), and
  1.8356 / 1.8364 / 1.8355 (decode).
- **Passes (i)** with a 15× margin on the KL bound. The q8 KV distance is ~7× below the fp16
  cast's own KL (4e-4) and ~900× below q4-vs-q5 weights. q8f is marginally closer to fp16
  than q8.

Side numbers from the same run (one pass each, not interleaved; the teacher-forced decode is
a synchronous step without step-ahead):
- **Prefill 32K:** fp16 287 s (16 slabs), q8 279 s (34 slabs), q8f 261 s (16 slabs).
- **Peak:** 19.65 / 18.83 / 18.80 GiB.
- **512 decode steps:** fp16 36.4 s, q8 45.9 s, q8f 44.3 s. q8 is ×0.79–0.82, i.e. +19 ms per
  step, of which attention (above) explains ~13 ms (≈ 2/3).

So for plain decode at 32K the phase-2 rule's first clause (q8 < 0.90× fp16) would already be
met in this synchronous loop, with attention at the 2/3 boundary. `ab_kv.py` (the worker's
real step-ahead loop, interleaved) decides.

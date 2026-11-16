# One-pass attention over 8-bit quantized KV (`QWEN_VATT_Q8`)

2026-11-16, M1 Max 32 GB (32 GPU cores), mlx 0.32.3, mlx-vlm 0.7.6, branch `vatt-q8`.
Model: Qwen3.8-27B (`weights/qwen38-27b-mlx-oq4e`, oQ4e; session 2 used the uniform 4-bit
`weights/qwen38-27b-mlx-4bit`, "u4") + DFlash2 drafter (`weights/qwen38-27b-dflash2`).
Labels: **[M]** measured, **[D]** derived, **[E]** estimate. Raw files: `raw/vatt_q8/`.

## Verdict: the 27B on q8 KV, yes

With `QWEN_KV_BITS=8 QWEN_VATT_Q8=1 QWEN_KVQ_PREFILL_FUSED=1` added to the 27B's long-context
config (oQ4e + DFlash2, `QWEN_PROMPT_CACHE_MOVE`), a 78K conversation behaves as follows [M]:

* **Memory.** The conversation's entry is **3.37 GiB** instead of 5.61 GiB with fp16 KV. With the
  entry held, a foreign 14.3K prompt (`prompt_cache: "read"`) prefills at a **23.24 GiB** peak
  instead of 25.68, which is 3.8 GiB under the 27 GiB wired limit instead of 1.3. The entry
  survives it, and the conversation's next turn hits (78 016 tokens reused, TTFT **1.44 s**).
* **Decode at 78K.** Plain is 11.97 tok/s vs 5.42 through the library's q8 attention. DFlash is
  18.90 vs 4.92.
* **Decode at 32K.** This was one process, fp16 vs q8 interleaved. Plain is 11.89 vs 11.29 tok/s
  (×1.05). The DFlash round costs the same (191.5 vs 195.7 ms). DFlash tok/s is 21.14 vs 24.13
  only because the fp16 run decoded a different text (3.74 vs 3.06 drafts accepted per round; fp16
  DFlash diverged from greedy, q8 DFlash did not).
* **Cost.** Cold prefill is ×1.05–1.06 slower. q8 attention is not bit-equal to fp16; the 2026-11-15
  gate measured KL 5.6e-5 / top-1 99.70 % for q8 vs fp16 KV at 32K.

Recipe: the production env plus `QWEN_KV_BITS=8 QWEN_KVQ_PREFILL_FUSED=1`. `QWEN_VATT_Q8=1` is
the default from this branch on (set it explicitly on older code). `QWEN_DECODE_ATT` does nothing
on q8 KV: it needs ≥ 8192 fp16 keys, and the cache turns q8 at 5000.

## What it is

With `QWEN_KV_BITS=8`, mlx-vlm keeps the full-attention layers' KV in a `QuantizedKVCache`:
(packed uint32, fp16 scales, fp16 biases), affine, group 64 along head_dim 256. Its attention is
`quantized_scaled_dot_product_attention`: two `quantized_matmul`s plus a softmax over materialized
scores. A DFlash verify block calls it once per query (the library's `_attention`), so a block of 8
reads the whole quantized KV 8 times. `QWEN_VERIFY_ATT` and `QWEN_DECODE_ATT` were fp16-only, so a q8
cache got none of the one-pass kernel's gains.

`vatt.py` now takes either format. The kernel body (steel MMA loop, split-KV pass 1 + LSE merge) is
unchanged. Only the tile loaders differ:

* `KVF16` uses steel's `BlockLoaderT`, the fp16 path. It is bit-identical to before the refactor
  (checked at 8 shapes, 700–81920 keys, T 1/8/16) [M].
* `KVQ8` uses `Q8Loader`. Each thread loads one uint4 (16 packed values) plus its group's scale and
  bias. It dequantizes in registers with the 0x6400 trick
  (`as_type<half2>((w & 0x00ff00ff) | 0x64006400) - 1024`, then `fma(h, scale, bias)`) and writes the
  fp16 values into the same threadgroup tiles the fp16 path fills.
  * K is stored transposed.
  * The thread → (key, 16-value chunk) mapping is "c": consecutive threads take consecutive chunks of
    one key, so the packed loads coalesce. The older key-major mapping (`VATT_Q8_KMAP=k`, microbench
    only) ran decode at 80K at 1.23 vs 0.89 ms a layer [M].

The cache's own buffers are read in place. They are `[..., :offset, :]` views of the step-256
buffers, and the strides are passed in (`ensure_row_contiguous=False`). There is no copy and no host
sync.

Wiring (`kernels.py`):

* **Verify.** In `_verify_attention`, a q8 `QuantizedKVCache` with T ≤ 16 goes to
  `vatt.attention_q8(q, cache.keys, cache.values, n, scale)`.
* **Decode.** The `_fused_sdpa` override (`q35.scaled_dot_product_attention`) routes a one-query step
  over a q8 cache the same way. That covers a lone row, and each row of a batch through
  `_rows_attention`.
* **Before `quantized_kv_start`** (5000 tokens) the cache is a plain `KVCache`, so
  `QWEN_VERIFY_ATT` / `QWEN_DECODE_ATT` apply.
* **Load self-test (`_probe_vatt_q8`).** It runs on random data (n 600 in a 768-slot buffer, garbage
  past n) for T 1, 3, 8 and 16. The error vs an fp32 reference on the dequantized KV must be ≤
  max(1.05 × the library's error, 4e-4); otherwise the knob turns off with a warning. A call that
  raises at run time turns the path off and falls back to the library.
* **CPU tests.** `tests/test_vatt_q8_path.py` covers the DFlash and decode routing, batch rows, the
  fallback on failure, and the fp16 cache keeping the fp16 paths. The kernel's GPU tests are in
  `tests/test_vatt.py` (`VATT_GPU=1`: 98 passed) [M].

## Microbenchmark (one layer, 27B layout: 24 query heads, 4 KV heads, head_dim 256)

`scripts/perf/dflash2/verify_att_bench.py`. Time per layer call, median [M]. The rms relative error
is vs an fp32 reference on the dequantized KV.

| keys | decode: library q8 | **vatt q8** | vatt fp16 | verify T=8: library q8 (8 calls) | **vatt q8** | vatt fp16 |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.462 ms | **0.199** | 0.213 | 3.64 ms | **0.487** | 0.457 |
| 32K | 1.725 | **0.463** | 0.564 | 15.13 | **1.452** | 1.369 |
| 60K | 3.349 | **0.753** | 0.961 | 29.69 | **2.648** | 2.556 |
| 80K | 4.572 | **0.910** | 1.178 | 40.47 | **3.486** | 3.227 |
| 128K | 5.551 | **1.576** | 1.800 | 56.71 | **5.606** | 4.991 |

* **Decode** (bandwidth-bound) on q8 beats fp16 vatt from 8K on, because it reads 0.53× the bytes
  (196 GB/s of q8 bytes at 80K). It is ×2.3–5.0 faster than the library q8 path.
* **Verify** (compute-bound: every 16-row query block dequantizes its tiles again) runs 4–12 % behind
  fp16 vatt (6 % at 8K, 12 % at 128K). It is ×7.5–11.6 faster than the library's per-query calls.
* **Errors.** vatt q8 has rms rel error 2.0–2.1e-4, the same as the fp16 kernel. The library q8 path
  has 1.4–2.9e-3: its scores are fp16.

Tile sweep (T 8: 16×16, 16×32, 24×24, 32×32, 48×24; T 1: 8×16, 8×32) and the loader-mapping sweep
are in `raw/vatt_q8/sweep_*.json` and `sweep2/`. The fp16 tiles (16×16 verify, 8×16 decode) stay
best for q8 too.

## Numerics gate (teacher-forced, 32K context)

`scripts/perf/kvq8/vattq8_kl.py`. It prefills 32K tokens as the worker does with q8 KV
(quantized after each slab). It then runs 64 verify blocks of 8 drafted positions and 128 decode
steps. The three variants are the library, vatt q8, and an exact fp32 reference attention on the
dequantized KV ("ref").

| oQ4e, 32K | positions | KL mean | p99 | top-1 | flips at ref top-2 gap |
|---|---:|---:|---:|---:|---|
| verify, lib ‖ vatt q8 | 512 | **2.15e-5** | 1.0e-4 | **99.80 %** | 0.0 (an exact tie) |
| verify, ref ‖ lib | 512 | 2.3e-5 | 9.2e-5 | 99.61 % | 0.0, 0.0 |
| verify, ref ‖ vatt q8 | 512 | 2.15e-5 | 1.2e-4 | 99.80 % | 0.0 |
| decode, lib ‖ vatt q8 | 128 | **1.67e-5** | 7.9e-5 | **100 %** | — |
| decode, ref ‖ lib | 128 | 1.7e-5 | 1.3e-4 | 100 % | — |
| decode, ref ‖ vatt q8 | 128 | 1.44e-5 | 6.3e-5 | 100 % | — |

**Gate (KL ≤ 1e-4, top-1 ≥ 99.7 %): PASS on oQ4e** [M]. vatt q8 is at least as close to the exact
attention as the library is, and closer on both mean and max |Δ logprob| (0.26 vs 0.96 verify, 0.64
vs 2.84 decode).

On u4 (session 2) the same comparison gave KL 1.44e-5 for both verify and decode, and top-1 100 %
decode but **99.41 % verify (3 flips in 512): FAIL** by the letter of the rule [M]. That run had no
ref variant. On oQ4e the library itself sits at 99.61 % top-1 vs the exact attention. The 99.7 %
line is at the noise floor of q8 attention numerics, not a property of this kernel.

## End-to-end A/B, 32K (in-process, interleaved, median of 3, 256 greedy tokens)

`scripts/perf/kvq8/ab_kv.py --contexts 32768 --variants fp16 q8f q8fv --prefill-rounds 1 --rounds 3
--decode-mode ''`. The run used MOVE mode (one KV copy), oQ4e, `QWEN_MRQMM4=1 QWEN_MRQMM6=1`,
**without `QWEN_MRQMM5`**. The oQ4e's 236 5-bit modules therefore ran the verify block through
`quantized_matmul`, so absolute DFlash rates are low. The ratios hold.

* fp16 = fp16 KV with `QWEN_VERIFY_ATT=2 QWEN_DECODE_ATT=8192`.
* q8f = q8 KV + `QWEN_KVQ_PREFILL_FUSED=1`, library attention.
* q8fv = the same with `QWEN_VATT_Q8=1`.

| 32K | fp16 (vatt) | q8f (library) | **q8fv (vatt q8)** | q8fv / q8f | q8fv / fp16 |
|---|---:|---:|---:|---:|---:|
| plain tok/s | 11.41 | 8.55 | **12.13** | ×1.42 | ×1.06 |
| DFlash tok/s | 7.55 | 5.14 | **7.90** | ×1.54 | ×1.05 |
| DFlash round | 614 ms | 902 ms | **587 ms** | | |
| prefill (cold) | 325 s, peak 21.61 GiB | 334 s, peak 23.61 GiB | (shares q8f's) | | |

* **Identity.** All 18 runs equal their variant's first plain run, and the plain answers are equal
  across fp16, q8f and q8fv. Acceptance was 3.65 drafts per round in every DFlash run [M].
* **Session 2 (u4, read mode, two KV copies).** q8fv / q8f was plain ×1.33 and DFlash ×2.27; q8fv /
  fp16 was ×1.06 / ×1.44. The fp16 side ran at 25.9 GiB peak, under memory pressure, so its ratio is
  not used [M].

## The production 27B config at 78K: the long entry and a foreign request beside it

Config (the 27B serving env; `QWEN_DECODE_ATT=8192` was added on both sides):

```
QWEN_MODEL=weights/qwen38-27b-mlx-oq4e QWEN_DRAFT=weights/qwen38-27b-dflash2
QWEN_MRQMM4=1 QWEN_MRQMM5=1 QWEN_MRQMM6=1 QWEN_TEXT_ONLY=1 QWEN_MAX_BATCH=1
QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_MOVE=1 QWEN_PROMPT_CACHE_GB=6 QWEN_PROMPT_CACHE_CKPTS=4
QWEN_PREFILL_RESERVE_GB=2.5 QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216
QWEN_WIRED_LIMIT_GB=27 QWEN_PRESENCE_PENALTY=0 QWEN_VERIFY_ATT=2 QWEN_DECODE_ATT=8192
q8 side:   QWEN_KV_BITS=8 QWEN_VATT_Q8=1 QWEN_KVQ_PREFILL_FUSED=1
fp16 side: QWEN_KV_BITS=0
```

`scripts/perf/dflash2/foreign_prefill.py --probe --writer 78000 --foreign 14300` does the following:

1. A conversation with a 78K-token system document gets one turn, cold. The entry it leaves is the
   prompt plus the reply.
2. An unrelated 14.3K-token prompt runs with `prompt_cache: "read"` (nothing stored).
3. The conversation's next turn runs. It must hit the entry.
4. An A/B of that next turn follows.

Each side ran in its own process, back to back. A short-prompt probe opened each process: plain
12.95 / 13.44 tok/s and DFlash 37.2 / 38.6 tok/s (q8 / fp16), so the machine ran at the same speed
within 4 % [M]. Raw: `raw/vatt_q8/prod78k_try1/`.

| 78K, production config | **q8 KV + vatt q8** | fp16 KV | |
|---|---:|---:|---|
| conversation prompt | 78 036 tokens | 78 037 | |
| cold prefill TTFT | 968.5 s | 924.5 s | q8 ×1.05 slower [M] |
| prefill peak (active before) | **22.54 GiB** (16.34) | 24.49 GiB (16.34) | [M] |
| **cache entry** (prompt + reply) | **3.37 GiB** | 5.61 GiB | −2.24 GiB [M] |
| active with the entry held | 19.71 GiB | 21.96 GiB | [M] |
| foreign 14.3K `read`: TTFT / prefill peak | 142 s / **23.24 GiB** | 127 s / 25.68 GiB | q8: 3.8 GiB under the 27 GiB limit, fp16: 1.3 [M] |
| entry after the foreign request | kept (3.37 GiB, 1 entry) | kept (5.61 GiB) | neither evicted, neither OOM [M] |
| next turn: reused / TTFT | 78 016 / **1.44 s** | 78 017 / 1.74 s | hit on both [M] |
| process peak | **23.24 GiB** | 25.68 GiB | [M] |

* **Decode at 78K, q8 side.** This is an in-process A/B of the next turn: 2 rounds, vatt q8 vs the
  library's quantized SDPA (`--ab-lib`) [M].
  * Plain: **11.97 vs 5.42 tok/s (×2.21)**.
  * DFlash: **18.90 vs 4.92 tok/s (×3.84)**, rounds 244–289 vs 955–976 ms, 4.25 vs 4.0 drafts accepted
    per round.
  * The answers were short (20–21 tokens, ending on a stop), so these rates are noisier than
    256-token runs.
  * Tokens: plain vatt and DFlash-lib equal the first plain run; DFlash-vatt and plain-lib differ
    from it only in the last token (21 vs 20 tokens before the stop).
* **Decode at 78K, fp16 side.** Plain decode was **9.90 tok/s** (256 tokens with `ignore_eos`, 3
  rounds) [M]. `ignore_eos` is a token option, and token options turn DFlash off (`_spec_ok`), so
  those "DFlash" runs were plain and fp16 DFlash at 78K was not measured. The next turn without
  `ignore_eos` (64 tokens, 1.91 drafts accepted per round) ran DFlash at 12.0 tok/s, on a different
  text than the q8 side's.
* **fp16 for reference** (`docs/perf/2026-11-15/design/oq4e_speed.md`, the same config at 60K):
  DFlash 26.4 tok/s on the cold turn, 19.1 on a hit; entry 4.84 GB; peak 23.42 GiB.

## The production config at 32K: fp16 vs q8 KV in one process

`ab_kv.py --contexts 32768 --variants fp16 q8fv --prefill-rounds 1 --rounds 3 --decode-mode ''`.
The env is the production one above with `QWEN_KV_BITS=8 QWEN_VATT_Q8=1 QWEN_KVQ_PREFILL_FUSED=1`
and `QWEN_MAX_CONTEXT=98304`. The variant switches the KV format in-process, and each format has
its own prefix cache (MOVE). Runs: 3 rounds, rotating order, 256 greedy tokens with no thinking [M].
Raw: `raw/vatt_q8/prod32k/`.

| 32K, production config | fp16 KV | **q8 KV + vatt q8** | q8 / fp16 |
|---|---:|---:|---:|
| plain tok/s (median; runs) | 11.29 (11.50, 11.28, 11.29) | **11.89** (11.97, 11.89, 11.18) | **×1.05** |
| DFlash tok/s (median; runs) | 24.13 (24.13, 24.41, 23.89) | **21.14** (21.18, 21.14, 19.96) | ×0.88 |
| DFlash round (median) | 195.7 ms | **191.5 ms** | ×0.98 |
| drafts accepted per round | 3.74 | 3.06 | |
| cold prefill | 295.5 s | 311.8 s | ×1.06 slower |

* **DFlash round cost.** A q8 round costs the same as an fp16 one (191.5 vs 195.7 ms), so the
  verify kernel's 4–12 % lag (microbench) is lost in the rest of the round.
* **DFlash tok/s.** The ×0.88 comes from acceptance (3.06 vs 3.74 drafts per round), and the
  acceptance comes from the text. fp16 DFlash diverged from fp16 plain greedy partway through the
  answer (`dflash=x` in all 3 runs: the batched-decode numerics class, as on the
  `QWEN_VERIFY_MRQMM` path). q8 DFlash produced exactly q8 plain greedy (3/3). The two DFlash runs
  therefore decoded different texts. The plain answers were equal across fp16 and q8, token for
  token [M].
* **Not comparable in this run: memory.** Both formats' entries sit in one process here, so the q8
  prefill peak includes the fp16 entry. The memory comparison is the 78K back-to-back one above.

## Verdict on the gates

| gate | result |
|---|---|
| kernel vs fp32 reference (rms rel) | 2.0–2.1e-4 at 8K–128K, T 1/8 (the fp16 kernel's class; the library q8 path 1.4–2.9e-3) — **pass** [M] |
| teacher-forced KL vs the library q8, 32K, oQ4e | verify 2.15e-5 / 99.80 %, decode 1.67e-5 / 100 % — **pass** [M] |
| same on u4 (session 2) | 1.44e-5 both; top-1 verify 99.41 % — **fail** by the rule, at the q8 noise floor (the library vs the exact attention is 99.61 % on oQ4e) [M] |
| in-run gain ≥ 3 % | vs the library q8: plain ×1.42 / DFlash ×1.54 (32K), ×2.21 / ×3.84 (78K) — **pass** [M] |
| DFlash identity | q8 DFlash == q8 plain 3/3 (32K, 256 tokens) and 6/6 (session 3); the 16-task short corpus was not run (short contexts stay below `quantized_kv_start`: fp16 KV, unchanged paths) |

**`QWEN_VATT_Q8` is default on (`1`).** It acts only with `QWEN_KV_BITS=8`, so the 9B (fp16 KV)
and the 35B (q4) are unchanged.

## Open issues

* **128K headline not run.** The coordinator stopped it to save GPU time. Estimate: a 126K cold
  prefill takes ~30 min [E, from 968 s at 78K]. The q8 row would be ~4.2 GiB [D: 34.8 KB/token],
  so the resting process would sit at ~21 GiB plus a ~6 GiB prefill transient, under the 27 GiB
  limit [E].
* **fp16 DFlash at 78K not measured.** The fp16 78K A/B used `ignore_eos`, and token options turn
  DFlash off. For fp16 the 60K figure from `oq4e_speed.md` stands in.
* **Verify q8 lags fp16 vatt by 4–12 % per layer call.** Every 16-row query block dequantizes its
  tiles again. Dequantizing once per chunk into threadgroup memory shared by the query blocks is
  the lever. It was not tried: in the e2e round the gap does not show.
* **Cold q8 prefill is ×1.05–1.06 slower** than fp16 even with `QWEN_KVQ_PREFILL_FUSED=1` (32K and
  78K).

## Knobs

* `QWEN_VATT_Q8` (`1`). `1` = verify + decode; `verify` / `decode` = one of them; `0` = the library.
  It acts only with `QWEN_KV_BITS=8`.
* `VATT_Q8_KMAP` (`c`). Microbench only: `k` = the older key-major K loader.

## Commands

All runs: `cd src/worker`, `setopt NO_BG_NICE`, the GPU otherwise idle (lease). `$ENV` is the
production env above.

```zsh
# kernel tests (GPU) and CPU suite
VATT_GPU=1 .venv/bin/python -m pytest -q tests/test_vatt.py
.venv/bin/python -m pytest -q --ignore tests/test_mrqmm.py
# microbench, one layer, 27B layout
for N in 8192 32768 61440 81920 131072; do .venv/bin/python ../../scripts/perf/dflash2/verify_att_bench.py $N 8 \
  --only q8single,q8rows,vq8,vq8_1,single,vatt1,vec,vatt --json $PERF_OUT/micro_q8_T8_$N.json; done
# tile sweeps: VATT_TILE=<WMxBK> VB_ROUNDS=5 ... verify_att_bench.py $N 8 --only vq8,vatt (T 1: vq8_1,vatt1)
# numerics gate, 32K
$ENV QWEN_KV_BITS=8 QWEN_VATT_Q8=1 QWEN_KVQ_PREFILL_FUSED=1 env -u QWEN_DRAFT .venv/bin/python \
  ../../scripts/perf/kvq8/vattq8_kl.py --tokens 32768 --blocks 64 --decode 128 --json $PERF_OUT/kl_32768.json
# 32K in-process fp16 vs q8 (production config)
$ENV QWEN_KV_BITS=8 QWEN_VATT_Q8=1 QWEN_KVQ_PREFILL_FUSED=1 QWEN_MAX_CONTEXT=98304 .venv/bin/python \
  ../../scripts/perf/kvq8/ab_kv.py --contexts 32768 --variants fp16 q8fv --prefill-rounds 1 --rounds 3 \
  --decode-mode '' --json $PERF_OUT/ab32.json
# 78K: long entry + foreign request + hit + A/B (q8 with --ab-lib; fp16: QWEN_KV_BITS=0, --ab 3 --ab-ignore-eos)
$ENV QWEN_KV_BITS=8 QWEN_VATT_Q8=1 QWEN_KVQ_PREFILL_FUSED=1 .venv/bin/python \
  ../../scripts/perf/dflash2/foreign_prefill.py --probe --writer 78000 --foreign 14300 --ab 2 --ab-lib \
  --json $PERF_OUT/kv8_w78000.json
```


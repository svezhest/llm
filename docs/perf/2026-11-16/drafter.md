# DFlash2 drafter on the 27B: linears on mrqmm (D1), drafter bits (D2), prose, sampling (S1)

Target `weights/qwen38-27b-mlx-4bit` (lmstudio-community 4-bit g64, text-only), drafter
`weights/qwen38-27b-dflash2` (z-lab DFlash2, 5 layers, hidden 5120, MLP 17408, block 8), M1 Max 32 GB,
mlx 0.32.3, mlx-vlm 0.7.6. Worker env for every run:

```zsh
QWEN_MODEL=…/qwen38-27b-mlx-4bit QWEN_DRAFT=…/qwen38-27b-dflash2 QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 \
QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27
```

Verify knobs at their defaults (`QWEN_VERIFY_MRQMM=1`: fixed block 8 through mrqmm, `QWEN_VERIFY_ATT=1`).
All numbers are **measured** in one process with variants interleaved (CLAUDE.md "Measuring performance");
raw output in `raw/drafter_*`.

## Summary

| lever | result | verdict |
|---|---|---|
| D1 `QWEN_DRAFT_MRQMM` (drafter linears on `mrqmm_wide`) | drafter block 31.3 → 17.4 ms; the same drafts on all 240 probes; DFlash decode 25.8 → 29.6 tok/s, **×1.117** in-run (code / en / ru 2K: ×1.129 / ×1.123 / ×1.111); identity vs plain unchanged (12/16, same divergence points) | **on by default** |
| D1 `QWEN_DRAFT_COMPILE` (compiled grouped conv) | block −0.65 ms; decode ×1.003 on top of D1 | off (under the 3 % gate) |
| D2 `QWEN_DRAFT_BITS` 5 / 8 | acceptance 3.21 / 3.25 vs 3.24 of 7 at 4 bits (flat); block +1.5 / +1.9 ms; e2e 8 bits ×0.986 of 4 | **stays 4** |
| D1 on short prose (greedy, ~40-token prompts) | ×1.125 vs the old drafter path (en ×1.136, ru ×1.113), but DFlash stays **slower than plain on Russian prose** (11.1 vs 14.8 tok/s: 0.95 drafts accepted per round) | D1 helps everywhere; short Russian prose needs a fallback, not a faster drafter |
| D2 on short prose | acceptance 1.01 / 0.98 / 0.94 of 7 at 4 / 5 / 8 bits (Russian 0.71 / 0.71 / 0.73) | bits do not move Russian |
| S1 `QWEN_DRAFT_SAMPLING=1` (T 0.6, top_p 0.95, top_k 20) | vs plain sampling, in-run: code2k ×1.41, en2k ×1.41, ru2k ×1.36, short en prose ×1.03, **short ru prose ×0.70** | stays opt-in (not ≥ ×1.1 on every class) |
| `QWEN_SAMPLER_TOPK_FAST` in DFlash sampling | +2.8 % on the DFlash rows (×1.360 vs ×1.323), +0.9 % plain | unchanged (off: its own knob, another RNG stream) |

## 1. What the drafter block spends (stage profile, each stage synced)

`draft_fast.py --stages`. One block: 8 rows (anchor + 7 masks), 5 new context rows; 12 iterations, the last 10
taken. A synced profile adds per-stage gaps, so its total (44 ms) exceeds the one-eval block (31 ms).

| stage (per block) | before (quantized_matmul in fp32) | `+mr+cc` |
|---|---|---|
| MLP (gate, up, down) × 5 | 17.11 ms | 5.68 ms |
| self_attn (q, k, v, o + context k, v + SDPA) × 5 | 7.91 | 6.42 |
| conv.prepare (kernel_projection + conv) × 10 | 4.22 | 3.51 |
| conv.finish + residual × 10 | 3.46 | 2.86 |
| lm_head (the target's, 7 rows, already mrqmm) | 4.05 | 4.00 |
| layernorms × 10 | 2.16 | 2.16 |
| selector (7 sequential steps) | 1.95 | 2.01 |
| fc + hidden_norm (context rows) | 1.30 | 0.72 |
| embed, final norm | 0.49 | 0.48 |
| **total, synced** | 44.04 | 29.23 |

The MLP is where the time went. The drafter runs fp32 activations (`QWEN_DRAFT_FP32`: its residual stream
overflows fp16), so its 4-bit linears ran `quantized_matmul` with fp32 x, dequantization-bound at 8 rows.
What is left after D1 is mostly small kernels: attention without its projections, the convolutions, norms,
the selector's 7 sequential steps. The compiled convolution (`+cc`) removes ~0.65 ms of them.

## 2. D1: `mrqmm_wide` for the drafter (`QWEN_DRAFT_MRQMM`)

**Kernel.** `mrqmm.mrqmm_wide(x16, w, scales16, biases16, rs)`: the mrqmm kernel (simdgroup MMA, fp32
accumulators) with one change, in the store: `y[m, n] = acc * rs[m]`, written as fp32 instead of being rounded
to fp16. The 8-bit source is unchanged (its pin in `tests/test_mrqmm_bits.py` holds); the wide variant is a
textual replacement of the two store lines (`_src_wide`).

**Prescale.** `kernels._prescale_rows` (compiled): per row, `p` = the power of two of max |x| (the max's exponent
field masked out of its fp32 bits); x is multiplied by `2^10 / p` and cast to fp16, so the row's max is in
[2^10, 2^11) and `rs = p / 2^10`. Both scalings are exact (powers of two); the only rounding is fp32 → fp16 of
x (11 bits), far below the 4-bit weights' own error. Away from fp16's max (a product with a dequantized
weight stays finite even if the MMA rounded it to fp16, for |w| < 32) and from its subnormals (elements 2^-24
below the row's max keep full precision).

**Install.** `_install_drafter_mrqmm` at load (after the target's mrqmm install, so the 4-bit self-test has
run): every drafter `QuantizedLinear` with a kernel width (4 / 5 / 8, g64, N % 8 == 0) whose fp32 scales and
biases are exactly representable in fp16 (checked) gets fp16 scales / biases and a `__call__` that takes
`mrqmm_wide` for 1..16 rows while `cfg.DRAFT_MRQMM` is on (read per call, so tools can flip it); more rows
(the first round's context, up to 2047 rows through `fc`) stay on `quantized_matmul`, which promotes the fp16
scales back to the identical fp32 values. 47 modules on the 4-bit drafter (all of them: q / k / v / o, gate /
up / down, the two kernel projections × 5 layers, `fc`, the selector's `hidden_projection`); 46 at 8 bits.
In the worker the fp32 scales are replaced (−~0.1 GB vs before); tools that A/B the knob keep both
(`keep_fp32=True`).

**Accuracy (GPU tests, `tests/test_mrqmm.py -k wide`).** `mrqmm_wide` at 1 / 3 / 8 / 9 / 16 rows, 4 / 5 / 8
bits, the drafter's shapes (5120 → 1280, 17408 → 5120): within fp16-dequantization rounding of an fp32
reference (the self-test's 1.5e-3 · max), and its outputs rounded to fp16 are bit-equal to `mrqmm`'s.
`_drafter_linear` on rows with max |x| 1e5 / 1e-3 / 3: finite, max error / max |ref| < 2e-3 against
`quantized_matmul` in fp32.

**Acceptance (`draft_fast.py`, teacher forcing as `draft_wide.py`).** 8 sequences (5 short code tasks + the
three 2K prompts of `prompts2k.py`), each the prompt + 96 greedy tokens of the target, a probe every 3
positions (240 probes), block 8, drafts compared with the target's greedy continuation:

| variant | mean accepted of 7 | code0..4 | code2k | en2k | ru2k |
|---|---|---|---|---|---|
| d4 (before) | 3.237 | 3.87 2.57 3.07 4.00 4.30 | 3.37 | 2.77 | 1.97 |
| d4 + mr | 3.237 | identical | 3.37 | 2.77 | 1.97 |
| d4 + mr + cc | 3.237 | identical | 3.37 | 2.77 | 1.97 |

The same accepted length on every probe (gate: within −2 %).

**Block time** (incremental blocks as the loop runs them: 5 new context rows each, 37 blocks per variant,
interleaved): d4 31.34 ms → d4+mr 17.44 → d4+mr+cc 16.79 (d4+cc alone 30.87).

**End to end (`ab_drafter.py --rounds 3`).** The real Engine and scheduler, 3 prompts (code / English /
Russian, ~2.1K tokens each, `prompts2k.py`, cut from repo files at 7437e62), 256 tokens, greedy, thinking
off, 3 rounds, variant order rotated and reversed per prompt. Median over prompts of the per-prompt median
over rounds; ratio = median over (round, prompt) pairs of tok/s / tok/s of `d4` in the same pair:

| variant | decode tok/s | code2k / en2k / ru2k | ratio vs d4 (per prompt) | drafts accepted / round | round |
|---|---|---|---|---|---|
| plain | 18.75 | 18.3 / 18.8 / 18.8 | ×0.709 | — | — |
| d4 (before) | 25.75 | 25.7 / 27.5 / 24.6 | ×1 | 2.71 | 142.6 ms |
| **d4 + mr** | **29.55** | 29.5 / 30.9 / 27.3 | **×1.117** (1.129 / 1.123 / 1.111) | 2.71 | 125.4 ms |
| d4 + mr + cc | 29.53 | 29.5 / 31.3 / 27.4 | ×1.120 (1.128 / 1.132 / 1.116) | 2.71 | 125.1 ms |
| d8 + mr + cc | 28.01 | 28.0 / 30.9 / 27.0 | ×1.104 (1.066 / 1.119 / 1.102) | 2.59 | 127.1 ms |

Every DFlash run of the speed phase produced the plain run's tokens (9/9 per variant). The round got
17 ms shorter for a 14 ms shorter block (not decomposed; a guess: fewer, cheaper launches to encode).

**Identity vs plain greedy** (16 wtest TASKS, 256 tokens): d4 12/16 (diverging at tokens 151, 56, 33, 202),
d4+mr+cc 12/16 at the same four positions — unchanged by the knob, as expected (the target verifies; the
drafts were identical anyway). The 12/16 itself is the 4-bit target's DFlash-vs-plain baseline (verify
through mrqmm, the batched-decode numerics class), not something this change moved; it is below the 15/16
of the 5-bit runs and worth a `diverge_margin.py` look by whoever owns the verify path.

Gates: acceptance within −2 % — yes (identical); ≥ +3 % in-run median on code / en / ru — yes (+11–13 % on
each); identity unchanged — yes. `QWEN_DRAFT_MRQMM` defaults to 1. `QWEN_DRAFT_COMPILE` adds +0.3 %:
stays 0.

## 3. D2: drafter bits (`QWEN_DRAFT_BITS` 4 / 5 / 8)

Two extra drafters loaded in the same process with the worker's code (`engine._load_drafter`), all three
with `+mr+cc` (and 5 / 8 without, for the block time):

| bits | drafter memory | mean accepted of 7 (240 probes) | block (incremental) | e2e vs 4 bits |
|---|---|---|---|---|
| 4 | (the engine's) | 3.237 | 17.89 ms | ×1 |
| 5 | +1.72 GB | 3.208 | 19.42 ms (34.11 without mr) | — |
| 8 | +2.40 GB | 3.254 | 19.83 ms (34.03 without mr) | ×0.986 (from the table above: 1.104 / 1.120) |

Acceptance is flat across widths on this target (±0.5 %); the 2026-11-10 probe on the 5-bit target had 5 bits
at +8 % over 4 (3.41 vs 3.16), which does not reproduce here. More bits only cost block time and memory:
**`QWEN_DRAFT_BITS` stays 4.**

## 4. Short prose (greedy) and drafter bits there

The coordinator's cross-framework smoke had DFlash at 17.6 tok/s on Russian prose vs 20.6 plain (64 tokens).
Two short prose prompts (`prompts2k.SHORT`: "write a ~300-word story about a lighthouse keeper…", English and
Russian), 256 tokens, greedy, thinking off.

Acceptance (`draft_fast.py --no-2k --short --gen 192`, 124 probes, mean accepted of 7):

| drafter | all | en_prose | ru_prose | block |
|---|---|---|---|---|
| 4 bits + mr | 1.008 | 1.31 | 0.71 | 15.93 ms |
| 5 bits + mr | 0.976 | 1.24 | 0.71 | 16.95 ms |
| 8 bits + mr | 0.944 | 1.16 | 0.73 | 16.79 ms |

End to end (`ab_drafter.py --no-2k --short --rounds 3`, `QWEN_DRAFT_MRQMM=0` at load so `d4` is the old path):

| variant | tok/s (en / ru) | ratio vs d4 (en / ru) | drafts / round | round |
|---|---|---|---|---|
| plain | 16.7 / 14.8 | ×1.049 / ×1.487 | — | — |
| d4 (before) | 15.1 / 9.9 | ×1 | 0.95 | 156.5 ms |
| d4 + mr | 17.1 / 11.1 | ×1.136 / ×1.113 | 0.95 | 139.2 ms |
| d8 + mr | 17.0 / 10.8 | ×1.107 / ×1.087 | 0.94 | 140.0 ms |

D1 helps here as everywhere (+11–14 %), and English prose moves from below plain to just above it (17.1 vs
16.7). Russian prose stays a loss (11.1 vs 14.8 tok/s, ×0.75 of plain): with < 1 draft accepted per round a
139 ms round yields ~1.95 tokens, i.e. 71 ms per token against ~67 ms for a plain step. Drafter bits do not
change that (the drafter does not predict this text, at any width). What would: a per-request fallback to plain
decoding when the running acceptance stays under ~1 draft per round (CLAUDE.md "Dead ends" (4) rejected it on
the 9B, where the verify was ×2.3 of a step and the drafter restarted cold; on the 27B a block-8 verify is
~2 plain steps, so the break-even is ~1 accepted draft per round). Not built here: the native-MTP / hybrid
work is on another branch.

(The plain rows of this run and of section 5 differ by ~15 % across the two processes, 20 minutes apart: machine
drift, which is why only in-run ratios are compared.)

## 5. S1: sampling (`QWEN_DRAFT_SAMPLING=1`) on the 4-bit

`ab_drafter.py --rounds 3 --short --sampling 0.6,0.95,20` (the 2K prompts + the two short prose prompts, 256
tokens, thinking off, presence penalty 0, one seed per (round, prompt) shared by every variant), the worker's
defaults otherwise (`QWEN_DRAFT_MRQMM=1` from load: this run is also the smoke test of the default-on path, 47
modules installed, active memory 16.51 → 16.40 GB with the fp16 scales). DFlash rows with
`QWEN_DRAFT_SAMPLING=1` (exact sample-match acceptance); `+tk` = `QWEN_SAMPLER_TOPK_FAST=1` (the target's
`pick` then runs the sampler's top-k fast path: one vocab argsort + a draw over the 20 survivors per row, instead
of topk + argsort + scatter + a full-vocab draw).

| variant | tok/s code2k / en2k / ru2k / en_prose / ru_prose | ratio vs plain+tk (per prompt) | drafts / round | round |
|---|---|---|---|---|
| plain | 17.7 / 18.1 / 18.3 / 18.7 / 18.4 | ×0.991 | — | — |
| plain + tk | 18.0 / 18.2 / 18.4 / 18.9 / 18.6 | ×1 | — | — |
| d4 + mr | 23.8 / 25.6 / 23.7 / 19.5 / 13.0 | ×1.323 (1.394 / 1.409 / 1.343 / 1.021 / 0.694) | 2.19 | 129.8 ms |
| d4 + mr + tk | 25.3 / 28.9 / 24.0 / 18.5 / 13.1 | ×1.360 (1.452 / 1.584 / 1.360 / 1.003 / 0.698) | 2.24 | 128.2 ms |

Sampling with DFlash pays on the 2K prompts (×1.34–1.41 vs plain sampling, code, English and Russian alike) and
is even on short English prose, but loses 30 % on short Russian prose, the same acceptance problem as greedy
(section 4). The plan's bar for recommending it in the 27B recipe (≥ ×1.1 on every class) is not met:
**`QWEN_DRAFT_SAMPLING` stays 0**; for document / code work at 2K+ context on the 27B it is a ×1.3–1.4 opt-in.
`pick` already gets the top-k fast path through `QWEN_SAMPLER_TOPK_FAST` (the sampler object is the same); it
adds 2.8 % on the DFlash rows, 0.9 % on plain — left as it is (off: another random stream per seed).

## Files

- `src/worker/mrqmm.py`: `mrqmm_wide`, `_src_wide`, `_get_kernel` / `_call(wide=…)`.
- `src/worker/qwen_worker/kernels.py`: `_prescale_rows`, `_drafter_linear`, `_install_drafter_mrqmm`,
  `_DraftQuantizedLinear`, `_grouped_conv` / `_install_drafter_compile`.
- `src/worker/qwen_worker/engine.py`: `_load_drafter` (the drafter load as a function, also for tools),
  `_setup_drafter_fast`.
- `src/worker/qwen_worker/config.py`: `QWEN_DRAFT_MRQMM` (1), `QWEN_DRAFT_COMPILE` (0).
- Tests: `tests/test_drafter_fast.py` (CPU), `tests/test_mrqmm.py -k wide` (Metal).
- Tools: `scripts/perf/dflash2/draft_fast.py` (acceptance + block time + stage profile, variants
  `d<bits>[+mr][+cc]`), `ab_drafter.py` (end to end; also `--sampling T,top_p,top_k` and `+tk`),
  `prompts2k.py`.

Commands (from `src/worker`, worker env above, `PYTHONPATH=.`):

```zsh
.venv/bin/python ../../scripts/perf/dflash2/draft_fast.py --variants d4,d4+mr,d4+mr+cc,d4+cc \
  --stages d4,d4+mr+cc --json $PERF_OUT/drafter/draft_fast_d4.json
.venv/bin/python ../../scripts/perf/dflash2/draft_fast.py --variants d4+mr+cc,d5+mr+cc,d8+mr+cc,d5,d8 \
  --json $PERF_OUT/drafter/draft_fast_bits.json
.venv/bin/python ../../scripts/perf/dflash2/ab_drafter.py --rounds 3 \
  --variants plain,d4,d4+mr,d4+mr+cc,d8+mr+cc --ident d4,d4+mr+cc --json $PERF_OUT/drafter/ab_greedy.json
QWEN_DRAFT_MRQMM=0 .venv/bin/python ../../scripts/perf/dflash2/draft_fast.py --tasks 0 --no-2k --short --gen 192 \
  --incr-rounds 10 --variants d4+mr,d5+mr,d8+mr --json $PERF_OUT/drafter/draft_fast_prose.json
QWEN_DRAFT_MRQMM=0 .venv/bin/python ../../scripts/perf/dflash2/ab_drafter.py --rounds 3 --no-2k --short \
  --variants plain,d4,d4+mr,d8+mr --skip-ident --json $PERF_OUT/drafter/ab_prose_greedy.json
.venv/bin/python ../../scripts/perf/dflash2/ab_drafter.py --rounds 3 --short --sampling 0.6,0.95,20 \
  --variants plain,plain+tk,d4+mr,d4+mr+tk --skip-ident --json $PERF_OUT/drafter/ab_sampling.json
```

(The first three ran with `QWEN_DRAFT_MRQMM` still defaulting to 0; with today's default set
`QWEN_DRAFT_MRQMM=0` in the env when the variant `d4` should be the old path with fp32 scales.)

# G0: where the 27B 4-bit decode time goes (plain step and DFlash2 round)

## Gate 1 verdicts (night plan; measured 2026-11-15 06:50–08:46)

| lever | rule | measured | verdict |
|---|---|---|---|
| **P1** fused GDN decode kernel | go iff small kernels ≥ 8 ms of the plain step | small kernels **4.6–4.7 ms at 2K, 4.9–6.6 at 32K** of a 51.3 / 74.3 ms step; P1's own scope (the whole GDN mixer incl. recurrence) **3.1–3.6 ms** | **NO-GO** |
| **R1b** recompute instead of GDN with-states | iff host bubble ≥ 5 ms | between rounds **2.2–2.3 ms**; with-states costs 1.9 ms over the plain T=8 kernel (5.5 vs 3.6 for 48 layers) — less than a recompute | **NO-GO** |
| R1a read drafts before the verify sync | (trivial) | the walk's host part is 0.4 ms once the targets are ready; the drafts are evaluated long before | **no gain, not done** |
| **P2** gate+up / q,k,v concat | bit-exact, ≥ 3 % | 1-row qmv on the real weights: **+0.01 ms (MLP) / +0.05 ms (qkv) per step**, bit-equal; the plain step already runs at its GPU time (51.5 ms) | **NO-GO** (not implemented) |
| P2 `MLX_MAX_OPS_PER_BUFFER` / `_MB_PER_BUFFER`, `MLX_METAL_FAST_SYNCH` | ≥ 3 % | plain period 51.5 → 50.3–51.6 ms, DFlash round 126.7 → 128.2 ms; MB=1000: prefill peak **+6.6 GiB, 8K prefill Metal OOM** | **rejected** |
| L1 one-pass verify attention (other agent) | go if ≥ 25 % of the round at ≥ 32K | verify attention **~5 % at 2K, ~36 % at 32K** (70 of 194 ms), 132 ms per round at 60K | **GO** (report only) |
| D1 drafter on mrqmm (other agent) | go if drafter ≥ 18 % of the round | drafter **23 % at 2K** (29.7 of 128 ms), 17 % at 32K | **GO** at short context (report only) |
| K1 mrqmm 4-bit sweep + gate steps 4–5 | — | see §5 | <!-- K1V --> |

What is left on the table besides L1 / D1: the DFlash round exposes **8–9 ms of host time
(4–7 %)**, mostly the verify graph build (4.6–6.4 ms of Python) on an idle GPU — MLX's
`async_eval` returns only ~4 ms before the submitted work ends, and no MLX env setting
changes that (§3). Plain decode is at 89 % of the bandwidth floor at 2K (§1).


Model `weights/qwen38-27b-mlx-4bit` (lmstudio-community, affine g64 4-bit), drafter
`weights/qwen38-27b-dflash2` (4-bit weights, fp32 activations), mlx 0.32.3, mlx-vlm 0.7.6,
M1 Max 32 GB (32 GPU cores), `pmset powermode 2`, `iogpu.wired_limit_mb=27648`.
Worker env everywhere: `QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 QWEN_KV_BITS=0 QWEN_MAX_BATCH=1
QWEN_WIRED_LIMIT_GB=27` (+ `QWEN_DRAFT`, `QWEN_PROMPT_CACHE=1 QWEN_PRESENCE_PENALTY=0` for
spec_prof). Greedy, thinking off. Labels: **[M]** measured here, **[D]** derived from
measured numbers, **[E]** estimate. Raw: `raw/g0_*`.


## 1. Plain decode step (B = 1)

`scripts/perf/prof/decode_classes.py` (worker model setup, the worker's B=1 call
`lm(y, cache, position_ids)`): K = 8 steps built lazily, each fed the previous step's
argmax, then one eval; eval/K = the step's GPU time (encode overlaps). Op classes by
removal variants, interleaved, medians of 5 rounds [M] (`raw/g0_decode_classes.json`):

| ms per step | 2K | 32K | plan's estimate (2K) |
|---|---:|---:|---:|
| **base (whole step, GPU)** | **51.3** | **74.3** | 58 (worker) |
| matmuls (353 qmv incl. lm_head) = base − nolin | 44.6 | 50.0 | 46.5 |
| full attention (16 layers) = base − noattn | 0.7 | 17.0 | 0.5–1 |
| GDN recurrence (48 layers) = base − norec | 1.4 | 2.4 | ~1 |
| GDN mixer = base − gdnmix (conv, SiLU, q/k norm, g/β, recurrence, gated norm: P1's scope) | 3.1 | 3.6 | — |
| **small kernels** = nolin − attn − rec (sub.) / `bare` (direct) | **4.6 / 4.7** | **4.9 / 6.6** | 8–10 |
| graph nodes in one step | 2381 | 2285 | ~1.5K dispatches |

- The matmuls read 14.4 GB per step [D]: 44.6 ms = 323 GB/s, i.e. **at the machine's
  read roof** (bwprobe at the start of the slot: sum 316, max 313, copy 344 GB/s [M]).
  The step's GPU time is 51.3 ms against a 45.6 ms floor (14.4 GB / 316 GB/s): 89 %.
- Attention at 32K: 17.0 ms = 1.06 ms/layer; `verify_att_bench` "single" (one query,
  fp16, 128 MiB of KV per layer) 1.20 ms/layer [M] = ~110 GB/s: the plain decode
  attention reads its KV at a third of the roof (60K: 2.81 ms/layer, 89 GB/s).
- **The worker's plain step runs at its GPU time**: step_prof B=1 at 2K (real scheduler,
  bare consumer, 2 × 300 tokens) period 51.5–51.8 ms, host wait 0.01 ms [M]; the
  standalone step-ahead loop (decode_classes `--pipe`) 51.5 ms vs 51.7 chained [M]; plain
  requests in spec_prof 50.0–50.4 ms/token (19.8–20.0 tok/s) [M]. (A first spec_prof run an
  hour earlier read 59.0 ms/token next to a 316 GB/s bwprobe; the later ones ran at
  336–341 GB/s: that was drift, not a pipeline loss. Every ratio below is in-process.)
- `mx.async_eval` of a step returns only ~4 ms before the step's GPU work ends
  (47.7 of 51.5 ms; `async_cpu` 8.7 ms: the rest is waiting) — MLX keeps the encoder a
  few command buffers ahead of the GPU. Any host work longer than that between two
  submissions idles the GPU. The plain step's host work (graph build 4.0 ms incl. the
  scheduler) fits; the DFlash round's does not (§2).

## 2. DFlash2 round (block 8, verify through mrqmm 4-bit, QWEN_VERIFY_ATT 5+3 queries)

`scripts/perf/dflash2/spec_prof.py`: the worker's own `_spec` rounds, wrapped from
outside; modes alternate per round (normal, stage, normal, deep), the first 4 rounds of
every request skipped. 2K: system document 2000 tokens + `wtest` task 0, 3 DFlash
requests × 512 tokens (first cold, then prompt-cache hits), 2 plain requests between
them; all 5 outputs identical token for token [M] (`raw/g0_spec_prof_2k.json`).

Second slot (08:34–08:46, bwprobe 336–341 GB/s), modes normal / stage alternated; 2K: 3 DFlash
× 512 tokens + 2 plain; 32K (`--tokens 32000`, prompt 32 067 tokens, cold prefill once):
3 DFlash × 384 + 2 plain [M] (`raw/g0_spec_prof_2k_b.json`, `raw/g0_spec_prof_32k.json`):

| ms per round (medians) | 2K normal | 2K stage | 32K normal | 32K stage |
|---|---:|---:|---:|---:|
| **period** | **128.0** | 131.2 | **193.7** | 220.2 |
| drafter graph build | 1.2 | 1.1 | 0.9 | 1.0 |
| `async_eval(drafts)` returns after | 29.7 | — | 30.7 | — |
| drafter, synced (incl. build) | — | **30.8** | — | **33.4** |
| verify graph build | **6.4** | 6.4 | **4.6** | 4.7 |
| `async_eval(targets, hidden)` returns after | 76.2 | — | 136.3 | — |
| verify forward + pick, synced (incl. build) | — | **95.7** | — | **158.2** |
| walk (`.tolist()`: the host waits for the GPU) | 13.0 | 0.4 | 17.3 | 10.4 |
| commit + `_own_states` (host; stage: + their GPU work) | 1.6 | 3.4 | 1.3 | 13.1 |
| between rounds: walk end → next draft submit (`bubble_lo`) | 2.3 | — | 2.2 | — |
| accepted drafts / round | 4.17–4.28 | | 3.67 | |
| **GPU work** [D] = (drafter − build) + (verify − build) | | **119.0** | | **185.9** |
| exposed host time [D] = normal period − GPU work | **9.0 (7 %)** | | **7.7 (4 %)** | |
| tok/s from normal rounds [D] (tokens per round / period) | **41.3** | | **24.1** | |
| plain decode, same process [M] | 19.8 tok/s = 50.4 ms | | 15.2 tok/s = 65.9 ms | |

- DFlash vs plain in-process: **×2.08 at 2K, ×1.59 at 32K** [D]. Outputs: every 2K request
  identical to the first (DFlash == plain greedy, 5/5); at 32K the plain answers differ from
  the DFlash ones (both DFlash repeats identical) — the verify-through-mrqmm numerics class
  (CLAUDE.md "DFlash2 on the 27B": not bit-equal to plain), not checked further here.
- The first 2K run (07:00, bwprobe 316 GB/s, modes incl. `deep`): normal period 141.6, stage
  GPU work 129.8, exposed 11.8 ms, plain 59.0 ms/token (`raw/g0_spec_prof_2k.json`) — the
  same shape on a ~10 % slower GPU.
- **Where the exposed ~8–9 ms go**: `async_eval(drafts)` returns only when the drafter is
  nearly done (29.7 ms ≈ its 29.7 ms of GPU work), so the verify graph build (4.6–6.4 ms
  of Python) runs on an idle GPU; + the drafter's build (~1 ms) and ~2.2 ms between rounds
  (walk end → commit, `_own_states`, emit, tick, `_regrow`, the next draft build). No order
  of the same calls hides the verify build: its graph needs the walk's result of the
  previous round, and MLX's encoder cannot run more than a few ms ahead (§3).
- Drafter share of the round: **23 % at 2K, 17 % at 32K**. Verify attention (vec 5+3 per
  layer × 16, §4): **~6 ms = 5 % at 2K, 70 ms = 36 % at 32K**, 132 ms at 60K [D]. The verify
  grows 89.9 → 153.5 ms of GPU work from 2K to 32K: +64 ms, matching the attention
  microbench's +64.
- `deep` mode (each linear / attention / GDN call synced inside the verify) is dominated
  by its 321 syncs × 0.21 ms (one tiny eval + sync measured at start): verify 296 ms vs
  102. Its shares are not usable; the verify's parts come from the microbenchmarks:

| verify part (2K) | ms | source |
|---|---:|---|
| linears, 8 rows mrqmm 4-bit (64 layers + lm_head) | ~76 | [Ms] plan (kern5_bench, ×1.64 of a solo row) |
| GDN with-states, T = 8, 48 layers | 5.5 | [M] `microbench_gdn --calls 48` (plain kernel at T=8: 3.6; T=1: 2.1) |
| block attention, 16 layers × `vec` 5+3 queries | ~6 | [M/D] `verify_att_bench` (8K: 1.20 ms/layer; 2K less) |
| everything else (norms, conv, gates, splits, KV append) × 8 rows | ~10 | [D] remainder of the first run's 97.4 ms |


## 3. MLX command-buffer settings (cross-process, same scripts)

| setting | plain step-ahead period (2K) | 2K prefill peak | DFlash round (2K, normal) |
|---|---:|---:|---:|
| default | 51.5 / 51.6 ms | 16.88 GiB | 126.7 ms |
| `MLX_MAX_MB_PER_BUFFER=1000` | 50.3 ms (async_eval still 46.3) | **23.46 GiB**; the 8K prefill then **failed: Metal OOM** | — |
| `MLX_METAL_FAST_SYNCH=1` | 51.6 ms | 16.88 GiB | — |
| `MLX_MAX_OPS_PER_BUFFER=200` | — | — | 128.2 ms (async_eval(drafts) still 29.5) |

None moves the point where `async_eval` returns; the MB limit costs 6.6 GiB of prefill
peak. **Rejected; no knob.** [M] (`raw/g0_pipe_*.json`, `raw/g0_spec_prof_2k_{ops200,default}.json`)

## 4. Microbenchmarks

| what | result | source |
|---|---|---|
| bwprobe (1 GiB fp16) | sum 316 → 336 → 341 GB/s over the night's slots (copy 344–356) | `raw/g0_bwprobe_*.json` |
| GDN recurrence, 48 layers, 27B shapes (Hv 48, Dk = Dv 128), one eval | T=1 plain 2.1 ms; T=8 plain 3.6, **with-states 5.5**, `mx.fast` 2.3 | `raw/g0_gdn_states.json` |
| verify attention per layer, 27B (H 24 / KV 4, D 256), T = 8 | 8K: vec 1.20 / rows 1.38 / single 0.25 ms; 32K: **vec 4.39** / rows 9.25 / single 1.20; 60K: **vec 8.28** / rows 23.8 / single 2.81 | `raw/g0_verify_att_bench.txt` |
| P2: 1-row qmv, gate+up concatenated vs separate (64 layers, real 4-bit weights) | 19.47 vs 19.48 ms (**+0.01 ms**), slices bit-equal | `raw/g0_concat_qmv.json` |
| P2: q/k/v concatenated vs separate (16 layers) | 1.99 vs 2.04 ms (**+0.05 ms**), bit-equal | same |
| drafter stages, 4-bit fp32, synced each (draft_prof; lm_head on qmm, the worker's is mrqmm) | MLP 15.4, lm_head 11.4, conv prepare + finish 7.7, attention 6.1, norms 2.5, selector 2.5, fc 1.2; one-eval draft_block 35.2 ms | `raw/g0_draft_prof.txt` |

Verify attention at 8 queries: the vector kernel's `vec` (5 + 3 queries per call) reads the
KV twice; 4.39 ms at 32K = 128 MiB × 2 / 4.39 ms = 61 GB/s, 60K 61 GB/s — a fifth of the
roof (one-pass L1 is the other agent's lever).



## 6. Commands

From `src/worker`, `ENV27="QWEN_MODEL=../../weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1
QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 PYTHONPATH=."`, `setopt NO_BG_NICE`, nothing
else on the GPU:

```zsh
.venv/bin/python ../../scripts/perf/microbench/bwprobe.py --json $PERF_OUT/bwprobe.json
env $ENV27 .venv/bin/python ../../scripts/perf/prof/decode_classes.py --ctx 2048,32768 --K 8 --rounds 5
env $ENV27 .venv/bin/python ../../scripts/perf/prof/decode_classes.py --ctx 2048,8192 --only-pipe --pipe 48 --rounds 3
env $ENV27 QWEN_BATCH_PREFILL_SHARE=1 .venv/bin/python ../../scripts/perf/prof/step_prof.py --B 1 --ctx 2k --gen 300 --consumer bare --rounds 2
env $ENV27 QWEN_DRAFT=../../weights/qwen38-27b-dflash2 QWEN_PROMPT_CACHE=1 QWEN_PRESENCE_PENALTY=0 \
  .venv/bin/python ../../scripts/perf/dflash2/spec_prof.py --tokens 2000 --reqs 3 --plain 2 --max-tokens 512 --modes normal,stage
#   32K: --tokens 32000 --max-tokens 384
MODEL=../../weights/qwen38-27b-mlx-4bit DRAFT=../../weights/qwen38-27b-dflash2 PYTHONPATH=. \
  .venv/bin/python ../../scripts/perf/dflash2/draft_prof.py
for N in 8192 32768 61440; do .venv/bin/python ../../scripts/perf/dflash2/verify_att_bench.py $N 8; done
.venv/bin/python -I ../../scripts/perf/microbench/microbench_gdn.py --label 27b --config ../../weights/qwen38-27b-mlx-4bit \
  --cases 1x1,1x8 --calls 48 --iters 30
.venv/bin/python ../../scripts/perf/microbench/microbench_concat_qmv.py --weights ../../weights/qwen38-27b-mlx-4bit
```

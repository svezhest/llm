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
| K1 mrqmm 4-bit sweep + gate steps 3–5 | config within 0.1 %; numerics in class; DFlash ×1.6; identity ≥ 15/16 | `_config` best at 8 rows (78.1 ms vs 78.0); batched step ×1.5–2.95 vs qmm; DFlash **×1.75** (32.3 vs 18.4 tok/s); teacher-forced top-1 100 / 100 / 99.80 % at B = 3/8/16, KL ≤ 1e-5 (stock qmm: 100 / 99.90 / 99.85 %); identity **12/16**, all 4 at near-ties (gap 0–0.031) | **no config change; QWEN_MRQMM4 stays default off** (identity count below 15/16, the 35B's 4-bit shared expert ungated) — use `QWEN_MRQMM4=1` for the 4-bit 27B |

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
| linears, 8 rows mrqmm 4-bit (64 layers + lm_head) | 78.1 | [M] §5 step 3 (kern5_bench on the real 4-bit weights, ×1.65 of a solo row; the plan said ~76) |
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



## 5. K1: mrqmm 4-bit, config sweep and gate steps 3–5 (`scripts/perf/dflash2/gate_mrqmm4.sh`)

**Step 3** (kern5_bench on the real 4-bit weights, no model; `raw/k1_kern4_bench_real.json`) [M]:
linears of one 27B forward (64 layers + lm_head):

| rows | qmv per row (1 row) | MLX `quantized_matmul` | mrqmm (`_config`) | 2x8 | 4x8 | 4x4 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 47.6 | 93.9 (×1.97) | 71.1 (×1.49) | | | |
| 4 | 47.1 | 121.3 (×2.58) | 71.7 (×1.52) | | | |
| 6 | 47.1 | 240.8 (×5.11) | 74.1 (×1.57) | | | |
| **8** | 47.2 | 243.7 (×5.16) | **78.1 (×1.65)** | 78.0 | 78.1 | 80.0 |
| 12 | 47.7 | 249.2 (×5.22) | 115.5 (×2.42) | | | |
| 16 | 47.1 | 246.7 (×5.24) | 116.1 (×2.47) | 115.9 | 116.6 | 115.5 |

`mrqmm._config`'s 4-bit choice (4x8 at ≤ 8 rows, 2x8 above) is within 0.1 % of the best
config at 8 rows (4x4 is 2.6 % slower) and within 0.5 % at 16: **no config change**. The
8-row verify block's linears cost ×1.65 a solo row's — the plan's [Ms] number, now on the
real weights.

**Step 4** (step_prof `--ab base,nomr --ab-steps 32`, one process, the model without the
drafter, ~600-token prompts; `raw/k1_step_ab_4bit.{json,log}`) [M]: decode step period, mrqmm vs
`quantized_matmul` for the same rows:

| B | mrqmm (base) | `quantized_matmul` (nomr) | |
|---:|---:|---:|---:|
| 3 | 80.2 ms | 120.2 ms | ×1.50 |
| 4 | 82.9 | 155.4 | ×1.88 |
| 8 | 103.2 | 304.5 | ×2.95 |
| 16 | 166.8 | 331.0 | ×1.98 |

Pass (base beats nomr at every B). step_prof does not configure INFO logging, so the load
log's self-test line is not in this file; the kernel was evidently taken (the A/B differs
×1.5–3); step 5's log shows it.

**Step 5** (`scripts/perf/dflash2/ab27.py`, plain / `m8/8` = DFlash block 8 with the verify through
mrqmm (`QWEN_VERIFY_MRQMM=1`, the default) / `e0/8` = block 8 through mlx-vlm's exact verify kernels,
5 tasks × 3 rounds interleaved, 256 tokens, greedy, thinking off, then a 16-task identity pass;
`raw/k1_ab27_4bit.{json,log}`) [M]. The load log has `mrqmm 4-bit g64 self-test passed`.

| variant | tok/s (median of tasks) | per task | accepted / round | round | identical to plain |
|---|---:|---|---:|---:|---:|
| plain | 18.44 | 18.2–18.6 | — | — | — |
| **m8/8** | **32.27 (×1.75)** | 22.6–37.9 | 3.40 | 136.7 ms | 12/15 (speed tasks), **12/16** (identity pass) |
| e0/8 | 13.22 (×0.72) | 9.3–15.7 | 3.40 | 331.0 ms | 15/15 |

Speed passes the gate's ×1.6 bar; without the kernel DFlash on the 4-bit model is slower than
plain (the exact verify kernels cost 2.4× a mrqmm round). Peak 17.97 GB, active 16.52 GB.
Identity: 12/16 with divergences at tokens 151, 56, 33, 202 — below the night rules' ≥ 15/16
(the 5-bit kernel had 15/16 without thinking on the same tasks). The e0/8 column shows the
drafter path itself is exact, so the four divergences come from the mrqmm verify rows (the
batched-decode numerics class). Whether they are near-ties, and the teacher-forced agreement
of the kernel against stock qmm at B = 3/8/16, follow.

**Near-ties** (`scripts/perf/dflash2/diverge_margin.py` with `MODEL=…-4bit`: plain greedy through
the library, single-row qmv, top-1 − top-2 logit gap at the diverging token; `raw/k1_diverge_margin_4bit.txt`,
`raw/k1_ab27_4bit_margins.json`) [M]:

| task | diverges at | gap there | median gap before | earlier steps with a smaller gap |
|---:|---:|---:|---:|---:|
| 3 | 151 | 0.0156 | 7.91 | 0 |
| 10 | 56 | 0.0312 | 8.23 | 0 |
| 11 | 33 | 0.0 | 2.30 | 0 |
| 15 | 202 | 0.0156 | 2.43 | 5 |

All 442 earlier steps: gap p1 0.016, p5 0.172, p50 4.09. Every divergence is at an fp16
tie or one ulp from it (fp16 logit spacing at these magnitudes is 0.0156).

**Teacher-forced agreement** (`scripts/perf/prof/agree.py`, `AG_BATCHES=3,8,16 AG_STEPS=128`, batched
vs solo decode of the same rows, the solo greedy stream fed to both; `raw/k1_agree_mrqmm4_{1,0}.txt`) [M]:

| B | `QWEN_MRQMM4=1` top-1 / KL mean / KL max | `QWEN_MRQMM4=0` (stock qmm) |
|---:|---|---|
| 3 | 100.00 % / 9.5e-6 / 4.4e-4 | 100.00 % / 0 / 0 (per-row qmv: bit-exact) |
| 8 | 100.00 % / 7.1e-6 / 4.4e-4 | 99.90 % / 6.8e-6 / 5.5e-4 |
| 16 | 99.80 % / 7.6e-6 / 1.0e-3 | 99.85 % / 7.4e-6 / 5.5e-4 |

Peak 19.1 / 22.2 GB at B = 8 / 16 (two copies of every row). The kernel is in the batched-decode
numerics class (KL ≤ 1e-4, top-1 ≥ 99.7 %) and no worse than stock qmm, whose 8–16 rows are
not bit-exact either; at B = 3 stock is per-row qmv (exact), the kernel is not.

**K1 verdict**: speed (×1.75 DFlash, ×1.5–2.95 batched step) and numerics (KL ≤ 1e-5,
top-1 ≥ 99.80 %) pass; DFlash identity is 12/16 — every divergence at a near-tie, but the
count is below the night rules' 15/16 (the 5-bit kernel is 15/16 without thinking and stays
default off at 12/16 with thinking). A default flip would also put the 35B's 4-bit shared
expert on the kernel, which no gate has measured. **`QWEN_MRQMM4` stays default off**; it
belongs in the 4-bit 27B recipe (without it DFlash on that model is ×0.72 of plain).
Lease hold for this section: 12:02–12:13.

## Open issues

- **The verify graph build, 4.6–6.7 ms of Python per round on an idle GPU** (§2). cProfile of
  the build in normal rounds (`spec_prof --cprofile`, 2K, 196 verify calls; `raw/g0_spec_prof_cprofile.txt`)
  [M]: 6.7 ms per build under the profiler, no single hot spot — 353 `mrqmm` calls per round
  (1.1 ms own time), mlx-vlm's `speculative_verifier._gated_delta` (3.0 ms cumulative, of which
  the with-states kernel wrapper 0.5 and the qk normalization 0.2), `_verify_linears` /
  `_mr_rows` dispatch (1.7 ms cumulative), `nn.Module.__getattr__` (0.4 ms, 1400 calls). Trimming
  the Python (cached kernel handles, fewer attribute lookups) would recover ≤ ~2 ms of the
  134 ms round (≤ 1.5 %) — below the 3 % bar, not done. Hiding it needs MLX to encode further
  ahead (§3: no env setting does) or a verify graph that does not depend on the previous
  walk (it does: the block's first token is the walk's bonus token).
- **4-bit DFlash ≠ plain greedy** on 4 of 16 tasks (§5 step 5); the exact verify path (e0/8)
  is identical but slower than plain. At 32K the DFlash output also differs from plain
  (§2, same numerics class).
- `step_prof.py` does not configure INFO logging: load-log lines (e.g. the mrqmm self-test)
  are not in its output.

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

# The 27B's native MTP head as a speculative drafter (QWEN_DRAFT=<MTP dir>)

Qwen3.8-27B ships one multi-token-prediction layer (`mtp_num_hidden_layers: 1`). The oQ4e
checkpoint (`weights/qwen38-27b-mlx-oq4e`) keeps it as `language_model.mtp.*`; the
uniform 4-bit lmstudio build (`qwen38-27b-mlx-4bit`, "u4" below) drops it. This report
covers the drafter built from that head, the exact multi-row qmv for 2–3-token verify
blocks, the adaptive draft count, the per-round DFlash / MTP / plain chooser, and the gate.
Machine: M1 Max 32 GB, mlx 0.32.3, mlx-vlm 0.7.6. Labels: [M] measured, [D] derived from
measured numbers, [E] estimated.

## Design

`src/worker/qwen_worker/mtp.py` + `_Scheduler._spec` (`sched_spec.py`), `src/worker/mrqmv.py`.

- **Split** (`scripts/perf/mtp/split_mtp.py`, CPU): the `language_model.mtp.*` keys of an
  MLX checkpoint (mlx-vlm 0.7.6's own `split_mtp` looks for `mtp.*` only) into an mlx-vlm
  `qwen3_5_mtp` drafter dir, tensors and per-module quantization untouched (oQ4e: 4-bit
  base, k / v 8-bit, o 5-bit; 308 MiB). `weights/qwen38-27b-mtp` is the oQ4e's head; the u4
  numbers below ran the same head on the u4 target (the u4 has none of its own).
- **Head** (mlx-vlm `Qwen3_5MTPDraftModel`): for position t it takes the target's post-norm
  hidden state h_t and the embedding of x_{t+1}, one full-attention decoder layer with its
  own KV cache, the target's norm + lm_head → x_{t+2}. `MTPDraft` keeps one head entry per
  target position: `push` (hidden states of positions that entered the target cache:
  prefill slabs, the kept part of a verify block, plain steps), `emit` (tokens in emission
  order), `flush` (the head over every complete (h, successor) pair at once, lazy).
  `draft(k)`: the seed (the head's output at the last entry) + k − 1 chained steps that feed
  the head its own output hidden as the stand-in for the target's (approximate entries);
  `round_end(accepted, h)` trims the rejected entries and keeps the accepted ones (as
  mlx-vlm; `QWEN_MTP_REFEED=1` recomputes them from the target's real hidden states).
  The head's KV is fp16 at 4 KiB / position on the 27B, charged to the row
  (`MTPDraft.token_bytes` in `kv_fp16`).
- **Round**: the same verify forward, `pick`, `_speculative_walk`, commit / rollback,
  thinking-budget cut, reply KV, demotion and prompt-cache paths as DFlash (one code path in
  `_spec`); greedy output is the verify's, so the drafter only moves acceptance. Sampling
  (`QWEN_DRAFT_SAMPLING=1`): the drafts are the head's argmax; the verify draws with the
  request's sampler and keeps a draft iff the draw equals it (exact for a deterministic
  drafter, as DFlash).
- **Hidden stream**: `return_hidden` gives the target's final hidden state as one more
  capture beside DFlash's target layers; DFlash reads `h[..., :layers × H]`, MTP
  `h[..., -H:]`. Prompt-cache tails (`QWEN_DRAFT_TAIL` / `_END`) carry the stream; a tail of
  another width (stored by another drafter set-up) is ignored.
- **Adaptive k** (`QWEN_MTP_K=0`, `MTPDraft.next_k`): per-depth acceptance EMAs
  p_i = P(draft i kept | drafts < i kept), a depth not drafted yet by the request inheriting
  the shallower one's; the process-wide table `MTPDraft.MS` of measured round wall times per
  k (prior `1 + QWEN_MTP_COST·k` scaled from the nearest measured k); k = argmax over
  1..`QWEN_MTP_KMAX` of (1 + Σ_i Π_{j≤i} p_j) / ms(k); every 8th round one deeper or (alternately)
  one shallower so the neighbours' estimates stay fresh.
- **Chooser** (`QWEN_DRAFT_HYBRID=1` with both drafters loaded, `QWEN_SPEC_PLAIN=1` adds a
  plain arm; `mtp.Chooser`): per round the arm with the most expected tokens per ms; tokens
  per round start at a prior worth 2 rounds (DFlash 3, MTP 2.5, plain 1) and follow an EMA;
  wall time per round (the first round of an arm replaced by its second: a DFlash round
  right after prefill feeds the whole prompt window); every arm is probed 2 rounds first and
  re-probed after 64 tokens without it. DFlash rounds the chooser skips hand their hidden
  states to `_DraftCtx` (the drafter catches up on its next round); MTP entries are pushed
  every round, whoever drafted.
- **mrqmv** (`QWEN_VERIFY_QMV=N`, N ≤ 3): verify blocks of 2..N tokens on 4-bit g64 modules
  (+ fused GDN in-projection groups of 4-bit g64) through a multi-row qmv whose every row is
  MLX's `qmv_fast` arithmetic (x prescaled by 1 / 16 / 256 / 4096, masked uint16 weights,
  `s · accum + sum · b`, `simd_sum`): row r == `quantized_matmul(x[r])` bit for bit
  (self-test per (K, N) at load against MLX, `array_equal`; a failing shape stays on the
  library). MLX's own paths: T = 2 goes to mlx-vlm's exact verify kernels (u4: 81–86 ms),
  T ≥ 3 to mrqmm (86–91 ms flat over 3..6); mrqmv T = 2: 66 ms [M, u4, synced round_prof].
  Rows ≥ 4 spill registers (×5–15 slower), hence the cap 3. On the oQ4e only its 4-bit
  modules take it (5 / 6 / 8-bit modules: the library's path).

## Gate method

In-process A/B (`scripts/perf/mtp/ab_mtp.py`, the real `Engine` + `_Scheduler`, no gRPC):
one worker with both drafters loaded, variants interleaved per (round, prompt), 256 generated
tokens, greedy, thinking off, the cross-framework bench27 prompts code-2k-0 / en-2k-0 / ru-2k-0
(~2K tokens of context each; `BENCH27` = the prompt dir of the bench27 harness). Decode tok/s =
(n − 1) / (t_last − t_first). Because the machine drifted hard during the runs (plain decode
of the same prompt 18.4 tok/s in a run's warm-up, 11.6–13.5 a few minutes later: ±30 % as
CLAUDE.md warns), the tables give the **ratio to plain decode of the same (round, prompt)
pair**, median over pairs [M], with the median tok/s beside it. oQ4e env:

    QWEN_MODEL=weights/qwen38-27b-mlx-oq4e QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 QWEN_MRQMM5=1 QWEN_MRQMM6=1
    QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_PRESENCE_PENALTY=0
    QWEN_DRAFT=weights/qwen38-27b-dflash2 QWEN_DRAFT_MTP=weights/qwen38-27b-mtp QWEN_VERIFY_QMV=3

Variants: `plain`; `dflash` (DFlash2 alone, main's defaults: `QWEN_DRAFT_MRQMM=1`,
`QWEN_VERIFY_ATT=2`, block 8); `mtp0+q` (MTP alone, adaptive k, mrqmv on every width);
`mtp0+q4` (mrqmv on the 4-bit modules only: what the first oQ4e run had); `mtp0+q+k6`
(`QWEN_MTP_KMAX=6`); `hyb0+q+p` (both drafters + the plain arm, `mtp.Chooser`).

The GPU budget ran out before the planned 3 full rounds: run A (`raw/mtp_oq4e_greedy_a.*`)
has round 0 and the start of round 1, run B (`raw/mtp_oq4e_greedy_b.*`) round 0 (both
stopped at the 25-minute hold cap). n = pairs per cell. **No sampling and no identity run on
the oQ4e** (a sampling run was started and killed within seconds when the hold ended).

## Results, oQ4e, greedy [M]

Ratio to plain of the same pair (median, [min–max]), median tok/s (throttled machine), drafts
accepted + 1 per round / round wall time:

| variant | code-2k | en-2k | ru-2k |
|---|---|---|---|
| plain | 13.1 tok/s (n 3) | 12.0 (n 2) | 12.1 (n 2) |
| dflash | **×2.14** [1.98–2.22], 26.6, 4.81 / 181 ms (n 3) | **×1.43** [1.42–1.44], 17.1, 3.07 / 179 ms (n 2) | ×0.89 [0.88–0.89], 10.7, 1.89 / 176 ms (n 2) |
| mtp0+q (all widths) | ×1.76, 23.2, 3.59 / 155 ms (n 1) | ×1.39, 16.8, 1.90 / 113 ms (n 1) | **×1.28**, 16.0, 1.85 / 115 ms (n 1) |
| mtp0+q4 (4-bit only) | ×1.61 [1.25–1.79], 2.68 / 144 ms (n 3) | ×1.32 [1.25–1.39], 2.11 / 132 ms (n 2) | ×1.24 [1.23–1.25], 1.88 / 125 ms (n 2) |
| mtp0+q+k6 | ×1.49 [1.34–1.64] (n 2) | ×1.26 (n 1) | ×1.25 (n 1) |
| hyb0+q+p | ×1.80 (n 1) | ×1.21 (n 1), 91 plain steps | ×0.99 (n 1), 179 plain steps |

- **DFlash wins code and English, MTP wins Russian** (DFlash is below plain there: 12.9 % of
  its drafts accepted). On code MTP's adaptive k varied between processes (run A 2.4–2.7
  tokens per round, mostly k ≤ 2; run B 3.6, mostly k = 4): the per-k round-time table
  (`MTPDraft.MS`) is learned per process, and in run A the slow machine made the deep
  rounds look expensive.
- **mrqmv on 5 / 6 / 8 bits**: 498 modules + 48 fused GDN groups instead of 255 + 33, all
  self-tests passed, load 7.7 s either way. In-run pair B: ru round 123 → 115 ms
  (×1.04 tok/s), en round 140 → 113 ms at the same tok/s (fewer drafts per round: the
  adaptive k moved), code equal. One pair per prompt: an indication, not a gate.
- **KMAX 6** did not help (×1.49 vs ×1.61 on code, run A): depths 5–6 of the chained head
  are rarely accepted (7 + 1 of 96 rounds kept 5 or 6 drafts).
- **The hybrid loses** to the better single drafter on every class. With the plain arm it
  took 91 (en) and 179 (ru) plain steps of 256 tokens: an arm's measured round time includes
  the catch-up after a switch (DFlash's drafter feeds every position since its last round,
  MTP flushes its pending pairs), so a drafter that was not used looks slower than it is and
  plain wins the comparison (round times of 293 / 481 ms on the hybrid rows). A fix would
  measure steady-state rounds only and add hysteresis; not done (no GPU time left), so the
  recommendation is one drafter per worker.

Derived absolute numbers (ratio × the cool-machine plain decode measured in the same
process's warm-up, 18.4 tok/s on code-2k) [D], against the cross-framework smoke of
llama.cpp q4_0 + MTP (64 tokens, one request each: code 34.5, ru 22.9, en with sampling 30.4):

| | code-2k greedy | ru-2k greedy | en-2k sampling |
|---|---|---|---|
| llama.cpp MTP (smoke) | 34.5 | 22.9 | 30.4 |
| ours, DFlash | **≈ 39** (warm-up rows measured: 34.9, 31.8) | ≈ 16 | not measured on oQ4e |
| ours, MTP | ≈ 32 (warm-up: 29.7, 26.0) | **≈ 23.6** | not measured |

So with the right drafter per class the greedy targets are met (code clearly, Russian by a
hair: ≈ 23.6 vs 22.9 is inside the run-to-run spread); with one drafter per worker they are
not met on both at once. English with sampling is open: no oQ4e sampling run, and main's
DFlash sampling gate (u4, `drafter.md` S1) gave ×1.41 on en-2k ≈ 26 tok/s [D] < 30.4.

## Results, u4 (lmstudio 4-bit), greedy [M], second column

Before the merge with main (no `QWEN_DRAFT_MRQMM`, no vatt), adaptive k before 8da61b7,
mrqmv 4-bit only, 3 full rounds (`raw/mtp_u4_greedy.*`):

| variant | code-2k | en-2k | ru-2k | identical to plain |
|---|---|---|---|---|
| plain | 18.0 | 18.2 | 18.2 | — |
| dflash | 28.2 (×1.59) | 19.0 (×1.06) | 13.6 (×0.74) | 6/9 |
| mtp0+q | 27.5 (×1.55) | 21.7 (×1.20) | 22.9 (×1.26) | 8/9 |
| mtp1+q | 25.2 (×1.40) | 21.8 (×1.22) | 23.2 (×1.28) | **9/9** |

Main's drafter work since then moved DFlash from ×1.59 to ×2.14 on code and from ×1.06 to
×1.43 on English (oQ4e), which is why DFlash and not MTP is the code / English drafter now.
Identity: `mtp1+q` (k = 1: every verify block 2 tokens on mrqmv, bit-exact rows) reproduces
plain greedy; deeper blocks go through mrqmm (T ≥ 4 on oQ4e), the batched-decode numerics
class as DFlash's (u4: 8/9). Not re-measured on the oQ4e.

## Recommended 27B recipe (oQ4e)

- **Greedy, code / English / agentic** (the default case): DFlash2 as today,
  `QWEN_DRAFT=weights/qwen38-27b-dflash2` (main's defaults). Code ×2.14, English ×1.43.
- **Greedy, Russian or other prose where DFlash accepts < 1 draft per round**: the MTP head,
  `QWEN_DRAFT=weights/qwen38-27b-mtp QWEN_MTP_K=0 QWEN_VERIFY_QMV=3` (+ the usual oQ4e
  `QWEN_MRQMM4/5/6=1`, fp16 KV). Russian ×1.28 (DFlash ×0.89), English ×1.39, code ×1.76.
- **Do not load both** (`QWEN_DRAFT_MTP` beside DFlash): the per-round chooser loses (above).
- **Sampling**: DFlash with `QWEN_DRAFT_SAMPLING=1` on code / English (main's S1 on u4:
  ×1.36–1.41 on 2K prompts; ×0.70 on short Russian prose → plain there). MTP with sampling is
  implemented (exact: the verify draws with the request's sampler and keeps a draft iff the
  draw equals it) and covered by CPU tests, but **not measured on the GPU**.

Defaults changed: `QWEN_MTP_K` 3 → 0 (adaptive; only read with an MTP drafter, and the fixed
k = 3 was never the best measured setting). Everything else stays opt-in: `QWEN_VERIFY_QMV`
`0` (bit-exact by its self-test, but its oQ4e gain rests on one pair), `QWEN_MTP_KMAX` 4.

## Open

- The chooser's switching cost (steady-state round times + hysteresis), then a re-gate: the
  only way to beat llama.cpp on code and Russian with one worker.
- oQ4e sampling A/B (plain / DFlash / MTP) and an identity run (`--ident mtp0+q,mtp1+q`).
- mrqmv rows ≥ 4 spill registers; T = 4..5 verify blocks stay on mrqmm (×1.6 of a row).
- The MTP draft step's lm_head (6-bit, 0.95 GB on oQ4e) is most of a draft step's ~3.5 ms; a
  reduced draft vocabulary would need frequency ranks (Russian tokens sit at ids ≥ 150K).
- The plain arm is synchronous (no step-ahead).

## Files

`src/worker/qwen_worker/mtp.py`, `sched_spec.py` (MTP / chooser hooks), `engine.py`,
`kernels.py` (`_setup_verify_qmv`, verify hooks), `src/worker/mrqmv.py`,
`tests/test_mtp.py`, `tests/test_mrqmv.py` (GPU), `scripts/perf/mtp/` (`split_mtp.py`,
`ab_mtp.py`, `round_prof.py`), raw runs in `raw/mtp_*`.

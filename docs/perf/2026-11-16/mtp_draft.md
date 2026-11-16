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
  recommendation is one drafter per worker. (2026-11-17: this explanation does not hold,
  see "`QWEN_DRAFT=auto`" below.)

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

## `QWEN_DRAFT=auto`: one setting, the chooser rewritten (2026-11-17, GPU gate pending)

Status: code + CPU tests on branch `chooser`; **nothing below is measured on the GPU yet**
(the tables are laid out, cells empty). [M] here = read from the raw files of the runs above.

**The diagnosis above was wrong** [M, `raw/mtp_oq4e_greedy_a.json`]. The "round times of
293 / 481 ms" on the hybrid rows are `ab_mtp.py` dividing the decode time by the drafter
rounds only (the plain steps' time included, their count not); the drafter rounds of those
requests took 150–166 ms, as alone. The catch-up after a switch is cheap (DFlash's context
positions pass `fc` + norm + the k/v projections, not its layers). What lost the rounds: the
chooser compared a noisy average (EMA 0.2) of the arm in use with the *frozen* estimates of
the arms not in use, frozen at the dip at which each lost, and re-probed them only after 64
tokens. The plain arm has no noise, so it collected the rounds (91 / 179 of 256 tokens)
although both drafters beat it.

**The chooser now** (`mtp.Chooser`, `mtp.Shadows`, `mtp.Clock`, `mtp.Avg`; `_spec`):

- *Shadow drafts.* An arm not taken still drafts now and then, without a verify; its drafts
  are scored against the tokens the request emits after the same anchor (the matching
  prefix = what a verify would have kept; greedy: the target's argmax, sampling: its
  draws). Cost: the draft alone (MTP ~3.5 ms per draft, a DFlash block ~17 ms [M: above, `drafter.md`],
  against 110–180 ms rounds). Every round until an arm has 3 drafts and while it looks
  better than the current arm from < 2.5 recent samples (a trial); else as often as costs
  1 % of the time (an arm within 0.8 of the current rate) or 0.4 %.
- *Tokens per round*: means weighted exp(−age / 2.5 s) with a prior worth 2 samples
  (DFlash 3.0 tokens; the head 0.5 per depth, at its best draft count); plain = 1.
- *Round times*: process-wide per request class (context size class < 6K / 24K / 64K / more,
  sampling, thinking budget), from steady rounds only (the same arm as the round before, no
  shadow drafts, a full block), kept as drift × per-kind constant so that the kinds not
  taken follow the machine's drift. A class's first request takes each arm 3 rounds (probe).
- *Choice*: a request starts on the MTP head (the smallest loss when it is the wrong arm:
  the oQ4e costs above, plain 80 : MTP k=1 113 : DFlash 180 ms, give a worst-case regret of
  ~18 % for the head, 33 % for DFlash, 53 % for plain [D]); the current arm is left only for
  one whose rate is 2 % higher and (a drafter) known from ≥ 2.5 recent samples.
- *The plain arm* is a burst of 8 plain steps with the plain loop's step-ahead (2 while a
  drafter is on trial); penalties are counted as in the batch path, a thinking budget ends
  the burst where it forces a token.
- `QWEN_DRAFT=auto` (config `_auto`, `engine._auto_drafters`): the drafters are found next
  to the model dir by their `config.json` (hidden size / layers / vocabulary of the target;
  `QWEN_DRAFT_DFLASH` / `QWEN_DRAFT_MTP` name one, `none` drops one; the load log says what
  was found), and the defaults become `QWEN_DRAFT_SAMPLING=1`, `QWEN_SPEC_PLAIN=1`,
  `QWEN_LOOKUP=draft`, `QWEN_VERIFY_QMV=3`. One drafter or none: it loads and runs (none:
  the plain loop). The explicit knobs stay (the A/B variants).
- Quantized KV: MTP and hybrid rounds follow DFlash's rules (no block across
  `quantized_kv_start`, requantize after every round and plain step). The head's own cache
  stays fp16 (4 KiB/token, now charged to quantized rows too); a prefill slab enters the
  head 512 rows at a time (its attention scores are rows × keys × heads: a 2048-row slab at
  32K keys was a 3 GiB transient — a candidate for the fp16 32K Metal OOM, not confirmed).

CPU tests (`tests/test_mtp.py`, `tests/test_kvq8.py`; tiny models): the chooser's rules
(start arm, margin, evidence, trial, shadow cadence, probe, the old failure as a regression
test), `Avg` / `Clock` / `Shadows`; scheduler level: shadow drafts leave the answer and both
drafters' state as their own rounds would (the head's cache == the head over the real
pairs); plain bursts × penalties × thinking budget == the plain loop; sampling through all
three arms is reproducible by seed; q8 KV: MTP (k 0 / 1 / 3) and the hybrid in every arm
order == plain greedy at 6–7 positions of `quantized_kv_start`, the final KV == the plain
loop's (up to lone one-level flips of q8 elements); drafter discovery.

### Gate (to run; `scripts/perf/mtp/gate_auto.sh`, in-run interleaved, ratio to plain of the same pair)

    WEIGHTS=<weights> PERF_OUT=<dir> scripts/perf/mtp/gate_auto.sh floor   # ~8 min
    WEIGHTS=<weights> PERF_OUT=<dir> scripts/perf/mtp/gate_auto.sh dump    # ~8 min + CPU replay
    WEIGHTS=<weights> PERF_OUT=<dir> scripts/perf/mtp/gate_auto.sh gate    # ~45 min (GATE_MODES=sampling | greedy: half)
    WEIGHTS=<weights> PERF_OUT=<dir> scripts/perf/mtp/gate_auto.sh q8      # ~20 min, QWEN_KV_BITS=8, ru-8k-0 greedy

Pass: `auto` ≥ 0.97 × the best of {plain, dflash, mtp0+q} in every cell (`gate_table.py`).
Cells: greedy / sampling (T 0.6, top_p 0.95, top_k 20) × code-2k-0, en-2k-0, ru-2k-0,
short-en, short-ru (`prompts2k.SHORT`); 256 tokens, thinking off, 2 rounds.

| cell | dflash × plain | mtp0+q × plain | auto × plain | plain tok/s | pairs | auto / best | |
|---|---|---|---|---|---|---|---|
| code-2k-0@s | | | | | | | |
| en-2k-0@s | | | | | | | |
| ru-2k-0@s | | | | | | | |
| short-en@s | | | | | | | |
| short-ru@s | | | | | | | |
| code-2k-0 | | | | | | | |
| en-2k-0 | | | | | | | |
| ru-2k-0 | | | | | | | |
| short-en | | | | | | | |
| short-ru | | | | | | | |
| ru-8k-0, `QWEN_KV_BITS=8` | | | | | | | |

The arms `auto` took (share of rounds / of tokens; shadow drafts and switches per request):

| cell | rounds: dflash / mtp / plain / lookup | tokens: dflash / mtp / plain / lookup | shadows (dflash / mtp) | switches |
|---|---|---|---|---|
| (as above) | | | | |

Floor (`floor`): the plain arm alone inside `_spec` (`pplain`) vs the plain loop, pass ≥ 0.97:

| cell | pplain × plain | pairs |
|---|---|---|
| en-2k-0@s / short-ru@s / en-2k-0 / short-ru | | |

Replay (`dump` → `chooser_sim.py`, CPU): the worker's own `Chooser` / `Shadows` / `MTPDraft`
classes over both drafters' drafts at every position of each cell's answer, with the round
times of the run's clock; `--set TAU=…,MARGIN=…,WARM=…,NEED=…,START=dflash/mtp/plain` to try
constants without the GPU, `--stitch a,b` for a change of the text's kind inside a request.

## Open

- The `QWEN_DRAFT=auto` gate (section above): nothing of the rewritten chooser is measured.
  Its constants (start arm, priors, TAU, margin, shadow budget) are first guesses from the
  greedy costs above; the replay on a dump is the way to set them.
- oQ4e sampling A/B (plain / DFlash / MTP) and an identity run (`--ident mtp0+q,mtp1+q`).
- mrqmv rows ≥ 4 spill registers; T = 4..5 verify blocks stay on mrqmm (×1.6 of a row).
- The MTP draft step's lm_head (6-bit, 0.95 GB on oQ4e) is most of a draft step's ~3.5 ms; a
  reduced draft vocabulary would need frequency ranks (Russian tokens sit at ids ≥ 150K).
- The plain arm inside `_spec` captures the drafters' hidden states every step (their
  context): its cost against the plain loop is the `floor` run.
- `QWEN_DRAFT=auto` without a drafter is the plain loop: lookup rounds between plain steps
  (`QWEN_LOOKUP=1`) are not turned on by it (fp16 KV only, not gated on the 9B).

## Files

`src/worker/qwen_worker/mtp.py`, `sched_spec.py` (MTP / chooser hooks), `engine.py`,
`kernels.py` (`_setup_verify_qmv`, verify hooks), `src/worker/mrqmv.py`,
`tests/test_mtp.py`, `tests/test_mrqmv.py` (GPU), `scripts/perf/mtp/` (`split_mtp.py`,
`ab_mtp.py`, `round_prof.py`, `gate_auto.sh`, `gate_table.py`, `chooser_sim.py`), raw runs in
`raw/mtp_*`; `tests/test_kvq8.py` (q8 parity of MTP / hybrid rounds).

# Speculative decoding: DFlash, DFlash2, MTP, lookup

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- Speculation runs only for a lone text row (greedy; sampling is opt-in); it is demoted to a
  batch row when another request would be admitted next to it.
- DFlash (9B) and DFlash2 (27B) speed code and English up ×1.4–2; on Russian prose DFlash
  loses to plain decoding (×0.74–0.83) and the target's own MTP head is the better drafter
  (×1.2–1.3 everywhere, ×1.8 on code).
- `QWEN_DRAFT=auto` loads both and picks per round (`mtp.Chooser`): on the 27B oQ4e,
  greedy at 2K, code ×1.94, English ×1.33, Russian ×1.20, short prose ×1.09–1.13 of plain
  (T 0.6: ×1.78 / ×1.23 / ×1.20 / ×1.07–1.09); 0.91–1.01 of the best single setting per
  cell, on q8 KV 1.00 (Russian 8K, English 2K), the no-draft floor 0.98–0.99 of plain
  ([perf/2026-11-17/auto_gate.md](perf/2026-11-17/auto_gate.md)). One setting for a mixed
  load; a prose-only load is still 4–9 % faster on the head with one fixed draft.
- The MTP head's draft count is its own gate (`QWEN_MTP_K=0`): at least one draft a round,
  all `KMAX` where the measured keep rates pay for the matrix kernel's block (code), plain
  steps only after four rounds that kept nothing. A 2-row verify block costs ×1.17 of a
  step, 3–5 rows ×1.48–1.55 (the kernels' floor on this machine), which caps the head at
  about (1 + keep rate) / 1.5.
- Prompt lookup (`QWEN_LOOKUP=draft`) pays on turns that copy earlier text; default off.
- Sampling: each token's noise is keyed by (seed, absolute position) ([api.md](api.md)), so a
  seeded request's speculative text is the plain loop's whatever the arm and block size (CPU
  tests; on the GPU within the verify's exactness class). Before (2026-11-17), the verify drew a
  block's rows with one key and sampled text left plain's within a few tokens: trajectory noise
  of up to 27 % per request in T 0.6 speed gates.
- The verify attention is the one-pass kernel (`QWEN_VERIFY_ATT=2`, `QWEN_VATT_Q8=1`,
  [kernels.md](kernels.md)). DFlash is not bit-equal to plain greedy (near-ties) but is
  deterministic run to run.

## DFlash (9B)

- **DFlash speculative decoding** (`QWEN_DRAFT=weights/qwen35-9b-dflash`, the
  z-lab drafter for base Qwen3.5-9B, 2.5 GB bf16, quantized to 4-bit at load via
  `QWEN_DRAFT_BITS`): works on Ornith with ~60% token acceptance (mean 3–5
  accepted per round), output identical to plain greedy on the test prompts
  (the target verifies, so drafter precision only moves speed). Decode
  ×1.43–1.64 on the 5 coding tasks (33 → 47–53 tok/s; 2026-10-12, mlx-vlm 0.7.6 in
  the batching worker: median 48 tok/s vs 37 plain, the old stack had fallen to 32 —
  i.e. slower than plain), 63.7 tok/s end-to-end on
  a code answer (67% acceptance), ~120 tok/s on trivially predictable text;
  holds at 8K context. Tiny answers (<10 tokens) come out slower — first rounds
  have no context. Block size: with `QWEN_VERIFY_MRQMM=1` (default) and `QWEN_DRAFT_BLOCK` 0 a fixed
  block 8; else adaptive ≥ fixed 4/8/16/32. The
  ceiling is an MLX kernel property: 8-bit qmm with 4–16 rows costs ~2.3× a
  1-row qmm on M1 (M=32 drops to 1.7×), so a 16-token verify forward ≈ 70 ms vs
  30 ms decode; the 4-bit drafter block ≈ 15 ms (lm_head over 15 rows alone
  9 ms). Greedy-only (temperature 0, presence_penalty 0), text-only (a
  thinking budget is honoured inside the rounds since 2026-11-11). Quantized KV since 2026-11-15 (before: fp16 only, the
  drafter was ignored): `_spec` quantizes like the plain loop (after every slab, step and committed round) and cuts a
  block at `quantized_kv_start` while the cache is fp16, so every position has the KV format of the plain loop (CPU:
  q8 DFlash == plain greedy); its verify attended quantized KV one query per call (`QWEN_VERIFY_ATT` is fp16-only) until `QWEN_VATT_Q8`
  (2026-11-16, default on: the one-pass kernel over q8 KV, `docs/perf/2026-11-16/vatt_q8.md`) (`docs/perf/2026-11-15/design/kv_q8.md`).
  mlx-vlm 0.7 verifies through exact multi-row qmv kernels (rows bit-identical to a
  1-token decode, so DFlash output == plain greedy, 15/15 corpus); they spill
  registers from ~10 rows, so `QWEN_VERIFY_EXACT_MAX_T=8` keeps them for blocks
  ≤ 8 and uses `quantized_matmul` above. Median tok/s on the 5 tasks (2 rounds):
  cap 0 = 35, 6 = 50, **8 = 46–52**, 10/12 = 51–54 but one task collapses to 27,
  16 = 39 (that task 23). Caps 6–16 all 15/15 identical to plain greedy; 0 → 14/15.

## DFlash2 on the 27B

- **DFlash2 on the 27B (2026-11-10/11, `docs/perf/2026-11-10/dflash2_27b.md`)**: `QWEN_DRAFT=weights/qwen38-27b-dflash2`
  (z-lab/Qwen3.8-27B-DFlash2, 5 layers, block 8, target layers 5/19/33/47/61) + `QWEN_KV_BITS=0` + `QWEN_MRQMM5=1` +
  wired-limit dance. Decode, in-run A/B, greedy: code without thinking 13.3 → 23.4 tok/s (×1.77, 3.9 tokens / 164 ms
  round), with thinking 13.4 → 19.0–19.3 (×1.41–1.44, 3.2 tokens / round at block 8; blocks 4 / 5 / 8 equal, adaptive
  worse: the verify cost is flat over 3..8 rows). Without the 5-bit kernel ×0.81–1.10 (a block-8 verify through
  `quantized_matmul` = 4.4 steps; mlx-vlm's exact verify kernels on 5 bits: 1.6 s at block 8). The drafter must run fp32
  activations (`QWEN_DRAFT_FP32`, auto: fp16 overflows → 0 % accepted); its block is 36 ms at 4 bits. Not bit-equal to
  plain greedy with the kernel (15/16 answers without thinking, 12/16 with; 4 of the 5 divergences at a top-2 logit gap
  ≤ 0.016, one at 0.25). mlx-vlm's default runtime block for DFlash2 is 5 (adaptive 3..5); `_setup_verify_mrqmm` fixes 8.
  A thinking budget no longer turns DFlash off (`_spec`: the block is cut where the budget forces the next token,
  `\n</think>` go through plain steps; 27B 8/8 == plain with the budget). Prompt cache + drafter (2026-11-11, report §7):
  hits work, output == cold (30/30 pairs incl. hits through a reply). The entry keeps the drafter's context
  (`QWEN_DRAFT_TAIL` / `QWEN_DRAFT_TAIL_END`: hidden states of the last positions before each state), so an exact repeat runs the cold
  run's rounds (before: 1 token of context, 5–20 % fewer drafts accepted) and so does the next turn; DFlash rows now
  store reply KV (also demoted ones; `_spec_reply`: `row.fed` = generated tokens, the cache holds the first
  pos − len(prompt); a stop token among accepted drafts with more after it → nothing stored) and a cancelled prefill's
  prefix. Next turn without thinking: TTFT 0.75–0.9 s vs 31–36 s cold. `_spec`'s prefill window bug fixed (the oldest
  chunk was dropped while all but the NEWEST covered the window: slabs 2048 + 2048 + 40 left the drafter 40 positions).
  States stored by plain rows carry no context. Peak 23.2 GB (26.3–26.7 GB = 24.5–24.9 GiB with a 3 GB cache of four entries, wired limit
  27 GiB). **Long context (2026-11-11, report §8)**: with the copies an entry is kept only to ~40K tokens (rows + cache
  share KV budget 3.24 GiB + prefill reserve 3.4 GiB, fp16 KV is 64 KiB/token and held twice: row + entry) — longer
  prompts were "not kept", every turn cold. `QWEN_PROMPT_CACHE_MOVE=1` (single stream: one KV copy, see Prompt cache)
  + `QWEN_PROMPT_CACHE_CKPTS=4 QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216
  QWEN_PROMPT_CACHE_GB=6 QWEN_MAX_CONTEXT=65536`: a 60.1K-token conversation, cold prefill 521 s, next turns TTFT
  1.4–2.1 s (thinking: resume at the last message start), decode 11.5–12.7 tok/s with thinking, peak 26.4 GiB
  (prefill) / 25.6–26.0 (decode) under the 27 GiB limit. Three things had to be fixed for it: (1) `_own_states`: after
  a partly accepted verify block mlx-vlm leaves each GDN layer's state as a slice of the block's per-token states,
  pinning all of them between rounds (7 × 147 MiB at block 8; decode transient above the resting row 2.05 → 1.27 GiB);
  (2) `QWEN_VERIFY_ATT`: mlx-vlm's verify attention is one query per call, i.e. the whole KV read per block token
  (round 570 ms at 60K keys vs 159 at 8K — slower than plain decode); MLX's vector attention kernel (queries × GQA
  factor ≤ 32 → 5 per call on the 27B) is bit-equal and reads it once: 296 ms; (3) a reference cycle (`_Own` ↔ its
  store closure ↔ the row) kept each finished row's states / KV until Python's cyclic GC (+0.4 GiB a turn with
  thinking, +1.3 without): `_Own.close`, and every request of `tests/test_pcache_move.py` checks that its row dies
  with its last reference. Above the wired limit MLX crawls before it fails (0.6 tok/s, then Metal OOM): the first
  60K runs. Without thinking at 60K: DFlash 14.3–16.0 tok/s vs plain 9.6 (×1.5–1.67), hits through the reply TTFT ~1 s;
  with thinking at 24K 17.0 vs plain 14.1. Full 64K and plain-with-thinking at 60K not measured.
  **Drafter (2026-11-16, 4-bit target, `docs/perf/2026-11-16/drafter.md`)**: the fp32 drafter's linears on
  `mrqmm_wide` (`QWEN_DRAFT_MRQMM=1`, default): block 31.3 → 17.4 ms, DFlash ×1.117 at 2K. `QWEN_DRAFT_BITS` 4 / 5 / 8:
  acceptance 3.24 / 3.21 / 3.25 of 7 (240 probes) at +1.5–2 ms per block for 5 / 8 bits, e2e 8 bits ×0.99 of 4: stays 4.
  Short Russian prose: 0.71 drafts of 7 accepted at any width, DFlash 11.1 vs plain 14.8 tok/s even with the fast
  drafter (break-even on the 27B ≈ 1 accepted draft per round): wants an acceptance-based fallback to plain.

## Dead ends (items (3)–(4) of the 2026-09-18 list; the rest is in [prefill.md](prefill.md#dead-ends))

(dequantization-bound) — a custom verify kernel has ≤1.3× headroom. (4) DFlash
with fallback to plain decoding on low acceptance: loses (8-round probe ≈0.5 s,
drafter restarts cold); spec for sampling requests is exact but only ×1.1–1.2
on code / ×0.8 on prose → opt-in `QWEN_DRAFT_SAMPLING=1`.

## Gotchas

- DFlash greedy and the plain loop are byte-identical only for verify blocks
  ≤ `QWEN_VERIFY_EXACT_MAX_T` (8): longer blocks go through `quantized_matmul`,
  which rounds differently from the 1-row decode, so a near-tie can flip there
  (the adaptive block reaches 16; 15/15 corpus identical at cap 8 nonetheless).
- DFlash's verify block was adaptive on acceptance stats; since mlx-vlm 0.7.6
  `draft.reset` clears them per request, so a request's blocks depend on itself only
  (checked by code, 2026-10-24). Since 2026-10-24 the default is `QWEN_VERIFY_MRQMM=1`
  with a fixed block 8 (verify rows through mrqmm as (T,1,K)): DFlash ≠ plain greedy
  bitwise (batched-decode class; 15/16 identical on the corpus, the old adaptive path
  13/16) but run-to-run deterministic (T ≤ 8 → always mrqmm's MB=1). `head` = only the
  drafter's lm_head through mrqmm (verify exact, +4.7 %).

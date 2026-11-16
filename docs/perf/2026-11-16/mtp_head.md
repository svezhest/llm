# The 27B's MTP head: 4-bit vs 8-bit vs fp16, acceptance per depth, chaining, cost

Status: **probe and tables written and checked on the CPU (tiny random model); no GPU run yet.**
Every number cell below is empty until `accept_probe.py` has run on the 27B. Labels as in
`mtp_draft.md`: [M] measured, [D] derived, [E] estimated.

Target `weights/qwen38-27b-mlx-oq4e`. Heads (same base weights; norms and `fc` identical):

| name | dir | quantization | size |
|---|---|---|---|
| q4 | `weights/qwen38-27b-mtp` | oQ4e's own: 4-bit g64, k / v 8-bit, o 5-bit, `fc` unquantized | 323 MB |
| q8 | `weights/qwen38-27b-mtp-q8` | from the bf16 checkpoint, uniform 8-bit g64 (`fc` too) | 451 MB |
| fp16 | `weights/qwen38-27b-mtp-bf16` | unquantized bf16, cast to fp16 at load (`QWEN_FP16`) | 849 MB |

## Method

`scripts/perf/mtp/accept_probe.py`: one process, one target load, no scheduler. Per text the
target runs twice (as `_spec` runs it: 2048-token prefill slabs, then 1-token steps with
`return_hidden`): its greedy continuation and one sampled continuation (temperature 0.6 /
top_p 0.95 / top_k 20, the worker's `_Sampler`, seed 1234), 320 tokens each. Kept per
position: the final hidden state and the sampler's filtered distribution (top 20). The heads
then replay the hidden states through the worker's own `MTPDraft` (`push` / `emit` / `flush`
/ `draft` / `round_end`), all three on the same positions.

Texts: bench27 `code-2k-0`, `en-2k-0`, `ru-2k-0` (as `ab_mtp.py`: the engine's render,
thinking off) + `en_prose`, `ru_prose` (`scripts/perf/dflash2/prompts2k.py` `SHORT`);
classes code / en / ru, pooled by counts. `--extra` adds bench27 ids.

- **table**: at every generated position the chain of 4 drafts from a head cache of real
  entries only (after each probe the chained entries are trimmed and the position is fed its
  real hidden state: `QWEN_MTP_REFEED=1`'s cache at any round start). p_i = P(draft i == the
  trajectory's token | drafts < i are), n = positions in the condition.
- **sampling**: the worker keeps a draft iff the target's own draw equals it, so draft i's
  expected acceptance is the target's filtered probability of it, read from the kept
  distribution (no draws). s_i = its mean over the positions whose drafts < i are the
  trajectory's tokens (past a draft that is not, the target's logits are off the trajectory).
  At the greedy positions that condition is "accepted and the argmax"; along the sampled
  trajectory it is "accepted", and s_i is an unbiased estimate of the sampling run's
  conditional acceptance. Tokens per round under sampling: 1 + the filtered probabilities of
  the drafts up to and with the first that is not the trajectory's, at the round starts of
  the simulated run.
- **oracle**: the head with the target's real hidden state at every depth = the table's
  depth-1 draft i − 1 positions later (no extra forward). Not usable at run time: the upper
  bound of any stand-in for the chained hidden.
- **rounds**: the rounds at fixed k = 2, 3, 4 along the trajectory with the chained entries
  of accepted drafts kept (the default, live); recomputed (`QWEN_MTP_REFEED=1`): from the
  table for k = 1..4, and live at k = 4 as a check of that derivation.
- **fp32** (`--wide`): the greedy table again with fp32 activations (scales, norms,
  unquantized weights and inputs fp32); + a count of non-finite head outputs in fp16.
- **costs**: per head `draft(k)` (a 1-row flush + k − 1 chained steps + k lm_head reads), an
  n-row flush, a 1-row forward, the lm_head read; interleaved with a plain decode step at the
  2K context, each stage synced, medians of 27.

`scripts/perf/mtp/accept_tables.py` prints every table below from the JSON and the speed
projection: speed-up over plain = tokens per round / (verify price(k + 1) + draft cost /
decode step), verify prices in decode steps: "today" T = 2 ×1.29, 3 ×1.6, 4 ×1.7, 5 ×1.8;
"target" T = 2 ×1.12, 3 ×1.20, 4–5 ×1.35.

Commands:

    cd src/worker && QWEN_MODEL=$WEIGHTS/qwen38-27b-mlx-oq4e QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 \
      QWEN_MRQMM5=1 QWEN_MRQMM6=1 QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 \
      QWEN_PRESENCE_PENALTY=0 QWEN_DRAFT=$WEIGHTS/qwen38-27b-mtp QWEN_MTP_K=0 QWEN_VERIFY_QMV=3 \
      HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python ../../scripts/perf/mtp/accept_probe.py \
      --heads q4=$WEIGHTS/qwen38-27b-mtp,q8=$WEIGHTS/qwen38-27b-mtp-q8,fp16=$WEIGHTS/qwen38-27b-mtp-bf16 \
      --wide code-2k-0,en-2k-0,ru-2k-0 --traj $PERF_OUT/mtp_head/traj --json $PERF_OUT/mtp_head/accept.json
    python3 scripts/perf/mtp/accept_tables.py $PERF_OUT/mtp_head/accept.json \
      --summary docs/perf/2026-11-16/raw/mtp_head_summary.json

## 1. Greedy acceptance per depth (table; n = positions in the condition)

| class | head | p1 (n) | p2 (n) | p3 (n) | p4 (n) | E drafts of 4 |
|---|---|---|---|---|---|---|
| code | q4 | | | | | |
| code | q8 | | | | | |
| code | fp16 | | | | | |
| en | q4 | | | | | |
| en | q8 | | | | | |
| en | fp16 | | | | | |
| ru | q4 | | | | | |
| ru | q8 | | | | | |
| ru | fp16 | | | | | |

Heads on the same positions (all classes):

| pair | depth 1: A right B wrong / B right A wrong | mean accepted of 4, A − B (± se) |
|---|---|---|
| q4 vs q8 | | |
| q4 vs fp16 | | |
| q8 vs fp16 | | |

## 2. Sampling (T 0.6 / top_p 0.95 / top_k 20): expected acceptance per depth

| class | head | greedy positions s1..s4 | sampled trajectory s1..s4 (n) | sampled trajectory, match p1..p4 |
|---|---|---|---|---|
| code | q4 | | | |
| code | q8 | | | |
| code | fp16 | | | |
| en | q4 | | | |
| en | q8 | | | |
| en | fp16 | | | |
| ru | q4 | | | |
| ru | q8 | | | |
| ru | fp16 | | | |

## 3. Chained vs oracle vs refeed

| class | head | chained p2 / p3 / p4 | oracle p2 / p3 / p4 | cosine stand-in vs target hidden, depth 1..4 |
|---|---|---|---|---|
| code | q4 | | | |
| en | q4 | | | |
| ru | q4 | | | |
| code | best head | | | |
| en | best head | | | |
| ru | best head | | | |

Tokens per round, greedy, kept (default) vs refeed:

| class | head | kept k=2 | kept k=3 | kept k=4 | refeed k=2 | refeed k=3 | refeed k=4 | refeed k=4 live |
|---|---|---|---|---|---|---|---|---|
| code | | | | | | | | |
| en | | | | | | | | |
| ru | | | | | | | | |

If the oracle is far above the chain: candidates for a cheap stand-in, none built — (a) a
fixed linear map from the head's output hidden to the target's hidden state, fit offline by
ridge regression on (output at t − 1, target hidden at t) pairs (one 5120 × 5120 fp16 GEMV
per chained step); (b) the same low-rank; (c) stopping the chain on the head's own top-1
probability (the `gate` columns of §5: no better drafts, fewer wasted ones).

## 4. Cost and fp32

| head | forward 1 row | lm_head | draft k=1 | k=2 | k=3 | k=4 | flush n=2 | n=5 |
|---|---|---|---|---|---|---|---|---|
| q4 | | | | | | | | |
| q8 | | | | | | | | |
| fp16 | | | | | | | | |

Plain decode step at the same context: ___ ms.

| head | non-finite outputs in fp16 | chains == fp32 | drafts right fp16 vs fp32 |
|---|---|---|---|
| q4 | | | |
| q8 | | | |
| fp16 | | | |

## 5. Bottom line: tokens per round and speed-up over plain

Tokens per round at k (kept), greedy / sampling:

| class | head | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| code | q4 | | | | |
| code | q8 | | | | |
| code | fp16 | | | | |
| en | q4 | | | | |
| en | q8 | | | | |
| en | fp16 | | | | |
| ru | q4 | | | | |
| ru | q8 | | | | |
| ru | fp16 | | | | |

Speed-up over plain [D], best fixed k per cell (greedy / sampling):

| class | head | today's verify prices | target verify prices | gated (best tau), today / target |
|---|---|---|---|---|
| code | q4 | | | |
| code | q8 | | | |
| code | fp16 | | | |
| en | q4 | | | |
| en | q8 | | | |
| en | fp16 | | | |
| ru | q4 | | | |
| ru | q8 | | | |
| ru | fp16 | | | |

End to end (`ab_mtp.py --alt-head <dir> --variants plain,mtp0+q,mtp0+q+h`, ratio to plain of
the same pair, ≥ 2 pairs per prompt):

| variant | code-2k | en-2k | ru-2k |
|---|---|---|---|
| mtp0+q (q4) | | | |
| mtp0+q+h (best head) | | | |

## Verdict

(empty until the run)

## Not verified

- Nothing here has run on the 27B: the probe was checked on the CPU on the tiny random model
  of the test suite only.
- The speed projection ignores the round's host time and takes the verify prices as given.
- One sampled trajectory per text (one seed).

## Files

`scripts/perf/mtp/accept_probe.py`, `scripts/perf/mtp/accept_tables.py`, `ab_mtp.py`
(`--alt-head`, `+h`).

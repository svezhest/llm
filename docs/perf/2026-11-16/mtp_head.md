# The 27B's MTP head: 4-bit vs 8-bit vs fp16, acceptance per depth, chaining, cost

> Later (2026-11-17): the draft-count rule and the chooser described here were replaced
> after the GPU gate — see [../2026-11-17/auto_gate.md](../2026-11-17/auto_gate.md). The
> knobs `QWEN_MTP_REFEED` / `QWEN_MTP_COST` and the replay tools `chooser_sim.py` /
> `gate_sim.py` named below no longer exist; the numbers stand as measured on that day.

Status: **measured on the 27B on 2026-11-17 01:22** (results and verdict below); the three heads
are level within 2 %, q4 stays. Labels as in
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

## Results (2026-11-17 01:22, [M] unless marked)

`accept_probe.py` on the 27B, one process, the three heads on the same positions; 5 texts
(3 × 2K + 2 short), 320 generated tokens per trajectory, greedy and one sampled (T 0.6).
Heads as loaded: q4 431 MiB, q8 456 MiB, fp16 1620 MiB. The machine was warm: the plain decode step read 88.2 ms (59 ms on a quiet one), so the speed-up columns are
a ranking of the heads, not the worker's speed. Verify prices in the projections are the
values assumed that day ("today": T = 2 ×1.29, 3 ×1.6, 4 ×1.7, 5 ×1.8); the block prices
measured later are in [../2026-11-17/auto_gate.md](../2026-11-17/auto_gate.md).
Raw: `raw/mtp_head_tables.md`, `raw/mtp_head_summary.json`.

### texts

| text | class | trajectory | prompt | generated | target's mean filtered p of its token |
|---|---|---|---|---|---|
| code-2k-0 | code | greedy | 2047 | 321 | 0.951 |
| en-2k-0 | en | greedy | 2048 | 321 | 0.895 |
| ru-2k-0 | ru | greedy | 2047 | 321 | 0.891 |
| en_prose | en | greedy | 43 | 321 | 0.850 |
| ru_prose | ru | greedy | 52 | 321 | 0.842 |
| code-2k-0 | code | sampled | 2047 | 321 | 0.907 |
| en-2k-0 | en | sampled | 2048 | 321 | 0.856 |
| ru-2k-0 | ru | sampled | 2047 | 321 | 0.848 |
| en_prose | en | sampled | 43 | 321 | 0.808 |
| ru_prose | ru | sampled | 52 | 321 | 0.789 |

### per-depth conditional acceptance, greedy trajectories

`match` = draft i equals the trajectory's token; `q` = mean filtered probability of draft i (expected sample-match acceptance); both given drafts < i equal the trajectory's tokens; n = positions in the condition.

| class | head | match p1 (n) | match p2 (n) | match p3 (n) | match p4 (n) | q s1 | q s2 | q s3 | q s4 | E drafts match / q |
|---|---|---|---|---|---|---|---|---|---|---|
| code | q4 | 0.921 (317) | 0.873 (292) | 0.835 (255) | 0.840 (213) | 0.904 | 0.870 | 0.834 | 0.835 | 2.962 / 2.936 |
| code | q4 oracle | 0.921 (317) | 0.928 (292) | 0.930 (271) | 0.937 (252) | 0.904 | 0.912 | 0.912 | 0.917 | 3.315 / 3.252 |
| code | q8 | 0.924 (317) | 0.881 (293) | 0.841 (258) | 0.843 (217) | 0.905 | 0.875 | 0.842 | 0.837 | 3.000 / 2.972 |
| code | q8 oracle | 0.924 (317) | 0.932 (293) | 0.930 (273) | 0.933 (254) | 0.905 | 0.913 | 0.912 | 0.916 | 3.334 / 3.268 |
| code | fp16 | 0.924 (317) | 0.877 (293) | 0.840 (257) | 0.847 (216) | 0.905 | 0.871 | 0.841 | 0.840 | 2.994 / 2.965 |
| code | fp16 oracle | 0.924 (317) | 0.932 (293) | 0.930 (273) | 0.933 (254) | 0.905 | 0.912 | 0.912 | 0.916 | 3.334 / 3.267 |
| en | q4 | 0.767 (634) | 0.638 (486) | 0.548 (310) | 0.459 (170) | 0.737 | 0.614 | 0.524 | 0.454 | 1.647 / 1.585 |
| en | q4 oracle | 0.767 (634) | 0.782 (486) | 0.782 (380) | 0.788 (297) | 0.737 | 0.748 | 0.749 | 0.759 | 2.203 / 2.115 |
| en | q8 | 0.767 (634) | 0.640 (486) | 0.566 (311) | 0.483 (176) | 0.736 | 0.615 | 0.536 | 0.472 | 1.669 / 1.601 |
| en | q8 oracle | 0.767 (634) | 0.780 (486) | 0.789 (379) | 0.786 (299) | 0.736 | 0.747 | 0.755 | 0.755 | 2.207 / 2.116 |
| en | fp16 | 0.763 (634) | 0.632 (484) | 0.562 (306) | 0.483 (172) | 0.733 | 0.610 | 0.531 | 0.478 | 1.648 / 1.584 |
| en | fp16 oracle | 0.763 (634) | 0.779 (484) | 0.788 (377) | 0.785 (297) | 0.733 | 0.745 | 0.753 | 0.755 | 2.194 / 2.104 |
| ru | q4 | 0.716 (634) | 0.551 (454) | 0.444 (250) | 0.396 (111) | 0.689 | 0.542 | 0.431 | 0.371 | 1.355 / 1.312 |
| ru | q4 oracle | 0.716 (634) | 0.718 (454) | 0.730 (326) | 0.706 (238) | 0.689 | 0.693 | 0.699 | 0.680 | 1.871 / 1.800 |
| ru | q8 | 0.732 (634) | 0.552 (464) | 0.457 (256) | 0.368 (117) | 0.701 | 0.540 | 0.443 | 0.349 | 1.388 / 1.339 |
| ru | q8 oracle | 0.732 (634) | 0.733 (464) | 0.750 (340) | 0.725 (255) | 0.701 | 0.705 | 0.715 | 0.699 | 1.962 / 1.881 |
| ru | fp16 | 0.729 (634) | 0.554 (462) | 0.457 (256) | 0.376 (117) | 0.698 | 0.544 | 0.445 | 0.357 | 1.386 / 1.340 |
| ru | fp16 oracle | 0.729 (634) | 0.729 (462) | 0.745 (337) | 0.717 (251) | 0.698 | 0.702 | 0.710 | 0.690 | 1.940 / 1.860 |
| all | q4 | 0.777 (1585) | 0.662 (1232) | 0.606 (815) | 0.609 (494) | 0.751 | 0.648 | 0.592 | 0.600 | 1.793 / 1.746 |
| all | q4 oracle | 0.777 (1585) | 0.793 (1232) | 0.806 (977) | 0.811 (787) | 0.751 | 0.767 | 0.777 | 0.786 | 2.293 / 2.216 |
| all | q8 | 0.784 (1585) | 0.664 (1243) | 0.618 (825) | 0.610 (510) | 0.756 | 0.648 | 0.603 | 0.599 | 1.823 / 1.770 |
| all | q8 oracle | 0.784 (1585) | 0.798 (1243) | 0.815 (992) | 0.813 (808) | 0.756 | 0.770 | 0.785 | 0.788 | 2.334 / 2.252 |
| all | fp16 | 0.782 (1585) | 0.661 (1239) | 0.617 (819) | 0.614 (505) | 0.754 | 0.647 | 0.601 | 0.605 | 1.813 / 1.763 |
| all | fp16 oracle | 0.782 (1585) | 0.797 (1239) | 0.813 (987) | 0.810 (802) | 0.754 | 0.768 | 0.782 | 0.786 | 2.321 / 2.239 |

### per-depth conditional acceptance, sampled trajectories

`match` = draft i equals the trajectory's token; `q` = mean filtered probability of draft i (expected sample-match acceptance); both given drafts < i equal the trajectory's tokens; n = positions in the condition.

| class | head | match p1 (n) | match p2 (n) | match p3 (n) | match p4 (n) | q s1 | q s2 | q s3 | q s4 | E drafts match / q |
|---|---|---|---|---|---|---|---|---|---|---|
| code | q4 | 0.868 (317) | 0.844 (275) | 0.823 (232) | 0.832 (191) | 0.874 | 0.854 | 0.823 | 0.816 | 2.703 / 2.709 |
| code | q4 oracle | 0.868 (317) | 0.887 (275) | 0.893 (244) | 0.908 (218) | 0.874 | 0.893 | 0.892 | 0.906 | 2.950 / 2.959 |
| code | q8 | 0.877 (317) | 0.849 (278) | 0.814 (236) | 0.812 (192) | 0.879 | 0.854 | 0.816 | 0.811 | 2.719 / 2.726 |
| code | q8 oracle | 0.877 (317) | 0.899 (278) | 0.908 (250) | 0.921 (227) | 0.879 | 0.898 | 0.900 | 0.911 | 3.041 / 3.028 |
| code | fp16 | 0.871 (317) | 0.851 (276) | 0.813 (235) | 0.822 (191) | 0.874 | 0.858 | 0.815 | 0.815 | 2.710 / 2.716 |
| code | fp16 oracle | 0.871 (317) | 0.891 (276) | 0.898 (246) | 0.914 (221) | 0.874 | 0.892 | 0.892 | 0.905 | 2.981 / 2.973 |
| en | q4 | 0.708 (634) | 0.601 (449) | 0.541 (270) | 0.418 (146) | 0.723 | 0.603 | 0.529 | 0.411 | 1.461 / 1.469 |
| en | q4 oracle | 0.708 (634) | 0.717 (449) | 0.720 (322) | 0.698 (232) | 0.723 | 0.736 | 0.731 | 0.711 | 1.838 / 1.875 |
| en | q8 | 0.707 (634) | 0.616 (448) | 0.529 (276) | 0.432 (146) | 0.721 | 0.618 | 0.530 | 0.436 | 1.472 / 1.488 |
| en | q8 oracle | 0.707 (634) | 0.717 (448) | 0.713 (321) | 0.699 (229) | 0.721 | 0.729 | 0.720 | 0.710 | 1.826 / 1.857 |
| en | fp16 | 0.707 (634) | 0.609 (448) | 0.531 (273) | 0.434 (145) | 0.720 | 0.612 | 0.528 | 0.438 | 1.465 / 1.480 |
| en | fp16 oracle | 0.707 (634) | 0.717 (448) | 0.713 (321) | 0.699 (229) | 0.720 | 0.729 | 0.720 | 0.710 | 1.826 / 1.857 |
| ru | q4 | 0.702 (634) | 0.539 (445) | 0.475 (240) | 0.395 (114) | 0.694 | 0.542 | 0.469 | 0.395 | 1.331 / 1.323 |
| ru | q4 oracle | 0.702 (634) | 0.728 (445) | 0.722 (324) | 0.726 (234) | 0.694 | 0.723 | 0.712 | 0.711 | 1.850 / 1.828 |
| ru | q8 | 0.699 (634) | 0.542 (443) | 0.471 (240) | 0.416 (113) | 0.695 | 0.552 | 0.467 | 0.421 | 1.330 / 1.332 |
| ru | q8 oracle | 0.699 (634) | 0.727 (443) | 0.724 (322) | 0.725 (233) | 0.695 | 0.728 | 0.723 | 0.722 | 1.841 / 1.836 |
| ru | fp16 | 0.703 (634) | 0.543 (446) | 0.483 (242) | 0.419 (117) | 0.697 | 0.551 | 0.475 | 0.417 | 1.347 / 1.343 |
| ru | fp16 oracle | 0.703 (634) | 0.729 (446) | 0.726 (325) | 0.729 (236) | 0.697 | 0.728 | 0.726 | 0.725 | 1.860 / 1.851 |
| all | q4 | 0.738 (1585) | 0.635 (1169) | 0.608 (742) | 0.588 (451) | 0.742 | 0.639 | 0.602 | 0.578 | 1.657 / 1.659 |
| all | q4 oracle | 0.738 (1585) | 0.761 (1169) | 0.769 (890) | 0.775 (684) | 0.742 | 0.768 | 0.768 | 0.773 | 2.065 / 2.073 |
| all | q8 | 0.738 (1585) | 0.643 (1169) | 0.600 (752) | 0.590 (451) | 0.742 | 0.649 | 0.599 | 0.592 | 1.664 / 1.673 |
| all | q8 oracle | 0.738 (1585) | 0.764 (1169) | 0.772 (893) | 0.781 (689) | 0.742 | 0.769 | 0.771 | 0.780 | 2.075 / 2.083 |
| all | fp16 | 0.738 (1585) | 0.641 (1170) | 0.604 (750) | 0.594 (453) | 0.742 | 0.647 | 0.601 | 0.591 | 1.667 / 1.672 |
| all | fp16 oracle | 0.738 (1585) | 0.762 (1170) | 0.769 (892) | 0.778 (686) | 0.742 | 0.767 | 0.769 | 0.778 | 2.071 / 2.077 |

### heads on the same positions (greedy trajectories, all classes)

| pair | depth-1: A right B wrong / B right A wrong | mean accepted of 4 drafts A − B (paired, ± se) |
|---|---|---|
| q4 vs q8 | 10 / 21 | -0.0297 ± 0.0121 (n 1585) |
| q4 vs fp16 | 11 / 18 | -0.0196 ± 0.0114 (n 1585) |
| q8 vs fp16 | 5 / 1 | +0.0101 ± 0.0066 (n 1585) |

### the chained stand-in: cosine of the head's output hidden with the target's hidden state it replaces (drafts so far right)

| class | head | depth 1 | depth 2 | depth 3 | depth 4 |
|---|---|---|---|---|---|
| code | q4 | 0.579 | 0.496 | 0.446 | 0.414 |
| code | q8 | 0.580 | 0.497 | 0.447 | 0.417 |
| code | fp16 | 0.580 | 0.498 | 0.448 | 0.416 |
| en | q4 | 0.677 | 0.564 | 0.508 | 0.486 |
| en | q8 | 0.678 | 0.565 | 0.513 | 0.488 |
| en | fp16 | 0.678 | 0.565 | 0.514 | 0.488 |
| ru | q4 | 0.614 | 0.512 | 0.486 | 0.485 |
| ru | q8 | 0.616 | 0.515 | 0.493 | 0.490 |
| ru | fp16 | 0.616 | 0.515 | 0.493 | 0.488 |

### fp32 activations (the table again with every activation fp32)

- q4: non-finite head outputs in fp16: 0; chains identical to fp32 at 956/960 positions; drafts right fp16 [793, 577, 388, 264] vs fp32 [795, 580, 391, 265] (n [951, 793, 577, 388] / [951, 795, 580, 391]); fp32 non-finite 0
- q8: non-finite head outputs in fp16: 0; chains identical to fp32 at 955/960 positions; drafts right fp16 [803, 587, 404, 273] vs fp32 [803, 587, 404, 272] (n [951, 803, 587, 404] / [951, 803, 587, 404]); fp32 non-finite 0
- fp16: non-finite head outputs in fp16: 0; chains identical to fp32 at 956/960 positions; drafts right fp16 [800, 581, 398, 272] vs fp32 [800, 581, 397, 272] (n [951, 800, 581, 398] / [951, 800, 581, 397]); fp32 non-finite 0

### costs (synced stages, medians of 27, context 2048; plain decode step 88.2 ms)

| head | forward 1 row | lm_head | draft k=1 | draft k=2 | draft k=3 | draft k=4 | flush n=1 | flush n=2 | flush n=3 | flush n=4 | flush n=5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4 | 1.85 | 5.36 | 7.13 | 13.76 | 19.97 | 26.55 | 6.83 | 8.36 | 9.29 | 10.36 | 11.28 |
| q8 | 2.29 | 4.96 | 7.25 | 13.66 | 21.21 | 27.79 | 7.17 | 8.77 | 10.28 | 11.85 | 13.45 |
| fp16 | 3.10 | 5.15 | 8.06 | 15.81 | 23.84 | 31.69 | 8.04 | 16.87 | 16.86 | 17.06 | 17.30 |

### tokens per round at fixed k, greedy trajectories (kept = the default's chained entries, live; refeed = recomputed, from the table; (rounds))

| class | head | kept k=1 | kept k=2 | kept k=3 | kept k=4 | refeed k=1 | refeed k=2 | refeed k=3 | refeed k=4 | refeed live |
|---|---|---|---|---|---|---|---|---|---|---|
| code | q4 | 1.922 (166) | 2.645 (121) | 3.312 (96) | 3.674 (86) | 1.922 (166) | 2.689 (119) | 3.302 (96) | 3.901 (81) | k=4: 3.901 |
| code | q8 | 1.933 (165) | 2.689 (119) | 3.278 (97) | 3.591 (88) | 1.933 (165) | 2.712 (118) | 3.268 (97) | 3.831 (83) | k=4: 3.831 |
| code | fp16 | 1.933 (165) | 2.689 (119) | 3.278 (97) | 3.591 (88) | 1.933 (165) | 2.712 (118) | 3.268 (97) | 3.831 (83) | k=4: 3.878 |
| en | q4 | 1.753 (365) | 2.251 (283) | 2.413 (264) | 2.590 (244) | 1.753 (365) | 2.227 (286) | 2.488 (256) | 2.633 (240) | k=4: 2.633 |
| en | q8 | 1.773 (361) | 2.204 (289) | 2.431 (262) | 2.569 (246) | 1.773 (361) | 2.235 (285) | 2.508 (254) | 2.689 (235) | k=4: 2.689 |
| en | fp16 | 1.763 (363) | 2.197 (290) | 2.422 (263) | 2.601 (243) | 1.763 (363) | 2.227 (286) | 2.498 (255) | 2.667 (237) | k=4: 2.667 |
| ru | q4 | 1.755 (364) | 2.127 (300) | 2.259 (282) | 2.310 (274) | 1.755 (364) | 2.120 (301) | 2.235 (285) | 2.292 (277) | k=4: 2.292 |
| ru | q8 | 1.770 (361) | 2.113 (302) | 2.275 (280) | 2.285 (277) | 1.770 (361) | 2.113 (302) | 2.267 (281) | 2.309 (275) | k=4: 2.309 |
| ru | fp16 | 1.770 (361) | 2.106 (303) | 2.251 (283) | 2.310 (274) | 1.770 (361) | 2.113 (302) | 2.259 (282) | 2.301 (276) | k=4: 2.301 |
| all | q4 | 1.785 (895) | 2.266 (704) | 2.480 (642) | 2.618 (604) | 1.785 (895) | 2.259 (706) | 2.498 (637) | 2.647 (598) | k=4: 2.647 |
| all | q8 | 1.802 (887) | 2.246 (710) | 2.491 (639) | 2.588 (611) | 1.802 (887) | 2.262 (705) | 2.517 (632) | 2.673 (593) | k=4: 2.673 |
| all | fp16 | 1.798 (889) | 2.240 (712) | 2.476 (643) | 2.613 (605) | 1.798 (889) | 2.259 (706) | 2.509 (634) | 2.659 (596) | k=4: 2.664 |

### tokens per round at fixed k, sampled trajectories (kept = the default's chained entries, live; refeed = recomputed, from the table; (rounds); expected of a sample-match verify [the one realization])

| class | head | kept k=1 | kept k=2 | kept k=3 | kept k=4 | refeed k=1 | refeed k=2 | refeed k=3 | refeed k=4 | refeed live |
|---|---|---|---|---|---|---|---|---|---|---|
| code | q4 | 1.862 [1.850] (173) | 2.574 [2.544] (125) | 3.045 [3.038] (105) | 3.294 [3.278] (97) | 1.862 [1.850] (173) | 2.594 [2.565] (124) | 3.025 [3.009] (106) | 3.459 [3.478] (92) | k=4: 3.478 |
| code | q8 | 1.857 [1.850] (173) | 2.553 [2.524] (126) | 3.012 [3.000] (106) | 3.320 [3.312] (96) | 1.857 [1.850] (173) | 2.590 [2.565] (124) | 3.013 [3.009] (106) | 3.395 [3.404] (94) | k=4: 3.404 |
| code | fp16 | 1.862 [1.850] (173) | 2.553 [2.524] (126) | 2.983 [2.972] (107) | 3.375 [3.347] (95) | 1.862 [1.850] (173) | 2.594 [2.565] (124) | 3.046 [3.038] (105) | 3.420 [3.441] (93) | k=4: 3.441 |
| en | q4 | 1.719 [1.711] (374) | 2.111 [2.095] (304) | 2.312 [2.291] (278) | 2.357 [2.324] (272) | 1.719 [1.711] (374) | 2.106 [2.089] (305) | 2.326 [2.303] (277) | 2.405 [2.376] (266) | k=4: 2.376 |
| en | q8 | 1.715 [1.699] (376) | 2.116 [2.095] (305) | 2.299 [2.271] (280) | 2.337 [2.315] (273) | 1.715 [1.699] (376) | 2.131 [2.116] (302) | 2.341 [2.325] (274) | 2.401 [2.376] (266) | k=4: 2.376 |
| en | fp16 | 1.714 [1.699] (376) | 2.111 [2.081] (307) | 2.308 [2.288] (278) | 2.344 [2.315] (273) | 1.714 [1.699] (376) | 2.130 [2.116] (302) | 2.341 [2.325] (274) | 2.410 [2.385] (265) | k=4: 2.385 |
| ru | q4 | 1.660 [1.675] (381) | 1.998 [2.006] (318) | 2.152 [2.167] (293) | 2.218 [2.220] (286) | 1.660 [1.675] (381) | 2.025 [2.042] (312) | 2.177 [2.201] (288) | 2.252 [2.261] (280) | k=4: 2.261 |
| ru | q8 | 1.661 [1.666] (383) | 1.997 [1.997] (319) | 2.168 [2.186] (290) | 2.207 [2.221] (285) | 1.661 [1.666] (383) | 2.018 [2.022] (315) | 2.190 [2.217] (286) | 2.269 [2.285] (277) | k=4: 2.285 |
| ru | fp16 | 1.666 [1.675] (381) | 2.002 [2.003] (318) | 2.156 [2.171] (292) | 2.207 [2.221] (285) | 1.666 [1.675] (381) | 2.031 [2.042] (312) | 2.211 [2.240] (283) | 2.285 [2.302] (275) | k=4: 2.293 |
| all | q4 | 1.722 [1.722] (928) | 2.141 [2.133] (747) | 2.357 [2.354] (676) | 2.435 [2.420] (655) | 1.722 [1.722] (928) | 2.154 [2.148] (741) | 2.373 [2.371] (671) | 2.490 [2.484] (638) | k=4: 2.484 |
| all | q8 | 1.719 [1.714] (932) | 2.138 [2.125] (750) | 2.354 [2.349] (676) | 2.424 [2.420] (654) | 1.719 [1.714] (932) | 2.160 [2.151] (741) | 2.383 [2.387] (666) | 2.490 [2.488] (637) | k=4: 2.488 |
| all | fp16 | 1.722 [1.717] (930) | 2.139 [2.123] (751) | 2.349 [2.346] (677) | 2.434 [2.424] (653) | 1.722 [1.717] (930) | 2.166 [2.160] (738) | 2.397 [2.402] (662) | 2.504 [2.504] (633) | k=4: 2.500 |

### speed-up over plain decode, greedy, verify prices "today" (T=2 x1.29, T=3 x1.6, T=4 x1.7, T=5 x1.8; draft cost / decode step 88.2 ms)

| class | head | k=1 | k=2 | k=3 | k=4 | best | refeed k=1 | refeed k=2 | refeed k=3 | refeed k=4 | gate 0.3 | gate 0.5 | gate 0.7 | gate 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code | q4 | 1.386 | 1.495 | 1.709 | 1.742 | **1.742** (k=4) | 1.389 | 1.515 | 1.689 | 1.827 | 1.857 | 1.874 | 1.853 | 1.752 |
| code | q8 | 1.392 | 1.520 | 1.680 | 1.691 | **1.691** (k=4) | 1.393 | 1.516 | 1.653 | 1.770 | 1.837 | 1.879 | 1.838 | 1.731 |
| code | fp16 | 1.311 | 1.447 | 1.614 | 1.628 | **1.628** (k=4) | 1.311 | 1.449 | 1.587 | 1.704 | 1.800 | 1.832 | 1.812 | 1.703 |
| en | q4 | 1.267 | 1.276 | 1.250 | 1.232 | **1.276** (k=2) | 1.270 | 1.259 | 1.279 | 1.242 | 1.297 | 1.360 | 1.348 | 1.236 |
| en | q8 | 1.279 | 1.250 | 1.250 | 1.213 | **1.279** (k=1) | 1.280 | 1.255 | 1.277 | 1.254 | 1.316 | 1.327 | 1.357 | 1.237 |
| en | fp16 | 1.209 | 1.203 | 1.215 | 1.198 | **1.215** (k=3) | 1.210 | 1.200 | 1.221 | 1.192 | 1.287 | 1.300 | 1.334 | 1.220 |
| ru | q4 | 1.268 | 1.206 | 1.171 | 1.099 | **1.268** (k=1) | 1.272 | 1.199 | 1.151 | 1.083 | 1.251 | 1.292 | 1.289 | 1.230 |
| ru | q8 | 1.277 | 1.199 | 1.171 | 1.080 | **1.277** (k=1) | 1.278 | 1.188 | 1.157 | 1.080 | 1.229 | 1.281 | 1.288 | 1.221 |
| ru | fp16 | 1.214 | 1.159 | 1.134 | 1.067 | **1.214** (k=1) | 1.214 | 1.140 | 1.106 | 1.032 | 1.204 | 1.262 | 1.261 | 1.208 |
| all | q4 | 1.290 | 1.284 | 1.284 | 1.244 | **1.290** (k=1) | 1.293 | 1.277 | 1.284 | 1.248 | 1.359 | 1.407 | 1.399 | 1.311 |
| all | q8 | 1.299 | 1.274 | 1.281 | 1.222 | **1.299** (k=1) | 1.300 | 1.270 | 1.282 | 1.247 | 1.355 | 1.389 | 1.400 | 1.305 |
| all | fp16 | 1.230 | 1.226 | 1.241 | 1.202 | **1.241** (k=3) | 1.230 | 1.216 | 1.226 | 1.190 | 1.326 | 1.362 | 1.375 | 1.288 |

### speed-up over plain decode, greedy, verify prices "target" (T=2 x1.12, T=3 x1.2, T=4 x1.35, T=5 x1.35; draft cost / decode step 88.2 ms)

| class | head | k=1 | k=2 | k=3 | k=4 | best | refeed k=1 | refeed k=2 | refeed k=3 | refeed k=4 | gate 0.3 | gate 0.5 | gate 0.7 | gate 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code | q4 | 1.579 | 1.931 | 2.086 | 2.214 | **2.214** (k=4) | 1.584 | 1.956 | 2.057 | 2.315 | 2.354 | 2.339 | 2.243 | 2.075 |
| code | q8 | 1.586 | 1.964 | 2.047 | 2.146 | **2.146** (k=4) | 1.587 | 1.952 | 2.009 | 2.234 | 2.324 | 2.346 | 2.219 | 2.050 |
| code | fp16 | 1.482 | 1.844 | 1.950 | 2.045 | **2.045** (k=4) | 1.482 | 1.843 | 1.912 | 2.130 | 2.264 | 2.270 | 2.173 | 2.009 |
| en | q4 | 1.444 | 1.650 | 1.527 | 1.567 | **1.650** (k=2) | 1.448 | 1.627 | 1.560 | 1.576 | 1.616 | 1.632 | 1.542 | 1.348 |
| en | q8 | 1.458 | 1.617 | 1.524 | 1.541 | **1.617** (k=2) | 1.459 | 1.619 | 1.554 | 1.587 | 1.638 | 1.594 | 1.556 | 1.347 |
| en | fp16 | 1.369 | 1.541 | 1.474 | 1.511 | **1.541** (k=2) | 1.369 | 1.530 | 1.472 | 1.492 | 1.595 | 1.556 | 1.526 | 1.326 |
| ru | q4 | 1.446 | 1.560 | 1.431 | 1.398 | **1.560** (k=2) | 1.450 | 1.550 | 1.404 | 1.376 | 1.518 | 1.519 | 1.465 | 1.345 |
| ru | q8 | 1.455 | 1.551 | 1.428 | 1.371 | **1.551** (k=2) | 1.457 | 1.532 | 1.409 | 1.368 | 1.490 | 1.505 | 1.464 | 1.334 |
| ru | fp16 | 1.374 | 1.486 | 1.376 | 1.347 | **1.486** (k=2) | 1.374 | 1.454 | 1.335 | 1.292 | 1.454 | 1.479 | 1.433 | 1.318 |
| all | q4 | 1.470 | 1.660 | 1.569 | 1.583 | **1.660** (k=2) | 1.474 | 1.650 | 1.565 | 1.585 | 1.677 | 1.683 | 1.608 | 1.448 |
| all | q8 | 1.481 | 1.647 | 1.562 | 1.552 | **1.647** (k=2) | 1.482 | 1.638 | 1.560 | 1.578 | 1.670 | 1.661 | 1.611 | 1.440 |
| all | fp16 | 1.392 | 1.569 | 1.505 | 1.516 | **1.569** (k=2) | 1.392 | 1.550 | 1.479 | 1.489 | 1.628 | 1.624 | 1.579 | 1.419 |

### speed-up over plain decode, sampled, verify prices "today" (T=2 x1.29, T=3 x1.6, T=4 x1.7, T=5 x1.8; draft cost / decode step 88.2 ms)

| class | head | k=1 | k=2 | k=3 | k=4 | best | refeed k=1 | refeed k=2 | refeed k=3 | refeed k=4 | gate 0.3 | gate 0.5 | gate 0.7 | gate 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code | q4 | 1.344 | 1.456 | 1.573 | 1.563 | **1.573** (k=3) | 1.347 | 1.463 | 1.550 | 1.624 | 1.648 | 1.667 | 1.723 | 1.657 |
| code | q8 | 1.338 | 1.444 | 1.544 | 1.564 | **1.564** (k=4) | 1.339 | 1.450 | 1.527 | 1.574 | 1.624 | 1.656 | 1.698 | 1.610 |
| code | fp16 | 1.270 | 1.382 | 1.474 | 1.534 | **1.534** (k=4) | 1.270 | 1.390 | 1.481 | 1.526 | 1.605 | 1.623 | 1.663 | 1.584 |
| en | q4 | 1.243 | 1.198 | 1.198 | 1.121 | **1.243** (k=1) | 1.246 | 1.192 | 1.197 | 1.136 | 1.215 | 1.264 | 1.268 | 1.206 |
| en | q8 | 1.238 | 1.201 | 1.182 | 1.104 | **1.238** (k=1) | 1.239 | 1.198 | 1.194 | 1.123 | 1.210 | 1.259 | 1.272 | 1.203 |
| en | fp16 | 1.181 | 1.161 | 1.160 | 1.082 | **1.181** (k=1) | 1.181 | 1.151 | 1.147 | 1.081 | 1.180 | 1.239 | 1.250 | 1.188 |
| ru | q4 | 1.201 | 1.134 | 1.116 | 1.055 | **1.201** (k=1) | 1.204 | 1.147 | 1.121 | 1.064 | 1.228 | 1.268 | 1.266 | 1.218 |
| ru | q8 | 1.200 | 1.134 | 1.115 | 1.043 | **1.200** (k=1) | 1.200 | 1.135 | 1.119 | 1.062 | 1.216 | 1.262 | 1.270 | 1.221 |
| ru | fp16 | 1.150 | 1.104 | 1.085 | 1.019 | **1.150** (k=1) | 1.150 | 1.099 | 1.085 | 1.026 | 1.200 | 1.238 | 1.247 | 1.204 |
| all | q4 | 1.245 | 1.214 | 1.221 | 1.158 | **1.245** (k=1) | 1.248 | 1.218 | 1.221 | 1.175 | 1.288 | 1.330 | 1.337 | 1.281 |
| all | q8 | 1.241 | 1.213 | 1.211 | 1.145 | **1.241** (k=1) | 1.242 | 1.214 | 1.215 | 1.163 | 1.278 | 1.324 | 1.338 | 1.275 |
| all | fp16 | 1.185 | 1.175 | 1.178 | 1.121 | **1.185** (k=1) | 1.185 | 1.170 | 1.174 | 1.123 | 1.255 | 1.300 | 1.314 | 1.257 |

### speed-up over plain decode, sampled, verify prices "target" (T=2 x1.12, T=3 x1.2, T=4 x1.35, T=5 x1.35; draft cost / decode step 88.2 ms)

| class | head | k=1 | k=2 | k=3 | k=4 | best | refeed k=1 | refeed k=2 | refeed k=3 | refeed k=4 | gate 0.3 | gate 0.5 | gate 0.7 | gate 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code | q4 | 1.532 | 1.881 | 1.920 | 1.987 | **1.987** (k=4) | 1.536 | 1.889 | 1.888 | 2.059 | 2.090 | 2.083 | 2.059 | 1.914 |
| code | q8 | 1.525 | 1.867 | 1.882 | 1.985 | **1.985** (k=4) | 1.526 | 1.868 | 1.857 | 1.989 | 2.057 | 2.062 | 2.036 | 1.856 |
| code | fp16 | 1.436 | 1.763 | 1.783 | 1.928 | **1.928** (k=4) | 1.437 | 1.769 | 1.785 | 1.910 | 2.022 | 2.012 | 1.986 | 1.821 |
| en | q4 | 1.417 | 1.549 | 1.463 | 1.426 | **1.549** (k=2) | 1.421 | 1.540 | 1.460 | 1.442 | 1.514 | 1.507 | 1.451 | 1.313 |
| en | q8 | 1.412 | 1.553 | 1.442 | 1.402 | **1.553** (k=2) | 1.413 | 1.545 | 1.454 | 1.422 | 1.503 | 1.501 | 1.451 | 1.309 |
| en | fp16 | 1.338 | 1.489 | 1.407 | 1.365 | **1.489** (k=2) | 1.338 | 1.468 | 1.384 | 1.354 | 1.460 | 1.471 | 1.424 | 1.291 |
| ru | q4 | 1.369 | 1.467 | 1.363 | 1.342 | **1.467** (k=2) | 1.373 | 1.482 | 1.368 | 1.352 | 1.477 | 1.473 | 1.422 | 1.321 |
| ru | q8 | 1.368 | 1.467 | 1.360 | 1.324 | **1.467** (k=2) | 1.369 | 1.465 | 1.362 | 1.345 | 1.464 | 1.465 | 1.422 | 1.327 |
| ru | fp16 | 1.303 | 1.416 | 1.317 | 1.286 | **1.416** (k=2) | 1.303 | 1.402 | 1.310 | 1.286 | 1.442 | 1.429 | 1.394 | 1.306 |
| all | q4 | 1.419 | 1.570 | 1.491 | 1.473 | **1.570** (k=2) | 1.423 | 1.575 | 1.489 | 1.492 | 1.585 | 1.580 | 1.529 | 1.405 |
| all | q8 | 1.415 | 1.569 | 1.476 | 1.454 | **1.569** (k=2) | 1.416 | 1.566 | 1.479 | 1.473 | 1.571 | 1.571 | 1.527 | 1.399 |
| all | fp16 | 1.342 | 1.505 | 1.429 | 1.414 | **1.505** (k=2) | 1.342 | 1.492 | 1.417 | 1.406 | 1.538 | 1.536 | 1.496 | 1.378 |

## Verdict

- The three heads accept the same drafts within 2 %: over all greedy positions the first
  draft is right at 0.777 (q4) / 0.784 (q8) / 0.782 (fp16), drafts kept of 4: 1.79 / 1.82 /
  1.81. Paired on the same 1585 positions q8 keeps 0.030 ± 0.012 more drafts of 4 than q4,
  fp16 0.020 ± 0.011 more; q8 and fp16 do not differ (0.010 ± 0.007).
- By projected speed q4 and q8 are level (all classes, greedy: ×1.290 / ×1.299 at one draft,
  ×1.407 / ×1.389 with the gate at 0.5; sampled ×1.245 / ×1.241) and fp16 is 3–5 % behind
  both: its draft costs 8.1 ms against 7.1–7.3 and its flush of 2+ rows 16.9 ms against
  8.4–8.8.
- The head's quantization is not what limits it: with the target's real hidden state at
  every depth ("oracle") the head would keep 2.29 drafts of 4 instead of 1.79 — the loss is
  in chaining on its own output (cosine with the state it stands in for: 0.58–0.68 at depth
  1, 0.41–0.49 at depth 4), the same for the three heads.
- fp16 activations are enough: no non-finite outputs, the chains equal fp32 at 955–956 of
  960 positions.
- Used since: **q4** (`weights/qwen38-27b-mtp`, the checkpoint's own head, 128 MB smaller
  than q8). It was the default before this run, and the run gives no reason to change it.

## Not verified

- The heads were compared by this probe (acceptance and stage costs), not in the worker's
  decode loop: the `ab_mtp.py --alt-head` row (`mtp0+q+h`) was not run, and every gate and
  benchmark of 2026-11-17 used q4.
- The speed projection ignores the round's host time and takes the verify prices as given.
- One sampled trajectory per text (one seed).

## Files

`scripts/perf/mtp/accept_probe.py`, `scripts/perf/mtp/accept_tables.py`, `ab_mtp.py`
(`--alt-head`, `+h`).

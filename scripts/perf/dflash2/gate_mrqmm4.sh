#!/usr/bin/env zsh
# GPU gate for the 4-bit multi-row kernel (mrqmm, QWEN_MRQMM4) on the 4-bit 27B
# (weights/qwen38-27b-mlx-4bit, lmstudio-community MLX 4-bit, affine g64
# throughout). Run from the repo root on an idle GPU (nothing else on it:
# CLAUDE.md "Measuring performance"); `setopt NO_BG_NICE` if backgrounded.
# Each step prints / writes its result to $PERF_OUT/mrqmm4/; pass criteria per step:
#
#  1. kernel unit tests (synthetic, ~5 s): tests/test_mrqmm.py all pass (the 4-bit
#     section: check_close at the 27B shapes, rows 3..16, the load self-test at
#     every shape incl. lm_head, rows independent of the row count).
#  2. real weights (~1.2 GB of tensors, no model): MRQMM_REAL_WEIGHTS=1
#     test_real_27b_4bit_weights passes (self-test + check_close per shape);
#     kern5_accuracy.py: worst rms rel k5 <= mlx's and <= ~3e-4 (5-bit: 2.3-2.4e-4),
#     rows_equal_across_M all true, row0_independent_of_neighbours all true.
#  3. kernel cost (no model): kern5_bench.py on the real 4-bit weights; read the
#     "step" ratio k5 / qmv1 at 3..8 rows (5-bit real: x1.48-1.61; random 4-bit
#     2026-11-10: x1.64 at 8 rows) and, with --sweep, whether 4x8 (8 rows) /
#     2x8 (16 rows) — mrqmm._config's 5/4-bit choice — is still the best config.
#  4. decode step A/B in one process (model, no drafter): step_prof --ab base,nomr
#     at B = 3,4,8,16; base must beat nomr at every B (5-bit: forward linears
#     249 -> 89 ms at 8 rows). The worker's load log must say
#     "mrqmm 4-bit g64 self-test passed" and "(4/8-bit, QWEN_MRQMM4)" or "(4-bit, ...)".
#  5. DFlash A/B in one process (model + DFlash2 drafter): ab27.py plain vs m8/8
#     (kernel) vs e0/8 (quantized_matmul verify); ident phase: m8/8 vs plain on
#     the 16-task corpus (5-bit: 15/16 without thinking, 12/16 with; divergences
#     only at near-ties, check with diverge_margin.py if lower). Pass: m8/8 /
#     plain >= ~x1.6 without thinking (5-bit: x1.77).
#  6. 4-bit vs 5-bit (two processes: both models do not fit at once): step 5 with
#     QWEN_MODEL=weights/qwen38-27b-mlx-5bit QWEN_MRQMM5=1. Compare the in-process
#     ratios (m8/8 / plain) first; absolute tok/s across processes only with each
#     run's own plain as the probe (the GPU drifts +-30 %).
#
# Usage: scripts/perf/dflash2/gate_mrqmm4.sh [step ...]   (default: 1 2 3 4 5 6)
set -eu
ROOT=${0:A:h}/../../..
W=$ROOT/src/worker
PY=${PY:-$W/.venv/bin/python}
OUT=${PERF_OUT:-/tmp/perf}/mrqmm4
M4=${M4:-$ROOT/weights/qwen38-27b-mlx-4bit}
M5=${M5:-$ROOT/weights/qwen38-27b-mlx-5bit}
DRAFT=${DRAFT:-$ROOT/weights/qwen38-27b-dflash2}
mkdir -p $OUT
cd $W
if (( $# )); then steps=("$@"); else steps=(1 2 3 4 5 6); fi
# the 27B's load recipe (CLAUDE.md "DFlash2 on the 27B"); needs the wired-limit dance:
#   sudo sysctl iogpu.wired_limit_mb=28672
RUN27=(QWEN_KV_BITS=0 QWEN_WIRED_LIMIT_GB=27 QWEN_PROMPT_CACHE=0 PYTHONPATH=.)

for s in $steps; do
  case $s in
  1) $PY -m pytest -q tests/test_mrqmm.py 2>&1 | tail -3 ;;
  2) MRQMM_REAL_WEIGHTS=1 MRQMM_WEIGHTS_27B4=$M4 $PY -m pytest -q tests/test_mrqmm.py \
       -k test_real_27b_4bit_weights 2>&1 | tail -3
     PYTHONPATH=. $PY ../../scripts/perf/dflash2/kern5_accuracy.py --weights $M4 \
       --out $OUT/kern4_accuracy_real.json ;;
  3) PYTHONPATH=. $PY ../../scripts/perf/dflash2/kern5_bench.py --weights $M4 --rounds 5 \
       --sweep 2x8,4x8,4x4 --out $OUT/kern4_bench_real.json ;;
  4) env $RUN27 QWEN_MODEL=$M4 QWEN_MRQMM4=1 QWEN_BATCH_PREFILL_SHARE=1 $PY \
       ../../scripts/perf/prof/step_prof.py --B 3,4,8,16 --ctx 1k --gen 200 \
       --ab base,nomr --ab-steps 32 --json $OUT/step_ab_4bit.json 2>&1 | tee $OUT/step_ab_4bit.log \
       | grep -E "self-test|mrqmm|base|nomr" ;;
  5) env $RUN27 QWEN_MODEL=$M4 QWEN_MRQMM4=1 QWEN_DRAFT=$DRAFT $PY \
       ../../scripts/perf/dflash2/ab27.py --rounds 3 --variants plain,m8/8,e0/8 --ident m8/8 \
       --json $OUT/ab27_4bit.json 2>&1 | tee $OUT/ab27_4bit.log | grep -vE "INFO|DEBUG" ;;
  6) env $RUN27 QWEN_MODEL=$M5 QWEN_MRQMM5=1 QWEN_DRAFT=$DRAFT $PY \
       ../../scripts/perf/dflash2/ab27.py --rounds 3 --variants plain,m8/8,e0/8 --ident m8/8 \
       --json $OUT/ab27_5bit.json 2>&1 | tee $OUT/ab27_5bit.log | grep -vE "INFO|DEBUG" ;;
  *) echo "unknown step $s" >&2; exit 2 ;;
  esac
done

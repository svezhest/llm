#!/bin/zsh
# Batch-invariant mode, GPU phase (docs/perf/2026-10-24/design/batch_invariant.md):
#   1. kernel cost per shape (no model, ~1.3 GB, ~2 min)
#   2. Metal tests (tiny shapes, < 0.3 GB, ~30 s)
#   3. real 9B: a row in a batch == the row alone (forced + free + control, ~8 + 5 min)
#   4. real 9B: decode step on / off A/B in-process (~7 min)
# Needs the GPU lease: no worker / gateway / other GPU job, Activity Monitor
# closed. One command; everything goes to $PERF_OUT (default
# /tmp/perf/batch_invariant), the summary lines to $PERF_OUT/summary.txt.
#   zsh scripts/perf/batch_invariant/run_gpu.zsh [step ...]   (default: 1 2 3 4)
# SMOKE=1: step 1 on tiny shapes (checks the plumbing, numbers meaningless).
setopt NO_BG_NICE
HERE=${0:A:h}
REPO=${HERE:h:h:h}
MAIN=${MAIN:-/Users/user/Projects/llm}                 # venv + weights (the main tree)
PY=${PY:-$MAIN/src/worker/.venv/bin/python}
MODEL=${QWEN_MODEL:-$MAIN/weights/ornith15-9b-mlx-8bit}
export PERF_OUT=${PERF_OUT:-/tmp/perf/batch_invariant}
mkdir -p $PERF_OUT
cd $REPO/src/worker
[[ -f worker_pb2.py ]] || cp $MAIN/src/worker/worker_pb2*.py .
steps=(${@:-1 2 3 4})
# the worker's 9B env (QWEN_FP16 / PFQMM / MRQMM / GDN_FUSE defaults), no
# prompt cache (no cascade, no reply KV: the matmul dispatch alone)
W9=(QWEN_MODEL=$MODEL QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=0 QWEN_PRESENCE_PENALTY=0
    QWEN_BATCH_INVARIANT=1 QWEN_MAX_BATCH=16 PYTHONPATH=. LOG_LEVEL=INFO)
log() { print -r -- "$(date +%H:%M:%S) $*" | tee -a $PERF_OUT/summary.txt }

if (( ${steps[(Ie)1]} )); then
  log "1 kernel bench"
  $PY ../../scripts/perf/batch_invariant/kern_bench.py --rounds 5 ${SMOKE:+--smoke} \
    --out $PERF_OUT/bi_kern.json > $PERF_OUT/bi_kern.txt 2>&1
  grep "step linears" $PERF_OUT/bi_kern.txt | tee -a $PERF_OUT/summary.txt
fi
if (( ${steps[(Ie)2]} )); then
  log "2 Metal tests"
  BATCH_INV_GPU=1 $PY -m pytest -q tests/test_batch_invariant_metal.py tests/test_mrqmm.py \
    > $PERF_OUT/bi_metal_tests.txt 2>&1 || true
  tail -2 $PERF_OUT/bi_metal_tests.txt | head -1 | tee -a $PERF_OUT/summary.txt
fi
if (( ${steps[(Ie)3]} )); then
  log "3 real 9B batch == solo"
  env $W9 $PY ../../scripts/perf/batch_invariant/check_real.py --out $PERF_OUT/bi_check.json \
    > $PERF_OUT/bi_check.txt 2>&1 || true
  grep -E "^(ref|solo|f |c |free|PASS|FAIL)" $PERF_OUT/bi_check.txt | tee -a $PERF_OUT/summary.txt
  grep -E "^coverage" $PERF_OUT/bi_check.txt | tee -a $PERF_OUT/summary.txt
  # short prompts, long answers: every batch size 1..16 decodes (the run above
  # is prefill-bound, mean batch ~4)
  env $W9 $PY ../../scripts/perf/batch_invariant/check_real.py --lens 300,600 --gen 256 --B 16 \
    --out $PERF_OUT/bi_check_full16.json > $PERF_OUT/bi_check_full16.txt 2>&1 || true
  grep -E "^(f |c |free|PASS|FAIL)" $PERF_OUT/bi_check_full16.txt | tee -a $PERF_OUT/summary.txt
fi
if (( ${steps[(Ie)4]} )); then
  log "4 real 9B step A/B"
  env $W9 QWEN_BATCH_PREFILL_SHARE=1 $PY ../../scripts/perf/batch_invariant/step_ab.py \
    --B 1,2,4,8,16 --rounds 3 --out $PERF_OUT/bi_step_ab.json > $PERF_OUT/bi_step_ab.txt 2>&1 || true
  grep -E "^B=" $PERF_OUT/bi_step_ab.txt | tee -a $PERF_OUT/summary.txt
fi
log "done -> $PERF_OUT"

#!/bin/zsh
# GPU plan of QWEN_VERIFY_MRQMM (docs/perf/2026-10-24/design/verify_mrqmm.md), one command:
#   zsh scripts/perf/verify_mrqmm/run.zsh
# Outputs in $PERF_OUT/verify_mrqmm (default /tmp/perf). Nothing else on the GPU meanwhile.
setopt NO_BG_NICE
set -e
HERE=${0:A:h}
ROOT=${HERE:h:h:h}
WD=${WEIGHTS:-/Users/user/Projects/llm/weights}
PY=${PY:-/Users/user/Projects/llm/src/worker/.venv/bin/python}
OUT=${PERF_OUT:-/tmp/perf}/verify_mrqmm
mkdir -p $OUT
cd $ROOT/src/worker
[[ -f worker_pb2.py ]] || cp /Users/user/Projects/llm/src/worker/worker_pb2*.py .
if pgrep -f "python worker.py|qwen-gateway" >/dev/null; then
  echo "WARNING: a worker/gateway is running (GPU and memory are shared): $(pgrep -fl 'python worker.py|qwen-gateway')"
fi
echo "== 0. Metal correctness (small shapes) ==" | tee $OUT/summary.txt
VERIFY_GPU=1 $PY -m pytest -q -p no:cacheprovider tests/test_gpu_verify_sampler.py 2>&1 | tail -3 | tee -a $OUT/summary.txt
echo "== 1. DFlash A/B in-process (9B + drafter): speed x 3 rounds, identity on 16 tasks ==" | tee -a $OUT/summary.txt
env QWEN_MODEL=$WD/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_DRAFT=$WD/qwen35-9b-dflash \
    QWEN_VERIFY_MRQMM=1 PYTHONPATH=. \
    $PY $HERE/dflash_ab.py --rounds ${ROUNDS:-3} --json $OUT/dflash_ab.json \
    2> $OUT/dflash_ab.log | tee $OUT/dflash_ab.txt
sed -n '/== speed/,$p' $OUT/dflash_ab.txt | tee -a $OUT/summary.txt
echo "done: $OUT/summary.txt"

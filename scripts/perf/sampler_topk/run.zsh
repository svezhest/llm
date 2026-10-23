#!/bin/zsh
# GPU plan of QWEN_SAMPLER_TOPK_FAST (docs/perf/2026-10-24/design/sampler_topk.md), one command:
#   zsh scripts/perf/sampler_topk/run.zsh            (E2E=0: skip the model run)
# Outputs in $PERF_OUT/sampler_topk (default /tmp/perf). Nothing else on the GPU meanwhile.
setopt NO_BG_NICE
set -e
HERE=${0:A:h}
ROOT=${HERE:h:h:h}
WD=${WEIGHTS:-/Users/user/Projects/llm/weights}
PY=${PY:-/Users/user/Projects/llm/src/worker/.venv/bin/python}
OUT=${PERF_OUT:-/tmp/perf}/sampler_topk
mkdir -p $OUT
cd $ROOT/src/worker
[[ -f worker_pb2.py ]] || cp /Users/user/Projects/llm/src/worker/worker_pb2*.py .
if pgrep -f "python worker.py|qwen-gateway" >/dev/null; then
  echo "WARNING: a worker/gateway is running (GPU and memory are shared): $(pgrep -fl 'python worker.py|qwen-gateway')"
fi
echo "== 0. Metal correctness at V = 248320 ==" | tee $OUT/summary.txt
VERIFY_GPU=1 $PY -m pytest -q -p no:cacheprovider tests/test_gpu_verify_sampler.py -k sample_rows 2>&1 | tail -3 | tee -a $OUT/summary.txt
echo "== 1. sampler microbench, off vs on interleaved ==" | tee -a $OUT/summary.txt
PYTHONPATH=. $PY $HERE/sampler_ab.py --json $OUT/sampler_ab.json | tee -a $OUT/summary.txt
if [[ ${E2E:-1} == 1 ]]; then
  echo "== 2. end-to-end, 9B, 16 card-sampled rows, rounds off/on/on/off ==" | tee -a $OUT/summary.txt
  env QWEN_MODEL=$WD/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_BATCH_PREFILL_SHARE=1 PYTHONPATH=. \
      $PY $HERE/e2e_ab.py --rounds ${ROUNDS:-8} --json $OUT/e2e_ab.json 2> $OUT/e2e_ab.log \
      | tee -a $OUT/summary.txt
fi
echo "done: $OUT/summary.txt"

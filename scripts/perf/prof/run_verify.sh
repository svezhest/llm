#!/bin/zsh
# prefill work verification: identity corpus (B=1), prompt-cache identity, 35B/27B smoke
setopt NO_BG_NICE
S=${PERF_OUT:-/tmp/perf}; B=$S/bump; N=$B/gpu/new
R=$S/results/pf/verify; mkdir -p $R
WD=/Users/user/Projects/llm/weights
cd /Users/user/Projects/llm/src/worker
PY=.venv/bin/python
W=(env PYTHONPATH=. $PY $S/gpu_tests/wtest.py)
NINE=(QWEN_MODEL=$WD/ornith15-9b-mlx-8bit QWEN_KV_BITS=0)
M27=(QWEN_MODEL=$WD/qwen38-27b-mlx-5bit QWEN_KV_BITS=8 QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512)
M35=(QWEN_MODEL=$WD/qwen36-35b-a3b-mlx-ud3 QWEN_KV_BITS=4 QWEN_WIRED_LIMIT_GB=27 QWEN_PREFILL_STEP=512)
stop() { pkill -f "worker.py --addr 127.0.0.1:50051"; sleep 6; }
set -x
$B/sw.sh $R/w9.log new $NINE QWEN_MAX_BATCH=1 || exit 1
$W corpus --out $R/corpus9.json > $R/corpus9.txt
$W compare $N/corpus9.json $R/corpus9.json > $R/compare9.txt
env PYTHONPATH=. $PY $B/tools/pc_ident.py --out $R/pc_cold.json > $R/pc_cold.txt
$W compare $N/pc_cold.json $R/pc_cold.json > $R/compare_pc_cold.txt
stop
$B/sw.sh $R/w9p.log new $NINE QWEN_PROMPT_CACHE=1 || exit 1
env PYTHONPATH=. $PY $B/tools/pc_ident.py --out $R/pc_hit.json > $R/pc_hit.txt
$W compare $N/pc_cold.json $R/pc_hit.json > $R/compare_pc_hit.txt
stop
$B/sw.sh $R/w35.log new $M35 || exit 1
$W corpus --out $R/corpus35.json > $R/corpus35.txt
$W compare $N/corpus35.json $R/corpus35.json > $R/compare35.txt
env PYTHONPATH=. $PY $B/tools/long_test.py prompt --chars 70000 --n 1 > $R/long35.txt
stop
$B/sw.sh $R/w27.log new $M27 || exit 1
$W corpus --out $R/corpus27.json > $R/corpus27.txt
$W compare $N/corpus27.json $R/corpus27.json > $R/compare27.txt
env PYTHONPATH=. $PY $B/tools/long_test.py prompt --chars 42000 --n 1 > $R/long27.txt
stop
echo VERIFY DONE
tail -2 $R/compare*.txt; cat $R/long35.txt $R/long27.txt; grep -h "pfqmm" $R/w*.log | cut -c1-200

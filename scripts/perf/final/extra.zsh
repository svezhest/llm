#!/bin/zsh
setopt NO_BG_NICE
cd /Users/user/Projects/llm
B=(uv run python scripts/bench_concurrency.py --url http://127.0.0.1:8090 --probe 64 --cooldown 30 --max-tokens 256 --prompt-tokens 600 --worker-log ${PERF_OUT:-/tmp/perf}/results/worker_final.log)
echo "$(date '+%T') extra desc start" >> ${PERF_OUT:-/tmp/perf}/results/final_status.txt
$B --concurrency 16,8,4 --out ${PERF_OUT:-/tmp/perf}/results/final_256_check_desc.json > ${PERF_OUT:-/tmp/perf}/results/final_256_check_desc.txt 2>&1
echo "$(date '+%T') extra asc start" >> ${PERF_OUT:-/tmp/perf}/results/final_status.txt
$B --concurrency 4,8 --out ${PERF_OUT:-/tmp/perf}/results/final_256_check_asc.json > ${PERF_OUT:-/tmp/perf}/results/final_256_check_asc.txt 2>&1
echo "$(date '+%T') extra done" >> ${PERF_OUT:-/tmp/perf}/results/final_status.txt

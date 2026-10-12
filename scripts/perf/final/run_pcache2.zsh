#!/bin/zsh
setopt NO_BG_NICE
# Phase-1 prompt cache GPU validation (docs/perf/2026-10-12/prompt_cache_phase1_gpu.md). Restarts only the :50051 worker and our :8090 gateway.
R=/Users/user/Projects/llm
SP=${PERF_OUT:-/tmp/perf}
S=$SP/pcache2; mkdir -p $S
ST=$S/pcache2_status.txt
GL=${GATEWAY_LOG:-$SP/results/gateway_final.log}
B=(uv run python scripts/bench_concurrency.py --url http://127.0.0.1:8090)
C=(--ignore-eos on --probe 64 --cooldown 30 --thinking off)
PY=(env PYTHONPATH=$R/src/worker $R/src/worker/.venv/bin/python $R/scripts/perf/final/pcache2_v1.py)
cnt() { local c; c=$(grep -c "$1" "$2" 2>/dev/null); echo ${c:-0}; }
note() { echo "$(date '+%F %T') $*" >> $ST; }
state() { note "therm: $(pmset -g therm 2>&1 | grep -v '^Note' | tr '\n' ' ') load: $(sysctl -n vm.loadavg)"; }
health() { note "health: $($PY health 2>/dev/null)"; }

stop_worker() {
  local p=$(lsof -nP -iTCP:50051 -sTCP:LISTEN -t)
  [[ -n $p ]] && { note "kill worker $p"; kill $p; }
  for i in {1..120}; do [[ -z $(lsof -nP -iTCP:50051 -sTCP:LISTEN -t) ]] && ! kill -0 $p 2>/dev/null && break; sleep 1; done
  sleep 3
}
start_worker() {  # $1 = log, rest = env
  local log=$1; shift
  local n0=$(cnt "worker serving on" $log)
  note "start worker env: $* log: $log"
  (cd $R/src/worker && env QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 "$@" nohup uv run python worker.py >> $log 2>&1 &)
  for i in {1..300}; do (( $(cnt "worker serving on" $log) > n0 )) && break; sleep 1; done
  note "worker up: $(grep 'batching:\|prompt cache:' $log | tail -2 | tr '\n' ' ')"
}
wait_gateway() {  # the gateway follows the worker's Health (polls every 15 s)
  local g0=$1
  for i in {1..60}; do (( $(cnt "Health changed" $GL) > g0 )) && break; sleep 1; done
  sleep 2
  note "gateway: $(grep 'Health changed\|max_batch' $GL | tail -2 | sed 's/\x1b\[[0-9;]*m//g' | tr '\n' ' ')"
}
restart_worker() {
  local g0=$(cnt "Health changed" $GL)
  stop_worker; start_worker "$@"; wait_gateway $g0
}
run() {  # name, args...
  local n=$1; shift
  note "phase $n start: $*"; state
  $B "$@" $C --worker-log $WL --out $S/pcache2_$n.json > $S/pcache2_$n.txt 2>&1
  note "phase $n end rc=$?"; state; health
}

cd $R
note "=== start"; state
# --- R1: cache ON, new gateway binary
WL=$S/worker_v1_on.log
gp=$(lsof -nP -iTCP:8090 -sTCP:LISTEN -t)
[[ -n $gp ]] && { note "kill gateway $gp"; kill $gp; for i in {1..30}; do kill -0 $gp 2>/dev/null || break; sleep 1; done; }
stop_worker
start_worker $WL QWEN_PROMPT_CACHE=1
note "start gateway (new binary)"
(cd $R && env GATEWAY_ADDR=127.0.0.1:8090 RUST_LOG=info nohup ./target/release/qwen-gateway >> $GL 2>&1 &)
for i in {1..60}; do [[ -n $(lsof -nP -iTCP:8090 -sTCP:LISTEN -t) ]] && break; sleep 1; done
note "gateway pid $(lsof -nP -iTCP:8090 -sTCP:LISTEN -t)"; sleep 2
note "phase V1on start"; state
$PY on --out $S/pcache2_v1_on.json > $S/pcache2_v1_on.txt 2>&1
note "phase V1on end rc=$?"; state
# --- R2: cache OFF
WL=$S/worker_v1_off.log
restart_worker $WL
note "phase V1off start"
$PY off --ref $S/pcache2_v1_on.json --out $S/pcache2_v1_off.json > $S/pcache2_v1_off.txt 2>&1
note "phase V1off end rc=$?"; state
$PY cmp $S/pcache2_v1_on.json $S/pcache2_v1_off.json > $S/pcache2_v1_cmp.txt 2>&1
note "V1 cmp rc=$? $(tail -1 $S/pcache2_v1_cmp.txt)"
# --- R3: cache ON, 1 GiB soft cap: V4 pressure
WL=$S/worker_v4_gb1.log
restart_worker $WL QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=1
run V4gb1 --prefix distinct --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
# --- R4: cache ON default: V2 (shared, distinct = V4 at 3 GB, shared), V3; left running
WL=$S/worker_v2.log
restart_worker $WL QWEN_PROMPT_CACHE=1
run V2s1 --prefix shared   --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
run V2d  --prefix distinct --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
run V2s2 --prefix shared   --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
run V3   --prefix shared   --concurrency 1  --requests-per-level 8  --max-tokens 256 --prompt-tokens 8000
note "=== done (worker left running, cache on, log $WL)"

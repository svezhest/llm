#!/bin/zsh
setopt NO_BG_NICE
# Segment-sharing prompt cache (Part A) GPU validation (docs/perf/2026-10-12/segments_partA_gpu.md).
# Same method as run_pcache2*.zsh. Restarts only the :50051 worker; the :8090 gateway stays.
# Phases: V1 (identity, vs the pcache2 cache-off references) / V2 (shared 8K x2 + distinct) /
# V3 (shared 32K c=16) / V4 (120K store bound, repeat, 32K next to it). Arg: phase list (default all).
R=/Users/user/Projects/llm
SP=${PERF_OUT:-/tmp/perf}
S=$SP/pcache3; mkdir -p $S
ST=$S/pcache3_status.txt
GL=${GATEWAY_LOG:-$SP/results/gateway_final.log}
REF=${PCACHE_REF:-$R/docs/perf/2026-10-12/raw}
B=(uv run python scripts/bench_concurrency.py --url http://127.0.0.1:8090)
C=(--ignore-eos on --probe 64 --cooldown 30 --thinking off)
PYB=(env PYTHONPATH=$R/src/worker $R/src/worker/.venv/bin/python)
PY=($PYB $R/scripts/perf/final/pcache2_v1.py)
PX=($PYB $R/scripts/perf/final/pcache2_v1x.py)
cnt() { local c; c=$(grep -c "$1" "$2" 2>/dev/null); echo ${c:-0}; }
note() { echo "$(date '+%F %T') $*" >> $ST; }
state() { note "therm: $(pmset -g therm 2>&1 | grep -v '^Note' | tr '\n' ' ') load: $(sysctl -n vm.loadavg)"; }
health() { note "health: $($PY health 2>/dev/null)"; }
gpu_free() {  # another agent's GPU tests: wait (30 s polls, up to 10 min)
  local i
  for i in {1..20}; do
    local o=$(pgrep -fl 'pytest|microbench|agree' | grep -v pgrep)
    [[ -z $o ]] && return 0
    note "GPU busy, waiting: $(echo $o | head -2 | cut -c1-160 | tr '\n' ' ')"; sleep 30
  done
  note "GPU still busy after 10 min: going on"
}

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
  local wp=$(lsof -nP -iTCP:50051 -sTCP:LISTEN -t)
  note "worker up pid $wp nice $(ps -o nice= -p $wp): $(grep 'batching:\|prompt cache:' $log | tail -2 | tr '\n' ' ')"
}
wait_gateway() {
  local g0=$1
  for i in {1..60}; do (( $(cnt "Health changed" $GL) > g0 )) && break; sleep 1; done
  sleep 2
  note "gateway: $(grep 'Health changed\|max_batch' $GL | tail -1 | sed 's/\x1b\[[0-9;]*m//g')"
}
restart_worker() {
  local g0=$(cnt "Health changed" $GL)
  gpu_free; stop_worker; start_worker "$@"; wait_gateway $g0
}
run() {  # name, args...
  local n=$1; shift
  gpu_free
  note "phase $n start: $*"; state
  $B "$@" $C --worker-log $WL --out $S/pcache3_$n.json > $S/pcache3_$n.txt 2>&1
  note "phase $n end rc=$?"; state; health
}

cd $R
PH=${1:-V1 V2 V3 V4}
note "=== start ($PH)"; state
if [[ $PH == *V1* ]]; then
  WL=$S/worker_v1.log
  restart_worker $WL QWEN_PROMPT_CACHE=1
  gpu_free; note "phase V1on start"; state
  $PY on --out $S/pcache3_v1_on.json > $S/pcache3_v1_on.txt 2>&1
  note "phase V1on end rc=$?"; state; health
  $PY cmp $S/pcache3_v1_on.json $REF/pcache2_v1_off.json > $S/pcache3_v1_cmp.txt 2>&1
  note "V1 cmp rc=$? $(tail -1 $S/pcache3_v1_cmp.txt)"
  WL=$S/worker_v1x.log
  restart_worker $WL QWEN_PROMPT_CACHE=1
  gpu_free; note "phase V1x on start"
  $PX on - $S/pcache3_v1x_on.json > $S/pcache3_v1x_on.txt 2>&1
  note "phase V1x on end rc=$?"; health
  $PX cmp $S/pcache3_v1x_on.json $REF/pcache2_v1x_off.json > $S/pcache3_v1x_cmp.txt 2>&1
  note "V1x cmp rc=$? $(tail -1 $S/pcache3_v1x_cmp.txt)"
fi
if [[ $PH == *V2* || $PH == *V3* ]]; then
  WL=$S/worker_v2.log
  restart_worker $WL QWEN_PROMPT_CACHE=1
  if [[ $PH == *V2* ]]; then
    run V2s1 --prefix shared   --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
    run V2d  --prefix distinct --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
    run V2s2 --prefix shared   --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
  fi
  [[ $PH == *V3* ]] && run V3s32k --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 128 --prompt-tokens 32000
fi
if [[ $PH == *V4* ]]; then
  WL=$S/worker_v4.log
  restart_worker $WL QWEN_PROMPT_CACHE=1
  gpu_free; note "phase V4 start"; state
  $PYB $R/scripts/perf/final/pcache3_v4.py $S/pcache3_V4.json $WL > $S/pcache3_V4.txt 2>&1 &
  cp=$!
  n0=$(wc -l < $WL)
  while kill -0 $cp 2>/dev/null; do   # abort if MLX active memory nears the 28 GiB cap
    a=$(tail -n +$n0 $WL | grep -o 'active=[0-9.]*GiB' | tail -1 | tr -dc '0-9.')
    if [[ -n $a ]] && (( a >= 26.5 )); then note "V4 ABORT: active $a GiB"; kill $cp; break; fi
    sleep 5
  done
  wait $cp; note "phase V4 end rc=$?"; state; health
fi
note "=== done (worker left running, cache on, log $WL)"

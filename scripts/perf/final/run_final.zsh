#!/bin/zsh
setopt NO_BG_NICE
R=/Users/user/Projects/llm
S=${PERF_OUT:-/tmp/perf}/results
ST=$S/final_status.txt
mv -f $ST $ST.aborted1 2>/dev/null
GL=$S/gateway_final.log
B=(uv run python scripts/bench_concurrency.py --url http://127.0.0.1:8090)
cnt() { local c; c=$(grep -c "$1" "$2" 2>/dev/null); echo ${c:-0}; }
note() { echo "$(date '+%F %T') $*" >> $ST; }
state() { note "therm: $(pmset -g therm 2>&1 | grep -v '^Note' | tr '\n' ' ') load: $(sysctl -n vm.loadavg)"; }

restart_worker() {  # $1 = log, rest = env assignments
  local log=$1; shift
  local p=$(lsof -nP -iTCP:50051 -sTCP:LISTEN -t)
  [[ -n $p ]] && { note "kill worker $p"; kill $p; }
  for i in {1..120}; do [[ -z $(lsof -nP -iTCP:50051 -sTCP:LISTEN -t) ]] && ! kill -0 $p 2>/dev/null && break; sleep 1; done
  sleep 3
  local n0=$(cnt "worker serving on" $log)
  local g0=$(cnt "Health changed" $GL)
  note "start worker env: $* log: $log"
  (cd $R/src/worker && env QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 "$@" nohup uv run python worker.py >> $log 2>&1 &)
  for i in {1..300}; do (( $(cnt "worker serving on" $log) > n0 )) && break; sleep 1; done
  note "worker up: $(grep 'batching:\|mrqmm' $log | tail -2 | tr '\n' ' ')"
  # wait for the gateway's Health poll to see the new worker (ready / max_batch)
  for i in {1..60}; do (( $(cnt "Health changed" $GL) > g0 )) && break; sleep 1; done
  sleep 2
  note "gateway: $(grep 'Health changed\|max_batch' $GL | tail -3 | sed 's/\x1b\[[0-9;]*m//g' | tr '\n' ' ')"
}

cd $R
note "=== start"; state
# b: final defaults
WL=$S/worker_final.log
restart_worker $WL
note "phase b256"; $B --concurrency 1,2,4,8,16 --repeat 2 --alternate --probe 64 --cooldown 30 --max-tokens 256 --prompt-tokens 600 --worker-log $WL --out $S/final_256.json > $S/final_256.txt 2>&1; note "b256 rc=$?"; state
note "phase b1024"; $B --concurrency 1,8,16 --repeat 2 --alternate --probe 64 --cooldown 30 --max-tokens 1024 --prompt-tokens 600 --worker-log $WL --out $S/final_1024.json > $S/final_1024.txt 2>&1; note "b1024 rc=$?"; state
for pt in 8000 32000; do
  note "phase prefill $pt"; $B --concurrency 1 --requests-per-level 2 --probe 64 --cooldown 30 --max-tokens 128 --prompt-tokens $pt --worker-log $WL --out $S/final_prefill_$pt.json > $S/final_prefill_$pt.txt 2>&1; note "prefill $pt rc=$?"; state
done
# c: ablation QWEN_MRQMM=0
WL=$S/worker_final_mrqmm0.log
restart_worker $WL QWEN_MRQMM=0
note "phase mrqmm0"; $B --concurrency 8,16 --repeat 1 --probe 64 --cooldown 30 --max-tokens 256 --prompt-tokens 600 --worker-log $WL --out $S/final_mrqmm0_256.json > $S/final_mrqmm0_256.txt 2>&1; note "mrqmm0 rc=$?"; state
# a: serialized reference, final code
WL=$S/worker_final_mb1.log
restart_worker $WL QWEN_MAX_BATCH=1
note "phase serial"; $B --concurrency 1,2,4,8 --repeat 2 --alternate --probe 64 --cooldown 30 --max-tokens 256 --prompt-tokens 600 --worker-log $WL --out $S/final_serial_256.json > $S/final_serial_256.txt 2>&1; note "serial rc=$?"; state
# leave defaults running
restart_worker $S/worker_final.log
note "=== done"

#!/bin/zsh
setopt NO_BG_NICE
# Prompt-cache bench (docs/perf/2026-10-12/prompt_cache_bench.md): gateway on :8090 stays up,
# the worker is restarted with QWEN_PROMPT_CACHE=1, then without it (left running).

R=/Users/user/Projects/llm
SP=${PERF_OUT:-/tmp/perf}
S=$SP/pcache; mkdir -p $S
ST=$S/pcache_status.txt
GL=${GATEWAY_LOG:-$SP/results/gateway_final.log}  # the running gateway's log (Health changes)
B=(uv run python scripts/bench_concurrency.py --url http://127.0.0.1:8090)
C=(--ignore-eos on --probe 64 --cooldown 30 --thinking off)
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
  note "worker up: $(grep 'batching:\|prompt cache:' $log | tail -2 | tr '\n' ' ')"
  for i in {1..60}; do (( $(cnt "Health changed" $GL) > g0 )) && break; sleep 1; done
  sleep 2
  note "gateway: $(grep 'Health changed\|max_batch' $GL | tail -3 | sed 's/\x1b\[[0-9;]*m//g' | tr '\n' ' ')"
}
run() {  # name, args...
  local n=$1; shift
  note "phase $n start: $*"
  $B "$@" $C --worker-log $WL --out $S/pcache_$n.json > $S/pcache_$n.txt 2>&1
  note "phase $n end rc=$?"; state
}

cd $R
note "=== start"; state
WL=$S/worker_pcache.log
restart_worker $WL QWEN_PROMPT_CACHE=1
run A1a --prefix shared   --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
run A2a --prefix distinct --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
run A1b --prefix shared   --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
run A2b --prefix distinct --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
run A3  --prefix shared   --concurrency 1  --requests-per-level 8  --max-tokens 256 --prompt-tokens 8000
run A4  --prefix distinct --concurrency 1  --requests-per-level 3  --max-tokens 256 --prompt-tokens 8000
if [[ -z $SKIP_A5 ]]; then
run A5  --prefix shared   --concurrency 8  --requests-per-level 8  --max-tokens 128 --prompt-tokens 32000
fi
WL=$S/worker_nocache.log
restart_worker $WL
run B1  --prefix shared   --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
note "=== done (worker left running, cache off, log $WL)"

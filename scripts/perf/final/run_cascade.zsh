#!/bin/zsh
setopt NO_BG_NICE
# Cascade attention (plan Part B, step B4) GPU validation (docs/perf/2026-10-12/cascade_gpu.md).
# Same method as run_pcache3.zsh: restarts only the :50051 worker; the :8090 gateway stays.
# Phases (arg: list, default all):
#   G1  exactness gate (agree_cascade.py; the worker is stopped while it runs)
#   G2  shared 8K wave c=16: ON (fresh) / OFF (fresh) / ON (fresh)  -> G3 + G4on run on the last ON worker
#   G3  shared 32K wave c=16, cascade ON
#   G4  shared 32K c=8 ON, 1K-prefix wave ON; then a fresh OFF worker: shared 32K c=8 OFF
#   G5  identity (pcache2_v1.py on, fresh ON worker) vs raw/pcache2_v1_off.json
#   G6  (extra) steady decode: 8K c=16 1024 tokens ON/OFF, 32K c=16 512 tokens ON
#   C1 C5 DEFEND  confirm run on the default (no QWEN_CASCADE env): waves / identity / leave a default worker
#   END worker left running with QWEN_CASCADE=0 + cache on
R=/Users/user/Projects/llm
SP=${PERF_OUT:-/tmp/perf}
S=$SP/cascade; mkdir -p $S
ST=$S/cascade_status.txt
GL=${GATEWAY_LOG:-$SP/results/gateway_final.log}
REF=$R/docs/perf/2026-10-12/raw
B=(uv run python scripts/bench_concurrency.py --url http://127.0.0.1:8090)
C=(--ignore-eos on --probe 64 --cooldown 30 --thinking off)
PYB=(env PYTHONPATH=$R/src/worker $R/src/worker/.venv/bin/python)
PY=($PYB $R/scripts/perf/final/pcache2_v1.py)
ON=(QWEN_PROMPT_CACHE=1 QWEN_CASCADE=1 QWEN_CASCADE_MIN_ROWS=8 QWEN_CASCADE_MIN_PREFIX=4096 QWEN_CASCADE_CHECK=1)
OFF=(QWEN_PROMPT_CACHE=1 QWEN_CASCADE=0)
cnt() { local c; c=$(grep -c "$1" "$2" 2>/dev/null); echo ${c:-0}; }
note() { echo "$(date '+%F %T') $*" >> $ST; }
state() { note "therm: $(pmset -g therm 2>&1 | grep -v '^Note' | tr '\n' ' ') load: $(sysctl -n vm.loadavg)"; }
health() { note "health: $($PY health 2>/dev/null)"; }
gpu_free() {  # another agent's GPU/CPU tests: wait (30 s polls, up to 10 min)
  local i
  for i in {1..20}; do
    local o=$(pgrep -fl 'pytest|microbench|agree' | grep -v pgrep | grep -v agree_cascade)
    [[ -z $o ]] && return 0
    note "GPU busy, waiting: $(echo $o | head -2 | cut -c1-160 | tr '\n' ' ')"; sleep 30
  done
  note "GPU still busy after 10 min: going on"
}
stop_worker() {
  local p=$(lsof -nP -iTCP:50051 -sTCP:LISTEN -t)
  [[ -n $p ]] && { note "kill worker $p"; kill $p; }
  for i in {1..120}; do [[ -z $(lsof -nP -iTCP:50051 -sTCP:LISTEN -t) ]] && { [[ -z $p ]] || ! kill -0 $p 2>/dev/null; } && break; sleep 1; done
  sleep 3
}
start_worker() {  # $1 = log, rest = env
  local log=$1; shift
  local n0=$(cnt "worker serving on" $log)
  note "start worker env: $* log: $log"
  (cd $R/src/worker && env QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 "$@" nohup uv run python worker.py >> $log 2>&1 &)
  for i in {1..300}; do (( $(cnt "worker serving on" $log) > n0 )) && break; sleep 1; done
  local wp=$(lsof -nP -iTCP:50051 -sTCP:LISTEN -t)
  note "worker up pid $wp nice $(ps -o nice= -p $wp): $(grep 'batching:\|cascade attention' $log | tail -2 | cut -c1-220 | tr '\n' ' ')"
}
wait_gateway() {
  local g0=$1
  for i in {1..60}; do (( $(cnt "Health changed" $GL) > g0 )) && break; sleep 1; done
  sleep 2
  note "gateway: $(grep 'Health changed\|max_batch' $GL | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200)"
}
restart_worker() {
  local g0=$(cnt "Health changed" $GL)
  gpu_free; stop_worker; start_worker "$@"; wait_gateway $g0
}
watch_mem() {  # pid: kill the bench client if MLX active memory nears the 28 GiB cap
  local cp=$1 n0=$2
  while kill -0 $cp 2>/dev/null; do
    a=$(tail -n +$n0 $WL | grep -o 'active=[0-9.]*GiB' | tail -1 | tr -dc '0-9.')
    if [[ -n $a ]] && (( a >= 27.0 )); then note "ABORT: active $a GiB"; kill $cp; break; fi
    sleep 3
  done
}
run() {  # name, args...
  local n=$1; shift
  gpu_free
  note "phase $n start: $*"; state
  local n0=$(( $(wc -l < $WL) + 1 ))
  $B "$@" $C --worker-log $WL --out $S/cascade_$n.json > $S/cascade_$n.txt 2>&1 &
  local cp=$!
  watch_mem $cp $n0
  wait $cp
  note "phase $n end rc=$?"; state; health
}

cd $R
PH=${1:-G1 G2 G3 G4 G5 END}
note "=== start ($PH)"; state
if [[ $PH == *G1* ]]; then
  gpu_free; stop_worker
  note "phase G1 start (worker stopped)"; state
  (cd $R/src/worker && QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 AGC_OUT=$S/cascade_g1.json \
     PYTHONPATH=. .venv/bin/python $R/scripts/perf/prof/agree_cascade.py > $S/cascade_g1.txt 2>&1)
  note "phase G1 end rc=$?"; state
fi
if [[ $PH == *G2* ]]; then
  WL=$S/worker_g2on1.log; restart_worker $WL $ON
  run G2on1 --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
  WL=$S/worker_g2off.log; restart_worker $WL $OFF
  run G2off --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
  WL=$S/worker_g2on2.log; restart_worker $WL $ON
  run G2on2 --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
fi
if [[ $PH == *G3* ]]; then
  [[ -z $WL || ${WL:t} != *on* ]] && { WL=$S/worker_g3on.log; restart_worker $WL $ON; }
  run G3on --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 128 --prompt-tokens 32000
fi
if [[ $PH == *G4* ]]; then
  [[ -z $WL || ${WL:t} != *on* ]] && { WL=$S/worker_g4on.log; restart_worker $WL $ON; }
  run G4on8 --prefix shared --concurrency 8 --requests-per-level 8 --max-tokens 128 --prompt-tokens 32000
  run G4on1k --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 1000
  WL=$S/worker_g4off.log; restart_worker $WL $OFF
  run G4off8 --prefix shared --concurrency 8 --requests-per-level 8 --max-tokens 128 --prompt-tokens 32000
fi
if [[ $PH == *G5* ]]; then
  WL=$S/worker_g5on.log; restart_worker $WL $ON
  gpu_free; note "phase G5 start"; state
  $PY on --out $S/cascade_g5_on.json > $S/cascade_g5_on.txt 2>&1
  note "phase G5 end rc=$?"; state; health
  $PY cmp $S/cascade_g5_on.json $REF/pcache2_v1_off.json > $S/cascade_g5_cmp.txt 2>&1
  note "G5 cmp rc=$? $(tail -1 $S/cascade_g5_cmp.txt)"
fi
if [[ $PH == *G6* ]]; then  # steady-state decode: longer answers (clean steps/s, active growth = donation)
  WL=$S/worker_g6on.log; restart_worker $WL $ON
  run G6on8k --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 1024 --prompt-tokens 8000
  run G6on32k --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 512 --prompt-tokens 32000
  WL=$S/worker_g6off.log; restart_worker $WL $OFF
  run G6off8k --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 1024 --prompt-tokens 8000
fi
# confirm run (defaults: no QWEN_CASCADE in the env -> the default, on)
DEF=(QWEN_PROMPT_CACHE=1 QWEN_CASCADE_CHECK=1)
if [[ $PH == *C1* ]]; then
  WL=$S/worker_confirm.log; restart_worker $WL $DEF
  run confirm_8k256 --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
  run confirm_8k1024 --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 1024 --prompt-tokens 8000
  run confirm_32k --prefix shared --concurrency 16 --requests-per-level 16 --max-tokens 128 --prompt-tokens 32000
  run confirm_8kdistinct --prefix distinct --concurrency 16 --requests-per-level 16 --max-tokens 256 --prompt-tokens 8000
fi
if [[ $PH == *C5* ]]; then
  WL=$S/worker_confirm_g5.log; restart_worker $WL $DEF
  gpu_free; note "phase confirm G5 start"; state
  $PY on --out $S/cascade_confirm_g5_on.json > $S/cascade_confirm_g5_on.txt 2>&1
  note "phase confirm G5 end rc=$?"; state; health
  $PY cmp $S/cascade_confirm_g5_on.json $REF/pcache2_v1_off.json > $S/cascade_confirm_g5_cmp.txt 2>&1
  note "confirm G5 cmp rc=$? $(tail -1 $S/cascade_confirm_g5_cmp.txt)"
fi
if [[ $PH == *DEFEND* ]]; then
  WL=$S/worker_default_end.log; restart_worker $WL QWEN_PROMPT_CACHE=1
fi
if [[ $PH == *END* && $PH != *DEFEND* ]]; then
  WL=$S/worker_end.log; restart_worker $WL $OFF
fi
note "=== done (worker log $WL)"

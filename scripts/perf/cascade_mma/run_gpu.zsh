#!/bin/zsh
# QWEN_CASCADE_KERNEL GPU plan (docs/perf/2026-10-24/design/cascade_mma.md, "GPU plan").
# One command, run from anywhere; everything goes to $OUT (default: this tree's
# docs/perf/2026-10-24/raw/cascade_mma), progress + machine state to $OUT/status.txt.
#
#   zsh scripts/perf/cascade_mma/run_gpu.zsh            # all phases: T M S A E
#   zsh scripts/perf/cascade_mma/run_gpu.zsh "M S"      # a subset
#
# Phases:
#   T  Metal correctness: tests/test_cascade_mma.py with CASCADE_MMA_GPU=1 (the
#      kernel vs fp32 at 9B shapes, grouped attention vs per-row) + the CPU suite
#      of the cascade files (fast) -- stop here if it fails
#   M  attention microbench (bench_attn.py, no model): kernel / fp32 cascade /
#      per-row SDPA / rattn at 8K / 32K x 8 / 16 rows, chunk sweep
#   S  in-model step A/B + decomposition (step_ab.py, loads the 9B): fp32 vs
#      kernel group steps, the "rest of the step" stub, isolated attention
#   A  exactness on the real 9B (agree_mma.py): teacher-forced top-1 / KL of
#      kernel-grouped rows vs solo, vs fp32-grouped, vs the per-row batch
#   E  end-to-end through a private worker (:50061) + gateway (:8095) of THIS
#      tree: 16 x 8K and 16 x 32K shared-prefix waves, kernel ON / OFF / ON
#      (fresh worker each), QWEN_CASCADE_CHECK=1; both are stopped at the end
#
# Needs the GPU lease: refuses to start while another worker.py / pytest /
# microbench runs (CMMA_FORCE=1 overrides). Never touches other processes.
emulate -L zsh
setopt NO_BG_NICE                                # background jobs at nice 0 (CLAUDE.md)
HERE=${0:A:h}
T=${HERE:h:h:h}                                  # this tree's root
MAIN=/Users/user/Projects/llm                    # venvs + weights + gateway binary
PY=$MAIN/src/worker/.venv/bin/python
TPY=$MAIN/.venv/bin/python
GW=$MAIN/target/release/qwen-gateway
OUT=${OUT:-$T/docs/perf/2026-10-24/raw/cascade_mma}
ST=$OUT/status.txt
MODEL=(QWEN_MODEL=$MAIN/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0)
PH=${1:-T M S A E}
mkdir -p $OUT
note() { echo "$(date '+%F %T') $*" | tee -a $ST; }
state() {
  note "state: therm $(pmset -g therm 2>&1 | grep -v '^Note' | tr '\n' ' ' | cut -c1-120) |" \
       "load $(sysctl -n vm.loadavg) | free $(( $(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}') * 16384 / 1048576 )) MiB" \
       "| swap $(sysctl -n vm.swapusage | awk '{print $6}')"
}
busy() {     # another model on the GPU: refuse
  pgrep -fl 'python worker.py|microbench|agree_cascade|cascade_step_probe|step_prof' | grep -v pgrep
}
noisy() {    # CPU tests of other agents: noise (+-20 %), accepted -- logged only
  pgrep -fl 'pytest' | grep -v pgrep | cut -c1-160
}
if [[ -z $CMMA_FORCE && -n $(busy) ]]; then
  note "REFUSED: the GPU is not free:"; busy | tee -a $ST; exit 2
fi
[[ -n $(noisy) ]] && note "WARNING: CPU pytest running (noise accepted): $(noisy | tr '\n' ' ')"
note "=== start ($PH), tree $T, git $(git -C $T rev-parse --short HEAD)$(git -C $T diff --quiet || echo '+dirty')"
state

if [[ $PH == *T* ]]; then
  note "phase T start"
  (cd $T/src/worker && CASCADE_MMA_GPU=1 $PY -m pytest -q tests/test_cascade_mma.py > $OUT/T_gpu_tests.txt 2>&1)
  rc=$?
  note "phase T Metal tests rc=$rc: $(grep -E 'passed|failed' $OUT/T_gpu_tests.txt | tail -1)"
  (cd $T/src/worker && $PY -m pytest -q tests/test_cascade_mma.py tests/test_cascade.py > $OUT/T_cpu_tests.txt 2>&1)
  rc2=$?
  note "phase T CPU tests rc=$rc2: $(grep -E 'passed|failed' $OUT/T_cpu_tests.txt | tail -1)"
  (( rc || rc2 )) && { note "STOP: tests failed"; exit 1; }
  state
fi

if [[ $PH == *M* ]]; then
  note "phase M start (bench_attn.py)"
  (cd $T/src/worker && PERF_OUT=$OUT PYTHONPATH=. $PY $HERE/bench_attn.py > $OUT/M_bench_attn.txt 2>&1)
  note "phase M end rc=$?: $(grep -c 'ms/step' $OUT/M_bench_attn.txt) lines"
  state
fi

if [[ $PH == *S* ]]; then
  note "phase S start (step_ab.py)"
  (cd $T/src/worker && env $MODEL QWEN_PROMPT_CACHE=1 QWEN_CASCADE_KERNEL=1 PERF_OUT=$OUT PYTHONPATH=. \
     $PY $HERE/step_ab.py > $OUT/S_step_ab.txt 2>&1)
  note "phase S end rc=$?: $(grep 'medians\|attention in the model\|x[0-9.]* vs fp32' $OUT/S_step_ab.txt | tr '\n' ' ' | cut -c1-600)"
  state
fi

if [[ $PH == *A* ]]; then
  note "phase A start (agree_mma.py)"
  (cd $T/src/worker && env $MODEL AGM_CONTROL_MAX_GB=${AGM_CONTROL_MAX_GB:-6} PERF_OUT=$OUT PYTHONPATH=. \
     $PY $HERE/agree_mma.py > $OUT/A_agree.txt 2>&1)
  note "phase A end rc=$?: $(grep 'vs_solo\|mma_vs_fp32' $OUT/A_agree.txt | cut -c10-120 | tr '\n' ' ' | cut -c1-900)"
  state
fi

if [[ $PH == *E* ]]; then
  WP=50061; GP=8095; WPID=; GPID=
  cleanup() {
    [[ -n $GPID ]] && kill $GPID 2>/dev/null
    [[ -n $WPID ]] && kill $WPID 2>/dev/null
    for i in {1..60}; do { [[ -z $WPID ]] || ! kill -0 $WPID 2>/dev/null; } && break; sleep 1; done
    GPID=; WPID=
  }
  trap cleanup EXIT INT TERM
  start() {  # $1 tag, rest: env
    local tag=$1; shift
    WL=$OUT/E_worker_$tag.log
    note "E: worker $tag env: $*"
    (cd $T/src/worker && exec env $MODEL QWEN_PROMPT_CACHE=1 QWEN_CASCADE=1 QWEN_CASCADE_CHECK=1 "$@" \
       $PY worker.py --addr 127.0.0.1:$WP > $WL 2>&1) &
    WPID=$!
    for i in {1..300}; do grep -q "worker serving on" $WL 2>/dev/null && break; kill -0 $WPID 2>/dev/null || break; sleep 1; done
    note "E: worker $tag up: $(grep 'cascade attention\|mma\|batching:' $WL | cut -c1-220 | tr '\n' ' ')"
    (WORKER_ADDR=127.0.0.1:$WP GATEWAY_ADDR=127.0.0.1:$GP RUST_LOG=info exec $GW > $OUT/E_gateway_$tag.log 2>&1) &
    GPID=$!
    for i in {1..60}; do curl -sf http://127.0.0.1:$GP/v1/models > /dev/null && break; sleep 1; done
  }
  wave() {  # $1 name, $2 prompt tokens, $3 max tokens, $4 concurrency (16)
    local n0=$(( $(wc -l < $WL) + 1 ))
    $TPY $T/scripts/bench_concurrency.py --url http://127.0.0.1:$GP --prefix shared --concurrency ${4:-16} \
      --requests-per-level ${4:-16} --max-tokens $3 --prompt-tokens $2 --ignore-eos on --probe 64 --cooldown 30 \
      --thinking off --worker-log $WL --out $OUT/E_$1.json > $OUT/E_$1.txt 2>&1 &
    local cp=$!
    while kill -0 $cp 2>/dev/null; do      # the 27 GiB guard of run_cascade.zsh
      a=$(tail -n +$n0 $WL | grep -o 'active=[0-9.]*GiB' | tail -1 | tr -dc '0-9.')
      if [[ -n $a ]] && (( a >= 27.0 )); then note "E: ABORT $1: active $a GiB"; kill $cp; break; fi
      sleep 3
    done
    wait $cp
    note "E: $1 rc=$? $(grep -E 'agg|tok/s' $OUT/E_$1.txt | tail -2 | tr '\n' ' ' | cut -c1-300)"
    note "E: $1 worker: $(tail -n +$n0 $WL | grep -c 'group formed') formations, $(tail -n +$n0 $WL | grep -c 'joined group') joins, $(tail -n +$n0 $WL | grep -c 'not grouped') refusals, $(tail -n +$n0 $WL | grep -ci 'traceback\|error') errors; $(tail -n +$n0 $WL | grep 'group formed' | head -1 | cut -c1-260)"
  }
  for tag in on1 off on2; do
    k=1; [[ $tag == off ]] && k=0
    start $tag QWEN_CASCADE_KERNEL=$k
    wave ${tag}_8k 8000 256
    wave ${tag}_32k 32000 128 ${E32_C:-16}
    cleanup
    state
  done
  trap - EXIT INT TERM
fi
note "=== done ($PH)"

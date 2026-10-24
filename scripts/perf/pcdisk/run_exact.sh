#!/bin/zsh
# exact_check.py on the real 9B: a worker of THIS worktree at QWEN_MAX_BATCH=1, reply KV off,
# QWEN_PROMPT_CACHE_GB=0.3 (one entry), the disk tier on; the gateway on :8090 (started if not
# running). usage: run_exact.sh [N_PAIRS=3]   (GPU lease held; ~6-10 min)
setopt NO_BG_NICE
N=${1:-3}
WT=${0:A:h:h:h:h}
MAIN=/Users/user/Projects/llm
PY=$MAIN/src/worker/.venv/bin/python
S=${PERF_OUT:-/tmp/perf}/pcdisk
R=$WT/docs/perf/2026-10-24/raw/pcdisk
mkdir -p $S $R
WLOG=$S/w_exact.log
DDIR=${PCDISK_DIR:-$S}/disk_exact      # PCDISK_DIR: where the cache files go (default $S)
pat='\.venv/bin/python3? ([^ ]*/)?(worker|sched_timer)\.py'
for pid in $(pgrep -f "$pat"); do kill -TERM $pid; done
for i in {1..90}; do pgrep -f "$pat" >/dev/null || break; sleep 1; done
pgrep -f "$pat" >/dev/null && { echo "worker did not exit"; exit 1; }
rm -rf $DDIR
echo "=== exact start $(date)" >> $WLOG
cd $WT/src/worker
env QWEN_MODEL=$MAIN/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=0.3 \
    QWEN_PROMPT_CACHE_REPLY=0 QWEN_MAX_BATCH=1 QWEN_PRESENCE_PENALTY=0 QWEN_PROMPT_CACHE_DISK=$DDIR \
    nohup $PY worker.py >> $WLOG 2>&1 &
cd $WT
for i in {1..240}; do grep -q 'worker serving on' $WLOG 2>/dev/null && break; sleep 1; done
grep -q 'worker serving on' $WLOG || { echo "worker failed to start"; tail -20 $WLOG; exit 1; }
grep 'prompt cache disk' $WLOG | tail -1
if ! curl -sf -m 5 127.0.0.1:8090/health >/dev/null; then
  (cd $MAIN && GATEWAY_ADDR=127.0.0.1:8090 GATEWAY_THINKING=off \
     GATEWAY_SERVED_MODEL_NAMES=ornith15-9b,ornith-ai/Ornith-1.5-9B-MLX-8bit RUST_LOG=info \
     nohup ./target/release/qwen-gateway >> $S/gateway_8090.log 2>&1 &)
fi
for i in {1..60}; do curl -sf -m 5 127.0.0.1:8090/health >/dev/null && break; sleep 2; done
sleep 20                                 # the gateway re-reads max_batch every 15 s
PERF_OUT=$S python3 $WT/scripts/perf/pcdisk/exact_check.py --n $N --worker-log $WLOG \
  ~/Projects/appworld/experiments/outputs/ornith9b_react_dev/tasks 2>&1 | tee $S/exact.txt
rc=${pipestatus[1]}
grep -E 'prompt cache disk' $WLOG | tail -20 >> $S/exact.txt
cp $S/exact.txt $S/pcdisk_exact.json $R/ 2>/dev/null
gzip -c $WLOG > $R/w_exact.log.gz
rm -rf $DDIR
exit $rc

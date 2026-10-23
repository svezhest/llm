#!/bin/zsh
# Disk tier of the prompt cache (QWEN_PROMPT_CACHE_DISK), AppWorld replay: one run per call.
# usage: run_replay_disk.sh NAME on|off [DURATION_MIN=10] [EPISODES=8] [PCACHE_GB=3]
# (GPU lease held; nothing else on the GPU). A fresh worker from THIS worktree's src/worker (the
# main tree's venv python, never uv: no sync), the gateway on :8090 (started if not running),
# memsampler, replay.py --own-replies, then the warm 8K probe; the disk tier's analysis
# (disk_log.py) and replay's summary in $S; raw copied to docs/perf/2026-10-24/raw/pcdisk/.
setopt NO_BG_NICE
NAME=$1; DISK=$2; DUR=${3:-10}; EP=${4:-8}; PCGB=${5:-3}
[[ $DISK == on || $DISK == off ]] || { echo "usage: $0 NAME on|off [DUR] [EP] [PCACHE_GB]"; exit 2; }
WT=${0:A:h:h:h:h}                       # the worktree (scripts/perf/pcdisk/ -> root)
MAIN=/Users/user/Projects/llm
PY=$MAIN/src/worker/.venv/bin/python
S=${PERF_OUT:-/tmp/perf}/pcdisk
R=$WT/docs/perf/2026-10-24/raw/pcdisk
mkdir -p $S $R
WLOG=$S/w_$NAME.log
DDIR=$S/disk_$NAME
pat='\.venv/bin/python3? ([^ ]*/)?(worker|sched_timer)\.py'
stop_worker() {
  for pid in $(pgrep -f "$pat"); do kill -TERM $pid; done
  for i in {1..90}; do pgrep -f "$pat" >/dev/null || break; sleep 1; done
  if pgrep -f "$pat" >/dev/null; then echo "worker did not exit"; exit 1; fi
  sleep 3
}
start_worker() {
  rm -rf $DDIR
  local extra=()
  [[ $DISK == on ]] && extra=(QWEN_PROMPT_CACHE_DISK=$DDIR)
  echo "=== $NAME start $(date) disk=$DISK pcache=$PCGB GB dur=$DUR ep=$EP worker $WT/src/worker" >> $WLOG
  cd $WT/src/worker
  env QWEN_MODEL=$MAIN/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 \
      QWEN_PROMPT_CACHE_GB=$PCGB QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 \
      QWEN_PROMPT_CACHE_REPLY=1 $extra PYTHONPATH=. \
      nohup $PY ../../scripts/perf/agentic/sched_timer.py >> $WLOG 2>&1 &
  cd $WT
  for i in {1..240}; do grep -q 'worker serving on' $WLOG 2>/dev/null && break; sleep 1; done
  grep -q 'worker serving on' $WLOG || { echo "worker failed to start"; tail -20 $WLOG; exit 1; }
  grep -E 'prompt cache disk' $WLOG | head -2
  if ! curl -sf -m 5 127.0.0.1:8090/health >/dev/null; then
    (cd $MAIN && GATEWAY_ADDR=127.0.0.1:8090 GATEWAY_THINKING=off \
       GATEWAY_SERVED_MODEL_NAMES=ornith15-9b,ornith-ai/Ornith-1.5-9B-MLX-8bit RUST_LOG=info \
       nohup ./target/release/qwen-gateway >> $S/gateway_8090.log 2>&1 &)
  fi
  for i in {1..60}; do curl -sf -m 5 127.0.0.1:8090/health >/dev/null && break; sleep 2; done
  sleep 20                               # the gateway re-reads max_batch every 15 s
}
probe() {  # $1 = tag
  python3 $WT/scripts/bench_concurrency.py --url http://127.0.0.1:8090 --concurrency 1 \
    --requests-per-level 1 --prompt-tokens 8000 --max-tokens 128 --worker-log $WLOG \
    --out $S/probe_$1.json > $S/probe_$1.txt 2>&1
}
echo "=== $NAME: disk $DISK, pcache $PCGB GB, $EP episodes, $DUR min ($(date))" | tee -a $S/driver.log
stop_worker
start_worker
ps -o pid,nice,command -p $(pgrep -f "$pat") | tee -a $S/driver.log
T0=$(date +%H:%M:%S)
python3 $WT/scripts/perf/agentic/memsampler.py --out $S/mem_$NAME.jsonl --duration-min $(( DUR + 6 )) \
  --pattern "$pat" > /dev/null 2>&1 &
MS=$!
sleep 5
python3 $WT/scripts/perf/agentic/replay.py --own-replies --ignore-eos off --episodes $EP --duration-min $DUR \
  --tag $NAME --worker-log $WLOG --mem-log $S/mem_$NAME.jsonl --out $S/replay_$NAME.json \
  > $S/replay_$NAME.txt 2> $S/replay_$NAME.err
echo "=== $NAME replay done $(date)" >> $WLOG
probe ${NAME}_warm
sleep 3; kill $MS 2>/dev/null
python3 $WT/scripts/perf/pcdisk/disk_log.py $WLOG --since $T0 --json $S/disk_$NAME.json > $S/disk_$NAME.txt 2>&1
du -sh $DDIR 2>/dev/null | tee -a $S/driver.log
rm -rf $DDIR
for f in replay_$NAME.json replay_$NAME.txt mem_$NAME.jsonl probe_${NAME}_warm.txt disk_$NAME.json disk_$NAME.txt; do
  cp $S/$f $R/ 2>/dev/null
done
gzip -c $WLOG > $R/w_$NAME.log.gz
echo "=== $NAME done ($(date))" | tee -a $S/driver.log
grep -E '^replay|^client|^split' $S/replay_$NAME.txt | tee -a $S/driver.log
cat $S/disk_$NAME.txt | tee -a $S/driver.log

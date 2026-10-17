#!/bin/zsh
# S1 (2026-10-17, replay_batch1.md): AppWorld replay, own replies, one QWEN_MAX_BATCH per run.
# usage: [WORKER_DIR=frozen src/worker copy] run_replay_batch.sh NAME MAX_BATCH [DURATION_MIN=10] [EPISODES=8]
# (GPU lock must be held)
# fresh worker (sched_timer: time split + per-slab lines) -> cool probe -> fresh worker -> replay +
# memsampler -> warm probe -> raw copied to docs/perf/2026-10-17/raw/replay_b1_<NAME>.*
setopt NO_BG_NICE
NAME=$1; MAXB=$2; DUR=${3:-10}; EP=${4:-8}
S=${PERF_OUT:-/tmp/scratch/b1}
mkdir -p $S
REPO=/Users/user/Projects/llm
R=$REPO/docs/perf/2026-10-17/raw
WLOG=$S/w_$NAME.log; PLOG=$S/w_${NAME}_probe.log
pat='\.venv/bin/python3 ([^ ]*/)?(worker|sched_timer)\.py'
stop_worker() {
  for pid in $(pgrep -f "$pat"); do kill -TERM $pid; done
  for i in {1..90}; do pgrep -f "$pat" >/dev/null || break; sleep 1; done
  if pgrep -f "$pat" >/dev/null; then echo "worker did not exit"; exit 1; fi
  sleep 3
}
start_worker() {   # $1 = log
  echo "=== $NAME start $(date) MAX_BATCH=$MAXB dur=$DUR ep=$EP worker ${WORKER_DIR:-src/worker}" >> $1
  cd $REPO/src/worker
  env QWEN_MODEL=/Users/user/Projects/llm/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 \
      QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 QWEN_PROMPT_CACHE_REPLY=1 \
      QWEN_MAX_BATCH=$MAXB PYTHONPATH=. SCHED_WORKER_DIR=${WORKER_DIR:-} nohup uv run python ../../scripts/perf/agentic/sched_timer.py >> $1 2>&1 &
  cd $REPO
  for i in {1..240}; do grep -q 'worker serving on' $1 2>/dev/null && break; sleep 1; done
  grep -q 'worker serving on' $1 || { echo "worker failed to start"; tail -20 $1; exit 1; }
  for i in {1..60}; do curl -sf -m 5 127.0.0.1:8090/health >/dev/null && break; sleep 2; done
  sleep 20   # the gateway re-reads max_batch every 15 s
  for i in {1..20}; do
    cap=$(curl -s 127.0.0.1:8090/metrics | awk '/^qwen_inflight_capacity/{print $2}')
    [[ "$cap" == "$MAXB" ]] && break; sleep 3
  done
  echo "gateway qwen_inflight_capacity=$cap (max_batch $MAXB)" | tee -a $1
}
probe() {  # $1 = tag, $2 = worker log
  python3 scripts/bench_concurrency.py --url http://127.0.0.1:8090 --concurrency 1 --requests-per-level 1 \
    --prompt-tokens 8000 --max-tokens 128 --worker-log $2 --out $S/probe_$1.json > $S/probe_$1.txt 2>&1
}
echo "=== $NAME: MAX_BATCH=$MAXB episodes $EP duration $DUR min ($(date))" | tee -a $S/driver.log
stop_worker
start_worker $PLOG
probe ${NAME}_cool $PLOG
stop_worker
start_worker $WLOG
grep -E 'batching:|serving on' $WLOG | tail -2 | cut -c1-400 | tee -a $S/driver.log
ps -o pid,nice,command -p $(pgrep -f "$pat") | tee -a $S/driver.log
python3 scripts/perf/agentic/memsampler.py --out $S/mem_$NAME.jsonl --duration-min $(( DUR + 6 )) --pattern "$pat" > /dev/null 2>&1 &
MS=$!
sleep 5
python3 scripts/perf/agentic/replay.py --own-replies --ignore-eos off --episodes $EP --duration-min $DUR --tag $NAME \
  --worker-log $WLOG --mem-log $S/mem_$NAME.jsonl --out $S/replay_$NAME.json > $S/replay_$NAME.txt 2> $S/replay_$NAME.err
echo "=== $NAME replay done $(date)" >> $WLOG
probe ${NAME}_warm $WLOG
sleep 3; kill $MS 2>/dev/null
cp $S/replay_$NAME.json $R/replay_b1_$NAME.json; cp $S/replay_$NAME.txt $R/replay_b1_$NAME.txt
cp $S/mem_$NAME.jsonl $R/replay_b1_mem_$NAME.jsonl
cp $S/probe_${NAME}_cool.txt $R/replay_b1_probe_${NAME}_cool.txt; cp $S/probe_${NAME}_warm.txt $R/replay_b1_probe_${NAME}_warm.txt
cp $WLOG $R/replay_b1_w_$NAME.log
echo "=== $NAME done ($(date))" | tee -a $S/driver.log
grep -E '^replay|^client|^split' $S/replay_$NAME.txt | tee -a $S/driver.log

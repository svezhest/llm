#!/bin/zsh
# gate 4: AppWorld replay with own replies, fresh worker (under sched_timer) per run, standard env
# usage: reply_kv_replay.sh NAME REPLY(0|1) [DURATION_MIN=10] [EPISODES=8]   (GPU lock must be held)
setopt NO_BG_NICE
NAME=$1; REPLY=$2; DUR=${3:-10}; EP=${4:-8}
S=/tmp/scratch/rk
REPO=/Users/user/Projects/llm
WLOG=$S/w_replay_$NAME.log
pat='\.venv/bin/python3 ([^ ]*/)?(worker|sched_timer)\.py'
for pid in $(pgrep -f "$pat"); do kill -TERM $pid; done
for i in {1..60}; do pgrep -f "$pat" >/dev/null || break; sleep 1; done
if pgrep -f "$pat" >/dev/null; then echo "worker did not exit"; exit 1; fi
sleep 3
echo "=== $NAME start $(date) REPLY=$REPLY dur=$DUR ep=$EP" >> $WLOG
cd $REPO/src/worker
env QWEN_MODEL=/Users/user/Projects/llm/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 \
    QWEN_PROMPT_CACHE_GB=6 QWEN_BATCH_RESERVE_TOKENS=1024 QWEN_PRESENCE_PENALTY=0 QWEN_PROMPT_CACHE_REPLY=$REPLY \
    PYTHONPATH=. nohup uv run python ../../scripts/perf/agentic/sched_timer.py >> $WLOG 2>&1 &
for i in {1..240}; do grep -q 'worker serving on' $WLOG 2>/dev/null && break; sleep 1; done
grep -q 'worker serving on' $WLOG || { echo "worker failed to start"; tail -20 $WLOG; exit 1; }
for i in {1..60}; do curl -sf -m 5 127.0.0.1:8090/health >/dev/null && break; sleep 2; done
sleep 20
cd $REPO
ps -o pid,nice,command -p $(pgrep -f "$pat")
python3 scripts/perf/agentic/memsampler.py --out $S/mem_$NAME.jsonl --duration-min $(( DUR + 4 )) --pattern "$pat" > /dev/null 2>&1 &
MS=$!
sleep 5
python3 scripts/perf/agentic/replay.py --own-replies --ignore-eos off --episodes $EP --duration-min $DUR --tag $NAME \
  --worker-log $WLOG --mem-log $S/mem_$NAME.jsonl --out $S/replay_$NAME.json > $S/replay_$NAME.txt 2> $S/replay_$NAME.err
kill $MS 2>/dev/null
echo "=== $NAME done $(date)" >> $WLOG
tail -40 $S/replay_$NAME.txt | head -60

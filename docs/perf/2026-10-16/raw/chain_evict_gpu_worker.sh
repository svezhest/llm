#!/bin/zsh
# usage: worker.sh LOG ENV=VAL ...   stops the :50051 worker (SIGTERM), starts a new one, waits for serving + gateway
setopt NO_BG_NICE
LOG=$1; shift
pat='\.venv/bin/python3 ([^ ]*/)?worker\.py'
for pid in $(pgrep -f "$pat"); do kill -TERM $pid; done
for i in {1..60}; do pgrep -f "$pat" >/dev/null || break; sleep 1; done
if pgrep -f "$pat" >/dev/null; then echo "worker did not exit"; exit 1; fi
sleep 3
cd /Users/user/Projects/llm/src/worker
env QWEN_MODEL=/Users/user/Projects/llm/weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 QWEN_PRESENCE_PENALTY=0 \
    QWEN_BATCH_RESERVE_TOKENS=1024 "$@" nohup uv run python worker.py >> $LOG 2>&1 &
for i in {1..240}; do grep -q 'worker serving on' $LOG 2>/dev/null && break; sleep 1; done
grep -q 'worker serving on' $LOG || { echo "worker failed to start"; tail -20 $LOG; exit 1; }
for i in {1..60}; do curl -sf -m 5 127.0.0.1:8090/health >/dev/null && break; sleep 2; done
sleep 20   # gateway re-reads max_batch every 15 s
ps -o pid,nice,command -p $(pgrep -f "$pat")
grep -E 'batching|prompt cache|cascade' $LOG | tail -4 | cut -c1-300

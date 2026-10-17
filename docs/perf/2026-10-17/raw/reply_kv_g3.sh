#!/bin/zsh
# gate 3: 8 conversations x 6 turns closed loop, 64-token ignore_eos answers; fresh worker per run
# usage: reply_kv_g3.sh NAME REPLY(0|1)
setopt NO_BG_NICE
S=/tmp/scratch/rk
R=/Users/user/Projects/llm/docs/perf/2026-10-17/raw
zsh $R/reply_kv_worker.sh $S/w_g3_$1.log QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=6 QWEN_PROMPT_CACHE_REPLY=$2 > /dev/null 2>&1 || exit 1
cd $R && python3 reply_kv_drive.py convs reply_kv_corpus.json g8 $S/g3_$1.json g3$1 6 64 ignore_eos | tail -1

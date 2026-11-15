#!/bin/zsh
# The GPU runs of docs/perf/2026-11-16/verify_att2.md: vatt_gate.sh micro | micro9 | sweep2 | kl [N] | short | long [N] [max_ctx] [rounds]
setopt NO_BG_NICE
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd $ROOT/src/worker
R=$ROOT/docs/perf/2026-11-16/raw/verify_att2; mkdir -p $R $R/sweep2
export QWEN_MODEL=$ROOT/weights/qwen38-27b-mlx-4bit QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 \
  QWEN_WIRED_LIMIT_GB=27 QWEN_CACHE_LIMIT_GB=1 QWEN_SDPA_FUSED_MIN_KV=8192 QWEN_PREFILL_BUDGET=16777216 \
  QWEN_VERIFY_ATT=2 QWEN_DECODE_ATT=8192 QWEN_PRESENCE_PENALTY=0
case "$1" in
micro)
  for N in 2048 8192 32768 61440 81920; do
    .venv/bin/python ../../scripts/perf/dflash2/verify_att_bench.py $N 8 --only single,vatt1,vec,vatt --json $R/micro_T8_$N.json 2>&1
  done
  for T in 2 4 16; do
    .venv/bin/python ../../scripts/perf/dflash2/verify_att_bench.py 61440 $T --only vec,vatt --json $R/micro_T${T}_61440.json 2>&1
  done ;;
sweep2)
  for N in 61440 8192; do for sp in 0 1; do for tile in 16x16 16x8 16x32 24x24; do
    echo "== N=$N split=$sp $tile"
    VATT_SPLIT=$sp VATT_TILE=$tile VB_ROUNDS=5 .venv/bin/python ../../scripts/perf/dflash2/verify_att_bench.py $N 8 --only vatt       --json $R/sweep2/${tile}_s${sp}_$N.json 2>&1 | grep -E "^vatt|FAILED|Error"
  done; done; done ;;
micro9)
  for N in 2048 8192 32768 61440; do
    VB_HQ=16 VB_HKV=4 .venv/bin/python ../../scripts/perf/dflash2/verify_att_bench.py $N 8 --only single,vatt1,vec,vatt --json $R/micro9b_T8_$N.json 2>&1
  done ;;
kl)
  .venv/bin/python ../../scripts/perf/dflash2/vatt_kl.py --tokens ${2:-32768} --blocks 64 --decode 128 --json $R/kl_${2:-32768}.json 2>&1 | grep -v "^20.*INFO" ;;
short)
  QWEN_DRAFT=$ROOT/weights/qwen38-27b-dflash2   .venv/bin/python ../../scripts/perf/dflash2/ab27.py --rounds 3 --variants plain,m8/8,m8/8:va     --ident m8/8,m8/8:va --json $R/ab27_2k.json 2>&1 | grep -Ev "^20.* INFO" ;;
long)
  QWEN_DRAFT=$ROOT/weights/qwen38-27b-dflash2 QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_MOVE=1 QWEN_PROMPT_CACHE_GB=6 \
  QWEN_PROMPT_CACHE_CKPTS=4 QWEN_MAX_CONTEXT=${3:-90112} \
  .venv/bin/python ../../scripts/perf/dflash2/long_ctx.py --tokens ${2:-60000} --turns 1 --no-thinking \
    --ab ${4:-3} --ab-mode '' --ab-variants plain,plain+d,vector,vatt,vatt+d --json $R/long_${2:-60000}.json 2>&1 \
    | grep -Ev "^20.* INFO .*(prompt cache|dflash:)" ;;
esac

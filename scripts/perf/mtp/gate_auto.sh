#!/bin/zsh
# The QWEN_DRAFT=auto gate on the 27B (oQ4e), one command per run, one JSON per run.
#   WEIGHTS=<dir with qwen38-27b-mlx-oq4e, qwen38-27b-dflash2, qwen38-27b-mtp> PERF_OUT=<dir> \
#     scripts/perf/mtp/gate_auto.sh <run>            (from the repo root, GPU otherwise idle)
# runs:
#   gate     10 cells (greedy, sampling T 0.6 / top_p 0.95 / top_k 20) x (code-2k, en-2k, ru-2k, short en /
#            ru prose) x plain, dflash, mtp0+q, auto, 2 interleaved rounds -> $PERF_OUT/mtp/gate_auto.json
#            + the tables (gate_table.py): auto >= 0.97 x the best of the three per cell, arms per cell.
#            GATE_MODES=sampling (or greedy): half of it (-> gate_auto_<mode>.json; gate_table.py takes both)
#   floor    the chooser's plain arm alone (pplain) vs the plain loop -> floor_auto.json
#   dump     both drafters' drafts at every position of each cell's answer (plain steps forced) ->
#            dump_auto.json, then the chooser's replay on the CPU (chooser_sim.py)
#   q8       QWEN_KV_BITS=8, 8K context (ru-8k-0 greedy; Q8_PROMPTS=code-8k-0,ru-8k-0 GATE_MODES=sampling,greedy
#            for more cells: ~75 s of prefill per request), the same four variants -> gate_auto_q8.json + tables
# The drafters are found by QWEN_DRAFT=auto next to the model dir (the load log says which);
# QWEN_DRAFT_DFLASH / QWEN_DRAFT_MTP name them explicitly.
setopt NO_BG_NICE
set -e
run=${1:?run: gate | floor | dump | q8}
: ${WEIGHTS:?} ${PERF_OUT:?}
root=${0:A:h:h:h:h}
out=$PERF_OUT/mtp
mkdir -p $out
cd $root/src/worker
export QWEN_MODEL=$WEIGHTS/qwen38-27b-mlx-oq4e QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 QWEN_MRQMM5=1 QWEN_MRQMM6=1 \
  QWEN_KV_BITS=0 QWEN_MAX_BATCH=1 QWEN_WIRED_LIMIT_GB=27 QWEN_PRESENCE_PENALTY=0 QWEN_DRAFT=auto \
  BENCH27=$root/scripts/perf/bench27 PYTHONPATH=. HF_HUB_OFFLINE=1
ab=(.venv/bin/python ../../scripts/perf/mtp/ab_mtp.py)
cells=(--prompts code-2k-0,en-2k-0,ru-2k-0,short-en,short-ru)
case $run in
  gate)
    modes=${GATE_MODES:-sampling,greedy}
    json=$out/gate_auto${GATE_MODES:+_$GATE_MODES}.json
    $ab --modes $modes $cells --variants plain,dflash,mtp0+q,auto --rounds ${GATE_ROUNDS:-2} \
      --warm-tokens 64 --json $json 2>${json:r}.log | tee ${json:r}.txt
    python3 ../../scripts/perf/mtp/gate_table.py $json | tee -a ${json:r}.txt ;;
  floor)
    json=$out/floor_auto.json
    $ab --modes sampling,greedy --prompts en-2k-0,short-ru --variants plain,pplain --rounds 2 \
      --warm-tokens 64 --json $json 2>${json:r}.log | tee ${json:r}.txt
    python3 ../../scripts/perf/mtp/gate_table.py --auto pplain $json | tee -a ${json:r}.txt ;;
  dump)
    json=$out/dump_auto.json
    $ab --modes sampling,greedy $cells --dump --skip-speed --warm-tokens 64 --json $json \
      2>${json:r}.log | tee ${json:r}.txt
    QWEN_DRAFT= .venv/bin/python ../../scripts/perf/mtp/chooser_sim.py $json | tee $out/sim_auto.txt ;;
  q8)
    export QWEN_KV_BITS=8
    json=$out/gate_auto_q8.json
    $ab --modes ${GATE_MODES:-greedy} --prompts ${Q8_PROMPTS:-ru-8k-0} --variants plain,dflash,mtp0+q,auto \
      --rounds 2 --warm-tokens 64 --json $json 2>${json:r}.log | tee ${json:r}.txt
    python3 ../../scripts/perf/mtp/gate_table.py $json | tee -a ${json:r}.txt ;;
  *) echo "unknown run $run"; exit 2 ;;
esac

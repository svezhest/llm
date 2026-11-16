#!/usr/bin/env zsh
# GPU gate for the mixed-precision 27B (weights/qwen38-27b-mlx-oq4e: oQ4e, base 4
# bits, 5 / 6 / 8-bit per-module overrides) against the uniform 4-bit one
# (weights/qwen38-27b-mlx-4bit): the 6-bit multi-row kernel (QWEN_MRQMM6), the GDN
# in-projection fused by width groups (QWEN_GDN_FUSE), plain and DFlash speed.
# Design: docs/perf/2026-11-15/design/oq4e_speed.md. Run from the repo root on an
# idle GPU (docs/measuring.md; `setopt NO_BG_NICE` if backgrounded).
# Every step writes to $PERF_OUT/oq4e/. Steps 1-3 load no model (tensors only);
# 4-8 load one 27B per process (wired-limit dance first:
#   sudo sysctl iogpu.wired_limit_mb=28672).
#
#  1. kernel unit tests (synthetic, ~1 min): tests/test_mrqmm.py all pass (6-bit:
#     check_close at 5120 x 1024 / 8192, every in-group offset exact, every config
#     with tails, the self-test at 248320 x 5120, rows independent of the row
#     count; the mixed tiny model: width groups bit-identical to the unfused model
#     at B <= 2 ON THE GPU, close above).
#  2. real weights (~1.3 GB of tensors): test_real_oq4e_weights (6-bit lm_head and
#     k / v, the qkv+z 4-bit and z+b+a 5-bit groups: self-test + check_close).
#  3. microbench, both checkpoints in ONE process (no model, ~9 GB of tensors):
#     oq4e_bench.py. Read:
#       qmv:    GB/s per (kind, width) at 1 row (H3/H4: 5 / 6-bit within ~10 % of
#               4-bit's GB/s?), "forward linears, 1 row" oq4e/none vs oq4e/groups vs
#               u4/groups (the drift-free plain-decode prediction: bytes ratio
#               14.85 / 14.09 = 1.054 + launches), u4 unfused / fused (H1 on the
#               uniform model).
#       verify: per-forward ms of a T=8 block, oq4e/none (today) vs oq4e/groups
#               (QWEN_MRQMM6 + groups) vs u4/groups; lm_head and drafter_head parts.
#       lmhead: 6-bit singleton8 vs mrqmm8 (expect ~25-30 ms -> ~4-5 ms); the
#               sweep: is 4x8 (the 4/5-bit choice, mrqmm._config) still the best at
#               8 rows? 8-bit re-quantized head: qmv1 vs 6-bit qmv1 (the cost the
#               rejected option would add to every plain token).
#       acc:    every width's rms rel <= quantized_matmul's at the same rows and
#               ~2-3e-4; rows_equal_across_M true everywhere.
#  4. plain decode A/B of the fusion in one process at B = 1 (step_prof --ab
#     base,nofuse): oq4e base <= nofuse (expect -2..4 %); the same on the uniform
#     model (its one-group fusion, unchanged code path; expect the old -2 %).
#  5. plain decode oq4e vs u4 across processes, interleaved A B A B A B with a
#     cooldown, each with its own bandwidth probe (decode_probe.py): compare
#     tok/s per 100 GB/s of probe. Expected without drift: oq4e / u4 ~0.90-0.95
#     (bytes + 6-bit head); the 2026-11-15 measurement said 0.68 (11.7 vs 17.2).
#  6. DFlash on oq4e, in-process ratios (ab27 plain,m8/8,e0/8): QWEN_MRQMM6=1 vs
#     QWEN_MRQMM6=0 (two processes; compare m8/8 / plain within each). Expect the
#     round 259-274 ms -> ~160-180 ms and m8/8 / plain from x1.45 to >= x2.
#     Also look at drafts/round (3.57 on 2026-11-15 vs 4.8 on the 4-bit): same
#     tasks for both models here (ab27's corpus), so a gap that remains is the
#     model, not the kernels.
#  7. identity: on oq4e plainnf == plain (the width-group fusion at B = 1: bit
#     for bit, 16/16) and m8/8 vs plain (verify through mrqmm: the batched-decode
#     class, 4-bit 27B reference: >= 14/16 without thinking, divergences at
#     near-ties: diverge_margin.py); on u4 plainnf == plain 16/16 (regression of
#     the uniform fusion).
#  8. load logs (from steps 4-7): "mrqmm 6-bit g64 self-test passed",
#     "GDN in-projections fused by width: 30 x [qkv+z 4-bit, b+a 5-bit]; 15 x
#     [z+b+a 5-bit, qkv alone]; 2 x [qkv+z 4-bit, b+a 8-bit]; 1 x [qkv+z 4-bit, b
#     alone, a alone]", "GDN decode in-projections fused (QWEN_GDN_FUSE): 48
#     layers", "(4/5/6-bit, ...)" in the mrqmm coverage line.
#
# Usage: scripts/perf/dflash2/gate_oq4e.sh [step ...]   (default: 1 2 3 4 5 6 7 8)
set -eu
ROOT=${0:A:h}/../../..
W=$ROOT/src/worker
PY=${PY:-$W/.venv/bin/python}
OUT=${PERF_OUT:-/tmp/perf}/oq4e
A=${A:-$ROOT/weights/qwen38-27b-mlx-oq4e}
B=${B:-$ROOT/weights/qwen38-27b-mlx-4bit}
DRAFT=${DRAFT:-$ROOT/weights/qwen38-27b-dflash2}
PAIRS=${PAIRS:-3}
COOL=${COOL:-30}
mkdir -p $OUT
cd $W
if (( $# )); then steps=("$@"); else steps=(1 2 3 4 5 6 7 8); fi
RUN27=(QWEN_KV_BITS=0 QWEN_WIRED_LIMIT_GB=27 QWEN_PROMPT_CACHE=0 QWEN_TEXT_ONLY=1
       QWEN_MRQMM4=1 QWEN_MRQMM5=1 QWEN_PRESENCE_PENALTY=0 PYTHONPATH=.)

for s in $steps; do
  case $s in
  1) $PY -m pytest -q tests/test_mrqmm.py 2>&1 | tail -3 ;;
  2) MRQMM_REAL_WEIGHTS=1 MRQMM_WEIGHTS_OQ4E=$A $PY -m pytest -q tests/test_mrqmm.py \
       -k test_real_oq4e_weights 2>&1 | tail -3 ;;
  3) PYTHONPATH=. $PY ../../scripts/perf/dflash2/oq4e_bench.py --a $A --b $B --rounds 5 \
       --out $OUT/oq4e_bench.json 2>&1 | tee $OUT/oq4e_bench.log ;;
  4) for M in $A $B; do
       n=${M:t}
       env $RUN27 QWEN_MRQMM6=1 QWEN_MODEL=$M QWEN_MAX_BATCH=1 $PY ../../scripts/perf/prof/step_prof.py \
         --B 1 --ctx 1k --gen 300 --ab base,nofuse --ab-steps 32 --json $OUT/step_fuse_$n.json 2>&1 \
         | tee $OUT/step_fuse_$n.log | grep -E "fused|self-test|mrqmm:|base|nofuse"
       sleep $COOL
     done ;;
  5) for i in $(seq 1 $PAIRS); do
       for M in $A $B; do
         n=${M:t}
         env $RUN27 QWEN_MRQMM6=1 QWEN_MODEL=$M QWEN_MAX_BATCH=1 $PY \
           ../../scripts/perf/dflash2/decode_probe.py --tokens 2000 --requests 4 \
           --json $OUT/probe_${n}_$i.json 2>&1 | tee $OUT/probe_${n}_$i.log | grep -E "SUMMARY|^r[0-9]"
         sleep $COOL
       done
     done
     $PY - $OUT ${A:t} ${B:t} <<'EOF'
import glob, json, statistics, sys
out, a, b = sys.argv[1:]
def load(n):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{out}/probe_{n}_*.json"))]
ra, rb = load(a), load(b)
for n, r in ((a, ra), (b, rb)):
    print(f"{n}: tps {[round(x['tps'], 2) for x in r]}  probe GB/s {[round(x['probe_gbps']) for x in r]}  "
          f"tps/100GBps {[round(x['tps_per_100gbps'], 3) for x in r]}")
pa = [x["tps_per_100gbps"] for x in ra]
pb = [x["tps_per_100gbps"] for x in rb]
print(f"{a} / {b}: tps/100GBps medians {statistics.median(pa) / statistics.median(pb):.3f}; "
      f"raw tps medians {statistics.median(x['tps'] for x in ra) / statistics.median(x['tps'] for x in rb):.3f}")
EOF
     ;;
  6) for K6 in 1 0; do
       env $RUN27 QWEN_MRQMM6=$K6 QWEN_MODEL=$A QWEN_DRAFT=$DRAFT $PY \
         ../../scripts/perf/dflash2/ab27.py --rounds 3 --variants plain,m8/8,e0/8 --skip-ident \
         --json $OUT/ab27_oq4e_mrqmm6_$K6.json 2>&1 | tee $OUT/ab27_oq4e_mrqmm6_$K6.log | grep -vE "INFO|DEBUG"
       sleep $COOL
     done ;;
  7) env $RUN27 QWEN_MRQMM6=1 QWEN_MODEL=$A QWEN_DRAFT=$DRAFT $PY \
       ../../scripts/perf/dflash2/ab27.py --skip-speed --ident plainnf,m8/8 \
       --json $OUT/ab27_oq4e_ident.json 2>&1 | tee $OUT/ab27_oq4e_ident.log | grep -vE "INFO|DEBUG"
     sleep $COOL
     env $RUN27 QWEN_MODEL=$B QWEN_DRAFT=$DRAFT $PY \
       ../../scripts/perf/dflash2/ab27.py --skip-speed --ident plainnf \
       --json $OUT/ab27_u4_ident.json 2>&1 | tee $OUT/ab27_u4_ident.log | grep -vE "INFO|DEBUG" ;;
  8) grep -hE "self-test|fused by width|in-projections fused|multi-row kernel|DFlash through mrqmm" \
       $OUT/*.log | sort | uniq -c ;;
  *) echo "unknown step $s" >&2; exit 2 ;;
  esac
done

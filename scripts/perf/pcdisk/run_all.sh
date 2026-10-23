#!/bin/zsh
# The disk tier's GPU plan in one command (docs/perf/2026-10-24/design/pcdisk.md, "GPU plan"):
#   1. io_bench.py: write / read of 1 and 2.5 GB entries, 3 interleaved rounds, bit-exact check;
#   2. run_exact.sh: the real 9B, a hit through the disk == cold (teacher-forced logprobs, greedy);
#   3. AppWorld replay, QWEN_PROMPT_CACHE_GB=3, 8 episodes x 10 min: off, on, off, on (interleaved);
#   4. summary.md: compare.py over the four runs + the disk tier's numbers.
# usage: run_all.sh [DURATION_MIN=10] [EPISODES=8] [PCACHE_GB=3]   (GPU lease held; ~65 min)
setopt NO_BG_NICE
DUR=${1:-10}; EP=${2:-8}; PCGB=${3:-3}
WT=${0:A:h:h:h:h}
S=${PERF_OUT:-/tmp/perf}/pcdisk
R=$WT/docs/perf/2026-10-24/raw/pcdisk
mkdir -p $S $R
PY=/Users/user/Projects/llm/src/worker/.venv/bin/python
pat='\.venv/bin/python3? ([^ ]*/)?(worker|sched_timer)\.py'
if pgrep -f "$pat" >/dev/null; then       # the bench needs the GPU and the memory to itself
  for pid in $(pgrep -f "$pat"); do kill -TERM $pid; done
  for i in {1..90}; do pgrep -f "$pat" >/dev/null || break; sleep 1; done
fi
echo "=== io_bench $(date)" | tee -a $S/driver.log
(cd $WT/src/worker && $PY ../../scripts/perf/pcdisk/io_bench.py --sizes-gb 1,2.5 --rounds 3 \
   --dir $S/io_dir --out $S/io_bench.json) 2>&1 | tee $S/io_bench.txt
cp $S/io_bench.json $S/io_bench.txt $R/
echo "=== exact $(date)" | tee -a $S/driver.log
$WT/scripts/perf/pcdisk/run_exact.sh 3 || echo "EXACT CHECK FAILED" | tee -a $S/driver.log
for n in off_a on_a off_b on_b; do
  $WT/scripts/perf/pcdisk/run_replay_disk.sh $n ${n%_*} $DUR $EP $PCGB
done
{
  echo "# pcdisk GPU run $(date)"
  echo; echo "## io_bench"; sed -n '/medians/,$p' $S/io_bench.txt
  echo; echo "## exact check (hit through the disk == cold)"; grep -E '^\{|pairs' $S/exact.txt
  echo; echo "## replay (QWEN_PROMPT_CACHE_GB=$PCGB, $EP episodes x $DUR min)"
  python3 $WT/scripts/perf/agentic/compare.py $S/replay_off_a.json $S/replay_on_a.json \
    $S/replay_off_b.json $S/replay_on_b.json
  for n in off_a on_a off_b on_b; do echo; echo "### $n"; cat $S/disk_$n.txt; done
} > $S/summary.md 2>&1
cp $S/summary.md $R/
cat $S/summary.md

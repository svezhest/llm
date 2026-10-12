#!/bin/zsh
while [ -z "$(grep '=== done' ${PERF_OUT:-/tmp/perf}/results/final_status.txt 2>/dev/null)" ]; do
  echo "$(date '+%T') $(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*\|"Renderer Utilization %"=[0-9]*\|"Tiler Utilization %"=[0-9]*' | sed 's/"//g;s/ Utilization %//' | tr '\n' ' ') ws=$(ps -o %cpu= -p $(pgrep -x WindowServer))"
  sleep 3
done

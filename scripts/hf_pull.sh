#!/bin/zsh
# Resumable HF repo pull: aria2c -c per file, sha256 from the tree API, infinite retries.
# Usage: scripts/hf_pull.sh <repo_id> <dest_dir>
# Re-run the same command after any interruption; finished files are skipped by checksum.
set -u
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH
REPO=$1; DEST=$2
export HF_HUB_DISABLE_XET=1
# aria2c ignores upper-case HTTPS_PROXY; direct access to HF is unreliable here
PROXY=${HTTPS_PROXY:-${https_proxy:-}}
px=(); [ -n "$PROXY" ] && px=(--all-proxy="$PROXY")
TOKEN=$(cat ~/.cache/huggingface/token)
mkdir -p "$DEST"
cd "$DEST"
LIST=$(curl -sL "https://huggingface.co/api/models/$REPO/tree/main" | python3 -c '
import json,sys
for f in json.load(sys.stdin):
    if f["type"]!="file": continue
    oid=(f.get("lfs") or {}).get("oid","")
    print(f["path"], f["size"], oid)')
while read -r fpath size oid; do
  [ -z "$fpath" ] && continue
  if [ -f "$fpath" ] && [ ! -f "$fpath.aria2" ]; then
    have=$(stat -f%z "$fpath")
    if [ "$have" = "$size" ]; then echo "skip $fpath"; continue; fi
  fi
  echo "get  $fpath ($size)"
  cs=(); [ -n "$oid" ] && cs=(--check-integrity=true --checksum=sha-256=$oid)
  until aria2c -c -x6 -s6 -k1M -j1 "${px[@]}" --max-tries=0 --retry-wait=5 --timeout=30 --connect-timeout=20 --lowest-speed-limit=32K --show-console-readout=false \
      --summary-interval=60 --console-log-level=warn "${cs[@]}" \
      --header="Authorization: Bearer $TOKEN" -o "$fpath" \
      "https://huggingface.co/$REPO/resolve/main/$fpath"; do
    echo "retry $fpath in 15s"; sleep 15
  done
done <<< "$LIST"
echo "ALL DONE $(date)"

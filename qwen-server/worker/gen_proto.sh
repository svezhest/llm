#!/usr/bin/env bash
# Regenerate worker_pb2*.py from ../proto/worker.proto.
# Run with: uv run --group proto bash gen_proto.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python -m grpc_tools.protoc \
  -I"$HERE/../proto" \
  --python_out="$HERE" \
  --grpc_python_out="$HERE" \
  "$HERE/../proto/worker.proto"
echo "generated: worker_pb2.py worker_pb2_grpc.py"

"""Prefill time + peak memory with the current MLX env (to compare
MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER settings across processes).
  cd src/worker; QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 PC_CTX=8192,32768 \
    .venv/bin/python $SP/prof/prefill_check.py"""
import os, sys, time
S = os.environ.get("PERF_OUT", "/tmp/perf") + ""
sys.path.insert(0, S + "/design")
import mlx.core as mx
import microbench_decode_batch as mb
from mlx_vlm.generate.common import generation_stream

model = mb.load_like_worker(mb.MODEL)
for ctx in [int(x) for x in os.environ.get("PC_CTX", "8192,32768").split(",")]:
    ids = [int(x) for x in mx.random.randint(1000, 150000, (ctx,)).tolist()]
    mx.clear_cache(); mx.reset_peak_memory()
    base = mx.get_active_memory()
    t = time.perf_counter()
    with mx.stream(generation_stream):
        cache, _ = mb.prefill_row(model, ids)
    dt = time.perf_counter() - t
    print(f"env ops={os.environ.get('MLX_MAX_OPS_PER_BUFFER', '-')} mb={os.environ.get('MLX_MAX_MB_PER_BUFFER', '-')} "
          f"ctx {ctx}: prefill {dt:.2f}s ({ctx / dt:.0f} tok/s), peak {mx.get_peak_memory() / 1e9:.2f} GB "
          f"(+{(mx.get_peak_memory() - base) / 1e9:.2f} over weights)", flush=True)
    del cache

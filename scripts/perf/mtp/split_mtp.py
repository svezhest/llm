"""Split the native MTP head of an MLX Qwen3.5/3.8 checkpoint into a standalone
mlx-vlm `qwen3_5_mtp` drafter directory (the worker's QWEN_DRAFT with an MTP head).

mlx-vlm 0.7.6's own splitter (`python -m mlx_vlm.split_mtp`) selects keys that
start with `mtp.`; MLX checkpoints converted with the head kept (e.g. the oQ4e
quant of Qwen3.8-27B) store it as `language_model.mtp.*` with per-module
quantization overrides under the same prefix, so it finds nothing there. This
script takes those keys, strips the prefix, keeps the tensors as they are (no
requantization: norms are already in MLX's +1 convention in an MLX source) and
writes the drafter config with the overrides renamed to the drafter's module
paths. CPU only (no Metal).

  src/worker/.venv/bin/python scripts/perf/mtp/split_mtp.py \
      weights/qwen38-27b-mlx-oq4e weights/qwen38-27b-mtp
"""
import json
import os
import shutil
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
PREFIX = "language_model.mtp."


def main(src, dst):
    with open(os.path.join(src, "config.json")) as f:
        conf = json.load(f)
    text = dict(conf.get("text_config") or conf)
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    files = sorted({v for k, v in wmap.items() if k.startswith(PREFIX)})
    if not files:
        sys.exit(f"no {PREFIX}* tensors in {src}")
    out = {}
    for fn in files:
        shard = mx.load(os.path.join(src, fn))
        out.update({k[len(PREFIX):]: v for k, v in shard.items() if k.startswith(PREFIX)})
    q = conf.get("quantization") or conf.get("quantization_config")
    quant = None
    if q is not None and any(k.endswith(".scales") for k in out):
        quant = {k: v for k, v in q.items() if not isinstance(v, dict)}
        for k, v in q.items():
            if isinstance(v, dict) and k.startswith(PREFIX):
                quant[k[len(PREFIX):]] = v
        # a module the source left unquantized (no .scales) stays so
        for k in out:
            if k.endswith(".weight") and k[:-7] + ".scales" not in out and out[k].ndim == 2:
                quant[k[:-7]] = False
    os.makedirs(dst, exist_ok=True)
    mx.save_safetensors(os.path.join(dst, "model.safetensors"), out, metadata={"format": "mlx"})
    depth = int(text.get("mtp_num_hidden_layers", 1))
    dconf = {"model_type": "qwen3_5_mtp", "text_config": text, "block_size": depth + 2,
             "tie_word_embeddings": bool(text.get("tie_word_embeddings", False))}
    if quant is not None:
        dconf["quantization"] = quant
        dconf["quantization_config"] = quant
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(dict(sorted(dconf.items())), f, indent=2)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))
    nbytes = sum(v.nbytes for v in out.values())
    print(f"{len(out)} tensors, {nbytes / 2**20:.0f} MiB -> {dst}; quantization overrides: "
          f"{ {k: v for k, v in (quant or {}).items() if isinstance(v, (dict, bool))} }")


if __name__ == "__main__":
    main(*sys.argv[1:3])

#!/usr/bin/env python3
"""A checkpoint oMLX's Lightning MTP accepts, assembled from an MLX model dir without the head and a
split head dir (scripts/perf/mtp/split_mtp.py or `mlx_vlm.split_mtp`), with no requantization and
no download (stdlib only, CPU, a copy of the head's bytes).

oMLX takes the head only from the model's own weights: it enables MTP when the config declares
`mtp_num_hidden_layers` and model.safetensors.index.json (else the shards' headers) lists a key under
`language_model.mtp.` (or `mtp.` / `model.mtp.`); the loaders read every model*.safetensors of the dir.
The output dir holds:
  - a relative symlink to every file of the model dir except config.json and the index,
  - model-mtp.safetensors: the head's tensors as stored, keys prefixed with `language_model.mtp.`,
  - model.safetensors.index.json: the model's map + the head's keys,
  - config.json: the model's, text_config.mtp_num_hidden_layers set, and quantization overrides for
    the head's modules wherever the head's quantization differs from the model's default (an
    unquantized module of the head -> false), under both quantization keys the model config uses.
The head's norms must already be in MLX's +1 convention (both splitters write them so from an MLX or
a raw HF source); oMLX's load-time norm check compares them with the backbone's.

  python3 scripts/perf/bench27/omlx_mtp_dir.py weights/ornith15-9b-mlx-8bit \\
      weights/_staging/ornith15-9b-mtp-q8 weights/_staging/ornith15-9b-mlx-8bit-mtp-omlx
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

PREFIX = "language_model.mtp."
SHARD = "model-mtp.safetensors"
QKEYS = ("quantization", "quantization_config")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def write_head(src, dst):
    """src's tensors under PREFIX: a new header, the data block copied as is (offsets unchanged)."""
    n, h = header(src)
    meta = h.pop("__metadata__", None) or {"format": "mlx"}
    out = {"__metadata__": meta}
    out.update({PREFIX + k: v for k, v in sorted(h.items())})
    hb = json.dumps(out, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        fi.seek(8 + n)
        while True:
            b = fi.read(1 << 24)
            if not b:
                break
            fo.write(b)
    return h


def head_quant(hconf, tensors):
    """The head's per-module quantization: {module: {bits, group_size, mode} | False}."""
    q = hconf.get("quantization") or hconf.get("quantization_config")
    mods = {k[:-len(".weight")] for k, v in tensors.items() if k.endswith(".weight") and len(v["shape"]) == 2}
    out = {}
    for mod in sorted(mods):
        if mod + ".scales" not in tensors:
            out[mod] = False
            continue
        if q is None:
            sys.exit(f"{mod}: quantized in the head, but the head's config has no quantization")
        ov = q.get(mod)
        if ov is False:
            sys.exit(f"{mod}: has scales but the head's config marks it unquantized")
        d = dict(ov) if isinstance(ov, dict) else {"group_size": q["group_size"], "bits": q["bits"]}
        d.setdefault("mode", q.get("mode", "affine"))
        out[mod] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("head")
    ap.add_argument("out")
    ap.add_argument("--force", action="store_true", help="replace an existing output dir's files")
    a = ap.parse_args()
    model, head, out = (os.path.abspath(p) for p in (a.model, a.head, a.out))
    mconf = json.load(open(os.path.join(model, "config.json")))
    hconf = json.load(open(os.path.join(head, "config.json")))
    idx_path = os.path.join(model, "model.safetensors.index.json")
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else None
    if index is None:       # a single-file model: its keys from the headers
        wmap = {}
        for fn in sorted(os.listdir(model)):
            if fn.startswith("model") and fn.endswith(".safetensors"):
                wmap.update({k: fn for k in header(os.path.join(model, fn))[1] if k != "__metadata__"})
        index = {"metadata": {}, "weight_map": wmap}
    have = [k for k in index["weight_map"] if k.startswith(PREFIX) or k.startswith("mtp.")]
    if have:
        sys.exit(f"{model} already has MTP tensors ({have[0]} ...): serve it as is")
    text = mconf.get("text_config", mconf)
    htext = hconf.get("text_config", hconf)
    for k in ("hidden_size", "vocab_size", "num_hidden_layers"):
        if text.get(k) != htext.get(k):
            sys.exit(f"head and model differ in {k}: {htext.get(k)} vs {text.get(k)}")
    if os.path.exists(out) and os.listdir(out) and not a.force:
        sys.exit(f"{out} exists and is not empty (--force replaces its files)")
    os.makedirs(out, exist_ok=True)

    for fn in sorted(os.listdir(model)):
        if fn in ("config.json", "model.safetensors.index.json", SHARD) or fn.startswith("."):
            continue
        link = os.path.join(out, fn)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.relpath(os.path.join(model, fn), out), link)

    tensors = write_head(os.path.join(head, "model.safetensors"), os.path.join(out, SHARD))
    index = dict(index, weight_map=dict(index["weight_map"], **{PREFIX + k: SHARD for k in tensors}))
    size = os.path.getsize(os.path.join(out, SHARD))
    index["metadata"] = dict(index.get("metadata") or {})
    if "total_size" in index["metadata"]:
        index["metadata"]["total_size"] = int(index["metadata"]["total_size"]) + size
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    conf = json.loads(json.dumps(mconf))
    ctext = conf.get("text_config", conf)
    depth = int(htext.get("mtp_num_hidden_layers") or 1)
    ctext["mtp_num_hidden_layers"] = max(int(ctext.get("mtp_num_hidden_layers") or 0), depth)
    hq = head_quant(hconf, tensors)
    added = {}
    for qk in QKEYS:
        mq = conf.get(qk)
        if mq is None:
            continue
        default = {"group_size": mq.get("group_size"), "bits": mq.get("bits"), "mode": mq.get("mode", "affine")}
        for mod, d in hq.items():
            if d is False or d != default:
                mq[PREFIX + mod] = d
                added[PREFIX + mod] = d
    if any(d is not False for d in hq.values()) and not any(conf.get(qk) is not None for qk in QKEYS):
        sys.exit("the head is quantized but the model's config has no quantization block")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(conf, f, indent=2)

    print(f"{out}: {len(tensors)} head tensors, {size / 2**20:.0f} MiB in {SHARD}; "
          f"{len(os.listdir(out)) - 3} symlinks; head quantization "
          f"{sorted({json.dumps(d) for d in hq.values()})}; overrides added: {added or 'none'}")


if __name__ == "__main__":
    main()

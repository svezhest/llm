#!/usr/bin/env python3
"""Cross-framework perplexity on one fixed corpus with identical token ids.

llama-perplexity's scheme, copied for MLX: the corpus is tokenized once (no BOS: the
Qwen vocab adds none), cut into floor(N / ctx) chunks of ctx tokens, every chunk
evaluated from an empty cache, and only its second half is scored: the logits at
positions ctx/2 .. ctx-2 against tokens ctx/2+1 .. ctx-1 (ctx/2 - 1 tokens per chunk).
PPL = exp(mean NLL); +/- = sqrt(var(NLL) / (count - 1)) * PPL (llama's formula).

  corpus  --name ppl_en --out F          write the corpus text (manifest's files at its commit)
  tokens  --gguf G --corpus F            llama-tokenize's ids vs the HF tokenizer's (CPU)
  mlx     --corpus F [--chunks N]        score an MLX checkpoint (QWEN_MODEL; GPU lease)
          run from src/worker: cd src/worker && QWEN_MODEL=... QWEN_KV_BITS=0 PYTHONPATH=. \\
            .venv/bin/python ../../scripts/perf/bench27/ppl.py mlx --corpus F --json out.json
  llama   --gguf G --corpus F [--chunks N]   llama-perplexity (GPU lease), parsed to JSON
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts as P  # noqa: E402


def hf_ids(text, model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    return tok.encode(text, add_special_tokens=False)


def cmd_corpus(a):
    man = P.load_manifest()
    text = P.corpus_text(man, a.name)
    want = man["corpora"][a.name].get("sha256")
    assert want is None or P.sha(text) == want, "corpus differs from the manifest's sha256"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {a.out}: {len(text)} chars")


def llama_ids(gguf, corpus):
    r = subprocess.run(["llama-tokenize", "-m", gguf, "-f", corpus, "--ids", "--log-disable", "--no-bos"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"llama-tokenize failed: {r.stderr[-2000:]}")
    m = re.search(r"\[([0-9,\s]*)\]", r.stdout)
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def cmd_tokens(a):
    text = open(a.corpus, encoding="utf-8").read()
    hf = hf_ids(text, a.model or os.path.join(P.weights_dir(), "qwen38-27b-mlx-4bit"))
    ll = llama_ids(a.gguf, a.corpus)
    first = next((i for i, (x, y) in enumerate(zip(hf, ll)) if x != y), None)
    res = {"hf": len(hf), "llama": len(ll), "identical": hf == ll, "first_diff": first}
    if first is not None:
        res["context"] = {"hf": hf[max(0, first - 3):first + 5], "llama": ll[max(0, first - 3):first + 5]}
    print(json.dumps(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def stats(nlls):
    n = len(nlls)
    mean = sum(nlls) / n
    var = sum(x * x for x in nlls) / n - mean * mean
    ppl = math.exp(mean)
    return {"ppl": round(ppl, 5), "ppl_pm": round(math.sqrt(max(var, 0) / (n - 1)) * ppl, 5), "nll_mean": mean,
            "scored_tokens": n}


def cmd_mlx(a):
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(HERE, "..", "prof"))
    import mlx.core as mx
    import pf_common as C
    from mlx_vlm.models.cache import make_prompt_cache
    t0 = time.time()
    path = os.environ["QWEN_MODEL"]
    model, proc = C.load(path)
    lm = model.language_model
    try:
        import pfqmm
        pfqmm.install(lm)
    except Exception as e:  # noqa: BLE001 -- the scorer runs without the prefill kernel too
        print("pfqmm not installed:", e)
    tok = getattr(proc, "tokenizer", proc)
    text = open(a.corpus, encoding="utf-8").read()
    ids = tok.encode(text, add_special_tokens=False)
    nch = len(ids) // a.ctx if a.chunks <= 0 else min(a.chunks, len(ids) // a.ctx)
    first = a.ctx // 2
    print(f"loaded {path} in {time.time() - t0:.0f} s; {len(ids)} tokens, {nch} chunks of {a.ctx}", flush=True)
    nlls, per_chunk = [], []
    for c in range(nch):
        chunk = ids[c * a.ctx:(c + 1) * a.ctx]
        cache = make_prompt_cache(lm)
        inp = mx.array([chunk])
        pos, out = 0, []
        while pos < a.ctx - 1:
            n = min(a.slab, a.ctx - 1 - pos)
            lg = lm(inp[:, pos:pos + n], cache=cache).logits[0]
            if pos + n > first:                          # positions >= first are scored
                lo = max(first, pos) - pos
                lgf = lg[lo:].astype(mx.float32)
                lp = lgf - mx.logsumexp(lgf, axis=-1, keepdims=True)
                tgt = mx.array(chunk[pos + lo + 1:pos + n + 1])
                nl = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
                mx.eval(nl, [cc.state for cc in cache])
                out.extend(nl.tolist())
            else:
                mx.eval([cc.state for cc in cache])
            pos += n
            mx.clear_cache()
        nlls.extend(out)
        per_chunk.append(round(math.exp(sum(nlls) / len(nlls)), 4))
        print(f"[{c + 1}] {per_chunk[-1]}", flush=True)
    res = dict(stats(nlls), system="mlx", model=os.path.basename(path.rstrip("/")), ctx=a.ctx, chunks=nch,
               corpus_tokens=len(ids), running_ppl=per_chunk, seconds=round(time.time() - t0, 1),
               peak_gib=round(mx.get_peak_memory() / 2**30, 2))
    print(json.dumps({k: v for k, v in res.items() if k != "running_ppl"}))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def cmd_llama(a):
    argv = ["llama-perplexity", "-m", a.gguf, "-f", a.corpus, "-c", str(a.ctx), "-ngl", "99", "-fa", "on",
            "-b", str(a.ctx)]
    if a.chunks > 0:
        argv += ["--chunks", str(a.chunks)]
    t0 = time.time()
    r = subprocess.run(argv, capture_output=True, text=True)
    log = r.stdout + r.stderr
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", log)
    nt = re.search(r"tokenizing the input \.\.\.\s*\n.*?(\d+) tokens", log)
    res = {"system": "llama.cpp", "model": os.path.basename(a.gguf), "ctx": a.ctx, "argv": argv,
           "returncode": r.returncode, "seconds": round(time.time() - t0, 1),
           "ppl": float(m.group(1)) if m else None, "ppl_pm": float(m.group(2)) if m else None,
           "chunks_line": (re.findall(r"calculating perplexity over (\d+) chunks.*", log) or [None])[0],
           "log_tail": log[-4000:]}
    print(json.dumps({k: v for k, v in res.items() if k != "log_tail"}))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if a.log:
        open(a.log, "w").write(log)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("--name", default="ppl_en")
    p.add_argument("--out", required=True)
    for name in ("tokens", "mlx", "llama"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", required=True)
        p.add_argument("--json", default="")
        p.add_argument("--ctx", type=int, default=2048)
        p.add_argument("--chunks", type=int, default=0, help="0 = all")
        p.add_argument("--gguf", default="")
        p.add_argument("--model", default="")
        p.add_argument("--slab", type=int, default=512)
        p.add_argument("--log", default="")
    a = ap.parse_args()
    {"corpus": cmd_corpus, "tokens": cmd_tokens, "mlx": cmd_mlx, "llama": cmd_llama}[a.cmd](a)


if __name__ == "__main__":
    main()

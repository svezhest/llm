#!/usr/bin/env python3
"""Phase 0 of "reply KV in the prompt cache" (reply_kv.md): CPU only, no model weights.

On recorded AppWorld traces (<tasks>/<id>/logs/lm_calls.jsonl, see scripts/perf/agentic/replay.py),
for every pair of consecutive calls (k, k+1) of a task: does the rendered prompt of call k+1 start
with  ids(prompt k) + tokenize(completion k) + [<|im_end|>]  ?  I.e. would the KV of the tokens the
model consumed while answering call k (prompt k + its reply, the EOS fed by the step-ahead) be a
prefix of call k+1, so call k+1 could resume after the reply.

Rendering is the worker's (Engine._render minus the checks): mlx-vlm's apply_chat_template with the
model's processor, enable_thinking as the gateway sends it (AppWorld ran with GATEWAY_THINKING=off),
then tokenizer.encode. Only the tokenizer / processor files are read; MLX is pinned to the CPU and
runs nothing.

    cd src/worker && uv run python ../../docs/perf/2026-10-17/design/reply_kv_phase0.py [--thinking on|off]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "worker"))
DEFAULT_TASKS = os.path.expanduser("~/Projects/appworld/experiments/outputs/ornith9b_react_dev/tasks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=DEFAULT_TASKS)
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "ornith15-9b-mlx-8bit"))
    ap.add_argument("--thinking", default="off", choices=["on", "off"])
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out", default=None, help="per-pair JSON lines")
    a = ap.parse_args()

    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from transformers import AutoTokenizer
    from worker import _bounds
    config = load_config(a.model)
    # a text-only checkpoint has no preprocessor_config.json: AutoProcessor
    # refuses it; the chat template is the tokenizer's (chat_template.jinja)
    proc = tok = AutoTokenizer.from_pretrained(a.model)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    nl = tok.encode("\n")
    think = a.thinking == "on"

    def render(msgs):
        msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in msgs]
        s = apply_chat_template(proc, config, msgs, add_generation_prompt=True, num_images=0,
                                enable_thinking=think)
        return tok.encode(s)

    def lcp(x, y):
        n = min(len(x), len(y))
        i = 0
        while i < n and x[i] == y[i]:
            i += 1
        return i

    files = sorted(glob.glob(os.path.join(a.tasks_dir, "*", "logs", "lm_calls.jsonl")))
    C = collections.Counter()
    gen_delta = collections.Counter()
    saved, new_tok, examples, rows = [], [], [], []
    for f in files:
        calls = [json.loads(line) for line in open(f)]
        task = f.split(os.sep)[-3]
        prev = None
        for ci, d in enumerate(calls):
            msgs = d["input"]["messages"]
            ids = render(msgs)
            out = d["output"]["choices"][0]["message"]
            content = out.get("content") or ""
            ct = (d["output"].get("usage") or {}).get("completion_tokens")
            cur = {"ids": ids, "content": content, "ct": ct, "msgs": msgs}
            if prev is not None:
                C["pairs"] += 1
                p_ids = prev["ids"]
                rep = tok.encode(prev["content"])
                if prev["ct"] is not None:
                    gen_delta[prev["ct"] - len(rep)] += 1
                # the history AppWorld sent back
                hist = msgs[len(prev["msgs"])] if len(msgs) > len(prev["msgs"]) else None
                same_hist = msgs[:len(prev["msgs"])] == prev["msgs"]
                C["history_prefix"] += same_hist
                C["assistant_content_verbatim"] += bool(hist and hist["role"] == "assistant"
                                                        and hist["content"] == prev["content"])
                C["prompt_prefix"] += ids[:len(p_ids)] == p_ids
                fed_eos = p_ids + rep + [im_end]          # step-ahead: the EOS went through the model
                fed_no = p_ids + rep                      # sync rows: it did not
                m_eos = ids[:len(fed_eos)] == fed_eos
                m_no = ids[:len(fed_no)] == fed_no
                C["match_with_eos"] += m_eos
                C["match_without_eos"] += m_no
                k = lcp(ids, fed_eos)
                where = ("full" if k == len(fed_eos) else
                         "before prompt end" if k < len(p_ids) else
                         "reply start" if k == len(p_ids) else
                         "in reply" if k < len(p_ids) + len(rep) else "at EOS")
                C["diverge: " + where] += 1
                # what the next turn would prefill today vs with the reply KV
                bounds = _bounds(ids, im_start)
                ends = set(bounds) | {len(ids) - 1}
                today = max((b for b in bounds if b <= len(p_ids) - 1), default=0)  # deepest ckpt <= old entry end
                after = len(fed_eos) if m_eos else len(fed_no) if m_no else today
                saved.append(after - today)
                new_tok.append(len(ids) - today)
                C["reply_end_is_slab_end"] += (len(fed_eos) in ends)
                rows.append({"task": task, "call": ci, "prompt_prev": len(p_ids), "reply": len(rep),
                             "ct": prev["ct"], "lcp": k, "where": where, "today_resume": today,
                             "reply_resume": after, "next_prompt": len(ids)})
                if where not in ("full",) and len(examples) < a.examples:
                    lo = max(len(p_ids), k - 8)
                    examples.append({"task": task, "call": ci, "where": where,
                                     "want": tok.decode(fed_eos[lo:k + 8]),
                                     "got": tok.decode(ids[lo:k + 8]),
                                     "want_ids": fed_eos[k:k + 4], "got_ids": ids[k:k + 4]})
            prev = cur
    n = C["pairs"]
    print(f"traces: {len(files)} tasks, {sum(1 for _ in rows) + len(files)} calls, {n} consecutive pairs "
          f"(thinking {a.thinking})")
    for k_, v in sorted(C.items()):
        print(f"  {k_:32s} {v:6d}  {100 * v / max(n, 1):5.1f} %")
    print("completion_tokens - len(tokenize(content)):",
          dict(sorted(gen_delta.items())))
    if saved:
        tot_new = sum(new_tok)
        print(f"next-turn prefill today: {tot_new} tokens ({tot_new / n:.0f}/call); "
              f"saved by the reply KV: {sum(saved)} ({100 * sum(saved) / tot_new:.1f} %, "
              f"{sum(saved) / n:.0f}/call)")
    for e in examples:
        print(json.dumps(e, ensure_ascii=False))
    if a.out:
        with open(a.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

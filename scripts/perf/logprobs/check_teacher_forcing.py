#!/usr/bin/env python3
"""Real-model check of the logprobs feature through the gateway (GPU; run only
with the GPU lease, against a worker at QWEN_MAX_BATCH=1 or 2 for bit-exact
decode).

For each recorded AppWorld LM call (lm_calls.jsonl lines: input.messages,
output.choices[0].message.content; generated at temperature 0 with thinking
off):
  gen      generation with logprobs:true on the recorded messages -> content,
           per-token logprobs (EOS included)
  gen_off  the same without logprobs -> content must be byte-identical to gen
           (the flag changes nothing) and, with the same worker build /
           knobs, to the recorded content (determinism)
  tf       teacher forcing: messages + the generated content as the last
           assistant message, prompt_logprobs:true, max_tokens 0,
           prompt_cache:false, add_generation_prompt:false -> the last
           assistant span's logprob sum must equal gen's sum (|d| small:
           prefill vs decode kernels)
  tf_hit   the same with prompt_cache "read" -> first > 1, the scored values
           equal tf's (hit == cold), the span sum equal where fully scored
  tf_var   the first user message changed (a sentence appended), "read": the tail
           after the change is recomputed; reports the span sums' deltas

Usage: check_teacher_forcing.py [--url http://127.0.0.1:8090/v1] [--n 5]
                                [--per-task 2] DIR_OR_JSONL...
PERF_OUT (default /tmp/perf) gets logprobs_check.json."""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out, time.time() - t


def calls(paths, per_task):
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "*/logs/lm_calls.jsonl"))) if os.path.isdir(p) else [p]
        for f in files:
            lines = [json.loads(x) for x in open(f)]
            # the longest contexts: the last calls of the task
            for rec in lines[-per_task:]:
                yield f, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="ornith15-9b-mlx-8bit")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    out_dir = os.environ.get("PERF_OUT", "/tmp/perf")
    os.makedirs(out_dir, exist_ok=True)
    base = dict(model=a.model, temperature=0, seed=100,
                chat_template_kwargs={"enable_thinking": False})
    rows = []
    for f, rec in list(calls(a.paths, a.per_task))[: a.n]:
        msgs = rec["input"]["messages"]
        recorded = rec["output"]["choices"][0]["message"]["content"]
        gen, t_gen = post(a.url, dict(base, messages=msgs, max_tokens=a.max_tokens, logprobs=True))
        gen_off, t_off = post(a.url, dict(base, messages=msgs, max_tokens=a.max_tokens))
        content = gen["choices"][0]["message"]["content"]
        lp = gen["choices"][0]["logprobs"]["content"]
        gen_sum = sum(x["logprob"] for x in lp)
        tf_msgs = msgs + [{"role": "assistant", "content": content}]
        tf_body = dict(base, messages=tf_msgs, max_tokens=0, prompt_logprobs=True,
                       add_generation_prompt=False)
        tf, t_tf = post(a.url, dict(tf_body, prompt_cache=False))
        hit, t_hit = post(a.url, dict(tf_body, prompt_cache="read"))
        var_msgs = [dict(m) for m in tf_msgs]
        i0 = next(i for i, m in enumerate(var_msgs) if m["role"] == "user")
        var_msgs[i0]["content"] += "\n\nNote: be concise."
        var, t_var = post(a.url, dict(tf_body, messages=var_msgs, prompt_cache="read"))

        def last_span(r):
            return r["prompt_logprobs_messages"][-1]

        s_tf, s_hit, s_var = last_span(tf), last_span(hit), last_span(var)
        plp_tf, plp_hit = tf["prompt_logprobs"], hit["prompt_logprobs"]
        first_hit = next((i for i, x in enumerate(plp_hit) if x is not None), None)
        same = all(x == y for x, y in zip(plp_tf[first_hit:], plp_hit[first_hit:])) if first_hit else None
        row = dict(
            file=f, call=rec["id"], prompt_tokens=gen["usage"]["prompt_tokens"],
            gen_tokens=len(lp), gen_sum=gen_sum, gen_eq_off=content == gen_off["choices"][0]["message"]["content"],
            gen_eq_recorded=content == recorded,
            tf_span=s_tf, tf_sum=s_tf["logprob"], d_tf_gen=s_tf["logprob"] - gen_sum,
            tf_scored_full=s_tf["scored"] == s_tf["tokens"],
            hit_first=first_hit, hit_cached=hit["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
            hit_values_equal=same, hit_span=s_hit,
            var_cached=var["usage"].get("prompt_tokens_details", {}).get("cached_tokens"),
            var_sum=s_var["logprob"], d_var=s_var["logprob"] - s_tf["logprob"],
            t=dict(gen=t_gen, gen_off=t_off, tf=t_tf, hit=t_hit, var=t_var),
        )
        rows.append(row)
        print(f"{rec['id'][:24]} P={row['prompt_tokens']} G={row['gen_tokens']} "
              f"gen={gen_sum:.3f} tf={s_tf['logprob']:.3f} d={row['d_tf_gen']:+.4f} "
              f"eq_off={row['gen_eq_off']} eq_rec={row['gen_eq_recorded']} "
              f"hit first={first_hit} cached={row['hit_cached']} same={same} "
              f"var d={row['d_var']:+.3f} cached={row['var_cached']} "
              f"t gen={t_gen:.1f}s tf={t_tf:.1f}s hit={t_hit:.1f}s var={t_var:.1f}s", flush=True)
    with open(os.path.join(out_dir, "logprobs_check.json"), "w") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False)
    bad = [r for r in rows if not (r["gen_eq_off"] and r["tf_scored_full"] and r["hit_values_equal"] is not False)]
    print(f"{len(rows)} calls, {len(bad)} with a failed invariant")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

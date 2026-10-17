# Builds the deterministic prompt corpus for the reply-KV GPU gate (stdlib + tokenizers).
#   cd src/worker && .venv/bin/python ../../docs/perf/2026-10-17/raw/reply_kv_build_corpus.py OUT.json
import json, sys
from tokenizers import Tokenizer
R = "/Users/user/Projects/llm/"
tok = Tokenizer.from_file(R + "weights/ornith15-9b-mlx-8bit/tokenizer.json")
def ntok(s): return len(tok.encode(s, add_special_tokens=False).ids)
def take(path, start_char, n):
    """text of `path` from start_char (snapped to a line start) with ~n tokens, whole lines"""
    t = open(R + path).read()
    start_char %= max(1, len(t) - 4000)          # wrap: every slice is a full one
    i = t.rfind("\n", 0, start_char) + 1 if start_char else 0
    out = []
    for ln in t[i:].split("\n"):
        out.append(ln)
        if ntok("\n".join(out)) >= n:
            break
    return "\n".join(out)

C = {}
# --- gate 1: one conversation, 2.5K system, ~300-token user turns, 6 turns
C["g1_system"] = ("You are a senior engineer reviewing a local LLM inference project. Answer "
                  "concisely and precisely; quote identifiers exactly. Project notes follow.\n\n"
                  + take("README.md", 9000, 2450))
g1q = ["What is the main point of this excerpt? One sentence.",
       "Which numbers in this excerpt matter most for performance, and why?",
       "Is there a risk in what this excerpt describes? Answer in two sentences.",
       "Name the components this excerpt touches.",
       "Explain this excerpt to a new team member.",
       "Summarize our whole conversation so far in one sentence."]
C["g1_turns"] = [f"Excerpt {k + 1}:\n\n" + take("BACKLOG.md", 3000 * k, 260) + "\n\n" + q
                 for k, q in enumerate(g1q)]
# gate 1b: the same system, other excerpts (gate 1's turn-2 reply mentioned `<think>`: not stored)
C["g1b_turns"] = [f"Excerpt {k + 1}:\n\n" + take("BACKLOG.md", 3000 * k + 1500, 260) + "\n\n" + q
                  for k, q in enumerate(g1q)]
# long-answer questions (64-token ignore_eos answers should not reach an EOS)
LQ = ["Explain in detail how this excerpt relates to the reference text; write at least 200 words.",
      "Walk through every mechanism this excerpt mentions, step by step, in detail (at least 200 words).",
      "Write a thorough review of this excerpt: strengths, weaknesses, open questions (at least 200 words).",
      "Describe how you would test what this excerpt describes, in detail (at least 200 words).",
      "Compare this excerpt with the earlier ones in this conversation, in detail (at least 200 words).",
      "Write a detailed design note expanding on this excerpt (at least 200 words)."]
srcs = [("CLAUDE.md", 0), ("CLAUDE.md", 25000), ("README.md", 0), ("BACKLOG.md", 0),
        ("src/gateway/src/request.rs", 0), ("src/gateway/src/api.rs", 0),
        ("docs/perf/2026-10-12/final_bench.md", 0), ("docs/perf/2026-10-16/replay_bench.md", 0)]
tsrc = ["docs/perf/2026-10-12/step_profile.md", "docs/perf/2026-10-12/mrqmm.md",
        "docs/perf/2026-10-12/prefill_profile.md", "docs/perf/2026-10-12/batching_gpu.md",
        "docs/perf/2026-10-12/cascade_gpu.md", "docs/perf/2026-10-12/prompt_cache_bench.md"]
def conv(c, sysp, nturns, off=0):
    return {"system": sysp,
            "turns": [f"Excerpt {c}.{t}:\n\n" + take(tsrc[t % len(tsrc)], 1200 * c + off, 280) + "\n\n" + LQ[t % len(LQ)]
                      for t in range(nturns)]}
# --- gate 2a / gate 3: 8 conversations, distinct ~3K systems (6 turns; gate 2a uses the first 4)
C["g8"] = [conv(c, f"You are assistant #{c} for the reviewer of project file {p}. Reference text:\n\n"
                + take(p, off, 2950), 6) for c, (p, off) in enumerate(srcs)]
# --- gate 2b: 16 conversations sharing one ~5K system prompt, 2 turns
shared = ("You are a reviewer of this project. Reference notes follow.\n\n" + take("CLAUDE.md", 0, 4950))
C["g16"] = [conv(c, shared, 2, off=300) for c in range(16)]
json.dump(C, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("g1 system", ntok(C["g1_system"]), "turns", [ntok(x) for x in C["g1_turns"]], "1b", [ntok(x) for x in C["g1b_turns"]])
print("g8 systems", [ntok(c["system"]) for c in C["g8"]], "turns", [[ntok(t) for t in c["turns"]] for c in C["g8"]][:2])
print("g16 system", ntok(shared), "turns", [[ntok(t) for t in c["turns"]] for c in C["g16"]][:3])

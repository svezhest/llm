# Builds the deterministic prompt corpus for the chain/evict GPU gate.
import json, sys
from tokenizers import Tokenizer
R = "/Users/user/Projects/llm/"
tok = Tokenizer.from_file(R + "weights/ornith15-9b-mlx-8bit/tokenizer.json")
def ntok(s): return len(tok.encode(s, add_special_tokens=False).ids)
def take(path, start_char, n):
    """text of `path` from start_char (snapped to a line start) with ~n tokens, whole lines"""
    t = open(R + path).read()
    i = t.rfind("\n", 0, start_char) + 1 if start_char else 0
    lines = t[i:].split("\n")
    out = []
    for ln in lines:
        out.append(ln)
        if ntok("\n".join(out)) >= n:
            break
    return "\n".join(out)

C = {}
# --- step 1/2: one long conversation
C["system"] = ("You are a senior engineer reviewing a local LLM inference project. Answer "
               "concisely and precisely; quote identifiers exactly. Project notes follow.\n\n"
               + take("README.md", 0, 2450))
C["document"] = take("src/worker/cascade.py", 0, 6000)
C["q1"] = "Using the document above: what does PrefixGroup store, and in which dtype? Answer in three sentences."
C["q1_sibling"] = "Using the document above: list the public methods of PrefixGroup with one short phrase each."
bl = open(R + "BACKLOG.md").read()
qs = ["Which items in this backlog excerpt are about memory? Name them.",
      "Summarize this backlog excerpt in two sentences.",
      "Which of these items would you do first and why? One paragraph.",
      "Does anything in this excerpt contradict the README? Answer briefly.",
      "Pick one item and list the risks of implementing it.",
      "Rewrite the first item of this excerpt as a one-line task title."]
C["followups"] = []
for k, q in enumerate(qs):
    C["followups"].append("Here is another excerpt of the backlog:\n\n" + take("BACKLOG.md", 2600 * k, 260) + "\n\n" + q)
# --- step 3: six independent conversations
srcs = [("CLAUDE.md", 0), ("CLAUDE.md", 20000), ("src/gateway/src/request.rs", 0),
        ("src/gateway/src/api.rs", 0), ("docs/perf/2026-10-12/final_bench.md", 0),
        ("docs/perf/2026-10-12/microbench.md", 0)]
turn_src = [("docs/perf/2026-10-12/step_profile.md", 0), ("docs/perf/2026-10-12/mrqmm.md", 0),
            ("docs/perf/2026-10-12/prefill_profile.md", 0), ("docs/perf/2026-10-12/batching_gpu.md", 0)]
tq = ["Relate this excerpt to the system notes in two sentences.",
      "What is the single most important number here, and why?",
      "List three follow-up questions an engineer should ask about this.",
      "Give a one-paragraph summary of the whole conversation so far."]
C["convs"] = []
for c, (p, off) in enumerate(srcs):
    sysp = (f"You are assistant #{c} for the reviewer of project file {p}. Be brief. Reference text:\n\n"
            + take(p, off, 2950))
    turns = []
    for t in range(4):
        tp, toff = turn_src[t]
        turns.append(f"Excerpt {c}.{t}:\n\n" + take(tp, toff + 1500 * c, 300) + "\n\n" + tq[t])
    C["convs"].append({"system": sysp, "turns": turns})
json.dump(C, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("system", ntok(C["system"]), "document", ntok(C["document"]),
      "followups", [ntok(x) for x in C["followups"]])
print("convs system", [ntok(c["system"]) for c in C["convs"]],
      "turns", [[ntok(t) for t in c["turns"]] for c in C["convs"]])

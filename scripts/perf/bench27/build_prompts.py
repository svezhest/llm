#!/usr/bin/env python3
"""Build prompts/manifest.json: the benchmark's fixed prompt set (run once, commit the
result BEFORE any measurement). CPU only (the HF tokenizer of the 27B, no MLX).

Classes (what the model writes; thinking off):
  code  wtest-style coding tasks; context = this repo's source code (worker + gateway)
  en    English prose tasks;     context = this repo's English docs
  ru    Russian prose tasks;     context = this repo's Russian docs/perf reports
Contexts (rendered prompt tokens, chat template, thinking off): 2K for every class
(3 tasks each), 8K and 32K for code and ru, 80K for code (one task each). The context
is the corpus' first N tokens (cut by tokens, as scripts/perf/dflash2/long_ctx.py),
N chosen so the whole rendered prompt is as close to the target as it gets from below.

  src/worker/.venv/bin/python scripts/perf/bench27/build_prompts.py [--commit 7437e62]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts as P  # noqa: E402

CODE_TASKS = [  # scripts/perf/prof/wtest.py TASKS[0..2]
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into seconds, then explain the edge cases.",
    "Implement an LRU cache in Rust with O(1) get and put, and walk through the ownership issues you run into.",
    "Write a Go worker pool that processes jobs from a channel with a bounded number of goroutines and graceful shutdown.",
]
EN_TASKS = [
    "Write a long, detailed essay (at least 800 words) about the history of the printing press and how it "
    "changed European society. Use full paragraphs, no lists or headings.",
    "Write a short story (at least 800 words) about a lighthouse keeper who finds a message in a bottle. "
    "Use rich, descriptive prose and dialogue.",
    "Explain to a curious teenager, in flowing prose of at least 800 words, why the Earth has seasons and "
    "why places near the equator barely notice them. No lists or headings.",
]
RU_TASKS = [
    "Напиши подробное эссе (не меньше 800 слов) об истории книгопечатания и о том, как оно изменило "
    "европейское общество. Пиши связными абзацами, без списков и заголовков.",
    "Напиши рассказ (не меньше 800 слов) о смотрителе маяка, который находит письмо в бутылке. "
    "Используй богатый описательный язык и диалоги.",
    "Объясни любознательному подростку связной прозой (не меньше 800 слов), почему на Земле бывают "
    "времена года и почему возле экватора их почти не замечают. Без списков и заголовков.",
]
HEADERS = {
    "code": "You are a coding assistant. Source files of a project follow; use them only if relevant.\n\n",
    "en": "You are a helpful writer. Notes of a software project follow; use them only if relevant.\n\n",
    "ru": "Ты — полезный помощник. Ниже — заметки программного проекта; используй их, только если они "
          "относятся к делу.\n\n",
}
FILE_HEADER = {"code": "// file: {path}\n", "en": "<!-- file: {path} -->\n", "ru": "<!-- файл: {path} -->\n"}
# (class, context target, tasks used, max_tokens list) -- the cells the plans pick from
LAYOUT = [("code", 2048, 3), ("en", 2048, 3), ("ru", 2048, 3),
          ("code", 8192, 1), ("ru", 8192, 1), ("code", 32768, 1), ("ru", 32768, 1), ("code", 81920, 1)]


def git_files(commit: str, pattern_dirs: list[str], exts: tuple[str, ...]) -> list[str]:
    import subprocess
    out = subprocess.run(["git", "-C", HERE, "ls-tree", "-r", "--full-tree", "--name-only", commit, "--", *pattern_dirs],
                         capture_output=True, text=True, check=True).stdout.split()
    return sorted(p for p in out if p.endswith(exts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="7437e62")
    ap.add_argument("--model", default="", help="tokenizer dir (default: weights/qwen38-27b-mlx-4bit)")
    ap.add_argument("--out", default=P.MANIFEST)
    a = ap.parse_args()
    import subprocess
    commit = subprocess.run(["git", "-C", HERE, "rev-parse", a.commit], capture_output=True, text=True,
                            check=True).stdout.strip()
    from transformers import AutoTokenizer
    model = a.model or os.path.join(P.weights_dir(), "qwen38-27b-mlx-4bit")
    tok = AutoTokenizer.from_pretrained(model)

    ru = []
    for p in git_files(commit, ["docs"], (".md",)):
        import re
        t = subprocess.run(["git", "-C", HERE, "show", f"{commit}:{p}"], capture_output=True,
                           check=True).stdout.decode()
        c, l = len(re.findall("[а-яА-ЯёЁ]", t)), len(re.findall("[a-zA-Z]", t))
        if c / (c + l + 1) > 0.4:
            ru.append(p)
    man = {
        "commit": commit,
        "tokenizer": os.path.basename(model.rstrip("/")),
        "render": "HF chat template, add_generation_prompt, enable_thinking=False",
        "corpora": {
            "code": {"file_header": FILE_HEADER["code"],
                     "files": git_files(commit, ["src/worker/qwen_worker"], (".py",))
                     + git_files(commit, ["src/gateway/src"], (".rs",)) + ["src/proto/worker.proto"]},
            "en": {"file_header": FILE_HEADER["en"], "files": ["README.md", "docs/TUTORIAL.md"]},
            "ru": {"file_header": FILE_HEADER["ru"], "files": ru},
            # perplexity corpora (ppl.py / llama-perplexity): plain concatenation
            "ppl_en": {"file_header": "", "files": ["README.md", "docs/TUTORIAL.md", "BACKLOG.md"]},
            "ppl_ru": {"file_header": "", "files": ru},
        },
        "prompts": [],
    }
    tasks = {"code": CODE_TASKS, "en": EN_TASKS, "ru": RU_TASKS}
    texts = {k: P.corpus_text(man, k) for k in man["corpora"]}
    for k in ("ppl_en", "ppl_ru"):
        man["corpora"][k]["sha256"] = P.sha(texts[k])
        man["corpora"][k]["hf_tokens"] = len(tok.encode(texts[k], add_special_tokens=False))
        print(f"{k}: {len(texts[k])} chars, {man['corpora'][k]['hf_tokens']} tokens")
    ids_all = {k: tok.encode(v, add_special_tokens=False) for k, v in texts.items()}

    def rendered(sys_text, task):
        return len(tok.apply_chat_template([{"role": "system", "content": sys_text},
                                            {"role": "user", "content": task}],
                                           tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                           return_dict=False))

    def nchars(cls, n):
        s = tok.decode(ids_all[cls][:n])
        while not texts[cls].startswith(s):        # a cut inside a character: back off
            n -= 1
            s = tok.decode(ids_all[cls][:n])
        return len(s)

    for cls, target, ntask in LAYOUT:
        ids = ids_all[cls]
        for k in range(ntask):
            pid = f"{cls}-{target // 1024}k-{k}"
            pr = {"id": pid, "cls": cls, "ctx": target, "corpus": cls, "header": HEADERS[cls],
                  "task": tasks[cls][k]}
            lo, hi = 0, len(ids)                       # largest n with rendered <= target
            assert rendered(P.system_text(man, dict(pr, n_chars=nchars(cls, hi)), "00"), pr["task"]) > target, \
                f"corpus {cls} too short for {target}"
            while lo < hi:
                mid = (lo + hi + 1) // 2
                pr["n_chars"] = nchars(cls, mid)
                if rendered(P.system_text(man, pr, "00"), pr["task"]) <= target:
                    lo = mid
                else:
                    hi = mid - 1
            pr["n_chars"] = nchars(cls, lo)
            sys_text = P.system_text(man, pr, "00")
            pr["n_tokens"] = rendered(sys_text, pr["task"])
            other = rendered(P.system_text(man, pr, "17"), pr["task"])
            assert other == pr["n_tokens"], (pid, other, pr["n_tokens"])   # tags don't move the count
            pr["sha256_tag00"] = P.sha(sys_text)
            man["prompts"].append(pr)
            print(f"{pid:12s} tokens {pr['n_tokens']:6d} context chars {pr['n_chars']:7d}", flush=True)
    with open(a.out, "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

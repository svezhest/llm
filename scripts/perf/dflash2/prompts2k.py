"""~2K-token prompts of three classes for the DFlash decode A/Bs (code /
English / Russian), cut by tokens from files of the repo at a fixed commit
(so the prompts do not move with the tree). No MLX here.

  prompts(tok, n=2000) -> [(name, text), ...]
"""
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REV = "7437e62"
SOURCES = (
    ("code", "src/worker/qwen_worker/sched_spec.py",
     "\n\nExplain step by step what the code above does, then rewrite the `pick` function so that it "
     "is simpler, with the full new code."),
    ("en", "README.md",
     "\n\nSummarize the text above in detail, section by section, in English."),
    ("ru", "docs/perf/2026-11-10/dflash2_27b.md",
     "\n\nПерескажи подробно по-русски, о чём этот текст: что измеряли, какие получились числа и выводы."),
)


SHORT = (
    ("en_prose", "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a "
                 "bottle. Plain prose, no headings."),
    ("ru_prose", "Напиши короткий рассказ (около 300 слов) о смотрителе маяка, который находит письмо в "
                 "бутылке. Обычная проза, без заголовков."),
)


def _show(path):
    return subprocess.run(["git", "-C", ROOT, "show", f"{REV}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def prompts(tok, n=2000):
    out = []
    for name, path, ask in SOURCES:
        ids = tok.encode(_show(path))
        out.append((f"{name}2k", tok.decode(ids[:n]) + ask))
    return out

"""The fixed prompt set of the cross-framework decode benchmark (stdlib only).

`prompts/manifest.json` (committed before any run, built by build_prompts.py) names
each prompt: its class (what the model writes: code / en prose / ru prose), the context
target, the corpus it cuts its context from, the user task and the expected rendered
token count. The corpora are files of this repository at one fixed commit, read with
`git show <commit>:<path>` (no copy of the text in the tree); every materialized
system message is checked against the manifest's sha256, so a prompt can never drift.

A request's system message = "Session <prompt id>/<tag>.\n" + header + context. The
first line makes every (prompt, tag) pair distinct from token ~3 on, so no server's
prefix cache can share one prompt's context with another's; the same tag twice = the
same prompt (a deliberate cache hit).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "prompts", "manifest.json")


def repo_root() -> str:
    """The main checkout (weights/ lives there, git-ignored), also from a worktree."""
    common = subprocess.run(["git", "-C", HERE, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return os.path.dirname(common)


def weights_dir() -> str:
    return os.environ.get("BENCH_WEIGHTS") or os.path.join(repo_root(), "weights")


def load_manifest(path: str = MANIFEST) -> dict:
    with open(path) as f:
        return json.load(f)


def corpus_text(man: dict, name: str) -> str:
    """The corpus' files at the manifest's commit, each under a one-line file header."""
    c = man["corpora"][name]
    parts = []
    for p in c["files"]:
        blob = subprocess.run(["git", "-C", HERE, "show", f"{man['commit']}:{p}"], capture_output=True,
                              check=True).stdout.decode("utf-8")
        parts.append(c["file_header"].format(path=p) + blob.rstrip("\n") + "\n\n")
    return "".join(parts)


_CACHE: dict[str, str] = {}


def system_text(man: dict, pr: dict, tag: str) -> str:
    name = pr["corpus"]
    if name not in _CACHE:
        _CACHE[name] = corpus_text(man, name)
    return f"Session {pr['id']}/{tag}.\n" + pr["header"] + _CACHE[name][:pr["n_chars"]]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def messages(man: dict, pr: dict, tag: str = "00", check: bool = True) -> list[dict]:
    sys_text = system_text(man, pr, tag)
    if check and tag == "00" and sha(sys_text) != pr["sha256_tag00"]:
        raise SystemExit(f"prompt {pr['id']}: system text does not match the manifest's sha256 "
                         f"(corpus commit {man['commit']})")
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": pr["task"]}]


def verify_all(man: dict) -> None:
    for pr in man["prompts"]:
        messages(man, pr, "00", check=True)

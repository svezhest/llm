#!/usr/bin/env python3
"""Markdown tables of the cross-system decode benchmark from run_matrix indexes, filled into the
report, and the raw sessions copied into the repo without private paths (stdlib only).

  tables  print every table (--h2: the 2K matrix, plan headline2; --ctx: plans ctx / ctx80)
  fill    replace the generated blocks of a report in place, each between
              <!-- report.py BLOCK -->  and  <!-- /report.py BLOCK -->
          (blocks: setup tokens medians ratios cells warmup prefill ctx-decode ctx-ratios ctx-prefill);
          a block whose data is missing keeps its text; prints the [[TODO ...]] left in the file
  raw     copy each listed session's session.json and the indexes to DEST, every absolute path
          of this machine replaced: a session's code tree -> $CODE, the weights -> $WEIGHTS, the
          main checkout -> $REPO, $PERF_OUT -> $PERF_OUT, home -> $HOME (+ --map PATH=NAME); writes
          nothing while a private path, the host or the user name is left

  python3 scripts/perf/bench27/report.py tables --h2 $PERF_OUT/bench9/index-*-h2a.json \\
      $PERF_OUT/bench9/index-*-h2b.json --ctx $PERF_OUT/bench9/index-*-ctx.json
  python3 scripts/perf/bench27/report.py fill docs/perf/2026-11-18/decode_9b_bench.md --h2 ... --ctx ...
  python3 scripts/perf/bench27/report.py raw docs/perf/2026-11-18/raw/bench9 $PERF_OUT/bench9/index-*.json

The numbers: a system's value for a class = the median over its sessions of each session's
median over the class's prompts (aggregate.py and chart.py show the same; the chart labels
round twice, so a label can read 0.1 below the table). Ratios are cell by cell: the median
over sessions of one cell (same prompt, mode, position in the session) against the other
system's, then the geometric mean over cells. Sessions an index marks off-probe are left
out (as aggregate.py does) unless --keep-off-probe. A system's column name and role: SYSTEMS;
the ratio rows: PAIRS (by role).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# system -> (role, name in the tables); a role orders the columns and pairs the ratios
SYSTEMS = {
    "ours-oq4e-auto": ("auto", "ours, `auto`"), "ours9-auto": ("auto", "ours, `auto`"),
    "ours-oq4e-plain": ("plain", "ours, plain"), "ours9-plain": ("plain", "ours, plain"),
    "ours-plain": ("plain", "ours, plain (4-bit)"),
    "ours-oq4e-dflash": ("dflash", "ours, DFlash"), "ours9-dflash": ("dflash", "ours, DFlash"),
    "ours-dflash": ("dflash", "ours, DFlash (4-bit)"),
    "ours-oq4e-mtp": ("mtp", "ours, MTP head"), "ours9-mtp": ("mtp", "ours, MTP head"),
    "llama-q4_0-mtp": ("llama-mtp", "llama.cpp + MTP"), "llama9-q8_0-mtp": ("llama-mtp", "llama.cpp + MTP"),
    "llama-q4_0": ("llama", "llama.cpp"), "llama9-q8_0": ("llama", "llama.cpp"),
    "llama-q4kxl": ("llama", "llama.cpp UD-Q4_K_XL"), "llama-q4_0-dflash": ("llama-dflash", "llama.cpp + DFlash"),
    "mlx-vlm": ("mlx-vlm", "mlx-vlm"), "mlx-vlm9": ("mlx-vlm", "mlx-vlm"),
    "mlx-vlm-dflash": ("mlx-vlm-dflash", "mlx-vlm + DFlash"),
}
ROLES = ["auto", "llama-mtp", "plain", "mlx-vlm", "llama", "dflash", "mtp", "llama-dflash", "mlx-vlm-dflash"]
PAIRS = [("auto", "llama-mtp"), ("plain", "llama"), ("plain", "mlx-vlm"), ("auto", "llama"), ("auto", "plain"),
         ("llama-mtp", "llama")]
PREFILL_PAIRS = [("plain", "llama"), ("plain", "mlx-vlm"), ("auto", "llama-mtp")]
CLS = {"code": "code", "en": "English", "ru": "Russian"}
MODE = {"greedy": "greedy", "sample": "sampling"}
BLOCKS = ["setup", "tokens", "medians", "ratios", "cells", "warmup", "prefill", "ctx-decode", "ctx-ratios",
          "ctx-prefill"]


def role(s):
    return SYSTEMS.get(s, (s, s))[0]


def name(s):
    return SYSTEMS.get(s, (s, s))[1]


def order_systems(systems):
    return sorted(systems, key=lambda s: (ROLES.index(role(s)) if role(s) in ROLES else len(ROLES), s))


def x(v):
    return f"×{v:.2f}"


def secs(t):
    return f"{t:.0f} s" if t >= 10 else f"{t:.1f} s"


# --------------------------------------------------------------------------- loading
def load(indexes, root="", keep_off=False):
    """-> list of session dicts (each once, in index order) with "_index" set."""
    out, seen = [], set()
    for ix in indexes:
        p = ix if os.path.isabs(ix) or not root else os.path.join(root, ix)
        idx = json.load(open(p))
        for e in idx["sessions"]:
            if e["session"] in seen:
                continue
            seen.add(e["session"])
            if e.get("off_probe") and not keep_off:
                print(f"note: {e['session']}: off-probe, left out", file=sys.stderr)
                continue
            sp = os.path.join(os.path.dirname(p), e["session"], "session.json")
            if not os.path.exists(sp):
                print(f"note: {sp}: missing, left out", file=sys.stderr)
                continue
            m = json.load(open(sp))
            if m.get("status") != "ok":
                print(f"note: {e['session']}: status {m.get('status')} {m.get('error') or ''}", file=sys.stderr)
            if not m.get("results"):
                continue
            m["_index"] = os.path.basename(p)
            out.append(m)
    return out


class Matrix:
    """The cells of a set of sessions: tab[system][cell] = [decode tok/s per session], cells in
    the order of the plan; cell = (cls, prompt, mode, max_tokens)."""

    def __init__(self, sessions):
        self.sessions = sessions
        self.tab, self.order, self.by_sys = {}, [], {}
        for m in sessions:
            self.by_sys.setdefault(m["system"], []).append(m)
            for r in m["results"]:
                if r.get("kind") != "cell" or not r.get("decode_tps"):
                    continue
                k = (r["cls"], r["prompt"], r["mode"], r["max_tokens"])
                if k not in self.order:
                    self.order.append(k)
                self.tab.setdefault(m["system"], {}).setdefault(k, []).append(r["decode_tps"])
        self.systems = order_systems(self.tab)
        mts = [k[3] for k in self.order]
        self.main_mt = max(set(mts), key=mts.count) if mts else 256
        self.classes = list(dict.fromkeys(k[0] for k in self.order))
        self.modes = [m for m in MODE if any(k[2] == m for k in self.order)]

    def groups(self):
        """(label, filter) per column: class x mode at the main length, then the other lengths."""
        g = [(f"{CLS.get(c, c)} {MODE[m]}", lambda k, c=c, m=m: k[0] == c and k[2] == m and k[3] == self.main_mt)
             for m in self.modes for c in self.classes]
        for c, m, mt in dict.fromkeys((k[0], k[2], k[3]) for k in self.order if k[3] != self.main_mt):
            g.append((f"{CLS.get(c, c)}{'' if m == 'greedy' else ' ' + MODE[m]}, {mt} tokens",
                      lambda k, c=c, m=m, mt=mt: k[0] == c and k[2] == m and k[3] == mt))
        return [(lab, f) for lab, f in g if any(f(k) for k in self.order)]

    def med(self, s, f):
        ks = [k for k in self.tab[s] if f(k)]
        if not ks:
            return None
        n = min(len(self.tab[s][k]) for k in ks)
        return st.median([st.median([self.tab[s][k][i] for k in ks]) for i in range(n)])

    def ratio(self, a, b, f=lambda k: True):
        v = [math.log(st.median(self.tab[a][k]) / st.median(self.tab[b][k]))
             for k in self.order if f(k) and k in self.tab[a] and k in self.tab[b]]
        if not v:
            return None
        return math.exp(sum(v) / len(v)), math.exp(min(v)), math.exp(max(v)), \
            sum(y >= math.log(0.995) for y in v), len(v)

    def pairs(self, pairs=PAIRS):
        out = []
        for ra, rb in pairs:
            out += [(a, b) for a in self.systems if role(a) == ra for b in self.systems if role(b) == rb]
        return out


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


# --------------------------------------------------------------------------- the 2K blocks
def b_setup(M, C):
    sess = M.sessions + (C.sessions if C else [])
    if not sess:
        return None
    rows = []
    for m in sess:
        v = m.get("versions") or {}
        rows.append([m["system"], m["plan"], m["session"][:15], m["started"][11:16],
                     f"`{(v.get('code_commit') or '?')[:7]}`" + (" (dirty)" if v.get("code_dirty") else ""),
                     str((m.get("probe") or {}).get("sum_gbs")), m.get("status", "?")])
    out = table(["system", "plan", "session", "started", "code", "probe GB/s", "status"], rows)
    facts = {}
    for m in sess:
        v, p = m.get("versions") or {}, m.get("preflight") or {}
        llama = v.get("llama.cpp")
        for k, val in (("mlx / mlx-vlm", v.get("mlx/mlx_vlm")),
                       ("llama.cpp", llama[0] if isinstance(llama, list) and llama else llama),
                       ("chip", v.get("chip")), ("GPU cores", (v.get("gpu_cores") or "").split("=")[-1].strip()),
                       ("iogpu.wired_limit_mb", p.get("wired_limit_mb")),
                       ("power mode", " ".join((p.get("powermode") or [""])[0].split())),
                       ("power source", (p.get("power_source") or [""])[0])):
            if val:
                facts.setdefault(k, [])
                if val not in facts[k]:
                    facts[k].append(val)
    out += "\n\n" + "\n".join(f"- {k}: {'; '.join(map(str, v))}" for k, v in facts.items())
    return out


def b_tokens(M, C):
    """Prompt tokens as each server counted them (usage.prompt_tokens) against the manifest's count
    (built with the tokenizer the manifest names): the systems must agree with each other."""
    sess = M.sessions + (C.sessions if C else [])
    if not sess:
        return None
    systems = order_systems({m["system"] for m in sess})
    got, exp = {}, {}
    for m in sess:
        for r in m["results"]:
            if r.get("kind") == "cell" and r.get("prompt_tokens") is not None:
                got.setdefault(r["prompt"], {}).setdefault(m["system"], set()).add(r["prompt_tokens"])
                exp[r["prompt"]] = r.get("expected_prompt_tokens")
    rows = []
    for pid, per in got.items():
        vals = [sorted(per.get(s, ())) for s in systems]
        agree = len({tuple(v) for v in vals if v}) == 1
        rows.append([pid, str(exp.get(pid))]
                    + [(str(v[0]) if v[0] == v[-1] else f"{v[0]}–{v[-1]}") if v else "–" for v in vals]
                    + ["yes" if agree else "**no**"])
    return table(["prompt", "manifest"] + [name(s) for s in systems] + ["systems agree"], rows)


def b_medians(M, _):
    if not M.systems:
        return None
    g = M.groups()
    rows = [[name(s)] + [f"{v:.1f}" if v is not None else "–" for v in (M.med(s, f) for _, f in g)]
            for s in M.systems]
    return table(["system"] + [lab for lab, _ in g], rows)


def b_ratios(M, _):
    pairs = M.pairs()
    if not pairs:
        return None
    g = [x for x in M.groups() if not x[0].endswith(" tokens")]
    rows = []
    for a, b in pairs:
        gm, lo, hi, n, N = M.ratio(a, b)
        per = [M.ratio(a, b, f) for _, f in g]
        rows.append([f"{name(a)} / {name(b)}", x(gm), f"{lo:.2f} – {hi:.2f}", f"{n} / {N}"]
                    + [x(p[0]) if p else "–" for p in per])
    N = len(M.order)
    return table(["ratio (geometric mean over cells)", f"all {N} cells", "min – max", "cells ≥ 1"]
                 + [lab for lab, _ in g], rows)


def b_cells(M, _):
    if not M.systems:
        return None
    rows = []
    for i, k in enumerate(M.order):
        rows.append([str(i + 1), f"{k[1]} {k[2]}{'' if k[3] == M.main_mt else f' {k[3]}'}"]
                    + [" / ".join(f"{v:.1f}" for v in M.tab[s].get(k, [])) or "–" for s in M.systems])
    return table(["#", "cell"] + [name(s) for s in M.systems], rows)


def b_warmup(M, _):
    """The greedy cells of the first class by position, per session: the warm-up of a session."""
    if not M.systems:
        return None
    pos = [i for i, k in enumerate(M.order) if k[0] == M.classes[0] and k[2] == "greedy"]
    rows = []
    for s in M.systems:
        for j, m in enumerate(M.by_sys[s]):
            v = {}
            for r in m["results"]:
                if r.get("kind") == "cell" and r.get("decode_tps"):
                    k = (r["cls"], r["prompt"], r["mode"], r["max_tokens"])
                    v.setdefault(k, r["decode_tps"])
            vals = [v.get(M.order[i]) for i in pos]
            have = [y for y in vals if y]
            drop = f"−{100 * (1 - have[-1] / have[0]):.0f} %" if len(have) > 1 else "–"
            rows.append([name(s), "ab"[j] if j < 2 else str(j + 1)]
                        + [f"{y:.1f}" if y else "–" for y in vals] + [drop])
    return table(["system", "session"] + [f"cell {i + 1} ({M.order[i][1]}"
                                          f"{'' if M.order[i][3] == M.main_mt else ', ' + str(M.order[i][3])})"
                                          for i in pos] + ["first → last"], rows)


def cold(m, greedy_only=True):
    """{cell + round: (ttft_s, prefill tok/s, ctx)} of a session's cold cells (nothing cached), in order."""
    out = {}
    for r in m["results"]:
        if r.get("kind") == "cell" and r.get("ttft_s") is not None and not r.get("cached_tokens") \
                and (r["mode"] == "greedy" or not greedy_only):
            out.setdefault((r["cls"], r["prompt"], r["mode"], r["max_tokens"], r.get("round", 0)),
                           (r["ttft_s"], r.get("prefill_tps"), r["ctx"]))
    return out


def b_prefill(M, _):
    if not M.systems:
        return None
    rows = []
    for s in M.systems:
        for j, m in enumerate(M.by_sys[s]):
            c = list(cold(m).values())
            if not c:
                continue
            t = [y[0] for y in c]
            pre = [y[1] for y in c if y[1]]
            rows.append([name(s), "ab"[j] if j < 2 else str(j + 1), str(len(c)),
                         f"{min(t):.1f} – {max(t):.1f}", f"{pre[0]:.0f} → {pre[-1]:.0f}" if pre else "–"])
    out = table(["system", "session", "cold greedy cells", "TTFT s, min – max",
                 "prefill tok/s, first → last cold cell"], rows)
    prs = []
    for a, b in M.pairs(PREFILL_PAIRS):
        per = []
        for ma, mb in zip(M.by_sys[a], M.by_sys[b]):
            ca, cb = cold(ma), cold(mb)
            v = [math.log(cb[k][0] / ca[k][0]) for k in ca if k in cb]
            if v:
                per.append(f"{x(math.exp(sum(v) / len(v)))} ({len(v)} cells)")
        if per:
            prs.append([f"{name(a)} / {name(b)}", " / ".join(per)])
    if prs:
        out += "\n\n" + table(["cold prefill speed (TTFT ratio, geometric mean, same cells and position)",
                               "session a / b"], prs)
    return out


# --------------------------------------------------------------------------- the long-context blocks
def ctx_cells(C):
    """{system: {(cls, ctx): [[tok/s per round] per session]}}, rows in plan order."""
    tab, rows = {}, []
    for m in C.sessions:
        per = {}
        for r in m["results"]:
            if r.get("kind") == "cell" and r.get("decode_tps") and r["mode"] == "greedy":
                k = (r["cls"], r["ctx"])
                per.setdefault(k, []).append(r["decode_tps"])
                if k not in rows:
                    rows.append(k)
        for k, v in per.items():
            tab.setdefault(m["system"], {}).setdefault(k, []).append(v)
    return tab, rows


def b_ctx_decode(_, C):
    if not C or not C.systems:
        return None
    tab, rows = ctx_cells(C)
    return table(["decode tok/s, round 1 / round 2"] + [name(s) for s in C.systems],
                 [[f"{CLS.get(c, c)} {n // 1024}K"] + ["; ".join(" / ".join(f"{v:.1f}" for v in sess)
                                                                for sess in tab.get(s, {}).get((c, n), [])) or "–"
                                                      for s in C.systems] for c, n in rows])


def b_ctx_ratios(_, C):
    if not C or not C.systems:
        return None
    tab, rows = ctx_cells(C)
    out = []
    for a, b in C.pairs():
        vals = []
        for k in rows:
            va, vb = tab.get(a, {}).get(k), tab.get(b, {}).get(k)
            vals.append(x(va[0][0] / vb[0][0]) if va and vb else "–")
        if any(v != "–" for v in vals):
            out.append([f"{name(a)} / {name(b)}"] + vals)
    if not out:
        return None
    return table(["decode ratio, round 1 (first session)"] + [f"{CLS.get(c, c)} {n // 1024}K" for c, n in rows], out)


def b_ctx_prefill(_, C):
    if not C or not C.systems:
        return None
    tt, ctxs, cls_by_ctx = {}, [], {}
    for m in C.sessions:
        for k, (t, _, n) in cold(m).items():
            c = k[0]
            if k[4] != 0:
                continue
            tt.setdefault(m["system"], {}).setdefault((c, n), t)
            if n not in ctxs:
                ctxs.append(n)
            cls_by_ctx.setdefault(n, [])
            if c not in cls_by_ctx[n]:
                cls_by_ctx[n].append(c)
    rows = [[f"{n // 1024}K ({' / '.join(CLS.get(c, c) for c in cls_by_ctx[n])})"]
            + [" / ".join(secs(tt[s][(c, n)]) if (c, n) in tt.get(s, {}) else "–" for c in cls_by_ctx[n])
               for s in C.systems] for n in sorted(ctxs)]
    t2 = {}
    for m in C.sessions:
        for r in m["results"]:
            if r.get("kind") == "turn2" and r.get("ttft_s") is not None:
                t2.setdefault((r["cls"], r["ctx"]), {}).setdefault(m["system"], []).append(
                    secs(r["ttft_s"]) + ("" if r.get("cached_tokens") else " (no cache)"))
    for (c, n), v in sorted(t2.items(), key=lambda kv: kv[0][1]):
        rows.append([f"follow-up turn on the {CLS.get(c, c)} {n // 1024}K conversation, round 1, 2"]
                    + [", ".join(v.get(s, [])) or "–" for s in C.systems])
    return table(["cold prefill (TTFT, round 1)"] + [name(s) for s in C.systems], rows)


FUNCS = {"setup": b_setup, "tokens": b_tokens, "medians": b_medians, "ratios": b_ratios, "cells": b_cells,
         "warmup": b_warmup, "prefill": b_prefill, "ctx-decode": b_ctx_decode, "ctx-ratios": b_ctx_ratios,
         "ctx-prefill": b_ctx_prefill}


def build(a):
    M = Matrix(load(a.h2, a.root, a.keep_off_probe))
    C = Matrix(load(a.ctx, a.root, a.keep_off_probe)) if a.ctx else None
    return {b: FUNCS[b](M, C) for b in BLOCKS}


def fill(path, blocks):
    text = open(path).read()
    for b, body in blocks.items():
        pat = re.compile(rf"(<!-- report\.py {re.escape(b)} -->\n)(.*?)(<!-- /report\.py {re.escape(b)} -->)", re.S)
        if not pat.search(text):
            print(f"{path}: no block {b}", file=sys.stderr)
            continue
        if body is None:
            print(f"{b}: no data, left as it is", file=sys.stderr)
            continue
        text = pat.sub(lambda mo: mo[1] + body.strip() + "\n" + mo[3], text)
        print(f"{b}: filled")
    open(path, "w").write(text)
    todo = [ln for ln in text.splitlines() if "[[TODO" in ln]
    print(f"{len(todo)} line(s) with [[TODO ...]] left in {path}")
    for ln in todo:
        print("  " + ln.strip()[:140])


# --------------------------------------------------------------------------- raw
def git(*args):
    r = subprocess.run(["git", "-C", HERE] + list(args), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def private_marks():
    home = os.path.expanduser("~")
    host = socket.gethostname()
    marks = {home, "/Users/", "/home/", "/private/", "/var/folders/", "/tmp/", host, host.split(".")[0]}
    user = os.environ.get("USER", "")
    if user:
        marks.add(f"/{user}/")
    for k in ("user.email", "user.name"):
        v = git("config", k)
        if v:
            marks.add(v)
    return {m for m in marks if m and len(m) > 3}


def raw(a):
    common = git("rev-parse", "--path-format=absolute", "--git-common-dir")
    weights = os.environ.get("BENCH_WEIGHTS") or (os.path.join(os.path.dirname(common), "weights") if common else "")
    base = {}
    for p, n in ((weights, "$WEIGHTS"), (os.path.dirname(common) if common else "", "$REPO"),
                 (os.environ.get("PERF_OUT", ""), "$PERF_OUT"), (os.path.expanduser("~"), "$HOME")):
        if p:
            base[os.path.realpath(p)] = n
            base[os.path.abspath(p)] = n
    for kv in a.map or []:
        p, _, n = kv.partition("=")
        base[os.path.abspath(p)] = n
    files = {}
    for ix in a.indexes:
        idx = json.load(open(ix))
        files[os.path.basename(ix)] = (ix, idx.get("code"))
        for e in idx["sessions"]:
            sp = os.path.join(os.path.dirname(ix), e["session"], "session.json")
            if os.path.exists(sp):
                files[os.path.join(e["session"], "session.json")] = (sp, json.load(open(sp)).get("code"))
            else:
                print(f"note: {sp}: missing", file=sys.stderr)
    out, bad = {}, []
    marks = private_marks()
    for rel, (src, code) in files.items():
        m = dict(base)
        if code:
            m[code] = "$CODE"
        text = open(src).read()
        for p in sorted(m, key=len, reverse=True):      # the longest prefix first: weights inside a checkout
            text = re.sub(re.escape(p) + r'(?=[/"\s:,;)\]\'}]|$)', lambda _, n=m[p]: n, text)
        json.loads(text)
        for i, line in enumerate(text.splitlines()):
            hit = [k for k in marks if k in line]
            if hit:
                bad.append(f"{rel}:{i + 1}: {hit}: {line.strip()[:160]}")
        out[rel] = text
    if bad:
        print("private text left (add --map PATH=$NAME), nothing written:\n" + "\n".join(bad[:40]), file=sys.stderr)
        sys.exit(1)
    for rel, text in out.items():
        d = os.path.join(a.dest, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        with open(d, "w") as f:
            f.write(text)
    print(f"wrote {len(out)} files to {a.dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("tables", "fill"):
        p = sub.add_parser(c)
        if c == "fill":
            p.add_argument("report")
        p.add_argument("--h2", nargs="+", default=[], help="index-*.json of the 2K matrix (plan headline2)")
        p.add_argument("--ctx", nargs="*", default=[], help="index-*.json of the long-context sessions")
        p.add_argument("--root", default="", help="where relative index paths live")
        p.add_argument("--keep-off-probe", action="store_true")
    p = sub.add_parser("raw")
    p.add_argument("dest")
    p.add_argument("indexes", nargs="+")
    p.add_argument("--map", action="append", help="PATH=$NAME: one more replacement")
    a = ap.parse_args()
    if a.cmd == "raw":
        return raw(a)
    blocks = build(a)
    if a.cmd == "fill":
        return fill(a.report, blocks)
    for b, body in blocks.items():
        print(f"## {b}\n\n{body if body is not None else '(no data)'}\n")


if __name__ == "__main__":
    main()

"""Split the original worker.py into the qwen_worker package (pure move).

  git show 46aa1dd:src/worker/worker.py > /tmp/worker_old.py
  cd src/worker && uv run python ../../scripts/refactor/split_worker.py /tmp/worker_old.py .

Kept as the record of how qwen_worker/ was produced from the single-file
worker.py (the line ranges below are that file's); verify_worker_split.py
checks the result. Run it with the worker's Python (symtable differs by version).
"""
import ast
import os
import re
import symtable
import sys
import builtins
import textwrap

ORIG, OUT = sys.argv[1], sys.argv[2]
SRC = open(ORIG).read()
LINES = SRC.splitlines(keepends=True)
N = len(LINES)
TREE = ast.parse(SRC)

# ---------------------------------------------------------------- the plan
# top-level line ranges -> module (inclusive, 1-based)
TOP = [
    ("facade", 1, 17), ("drop", 18, 54),
    ("config", 55, 64), ("chat", 65, 320), ("config", 321, 338), ("chat", 339, 493),
    ("config", 494, 578), ("sampler", 579, 826), ("config", 827, 1099),
    ("pcache", 1100, 1922), ("sizing", 1923, 1985), ("stream", 1986, 2056),
    ("grammar", 2057, 2307), ("scheduler", 2308, 2331), ("config", 2332, 2420),
    ("nosync", 2421, 2474), ("config", 2475, 2543), ("rows", 2544, 2665),
    ("sched_prefill", 2666, 2671), ("sizing", 2672, 2678), ("engine", 2679, 2686),
    ("sizing", 2687, 2704), ("rows", 2705, 2719), ("config", 2720, 2769),
    ("kernels", 2770, 2773), ("config", 2774, 2784), ("kernels", 2785, 2841),
    ("config", 2842, 2873), ("kernels", 2874, 3063), ("sched_spec", 3064, 3078),
    ("kernels", 3079, 3087), ("config", 3088, 3101), ("kernels", 3102, 3278),
    ("sizing", 3279, 3313), ("rows", 3314, 3365), ("logprobs", 3366, 3506),
    ("request", 3507, 3643), ("SCHED", 3644, 6597), ("engine", 6598, 7354),
    ("stream", 7355, 7508), ("server", 7509, 7634), ("facade", 7635, 7638),
]

MIXINS = {   # _Scheduler method -> module
    "sched_admit": "_lane_first _bytes _step_bytes _tokens _reserve _regrow _tasks _parents _fits "
                   "_peek _lookup_ids _wanted _bounds _ends _admit _setup _lane_tokens _admit_lane "
                   "_spec_ok _advance _advance_lane _next _task_done",
    "sched_prefill": "_prompt_cache _slab_done _prefill _store_prefix",
    "sched_spec": "_spec _spec_reply",
    "sched_store": "_reply_cache _move_on move_state _lost _settle_prefix _settle_prefix_ _settle "
                   "_reply_plan _store_reply _disk_tick _disk_want _held_wait _disk_room "
                   "_disk_take _disk_insert",
    "sched_cascade": "_compact _cascade_on cascade_state set_cascade _cascade_count _unlink "
                     "_unlink_row _drop_groups _member_bytes _group_suffix _group_bytes "
                     "_group_wins _charge_fits _cascadable _fits_prefix _same_prefix "
                     "_cascade_join _group_build _group_add _cascade_tidy _tidy_choice "
                     "_tidy_group _plan _grouped_forward",
    "sched_decode": "_hidden_dtype _join _drop_done _end _finish _fail _fail_rows _step _logits "
                    "_sample _iterate _preempt_rows _preemptable _preempt _store_preempt",
}
CORE = ("__init__ start alive submit wake stop _run _close _busy _idle tick fit_max_batch "
        "_log").split()
MIXIN_CLASS = {
    "sched_admit": ("_AdmitMixin", "Admission and the memory budget: reservations, _fits / _regrow, the\n"
                    "    prefill lane, the task queue (_next / _advance)."),
    "sched_prefill": ("_PrefillMixin", "Prefill: the prompt-cache lookup / checkpoint plan "
                      "(_prompt_cache), the\n    slab loop (_prefill) and the partial store."),
    "sched_spec": ("_SpecMixin", "DFlash speculative decoding of a lone row (_spec) and its reply "
                   "store."),
    "sched_store": ("_StoreMixin", "Prompt-cache stores at a row's end: reply KV, one KV copy\n"
                    "    (QWEN_PROMPT_CACHE_MOVE: _settle*), the disk tier's loads."),
    "sched_cascade": ("_CascadeMixin", "Cascade groups (QWEN_CASCADE): formation, joins, tidy, "
                      "the step plan."),
    "sched_decode": ("_DecodeMixin", "The batch: join / drop rows, the decode step and its "
                     "sampling, row ends,\n    preemption (QWEN_PREEMPT)."),
}
ORDER = ["config", "chat", "sampler", "pcache", "sizing", "stream", "grammar", "nosync", "rows",
         "kernels", "logprobs", "request", "sched_admit", "sched_prefill", "sched_spec",
         "sched_store", "sched_cascade", "sched_decode", "scheduler", "engine", "server"]
DOC = {
    "config": "Env knobs (QWEN_*) and process-wide constants, read once at import.\n\n"
              "Feature-local knobs stay with their code: QWEN_JSON_MAX_WS (grammar.py),\n"
              "QWEN_LOGPROBS_CHUNK (logprobs.py). SDPA_FUSED, DRAFT_BLOCK and BATCH_INVARIANT\n"
              "are also changed at load (kernels.py): other modules read them as `cfg.NAME`.",
    "chat": "Request side: tool-call parsing (vLLM qwen3_xml), images, chat messages and the\n"
            "prompt tokenization with QWEN_SPLIT_SPECIAL_TOKENS.",
    "sampler": "Sampling in vLLM's order: penalties, temperature, min_p, top_k, top_p, draw.",
    "pcache": "Prompt cache: cache-array helpers, KV segments, entries, the prefix cache.",
    "sizing": "Prefill slab policy (_step / _bounds) and the memory budget arithmetic.",
    "stream": "Handler side of a generation: detokenizer, stop strings, request controls,\n"
              "the reasoning / answer split and the Done event.",
    "grammar": "Structured output: response_format / tool-call grammars (llguidance).",
    "nosync": "QWEN_DEBUG_NO_SYNC: host waits raise while a decode step is built.",
    "rows": "Batched decode over per-row caches: RowsKV, the patched attention, cascade\n"
            "step plans, the stacked GDN state.",
    "kernels": "mlx-vlm patches and fast paths: fused SDPA, fast GDN, the DFlash verify\n"
               "(mrqmm, vector attention), the drafter's fp32 mode, the fused GDN\n"
               "in-projection, batch-invariant checks.",
    "logprobs": "Log-probabilities (logprobs / top_logprobs / prompt_logprobs) and the\n"
                "assistant spans of a prompt.",
    "request": "A request (_Req), a running row (_Row) and a moved prompt-cache lease (_Own).",
    "sched_admit": "_Scheduler: admission, budget, prefill lane, task queue.",
    "sched_prefill": "_Scheduler: prompt-cache lookup and the prefill slab loop.",
    "sched_spec": "_Scheduler: DFlash speculative decoding.",
    "sched_store": "_Scheduler: prompt-cache stores at a row's end, one KV copy, the disk tier.",
    "sched_cascade": "_Scheduler: cascade groups.",
    "sched_decode": "_Scheduler: the batch, the decode step, row ends, preemption.",
    "scheduler": "The continuous-batching scheduler (_Scheduler): threads, ticks, logging; the\n"
                 "rest of its methods come from the sched_* mixins.",
    "engine": "Engine: model load, request checks / rendering, generate() for the handlers.",
    "server": "gRPC servicer, serve() and main().",
}
ALIAS = {"config": "cfg", "sizing": "sizing", "kernels": "kernels", "rows": "rows_mod",
         "nosync": "nosync"}
LOCAL_TOP = {"cascade", "cascade_mma", "pcdisk", "worker_pb2", "worker_pb2_grpc"}


# ---------------------------------------------------------------- coverage
owner = [None] * (N + 1)
for mod, a, b in TOP:
    for i in range(a, b + 1):
        assert owner[i] is None, (i, mod)
        owner[i] = mod
assert all(owner[1:]), [i for i in range(1, N + 1) if owner[i] is None]

# _Scheduler body segments
sched = next(s for s in TREE.body if isinstance(s, ast.ClassDef) and s.name == "_Scheduler")
meth_mod = {}
for mod, names in MIXINS.items():
    for n in names.split():
        assert n not in meth_mod, n
        meth_mod[n] = mod
for n in CORE:
    meth_mod[n] = "scheduler"
body = sched.body
assert isinstance(body[0], ast.Expr)
doc_end = body[0].end_lineno
sched_segs = []   # (module, start, end)
prev = doc_end
for m in body[1:]:
    assert isinstance(m, ast.FunctionDef), m
    assert m.name in meth_mod, m.name
    sched_segs.append((meth_mod[m.name], prev + 1, m.end_lineno, m.name))
    prev = m.end_lineno
assert prev == sched.end_lineno
assert set(meth_mod) == {m.name for m in body[1:]}


def text(a, b):
    return "".join(LINES[a - 1:b])


# ---------------------------------------------------------------- assemble
parts = {m: [] for m in ORDER}
for mod, a, b in TOP:
    if mod in ("facade", "drop", "SCHED"):
        continue
    parts[mod].append(text(a, b))

classes = {}
for mod, a, b, name in sched_segs:
    classes.setdefault(mod, []).append(text(a, b))

assembled = {}
for mod in ORDER:
    top = "".join(parts[mod]).strip("\n")
    chunks = [top] if top else []
    if mod in MIXIN_CLASS:
        cname, cdoc = MIXIN_CLASS[mod]
        chunks.append(f"class {cname}:\n    \"\"\"{cdoc}\"\"\"\n" + "".join(classes[mod]).rstrip("\n"))
    if mod == "scheduler":
        bases = ", ".join(MIXIN_CLASS[m][0] for m in MIXIN_CLASS)
        hdr = LINES[sched.lineno - 1]
        assert hdr == "class _Scheduler:\n", hdr
        chunks.append(f"class _Scheduler({bases}):\n" + text(sched.lineno + 1, doc_end)
                      + "".join(classes["scheduler"]).rstrip("\n"))
    assembled[mod] = "\n\n\n".join(chunks) + "\n"

# ---------------------------------------------------------------- homes
def bound_names(tree):
    out = []
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            out.append(st.name)
        elif isinstance(st, ast.Assign):
            for t in st.targets:
                out += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
        elif isinstance(st, (ast.AnnAssign, ast.AugAssign)):
            out.append(st.target.id)
        elif isinstance(st, (ast.Import, ast.ImportFrom, ast.Expr, ast.If)):
            pass
        else:
            raise ValueError(ast.dump(st)[:200])
    return out


home = {}
for mod in ORDER:
    for n in bound_names(ast.parse(assembled[mod])):
        assert n not in home or home[n] == mod, (n, home.get(n), mod)
        home[n] = mod

MUTATED = set()
for node in ast.walk(TREE):
    if isinstance(node, ast.Global):
        MUTATED.update(node.names)
print("mutated:", sorted(MUTATED))


# ---------------------------------------------------------------- qualify
def global_refs(src):
    """Names a module's code looks up in its globals (any scope)."""
    st = symtable.symtable(src, "m", "exec")
    out = set()

    def walk(t):
        for s in t.get_symbols():
            if t.get_type() == "module":
                if s.is_referenced() or s.is_assigned():
                    out.add(s.get_name())
            elif s.is_global():
                out.add(s.get_name())
        for c in t.get_children():
            walk(c)
    walk(st)
    return out


def qualify(mod, src):
    """Mutated names homed elsewhere -> alias.NAME; their `global` lines dropped."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits = []   # (line, col, old_len, new)
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            keep = [n for n in node.names if home[n] == mod]
            if len(keep) != len(node.names):
                line = lines[node.lineno - 1]
                assert line.strip().startswith("global ") and node.lineno == node.end_lineno
                if keep:
                    ind = line[:len(line) - len(line.lstrip())]
                    lines[node.lineno - 1] = f"{ind}global {', '.join(keep)}\n"
                else:
                    drop.add(node.lineno)
        if isinstance(node, ast.Name) and node.id in MUTATED and home[node.id] != mod:
            edits.append((node.lineno, node.col_offset, len(node.id),
                          f"{ALIAS[home[node.id]]}.{node.id}"))
    for ln, col, k, new in sorted(edits, reverse=True):
        s = lines[ln - 1]
        assert s[col:col + k] == new.split(".")[1], (s, new)
        lines[ln - 1] = s[:col] + new + s[col + k:]
    lines = [l for i, l in enumerate(lines, 1) if i not in drop]
    return "".join(lines), len(edits), len(drop)


# ---------------------------------------------------------------- manual edits
EDITS = {
    "config": [(
        "# Repo-relative default: <repo>/weights/<name>. worker.py lives at src/worker/,\n"
        "# so the repo root is two directories up. Override with --model or QWEN_MODEL.\n"
        "_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"..\"))\n",
        "# Repo-relative default: <repo>/weights/<name>. This file lives at\n"
        "# src/worker/qwen_worker/, so the repo root is three directories up. Override\n"
        "# with --model or QWEN_MODEL.\n"
        "_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"..\", \"..\"))\n",
    )],
    "engine": [(
        "        here = os.path.dirname(os.path.abspath(__file__))\n"
        "        src = hashlib.sha256()\n"
        "        for f in (\"worker.py\", \"pfqmm.py\", \"mrqmm.py\", \"cascade.py\", \"cascade_mma.py\", \"pcdisk.py\"):\n",
        "        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/worker\n"
        "        pkg = sorted(os.path.join(\"qwen_worker\", n) for n in os.listdir(os.path.join(here, \"qwen_worker\"))\n"
        "                     if n.endswith(\".py\"))\n"
        "        src = hashlib.sha256()\n"
        "        for f in (\"worker.py\", *pkg, \"pfqmm.py\", \"mrqmm.py\", \"cascade.py\", \"cascade_mma.py\", \"pcdisk.py\"):\n",
    )],
}

# original imports: bound name -> (group, statement)
IMPORTS = {}
for st in TREE.body:
    if isinstance(st, ast.Import):
        for a in st.names:
            name = a.asname or a.name.split(".")[0]
            stmt = f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
            top = a.name.split(".")[0]
            grp = 2 if top in LOCAL_TOP else (1 if top in ("grpc", "mlx", "numpy") else 0)
            IMPORTS[name] = (grp, stmt, None)
    elif isinstance(st, ast.ImportFrom):
        for a in st.names:
            name = a.asname or a.name
            grp = 0 if st.module == "concurrent" else 1
            IMPORTS[name] = (grp, None, (st.module, a.name))

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}


def wrap_from(prefix, names, width=100):
    s = f"{prefix} import {', '.join(names)}"
    if len(s) <= width:
        return s + "\n"
    out, line = [], f"{prefix} import ("
    ind = " " * len(line)
    for i, n in enumerate(names):
        piece = n + ("," if i < len(names) - 1 else ")")
        if len(line) + len(piece) + 1 > width and not line.endswith("("):
            out.append(line.rstrip() + "\n")
            line = ind
        line += piece + " "
    out.append(line.rstrip() + "\n")
    return "".join(out)


deps = {m: set() for m in ORDER}
final = {}
stats = {}
for mod in ORDER:
    src = assembled[mod]
    for old, new in EDITS.get(mod, []):
        assert src.count(old) == 1, (mod, old)
        src = src.replace(old, new)
    src, nq, ng = qualify(mod, src)
    stats[mod] = (nq, ng)
    refs = global_refs(src)
    mine = set(bound_names(ast.parse(src)))
    std, third, local, pkg_mods, pkg_names = [], [], [], set(), {}
    from_groups = {}
    for n in sorted(refs - mine):
        if n in BUILTINS:
            continue
        if n in ALIAS.values():
            m = next(k for k, v in ALIAS.items() if v == n)
            pkg_mods.add(m)
            deps[mod].add(m)
        elif n in home:
            assert n not in MUTATED, (mod, n)
            pkg_names.setdefault(home[n], []).append(n)
            deps[mod].add(home[n])
        elif n in IMPORTS:
            grp, stmt, frm = IMPORTS[n]
            if stmt:
                [std, third, local][grp].append(stmt)
            else:
                from_groups.setdefault((grp, frm[0]), []).append(frm[1])
        else:
            raise SystemExit(f"{mod}: unresolved global {n!r}")
    for (grp, modname), names in from_groups.items():
        # keep the original import order of the names
        order = [a.name for st in TREE.body if isinstance(st, ast.ImportFrom) and st.module == modname
                 for a in st.names]
        names = sorted(names, key=order.index)
        [std, third, local][grp].append(wrap_from(f"from {modname}", names).rstrip("\n"))

    def key(s):
        return (s.startswith("from "), s.split()[1].lower())
    blocks = []
    for g in (std, third, local):
        if g:
            blocks.append("".join(s + "\n" for s in sorted(g, key=key)))
    rel = []
    for m in ORDER:
        if m in pkg_mods:
            rel.append(f"from . import {m} as {ALIAS[m]}\n" if ALIAS[m] != m else f"from . import {m}\n")
    for m in ORDER:
        if m in pkg_names:
            rel.append(wrap_from(f"from .{m}", sorted(pkg_names[m], key=lambda x: (x.lower(), x))))
    if rel:
        blocks.append("".join(rel))
    doc = '"""' + DOC[mod] + '"""\n\n'
    final[mod] = doc + "\n".join(blocks) + "\n\n" + src

# cycles
seen, temp, topo = set(), set(), []


def visit(m, path):
    if m in temp:
        raise SystemExit(f"import cycle: {' -> '.join(path + [m])}")
    if m in seen:
        return
    temp.add(m)
    for d in sorted(deps[m]):
        visit(d, path + [m])
    temp.discard(m)
    seen.add(m)
    topo.append(m)


for m in ORDER:
    visit(m, [])
for m in ORDER:
    for d in deps[m]:
        assert ORDER.index(d) < ORDER.index(m), (m, d)

# alias names must not be locals anywhere they are introduced
for mod in ORDER:
    st = symtable.symtable(final[mod], mod, "exec")

    def walk(t):
        for s in t.get_symbols():
            if s.get_name() in ALIAS.values() and t.get_type() != "module" and not s.is_global():
                raise SystemExit(f"{mod}: alias {s.get_name()} shadowed in {t.get_name()}")
        for c in t.get_children():
            walk(c)
    walk(st)

pkg = os.path.join(OUT, "qwen_worker")
os.makedirs(pkg, exist_ok=True)
for mod in ORDER:
    with open(os.path.join(pkg, mod + ".py"), "w") as f:
        f.write(final[mod])
    print(f"{mod:14s} {final[mod].count(chr(10)):5d} lines  qualified={stats[mod][0]} "
          f"globals_dropped={stats[mod][1]}  deps={sorted(deps[mod])}")


# ---------------------------------------------------------------- package + facade
mods = ", ".join(ORDER)
init = ('"""The MLX worker as a package: worker.py is its entry point (`uv run python\n'
        'worker.py`) and a facade over it (`import worker`: every name of these modules\n'
        'as one namespace, as the single-file worker was). MODULES in import order."""\n'
        + wrap_from("from .", ORDER).replace("from . import", "from . import", 1)
        + "\n" + "\n".join(textwrap.wrap("MODULES = (" + mods + ")", 99, subsequent_indent=" " * 11))
        + "\n")
init = init.replace("from . import", "from . import", 1)
init = init.replace('"""\nfrom . ', '"""\n\nfrom . ', 1)
with open(os.path.join(pkg, "__init__.py"), "w") as f:
    f.write(init)

facade = (text(1, 17) + '''# The code lives in the qwen_worker package (module map: CLAUDE.md "Layout").
# This file is the entry point and a facade: `import worker` is the package's
# modules as one namespace, as the single-file worker.py was.
#   - Reading worker.NAME gives the live value from the module that holds it.
#   - Setting worker.NAME (tests' monkeypatch.setattr(worker, ...), scripts'
#     W.NAME = v) rebinds NAME in every package module that binds it: its home
#     and the modules that took it with `from .x import NAME` -- all of them
#     hold the same object, so the patch reaches every reader, as it did when
#     there was one module.
#   - Names the code itself rebinds at run time (`global` in their home,
#     `cfg.SDPA_FUSED = ...`) are imported by no other module (they read
#     `cfg.NAME` / `sizing._FIXED_SLAB`), so they have one binding.
# tests/test_facade.py checks both rules and that every name the tests and
# scripts use through `worker` resolves.
import sys as _sys
import types as _types

import qwen_worker as _pkg


class _Facade(_types.ModuleType):
    def __getattr__(self, name):
        for m in _pkg.MODULES:
            d = m.__dict__
            if name in d:
                return d[name]
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    @staticmethod
    def _homes(name):
        """The package modules binding `name` (all the same object, else an error)."""
        if name.startswith("__"):
            return []
        homes = [m for m in _pkg.MODULES if name in m.__dict__]
        if any(m.__dict__[name] is not homes[0].__dict__[name] for m in homes):
            raise AttributeError(
                f"worker.{name}: the modules {', '.join(m.__name__ for m in homes)} hold different "
                "objects under this name; set it on the module that owns it")
        return homes

    def __setattr__(self, name, value):
        homes = self._homes(name)
        if not homes:
            return super().__setattr__(name, value)
        for m in homes:
            setattr(m, name, value)

    def __delattr__(self, name):
        homes = self._homes(name)
        if not homes:
            return super().__delattr__(name)
        for m in homes:
            delattr(m, name)

    def __dir__(self):
        return sorted(set(super().__dir__()).union(*(vars(m) for m in _pkg.MODULES)))


''' + text(7637, 7637) + "    from qwen_worker.server import main\n" + text(7638, 7638)
          + "else:\n    _sys.modules[__name__].__class__ = _Facade\n")
with open(os.path.join(OUT, "worker.py"), "w") as f:
    f.write(facade)

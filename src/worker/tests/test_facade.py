"""worker.py is a facade over the qwen_worker package (see its header): a name
set through `worker` (monkeypatch.setattr(worker, ...), scripts' W.NAME = v)
is rebound in every package module that binds it. These checks make sure
that reaches every reader, i.e. that no test silently runs on defaults."""
import ast
import builtins
import pathlib
import re
import symtable

import pytest

import qwen_worker
import worker

WORKER = pathlib.Path(worker.__file__).resolve().parent
REPO = WORKER.parent.parent
MODULES = qwen_worker.MODULES


def _tree(m):
    return ast.parse(pathlib.Path(m.__file__).read_text())


def _aliases(m, tree):
    """`from . import x [as y]` in module m -> {y: the module x}."""
    out = {}
    for st in tree.body:
        if isinstance(st, ast.ImportFrom) and st.level == 1 and st.module is None:
            for a in st.names:
                out[a.asname or a.name] = getattr(qwen_worker, a.name)
    return out


def _global_refs(m):
    """Names module m's code looks up in its own globals, from any scope."""
    out = set()

    def walk(t):
        for s in t.get_symbols():
            if (s.is_referenced() if t.get_type() == "module" else s.is_global()):
                out.add(s.get_name())
        for c in t.get_children():
            walk(c)
    walk(symtable.symtable(pathlib.Path(m.__file__).read_text(), m.__file__, "exec"))
    return out


def _qualified_refs(m, tree):
    """(module, NAME) for every `alias.NAME` of a package module in m."""
    al = _aliases(m, tree)
    return {(al[n.value.id], n.attr) for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in al}


def _bound():
    out = {}
    for m in MODULES:
        for k in vars(m):
            if not (k.startswith("__") and k.endswith("__")):
                out.setdefault(k, []).append(m)
    return out


def test_one_object_per_name():
    """Every name bound in several package modules (a `from .x import NAME`
    beside its home, a shared stdlib import) is the same object in all of them."""
    bad = {k: [m.__name__ for m in ms] for k, ms in _bound().items()
           if any(vars(m)[k] is not vars(ms[0])[k] for m in ms)}
    assert not bad


def test_rebound_names_have_one_binding():
    """Names the code rebinds at run time (`global NAME` in their home,
    `cfg.NAME = ...` from elsewhere) live in exactly one module: a copy taken
    with `from .x import NAME` would keep the old value."""
    rebound = {}
    for m in MODULES:
        tree = _tree(m)
        al = _aliases(m, tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Global):
                for g in n.names:
                    rebound.setdefault(g, set()).add(m.__name__)
            if (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                    and isinstance(n.value, ast.Name) and n.value.id in al):
                rebound.setdefault(n.attr, set()).add(al[n.value.id].__name__)
    assert {"SDPA_FUSED", "DRAFT_BLOCK", "BATCH_INVARIANT", "_FIXED_SLAB", "_VERIFY_ATT_Q"} <= set(rebound)
    bound = _bound()
    for name, homes in rebound.items():
        assert len(homes) == 1, (name, homes)
        assert [m.__name__ for m in bound[name]] == list(homes), (name, homes)


def test_every_global_is_bound_where_it_is_read():
    """Each module binds every global it reads (no NameError at run time) and
    every `alias.NAME` it reads exists in the aliased module."""
    for m in MODULES:
        missing = {n for n in _global_refs(m) - set(vars(m)) if not hasattr(builtins, n)}
        assert not missing, (m.__name__, missing)
        for mod, name in _qualified_refs(m, _tree(m)):
            assert hasattr(mod, name), (m.__name__, mod.__name__, name)


def test_a_patch_through_worker_reaches_every_reader(monkeypatch):
    """For EVERY module-level name of the package: set through `worker`, it
    is what every module reading it sees (bare or as alias.NAME)."""
    readers = {}
    for m in MODULES:
        tree = _tree(m)
        for n in _global_refs(m):
            readers.setdefault(n, []).append((m, n))
        for mod, n in _qualified_refs(m, tree):
            readers.setdefault(n, []).append((mod, n))
    names = sorted(set(_bound()) & set(readers))
    assert len(names) > 250
    for name in names:
        sentinel = object()
        with monkeypatch.context() as mp:
            mp.setattr(worker, name, sentinel)
            assert getattr(worker, name) is sentinel
            for mod, n in readers[name]:
                assert vars(mod)[n] is sentinel, (name, mod.__name__)
        for mod, n in readers[name]:
            assert vars(mod)[n] is not sentinel


def _users():
    """Names used as worker.NAME / W.NAME in the tests and scripts (files that
    import worker), and the string names of setattr(worker, "NAME", ...)."""
    files = list((WORKER / "tests").glob("*.py")) + list((REPO / "scripts").rglob("*.py"))
    names = {}
    for f in files:
        tree = ast.parse(f.read_text())
        al = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
              for a in n.names if a.name == "worker"}
        # helpers handed the module (scripts/perf/prof/variants_pf.py: make(W, ...))
        al |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               for a in n.args.args if a.arg == "W"}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in al:
                names.setdefault(n.attr, set()).add(f.name)
            if (isinstance(n, ast.Call) and len(n.args) >= 2 and isinstance(n.args[0], ast.Name)
                    and n.args[0].id in al and isinstance(n.args[1], ast.Constant)
                    and isinstance(n.args[1].value, str)
                    and getattr(n.func, "attr", getattr(n.func, "id", None)) == "setattr"):
                names.setdefault(n.args[1].value, set()).add(f.name)
    return names


def test_names_used_through_worker_resolve():
    names = _users()
    assert len(names) > 140
    # gone from worker.py before the package split (ab_prefill.py checks hasattr;
    # kl_tail.py is a stale probe of the old slab reservation)
    stale = {"_reserve_kv"}
    missing = {n: sorted(f) for n, f in names.items() if not hasattr(worker, n) and n not in stale}
    assert not missing


@pytest.mark.parametrize("name", ["PROMPT_CACHE_GB", "time", "_block_sdpa", "_VERIFY_ATT_Q"])
def test_patch_and_undo(monkeypatch, name):
    old = getattr(worker, name)
    homes = [m for m in MODULES if name in vars(m)]
    with monkeypatch.context() as mp:
        mp.setattr(worker, name, 12345)
        assert all(vars(m)[name] == 12345 for m in homes)
    assert all(vars(m)[name] is old for m in homes)

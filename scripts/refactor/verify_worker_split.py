"""Mechanical check that the package is a pure move of the old worker.py.

  git show 46aa1dd:src/worker/worker.py > /tmp/worker_old.py
  cd src/worker && uv run python ../../scripts/refactor/verify_worker_split.py /tmp/worker_old.py .

(46aa1dd = the last commit with the single-file worker; split_worker.py beside
this file generated the package from it.)

1. AST: every function / class / method (by qualified name, mixin methods as
   _Scheduler.<name>) and every top-level statement of the old file has an
   identical AST in the new modules, after undoing the qualification
   (cfg.NAME / sizing.NAME -> NAME for the rebound names) and dropping the
   `global` names that became qualified. Imports and the new modules'
   docstrings / mixin headers are excluded.
2. Lines: the multiset of non-blank source lines, old vs new, with the same
   two normalizations; prints every line on one side only.
"""
import ast
import collections
import os
import re
import sys

OLD, NEW = sys.argv[1], sys.argv[2]
PKG = os.path.join(NEW, "qwen_worker")
QUAL = {"cfg": {"SDPA_FUSED", "DRAFT_BLOCK", "BATCH_INVARIANT"}, "sizing": {"_FIXED_SLAB"}}
QNAMES = set().union(*QUAL.values())
MIXINS = {"_AdmitMixin", "_PrefillMixin", "_SpecMixin", "_StoreMixin", "_CascadeMixin",
          "_DecodeMixin"}


class Unqualify(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in QUAL and node.attr in QUAL[node.value.id]:
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node

    def visit_Global(self, node):
        names = [n for n in node.names if n not in QNAMES]
        if not names:
            return None
        node.names = names
        return node


def dump(node):
    return ast.dump(Unqualify().visit(node), include_attributes=False)


def units(tree, mixin_as=None):
    """qualified name -> dump; plus the list of other top-level statements' dumps."""
    out, other = {}, []
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[st.name] = dump(st)
        elif isinstance(st, ast.ClassDef):
            cname = "_Scheduler" if st.name in MIXINS else st.name
            body = st.body
            if st.name in MIXINS:
                body = body[1:]                          # the mixin's own docstring
            rest = []
            for m in body:
                if isinstance(m, ast.FunctionDef):
                    key = f"{cname}.{m.name}"
                    assert key not in out, key
                    out[key] = dump(m)
                else:
                    rest.append(dump(m))
            if st.name not in MIXINS:
                # bases + decorators + non-method body; _Scheduler's bases are the mixins (plumbing)
                bases = [] if cname == "_Scheduler" else [dump(b) for b in st.bases]
                out[f"{cname}.<class>"] = repr((bases, [dump(d) for d in st.decorator_list],
                                                [dump(k) for k in st.keywords], rest))
        else:
            other.append(dump(st))
    return out, other


old_tree = ast.parse(open(OLD).read())
old_units, old_other = units(old_tree)
old_other = [d for d in old_other if "Name(id='__name__'" not in d]          # if __name__ == ...
old_other = old_other[1:]                                                      # module docstring

new_units, new_other = {}, []
files = sorted(f for f in os.listdir(PKG) if f.endswith(".py") and f != "__init__.py")
for f in files:
    tree = ast.parse(open(os.path.join(PKG, f)).read())
    assert isinstance(tree.body[0], ast.Expr)                                 # the new docstring
    tree.body = tree.body[1:]
    u, o = units(tree)
    for k, v in u.items():
        assert k not in new_units, (k, f)
        new_units[k] = v
    new_other += o

missing = sorted(set(old_units) - set(new_units))
extra = sorted(set(new_units) - set(old_units))
changed = sorted(k for k in set(old_units) & set(new_units) if old_units[k] != new_units[k])
print(f"AST units: old {len(old_units)}, new {len(new_units)}; missing {missing}; extra {extra}")
print(f"AST units changed: {changed}")
oc, nc = collections.Counter(old_other), collections.Counter(new_other)
print(f"top-level statements: old {sum(oc.values())}, new {sum(nc.values())}, "
      f"old-only {sum((oc - nc).values())}, new-only {sum((nc - oc).values())}")
for d in (oc - nc):
    print("  OLD-ONLY", d[:200])
for d in (nc - oc):
    print("  NEW-ONLY", d[:200])

# ---- lines
def norm(line):
    for a, names in QUAL.items():
        line = re.sub(rf"\b{a}\.({'|'.join(names)})\b", r"\1", line)
    return line.rstrip("\n")


def code_lines(path, skip_ranges):
    lines = open(path).read().splitlines()
    skip = set()
    for a, b in skip_ranges:
        skip.update(range(a, b + 1))
    return [norm(l) for i, l in enumerate(lines, 1) if i not in skip and l.strip()]


def skips(tree, new):
    out = []
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            out.append((st.lineno, st.end_lineno))
        if isinstance(st, ast.ClassDef) and st.name in MIXINS:
            out.append((st.lineno, st.body[0].end_lineno))       # class line + docstring
        if isinstance(st, ast.ClassDef) and st.name == "_Scheduler":
            out.append((st.lineno, st.lineno))
    if new:
        out.append((tree.body[0].lineno, tree.body[0].end_lineno))
    return out


old_src = open(OLD).read()
ot = ast.parse(old_src)
old_skip = skips(ot, False) + [(1, 17)]
main_if = ot.body[-1]
old_skip.append((main_if.lineno, main_if.end_lineno))
old_lines = collections.Counter(code_lines(OLD, old_skip))
new_lines = collections.Counter()
for f in files:
    p = os.path.join(PKG, f)
    new_lines.update(code_lines(p, skips(ast.parse(open(p).read()), True)))
print(f"lines: old {sum(old_lines.values())}, new {sum(new_lines.values())}")
for l in sorted((old_lines - new_lines).elements()):
    print("  OLD-ONLY |" + l)
for l in sorted((new_lines - old_lines).elements()):
    print("  NEW-ONLY |" + l)
# the facade keeps the old docstring and the __main__ guard
fac = open(os.path.join(NEW, "worker.py")).read().splitlines()
assert fac[:17] == old_src.splitlines()[:17], "facade docstring"
print("facade: lines 1-17 (shebang + docstring) identical")

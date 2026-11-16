"""First GPU contact of vatt.py's q8 kernels (QWEN_VATT_Q8_KERNEL = tile / reg / regf), a few
seconds, no model:

  1. compile + numbers: every kernel (reg / regf at every tile of --tiles) on 12 shapes (n 1..33000,
     T 1..16, 27B and 9B-like head layouts, garbage past n): rms rel error vs the fp32 reference
     over mx.dequantize's fp16 KV ("ref") and over the affine values ("exact"). Pass: tile / reg
     <= 4e-4 vs ref, regf <= 4e-4 vs exact, no NaN (the tests' bound; the gate's 2.5e-4 is the
     microbench's);
  2. the worker's load self-test (kernels._vatt_q8_selftest, the body of _probe_vatt_q8, unchanged)
     for each kernel, 27B head layout, T 1..16: "" = the knob would turn on;
  3. the timing variants (VATT_Q8_TIMING 1..3) of tile and reg compile and run;
  4. the fp16 path (vatt.attention) is bit-identical to the one of --base (a git revision; its
     vatt.py is loaded beside this one under other kernel names) at 8 shapes.

Prints one line per item and "ALL OK" / "FAILED: ..." last; exit code 1 on a failure. A compile
error prints the Metal log's head (--log N characters).

  cd src/worker && .venv/bin/python ../../scripts/perf/dflash2/vatt_q8_check.py [--base cbd0e7a] [--json f]
"""
import importlib.util, json, os, subprocess, sys, tempfile, types
import mlx.core as mx
sys.path.insert(0, os.getcwd())
import vatt  # noqa: E402
from qwen_worker import kernels  # noqa: E402


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


BASE = opt("--base", "cbd0e7a")
LOG = int(opt("--log", "3000"))
JS = opt("--json", "")
DTILES = opt("--decode-tiles", "1x2x1,1x4x1,1x2x2,1x1x1").split(",")
VTILES = opt("--verify-tiles", "2x4x1,3x4x1,2x2x1,1x2x1,2x4x2,1x1x1").split(",")
D = 256
R0 = (vatt.Q8R_DECODE, vatt.Q8R_VERIFY)
fails, res = [], dict(cases={}, selftest={}, timing={}, fp16_bits=None)
print("vatt:", os.path.relpath(vatt.__file__), "| default kernel", vatt.Q8_KERNEL, "| reg tiles", R0, flush=True)


def rr(a, ref):
    d = a.astype(mx.float32) - ref
    return (mx.sqrt(mx.mean(d * d)) / mx.sqrt(mx.mean(ref * ref))).item()


def kvq(n, hkv, spare=37, seed=0):
    mx.random.seed(seed)
    k = mx.random.normal((1, hkv, n, D)).astype(mx.float16)
    v = mx.random.normal((1, hkv, n, D)).astype(mx.float16)
    pad = [(0, 0), (0, 0), (0, spare), (0, 0)]
    kq = [mx.pad(a, pad) for a in mx.quantize(k, 64, 8)]
    vq = [mx.pad(a, pad) for a in mx.quantize(v, 64, 8)]
    kq[0][..., n:, :] = 0x7f3a91c5
    vq[0][..., n:, :] = 0x12ffee01
    kq[1][..., n:, :] = 1000.0          # garbage past n: never read
    vq[1][..., n:, :] = float("nan")
    mx.eval(kq, vq)
    return tuple(kq), tuple(vq)


# 1. every kernel and tile
broken = set()
for (n, T, hq, hkv) in [(600, 1, 24, 4), (600, 8, 24, 4), (1, 1, 24, 4), (9, 8, 24, 4), (5000, 1, 24, 4),
                        (5000, 3, 24, 4), (5000, 16, 24, 4), (4099, 8, 24, 4), (3000, 8, 16, 4), (3000, 1, 16, 2),
                        (33000, 8, 24, 4), (33000, 1, 24, 4)]:
    kq, vq = kvq(n, hkv, seed=n % 7)
    mx.random.seed(T)
    q = (mx.random.normal((1, hq, T, D)) * 2).astype(mx.float16)
    sc = D ** -0.5
    ref, refx = vatt.reference_q8(q, kq, vq, n, sc), vatt.reference_q8(q, kq, vq, n, sc, exact=True)
    mx.eval(ref, refx)
    tiles = DTILES if hq // hkv * T <= 8 else VTILES
    line = []
    for name in ["tile"] + [f"{k}:{t}" for k in ("reg", "regf") for t in tiles]:
        if name in broken:
            continue
        kernel, _, tile = name.partition(":")
        if tile:
            vatt.Q8R_DECODE = vatt.Q8R_VERIFY = tuple(int(x) for x in tile.split("x"))
        try:
            o = vatt.attention_q8(q, kq, vq, n, sc, kernel=kernel)
            mx.eval(o)
            e, ex, nan = rr(o, ref), rr(o, refx), bool(mx.any(mx.isnan(o)).item())
            ok = not nan and (ex if kernel == "regf" else e) <= 4e-4
            res["cases"][f"{n}/{T}/{hq}/{hkv}/{name}"] = dict(rel_ref=e, rel_exact=ex, nan=nan, ok=ok)
            line.append(f"{name} {e:.1e}/{ex:.1e}{'' if ok else ' BAD'}")
            if not ok:
                fails.append(f"{name} n={n} T={T} {hq}/{hkv}: vs ref {e:.2e}, vs exact {ex:.2e}{', NaN' if nan else ''}")
        except Exception as ex_:                   # a compile error: once per kernel
            broken.add(name)
            fails.append(f"{name}: {type(ex_).__name__}")
            res["cases"][f"{n}/{T}/{hq}/{hkv}/{name}"] = dict(failed=str(ex_)[:LOG])
            print(f"{name} FAILED at n={n} T={T}: {str(ex_)[:LOG]}", flush=True)
    print(f"n={n:5d} T={T:2d} {hq}/{hkv}  (vs ref/vs exact)  " + " | ".join(line), flush=True)
vatt.Q8R_DECODE, vatt.Q8R_VERIFY = R0

# 2. the worker's load self-test per kernel (default reg tiles)
lm = types.SimpleNamespace(args=types.SimpleNamespace(num_attention_heads=24, num_key_value_heads=4, head_dim=D))
k0 = vatt.Q8_KERNEL
for kernel in vatt.Q8_KERNELS:
    vatt.Q8_KERNEL = kernel
    why, worst = kernels._vatt_q8_selftest(lm, tuple(range(1, vatt.MAX_T + 1)))
    res["selftest"][kernel] = dict(why=why, worst=worst)
    print(f"load self-test, QWEN_VATT_Q8_KERNEL={kernel}: {'PASS' if not why else 'FAIL: ' + why[:LOG]} "
          f"(worst rms rel error vs ref {worst:.2e})", flush=True)
    if why and kernel != "regf":                   # regf vs the fp16 reference: a gate decision, not a bug
        fails.append(f"self-test {kernel}")
vatt.Q8_KERNEL = k0

# 3. timing variants
kq, vq = kvq(2000, 4, seed=1)
for kernel in ("tile", "reg"):
    for t in (1, 2, 3):
        try:
            for T in (1, 8):
                q = mx.random.normal((1, 24, T, D)).astype(mx.float16)
                mx.eval(vatt.attention_q8(q, kq, vq, 2000, D ** -0.5, kernel=kernel, timing=t))
            res["timing"][f"{kernel}@{t}"] = True
        except Exception as ex_:
            res["timing"][f"{kernel}@{t}"] = str(ex_)[:LOG]
            fails.append(f"timing {kernel}@{t}")
            print(f"timing variant {kernel}@{t} FAILED: {str(ex_)[:LOG]}", flush=True)
print("timing variants:", {k: v is True for k, v in res["timing"].items()}, flush=True)

# 4. the fp16 path's bits vs the base revision's vatt.py
try:
    src = subprocess.run(["git", "show", f"{BASE}:src/worker/vatt.py"], capture_output=True, text=True,
                         check=True).stdout.replace('name=f"vatt', 'name=f"vbase')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "vatt_base.py")
        open(path, "w").write(src)
        spec = importlib.util.spec_from_file_location("vatt_base", path)
        vb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vb)
    same = []
    for (n, T) in [(700, 1), (700, 8), (5000, 16), (12345, 8), (33000, 1), (33000, 8), (81920, 8), (81920, 1)]:
        mx.random.seed(n + T)
        k = mx.random.normal((1, 4, n + 50, D)).astype(mx.float16)
        v = mx.random.normal((1, 4, n + 50, D)).astype(mx.float16)
        q = (mx.random.normal((1, 24, T, D)) * 2).astype(mx.float16)
        same.append(bool(mx.array_equal(vatt.attention(q, k, v, n, D ** -0.5), vb.attention(q, k, v, n, D ** -0.5)).item()))
    res["fp16_bits"] = same
    print(f"fp16 path bit-identical to {BASE} at 8 shapes: {same}", flush=True)
    if not all(same):
        fails.append("fp16 bits")
except Exception as ex_:
    fails.append("fp16 bits check did not run")
    print(f"fp16 bit check FAILED to run: {str(ex_)[:LOG]}", flush=True)

res["fails"] = fails
if JS:
    os.makedirs(os.path.dirname(JS) or ".", exist_ok=True)
    json.dump(res, open(JS, "w"), indent=1)
print("ALL OK" if not fails else "FAILED: " + "; ".join(fails), flush=True)
sys.exit(1 if fails else 0)

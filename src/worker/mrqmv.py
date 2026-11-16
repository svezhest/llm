"""Multi-row 4-bit qmv for short verify blocks (QWEN_VERIFY_QMV): T = 2..MAX_ROWS
rows of x against 4-bit affine g64 weights, each row's result bit for bit
MLX's own single-row kernel (affine_qmv_fast) on that row alone.

An MTP round on the 27B verifies 2..5 tokens. mrqmm (simdgroup MMA, rows padded
to 8) costs a flat ~x1.6 of a solo row there, mlx-vlm's exact multi-row qmv
(quantized_verifier: per row its own unpack of every weight, so T x the ALU work)
more. Here each thread does MLX's qmv_fast work - 16 weights per 512-wide block,
4 output features per simdgroup, 2 simdgroups - with the weights unpacked once
per block into registers and T dot products over them:

  * load_vector (MLX's 4-bit form): x[i], x[i+1] / 16, x[i+2] / 256,
    x[i+3] / 4096 and the block's sum, per row;
  * qdot (MLX's 4-bit form): accum += x0 * (w & 0x000f) + x1 * (w & 0x00f0)
    + x2 * (w & 0x0f00) + x3 * (w & 0xf000) per uint16, the masked weights as
    floats computed once for all T rows;
  * result += scale * accum + sum * bias per block; simd_sum; one cast to T.

The arithmetic of a row is MLX's for one row in the same order, so the bits
are the same as long as the compiler contracts it the same way: checked at
load on the first weights offered (`_selftest`: rows 2..MAX_ROWS vs
mx.quantized_matmul of each row alone, mx.array_equal), and a mismatch or
error leaves every call on the caller's own path (the knob's promise is
"the verify == T plain steps", not just speed).

Shapes: K % 512 == 0 (qmv_fast's own condition), N % 8 == 0, fp16 x / scales /
biases, (1, T, K) inputs. Everything else stays with the caller.
"""
from __future__ import annotations

import logging
import threading

import mlx.core as mx

log = logging.getLogger("worker.mrqmv")

BITS, GROUP = 4, 64
BLOCK = 512                     # K per iteration: 32 threads x 16 values
RPS, NSG = 4, 2                 # output features per simdgroup, simdgroups per threadgroup
CHUNK = 3                       # rows whose x a thread holds at once (registers)
MAX_ROWS = 8                    # set by install (QWEN_VERIFY_QMV)

_SRC = r"""
    const uint n_tile = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_simdgroup;
    constexpr int KW = K / 4;                 // uint16 per weight row (4 values each)
    constexpr int KG = K / 64;                // groups per row
    const int out_row = int(n_tile) * (2 * 4) + int(sg) * 4;

    const device uint16_t* ws = (const device uint16_t*)w + out_row * KW + lid * 4;
    const device half* sc = scales + out_row * KG + lid / 4;
    const device half* bi = biases + out_row * KG + lid / 4;
    const device half* xk = x + lid * 16;

    float result[TT][4];
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < TT; t++) {
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 4; r++)
            result[t][r] = 0.0f;
    }
    float xt[CH][16];
    float sum[CH];

    for (int k = 0; k < K; k += 512) {
        // rows in chunks of CH (registers: x of CH rows at a time); the
        // chunk's second pass over the weights reads them from the cache
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < TT; c += CH) {
            _Pragma("clang loop unroll(full)")
            for (int u = 0; u < CH; u++) {
                if (c + u < TT) {
                    const device half* xr = xk + (c + u) * K;
                    float s = 0.0f;
                    _Pragma("clang loop unroll(full)")
                    for (int i = 0; i < 16; i += 4) {
                        s += xr[i] + xr[i + 1] + xr[i + 2] + xr[i + 3];
                        xt[u][i] = xr[i];
                        xt[u][i + 1] = xr[i + 1] / 16.0f;
                        xt[u][i + 2] = xr[i + 2] / 256.0f;
                        xt[u][i + 3] = xr[i + 3] / 4096.0f;
                    }
                    sum[u] = s;
                }
            }
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < 4; r++) {
                const device uint16_t* wl = ws + r * KW;
                const float s = sc[r * KG];
                const float b = bi[r * KG];
                float accum[CH];
                _Pragma("clang loop unroll(full)")
                for (int u = 0; u < CH; u++)
                    accum[u] = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < 4; i++) {
                    const uint16_t p = wl[i];
                    const float w0 = float(p & 0x000f), w1 = float(p & 0x00f0);
                    const float w2 = float(p & 0x0f00), w3 = float(p & 0xf000);
                    _Pragma("clang loop unroll(full)")
                    for (int u = 0; u < CH; u++) {
                        if (c + u < TT) {
                            accum[u] +=
                                (xt[u][4 * i] * w0 +
                                 xt[u][4 * i + 1] * w1 +
                                 xt[u][4 * i + 2] * w2 +
                                 xt[u][4 * i + 3] * w3);
                        }
                    }
                }
                _Pragma("clang loop unroll(full)")
                for (int u = 0; u < CH; u++)
                    if (c + u < TT)
                        result[c + u][r] += s * accum[u] + sum[u] * b;
            }
        }
        ws += 512 / 4;
        sc += 512 / 64;
        bi += 512 / 64;
        xk += 512;
    }

    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; r++) {
        _Pragma("clang loop unroll(full)")
        for (int t = 0; t < TT; t++) {
            float v = simd_sum(result[t][r]);
            if (lid == 0) {
                y[t * N + out_row + r] = half(v);
            }
        }
    }
"""

_KERNELS = {}                   # (T, K, N) -> metal_kernel
_LOCK = threading.Lock()
_OK = {}                        # (K, N) -> self-test verdict (None = passed, else why not)


def _kernel(T: int, K: int, N: int):
    key = (T, K, N)
    k = _KERNELS.get(key)
    if k is None:
        with _LOCK:
            k = _KERNELS.get(key)
            if k is None:
                consts = f"    constexpr int TT = {T}, K = {K}, N = {N}, CH = {min(T, CHUNK)};\n"
                k = _KERNELS[key] = mx.fast.metal_kernel(
                    name=f"mrqmv4_g64_t{T}_k{K}_n{N}", input_names=["x", "w", "scales", "biases"],
                    output_names=["y"], source=consts + _SRC)
    return k


def shape_ok(w: mx.array, scales: mx.array, biases, bits: int = BITS, group_size: int = GROUP) -> bool:
    """4-bit g64 affine fp16 weights of a shape qmv_fast takes."""
    if biases is None or w.dtype != mx.uint32 or w.ndim != 2 or scales.ndim != 2:
        return False
    N, K = w.shape[0], w.shape[1] * 32 // BITS
    return (bits == BITS and group_size == GROUP and scales.dtype == mx.float16
            and biases.dtype == mx.float16 and K % BLOCK == 0 and N % 8 == 0
            and scales.shape == (N, K // GROUP))


def rows(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """x (T, K) fp16, 2 <= T <= MAX_ROWS -> (T, N): row t ==
    mx.quantized_matmul(x[t:t + 1], ...) bit for bit on shapes `tested`
    passed. Call shape_ok first."""
    T, K = x.shape
    N = w.shape[0]
    return _kernel(T, K, N)(inputs=[x, w, scales, biases], grid=(32, N // RPS, 1),
                            threadgroup=(32, NSG, 1), output_shapes=[(T, N)],
                            output_dtypes=[mx.float16])[0]


def _selftest(w, scales, biases, max_rows) -> str | None:
    """None when every row of 2..max_rows-row calls equals MLX's own 1-row
    quantized_matmul of it, bit for bit, on these weights (random x at a
    realistic scale, one row of large values); else why not."""
    K = scales.shape[1] * GROUP
    for T in range(2, max_rows + 1):
        x = mx.random.normal((T, K), key=mx.random.key(T)) * 0.5
        if T >= 3:
            x = mx.concatenate([x[:2], x[2:3] * 40, x[3:]])
        x = x.astype(mx.float16)
        y = rows(x, w, scales, biases)
        for t in range(T):
            one = mx.quantized_matmul(x[t:t + 1], w, scales, biases, transpose=True,
                                      group_size=GROUP, bits=BITS)
            if not mx.array_equal(y[t:t + 1], one).item():
                d = mx.abs(y[t:t + 1].astype(mx.float32) - one.astype(mx.float32)).max().item()
                return f"{T} rows: row {t} differs from its solo qmv (max |d| {d:.3g})"
    return None


def tested(w, scales, biases, max_rows: int = MAX_ROWS) -> bool:
    """The self-test's verdict for this weight shape, run once per (K, N) per
    process (scheduler thread or load)."""
    key = (w.shape[1] * 32 // BITS, w.shape[0])
    if key not in _OK:
        try:
            _OK[key] = _selftest(w, scales, biases, max_rows)
        except Exception as e:  # noqa: BLE001 - any failure = off for this shape
            _OK[key] = f"error: {e}"
        if _OK[key] is not None:
            log.warning("mrqmv: self-test FAILED for K %d N %d, those verify calls keep their "
                        "old path: %s", *key, _OK[key])
    return _OK[key] is None


def install(lm, max_rows: int, fused=()) -> tuple[int, int]:
    """Marks every 4-bit g64 quantized linear of `lm` whose shape passes the
    self-test (module.__dict__["_qmv"] = True) and the fused GDN in-projection
    groups `fused` hands over ((module, [(member positions, w, scales, biases,
    split), ...], ids of the four modules) -> its __dict__["_qmv_fused"]: the
    groups that pass). -> (modules, layers with a fused group) marked."""
    global MAX_ROWS
    MAX_ROWS = max(2, min(int(max_rows), 3))   # 4+ rows spill registers (2026-11-15: x5-x15 slower)
    n = 0
    for _, m in lm.named_modules():
        if not isinstance(m, dict):
            continue
        w, sc, b = (m.get(k) for k in ("weight", "scales", "biases"))
        if (sc is None or not isinstance(w, mx.array) or getattr(m, "mode", "affine") != "affine"
                or not shape_ok(w, sc, b, getattr(m, "bits", 0), getattr(m, "group_size", 0))):
            continue
        if tested(w, sc, b, MAX_ROWS):
            m.__dict__["_qmv"] = True
            n += 1
    nf = 0
    for m, groups, ids in fused:
        ok = tuple(g for g in groups if shape_ok(*g[1:4]) and tested(*g[1:4], MAX_ROWS))
        if ok:
            m.__dict__["_qmv_fused"] = (ids, ok)
            nf += 1
    return n, nf

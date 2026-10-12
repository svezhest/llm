"""Multi-row 8-bit quantized matmul for batched decode on Apple GPUs (M1/M2).

mx.quantized_matmul runs one qmv per row below ~6 rows and a qmm (32-row
tiles, weights dequantized through threadgroup memory) above: on M1 an 8-bit
decode step costs x1.9 at 4 rows and a flat x2.6 from 6 to 32 rows. This
kernel reads each weight once for up to 16 rows of x with simdgroup MMAs:

    y[m, n] = sum_k x[m, k] * (s[n, g] * q[n, k] + b[n, g])     (fp32 accumulate)

computed as C^T = W . x^T. Each simdgroup owns 8*NA output features (NA 8x8
A fragments) and all M (padded to 8 or 16) rows; the KS simdgroups of a
threadgroup split K (64-wide groups, interleaved) and add their partial sums
through threadgroup memory once at the end. No threadgroup memory and no
barrier in the K loop:
  * The k order inside a 64-group is permuted (the same for A and B, so the
    dot product is unchanged): logical k c (0..7) of MMA step st (0..7) is
    physical k 8*c + st.
  * A (weights) is built in registers: per 64-group each thread loads ONE
    16-byte uint4 per fragment (8 rows x 64 bytes per simdgroup = 8 full
    cache lines); its two A elements of step st are byte st%4 of words st/4
    and 2 + st/4. Two weights become a half2 with the 0x6400 trick (mask,
    insert, or, subtract 1024) and one half2 fma with the group's scale and
    bias (fp16 rounding of the dequantized weight, as MLX's qmm loader).
  * B (x^T) is read straight from device memory (x is tiny, L1/L2-resident):
    per 64-group a thread's B elements of each of its two x rows are one
    16-byte uint4 (8 consecutive halves). This order costs a few more integer
    ops on A than one where both A elements sit in one word, but needs a
    quarter of the x load instructions: equal at 8 rows, -12% at 16.
Rows m >= M0 of the padded M are computed from row 0 and never stored.

Used only for 8-bit affine, group size 64, fp16 x/scales/biases, 3 <= rows
<= 16 of a decode-shaped input (..., 1, K), K % 64 == 0, N % 8 == 0, N >= 512
(MIN_N); every other call goes to mx.quantized_matmul unchanged (M <= 2 stays
bit-identical to a solo row).

    import mrqmm; mrqmm.install(model)        # QWEN_MRQMM=0 disables

The GDN decode in-projections (qkv, z, b, a as one matmul over concatenated
weights) are fused by the worker (`worker._install_gdn_fuse`, QWEN_GDN_FUSE),
which calls `mrqmm` for 3..16 rows itself; mlx-vlm 0.7 has no fused hook of
its own (0.6.2's `_decode_quantized_linears_fused`, which this module used to
route, is gone).
"""
from __future__ import annotations

import logging
import os
import threading

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger("worker.mrqmm")

MIN_ROWS = 3
MAX_ROWS = 16
# Narrower layers stay on quantized_matmul: launch-bound, no gain (35B MoE
# router N=256: x0.96-1.03 at 3..16 rows), and the routing top-k keeps the
# library's numerics. The worker's fused GDN in-projection is judged by its
# total N.
MIN_N = 512
BITS = 8
GROUP = 64

_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# Constants (prepended per kernel, see _get_kernel): K, N, MB (M/8: 1 or 2),
# NA (A frags per simdgroup), KS (simdgroups splitting K); T = half.
# Input `m0` = int32[1] valid rows.
_SRC = r"""
    constexpr int G = K / 64;
    threadgroup float2 red[(KS > 1 ? KS - 1 : 1) * NA * MB * 32];
    const uint lid = thread_index_in_simdgroup;
    const int ks = int(simdgroup_index_in_threadgroup);
    const int n0 = int(threadgroup_position_in_grid.x) * (8 * NA);
    const int M0 = m0[0];
    const short qid = short(lid) / 4;
    const short fm = (qid & 4) + ((short(lid) / 2) % 4);
    const short fn = (qid & 2) * 2 + (short(lid) % 2) * 2;

    // weights: thread reads bytes [16*(fn/2), +16) of each 64-byte group of row n0+8i+fm
    const device uint4* wp[NA];
    const device T* sp[NA];
    const device T* bp[NA];
    bool act[NA];
    for (int i = 0; i < NA; ++i) {
      act[i] = n0 + 8 * i < N;
      const int r = act[i] ? n0 + 8 * i + fm : fm;
      wp[i] = (const device uint4*)((const device uint8_t*)w + size_t(r) * K) + fn / 2;
      sp[i] = scales + size_t(r) * G;
      bp[i] = biases + size_t(r) * G;
    }
    // x: thread's B elements of step st are x[8mb+fn+e][g*64 + 8*fm + st], i.e.
    //    half st%2 of word st/2 of uint4 fm of the group (8 uint4)
    const device uint4* xp[MB][2];
    for (int mb = 0; mb < MB; ++mb)
      for (int e = 0; e < 2; ++e) {
        const int m = 8 * mb + fn + e;
        xp[mb][e] = (const device uint4*)(x + size_t(m < M0 ? m : 0) * K) + fm;
      }

    simdgroup_matrix<float, 8, 8> C[NA][MB];
    for (int i = 0; i < NA; ++i)
      for (int mb = 0; mb < MB; ++mb) C[i][mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // K loop over 64-groups, the KS simdgroups interleaved (adjacent groups of a
    // row are read by sibling simdgroups at the same time). Software pipelining
    // (loading the next group's weights before this group's MMAs), 2-4 groups
    // per iteration were measured slower or equal:
    // the extra registers cost more occupancy than the latency they hide.
    for (int g = ks; g < G; g += KS) {
      uint4 xw[MB][2];
      for (int mb = 0; mb < MB; ++mb)
        for (int e = 0; e < 2; ++e) xw[mb][e] = xp[mb][e][8 * g];
      uint4 wq[NA];
      vec<T, 2> s2[NA], b2[NA];
      for (int i = 0; i < NA; ++i) {
        wq[i] = wp[i][4 * g];
        s2[i] = vec<T, 2>(sp[i][g]);
        b2[i] = vec<T, 2>(bp[i][g]);
      }
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        simdgroup_matrix<T, 8, 8> B[MB];
        for (int mb = 0; mb < MB; ++mb) {
          // B elements (x[8mb+fn][k], x[8mb+fn+1][k]): half st%2 of word st/2 of each row
          const uint u0 = xw[mb][0][st / 2], u1 = xw[mb][1][st / 2];
          const uint u = (st % 2) ? insert_bits(u1, u0 >> 16, 0u, 16u) : insert_bits(u0, u1, 16u, 16u);
          const vec<T, 2> h = as_type<vec<T, 2>>(u);
          B[mb].thread_elements()[0] = h.x;
          B[mb].thread_elements()[1] = h.y;
        }
        for (int i = 0; i < NA; ++i) {
          // A elements (k = 8fn + st, 8fn + 8 + st): byte st%4 of words st/4 and 2 + st/4
          // -> half2 (0x6400 | q) - 1024 = q, exact
          const uint w0 = wq[i][st / 4], w1 = wq[i][2 + st / 4];
          const uint m0 = ((st % 2) ? (w0 >> 8) : w0) & 0x00ff00ffu;
          const uint m1 = ((st % 2) ? (w1 >> 8) : w1) & 0x00ff00ffu;
          const uint q = ((st / 2) % 2) ? insert_bits(m1, m0 >> 16, 0u, 16u) : insert_bits(m0, m1, 16u, 16u);
          const vec<T, 2> a = fma(as_type<vec<T, 2>>(q | 0x64006400u) - vec<T, 2>(1024), s2[i], b2[i]);
          simdgroup_matrix<T, 8, 8> A;
          A.thread_elements()[0] = a.x;
          A.thread_elements()[1] = a.y;
          for (int mb = 0; mb < MB; ++mb) simdgroup_multiply_accumulate(C[i][mb], A, B[mb], C[i][mb]);
        }
      }
    }

    // thread holds C[i][mb] at (n = n0+8i+fm, m = 8mb+fn+{0,1})
    float2 acc[NA][MB];
    for (int i = 0; i < NA; ++i)
      for (int mb = 0; mb < MB; ++mb) {
        auto e = C[i][mb].thread_elements();
        acc[i][mb] = float2(e[0], e[1]);
      }
    if (KS > 1) {
      if (ks > 0)
        for (int i = 0; i < NA; ++i)
          for (int mb = 0; mb < MB; ++mb) red[(((ks - 1) * NA + i) * MB + mb) * 32 + lid] = acc[i][mb];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ks > 0) return;
      for (int s = 0; s < KS - 1; ++s)
        for (int i = 0; i < NA; ++i)
          for (int mb = 0; mb < MB; ++mb) acc[i][mb] += red[((s * NA + i) * MB + mb) * 32 + lid];
    }
    for (int i = 0; i < NA; ++i) {
      if (!act[i]) continue;
      const int n = n0 + 8 * i + fm;
      for (int mb = 0; mb < MB; ++mb) {
        const int m = 8 * mb + fn;
        if (m < M0) y[size_t(m) * N + n] = T(acc[i][mb].x);
        if (m + 1 < M0) y[size_t(m + 1) * N + n] = T(acc[i][mb].y);
      }
    }
"""

_KERNELS = {}                 # (K, N, MB, NA, KS) -> compiled metal_kernel
_CALLS = {}                   # (x.shape, x.dtype, w.shape, cfg) -> (kernel, kwargs, m0 array)
_LOCK = threading.Lock()      # cache misses only (warm-up at load vs the scheduler thread)


def _config(K: int, N: int, MB: int) -> tuple[int, int]:
    """(NA, KS): A fragments (8 output features each) per simdgroup and
    simdgroups splitting K, per row bucket. Measured on M1 Max (results/mrqmm.md):
    8 rows -> 2 x 8, 16 rows -> 4 x 4 (more MMAs per x fragment); a narrow
    layer keeps NA 2 so that it still spreads over enough simdgroups."""
    na, ks = (2, 8) if MB == 1 else (4, 4)
    if N < 4096 or N % (8 * na):
        na = 2 if N % 16 == 0 else 1
    while ks > 1 and (K // GROUP) % ks:
        ks //= 2          # not needed for correctness (groups are strided), keeps KS | G
    return na, ks


def _get_kernel(K, N, MB, na, ks):
    """One metal_kernel per shape/config with the constants baked into the
    source: template arguments cost ~6 us of host time per call (mlx 0.31),
    a plain call ~1.5 us."""
    key = (K, N, MB, na, ks)
    k = _KERNELS.get(key)
    if k is None:
        consts = (f"using T = half;\n    constexpr int K = {K}, N = {N}, MB = {MB}, NA = {na}, "
                  f"KS = {ks};\n")
        k = _KERNELS[key] = mx.fast.metal_kernel(
            name=f"mrqmm8_g64_k{K}_n{N}_mb{MB}_a{na}_s{ks}",
            input_names=["x", "w", "scales", "biases", "m0"], output_names=["y"],
            source=consts + _SRC, header=_HEADER)
    return k


def _call(K, N, rows, cfg=None):
    MB = 1 if rows <= 8 else 2
    na, ks = cfg or _config(K, N, MB)
    tiles = -(-N // (8 * na))
    kw = dict(grid=(32 * ks * tiles, 1, 1), threadgroup=(32 * ks, 1, 1),
              output_shapes=[(rows, N)], output_dtypes=[mx.float16])
    return _get_kernel(K, N, MB, na, ks), kw


def supported(w: mx.array, scales: mx.array, biases, bits: int, group_size: int, mode="affine") -> bool:
    """Weights the kernel can take (any row count)."""
    if bits != BITS or group_size != GROUP or mode != "affine" or biases is None:
        return False
    if w.dtype != mx.uint32 or w.ndim != 2 or scales.dtype != mx.float16 or biases.dtype != mx.float16:
        return False
    N, K = w.shape[0], w.shape[1] * 4
    return N % 8 == 0 and K % GROUP == 0 and scales.shape == (N, K // GROUP)


def mrqmm(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, cfg=None) -> mx.array:
    """x[..., K] (fp16, 1..16 rows) @ dequant(w)^T for 8-bit g64 affine weights
    -> [..., N]. Checks only the x side (rows, dtype, K) once per x / w shape: call
    `supported` on the weights first. x is made row-contiguous by metal_kernel
    (a strided x is copied); a contiguous view starts on a row, so its 16-byte
    loads stay aligned (K % 64 == 0)."""
    N = w.shape[0]
    key = (x.shape, x.dtype, w.shape, cfg)      # w.shape: a hit skips the K check below
    c = _CALLS.get(key)
    if c is None:
        K = x.shape[-1]
        rows = x.size // K
        if not (1 <= rows <= MAX_ROWS and x.dtype == mx.float16 and K == w.shape[1] * 4
                and K % GROUP == 0):
            raise ValueError(f"mrqmm: x {x.shape} {x.dtype} vs w {w.shape}")
        with _LOCK:
            kernel, kw = _call(K, N, rows, cfg)
            kw["output_shapes"] = [(*x.shape[:-1], N)]
            c = _CALLS[key] = (kernel, kw, mx.array([rows], dtype=mx.int32))
    kernel, kw, m0 = c
    return kernel(inputs=[x, w, scales, biases, m0], **kw)[0]


def _use(x: mx.array) -> bool:
    """Decode-shaped fp16 input with MIN_ROWS..MAX_ROWS rows."""
    if x.dtype != mx.float16 or x.ndim < 2 or x.shape[-2] != 1:
        return False
    rows = x.size // x.shape[-1]
    return MIN_ROWS <= rows <= MAX_ROWS


class MRQuantizedLinear(nn.QuantizedLinear):
    """nn.QuantizedLinear whose decode-shaped 3..16-row calls take mrqmm.
    Installed by swapping __class__ (parameters and state are untouched)."""

    def __call__(self, x):
        if not _use(x):
            return super().__call__(x)
        y = mrqmm(x, self["weight"], self["scales"], self["biases"])
        if "bias" in self:
            y = y + self["bias"]
        return y


def _module_ok(m) -> bool:
    """Weights the kernel takes (8-bit affine g64, fp16 scales and biases)."""
    return (type(m) in (nn.QuantizedLinear, MRQuantizedLinear)
            and supported(m["weight"], m["scales"], m.get("biases"), m.bits, m.group_size,
                          getattr(m, "mode", "affine")))


def enabled() -> bool:
    return os.environ.get("QWEN_MRQMM", "1") != "0" and mx.metal.is_available()


def install(model, warmup: bool = True) -> int:
    """Swap every supported nn.QuantizedLinear of `model` (8-bit, g64, fp16
    scales/biases, N >= MIN_N) to MRQuantizedLinear; every other module is
    left alone. Logs the coverage once. Returns the number of modules covered
    (0 = no-op). The GDN decode in-projections fused into one matmul are the
    worker's (`_install_gdn_fuse`, after this; it calls `warm` for its shapes)."""
    if not enabled():
        log.info("mrqmm: off (QWEN_MRQMM=0 or no Metal)")
        return 0
    covered, skipped = [], {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.QuantizedLinear):
            continue
        if _module_ok(m) and m.weight.shape[0] >= MIN_N:
            m.__class__ = MRQuantizedLinear
            covered.append((name, m))
        elif type(m) is nn.QuantizedLinear:
            why = "" if (m.bits, m.group_size) != (BITS, GROUP) else (
                " N % 8" if m.weight.shape[0] % 8 else " not fp16" if m.scales.dtype != mx.float16
                else f" N < {MIN_N}" if m.weight.shape[0] < MIN_N else "")
            k = f"{m.bits}-bit g{m.group_size}{why}"
            skipped[k] = skipped.get(k, 0) + 1
    if not covered:
        log.info("mrqmm: no supported module, all on quantized_matmul (%s)",
                 ", ".join(f"{k} x{v}" for k, v in skipped.items()) or "nothing quantized")
        return 0
    kinds = {}
    for name, _ in covered:
        k = ".".join(p for p in name.split(".") if not p.isdigit())
        for pre in ("language_model.", "model.layers.", "layers."):
            k = k[len(pre):] if k.startswith(pre) else k
        kinds[k] = kinds.get(k, 0) + 1
    log.info("mrqmm: %d modules on the multi-row kernel for %d..%d decode rows (%s); "
             "left on quantized_matmul: %s", len(covered), MIN_ROWS, MAX_ROWS,
             ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())),
             ", ".join(f"{k} x{v}" for k, v in skipped.items()) or "none")
    if warmup:
        shapes = {}
        for _, m in covered:
            shapes.setdefault((m.weight.shape[1] * 4, m.weight.shape[0]), (m.weight, m.scales, m.biases))
        for w in shapes.values():
            warm(*w)
    return len(covered)


def warm(w: mx.array, scales: mx.array, biases: mx.array):
    """Compile the kernels of this weight shape for both row buckets (3..8 and
    9..16 rows) now, not on the first batched decode step (~60 ms of Metal
    compile each, on the scheduler thread). Zero inputs: only the shapes
    matter."""
    K = w.shape[1] * 4
    with mx.stream(mx.gpu):
        mx.eval([mrqmm(mx.zeros((rows, 1, K), mx.float16), w, scales, biases)
                 for rows in (MIN_ROWS, MAX_ROWS)])


def uninstall(model) -> int:
    """Undo install on `model` (its modules go back to nn.QuantizedLinear)."""
    n = 0
    for _, m in model.named_modules():
        if type(m) is MRQuantizedLinear:
            m.__class__ = nn.QuantizedLinear
            n += 1
    return n

"""Prefill quantized matmul on Apple GPUs without NAX (M1/M2): MLX's own
qmm_t with 64-row tiles.

On these GPUs mx.quantized_matmul runs a transposed qmm with 32x32 output
tiles (BM = BN = BK = 32, 2x2 simdgroups of 16x16). For prefill-sized row
counts it reaches ~8.1 TFLOPS on M1 Max where MLX's fp16 GEMM reaches ~9.5.
The same kernel code (mlx's QuantizedBlockLoader + steel BlockMMA, taken
from the headers the mlx wheel ships) instantiated with BM = 64 (2x2
simdgroups of 32x16) runs ~9.0 TFLOPS on every 9B prefill shape: x1.10-1.14
per matmul at 8 bits (measured x1.03-1.28 over 3..8-bit shapes).

The output is bit-identical to mx.quantized_matmul's qmm_t: every output
element accumulates the same BK = 32 steps of the same 8x8 MMAs in the same
order, only the tile of outputs a threadgroup owns changes. A call is taken
only where MLX itself would run that qmm_t (not qmv, not split-K, see
_mlx_uses_qmm), and install() checks bit-equality on the real weights once
per (bits, group size), on a module and row count where MLX surely runs
qmm_t; a key it cannot check that way, or that differs (e.g. an mlx bump
that changes qmm_t), stays off.

Used for fp16 x / scales / biases, affine quantization with bits in
{3,4,5,6,8} and group size 64 (the tested ones), K % 32 == 0, N % 32 == 0,
x of shape (1, ..., 1, M, K) (row-contiguous as MLX sees it). Everything
else goes to mx.quantized_matmul unchanged. QWEN_FP16=0 (bf16 scales)
leaves it off.

    import pfqmm; pfqmm.install(language_model)        # QWEN_PFQMM=0 disables
"""
from __future__ import annotations

import logging
import os
import re
import threading

import mlx.core as mx
import mlx.nn as nn

log = logging.getLogger("worker.pfqmm")

BM, BK, BN = 64, 32, 32
BITS = (3, 4, 5, 6, 8)          # 2-bit / other group sizes: untested, left to MLX
GROUPS = (64,)
TEST_MIN_ROWS, TEST_MAX_ROWS = 256, 4096   # the load self-test's row range

_SRC = r"""
  constexpr int BK_padded = BK + 16 / sizeof(T);
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  pf_qmm_t<T, GS, BITS, BM, BK, BN>(w, scales, biases, x, y, Xs, Ws, dims[0], dims[1], dims[2],
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

# mlx 0.32.3 kernels/quantized.h qmm_t_impl (aligned N, K % BK == 0, WM = WN = 2),
# with the dimensions as plain ints.
_IMPL = r"""
template <typename T, int group_size, int bits, int BM, int BK, int BN>
METAL_FUNC void pf_qmm_t(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
    int M, int K, int N, uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}
"""

_HEADER = None
_KERNELS = {}                 # (bits, group_size) -> metal_kernel
_DIMS = {}                    # (M, K, N) -> int32[3]; cleared past _DIMS_MAX entries
_DIMS_MAX = 1024
_LOCK = threading.Lock()
_ORIG_CALL = None             # nn.QuantizedLinear.__call__
_ATTR = "_pfqmm"              # set on the modules install() covers


def _header() -> str:
    """The wheel's steel GEMM + quantized headers, includes inlined (the
    runtime Metal compiler has no include path); metal_kernel prepends
    kernels/utils.h itself, so it and its includes are skipped."""
    global _HEADER
    if _HEADER is None:
        root = os.path.join(os.path.dirname(mx.__file__), "include")
        seen = set()

        def inline(rel):
            if rel in seen:
                return ""
            seen.add(rel)
            out = []
            with open(os.path.join(root, rel)) as f:
                for line in f.read().splitlines():
                    m = re.match(r'\s*#include\s+"(.+)"', line)
                    if m:
                        out.append(inline(m.group(1)))
                    elif line.strip() != "#pragma once":
                        out.append(line)
            return "\n".join(out)

        k = "mlx/backend/metal/kernels/"
        inline(k + "utils.h")
        _HEADER = "\n".join(inline(k + f) for f in (
            "steel/gemm/gemm.h", "quantized_utils.h", "quantized.h")) + _IMPL
    return _HEADER


def _kernel(bits: int, gs: int):
    key = (bits, gs)
    k = _KERNELS.get(key)
    if k is None:
        with _LOCK:
            k = _KERNELS.get(key)
            if k is None:
                consts = (f"using T = half;\n  constexpr int BM = {BM}, BK = {BK}, BN = {BN}, "
                          f"BITS = {bits}, GS = {gs};\n")
                k = _KERNELS[key] = mx.fast.metal_kernel(
                    name=f"pfqmm_b{bits}_g{gs}_{BM}x{BK}x{BN}",
                    input_names=["w", "scales", "biases", "x", "dims"], output_names=["y"],
                    source=consts + _SRC, header=_header())
    return k


def pfqmm(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, bits: int,
          group_size: int) -> mx.array:
    """x[..., K] @ dequant(w)^T (fp16) -> [..., N]; the caller checks `_use`."""
    K = x.shape[-1]
    N = w.shape[0]
    M = x.size // K
    key = (M, K, N)
    dims = _DIMS.get(key)
    if dims is None:
        if len(_DIMS) >= _DIMS_MAX:     # every last-slab length x shape: keep it bounded
            _DIMS.clear()
        dims = _DIMS[key] = mx.array([M, K, N], dtype=mx.int32)
    y = _kernel(bits, group_size)(
        inputs=[w, scales, biases, x, dims],
        grid=(N // BN * 64, -(-M // BM) * 2, 1), threadgroup=(64, 2, 1),
        output_shapes=[(*x.shape[:-1], N)], output_dtypes=[mx.float16])[0]
    return y


def _mlx_uses_qmm(M: int, K: int, N: int, gs: int) -> bool:
    """mlx QuantizedMatmul::eval_gpu (transposed, non-batched) takes qmm_t for
    these rows: not a qmv (M below the vector limit, which is <= 33 on every
    device; `_use` needs M >= 128 anyway) and not split-K (qmm_splitk: split = 512 // 32x32 tiles, capped
    by K's groups and K % (split * group) == 0, qmm when it ends <= 1)."""
    if M < 32:
        return False
    tiles = -(-N // 32) * -(-M // 32)
    split = min(max(1, 512 // tiles), K // gs)
    while split > 1 and K % (split * gs):
        split -= 1
    return split <= 1


def supported(m) -> bool:
    if not isinstance(m, nn.QuantizedLinear) or getattr(m, "mode", "affine") != "affine":
        return False
    w, s, b = m.get("weight"), m.get("scales"), m.get("biases")
    if b is None or m.bits not in BITS or m.group_size not in GROUPS or w.ndim != 2:
        return False
    if w.dtype != mx.uint32 or s.dtype != mx.float16 or b.dtype != mx.float16:
        return False
    N, K = w.shape[0], w.shape[1] * 32 // m.bits
    return N % BN == 0 and K % BK == 0 and K % m.group_size == 0 and s.shape == (N, K // m.group_size)


def _use(m, x) -> bool:
    """fp16 rows that MLX would run through qmm_t, and not so few past a
    multiple of 64 that the half-empty last tile eats the gain (64-row tiles
    are x1.10-1.14 per row; M = 96 is x0.8)."""
    if x.dtype != mx.float16 or x.ndim < 2:
        return False
    K = x.shape[-1]
    M = x.size // K
    if M != x.shape[-2]:
        # leading dims > 1: MLX may see a non-row-contiguous x and run a
        # batched qmm / qmv by shape[-2] instead of the qmm_t mirrored here
        return False
    return (M >= 128 and -(-M // 64) * 64 <= 1.06 * (-(-M // 32) * 32)
            and _mlx_uses_qmm(M, K, m["weight"].shape[0], m.group_size))


ACTIVE = True                 # runtime switch (in-process A/B); install() sets the modules up


def _call(self, x):
    if ACTIVE and self.__dict__.get(_ATTR) and _use(self, x):
        y = pfqmm(x, self["weight"], self["scales"], self["biases"], self.bits, self.group_size)
        if "bias" in self:
            y = y + self["bias"]
        return y
    return _ORIG_CALL(self, x)


def enabled() -> bool:
    return os.environ.get("QWEN_PFQMM", "1") != "0" and mx.metal.is_available()


def _test_rows(K: int, N: int, gs: int) -> int | None:
    """Rows at which the load self-test can compare against qmm_t: the
    smallest multiple of 64 in [TEST_MIN_ROWS, TEST_MAX_ROWS] where MLX runs
    qmm_t (no split-K) both there and 8 rows further (a ragged tile); None
    when this shape never does in that range."""
    for M in range(TEST_MIN_ROWS, TEST_MAX_ROWS + 1, 64):
        if _mlx_uses_qmm(M, K, N, gs) and _mlx_uses_qmm(M + 8, K, N, gs):
            return M
    return None


def _selftest(m, M: int) -> str | None:
    """None when the kernel matches mx.quantized_matmul bit for bit on this
    module at M rows (a full 64-row tile) and M + 8 (a ragged one), else why
    not. M must be one where MLX runs qmm_t (`_test_rows`): never vacuous."""
    N, K = m["weight"].shape[0], m["weight"].shape[1] * 32 // m.bits
    for rows in (M, M + 8):
        if not _mlx_uses_qmm(rows, K, N, m.group_size):
            raise ValueError(f"self-test at {rows} rows: MLX would not run qmm_t (K={K} N={N})")
        x = (mx.random.normal((1, rows, K), key=mx.random.key(rows)) * 0.5).astype(mx.float16)
        ref = mx.quantized_matmul(x, m["weight"], m["scales"], m["biases"], transpose=True,
                                  group_size=m.group_size, bits=m.bits)
        y = pfqmm(x, m["weight"], m["scales"], m["biases"], m.bits, m.group_size)
        if not mx.array_equal(y, ref).item():
            return f"K={K} N={N} M={rows}: differs from quantized_matmul " \
                   f"(max |d| {mx.abs(y - ref).max().item():.3g})"
    return None


def install(model) -> int:
    """Route the prefill-sized calls of every supported nn.QuantizedLinear of
    `model` (subclasses included: mrqmm's decode class calls through) to
    pfqmm. Per (bits, group size): the first module (in named_modules order)
    whose shape lets MLX run qmm_t at TEST_MIN_ROWS..TEST_MAX_ROWS rows is
    checked bit for bit; a key with no such module ("untested") or a mismatch
    / error ("failed") stays on quantized_matmul. Logs each key. -> modules
    covered (0 = off)."""
    global _ORIG_CALL
    if not enabled():
        log.info("pfqmm: off (QWEN_PFQMM=0 or no Metal)")
        return 0
    by_key = {}
    for n, m in model.named_modules():
        if supported(m):
            by_key.setdefault((m.bits, m.group_size), []).append((n, m))
    if not by_key:
        log.info("pfqmm: off, no fp16-scale affine %s-bit g64 QuantizedLinear "
                 "(QWEN_FP16=0 leaves bf16 scales)", "/".join(map(str, BITS)))
        return 0
    status = {}                                     # key -> ("tested" | "untested" | "failed", detail)
    for (bits, gs), ms in sorted(by_key.items()):
        pick = None
        for n, m in ms:
            K, N = m["weight"].shape[1] * 32 // bits, m["weight"].shape[0]
            M = _test_rows(K, N, gs)
            if M is not None:
                pick = (n, M, K, N, m)
                break
        if pick is None:
            status[(bits, gs)] = ("untested", f"no module where MLX runs qmm_t at "
                                              f"<= {TEST_MAX_ROWS} rows ({len(ms)} modules)")
            continue
        n, M, K, N, m = pick
        try:
            why = _selftest(m, M)
        except Exception as e:  # noqa: BLE001 — no headers / compile error: stay on qmm
            why = f"{type(e).__name__}: {str(e)[:300]}"
        status[(bits, gs)] = ("failed", why) if why else (
            "tested", f"{n} K={K} N={N} M={M}/{M + 8}")
    covered = []
    for key, ms in by_key.items():
        if status[key][0] == "tested":
            covered += [m for _, m in ms]
    for m in covered:
        object.__setattr__(m, _ATTR, True)          # a plain attribute, not a parameter
    if covered and _ORIG_CALL is None:
        _ORIG_CALL, nn.QuantizedLinear.__call__ = nn.QuantizedLinear.__call__, _call
    mx.clear_cache()
    for (bits, gs), (st, detail) in sorted(status.items()):
        (log.info if st == "tested" else log.warning)(
            "pfqmm %d-bit g%d: %s, %d modules %s (%s)", bits, gs, st, len(by_key[(bits, gs)]),
            "on" if st == "tested" else "left on quantized_matmul", detail)
    log.info("pfqmm: %d modules on the 64-row prefill qmm (%s)", len(covered),
             ", ".join(f"{b}-bit g{g} {st}" for (b, g), (st, _) in sorted(status.items())))
    return len(covered)


def uninstall(model) -> int:
    n = 0
    for _, m in model.named_modules():
        if m.__dict__.get(_ATTR):
            object.__setattr__(m, _ATTR, False)
            n += 1
    return n

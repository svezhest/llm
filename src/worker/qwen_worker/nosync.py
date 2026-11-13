"""QWEN_DEBUG_NO_SYNC: host waits raise while a decode step is built."""

import contextlib
import functools
import sys
import threading

import mlx.core as mx
import numpy as np

from .config import DEBUG_NO_SYNC


_NO_SYNC = threading.local()
_NO_SYNC_ON = False                     # the wrappers are installed (once, at first use)
# mlx-vlm 0.7.6's one-time evals inside a decode forward (each caches its
# result on first use: a kernel launch probe, the conv weight's fp32 copy):
# a wait on the first step after load only, not a step-ahead regression
_NO_SYNC_ALLOW = frozenset({"_qwen3_5_try_launch", "_causal_conv1d_decode"})


class SyncInStep(RuntimeError):
    """QWEN_DEBUG_NO_SYNC: a host wait while a decode step was being built."""


def _install_no_sync():
    """Wrap the waits once: mx.array methods are patchable at class level
    (nanobind), numpy's buffer-protocol reads are not, so np.asarray /
    np.array are wrapped as module functions (what `np.` callers see)."""
    global _NO_SYNC_ON
    if _NO_SYNC_ON:
        return

    def guard(name, fn, arg=False):
        @functools.wraps(fn)
        def f(*a, **k):
            if (getattr(_NO_SYNC, "on", False) and (not arg or (a and isinstance(a[0], mx.array)))
                    and sys._getframe(1).f_code.co_name not in _NO_SYNC_ALLOW):
                raise SyncInStep(f"QWEN_DEBUG_NO_SYNC: {name} while a decode step is built "
                                 "(_Scheduler._step, before its async_eval): it waits for the GPU "
                                 "and breaks the step-ahead")
            return fn(*a, **k)
        return f
    for name in ("eval", "synchronize"):
        setattr(mx, name, guard(f"mx.{name}()", getattr(mx, name)))
    for name in ("item", "tolist", "__int__", "__float__", "__bool__", "__index__"):
        setattr(mx.array, name, guard(f"mx.array.{name}()", getattr(mx.array, name)))
    np.asarray = guard("np.asarray(mx.array)", np.asarray, arg=True)
    np.array = guard("np.array(mx.array)", np.array, arg=True)
    _NO_SYNC_ON = True


class _NoSync:
    def __enter__(self):
        _install_no_sync()
        _NO_SYNC.on = True

    def __exit__(self, *exc):
        _NO_SYNC.on = False


_NULL_CTX = contextlib.nullcontext()


def _no_sync():
    """The QWEN_DEBUG_NO_SYNC context for building a step (a no-op when off)."""
    return _NoSync() if DEBUG_NO_SYNC else _NULL_CTX

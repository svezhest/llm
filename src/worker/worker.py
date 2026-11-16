#!/usr/bin/env python3
"""Thin gRPC MLX worker for the Qwen3.x-family VL models (mlx-vlm).

Loads the model once, then serves the `Worker` service from worker.proto:
  - Generate: server-streaming Start/Token/Done/Error
  - Cancel:   cooperative stop of an in-flight generation
  - Health:   ready / model / load time
  - Shutdown: graceful stop

Generations are batched: one scheduler thread owns MLX and decodes every
running request in one forward per step (continuous batching, QWEN_MAX_BATCH);
the gRPC handler threads only render, tokenize and turn token ids into events.

Run:
  uv run --group proto bash gen_proto.sh   # once, to codegen
  uv run python worker.py                   # serve on 127.0.0.1:50051
"""
# The code lives in the qwen_worker package (module map: docs/layout.md).
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


if __name__ == "__main__":
    from qwen_worker.server import main
    main()
else:
    _sys.modules[__name__].__class__ = _Facade

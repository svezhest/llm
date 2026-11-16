"""The MLX worker as a package: worker.py is its entry point (`uv run python
worker.py`) and a facade over it (`import worker`: every name of these modules
as one namespace, as the single-file worker was). MODULES in import order."""

from . import (config, chat, sampler, pcache, sizing, stream, grammar, nosync, rows, kernels,
               lookup, mtp, logprobs, request, sched_admit, sched_prefill, sched_spec, sched_store,
               sched_cascade, sched_decode, scheduler, engine, server)

MODULES = (config, chat, sampler, pcache, sizing, stream, grammar, nosync, rows, kernels, lookup, mtp, logprobs,
           request, sched_admit, sched_prefill, sched_spec, sched_store, sched_cascade,
           sched_decode, scheduler, engine, server)

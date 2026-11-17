# Code layout: the worker facade and its modules

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- `src/worker/worker.py` is the entry point and a facade over `qwen_worker/`; tests patch
  names through it, and `tests/test_facade.py` checks that each patch reaches its readers.
- The repository layout and run order are in [CLAUDE.md](../CLAUDE.md).

## `src/worker/`

`worker.py` = the entry point (`uv run python worker.py`) + a facade over the package
`qwen_worker/`: `import worker` still gives every name (`W._Scheduler`, `W.PREFILL_STEP`, …),
and `setattr(worker, NAME, v)` / `monkeypatch.setattr(worker, …)` rebinds NAME in every
module that holds it (`tests/test_facade.py` checks that each patch reaches its readers).
So "worker.py" / `_Scheduler._fits` in the notes below = a name in `qwen_worker/`.
Modules: `config` (QWEN_* knobs, constants; `SDPA_FUSED` / `DRAFT_BLOCK` / `BATCH_INVARIANT`
are changed at load, read as `cfg.NAME`), `chat` (tool-call parser, images, messages,
`_PlainContent`), `sampler`, `pcache` (`_Seg` / `_Entry` / `_PrefixCache`, cache-array
helpers), `sizing` (`_step` / `_bounds` slab policy, `_FIXED_SLAB`, budget math), `stream`
(detokenizer, `_StopFilter`, `_Control`, `_GenState`, `_done`), `grammar` (llguidance;
`QWEN_JSON_MAX_WS`), `nosync`, `rprof` (`QWEN_SPEC_PROF`: a request's round / step timeline), `rows` (`RowsKV`, `_rows_attention`, stacked GDN), `kernels`
(mlx-vlm patches: fused SDPA, DFlash verify, GDN fuse, batch-invariant checks), `lookup` (prompt lookup drafts: `_Lookup`, `_Gate`, `_DraftCtx`), `logprobs`,
`request` (`_Req` / `_Row` / `_Own`), `scheduler` (`_Scheduler` core) + its mixins
`sched_admit` / `sched_prefill` / `sched_spec` / `sched_store` / `sched_cascade` /
`sched_decode`, `engine` (`Engine`), `server` (servicer, `serve`, `main`). A new module-level
name a test patches must be read bare in its home or via `from .home import NAME`; a name
the code rebinds at run time must live in one module and be read qualified elsewhere.

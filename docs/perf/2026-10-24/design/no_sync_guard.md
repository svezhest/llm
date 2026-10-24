# Debug guard: no host wait while a decode step is built (`QWEN_DEBUG_NO_SYNC`)

2026-10-24. This is DeepInfer's `forbid_cuda_sync()` (`deepinfer/utils.py:119`, used around the step
build in `worker/model_runner.py:324,359`). Our step-ahead has the scheduler queue step k+1 before it
reads step k's tokens. It is broken by any host wait inside `_Scheduler._step`. The last such regression
(`concatenate(...).tolist()`, −5…12 % step time, `docs/perf/2026-10-12/step_profile.md`) was found by
hand.

## What

With `QWEN_DEBUG_NO_SYNC=1`, `_iterate` builds the step inside `_no_sync()` (both call sites). While it
is active, on the scheduler thread only, the following raise `SyncInStep` with a message naming the call
and `_Scheduler._step`:

- `mx.eval` and `mx.synchronize`;
- `mx.array.item` / `tolist` / `__int__` / `__float__` / `__bool__` / `__index__`;
- `np.asarray` / `np.array` of an `mx.array`.

How the waits are patched:

- **mx.array methods** are patched at class level. Nanobind types accept it; checked: `int()`, `float()`,
  `bool()`, `operator.index()` and `.item()` go through the patched attributes.
- **numpy** reads mx arrays through the C buffer protocol, which cannot be patched (`__buffer__` set on
  the class is ignored by numpy). So `np.asarray` / `np.array` are wrapped as module functions: what
  every `np.` caller sees.
- **Installation.** The wrappers are installed once, at first use, and only check a thread-local flag.
  Handler threads, which read numpy from MLX constants (`to_mlx`), are never affected. With the knob off
  nothing is installed and `_no_sync()` returns a shared `nullcontext`.
- **Allowlist.** mlx-vlm 0.7.6 has two one-time evals inside a decode forward:
  `_qwen3_5_try_launch` (a kernel launch probe, first use per pipeline) and `_causal_conv1d_decode` (the
  conv weight's fp32 copy, cached). They are allowlisted by their caller's code name (`_NO_SYNC_ALLOW`):
  each is a wait on the first step after load, not a step-ahead regression.
- **`mx.async_eval` is allowed.** The step ends with it.
- **Outside the step.** The token read (`r.y.item()`), the preemption read, the prompt-cache stores and
  the prefill are outside the step and unaffected.

A violation propagates like any exception in a step:

- inside a row's sampling `try`, that row fails with `SyncInStep`;
- elsewhere the step fails (`_fail_rows`, the strike counter).

The tests see it either way.

## Findings

The whole CPU suite runs with `QWEN_DEBUG_NO_SYNC=1` and passes, with one exception:
`test_cascade_sched.py::test_wave_forms_joins_and_dissolves`. Its spy reads logits with `np.array` inside
the step on purpose. So `_step` is clean today on the CPU paths: B = 1, stacked B ≥ 2, GDN fuse,
cascade, grouped sampling, logprobs, penalties, grammar sync mode, `_regrow` with evictions. The Metal
paths (mrqmm, the Metal SDPA / RoPE kernels) are checked in phase 0 of `scripts/perf/preempt/preempt_check.py`.

## Tests

`src/worker/tests/test_no_sync.py`:

- B = 1 / 4 / 8 on `dense` and `dense_fused`: the `mixed` set (seeded sampling, penalties, max 0 / 1);
- logprobs with and without penalties, a grammar row (sync mode), bad words + min_tokens, and the prompt
  cache, all answering exactly as with the guard off;
- an injected `.tolist()` / `.item()` / `mx.eval` / `np.asarray` in the step raises `SyncInStep`. Outside
  the step the wrappers pass through.

`test_lane.py` and `test_preempt.py` also run with the guard on.

It could be turned on for the whole CI (`conftest`: `worker.DEBUG_NO_SYNC = True`) once the cascade spy
copies its logits after the step. Not done here: other branches add code to `_step` in parallel.

## GPU results (2026-10-24, 9B 8-bit, M1 Max; raw: `../raw/preempt/preempt_check.*`)

Phase 0 of `scripts/perf/preempt/preempt_check.py` ran with `worker.DEBUG_NO_SYNC = True`:

- 8 rows of 64 tokens (greedy, seeded top-p/top-k, penalties, logprobs top 5). The batch reached B = 8,
  and the log shows mean rows 4–6, which covers mrqmm, the GDN fuse and grouped sampling.
- Then 1 row alone with logprobs.

Result: **PASS**. No `SyncInStep` and no errors. The Metal paths of `_step` hold no host wait, and the
allowlisted mlx-vlm one-time evals did not fire outside the allowlist. The knob stays a debug switch
(default off).

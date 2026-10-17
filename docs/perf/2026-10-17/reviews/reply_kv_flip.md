# Review: reply-KV default flip + F1/F2/F3 (2026-10-17)

Patch: worktree `.claude/worktrees/agent-aee3fec962d4a9caa`, `git diff` vs 744f9e3. CPU only,
nothing edited. Line numbers below are the worktree's `src/worker/worker.py` / tests.

**Verdict: nothing above LOW.** CPU suite (`--ignore test_mrqmm/test_pfqmm/test_rattn`), one
run: **278 passed, 3 skipped** (319 s).

## 1. Handler: `_GenState.for_prompt` vs the old inline logic (`:4515`, `:4766`)

Old: `inside = next((t == self._think[0] for t in reversed(ids) if t in self._think), False)`;
`state = _GenState(inside)`; `state.counting = inside`.
New: `for_prompt` = `cls(bool(think) and <the same next(...)>)`; `__init__` sets
`counting = thinking`; `inside = state.in_thinking`.

- Same expression on the same `ids`; `Engine._think` is always a 2-element list
  (`:4216`), so the `bool(think)` guard is always true in the handler (it only matters for
  the scheduler's `getattr(eng, "_think", None)`). `counting` moved into `__init__`: there
  are no other `_GenState(...)` constructions (grep), so nothing else changes.
- `inside` still gates the thinking budget and `_Guide(active=not inside)` exactly as before.
- Checked by a fuzz (20k random id sequences over {<think>, </think>, other}, plus the named
  shapes below, with `think` = [TH, THC] / [None, None] / [TH, None]): `(in_thinking,
  counting)` identical to the old code in every case, `in_thinking` always a bool, and the
  same `reasoning_tokens` / `gen_tokens` after a token stream.
  - thinking on (gen prompt ends `<think>\n`): inside, as before;
  - thinking off (`<think>\n\n</think>\n\n`): last tag `</think>`, not inside;
  - `enable_thinking` absent: `_render` defaults it to True → pre-opened → inside;
  - history whose last think tag is `</think>`, then a thinking-on gen prompt: the gen
    prompt's `<think>` is last → inside; with a thinking-off gen prompt → not inside;
  - continue_final_message with an unclosed `<think>`: inside (unchanged);
  - forced `</think>` (budget) and tool-call-ending reasoning: these happen in the output,
    handled by `feed()` / `token()`, which the patch does not touch.

The API reasoning/content split is unchanged for every prompt shape.

## 2. `state.in_thinking` at the end of the row (`:3458-3465`)

The skip tests `state.in_thinking` on the **fresh** `for_prompt` state, before any fed token
is replayed; `token()` never changes `in_thinking` (only `feed()` does, and it is not called
here). So the decision is "generation started inside an open `<think>`", i.e. thinking on →
**every** reply is skipped, whether the reasoning closed (`</think>` then content), ended on
`<tool_call>`, or never closed. That matches the design intent ("skip rows that generated a
reasoning section"). The replay loop only feeds the log's reasoning count.

Coverage of the closed-reasoning case: `test_reasoning_replies_are_not_stored[think-ids-known]`
(`tests/test_reply_kv.py:158`) — the budget forces `\n</think>` at `toks[4:6]`, then 3 content
tokens; asserted skipped, prompt entry and its end state kept. A model-emitted (unforced)
`</think>` takes the identical code path (no test, none needed). Mutants: the old
"any think id in fed" rule fails `test_think_id_in_the_content_with_thinking_off_is_stored`;
dropping the `in_thinking` check fails `test_thinking_on_reply_without_think_ids_is_not_stored`.

- LOW (info): with thinking off, a `</think>` id the model emits inside its content is
  *dropped* from the API content by `_GenState.feed` (`.replace(THINK_CLOSE, "")`), while the
  reply KV holds it. The client's next turn re-renders without it, so the hit stops at the
  last slab end before it: harmless fallback, only the copy and the prompt-end state (a
  regenerate's full hit) are lost. `<think>` (plain text in the content) round-trips. Rare;
  optional fix: with thinking off also skip when `think[1] in fed[:n]`. Not worth a change
  unless it shows up in logs.

## 3. Fixtures (`small`, `prod`, `casc`)

- Before the patch the default was off, so `PROMPT_CACHE_REPLY=False` in these fixtures
  reproduces exactly the pre-patch environment for every test using them, including
  `test_partial_store.py` / `test_segments.py`, which import `small` from
  `test_prompt_cache`. Nothing that exercised the prefill-only store stopped doing so;
  nothing passes vacuously.
- Tests that turn the cache on outside these fixtures now run with the reply KV **on**:
  `test_scheduler.py:257 test_prompt_cache_while_batching` (only one that runs requests),
  `test_cascade_sched.py:238` and `test_robustness.py:243` (no generation: knobs / budget
  formulas, unaffected). LOW (info): `test_prompt_cache_while_batching` still checks what it
  claims (hits at the system prefix's slab ends, output == cold) and now runs under the
  production default; its later arrivals hit c0's prompt entry while c0 decodes. If it is
  meant as a prefill-store-only test, pin `PROMPT_CACHE_REPLY=False` there for symmetry;
  otherwise fine as is.
- New tests use their own setting explicitly (`test_reply_kv.small` on;
  `test_reply_kv_skips_grouped_rows_with_a_log_line` sets it on over `casc`).

## 4. F1 log line and the parsers

- `_reply_cache` (`:3391-3399`): the group check moved before the `keys` check (grouped
  rows hold `_GroupRef` caches without `keys`, so before the patch they returned None in
  the same condition); now logged once per candidate row (stop/length, text, not DFlash),
  never for cancelled rows (tested). `row.group` is a `_PrefixGroup` with `.P` (`:1849`).
  At most one INFO line per request: fine.
- `scripts/perf/agentic/worker_log.py:106` and `docs/perf/2026-10-16/analyze_worker_log.py:29`
  match `rest.startswith("prompt cache reply skipped: in a cascade group")` after the
  `id=(\S+) ` split, which is exactly the emitted text; the counters are printed through the
  existing `events {E}` / `reply KV: {REPLY}` lines. No other reply-line consumer to update.

## 5. Docs (F3)

The `length` correction is right: `_iterate` finishes a length row at the *next* tick
(`r.n == max_tokens`), after the step on its last read token was queued (step-ahead) or
taken (sync rows step after reading, `if not ahead:`), so the last generated token is fed in
both modes; only the token sampled from it is never read. A sync row's EOS is not fed
(`_finish` + `_compact` before the step), as the table says. README / CLAUDE.md / design doc
defaults updated consistently (`1`, `0` = bit-exact way back).

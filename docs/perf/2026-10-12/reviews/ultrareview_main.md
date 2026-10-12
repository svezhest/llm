# Cloud review (ultrareview) of main 718d1b5..ca28690 — findings to fix during the optimization work

1. [normal] gateway api.rs:108-144 — `client.check()` runs for EVERY request BEFORE `pool.try_admit()`;
   a full queue no longer sheds load for free, and Check (template render + tokenizer.encode, GIL) shares
   the worker's 4-thread gRPC pool with Generate/Cancel/Health → can starve it. Fix: admit (try_admit) first,
   then Check; and give the worker pool enough threads (batching needs more anyway).
2. [normal] worker.py ~1830-1846 (`_Control.__init__`) — bad_words / allowed_token_ids / stop_token_ids have
   no size cap (request.rs validate() only checks sign/non-empty); tokenizer.encode per bad_word ×2 + mx.array
   build happen under the global generation lock. Fix: cap list lengths in request.rs validate() (400).
3. [normal] request.rs:482-508 `template_kwargs()` — reasoning_effort's derived enable_thinking is inserted
   before the top-level `enable_thinking`, so `{"reasoning_effort":"high","enable_thinking":false}` renders
   with thinking ON, contradicting the documented precedence (chat_template_kwargs < reasoning_effort <
   top-level enable_thinking < GATEWAY_THINKING). Fix the precedence.
4. [normal] worker.py:280-289 `materialize_images` — temp files of earlier images leak when a later image's
   base64 fails (function raises before returning `tmp`). Fix: try/except that unlinks created files.
5. [nit] request.rs:627 — stop_token_ids / allowed_token_ids cast i64→u32 without an upper bound (wraps to 0).
   Fix: validate ≤ u32::MAX (or ≤ vocab size — the worker knows vocab) → 400.
6. [nit] api.rs:112 — the whole pb::Request (incl. base64 image URLs) is cloned for Check. Minor.
7. [nit] journal.rs:31-51 — synchronous mutex-guarded file write on the tokio hot path; move to
   spawn_blocking or a writer task.

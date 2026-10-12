# Gateway diff review (uncommitted: src/gateway, src/proto, deploy, README) — 2026-10-11

Read-only review; nothing built or run. `target/release/qwen-gateway` (20:26) is newer than every
changed source (≤20:25:55), and the generated `qwenworker.rs` derives `Copy` for `Usage`, so
`d.usage.unwrap_or_default()` / `if let Some(u) = d.usage` on `&Done` compile.

## Findings (ranked)

### 1. [Medium] A client's `request_id` reused across concurrent requests now cross-cancels (api.rs:100-101, worker.py:2065-2083)
- The worker id is `chatcmpl-<request_id>` when the client sends `request_id`. The worker keeps
  `_cancels[req.id]` (one entry per id): `_register` overwrites it and `_unregister` pops it by id.
- Before this diff the in-flight cap was 1, so two requests with the same id almost never overlapped
  in the worker. The only overlap was a narrow race: the permit is freed before the Cancel is spawned.
  Now up to N overlap.
- Scenario: a client retries with the same `request_id` (an idempotency-key pattern) while the first
  request still runs, then drops the first. The Cancel sets B's event, so B is cancelled. Also, when
  A finishes, `_unregister` removes B's entry, so B's later Cancel is a no-op (only the gRPC
  context callback still saves it).
- Fix: send the worker a unique internal id (uuid, or `chatcmpl-<request_id>-<uuid8>`) and keep
  `chatcmpl-<request_id>` only in the response and journal. Alternatively, refuse a duplicate
  in-flight id.

### 2. [Medium] Worker gRPC thread pool vs gateways that oversubscribe: Check, Health and Cancel starve and Check has no deadline (api.rs:120, worker.py:2610, CLAUDE.md "a mismatch is harmless")
- The worker pool has `2*max_batch + 8` threads (24 at batch 8). Every Generate holds a thread for
  its whole life, including while it waits in the scheduler.
- The gateway's capacity is per gateway. Three gateways already share :50051 (8080, 8081, 8090).
  Once all three run this binary, they can hold 3×8 = 24 Generate streams, which is every thread.
- The same happens with `GATEWAY_MAX_INFLIGHT` > max_batch. It also happens after the worker
  restarts with a smaller `QWEN_MAX_BATCH`, or with a non-qwen3_5 model that falls back to 1
  (10 threads), while the gateways keep N=8.
- Result: a new request's `Check` waits in grpcio's executor queue until some generation finishes,
  which can take minutes at long context. During that time it holds a `MAX_QUEUE` slot, and the
  client sees nothing. Cancel and Health (`/v1/models`) wait as well. This is not a deadlock, because
  generations progress, but the claim "a mismatch is harmless: the worker queues internally" is false.
- Fixes, from cheapest:
  - (a) Put a deadline on Check (`tonic::Request::set_timeout`, ~10–30 s) and map a timeout to 503.
  - (b) Warn when `GATEWAY_MAX_INFLIGHT` > max_batch. Document that N gateways should split
    max_batch between them.
  - (c) On the worker, size threads for the control RPCs separately. For example, use a larger
    `QWEN_GRPC_THREADS` default (`4*max_batch+8`), or serve Check/Cancel/Health on their own server
    or executor.
  - Correct the CLAUDE.md sentence.

### 3. [Low-Medium] Capacity is read once at connect (main.rs:38-41)
- If the gateway started against an old worker (no `max_batch` → 1), it stays serialized after the
  worker is upgraded. If it started against a batch-8 worker, it keeps sending 8 after a restart to
  max_batch 1 (see #2). tonic reconnects silently, so the only signal is the startup log.
- Fix: re-read Health periodically (e.g. 30 s). Adjust with `Semaphore::add_permits` and
  `forget_permits`, and update `qwen_inflight_capacity`. At minimum, log at warn when a later Health
  reports a different max_batch.

### 4. [Low, pre-existing, confirmed] The TTFT and request-duration panels can never show data (deploy/grafana/dashboards/qwen.json panels 5, 6)
- Both panels query `histogram_quantile(..., rate(qwen_*_seconds_bucket[5m]))`. `metrics.rs` builds
  `PrometheusBuilder::new()` with no `set_buckets` or `set_buckets_for_metric`, so every histogram
  renders as a summary (`{quantile=...}`, `_sum`, `_count`). No `_bucket` series exist.
- The new panels 11 and 12 correctly use `{quantile="0.5"|"0.95"}`, which are in the exporter's
  default quantiles (0, .5, .9, .95, .99, .999, 1). Those quantiles cover a rolling window of about
  60 s.
- Fix, either:
  - Add `.set_buckets_for_metric(Matcher::Full("qwen_ttft_seconds".into()), &[...])?` and the same
    for `qwen_request_duration_seconds`. Then the panels work and can be aggregated, and the README
    sentence "Histograms render as summaries" needs an exception.
  - Or rewrite panels 5 and 6 as `qwen_ttft_seconds{quantile="0.5"}` and the other quantiles, like
    the new panels.

### 5. [Low] Admitting before Check changes which error wins when the queue is full (api.rs:109-145)
- With a full queue, a request with an unknown `model` (formerly 404), a template or context error
  (400), or a dead worker (502) now gets 503 "queue full".
- This is intended (review finding #1) and fine under OpenAI semantics, since 503 is retryable.
  Mentioned because the 404 used to be deterministic. Gateway-side validation (`validate`, `to_pb`)
  still runs before admission, so those 400s are unaffected.

### 6. [Low] `qwen_worker_queue_seconds` records 0 for every request from an older worker (api.rs:327-333)
- An old worker leaves `queue_ms` unset, which reads as 0, so p50 and p95 sit at 0 and look healthy.
- Fix: record it only when `u.batch_mean > 0.0`. A new worker always sets `batch_mean` ≥ 1, so it
  already tells new workers from old ones.

### 7. [Low, semantics] The gateway-side wait is measured nowhere (api.rs:191-195, 360-365)
- `start` is taken after `acquire_inflight()`. So `qwen_request_duration_seconds` ("End-to-end
  request duration") and `qwen_ttft_seconds` exclude the wait for an in-flight slot. With
  capacity > 1, that wait is now the main client-visible queue.
- Fix: take an `Instant` at handler entry. Record a `qwen_gateway_queue_seconds` histogram, measure
  TTFT from entry, or both.

### 8. [Low] Signal handling (main.rs:74-97)
- SIGINT and SIGTERM now exit with code 0. Before, the process was killed by the signal (exit status
  130/143).
- During the flush (up to 5 s, and only if the journal is on), a second Ctrl-C is swallowed.
- Streams are cut exactly as before: `process::exit`, no graceful drain. Workers see the connection
  close, and the context callback frees their rows. A stream that finishes during the flush window
  may or may not get its journal line, which is acceptable.
- launchd: the gateway plist has `KeepAlive <true/>`, which restarts on any exit, so exit 0 is
  compatible. `watchdog.sh` uses `kickstart -k`, which is unaffected.
- Before `serve`, while `connect()` waits for the worker, no handler is installed, so the default
  disposition applies. That is fine.

## Nits
- journal.rs:47 — `mpsc::channel` is unbounded. If the filesystem hangs, every finished request
  stays in memory (its body `Bytes` plus the response `Value`). A `sync_channel(1024)` with
  `try_send` and a dropped-lines counter would cap it. On a local disk this is not a practical
  problem.
- request.rs `thinking_precedence` test asserts `forced_thinking() == None`. It fails if
  `GATEWAY_THINKING` is exported in the shell that runs `cargo test`, because the value is read
  once per process.
- main.rs:35 — a bad `MAX_QUEUE` silently falls back to 32. `MAX_QUEUE=0` turns every request into
  a 503. Both are pre-existing. Consider the same warn-and-fallback that `capacity()` does.
- api.rs `stream_response` — the 200 and SSE headers go out before the in-flight slot is acquired,
  and no `Sse::keep_alive` is set. Under load, a streaming client sees silence until it gets a slot.
  Pre-existing, but more likely with gateway queueing; `.keep_alive(KeepAlive::default())` costs
  nothing for OpenAI clients, which ignore comment lines.
- request.rs:313-317 — a negative or ≥2^32 id now reports `contains an invalid token id: N.` with
  the correct `param`. Before, `stop_token_ids` was used even for `allowed_token_ids`. vLLM's own
  messages differ (`allowed_token_ids contains out-of-vocab token id!`); cosmetic.
- `seed` values other than -1 that are negative are cast `as u64` and wrap. This is pre-existing and
  not in the diff.
- Admission order is FIFO at the in-flight semaphore (tokio's semaphore is fair). But Check
  latency differs per request, so the order of arrival is not strictly kept.

## Verified OK
- **Slot gauges are exact on every exit path.** `qwen_queue_depth` and `qwen_in_flight` are set to
  0 in `Pool::new`, incremented when a permit is taken, and decremented in `Slot::drop`. The paths:
  - 503: no slot is taken.
  - Check returns 400, 404, 500 or 502 after admission: `admit` drops on return.
  - Client disconnects during Check: the handler future is dropped, which drops `admit`.
  - Client disconnects while waiting for an in-flight slot: in the stream case the generator
    captures `admit`; in the JSON case the handler future owns it. Either way it is dropped, the
    pending `acquire_owned` future is dropped, and no gauge is touched.
  - Mid-stream or after the response: `ReqGuard` drops both slots.
  - Panic: there is no `panic = "abort"` profile, so unwinding drops the generator and the guard.
  - `pool.rs` `permits_and_gauges` covers these cases with a local recorder.
- **Worker.Cancel fires only after a slot was taken and the guard was armed.** `ReqGuard` is built
  after `acquire_inflight`. A disconnect while waiting for admission, Check or a slot sends no
  Cancel. `armed = false` is set before the journal line and `[DONE]`, and in the JSON path before
  the response is built. The pre-existing case is unchanged: a disconnect while
  `client.generate()` is still connecting sends a Cancel for an unregistered id, which the worker
  logs at debug and ignores.
- **`capacity()`:** a positive integer is used, with whitespace trimmed. Empty, 0, negative or
  junk values warn and fall back to `max(max_batch, 1)`. `max_batch` 0 becomes 1. There is also a
  warn when `MAX_QUEUE < cap`.
- **Proto:** the new fields are additive, so compatibility holds both ways. The old binaries on
  :8080 and :8081 work with the new worker. A new gateway with an old worker gets max_batch 0 → 1,
  and queue_ms and batch_mean read as 0. `Health.max_batch` is set after `Engine.load()` (including
  the fallback to 1 for non-qwen3_5 models), before the server starts.
- **Thinking precedence:** the code matches README, CLAUDE.md and the tests. The order is
  `chat_template_kwargs` < top-level `enable_thinking` < `GATEWAY_THINKING`, and `reasoning_effort`
  fills `enable_thinking` only when `chat_template_kwargs` has none. The `reasoning_effort` key
  itself overrides the copy in `chat_template_kwargs`.
  - `reasoning_effort: "none"` alone still gives `enable_thinking: false`, so no regression. With
    `chat_template_kwargs.enable_thinking: true` it is still true, as before.
  - The one intended behaviour change: top-level `enable_thinking` now beats `chat_template_kwargs`
    (before, `or_insert` made it lose).
- **Clamps:**
  - `top_k`: -1 → clamp → 0, which means disabled in the worker; values ≥2^32 → u32::MAX; omitted
    stays None or the OpenAI default 0. `validate` still rejects values below -1.
  - `min_tokens`: validated ≥0 and ≤ max_tokens on i64 before the clamp.
  - `thinking_token_budget`: -1 → None (unlimited), values below -1 are a 400, large values → u32::MAX.
  - `stop_token_ids` and `allowed_token_ids` are range-checked in `validate`, so `as u32` in `to_pb`
    is safe; `to_pb` is only called after `validate`.
  - The caps apply to `stop`, `stop_token_ids`, `bad_words` and `allowed_token_ids`. The body limit
    is 32 MB, so the 2^18 cap can be reached.
- **Journal:** one writer thread, and the channel keeps FIFO order. The `Flush` reply comes only
  after every earlier line has been written, and each line is one `write_all` on an unbuffered
  `File`. `process::exit` runs after the flush. stdout logging is line-buffered, so no log lines
  are lost either.
- **Metrics:** existing names are unchanged. `qwen_inflight_capacity`, `qwen_worker_queue_seconds`
  and `qwen_batch_mean` are described, and the two histograms are recorded once per Done event on
  both the stream and JSON paths. The dashboard JSON parses. The new panels reference metrics that
  exist and use quantile labels. Prometheus scrapes only :8080, so there are no multi-gateway
  series clashes.
- **OpenAI shape:** the error JSON `{"error":{message,type,param,code}}` is unchanged. Status codes
  are unchanged apart from the ordering in #5. 503 still uses type `ServiceUnavailableError`.
- **Tool calls and structured output:** the diff does not touch `tool_calls_json`, the Done
  chunk's `tool_calls` delta, `finish_reason`, `response_format_json`, `check_tools` or tool
  grammars. The only Done changes are the `on_done` log and metrics and `Usage` copied instead of
  cloned, so the tool and JSON-schema paths behave exactly as before.

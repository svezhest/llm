# API, request handling and the gateway

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- Behaviour reference is vLLM v0.30 (`vllm serve` + the card recipe): what vLLM does that we
  don't is a 400; vLLM-unknown fields are ignored and logged.
- Tool calls: a port of vLLM's `qwen3_xml`; structured output and tool grammars: llguidance;
  reasoning: vLLM's `qwen3` parser + mlx-vlm's thinking budget.
- Sampler: vLLM's order in fp32 with an explicit per-request key (seeded requests reproduce).
- Gateway: admission (queue → Check → in-flight slot → Generate), `ReqGuard` owns the slots
  and cancels on disconnect. Multi-model routing is planned, not built.

## Request handling

- **Reasoning budget = native mlx-vlm `ThinkingBudgetCriteria`** (forces `\n</think>`
  into the token stream; KV cache stays consistent; 0.7: `prompt_preopens_thinking=True`
  or it never counts, `pop_forced_token_id()`). mlx-vlm ≥ 0.6.3 renders
  `enable_thinking=False` when unset — `_render` defaults it to True (template default). Do NOT hand-splice + restart
  generation — the earlier crutch corrupted answers and is removed. Thinking on/off
  is the template's `enable_thinking` (gateway, later wins: chat_template_kwargs <
  top-level enable_thinking < GATEWAY_THINKING; reasoning_effort fills it only when
  chat_template_kwargs did not; unset = template default = thinking). The template *pre-opens*
  `<think>`, so the worker decides "generation starts inside reasoning" from the
  prompt ids (last of `<think>`/`</think>` is an open one), not from a flag.
  `_GenState` = vLLM's `qwen3` reasoning parser (see README).
- **Prompt-length validation** (worker, before generation → `context_length_exceeded`
  → HTTP 400): count image tokens via the processor output (`_vision_inputs`, mlx-vlm `prepare_inputs` minus MLX),
  NOT a text-only `encode` — a 4096×2645 image expands to ~10644 tokens (text-only
  would say ~15). **No text fallback for vision** (would undercount → OOM); raises
  `prompt_count_failed` instead. Text-only uses a plain `encode`.
- **Tool calls**: Qwen3.5 emits `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n
  </parameter>\n</function>\n</tool_call>`. `parse_tool_calls` is a port of vLLM's
  `qwen3_xml` (vllm/parser/qwen3.py + parser_engine coercion, v0.30): one newline
  trimmed each side, types only from the tool schema (`coerce_to_schema_type` order
  null > integer > number > boolean > object/array > string, else json.loads, else
  the string). No `json.loads` of schema-less values — `"false"` stays a string.
- **Behaviour reference = vLLM v0.30** (`vllm serve` + card recipe: qwen3_xml tool
  parser, qwen3 reasoning parser). Anything vLLM does that we don't is a 400, never
  "accept and ignore"; vLLM-unknown fields are ignored + logged (as vLLM, which logs
  at debug). Request checks: gateway `request.rs` (field table, SamplingParams ranges,
  tool_choice rules) + worker `Check` RPC (template, context length, grammars, images
  on text-only) before queueing → 400 for streams too.
- **Sampler** = vLLM order on float32: penalties (presence/frequency over generated
  tokens, repetition over prompt+output; `_Penalties`) → temperature → min_p → top_k →
  top_p → draw (`make_sampler`, per-request key). Greedy stays argmax of
  `logits - logsumexp` in the model dtype — byte-identical to the old path. The old
  512-token presence window is gone (vLLM has none).
- **Log-probabilities (2026-10-22, `docs/perf/2026-10-21/design/logprobs.md`, review
  `reviews/logprobs.md`)**: `logprobs` / `top_logprobs` (≤ 20) per generated token (reasoning,
  answer and the final EOS / stop token; raw logits before penalties / temperature, fp32
  `logit − logsumexp`, built in `_step` on the same `async_eval` as the token and read after
  `r.y.item()` only for rows that asked — the off path adds no op) and `prompt_logprobs`
  (teacher forcing: the prefill keeps `return_hidden`'s post-norm hidden state and runs
  `lm_head` in `QWEN_LOGPROBS_CHUNK` (256) row chunks, fp32 log-softmax of the next prompt
  token; ~+10–12 % of a slab's FLOPs, only for such requests). Scored positions
  `[start + 1, n)`: position 0 never, nothing at or below the cache resume point (its logits
  are not kept) — `prompt_cache: false`/`"off"` = no lookup (cold, every position; still
  stored), `"read"` = lookup, nothing stored (variants of one baseline); the lookup for
  prompt_logprobs is over `ids[:-1]` so the last token is always scored. `max_tokens: 0`
  (only with prompt_logprobs) ends the row inside `_prefill` after the store (`length`, no
  `_join`, no decode step; the entry = prompt[:-1] as a generation's). Spans
  (`_assistant_spans`): the k-th `<|im_start|>assistant` block ↔ the k-th assistant message,
  matched from the end on a count mismatch (logged); `[start, end)` = after the header + the
  template's `<think>` pre-open ([`<think>`, `\n\n`, `</think>`, `\n\n`] = thinking off,
  [`<think>`, `\n`] = on) through `<|im_end|>`. The Ornith template renders a history
  assistant message exactly as generation prompt + reply (tested on the real tokenizer), so
  teacher-forced sums equal generation logprobs up to prefill-vs-decode kernel rounding
  (CPU 8-bit fp16: ≤ 3e-3 per token). Wire: `Token.lp` (not aligned with `text`; an empty
  Token may carry the EOS's lp), one `PromptLogprobs` event before Done; gateway: OpenAI
  `choices[0].logprobs` (always present, null when off), top-level `prompt_logprobs` (null
  where not scored) + `prompt_logprobs_messages` (per span: tokens / scored / logprob sum),
  journal keeps only counts and sums. DFlash never takes such requests. Tests
  `tests/test_logprobs.py`; real-model check `scripts/perf/logprobs/check_teacher_forcing.py` (9B, 2026-10-22,
  `docs/perf/2026-10-21/logprobs_check_real.json`: 5 AppWorld calls, generation byte-equal with the flags
  off, |Σtf − Σgen| ≤ 0.019 nat on 23–50 tokens, hit == cold values, 9.5–12.4K cold scored in 28–38 s).
- **Structured output / constrained tools** = **llguidance** (already an mlx-vlm
  dependency; Metal bitmask kernel; no grammar compile step, mask ~0.05–0.2 ms on the
  248K vocab; best JSON Schema coverage). `_Guide` masks logits before the sampler, so
  seed/sampler semantics are unchanged. The mask for token n+1 needs token n, so the
  guided loop does **not** queue the next step ahead (a step-ahead + separate masked
  sample measured 18 tok/s and degrading; mask-in-step is 30 vs 31 plain). Thinking:
  grammar activates on `</think>` (or a `<tool_call>` ending reasoning), before it only
  EOS is masked. JSON whitespace free (llguidance's lexer skips whitespace, so a
  `whitespace_pattern` does NOT bound runs — `QWEN_JSON_MAX_WS` masks by leading
  whitespace per token instead). Tool grammars (`_grammar`, Lark): one rule per
  function in xgrammar's `qwen_3_coder` layout, params in schema order, string values
  lazy raw text up to `\n</parameter>\n`, others `%json`. llguidance treats every
  added token as special: `<tool_call>` goes by id (`<[248058]>`), and free-text
  regexes can never produce it. DFlash is skipped for grammar requests.
  `required` = call first, no text before (parallel_tool_calls false: one call, nothing
  after); named = exactly that call; checked against xgrammar 0.2.8's own
  `get_model_structural_tag("qwen_3_coder")`. Schemas go through `_strict`
  (xgrammar strict_mode: no extra keys unless the schema allows them).
- **Request shaping as vLLM's renderer**: `developer` → `system` + all system messages
  merged at the front (`_developer_to_system`, renderers/hf.py); chat_template_kwargs:
  `chat_template`/`tokenize` → 400, keys not in the template dropped (logged), the
  request's `add_generation_prompt`/`continue_final_message` win (gateway). `Check`
  never 502s on a worker exception: template errors → 400, anything else → 500.
  No tools: `<tool_call>` + all text after it is cut from the answer (vLLM's parser).
- **Key order**: gateway serde_json has `preserve_order` — tools, tool_call history,
  response_format reach the worker in the client's order (sorted before: a tools
  prompt changed under the client's feet; schema property order is what the grammar
  emits).
- **Model id** = the worker's weights dir name (Health.model, Start.model, Check.model);
  `model` in a request must be it or a GATEWAY_SERVED_MODEL_NAMES alias, else 404
  (vLLM). `MODEL_NAME` is gone.
- **Gateway `ReqGuard`** (Drop) owns both slots (`pool::Slot`: permit + its gauge,
  decremented on drop) AND fires `Worker.Cancel` on client disconnect. Without
  owning the permits there, `queue_depth` stuck non-zero after a mid-stream disconnect.
- **Gateway admission**: `MAX_QUEUE` admits (waiting + running; 503 beyond, before any
  worker call) → `Check` (400/500) → wait for one of N in-flight slots → `Generate`.
  N = the worker's `HealthReply.max_batch` (0/1 = serialized; re-read every 15 s,
  the semaphore resized) or a fixed `GATEWAY_MAX_INFLIGHT`. Several gateways on one
  worker each send up to their own N, so they should split `max_batch`. Oversubscribing
  is NOT harmless: every Generate holds a worker gRPC thread while it waits in the
  worker, so Check/Cancel/Health starve, and the worker caps concurrent Generates with
  `RESOURCE_EXHAUSTED` → gateway 503 (slots freed). Check has a deadline
  (`GATEWAY_CHECK_TIMEOUT_S`, 30) → 503. The worker gets a per-generation id
  (Cancel goes by it; a reused client `request_id` must not cancel another request);
  responses keep `chatcmpl-<request_id>`. Duration/TTFT metrics start at handler
  entry; `qwen_gateway_queue_seconds` = wait for a slot. All histograms have buckets.
- **Gateway list caps** (vLLM has none): `stop` ≤ 64, `stop_token_ids` ≤ 1024,
  `bad_words` ≤ 1024, `allowed_token_ids` ≤ 2^18; token ids must fit uint32 (the
  vocab check is the worker's). Journal lines go through a writer thread (bounded
  10k-line channel, ordered, dropped + counted when full; flushed on SIGINT/SIGTERM,
  exit 130/143).

- **Stop sequences**: `_StopFilter` in the chunk loop keeps back the longest
  stream tail that is a prefix of a stop string; earliest match wins; applied to the
  whole output, reasoning included (vLLM's detokenizer does the same).

## Direction: multi-model routing (planned, not yet built)

Goal: several models / inference backends behind one gateway. `llama-server`
already speaks OpenAI API, so for a single model the gateway is optional — its
value is **routing + queue + metrics**. Plan: gateway keeps a registry
`model name → backend addr` and proxies by the request's `model` field (e.g.
`qwen3.6-35b-a3b`→:50051, `ornith-9b`→:50052). Since both current models are
mlx-vlm, this is "pick one of N same-protocol workers", not multi-protocol —
keep it that simple until a non-MLX (GGUF/llama.cpp) backend actually lands.

## Gotchas

- **MLX random state is per thread, and mlx_lm's sampler is compiled against the
  main thread's key.** gRPC handlers run in pool threads, so `mlx_lm.make_sampler`
  there drew every token from the same never-advancing key: temperature 0.7
  returned byte-identical answers, and `mx.random.seed` from the handler thread
  changed nothing. The worker's own `make_sampler` (worker.py, no mlx_lm since the
  0.7.6 bump) filters top_p/min_p/top_k itself and draws with an explicit per-request key split
  per step; request `seed` feeds that key (reproducible, verified through the
  gateway). Test any sampler change from a thread, not the REPL main thread.

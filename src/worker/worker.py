#!/usr/bin/env python3
"""Thin gRPC MLX worker for the Qwen3.x-family VL models (mlx-vlm).

Loads the model once, then serves the `Worker` service from worker.proto:
  - Generate: server-streaming Start/Token/Done/Error
  - Cancel:   cooperative stop of an in-flight generation
  - Health:   ready / model / load time
  - Shutdown: graceful stop

The model is not reentrant, so generations are serialized with a lock
(the gateway already keeps a single request in flight; this is belt-and-suspenders).

Run:
  uv run --group proto bash gen_proto.sh   # once, to codegen
  uv run python worker.py                   # serve on 127.0.0.1:50051
"""
import argparse
import base64
import functools
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent import futures

import grpc
import mlx.core as mx

import worker_pb2 as pb
import worker_pb2_grpc as pb_grpc

log = logging.getLogger("qwen.worker")

# Repo-relative default: <repo>/weights/<name>. worker.py lives at src/worker/,
# so the repo root is two directories up. Override with --model or QWEN_MODEL.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL = os.environ.get(
    "QWEN_MODEL",
    os.path.join(_REPO_ROOT, "weights", "qwen36-35b-a3b-mlx-ud3"),
)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Qwen3 emits tool calls inside <tool_call>...</tool_call>. The payload is EITHER
# JSON ({"name":..,"arguments":{..}}) OR Hermes XML (<function=NAME>
# <parameter=KEY>VALUE</parameter>...</function>) — this MLX build uses the XML form.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def _coerce(v: str):
    """Best-effort: turn a parameter string into a JSON scalar (number/bool/obj),
       else keep it as a string."""
    v = v.strip()
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def _parse_one(payload: str):
    """Parse one tool_call body (JSON or Hermes XML) -> (name, arguments_dict)."""
    body = payload.strip()
    if body.startswith("{"):                         # JSON form
        try:
            obj = json.loads(body)
            return obj.get("name", ""), obj.get("arguments", {})
        except json.JSONDecodeError:
            pass
    fm = _FUNC_RE.search(body)                        # Hermes XML form
    if fm:
        name = fm.group(1).strip()
        args = {k.strip(): _coerce(val) for k, val in _PARAM_RE.findall(fm.group(2))}
        return name, args
    return None, None


def parse_tool_calls(text: str):
    """Extract tool calls from generated text (handles both JSON and Hermes XML).

    Returns (tool_calls, leftover_text): tool_calls is a list of pb.ToolCall
    (arguments serialized to a JSON string, per the OpenAI shape); leftover_text
    is the answer with the tool_call blocks stripped out."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        name, args = _parse_one(m.group(1))
        if not name:
            continue
        args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        calls.append(pb.ToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}", name=name, arguments=args_str))
    leftover = _TOOL_CALL_RE.sub("", text).strip()
    return calls, leftover


def materialize_images(images):
    """mlx-vlm loads images from file paths / http URLs, not base64 `data:` URIs
       (which is what OpenAI clients like opencode/ai-sdk send). Decode any data:
       URL to a temp file and rewrite file:// to a plain path.

       Returns (paths, temp_files) — temp_files must be unlinked by the caller."""
    out, tmp = [], []
    for img in images:
        if img.startswith("data:"):
            header, _, b64 = img.partition(",")
            ext = "png" if "png" in header.lower() else "jpg"
            fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix="qwen_img_")
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64decode(b64))
            out.append(path)
            tmp.append(path)
        elif img.startswith("file://"):
            out.append(img[len("file://"):])
        else:
            out.append(img)
    return out, tmp


def _rebuild_messages(pb_messages):
    """pb.Message -> OpenAI-style dicts for the tokenizer chat template, preserving
       tool-call history (assistant tool_calls, tool-result messages)."""
    out = []
    for m in pb_messages:
        d = {"role": m.role, "content": m.content or ""}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls_json:
            try:
                d["tool_calls"] = json.loads(m.tool_calls_json)
            except json.JSONDecodeError:
                pass
        out.append(d)
    return out

# Context ceiling (prompt + generation). Model supports 262144 natively, but on
# 32 GB it's bounded by KV-cache memory. The weights are already 3-bit, so a
# 4-bit KV cache (~20 KiB/token) is a balanced match — 128K costs ~2.6 GB
# (model ~18 GB + KV still fits the default ~21.8 GB Metal wired limit; raise
# iogpu.wired_limit_mb for more headroom if needed).
# Context window (prompt + generation). 131072 is the shipped default for all
# three models; the 27B can run 200K (`QWEN_MAX_CONTEXT=204800`) — see README
# "Context & memory" for the KV budget before raising it.
MAX_CONTEXT = int(os.environ.get("QWEN_MAX_CONTEXT", "131072"))

# KV cache quantization (the MLX equivalent of llama.cpp's --cache-type-k q4_0).
# kv_bits applies to BOTH K and V uniformly here. quantized_kv_start keeps the
# first N tokens in fp16, then quantizes the tail.
#
# Per-model via QWEN_KV_BITS (default 4). The right value tracks the *weight*
# precision + the memory budget, so it differs by model:
#   - 35B-A3B (3-bit weights, ~18 GB, tight on 32 GB): kv_bits=4 — a 4-bit KV is
#     the balanced match for 3-bit weights, and q8 KV would blow the budget.
#   - 9B (8-bit weights, ~10 GB, tons of headroom): kv_bits=8 — near-lossless KV
#     costs only ~2 GB at 128K, so there's no reason to compress it to 4-bit.
#   - 27B (5-bit weights, ~18 GB): kv_bits=8 — ~32 KB/token (16 full-attention
#     layers x 4 KV heads x head_dim 256), ~3.9 GB at 128K; fits with the wired bump.
# Only the full-attention layers' KVCache is quantized; the linear-attention
# (Gated DeltaNet) layers keep a small constant recurrent state that mlx-lm's
# maybe_quantize_kv_cache skips (ArraysCache has no to_quantized).
KV_BITS = int(os.environ.get("QWEN_KV_BITS", "4"))
KV_KWARGS = {"kv_bits": KV_BITS, "kv_group_size": 64, "quantized_kv_start": 5000}

# Chunked prefill: process a long prompt in N-token slabs instead of one giant
# matmul, so the prefill ACTIVATION peak (not the KV cache) stays bounded. A 39K
# prompt prefilled in one shot is what spiked us over the wired limit. 512 is a
# safe slab; raise via QWEN_PREFILL_STEP if you have headroom and want speed.
PREFILL_STEP = int(os.environ.get("QWEN_PREFILL_STEP", "512"))

# Metal wired-memory ceiling. Default on a 32 GB Mac is ~21.8 GB — barely above
# the ~18 GB of 3-bit weights, leaving almost nothing for KV + prefill. Raise it
# in-process when QWEN_WIRED_LIMIT_GB is set. NOTE: going above the OS-recommended
# max also needs `sudo sysctl iogpu.wired_limit_mb=<MB>` — set_wired_limit alone
# cannot exceed what the kernel allows.
_WIRED_GB = float(os.environ.get("QWEN_WIRED_LIMIT_GB", "0") or "0")

# Sampling (native mlx-lm sampler + logits processors via mlx-vlm — no custom code).
# Env sets the process-wide defaults; a request's temperature/top_p (sentinel
# -1 / 0 from the gateway when omitted) and top_k/min_p/presence_penalty
# (proto optional fields) override them. Default:
# greedy/argmax (temperature=0 => top_p/top_k/min_p irrelevant) with an ADDITIVE
# presence penalty 1.5 over a 512-token window (the Qwen card's *non-thinking*
# setting). Multiplicative repetition penalty is OFF.
# For reasoning / benchmarks set QWEN_PRESENCE_PENALTY=0 and the card's sampler
# (thinking: temperature 1.0, top_p 0.95, top_k 20, min_p 0) — see README.
SAMPLING_KWARGS = {
    "temperature": float(os.environ.get("QWEN_TEMPERATURE", "0.0")),    # 0 => argmax
    "top_p": float(os.environ.get("QWEN_TOP_P", "1.0")),                # 1.0 => off
    "top_k": int(os.environ.get("QWEN_TOP_K", "0")),                    # 0 => off
    "min_p": float(os.environ.get("QWEN_MIN_P", "0.0")),                # 0 => off
    "presence_penalty": float(os.environ.get("QWEN_PRESENCE_PENALTY", "1.5")),
    "presence_context_size": int(os.environ.get("QWEN_PRESENCE_CONTEXT", "512")),
}

# MLX buffer pool. Freed Metal buffers go to an allocator free-list ("cache")
# instead of back to the OS; they stay resident and count against the wired
# working set. The pool limit defaults to the memory limit, i.e. unbounded in
# practice, and the list is size-bucketed: after request N the pool holds that
# request's prefill activations + KV growth buffers, and request N+1's prefill
# asks for buffers of other sizes, so it allocates on top of the retained ones
# — that's the second-request OOM on 32 GB. Two knobs:
#   QWEN_CLEAR_CACHE=1  (default) drop the whole pool after every generation
#                       (and between prefill chunks) — the old behaviour.
#   QWEN_CACHE_LIMIT_GB bound the pool via mx.set_cache_limit; with
#                       QWEN_CLEAR_CACHE=0 buffers are reused across requests
#                       (faster decode start) and the pool can't grow past this.
CLEAR_CACHE = os.environ.get("QWEN_CLEAR_CACHE", "1") == "1"
CACHE_LIMIT_GB = float(os.environ.get("QWEN_CACHE_LIMIT_GB", "3"))

# Prompt cache (QWEN_PROMPT_CACHE=1): single-slot snapshot of the KV/recurrent
# state taken after the prefill of the previous text-only prompt. The next
# prompt that extends it (multi-turn chat) only prefills the delta. No prefix
# trimming: the gated-delta-net layers keep a recurrent ArraysCache that cannot
# be rewound, and quantized KV keys are tuples the library's trim mishandles.
PROMPT_CACHE = os.environ.get("QWEN_PROMPT_CACHE", "0") == "1"


def _copy_cache(caches):
    """Fresh cache objects holding the same (immutable) arrays.

    MLX `a[idx] = v` rebinds the Python array object in place, so two cache
    objects must not share array objects — slicing gives new ones over the same
    buffer (copy-on-write from there). ArraysCache entries are replaced whole
    by the model, so a new list is enough."""
    out = []
    for c in caches:
        if hasattr(c, "cache"):                       # ArraysCache (recurrent state)
            n = type(c)(size=len(c.cache))
            n.cache = list(c.cache)
        elif isinstance(c.keys, tuple):               # QuantizedKVCache
            n = type(c)(group_size=c.group_size, bits=c.bits)
            n.keys = tuple(k[..., : c.offset, :] for k in c.keys)
            n.values = tuple(v[..., : c.offset, :] for v in c.values)
            n.offset = c.offset
        else:                                         # KVCache
            n = type(c)()
            n.keys = c.keys[..., : c.offset, :]
            n.values = c.values[..., : c.offset, :]
            n.offset = c.offset
        out.append(n)
    return out


def _cache_arrays(caches):
    arrs = []
    for c in caches:
        if hasattr(c, "cache"):
            arrs += [a for a in c.cache if a is not None]
        else:
            for x in (c.keys, c.values):
                arrs += list(x) if isinstance(x, tuple) else [x]
    return arrs


class _Snapshot:
    def __init__(self, ids, cache):
        self.ids = ids
        self.cache = cache
        arrs = _cache_arrays(cache)
        mx.eval(arrs)
        self.nbytes = sum(a.nbytes for a in arrs)


class _StopFilter:
    """Holds back the tail of the stream that could be the start of a stop
    sequence, so a stop split across chunks is still caught and never emitted."""

    def __init__(self, stops):
        self.stops = [x for x in stops if x]
        self.buf = ""
        self.hit = False

    def feed(self, text):
        if not self.stops:
            return text
        self.buf += text
        for st in self.stops:
            i = self.buf.find(st)
            if i >= 0:
                self.hit = True
                out, self.buf = self.buf[:i], ""
                return out
        keep = 0
        for st in self.stops:
            for k in range(min(len(st) - 1, len(self.buf)), 0, -1):
                if st.startswith(self.buf[-k:]):
                    keep = max(keep, k)
                    break
        out, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        return out

    def flush(self):
        out, self.buf = self.buf, ""
        return out


class Engine:
    """Owns the MLX model and runs (serialized) generations."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ready = False
        self.load_ms = 0
        self._lock = threading.Lock()
        self._cancels: dict[str, threading.Event] = {}
        self._cancels_lock = threading.Lock()
        self._snap: _Snapshot | None = None

    def load(self):
        from mlx_vlm import load
        from mlx_vlm.prompt_utils import apply_chat_template  # noqa: F401
        from mlx_vlm.utils import load_config

        log.info("loading model: %s", self.model_path)
        t = time.time()
        self.model, self.processor = load(self.model_path)
        self.config = load_config(self.model_path)
        self.load_ms = int((time.time() - t) * 1000)
        if _WIRED_GB > 0:
            want = int(_WIRED_GB * 1024**3)
            cap = int(mx.device_info()["max_recommended_working_set_size"])
            if want > cap:
                log.warning(
                    "requested wired limit %.1f GB exceeds kernel max %.1f GB; "
                    "clamping. Raise it with: sudo sysctl iogpu.wired_limit_mb=%d",
                    _WIRED_GB, cap / 1024**3, int(_WIRED_GB * 1024))
                want = cap
            mx.set_wired_limit(want)
            log.info("wired limit set to %.1f GB", want / 1024**3)
        mx.set_cache_limit(int(CACHE_LIMIT_GB * 1024**3))
        log.info("buffer pool limit %.1f GB, clear after request: %s, prompt cache: %s",
                 CACHE_LIMIT_GB, CLEAR_CACHE, PROMPT_CACHE)
        self.ready = True
        log.info("model loaded in %d ms", self.load_ms)

    # --- cancellation -----------------------------------------------------
    def cancel(self, req_id: str) -> bool:
        with self._cancels_lock:
            ev = self._cancels.get(req_id)
        if ev is not None:
            ev.set()
            log.info("cancel requested: %s", req_id)
            return True
        log.debug("cancel for unknown id: %s", req_id)
        return False

    def _register(self, req_id: str) -> threading.Event:
        ev = threading.Event()
        with self._cancels_lock:
            self._cancels[req_id] = ev
        return ev

    def _unregister(self, req_id: str):
        with self._cancels_lock:
            self._cancels.pop(req_id, None)

    # --- prompt preparation ----------------------------------------------
    def _prepare(self, formatted: str, images: list):
        """Tokenize once -> (input_ids, id list, pixel_values, mask, extra kwargs).

        Vision goes through mlx-vlm's prepare_inputs (the image preprocessor is
        the expensive part); the result is fed straight into generation, so the
        images are processed exactly once. The id count is exact, image tokens
        included (a 4K image ≈ 10k tokens vs ~15 for the text placeholder) —
        there is deliberately no text-only fallback for vision, an undercount
        would let an oversized prompt through to an OOM.
        """
        if images:
            from mlx_vlm.utils import prepare_inputs
            inputs = prepare_inputs(
                self.processor, images=images, prompts=formatted,
                image_token_index=getattr(self.model.config, "image_token_index", None),
                add_special_tokens=True, padding=False,
            )
            input_ids = inputs["input_ids"]
            extra = {k: v for k, v in inputs.items()
                     if k not in ("input_ids", "pixel_values", "attention_mask")}
            return (input_ids, input_ids.flatten().tolist(), inputs.get("pixel_values"),
                    inputs.get("attention_mask"), extra)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        ids = tokenizer.encode(formatted)
        return mx.array([ids]), ids, None, None, {}

    def _run(self, input_ids, pixel_values, mask, extra, prompt_cache, *,
             max_tokens, sampling, thinking, on_prefilled=None):
        """Prefill + decode, yielding (token_id, text) pairs; sets self._finish.

        A trimmed-down generate_step/stream_generate from mlx-vlm 0.6.2. Kept
        local so that (a) the prompt cache snapshot can be taken at the exact
        point where the cache holds the prompt minus its last token and nothing
        else, (b) pre-tokenized vision inputs go in directly, and (c) the
        finish reason and buffer-pool clearing are under our control.
        """
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        from mlx_vlm.generate.common import (generation_stream, maybe_quantize_kv_cache,
                                             wired_limit)
        from mlx_vlm.tokenizer_utils import make_streaming_detokenizer

        model = self.model
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        sampler = make_sampler(temp=sampling["temperature"], top_p=sampling["top_p"],
                               min_p=sampling["min_p"], top_k=sampling["top_k"])
        processors = make_logits_processors(
            presence_penalty=sampling["presence_penalty"] or None,
            presence_context_size=sampling["presence_context_size"])
        quantize = functools.partial(maybe_quantize_kv_cache, **{
            "quantized_kv_start": KV_KWARGS["quantized_kv_start"],
            "kv_group_size": KV_KWARGS["kv_group_size"], "kv_bits": KV_KWARGS["kv_bits"]})
        kwargs = dict(extra)
        tokens = mx.array([], dtype=input_ids.dtype)

        def step(y, inputs_embeds=None):
            nonlocal tokens, kwargs
            with mx.stream(generation_stream):
                out = model.language_model(y, inputs_embeds=inputs_embeds,
                                           cache=prompt_cache, **kwargs)
                logits = out.logits[:, -1, :]
                if processors:
                    tokens = mx.concat([tokens, y.flatten()])
                    for proc in processors:
                        logits = proc(tokens, logits)
                quantize(prompt_cache)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                kwargs = {}
                return sampler(logprobs), logprobs.squeeze(0)

        with wired_limit(model, [generation_stream]):
            with mx.stream(generation_stream):
                emb = model.get_input_embeddings(input_ids, pixel_values, mask=mask, **kwargs)
                inputs_embeds = emb.inputs_embeds
                kwargs.update({k: v for k, v in emb.to_dict().items()
                               if k != "inputs_embeds" and v is not None})
                # Everything but the last prompt token goes through the cache in
                # PREFILL_STEP slabs; the last token is the first decode step.
                while inputs_embeds.shape[1] > 1:
                    n = min(PREFILL_STEP, inputs_embeds.shape[1] - 1)
                    model.language_model(input_ids[:, :n], inputs_embeds=inputs_embeds[:, :n],
                                         cache=prompt_cache, **kwargs)
                    quantize(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    inputs_embeds = inputs_embeds[:, n:]
                    input_ids = input_ids[:, n:]
                    if CLEAR_CACHE:
                        mx.clear_cache()
                if on_prefilled is not None:
                    on_prefilled(prompt_cache)
                y, logprobs = step(input_ids, inputs_embeds)
            mx.async_eval(y, logprobs)

            detok = make_streaming_detokenizer(self.processor)
            self._finish = "length"
            n = 0
            while True:
                if n != max_tokens:
                    next_y, next_lp = step(y[None])
                    mx.async_eval(next_y, next_lp)
                if n == 0:
                    mx.eval(y)
                if n == max_tokens:
                    break
                tok = y.item()
                if thinking is not None:
                    thinking(tok)
                if tokenizer.stopping_criteria(tok):
                    self._finish = "stop"
                    break
                detok.add_token(tok)
                yield tok, detok.last_segment
                if n % 256 == 0 and CLEAR_CACHE:
                    mx.clear_cache()
                if thinking is not None:
                    next_y = thinking.apply_forced_token(next_y)
                y, logprobs = next_y, next_lp
                n += 1
            detok.finalize()
            seg = detok.last_segment   # property with side effects: read once
            if seg:
                yield None, seg

    # --- generation -------------------------------------------------------
    def generate(self, req: pb.Request):
        """Yield pb.Event. Serialized across requests via self._lock."""
        from mlx_vlm.prompt_utils import apply_chat_template

        p = req.params
        cancel_ev = self._register(req.id)
        # Tools mode: tools provided AND not explicitly disabled. Uses mlx-vlm's
        # apply_chat_template with the FULL message history + tools (the normal path
        # only uses the last message); image tokens are still inserted for num_images.
        tools_mode = bool(p.tools_json) and p.tool_choice != "none"
        tmp_files = []
        try:
            with self._lock:
                yield pb.Event(id=req.id, start=pb.Start(role="assistant"))

                images, tmp_files = materialize_images(list(req.images))
                # One path for both modes: mlx-vlm's apply_chat_template takes a FULL
                # message list (system + history + tool messages), forwards `tools` to
                # the Qwen Jinja template, AND inserts image tokens for num_images.
                # The old non-tools branch passed only the LAST message, silently
                # dropping system prompts and history — a long-standing footgun.
                oai_messages = _rebuild_messages(req.messages)
                extra = {"tools": json.loads(p.tools_json)} if tools_mode else {}
                formatted = apply_chat_template(
                    self.processor, self.config, oai_messages,
                    num_images=len(images), enable_thinking=p.enable_thinking,
                    **extra,
                )

                # Tokenize once (images preprocessed once), reject prompts that
                # don't fit, and clamp the generation to the remaining budget.
                try:
                    input_ids, ids, pixel_values, mask, extra = self._prepare(formatted, images)
                except Exception as e:  # noqa: BLE001 — never proceed un-measured
                    log.exception("prompt preparation failed: %s", req.id)
                    yield pb.Event(id=req.id, error=pb.Error(
                        kind="prompt_count_failed",
                        msg=f"could not measure prompt length: {e}"))
                    return
                prompt_tokens = len(ids)
                log.info("id=%s prompt_tokens=%d (images=%d)",
                         req.id, prompt_tokens, len(images))
                if prompt_tokens >= MAX_CONTEXT:
                    yield pb.Event(id=req.id, error=pb.Error(
                        kind="context_length_exceeded",
                        msg=f"prompt is {prompt_tokens} tokens, exceeds context "
                            f"limit {MAX_CONTEXT}"))
                    return

                state = _GenState(p)
                t0 = time.time()
                ttft_ms = 0
                mx.reset_peak_memory()

                remaining = MAX_CONTEXT - prompt_tokens
                max_tokens = min(p.max_tokens or 1024, remaining)

                # Reasoning budget: forces \n</think> into the token stream when
                # exceeded (the KV cache stays consistent). Same criteria object
                # mlx-vlm's stream_generate builds.
                thinking = None
                if p.enable_thinking and p.thinking_budget > 0:
                    from mlx_vlm.utils import ThinkingBudgetCriteria
                    tokenizer = getattr(self.processor, "tokenizer", self.processor)
                    start_id = tokenizer.encode(THINK_OPEN, add_special_tokens=False)[-1]
                    thinking = ThinkingBudgetCriteria(
                        tokenizer=tokenizer, thinking_budget=p.thinking_budget,
                        thinking_end_token=THINK_CLOSE, thinking_start_token=THINK_OPEN,
                        enable_thinking=start_id in ids)

                sampling = dict(SAMPLING_KWARGS)
                if p.temperature >= 0:
                    sampling["temperature"] = p.temperature
                if p.top_p > 0:
                    sampling["top_p"] = p.top_p
                if p.HasField("top_k"):
                    sampling["top_k"] = p.top_k
                if p.HasField("min_p"):
                    sampling["min_p"] = p.min_p
                if p.HasField("presence_penalty"):
                    sampling["presence_penalty"] = p.presence_penalty
                if p.HasField("seed"):
                    mx.random.seed(p.seed)

                # Prompt cache: reuse the snapshot when the new prompt extends it
                # (the snapshot holds ids[:-1] of the previous prompt, so an exact
                # repeat is a hit too). Text-only conversations only.
                from mlx_vlm.models.cache import make_prompt_cache
                lm = self.model.language_model
                snap = self._snap if (PROMPT_CACHE and not images) else None
                reused = 0
                if snap is not None and len(ids) > len(snap.ids) \
                        and ids[: len(snap.ids)] == snap.ids:
                    reused = len(snap.ids)
                    prompt_cache = _copy_cache(snap.cache)
                    input_ids = mx.array([ids[reused:]])
                    # Text-only mRoPE continues from the cache offset with a zero
                    # delta; a preceding vision request leaves its own delta here.
                    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
                    lm._position_ids = None
                    log.info("id=%s prompt cache HIT: %d/%d tokens reused, prefilling %d",
                             req.id, reused, prompt_tokens, prompt_tokens - reused)
                else:
                    prompt_cache = make_prompt_cache(lm)
                    if PROMPT_CACHE:
                        common = 0
                        if snap is not None:
                            for a, b in zip(snap.ids, ids):
                                if a != b:
                                    break
                                common += 1
                        log.info("id=%s prompt cache MISS (common prefix %d, snapshot %d, "
                                 "images=%d)", req.id, common,
                                 len(snap.ids) if snap else 0, len(images))

                def snapshot(cache):
                    if not PROMPT_CACHE or images:
                        return
                    if any(getattr(c, "keys", 1) is None for c in cache):
                        return
                    self._snap = _Snapshot(ids[:-1], _copy_cache(cache))
                    log.info("id=%s prompt cache snapshot: %d tokens, %.1f MB",
                             req.id, len(self._snap.ids), self._snap.nbytes / 1e6)

                stop = _StopFilter(list(p.stop))
                answer_buf = []   # tools mode: buffer answer tokens, parse at the end
                gen = self._run(input_ids, pixel_values, mask, extra, prompt_cache,
                                max_tokens=max_tokens, sampling=sampling,
                                thinking=thinking, on_prefilled=snapshot)
                for tok, text in gen:
                    if cancel_ev.is_set():
                        gen.close()
                        yield _done(req.id, "cancelled", state, t0, ttft_ms, prompt_tokens)
                        return
                    if ttft_ms == 0:
                        ttft_ms = int((time.time() - t0) * 1000)
                    if tok is not None:
                        state.gen_tokens += 1
                    think_part, ans_part = state.feed(text)
                    if think_part:
                        yield pb.Event(id=req.id, token=pb.Token(text=think_part, thinking=True))
                    if ans_part:
                        ans_part = stop.feed(ans_part)
                    if ans_part:
                        if tools_mode:
                            # Don't stream raw answer (may contain <tool_call> XML); buffer it.
                            answer_buf.append(ans_part)
                        else:
                            yield pb.Event(id=req.id, token=pb.Token(text=ans_part, thinking=False))
                    if stop.hit:
                        gen.close()
                        break
                finish = "stop" if stop.hit else self._finish
                tail = stop.flush() if not stop.hit else ""

                gen_s = time.time() - t0 - ttft_ms / 1000
                tps = state.gen_tokens / gen_s if gen_s > 0 else 0.0
                log.info("id=%s finish=%s gen_tokens=%d ttft_ms=%d tps=%.1f cached=%d "
                         "peak=%.2fGB active=%.2fGB pool=%.2fGB",
                         req.id, finish, state.gen_tokens, ttft_ms, tps, reused,
                         mx.get_peak_memory() / 1e9, mx.get_active_memory() / 1e9,
                         mx.get_cache_memory() / 1e9)

                if tools_mode:
                    answer_buf.append(tail)
                    tool_calls, leftover = parse_tool_calls("".join(answer_buf))
                    if leftover:
                        yield pb.Event(id=req.id, token=pb.Token(text=leftover, thinking=False))
                    if tool_calls:
                        finish = "tool_calls"
                    yield _done(req.id, finish, state, t0, ttft_ms, prompt_tokens, tool_calls, tps)
                else:
                    if tail:
                        yield pb.Event(id=req.id, token=pb.Token(text=tail, thinking=False))
                    yield _done(req.id, finish, state, t0, ttft_ms, prompt_tokens, tps=tps)
        except Exception as e:  # noqa: BLE001 — map any failure to a wire Error
            log.exception("generation failed: %s", req.id)
            yield pb.Event(id=req.id, error=pb.Error(kind=type(e).__name__, msg=str(e)))
        finally:
            self._unregister(req.id)
            for f in tmp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass
            # See CLEAR_CACHE: drop the buffer pool, or keep it (bounded by
            # QWEN_CACHE_LIMIT_GB) for the next request.
            if CLEAR_CACHE:
                mx.clear_cache()


class _GenState:
    """Tracks whether the current token is reasoning text (for tagging only).

    The reasoning budget itself is enforced natively by mlx-vlm; this just
    classifies each chunk as thinking vs answer so the gateway can route it to
    `reasoning_content` vs `content`.
    """

    def __init__(self, params: pb.Params):
        # When thinking is enabled, the chat template already opens <think> in the
        # prompt, so generation STARTS inside the reasoning block (the literal
        # <think> token is never emitted). Seed the state accordingly.
        self.in_thinking = bool(params.enable_thinking)
        self.gen_tokens = 0

    def feed(self, text: str) -> tuple[str, str]:
        """Split a chunk into (thinking_part, answer_part).

        A single chunk can straddle the </think> boundary — notably when
        thinking_budget force-injects "\\n</think>" and the answer follows in
        the same chunk. Classifying whole chunks put the answer into
        reasoning_content and left content empty; split instead.
        """
        if THINK_OPEN in text:
            self.in_thinking = True
        if self.in_thinking:
            if THINK_CLOSE in text:
                self.in_thinking = False
                pre, _, post = text.partition(THINK_CLOSE)
                return (pre + THINK_CLOSE, post)
            return (text, "")
        return ("", text)


def _done(req_id, finish, state, t0, ttft_ms, prompt_tokens, tool_calls=None, tps=0.0):
    usage = pb.Usage(
        prompt_tokens=prompt_tokens,
        gen_tokens=state.gen_tokens,
        tps=tps,
        ttft_ms=ttft_ms,
        peak_gb=mx.get_peak_memory() / 1e9,
    )
    return pb.Event(id=req_id, done=pb.Done(
        finish=finish, usage=usage, tool_calls=tool_calls or []))


class WorkerServicer(pb_grpc.WorkerServicer):
    def __init__(self, engine: Engine, stop_event: threading.Event):
        self.engine = engine
        self.stop_event = stop_event

    def Generate(self, request, context):
        log.info("Generate id=%s images=%d max_tokens=%d think=%s budget=%d",
                 request.id, len(request.images), request.params.max_tokens,
                 request.params.enable_thinking, request.params.thinking_budget)
        yield from self.engine.generate(request)

    def Cancel(self, request, context):
        self.engine.cancel(request.id)
        return pb.Ack()

    def Health(self, request, context):
        return pb.HealthReply(ready=self.engine.ready,
                              model=self.engine.model_path,
                              load_ms=self.engine.load_ms)

    def Shutdown(self, request, context):
        log.warning("shutdown requested")
        self.stop_event.set()
        return pb.Ack()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:50051")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )

    engine = Engine(args.model)
    engine.load()

    stop_event = threading.Event()
    # Single MLX worker: one generation at a time, but allow Health/Cancel concurrently.
    # Vision prompts carry base64 image data URIs: a detailed map snapshot is several MB,
    # past gRPC's 4 MB default receive cap. Lift it so high-res vision requests get through.
    _MAX_MSG = 64 * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", _MAX_MSG),
            ("grpc.max_send_message_length", _MAX_MSG),
        ],
    )
    pb_grpc.add_WorkerServicer_to_server(WorkerServicer(engine, stop_event), server)
    server.add_insecure_port(args.addr)
    server.start()
    log.info("worker serving on %s", args.addr)
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        server.stop(grace=2).wait()
        log.info("worker stopped")


if __name__ == "__main__":
    main()

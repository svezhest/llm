#!/usr/bin/env python3
"""Thin gRPC MLX worker for Qwen3.6-35B-A3B (vision-language).

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
MAX_CONTEXT = 131072

# KV cache quantization (the MLX equivalent of llama.cpp's --cache-type-k q4_0).
# kv_bits applies to BOTH K and V uniformly here. quantized_kv_start keeps the
# first N tokens in fp16, then quantizes the tail.
KV_KWARGS = {"kv_bits": 4, "kv_group_size": 64, "quantized_kv_start": 5000}

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

# Sampling (native mlx-lm logits processors via mlx-vlm — no custom code):
# greedy/argmax decoding (temperature=0, so top_p is irrelevant) with an ADDITIVE
# presence penalty over a 512-token window. Multiplicative repetition penalty and
# frequency penalty are OFF (left as None). Env-overridable for the run.
SAMPLING_KWARGS = {
    "temperature": float(os.environ.get("QWEN_TEMPERATURE", "0.0")),    # 0 => argmax
    "top_p": 1.0,                                                       # off
    "presence_penalty": float(os.environ.get("QWEN_PRESENCE_PENALTY", "1.5")),
    "presence_context_size": int(os.environ.get("QWEN_PRESENCE_CONTEXT", "512")),
}


class Engine:
    """Owns the MLX model and runs (serialized) generations."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ready = False
        self.load_ms = 0
        self._lock = threading.Lock()
        self._cancels: dict[str, threading.Event] = {}
        self._cancels_lock = threading.Lock()

    def load(self):
        from mlx_vlm import load, stream_generate  # noqa: F401  (import cost up-front)
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

    # --- context accounting ----------------------------------------------
    def _count_prompt_tokens(self, formatted: str, images: list) -> int:
        """Exact prompt token count, including expanded image tokens.

        Vision: counted via mlx-vlm's prepare_inputs (the same path generation
        takes). There is deliberately NO text-only fallback for images — a text
        `encode` ignores image tokens (a 4K image ≈ 10k tokens vs ~15 for the
        placeholder), so falling back would massively undercount and let an
        oversized prompt through → OOM. If image counting fails we raise, so the
        request is rejected loudly instead of silently mis-measured.
        """
        if images:
            from mlx_vlm.utils import prepare_inputs
            inputs = prepare_inputs(
                self.processor, images=images, prompts=formatted,
                add_special_tokens=False, padding=False,
            )
            return int(inputs["input_ids"].shape[-1])
        # Text-only path: a plain encode is exact.
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return len(tokenizer.encode(formatted))

    # --- generation -------------------------------------------------------
    def generate(self, req: pb.Request):
        """Yield pb.Event. Serialized across requests via self._lock."""
        from mlx_vlm import stream_generate
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

                # Reject prompts that don't fit the context window, and clamp the
                # generation to whatever budget is left (prompt + output <= 64K).
                # For vision, image placeholders expand to MANY tokens, so count
                # the real input_ids via prepare_inputs (what generation uses
                # internally) — a text-only encode would badly undercount images.
                try:
                    prompt_tokens = self._count_prompt_tokens(formatted, images)
                except Exception as e:  # noqa: BLE001 — never proceed un-measured
                    log.exception("prompt token count failed: %s", req.id)
                    yield pb.Event(id=req.id, error=pb.Error(
                        kind="prompt_count_failed",
                        msg=f"could not measure prompt length: {e}"))
                    return
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
                last = None

                remaining = MAX_CONTEXT - prompt_tokens
                max_tokens = min(p.max_tokens or 1024, remaining)

                # Reasoning budget: mlx-vlm enforces it natively (forces \n</think>
                # into the token stream when the budget is exceeded, keeping the KV
                # cache consistent). Just pass the kwargs through.
                think_kwargs = {"enable_thinking": p.enable_thinking}
                if p.enable_thinking and p.thinking_budget > 0:
                    think_kwargs["thinking_budget"] = p.thinking_budget

                answer_buf = []   # tools mode: buffer answer tokens, parse at the end
                for chunk in stream_generate(
                    self.model, self.processor, formatted, images,
                    max_tokens=max_tokens, prefill_step_size=PREFILL_STEP,
                    **SAMPLING_KWARGS, **KV_KWARGS, **think_kwargs,
                ):
                    if cancel_ev.is_set():
                        yield _done(req.id, "cancelled", state, t0, ttft_ms, last)
                        return
                    last = chunk
                    if ttft_ms == 0:
                        ttft_ms = int((time.time() - t0) * 1000)

                    think_part, ans_part = state.feed(chunk.text)
                    if think_part:
                        yield pb.Event(id=req.id, token=pb.Token(text=think_part, thinking=True))
                    if ans_part:
                        if tools_mode:
                            # Don't stream raw answer (may contain <tool_call> XML); buffer it.
                            answer_buf.append(ans_part)
                        else:
                            yield pb.Event(id=req.id, token=pb.Token(text=ans_part, thinking=False))

                if tools_mode:
                    tool_calls, leftover = parse_tool_calls("".join(answer_buf))
                    if leftover:
                        yield pb.Event(id=req.id, token=pb.Token(text=leftover, thinking=False))
                    finish = "tool_calls" if tool_calls else "stop"
                    yield _done(req.id, finish, state, t0, ttft_ms, last, tool_calls)
                else:
                    yield _done(req.id, "stop", state, t0, ttft_ms, last)
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
            # Release MLX's reclaimable buffer pool. Without this the cache from
            # this generation stays resident and the NEXT request's prefill OOMs
            # (exactly the second-request crash we saw on the 32 GB box).
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
        self.gen_tokens += 1
        if THINK_OPEN in text:
            self.in_thinking = True
        if self.in_thinking:
            if THINK_CLOSE in text:
                self.in_thinking = False
                pre, _, post = text.partition(THINK_CLOSE)
                return (pre + THINK_CLOSE, post)
            return (text, "")
        return ("", text)


def _done(req_id, finish, state, t0, ttft_ms, last, tool_calls=None):
    usage = pb.Usage(
        prompt_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
        gen_tokens=state.gen_tokens,
        tps=float(getattr(last, "generation_tps", 0.0) or 0.0),
        ttft_ms=ttft_ms,
        peak_gb=float(getattr(last, "peak_memory", 0.0) or 0.0),
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

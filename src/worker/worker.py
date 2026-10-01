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
import bisect
import functools
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from concurrent import futures

import grpc
import mlx.core as mx
from mlx_vlm.tokenizer_utils import BPEStreamingDetokenizer, _remove_space

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

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
FUNC_OPEN = "<function="
FUNC_CLOSE = "</function>"

# Tool calls, parsed as vLLM's `qwen3_xml` parser does (vllm/parser/qwen3.py,
# v0.30): <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n
# </function>\n</tool_call>. A value loses exactly one leading and one trailing
# newline and nothing else; a parameter ends at </parameter> or at the next
# <parameter= (an unterminated last one is dropped).
_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
                       re.DOTALL)
_TAG_RE = re.compile("|".join(re.escape(t) for t in (TOOL_OPEN, TOOL_CLOSE, FUNC_OPEN, FUNC_CLOSE)))


def _trim_newlines(v):
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _schema_types(schema):
    """vLLM's extract_types_from_schema: type / enum / anyOf-oneOf-allOf, else string."""
    if not isinstance(schema, dict):
        return ["string"]
    types = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    for v in schema.get("enum") or []:
        types.add("null" if v is None else "boolean" if isinstance(v, bool)
                  else "integer" if isinstance(v, int) else "number" if isinstance(v, float)
                  else "string" if isinstance(v, str) else "array" if isinstance(v, list)
                  else "object")
    for k in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(k) or []:
            types.update(_schema_types(sub))
    return list(types) if types else ["string"]


_TYPE_ALIASES = {"str": "string", "text": "string", "varchar": "string", "char": "string",
                 "enum": "string", "int": "integer", "int32": "integer", "int64": "integer",
                 "uint": "integer", "uint32": "integer", "uint64": "integer", "long": "integer",
                 "short": "integer", "unsigned": "integer", "float": "number", "float32": "number",
                 "float64": "number", "double": "number", "bool": "boolean", "dict": "object",
                 "arr": "array", "list": "array", "sequence": "array"}


def _finite(x):
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, list):
        return all(_finite(v) for v in x)
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    return True


def _coerce(value, types):
    """vLLM's coerce_to_schema_type: the first type (null > integer > number >
    boolean > object > array > string) the raw string parses as; else JSON if
    it is JSON, else the string itself."""
    norm = {_TYPE_ALIASES.get(t.strip().lower(), t.strip().lower()) for t in types}
    for t in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if t not in norm:
            continue
        if t == "null" and value.lower() == "null":
            return None
        if t == "string":
            return value
        if t == "integer":
            try:
                return int(value)
            except ValueError:
                continue
        if t == "number":
            try:
                f = float(value)
            except ValueError:
                continue
            if math.isfinite(f):
                return int(f) if f == int(f) else f
        if t == "boolean":
            if value.lower().strip() in ("true", "1"):
                return True
            if value.lower().strip() in ("false", "0"):
                return False
        if t in ("object", "array"):
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if _finite(parsed):
                return parsed
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return parsed if _finite(parsed) else value


def _coerce_value(value, schema):
    if isinstance(value, str):
        return _coerce(value, _schema_types(schema))
    if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        return {k: _coerce_value(v, schema["properties"][k])
                if isinstance(schema["properties"].get(k), dict) else v for k, v in value.items()}
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [_coerce_value(v, schema["items"]) for v in value]
    return value


def _tool_properties(tools, name):
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            return (fn.get("parameters") or {}).get("properties") or {}
    return {}


def _call_id(seed_key, index):
    """Deterministic tool-call id in vLLM's format (chatcmpl-tool-<16 hex>): the
    same request (prompt, seed) and the same answer give the same ids."""
    return "chatcmpl-tool-" + hashlib.sha256(f"{seed_key}:{index}".encode()).hexdigest()[:16]


def parse_tool_calls(text, tools, seed_key):
    """vLLM qwen3_xml, non-streaming. -> (content, [(name, arguments_json)]).

    content is the text before the first call, stripped (None if empty) when
    there are calls, the whole text otherwise. Text between </function> and
    </tool_call> and after a call is not content (vLLM defers it and never
    flushes it in a non-streaming answer); it is logged, not silently lost.
    A call cut off by the end of the output is still a call (its unterminated
    last parameter dropped); a <tool_call> with no <function= is no call."""
    calls, before, dropped = [], None, []
    state, name, args, pos = "content", "", "", 0

    def end_call():
        nonlocal name, args
        nm = name.strip()
        if nm:
            params = {m.group(1): _trim_newlines(m.group(2)) for m in _PARAM_RE.finditer(args)}
            props = _tool_properties(tools, nm)
            params = {k: _coerce_value(v, props[k]) if isinstance(props.get(k), dict) else v
                      for k, v in params.items()}
            calls.append((nm, json.dumps(params, ensure_ascii=False)))
        name, args = "", ""

    for m in _TAG_RE.finditer(text):
        chunk, tag = text[pos:m.start()], m.group()
        pos = m.end()
        if state == "name":                       # NAME> closes the name
            name, chunk = name + chunk, ""
            if ">" in name:
                name, _, args = name.partition(">")
                state = "args"
        if state == "content":
            if before is None and tag not in (TOOL_OPEN, FUNC_OPEN):
                continue                          # a stray closing tag stays text
            if before is None:
                before = text[:m.start()]
            elif not calls:                       # after a <tool_call> that held no call
                before += chunk if tag in (TOOL_OPEN, FUNC_OPEN) else chunk + tag
                if tag not in (TOOL_OPEN, FUNC_OPEN):
                    continue
            elif chunk.strip():
                dropped.append(chunk)
            if tag in (TOOL_OPEN, FUNC_OPEN):
                state = "preamble" if tag == TOOL_OPEN else "name"
            else:
                dropped.append(tag)
        elif state in ("preamble", "between"):
            if chunk.strip():
                dropped.append(chunk)
            if tag == TOOL_CLOSE:
                state = "content"
            elif tag == FUNC_OPEN:
                state = "name"
            elif tag == TOOL_OPEN:
                state = "preamble"
        elif state == "name":                     # malformed: no '>' before this tag
            if tag == FUNC_CLOSE:
                end_call()
                state = "between"
            else:
                name += tag
        elif state == "args":
            args += chunk
            if tag == FUNC_CLOSE:
                end_call()
                state = "between"
            else:
                args += tag
    rest = text[pos:]
    if before is None:
        return text, []
    if state == "name":
        name += rest
        if ">" in name:
            name, _, args = name.partition(">")
        end_call()
    elif state == "args":
        args += rest
        end_call()
    elif not calls and state == "content":
        before += rest
    elif rest.strip():
        dropped.append(rest)
    if dropped:
        log.warning("tool calls: text outside the calls not returned (as vLLM): %r",
                    "".join(dropped)[:500])
    if not calls:
        return before or None, []
    return before.strip() or None, [pb.ToolCall(id=_call_id(seed_key, i), name=n, arguments=a)
                                    for i, (n, a) in enumerate(calls)]


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


def _developer_to_system(messages):
    """vLLM (renderers/hf.py): developer -> system, then every system message merged
    into one at the front ("\n\n"-joined), since this template wants it first."""
    msgs = [dict(m, role="system") if m["role"] == "developer" else m for m in messages]
    sys_ = [m for m in msgs if m["role"] == "system"]
    if len(sys_) <= 1 and (not sys_ or msgs[0]["role"] == "system"):
        return msgs
    merged = {"role": "system", "content": "\n\n".join(m["content"] for m in sys_ if m["content"])}
    return [merged] + [m for m in msgs if m["role"] != "system"]


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
            d["tool_calls"] = json.loads(m.tool_calls_json)   # the gateway checked the shape
            for c in d["tool_calls"]:
                args = (c.get("function") or {}).get("arguments")
                if isinstance(args, str) and args:
                    try:
                        ok = isinstance(json.loads(args), dict)
                    except ValueError:
                        ok = False
                    if not ok:
                        # mlx-vlm renders it as {} (vLLM does the same, with this warning)
                        log.warning("tool call %r in the history: arguments are not a JSON "
                                    "object; rendered as {}", (c.get("function") or {}).get("name"))
        if m.HasField("reasoning"):
            # the template reads reasoning_content; vLLM passes both names
            d["reasoning"] = d["reasoning_content"] = m.reasoning
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
# QWEN_KV_BITS=0 disables KV quantization (fp16 cache). Measured on the 9B at
# 32K: fp16 KV beats q8 on BOTH prefill (72 vs 82 s) and decode (31.9 vs 27.3
# tok/s) — quantized-cache prefill dequantizes per slab and the q8 decode SDPA
# kernel is slower than the fp16 one — and it costs 32 KB/token (~4 GB @128K),
# which the 9B (11.6 GB loaded) can afford. Keep q8 for the 27B (64 KB/token
# fp16 = 8.4 GB @128K, over budget) and q4 for the 35B.
KV_BITS = int(os.environ.get("QWEN_KV_BITS", "4"))
KV_KWARGS = {"kv_bits": KV_BITS or None, "kv_group_size": 64, "quantized_kv_start": 5000}

# Chunked prefill: process a long prompt in N-token slabs instead of one giant
# matmul, so the prefill ACTIVATION peak (not the KV cache) stays bounded. A 39K
# prompt prefilled in one shot is what spiked us over the wired limit. 512 is a
# safe slab; raise via QWEN_PREFILL_STEP if you have headroom and want speed.
# Measured on the 9B (fp16): 512 -> 2048 is +15% prefill at 8K; beyond 2048 no
# gain. The slab is shrunk automatically at long context (see _run) so that
# slab * cache_offset <= QWEN_PREFILL_BUDGET (default 2048 * 32K); at 120K the
# slab is ~512.
PREFILL_STEP = int(os.environ.get("QWEN_PREFILL_STEP", "2048"))
PREFILL_MIN_STEP = int(os.environ.get("QWEN_PREFILL_MIN_STEP", "256"))
PREFILL_BUDGET = int(os.environ.get("QWEN_PREFILL_BUDGET", str(2048 * 32768)))

# Metal wired-memory ceiling. Default on a 32 GB Mac is ~21.8 GB — barely above
# the ~18 GB of 3-bit weights, leaving almost nothing for KV + prefill. Raise it
# in-process when QWEN_WIRED_LIMIT_GB is set. NOTE: going above the OS-recommended
# max also needs `sudo sysctl iogpu.wired_limit_mb=<MB>` — set_wired_limit alone
# cannot exceed what the kernel allows.
_WIRED_GB = float(os.environ.get("QWEN_WIRED_LIMIT_GB", "0") or "0")

# Sampling, as vLLM does it (the reference server for these models). Env sets
# the process-wide defaults, a request overrides each field it sets (proto
# `optional` fields; absent = env default). Order, all on float32 logits:
# penalties (presence/frequency over the generated tokens, repetition over
# prompt + generated) -> temperature -> min_p -> top_k -> top_p -> draw; each
# filter sees the distribution the previous one left. Temperature 0 = argmax
# of the penalized logits. Default: greedy with an additive presence penalty
# 1.5 (the Qwen card's *non-thinking* setting). For reasoning / benchmarks set
# QWEN_PRESENCE_PENALTY=0 and the card's sampler (thinking: temperature 1.0,
# top_p 0.95, top_k 20, min_p 0) — see README.
SAMPLING_KWARGS = {
    "temperature": float(os.environ.get("QWEN_TEMPERATURE", "0.0")),    # 0 => argmax
    "top_p": float(os.environ.get("QWEN_TOP_P", "1.0")),                # 1.0 => off
    "top_k": int(os.environ.get("QWEN_TOP_K", "0")),                    # 0 => off
    "min_p": float(os.environ.get("QWEN_MIN_P", "0.0")),                # 0 => off
    "presence_penalty": float(os.environ.get("QWEN_PRESENCE_PENALTY", "1.5")),
    "frequency_penalty": float(os.environ.get("QWEN_FREQUENCY_PENALTY", "0.0")),
    "repetition_penalty": float(os.environ.get("QWEN_REPETITION_PENALTY", "1.0")),
}


def make_sampler(sampling, seed=None):
    """logits (rows, V) -> token ids, with an explicit per-request RNG key.

    vLLM's order: temperature, then min_p, top_k and top_p, each on the
    distribution the previous one left (top_p over the top_k survivors,
    renormalized; at least one token always survives).

    The key is ours, not the global one: mlx_lm's sampler draws through a
    function compiled with ``inputs=outputs=mx.random.state``, which captures
    the global key of the thread that first traced it. MLX random state is per
    thread, so in the gRPC handler threads the captured key never advanced and
    temperature 0.7 returned byte-identical answers. Here the key is split per
    step; ``seed`` (request field) makes a request reproducible.
    """
    temp = sampling["temperature"]
    if temp <= 0:                       # vLLM: 0 = greedy, (0, 0.01) is raised to 0.01
        return lambda logits: mx.argmax(logits - mx.logsumexp(logits, axis=-1, keepdims=True),
                                        axis=-1)
    top_p, min_p, top_k = sampling["top_p"], sampling["min_p"], sampling["top_k"]
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    key = mx.random.key(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    temp = max(temp, 0.01)
    ninf = mx.array(-float("inf"), dtype=mx.float32)

    def sampler(logits):
        nonlocal key
        x = logits.astype(mx.float32) / temp
        if min_p > 0:
            p = mx.softmax(x, axis=-1)
            x = mx.where(p < p.max(axis=-1, keepdims=True) * min_p, ninf, x)
        if 0 < top_k < x.shape[-1]:
            kth = mx.topk(x, top_k, axis=-1).min(axis=-1, keepdims=True)
            x = mx.where(x < kth, ninf, x)
        if top_p < 1.0:
            idx = mx.argsort(x, axis=-1)
            xs = mx.take_along_axis(x, idx, axis=-1)
            drop = mx.cumsum(mx.softmax(xs, axis=-1), axis=-1) <= 1 - top_p
            drop[..., -1] = False
            x = mx.put_along_axis(x, idx, mx.where(drop, ninf, xs), axis=-1)
        key, sub = mx.random.split(key)
        return mx.random.categorical(x, key=sub)

    return sampler


class _Penalties:
    """vLLM's (= OpenAI's) penalties on the raw logits: presence and frequency
    are additive over the tokens generated so far; repetition is the
    multiplicative HF/vLLM one (positive logits / p, negative * p) over every
    token seen in the prompt or the output. `extra` adds per-row counts (the
    drafts before each verify row in speculative decoding)."""

    def __init__(self, sampling, prompt_ids, vocab):
        self.presence = sampling["presence_penalty"]
        self.frequency = sampling["frequency_penalty"]
        self.repetition = sampling["repetition_penalty"]
        self.active = bool(self.presence or self.frequency or self.repetition != 1.0)
        self.counts = mx.zeros((vocab,), dtype=mx.float32)
        self.prompt = None
        if self.repetition != 1.0:
            self.prompt = mx.zeros((vocab,), dtype=mx.bool_)
            self.prompt[mx.array(prompt_ids)] = True

    def add(self, toks):
        self.counts = self.counts.at[toks.reshape(-1)].add(1.0)

    def __call__(self, logits, extra=None):
        x = logits.astype(mx.float32)
        c = self.counts if extra is None else self.counts + extra
        if self.repetition != 1.0:
            seen = (c > 0) | self.prompt
            x = mx.where(seen, mx.where(x > 0, x / self.repetition, x * self.repetition), x)
        if self.frequency:
            x = x - self.frequency * c
        if self.presence:
            x = x - self.presence * (c > 0)
        return x

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

# Prompt cache (QWEN_PROMPT_CACHE=1): multi-slot prefix cache over text-only
# prompts. Each entry holds the cache state after the prefill of prompt[:-1].
# Full-attention KV can be cut to ANY prefix by slicing; the gated-delta-net
# layers keep a recurrent state that cannot be rewound, so that one is
# checkpointed during prefill (~52 MB each on the 9B: 24 layers x fp32
# 32x128x128) at every QWEN_PROMPT_CACHE_STRIDE tokens and at the point where
# the prompt diverged from the closest cached one (the likely end of a shared
# system prompt / tool block). A new prompt resumes from the deepest checkpoint
# inside its longest common prefix with any entry, so a different conversation
# over the same system prompt, an edited history or interleaved sessions all
# hit. Entries are LRU-evicted past QWEN_PROMPT_CACHE_GB (the newest always
# stays). The library's own trim is never used (broken for quantized tuples).
PROMPT_CACHE = os.environ.get("QWEN_PROMPT_CACHE", "0") == "1"
PROMPT_CACHE_GB = float(os.environ.get("QWEN_PROMPT_CACHE_GB", "3"))
PROMPT_CACHE_STRIDE = int(os.environ.get("QWEN_PROMPT_CACHE_STRIDE", "2048"))
PROMPT_CACHE_CKPTS = int(os.environ.get("QWEN_PROMPT_CACHE_CKPTS", "12"))

# Compute dtype (QWEN_FP16=1): cast every bfloat16 parameter (quant scales/
# biases, norms, embeddings, A_log...) to float16 after load. M1/M2-class GPUs
# have no native bf16: an 8-bit quantized_matmul with M>=512 rows runs ~1.7x
# slower in bf16 than fp16 (4.5 vs 7.8 TFLOPS on M1 Max), so PREFILL is ~1.5x
# faster in fp16; decode (M=1, bandwidth-bound) is unchanged. Weights stay 8/5/3
# bit; only the fp16 range (65504) is a risk. Verified on the 9B (teacher-forced,
# 3.4K tokens): KL(bf16||fp16) = 0.0003, top-1 agreement 99.4% — below the
# 8-bit weight quantization noise itself. QWEN_FP16=0 restores bf16.
FP16 = os.environ.get("QWEN_FP16", "1") == "1"

# Speculative decoding (QWEN_DRAFT=<dir>): a DFlash block-diffusion drafter
# (e.g. z-lab/Qwen3.5-9B-DFlash, trained on the base Qwen3.5-9B — it also works
# on the Ornith finetunes, ~60% token acceptance). Exact-greedy walk, so it is
# only used when the request is greedy (temperature 0, no presence penalty),
# text-only and without a thinking budget; otherwise the plain loop runs.
# Output is identical to the plain greedy loop up to fp16 noise.
DRAFT = os.environ.get("QWEN_DRAFT", "")
DRAFT_BLOCK = int(os.environ.get("QWEN_DRAFT_BLOCK", "0"))   # 0 = adaptive
# Drafter weight bits (0 = keep fp16). The target verifies every draft, so the
# drafter's precision only moves acceptance, not output: measured 4-bit ≈ 8-bit
# ≈ fp16 acceptance (~60%), and the round gets cheaper (fp16 ~100 ms → 4-bit
# ~85 ms): decode ×1.16–1.52 (fp16) → ×1.43–1.64 (4-bit).
DRAFT_BITS = int(os.environ.get("QWEN_DRAFT_BITS", "4"))
# Sampling / presence-penalty requests can take the speculative path too
# (QWEN_DRAFT_SAMPLING=1, exact "sample-match" acceptance, see _run_spec), but a
# round (draft ~15 ms + a 4..16-row verify, flat ~2.6x a 1-row step on M1: the
# 8-bit qmm is dequantization-bound from 6 to 32 rows) costs about 3 plain
# steps and acceptance drops with temperature: measured x1.1-1.2 on code and
# x0.8 on prose (accept ~1.1/round), so it is off by default. Falling back to
# plain decoding on low acceptance was tried and lost: a probe of 8 rounds
# costs ~0.5 s, and the drafter restarts cold (~130 ms rounds, lower accept).
DRAFT_SAMPLING = os.environ.get("QWEN_DRAFT_SAMPLING", "0") == "1"


def _copy_cache(caches, upto=None, states=None):
    """Fresh cache objects holding the same (immutable) arrays.

    `upto` cuts the KV to that many tokens and `states` (a recurrent-state
    checkpoint taken at the same position) replaces the ArraysCache contents.

    MLX `a[idx] = v` rebinds the Python array object in place, so two cache
    objects must not share array objects — slicing gives new ones over the same
    buffer (copy-on-write from there). ArraysCache entries are replaced whole
    by the model, so a new list is enough."""
    out = []
    states = iter(states) if states is not None else None
    for c in caches:
        if hasattr(c, "cache"):                       # ArraysCache (recurrent state)
            n = type(c)(size=len(c.cache))
            n.cache = list(next(states)) if states is not None else list(c.cache)
            out.append(n)
            continue
        end = c.offset if upto is None else upto
        if isinstance(c.keys, (tuple, list)):         # QuantizedKVCache (list after its tree_map resize)
            n = type(c)(group_size=c.group_size, bits=c.bits)
            n.keys = type(c.keys)(k[..., :end, :] for k in c.keys)
            n.values = type(c.values)(v[..., :end, :] for v in c.values)
        else:                                         # KVCache
            n = type(c)()
            n.keys = c.keys[..., :end, :]
            n.values = c.values[..., :end, :]
        n.offset = end
        out.append(n)
    return out


def _recurrent_states(caches):
    return [list(c.cache) for c in caches if hasattr(c, "cache")]


def _cache_arrays(caches):
    arrs = []
    for c in caches:
        if hasattr(c, "cache"):
            arrs += [a for a in c.cache if a is not None]
        else:
            for x in (c.keys, c.values):
                arrs += list(x) if isinstance(x, (tuple, list)) else [x]
    return arrs


def _lcp(a, b):
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n                                     # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


class _Entry:
    def __init__(self, ids, cache, ckpts):
        self.ids = ids
        self.cache = cache
        self.ckpts = ckpts                            # pos -> recurrent states, pos < len(ids)
        self.used = time.time()
        self.quantized = any(isinstance(getattr(c, "keys", None), (tuple, list)) for c in cache)
        arrs = _cache_arrays(cache)
        mx.eval(arrs)
        self.nbytes = sum(a.nbytes for a in arrs) + sum(
            a.nbytes for st in ckpts.values() for layer in st for a in layer)

    def best(self, ids, ends):
        """Deepest resumable position within the common prefix (one token is
        always left to feed). Only a slab end of the new prompt (`ends`) will
        do: resuming anywhere else would cut the rest of the prefill
        differently from a cold run. A quantized KV cut below the
        quantization start would not match a cold run either."""
        common = min(_lcp(self.ids, ids), len(ids) - 1)
        low = KV_KWARGS["quantized_kv_start"] if self.quantized else 0
        pos = [p for p in (*self.ckpts, len(self.ids)) if low <= p <= common and p in ends]
        return max(pos, default=0), common


class _PrefixCache:
    def __init__(self):
        self.entries: list[_Entry] = []

    def lookup(self, ids, ends):
        """-> (entry, resume position, longest common prefix over all entries)."""
        hit, at, split = None, 0, 0
        for e in self.entries:
            pos, common = e.best(ids, ends)
            split = max(split, common)
            if pos > at:
                hit, at = e, pos
        if hit is not None:
            hit.used = time.time()
        return hit, at, split

    def restore(self, entry, at):
        if at == len(entry.ids):
            return _copy_cache(entry.cache)
        return _copy_cache(entry.cache, upto=at, states=entry.ckpts[at])

    def store(self, ids, cache, ckpts, parent=None, at=0):
        if parent is not None:                        # inherit the shared part
            ckpts = {**{p: s for p, s in parent.ckpts.items() if p <= at}, **ckpts}
            if at == len(parent.ids) and at < len(ids):
                ckpts[at] = _recurrent_states(parent.cache)
        ckpts.pop(len(ids), None)
        while len(ckpts) > PROMPT_CACHE_CKPTS:        # thin: drop the closest pair's left one
            ps = sorted(ckpts)
            gaps = [(ps[i + 1] - (ps[i - 1] if i else 0), ps[i]) for i in range(len(ps) - 1)]
            ckpts.pop(min(gaps)[1])
        new = _Entry(ids, _copy_cache(cache), ckpts)
        # An entry that is a prefix of the new one is fully covered by it.
        self.entries = [e for e in self.entries
                        if not (len(e.ids) <= len(ids) and ids[: len(e.ids)] == e.ids
                                and (len(e.ids) in ckpts or len(e.ids) == len(ids)))]
        self.entries.append(new)
        budget = PROMPT_CACHE_GB * 1024**3
        while len(self.entries) > 1 and sum(e.nbytes for e in self.entries) > budget:
            self.entries.remove(min(self.entries[:-1], key=lambda e: e.used))
        return new


def _step(pos):
    # shrunk with context: the attention scratch scales with slab x context
    # (a 2048 slab at 120K peaked at 26.7 GB on the 9B)
    return max(PREFILL_MIN_STEP, min(PREFILL_STEP, PREFILL_BUDGET // max(pos, 1)))


def _slab(pos, left, cuts):
    """Prefill slab at absolute position `pos`, ended on the next cut."""
    n = _step(pos)
    nxt = min((c for c in cuts if c > pos), default=pos + n)
    return min(n, left, nxt - pos)


def _bounds(ids, mark):
    """Slab ends for the prefill of ids[:-1]: a message start (`mark`) at least
    PREFILL_MIN_STEP past the previous end, else a full step.

    The kernels round differently depending on the slab shape (split-K in the
    thin fp16 matmuls, SDPA over a key length that is not a multiple of the key
    block), so the same prompt cut into other slabs gives other logits, and a
    greedy answer can flip. Every end here depends only on the tokens before
    it, so a prompt resumed from a cached prefix is cut exactly as a cold one
    and the output does not depend on what the prompt cache holds."""
    out, last = [], 0
    for i in range(1, len(ids) - 1):
        if i - last >= _step(last) or (ids[i] == mark and i - last >= PREFILL_MIN_STEP):
            out.append(i)
            last = i
    return out


def _eos_ids(config):
    """Все eos_token_id из конфига модели: верхний (уже с generation_config) и text_config."""
    ids = set()
    for c in (config, config.get("text_config") or {}):
        e = c.get("eos_token_id")
        if e is not None:
            ids.update(e if isinstance(e, list) else [e])
    return ids


class _BPEDetokenizer(BPEStreamingDetokenizer):
    """BPE-детокенизатор mlx-vlm 0.6.2, но байты, не складывающиеся в UTF-8,
    заменяются на U+FFFD (как tokenizer.decode в HF и детокенизатор mlx-lm),
    а не роняют генерацию. Недописанный символ по-прежнему ждёт следующих
    байтов: слово сбрасывается только на токене, начинающемся с пробела."""

    def _flush(self):
        text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
            "utf-8", errors="replace")
        self.text += text if self.text or not self.trim_space else _remove_space(text)

    def add_token(self, token, skip_special_token_ids=()):
        if token in skip_special_token_ids:
            return
        v = self.tokenmap[token]
        if self._byte_decoder[v[0]] == 32:
            self._flush()
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        self._flush()
        self._unflushed = ""


class _StopFilter:
    """Holds back the tail of the stream that could be the start of a stop
    sequence, so a stop split across chunks is still caught and never emitted."""

    def __init__(self, stops, keep=False):
        self.stops = [x for x in stops if x]
        self.keep = keep        # include_stop_str_in_output
        self.buf = ""
        self.hit = False

    def feed(self, text):
        if not self.stops:
            return text
        self.buf += text
        # the earliest match wins, whichever stop string it is
        hits = [(j, st) for st in self.stops if (j := self.buf.find(st)) >= 0]
        if hits:
            i, st = min(hits, key=lambda h: (h[0] + len(h[1]), h[0]))   # first to complete
            self.hit = True
            out, self.buf = self.buf[:i + len(st) if self.keep else i], ""
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


# Structured output (`response_format`, constrained tool calls) is decoded
# under a grammar by llguidance: mlx-vlm 0.6.2 already depends on it (its
# server's json_schema path), it ships a Metal kernel for the bitmask, needs no
# grammar compile step (the mask is computed lazily, ~50 us per token on the 9B
# vocab) and covers JSON Schema well ($ref/$defs, anyOf, const/enum, min/max
# length and items). JSON whitespace is free, as with vLLM (xgrammar
# any_whitespace=True). QWEN_JSON_MAX_WS=N caps a run of whitespace in a
# response_format answer at N characters (inside strings too; 0 = off): a token
# may continue a run only while its leading whitespace fits under the cap.
# Not applied to tool-call grammars (raw values, \n</parameter> terminators).
JSON_MAX_WS = int(os.environ.get("QWEN_JSON_MAX_WS", "0"))


def _response_schema(fmt_json):
    """OpenAI `response_format` JSON -> the JSON schema to enforce, None for text.
    Raises ValueError for anything else."""
    if not fmt_json:
        return None
    try:
        rf = json.loads(fmt_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"response_format is not valid JSON: {e}") from None
    kind = rf.get("type") if isinstance(rf, dict) else None
    if kind == "text":
        return None
    if kind == "json_object":
        return {"type": "object"}
    if kind == "json_schema":
        js = rf.get("json_schema")
        schema = js.get("schema") if isinstance(js, dict) else None
        if not isinstance(schema, dict):
            raise ValueError("response_format.json_schema.schema must be a JSON Schema object")
        return _strict(schema)
    raise ValueError(f"unsupported response_format type {kind!r} "
                     "(supported: text, json_object, json_schema)")


def _strict(schema):
    """vLLM compiles json_schema with xgrammar's strict_mode=True: an object that does
    not say otherwise takes no properties beyond its `properties` (additionalProperties
    and unevaluatedProperties default to false). llguidance follows JSON Schema (extra
    keys allowed), so the default is written into the schema."""
    if isinstance(schema, list):
        return [_strict(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
            out[k] = {n: _strict(x) for n, x in v.items()} if isinstance(v, dict) else v
        elif k in ("const", "enum", "default", "examples", "required"):
            out[k] = v
        else:
            out[k] = _strict(v)
    t = out.get("type")
    is_obj = t == "object" or (isinstance(t, list) and "object" in t) or "properties" in out
    if is_obj and "additionalProperties" not in out and "unevaluatedProperties" not in out \
            and "patternProperties" not in out:
        out["additionalProperties"] = False
    return out


def _tool_choice(p):
    """-> ("auto" | "none" | "required" | "named", function name or None)."""
    tc = p.tool_choice
    if tc in ("", "auto", "none", "required"):
        return tc or "auto", None
    try:
        return "named", json.loads(tc)["function"]["name"]
    except (ValueError, KeyError, TypeError):
        raise ValueError(f"invalid tool_choice: {tc!r}") from None


def _lit(s):
    return json.dumps(s, ensure_ascii=False)


def _tool_call_rules(tools, tags, rules):
    """Lark rules for <tool_call> blocks, one per function: the name is fixed
    and the parameters follow its JSON schema in the qwen_xml layout xgrammar's
    `qwen_3_coder` structural tag uses (what vLLM constrains required / named /
    strict tools with): properties in schema order, required ones present, no
    others; a string value is raw text up to \n</parameter>, anything else is
    JSON under its sub-schema. A function whose parameters are not an object
    with properties takes any parameters. -> {name: rule}."""
    out = {}
    for i, t in enumerate(tools):
        fn = t.get("function") or {}
        params = fn.get("parameters")
        if fn.get("strict") is False or not isinstance(params, dict):
            params = {}
        defs = {k: params[k] for k in ("$defs", "definitions") if k in params}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else None
        body = []
        if props is None or not props:
            if params.get("additionalProperties") is not False and props is None:
                rules.append(f'any_{i}: "<parameter=" /[^>\\n]+/ ">\\n" any_v_{i}')
                rules.append(f'any_v_{i}[lazy]: /(?s:.*)\\n<\\/parameter>\\n/')
                body.append(f"any_{i}*")
        else:
            required = set(params.get("required") or [])
            for j, (k, sub) in enumerate(props.items()):
                rules.append(f"p_{i}_{j}: {_lit('<parameter=' + k + '>' + chr(10))} v_{i}_{j}")
                rules.append(_value_rule(f"v_{i}_{j}", sub, defs))
                body.append(f"p_{i}_{j}" + ("" if k in required else "?"))
        rules.append(f"call_{i}: <[{tags[0]}]> {_lit(chr(10) + '<function=' + fn.get('name', '') + '>' + chr(10))} "
                     f"{' '.join(body)} {_lit('</function>' + chr(10))} <[{tags[1]}]>")
        out[fn.get("name")] = f"call_{i}"
    return out


def _value_rule(rule, schema, defs):
    end = r"\n<\/parameter>\n"
    sch = schema if isinstance(schema, dict) else {}
    ref = sch.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/") and len(sch) == 1:
        node = {**defs}
        for part in ref[2:].split("/"):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        sch = node if isinstance(node, dict) else {}
    types = set(_schema_types(sch))
    if types == {"string"} and isinstance(sch.get("enum"), list) and sch["enum"]:
        return f"{rule}: " + " | ".join(_lit(v + "\n</parameter>\n") for v in sch["enum"])
    if types == {"string"} and isinstance(sch.get("const"), str):
        return f"{rule}: {_lit(sch['const'] + chr(10) + '</parameter>' + chr(10))}"
    if "string" in types:                          # raw text (vLLM keeps it a string)
        return f"{rule}[lazy]: /(?s:.*){end}/"
    return f'{rule}: %json {json.dumps(_strict({**schema, **defs}), ensure_ascii=False)} "\\n</parameter>\\n"'


def _grammar(answer, tools, choice, name, parallel, tags):
    """llguidance grammar for one request, None when nothing is constrained.

    - response_format only: the JSON schema.
    - tools, choice "required": a call first, then text / more calls (exactly one
      call if parallel_tool_calls is false); "named": exactly that call, nothing else
      (xgrammar's qwen_3_coder tags, as vLLM applies them).
    - tools, "auto": constrained only when a tool is `strict` (text with calls
      anywhere, as vLLM) or with a response_format: then the answer is either
      the JSON or tool calls (vLLM would apply the JSON grammar to the whole
      output and make tool calls impossible).
    - "none": only response_format, if any; tools stay in the prompt."""
    from llguidance import LLMatcher
    strict = any((t.get("function") or {}).get("strict") is True for t in tools)
    if not tools or choice == "none" or (choice == "auto" and not strict and answer is None):
        return None if answer is None else LLMatcher.grammar_from_json_schema(answer)
    if tags is None:
        raise ValueError("constrained tool calls need <tool_call> tokens in the vocabulary")
    rules = []
    calls = _tool_call_rules(tools, tags, rules)
    rules.append("call: " + " | ".join(calls.values()))
    rules.append("TEXT: /(?s:.+)/")
    many = 'call (WS? call)*' if parallel else 'call'
    rules.append("WS: /[ \\t\\r\\n]+/")
    if choice == "named":
        if name not in calls:
            raise ValueError(f"tool_choice names an unknown function: {name!r}")
        start = calls[name]
    elif choice == "required":
        # xgrammar's qwen_3_coder "required": the output opens with a call (no text
        # before it, so the model cannot talk until max_tokens), text may follow;
        # parallel_tool_calls false (stop_after_first): that one call, nothing after
        start = "call (TEXT? call)* TEXT?" if parallel else "call"
    elif answer is not None:
        rules.append(f"answer: %json {json.dumps(answer, ensure_ascii=False)}")
        start = f"answer | {many}"
    else:
        start = "TEXT? (call TEXT?)*" if parallel else "TEXT? (call)?"
    return LLMatcher.grammar_from_lark("\n".join([f"start: {start}"] + rules) + "\n")


class _Guide:
    """Grammar-constrained sampling for one request.

    `sample` masks the logits with the grammar state after the tokens
    consumed so far (the mask is computed on the CPU when the graph is built,
    ~0.2 ms per token on the 9B vocab). When generation begins inside the
    thinking block the grammar starts after </think> (or at a <tool_call> that
    ends the reasoning, as vLLM's parser treats it); until then only EOS is
    masked out, so the answer cannot be skipped. EOS is allowed by the grammar
    only where the output is complete."""

    def __init__(self, llg_tok, grammar, *, active, think_close, tool_open=None, ws=None):
        import llguidance
        import llguidance.numpy
        self.llg_tok = llg_tok
        self.matcher = llguidance.LLMatcher(llg_tok, grammar)
        if self.matcher.is_error():
            raise ValueError(self.matcher.get_error())
        self.n_vocab = llg_tok.vocab_size
        self.active = active
        self.think_close = think_close
        self.tool_open = tool_open
        self.no_eos = llguidance.numpy.allocate_token_bitmask(1, self.n_vocab)
        bits = self.no_eos.view("uint32")
        for t in llg_tok.eos_tokens:
            bits[0, t // 32] &= ~(1 << (t % 32)) & 0xFFFFFFFF
        self.ws_masks = ws      # QWEN_JSON_MAX_WS: [r] = tokens with <= r leading whitespace
        self.ws_run = 0

    def consume(self, tok):
        if not self.active:
            if tok == self.think_close:
                self.active = True
                return
            if tok != self.tool_open:
                return
            self.active = True
        if not self.matcher.consume_token(tok):
            raise RuntimeError(f"grammar: token {tok} rejected: {self.matcher.get_error()}")
        if self.ws_masks is not None:
            b = self.llg_tok.decode_bytes([tok])
            tail = len(b) - len(b.rstrip(b" \t\r\n"))
            self.ws_run = self.ws_run + len(b) if tail == len(b) else tail

    def sample(self, logits, sampler):
        import llguidance.mlx
        import llguidance.numpy
        if self.active:
            mask = llguidance.numpy.allocate_token_bitmask(1, self.n_vocab)
            llguidance.numpy.fill_next_token_bitmask(self.matcher, mask)
            if self.matcher.is_error():
                raise RuntimeError(f"grammar: {self.matcher.get_error()}")
            if self.ws_masks is not None:
                capped = mask & self.ws_masks[max(0, JSON_MAX_WS - self.ws_run)]
                if capped.any():                  # never leave the grammar no way out
                    mask = capped
        else:
            mask = self.no_eos
        return sampler(llguidance.mlx.apply_token_bitmask(logits, mask))


def _ws_masks(llg_tok, cap):
    """[r] = bitmask of the tokens that start with at most r whitespace chars
    (a whitespace-only token counts whole), r = 0..cap, for QWEN_JSON_MAX_WS:
    a token may continue a whitespace run only as far as the cap. A token's
    own trailing whitespace is not bounded (rare in practice)."""
    import llguidance.numpy
    import numpy as np
    n = 32 * ((llg_tok.vocab_size + 31) // 32)
    lead = np.zeros(n, dtype=np.int64)
    for t in range(llg_tok.vocab_size):
        b = llg_tok.decode_bytes([t])
        lead[t] = len(b) - len(b.lstrip(b" \t\r\n"))
    out = []
    for r in range(cap + 1):
        mask = llguidance.numpy.allocate_token_bitmask(1, llg_tok.vocab_size)
        mask[0] = np.packbits(lead <= r, bitorder="little").view(np.int32)
        out.append(mask)
    return out


class Engine:
    """Owns the MLX model and runs (serialized) generations."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        # the id clients see (/v1/models, "model" in responses): the weights dir name
        self.model_id = os.path.basename(os.path.normpath(model_path))
        self.ready = False
        self.load_ms = 0
        self._lock = threading.Lock()
        self._cancels: dict[str, threading.Event] = {}
        self._cancels_lock = threading.Lock()
        self._pcache = _PrefixCache()
        self.draft = None

    def load(self):
        from mlx_vlm import load
        from mlx_vlm.prompt_utils import apply_chat_template  # noqa: F401
        from mlx_vlm.utils import load_config

        log.info("loading model: %s", self.model_path)
        t = time.time()
        self.config = load_config(self.model_path)
        if "vision_config" not in self.config:
            # Text-only checkpoint of a VL architecture (e.g. Ornith-1.5-9B-MLX):
            # mlx-vlm still builds the vision tower and load_weights() is strict,
            # so let the tower keep its random init. Never used for text.
            log.warning("no vision_config: text-only checkpoint, loading weights non-strictly "
                        "(vision tower uninitialised; image requests will not work)")
            import mlx.nn as nn
            _orig = nn.Module.load_weights
            nn.Module.load_weights = lambda m, w, strict=True: _orig(m, w, strict=False)
            try:
                self.model, self.processor = load(self.model_path)
            finally:
                nn.Module.load_weights = _orig
        else:
            self.model, self.processor = load(self.model_path)
        if FP16:
            lm = self.model.language_model
            lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
            mx.eval(lm.parameters())
            log.info("language model cast to float16 (QWEN_FP16=1)")
        if DRAFT and KV_KWARGS["kv_bits"] is not None:
            # The spec loop never re-quantizes the cache and rollback trims a
            # quantized cache wrongly; the drafter is only worth it on the 9B,
            # which runs an fp16 cache anyway.
            log.warning("QWEN_DRAFT ignored: speculative decoding needs QWEN_KV_BITS=0")
        elif DRAFT:
            from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
            draft, kind = load_drafter(DRAFT)
            if kind != "dflash":
                raise RuntimeError(f"unsupported drafter kind {kind!r} (only dflash)")
            validate_drafter_compatibility(self.model, draft, kind)
            if FP16:
                draft.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
            if DRAFT_BITS:
                import mlx.nn as nn
                nn.quantize(draft, group_size=64, bits=DRAFT_BITS,
                            class_predicate=lambda _p, m: isinstance(m, nn.Linear)
                            and m.weight.shape[1] % 64 == 0)
            mx.eval(draft.parameters())
            if DRAFT_BLOCK:
                draft.prefer_requested_block_size = True
            self.draft = draft
            log.info("dflash drafter loaded: %s (%s, block %s, target layers %s)", DRAFT,
                     f"{DRAFT_BITS}-bit" if DRAFT_BITS else "fp16", DRAFT_BLOCK or "adaptive",
                     list(draft.config.target_layer_ids))
        _tok = getattr(self.processor, "tokenizer", self.processor)
        # mlx-vlm стопит только на верхнем eos_token_id; у qwen их два
        # (<|im_end|> и <|endoftext|>), стоп на всех — как в HF/vLLM
        crit = _tok.stopping_criteria
        crit.eos_token_ids = sorted(set(crit.eos_token_ids or []) | _eos_ids(self.config))
        log.info("eos token ids: %s", crit.eos_token_ids)
        self._im_start = _tok.convert_tokens_to_ids("<|im_start|>")
        tpl = getattr(self.processor, "chat_template", None) or getattr(_tok, "chat_template", "") or ""
        import jinja2
        import jinja2.meta
        self._template_vars = jinja2.meta.find_undeclared_variables(
            jinja2.Environment().parse(tpl)) - {"messages", "tools", "add_generation_prompt"}
        # vLLM: a template that never names "developer" gets it as system
        self._developer = '"developer"' in tpl or "'developer'" in tpl
        self._think = [_tok.convert_tokens_to_ids(t) for t in (THINK_OPEN, THINK_CLOSE)]
        tags = [_tok.convert_tokens_to_ids(t) for t in ("<tool_call>", "</tool_call>")]
        self._tool_tags = tags if _tok.convert_ids_to_tokens(tags) == ["<tool_call>", "</tool_call>"] \
            else None
        # response_format: llguidance's view of the vocab (~2 s to build, once).
        # Masks are as wide as the logits; EOS = every id the loop stops on.
        # logits width (the embedding is padded past the tokenizer's vocab)
        self.vocab = int((self.config.get("text_config") or {}).get("vocab_size")
                         or self.config["vocab_size"])
        try:
            import llguidance.hf
            self._llg_tok = llguidance.hf.from_tokenizer(
                _tok, n_vocab=self.vocab, eos_token=crit.eos_token_ids)
        except Exception:  # noqa: BLE001 — the rest of the worker does not need it
            log.exception("llguidance tokenizer failed; response_format requests will be rejected")
            self._llg_tok = None
        self._not_ws = None
        if JSON_MAX_WS and self._llg_tok is not None:
            self._not_ws = _ws_masks(self._llg_tok, JSON_MAX_WS)
            log.info("constrained answers: whitespace runs capped at %d chars", JSON_MAX_WS)
        d = self.processor.detokenizer
        if type(d) is BPEStreamingDetokenizer:
            self.processor.detokenizer = _BPEDetokenizer(_tok, trim_space=d.trim_space)
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

    # --- structured output --------------------------------------------------
    def grammar_for(self, p: pb.Params):
        """llguidance grammar for the request's response_format / tool calls
        (None = unconstrained). Raises ValueError when a format is unsupported
        or does not compile."""
        answer = _response_schema(p.response_format_json)
        choice, name = _tool_choice(p)
        tools = json.loads(p.tools_json) if p.tools_json else []
        parallel = not p.HasField("parallel_tool_calls") or p.parallel_tool_calls
        what = "response_format" if answer is not None else "tools"
        try:
            grammar = _grammar(answer, tools, choice, name, parallel, self._tool_tags)
        except ValueError as e:
            raise ValueError(f"{what}: {e}") from None
        if grammar is None:
            return None
        if self._llg_tok is None:
            raise ValueError(f"{what}: constrained decoding unavailable (no llguidance tokenizer)")
        from llguidance import LLMatcher
        bad, msgs = LLMatcher.validate_grammar_with_warnings(grammar, self._llg_tok)
        if bad:
            raise ValueError(f"{what}: schema does not compile: {msgs[0]}")
        for m in msgs:
            log.warning("%s: %s", what, m)
        return grammar

    def _render(self, req, images):
        """Checks that need no model, then the chat template -> (prompt, grammar).
        Shared by Check and Generate; raises _Reject."""
        from mlx_vlm.prompt_utils import apply_chat_template
        p = req.params
        if images and "vision_config" not in self.config:
            raise _Reject("invalid_request", f"{self.model_id} is a text-only checkpoint: "
                          "image inputs are not supported")
        try:
            grammar = self.grammar_for(p)
        except ValueError as e:
            raise _Reject("invalid_request", str(e)) from None
        kwargs = json.loads(p.chat_template_kwargs_json) if p.chat_template_kwargs_json else {}
        # vLLM's resolve_chat_template_kwargs: these two are refused, the rest is
        # kept only if the template uses it (the request's own fields win)
        bad = {"chat_template", "tokenize"} & kwargs.keys()
        if bad:
            raise _Reject("invalid_request",
                          f"Found unexpected chat template kwargs from request: {bad}")
        dropped = [k for k in kwargs if k not in self._template_vars]
        if dropped:
            log.info("chat_template_kwargs not used by the template, ignored (as vLLM): %s", dropped)
        kwargs = {k: v for k, v in kwargs.items() if k in self._template_vars}
        if p.tools_json:
            # also for tool_choice "none": vLLM keeps the tools in the prompt
            kwargs["tools"] = json.loads(p.tools_json)
        add_gen = not p.HasField("add_generation_prompt") or p.add_generation_prompt
        if p.continue_final_message:
            kwargs["continue_final_message"] = True
        messages = _rebuild_messages(req.messages)
        if any(m["role"] == "developer" for m in messages) and not self._developer:
            messages = _developer_to_system(messages)
        try:
            formatted = apply_chat_template(self.processor, self.config, messages,
                                            add_generation_prompt=add_gen,
                                            num_images=len(images), **kwargs)
        except Exception as e:  # noqa: BLE001 — the template's own raise_exception, bad kwargs
            raise _Reject("invalid_request", f"chat template: {e}") from None
        return formatted, grammar

    def _limit(self, p, prompt_tokens):
        """Generation limit for a prompt this long; vLLM's rules and messages."""
        if p.HasField("max_tokens"):
            if prompt_tokens + p.max_tokens > MAX_CONTEXT:
                raise _Reject("context_length_exceeded",
                              f"This model's maximum context length is {MAX_CONTEXT} tokens. "
                              f"However, you requested {p.max_tokens} output tokens and your "
                              f"prompt contains {prompt_tokens} input tokens, for a total of "
                              f"{prompt_tokens + p.max_tokens} tokens. Please reduce the length "
                              "of the input prompt or the number of requested output tokens.")
            return p.max_tokens
        remaining = MAX_CONTEXT - prompt_tokens
        if remaining < 1:
            raise _Reject("context_length_exceeded",
                          f"This model's maximum context length is {MAX_CONTEXT} tokens. "
                          f"However, your request has {prompt_tokens} input tokens. "
                          "Please reduce the length of the input messages.")
        return min(p.default_max_tokens or remaining, remaining)

    def check(self, req: pb.Request) -> str:
        """What Generate would reject before generating, without the model lock
        (text prompts are tokenized to check the length; image prompts only
        when the worker runs them)."""
        try:
            formatted, _ = self._render(req, list(req.images))
            if not req.images:
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                self._limit(req.params, len(tokenizer.encode(formatted)))
        except _Reject as e:
            return str(e), False
        except Exception as e:  # noqa: BLE001 — a worker fault, not the request's
            log.exception("check failed: %s", req.id)
            return f"{type(e).__name__}: {e}", True
        return "", False

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
             max_tokens, sampling, thinking, on_prefilled=None, start=0, cuts=(),
             on_ckpt=None, guide=None, pen=None, ctl=None):
        """Prefill + decode, yielding (token_id, text) pairs; sets self._finish.

        A trimmed-down generate_step/stream_generate from mlx-vlm 0.6.2. Kept
        local so that (a) the prompt cache snapshot can be taken at the exact
        point where the cache holds the prompt minus its last token and nothing
        else, (b) pre-tokenized vision inputs go in directly, and (c) the
        finish reason and buffer-pool clearing are under our control.

        With a `guide` (response_format) the mask for a token needs the one
        before it on the CPU, so the next step is queued only once the token is
        known (the detokenizer and the yield still overlap the forward pass).
        """
        from mlx_vlm.generate.common import (generation_stream, maybe_quantize_kv_cache,
                                             wired_limit)
        from mlx_vlm.tokenizer_utils import make_streaming_detokenizer

        greedy = sampling["temperature"] <= 0 and not pen.active
        spec = (self.draft is not None and pixel_values is None and not extra
                and thinking is None and (greedy or DRAFT_SAMPLING))
        if spec and (guide is not None or ctl.active):
            # DFlash drafts are not grammar-masked: the verify rows would each
            # need a mask over the draft prefix. Plain decoding instead.
            log.info("grammar / token options: speculative decoding off for this request")
        elif spec:
            yield from self._run_spec(input_ids, prompt_cache, max_tokens=max_tokens,
                                      sampling=sampling, pen=pen,
                                      on_prefilled=on_prefilled, start=start, cuts=cuts,
                                      on_ckpt=on_ckpt)
            return

        model = self.model
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        sampler = make_sampler(sampling, seed=sampling.get("seed"))
        quantize = functools.partial(maybe_quantize_kv_cache, **{
            "quantized_kv_start": KV_KWARGS["quantized_kv_start"],
            "kv_group_size": KV_KWARGS["kv_group_size"], "kv_bits": KV_KWARGS["kv_bits"]})
        kwargs = dict(extra)

        def step(y, inputs_embeds=None):
            nonlocal kwargs
            with mx.stream(generation_stream):
                out = model.language_model(y, inputs_embeds=inputs_embeds,
                                           cache=prompt_cache, **kwargs)
                logits = out.logits[:, -1, :]
                if pen.active:
                    if inputs_embeds is None:         # y is a generated token, not the prompt
                        pen.add(y)
                    logits = pen(logits)
                if ctl.active:
                    if inputs_embeds is None:
                        ctl.n += 1
                    logits = ctl.mask(logits)
                quantize(prompt_cache)
                kwargs = {}
                if guide is not None:
                    return guide.sample(logits, sampler), None
                return sampler(logits), None

        with wired_limit(model, [generation_stream]):
            with mx.stream(generation_stream):
                emb = model.get_input_embeddings(input_ids, pixel_values, mask=mask, **kwargs)
                inputs_embeds = emb.inputs_embeds
                kwargs.update({k: v for k, v in emb.to_dict().items()
                               if k != "inputs_embeds" and v is not None})
                # Everything but the last prompt token goes through the cache in
                # PREFILL_STEP slabs; the last token is the first decode step.
                pos = start if pixel_values is None else \
                    int(prompt_cache[model.language_model.model.fa_idx].offset)
                while inputs_embeds.shape[1] > 1:
                    n = _slab(pos, inputs_embeds.shape[1] - 1, cuts)
                    # skip_logits: no lm_head over the slab (248K vocab = ~12% of
                    # the per-token FLOPs); only the cache matters here.
                    model.language_model(input_ids[:, :n], inputs_embeds=inputs_embeds[:, :n],
                                         cache=prompt_cache, skip_logits=True, **kwargs)
                    quantize(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    pos += n
                    if pos in cuts and on_ckpt is not None:
                        on_ckpt(pos, prompt_cache)
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
            ahead = guide is None and not ctl.sync    # else the next mask needs this token
            while True:
                if n != max_tokens and ahead:
                    next_y, next_lp = step(y[None])
                    mx.async_eval(next_y, next_lp)
                if n == 0:
                    mx.eval(y)
                if n == max_tokens:
                    break
                tok = y.item()
                if thinking is not None:
                    thinking(tok)
                why = ctl.stop(tok) if ctl.active else ("stop" if tokenizer.stopping_criteria(tok)
                                                        else None)
                if why == "stop":
                    self._finish = "stop"
                    break
                if not ahead:
                    if guide is not None:
                        guide.consume(tok)
                    ctl.history.append(tok)
                    next_y, next_lp = step(y[None])
                    mx.async_eval(next_y)
                if why != "skip":                     # an ignored EOS is not text
                    detok.add_token(tok)
                yield tok, detok.last_segment
                if n % 256 == 0 and CLEAR_CACHE:
                    mx.clear_cache()
                if thinking is not None:
                    next_y = thinking.apply_forced_token(next_y)
                y, logprobs = next_y, next_lp
                n += 1
            detok.finalize()
            # не last_segment: он придерживает текст, кончающийся на U+FFFD
            seg = detok.text[detok.offset:]
            if seg:
                yield None, seg

    def _run_spec(self, input_ids, prompt_cache, *, max_tokens, sampling, pen, on_prefilled=None,
                  start=0, cuts=(), on_ckpt=None):
        """Greedy text-only prefill + DFlash speculative decode; same protocol as _run.

        Prefill goes through the inner Qwen3_5Model directly so the hidden states
        of the drafter's target layers can be captured WITHOUT flipping the model
        into its per-token "verify" kernels (LanguageModel.__call__ does that
        whenever capture_layer_ids is set). Only the last sliding_window-1 prompt
        positions are kept: that is all the drafter's rotating cache looks at.
        Each round then drafts a block (greedy), verifies it with one target
        forward, keeps the longest matching prefix and rolls the caches back.

        Sampling is exact ("sample-match"): every verify row is sampled from the
        target distribution for that position - the penalties over the output
        plus the drafts before the row, then temperature/min-p/top-k/top-p -
        and drafts are kept while they equal the target's own sample. Each
        emitted token is therefore a target sample given an accepted prefix,
        i.e. the same distribution as the plain loop; temperature only lowers
        the acceptance rate.
        """
        from mlx_vlm.generate.common import generation_stream, wired_limit
        from mlx_vlm.speculative.common import (_dflash_block_total, _record_speculative_round,
                                                _speculative_walk)
        from mlx_vlm.speculative.dflash import _dflash_next_block_size
        from mlx_vlm.tokenizer_utils import make_streaming_detokenizer

        model = self.model
        lm = model.language_model
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        draft = self.draft
        layers = list(draft.config.target_layer_ids)
        win = draft.config.sliding_window - 1
        argmax = lambda logits: mx.argmax(logits, axis=-1)
        sampler = make_sampler(sampling, seed=sampling.get("seed"))
        dtype = input_ids.dtype

        def pick(logits, fed):
            """Target tokens for every row of `logits` (1, T, V); row i follows
            the output so far plus the drafts fed[1:i+1]."""
            logits = logits[0]
            if pen.active:
                extra = None
                if fed.shape[0] > 1:
                    hot = (fed[1:, None] == mx.arange(logits.shape[-1])[None]).astype(mx.float32)
                    extra = mx.concatenate([mx.zeros((1, logits.shape[-1])),
                                            mx.cumsum(hot, axis=0)])
                logits = pen(logits, extra)
            if sampling["temperature"] <= 0:
                return mx.argmax(logits, axis=-1)[None]
            return sampler(logits)[None]

        with wired_limit(model, [generation_stream]):
            with mx.stream(generation_stream):
                lm._rope_deltas = None
                ctx = input_ids[:, :-1]
                hidden = []
                pos = start
                while ctx.shape[1] > 0:
                    n = _slab(pos, ctx.shape[1], cuts)
                    sink = []
                    lm.model(ctx[:, :n], cache=prompt_cache, capture_layer_ids=layers,
                             hidden_sink=sink, gdn_sink=None)
                    h = mx.concatenate(sink, axis=-1)
                    hidden.append(h)
                    while sum(t.shape[1] for t in hidden[:-1]) >= win:
                        hidden.pop(0)
                    mx.eval(h, [c.state for c in prompt_cache])
                    pos += n
                    if pos in cuts and on_ckpt is not None:
                        on_ckpt(pos, prompt_cache)
                    ctx = ctx[:, n:]
                    if CLEAR_CACHE:
                        mx.clear_cache()
                if on_prefilled is not None:
                    on_prefilled(prompt_cache)
                out = lm(input_ids[:, -1:], cache=prompt_cache, capture_layer_ids=layers)
                hidden.append(mx.concatenate(out.hidden_states, axis=-1))
                hidden = mx.concatenate(hidden, axis=1)[:, -win:]
                first = pick(out.logits[:, -1:, :], input_ids[0, -1:])
                mx.eval(first, hidden)
            first = first.item()

            detok = make_streaming_detokenizer(self.processor)
            self._finish = "length"
            n = 0

            def emit(tok):
                if tokenizer.stopping_criteria(tok):
                    self._finish = "stop"
                    return False
                if pen.active:
                    pen.add(mx.array([tok]))
                detok.add_token(tok)
                return True

            if max_tokens > 0:
                ok = emit(first)
                if ok:
                    yield first, detok.last_segment
                    n = 1
                    # mlx-vlm 0.6.2 `_dflash_rounds`, with `pick` as the target sampler.
                    block_total = _dflash_block_total(draft, DRAFT_BLOCK or None)
                    draft_cache = draft.reset(model)
                    b, done = first, False
                    acc, drafted = 0, 0
                    while n < max_tokens and not done:
                        bs = _dflash_next_block_size(draft, block_total, max_tokens - n + 1)
                        if bs <= 1:
                            break
                        drafts = draft.draft_block(b, hidden, draft_cache, bs, argmax, dtype)
                        mx.async_eval(drafts)
                        with mx.stream(generation_stream):
                            fed = mx.concatenate([mx.array([[b]], dtype=dtype), drafts], axis=1)
                            out = lm(fed, cache=prompt_cache, capture_layer_ids=layers)
                            hidden = mx.concatenate(out.hidden_states, axis=-1)
                            targets = pick(out.logits, fed[0])
                        mx.async_eval(targets, hidden)
                        accepted, new = _speculative_walk(drafts, targets, max_tokens - n)
                        _record_speculative_round(draft, accepted, bs - 1)
                        acc, drafted = acc + accepted, drafted + bs - 1
                        for tok in new:
                            if not emit(tok):
                                done = True
                                break
                            yield tok, detok.last_segment
                            n += 1
                        if not new:
                            break
                        b = new[-1]
                        if accepted < bs - 1:
                            hidden = hidden[:, : accepted + 1, :]
                            with mx.stream(generation_stream):
                                lm.rollback_speculative_cache(prompt_cache, out.gdn_states,
                                                              accepted, bs)
                        if n % 256 < len(new) and CLEAR_CACHE:
                            mx.clear_cache()
            if max_tokens > 1:
                log.info("dflash: %d tokens, accept rate %.0f%% (%d/%d drafts)",
                         n, 100.0 * acc / max(1, drafted), acc, drafted)
            detok.finalize()
            seg = detok.text[detok.offset:]
            if seg:
                yield None, seg

    # --- generation -------------------------------------------------------
    def generate(self, req: pb.Request):
        """Yield pb.Event. Serialized across requests via self._lock."""
        p = req.params
        cancel_ev = self._register(req.id)
        tools = json.loads(p.tools_json) if p.tools_json else []
        tmp_files = []
        try:
            with self._lock:
                yield pb.Event(id=req.id, start=pb.Start(role="assistant", model=self.model_id))
                images, tmp_files = materialize_images(list(req.images))
                # mlx-vlm's apply_chat_template takes the FULL message list (system +
                # history + tool messages), forwards `tools` and the template kwargs
                # to the Jinja template, AND inserts image tokens for num_images.
                try:
                    formatted, grammar = self._render(req, images)
                except _Reject as e:
                    yield pb.Event(id=req.id, error=pb.Error(kind=e.kind, msg=str(e)))
                    return

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
                try:
                    max_tokens = self._limit(p, prompt_tokens)
                except _Reject as e:
                    yield pb.Event(id=req.id, error=pb.Error(kind=e.kind, msg=str(e)))
                    return

                # Thinking = the prompt leaves <think> open (the template does that
                # unless enable_thinking is false).
                inside = next((t == self._think[0] for t in reversed(ids)
                               if t in self._think), False)
                state = _GenState(inside)
                state.counting = inside
                t0 = time.time()
                ttft_ms = 0
                mx.reset_peak_memory()

                # Reasoning budget: forces \n</think> into the token stream when
                # exceeded (the KV cache stays consistent). Same criteria object
                # mlx-vlm's stream_generate builds.
                thinking = None
                if inside and p.HasField("thinking_budget"):
                    from mlx_vlm.utils import ThinkingBudgetCriteria
                    thinking = ThinkingBudgetCriteria(
                        tokenizer=getattr(self.processor, "tokenizer", self.processor),
                        thinking_budget=p.thinking_budget, thinking_end_token=THINK_CLOSE,
                        thinking_start_token=THINK_OPEN, enable_thinking=True)

                sampling = dict(SAMPLING_KWARGS)
                for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                          "frequency_penalty", "repetition_penalty", "seed"):
                    if p.HasField(f):
                        sampling[f] = getattr(p, f)
                pen = _Penalties(sampling, ids, self.vocab)
                tok_ = getattr(self.processor, "tokenizer", self.processor)
                ctl = _Control(p, tok_.stopping_criteria.eos_token_ids, tok_, self.vocab)

                # Prompt cache: resume from the deepest checkpoint shared with a
                # cached prompt. Text-only conversations only.
                from mlx_vlm.models.cache import make_prompt_cache
                lm = self.model.language_model
                use_cache = PROMPT_CACHE and not images
                bounds = [] if images else _bounds(ids, self._im_start)
                ends = set(bounds) | {prompt_tokens - 1}
                parent, reused, split = (self._pcache.lookup(ids, ends) if use_cache
                                         else (None, 0, 0))
                if parent is not None:
                    prompt_cache = self._pcache.restore(parent, reused)
                    input_ids = mx.array([ids[reused:]])
                    # Text-only mRoPE continues from the cache offset with a zero
                    # delta; a preceding vision request leaves its own delta here.
                    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
                    lm._position_ids = None
                    log.info("id=%s prompt cache HIT: %d/%d tokens reused (common prefix %d), "
                             "prefilling %d", req.id, reused, prompt_tokens, split,
                             prompt_tokens - reused)
                else:
                    prompt_cache = make_prompt_cache(lm)
                    if use_cache:
                        log.info("id=%s prompt cache MISS (common prefix %d, %d entries)",
                                 req.id, split, len(self._pcache.entries))
                keep, ckpts = set(), {}
                if use_cache:
                    end = prompt_tokens - 1
                    want = set(range(PROMPT_CACHE_STRIDE, end, PROMPT_CACHE_STRIDE))
                    want.add(split)
                    # Message boundaries are where prompts naturally diverge: the
                    # end of the system/tools block (another conversation), the
                    # last two messages (regenerate / edited turn) and the reply
                    # (the next turn appends to it).
                    starts = [i for i, t in enumerate(ids) if t == self._im_start]
                    want.update(starts[1:2] + starts[-3:])
                    # checkpoints only at slab ends: the closest one at or below
                    for c in want:
                        k = bisect.bisect_right(bounds, c)
                        if k and reused < bounds[k - 1] < end:
                            keep.add(bounds[k - 1])

                def checkpoint(pos, cache):
                    if pos in keep:
                        ckpts[pos] = _recurrent_states(cache)

                def snapshot(cache):
                    if not use_cache:
                        return
                    if any(getattr(c, "keys", 1) is None for c in cache):
                        return
                    e = self._pcache.store(ids[:-1], cache, ckpts, parent, reused)
                    log.info("id=%s prompt cache store: %d tokens, %d checkpoints, %.1f MB "
                             "(%d entries, %.1f MB total)", req.id, len(e.ids), len(e.ckpts),
                             e.nbytes / 1e6, len(self._pcache.entries),
                             sum(x.nbytes for x in self._pcache.entries) / 1e6)

                guide = None
                if grammar is not None:
                    guide = _Guide(self._llg_tok, grammar, think_close=self._think[1],
                                   active=not inside, ws=None if tools else self._not_ws,
                                   tool_open=self._tool_tags[0] if tools and self._tool_tags else None)
                    log.info("id=%s constrained: response_format=%s tool_choice=%s%s", req.id,
                             json.loads(p.response_format_json).get("type")
                             if p.response_format_json else "-", p.tool_choice or "auto",
                             "" if guide.active else " (after </think>)")

                stop = _StopFilter(list(p.stop), keep=p.include_stop_str_in_output)
                answer_buf = []   # tools mode: buffer answer tokens, parse at the end
                cut = []          # no tools: a <tool_call> in the answer was seen

                def split(text):
                    think_part, ans_part = state.feed(text)
                    if think_part:
                        yield pb.Event(id=req.id, token=pb.Token(text=think_part, thinking=True))
                    if ans_part and not tools and not cut and TOOL_OPEN in ans_part:
                        # vLLM's parser runs without tools too: the call and all text
                        # after it leave the answer (tool_calls are not returned)
                        ans_part, _, rest = ans_part.partition(TOOL_OPEN)
                        cut.append(TOOL_OPEN + rest)
                    elif cut:
                        cut.append(ans_part)
                        ans_part = ""
                    if ans_part:
                        if tools:
                            # Don't stream raw answer (may contain <tool_call> XML); buffer it.
                            answer_buf.append(ans_part)
                        else:
                            yield pb.Event(id=req.id, token=pb.Token(text=ans_part, thinking=False))

                gen = self._run(input_ids, pixel_values, mask, extra, prompt_cache,
                                max_tokens=max_tokens, sampling=sampling,
                                thinking=thinking, on_prefilled=snapshot, start=reused,
                                cuts=set(bounds), on_ckpt=checkpoint, guide=guide, pen=pen,
                                ctl=ctl)
                for tok, text in gen:
                    if cancel_ev.is_set():
                        gen.close()
                        yield _done(req.id, "cancelled", state, t0, ttft_ms, prompt_tokens)
                        return
                    if ttft_ms == 0:
                        ttft_ms = int((time.time() - t0) * 1000)
                    if tok is not None:
                        state.token(tok, self._think[1], self._tool_tags[0] if self._tool_tags else -1)
                    # vLLM: stop strings apply to the whole output, reasoning included,
                    # and only once min_tokens are out
                    yield from split(stop.feed(text) if state.gen_tokens > p.min_tokens else text)
                    if stop.hit:
                        gen.close()
                        break
                finish = "stop" if stop.hit else self._finish
                if not stop.hit:
                    yield from split(stop.flush())

                gen_s = time.time() - t0 - ttft_ms / 1000
                tps = state.gen_tokens / gen_s if gen_s > 0 else 0.0
                log.info("id=%s finish=%s gen_tokens=%d ttft_ms=%d tps=%.1f cached=%d "
                         "peak=%.2fGB active=%.2fGB pool=%.2fGB",
                         req.id, finish, state.gen_tokens, ttft_ms, tps, reused,
                         mx.get_peak_memory() / 1e9, mx.get_active_memory() / 1e9,
                         mx.get_cache_memory() / 1e9)

                if cut:
                    log.info("id=%s no tools: <tool_call> in the answer, it and %d chars after "
                             "it not returned (as vLLM)", req.id, len("".join(cut)))
                calls = []
                if tools:
                    answer = "".join(answer_buf)
                    key = hashlib.sha256(f"{ids}|{sampling.get('seed')}|{answer}".encode()).hexdigest()
                    content, calls = parse_tool_calls(answer, tools, key)
                    choice, _ = _tool_choice(p)
                    if calls and choice == "none":
                        log.info("id=%s tool_choice none: %d tool call(s) not returned",
                                 req.id, len(calls))
                        calls = []
                    if len(calls) > 1 and p.HasField("parallel_tool_calls") and not p.parallel_tool_calls:
                        log.info("id=%s parallel_tool_calls false: %d calls, first kept",
                                 req.id, len(calls))
                        calls = calls[:1]
                    if content:
                        yield pb.Event(id=req.id, token=pb.Token(text=content, thinking=False))
                    # vLLM: auto -> "tool_calls" whenever a call was parsed (even at the
                    # length limit); required -> only instead of "stop"; named keeps it
                    if calls and (choice == "auto" or (choice == "required" and finish == "stop")):
                        finish = "tool_calls"
                yield _done(req.id, finish, state, t0, ttft_ms, prompt_tokens, calls, tps)
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


class _Control:
    """vLLM's token-level request options: allowed_token_ids, bad_words, min_tokens
    (EOS and stop_token_ids masked until then), stop_token_ids, ignore_eos.
    `sync`: a multi-token bad word needs the tokens so far when the mask is built,
    so the decode loop does not step ahead (as with a grammar)."""

    def __init__(self, p, eos, tokenizer, vocab):
        self.eos = set(eos)
        self.stop_ids = set(p.stop_token_ids)
        self.ignore_eos = p.ignore_eos
        self.min_tokens = p.min_tokens
        self.vocab = vocab
        self.n = 0
        self.history = []
        base = None
        if p.allowed_token_ids:
            base = mx.zeros((vocab,), dtype=mx.bool_)
            base[mx.array(list(p.allowed_token_ids))] = True
        self.seqs = []                         # (prefix, last) of multi-token bad words
        single = set()
        for w in p.bad_words:                  # vLLM: the word and " word"
            for v in {w, " " + w.lstrip()}:
                ids = tokenizer.encode(v, add_special_tokens=False)
                if len(ids) == 1:
                    single.add(ids[0])
                elif ids:
                    self.seqs.append((ids[:-1], ids[-1]))
        if single:
            if base is None:
                base = mx.ones((vocab,), dtype=mx.bool_)
            base[mx.array(sorted(single))] = False
        self.base = base
        self.early = None
        if self.min_tokens:
            self.early = mx.ones((vocab,), dtype=mx.bool_)
            self.early[mx.array(sorted(self.eos | self.stop_ids))] = False
        self.sync = bool(self.seqs)
        self.active = base is not None or self.early is not None or self.sync \
            or bool(self.stop_ids) or self.ignore_eos

    def mask(self, logits):
        """Token n's logits (n = tokens generated before it)."""
        ok = self.base
        if self.early is not None and self.n < self.min_tokens:
            ok = self.early if ok is None else ok & self.early
        if self.seqs:
            ban = [last for pre, last in self.seqs
                   if len(self.history) >= len(pre) and self.history[-len(pre):] == pre]
            if ban:
                m = mx.ones((self.vocab,), dtype=mx.bool_)
                m[mx.array(ban)] = False
                ok = m if ok is None else ok & m
        if ok is None:
            return logits
        return mx.where(ok, logits, mx.array(-float("inf"), dtype=logits.dtype))

    def stop(self, tok):
        """-> "stop" (end, token not shown), "skip" (an ignored EOS: go on, not shown)
        or None."""
        if tok in self.stop_ids:
            return "stop"
        if tok in self.eos:
            return "skip" if self.ignore_eos else "stop"
        return None


class _Reject(Exception):
    """A request Generate refuses before generating (-> HTTP 400)."""

    def __init__(self, kind, msg):
        super().__init__(msg)
        self.kind = kind


class _GenState:
    """Splits the output into reasoning and answer as vLLM's `qwen3` parser does
    (vllm/parser/qwen3.py, v0.30, one grammar for both modes).

    Thinking on (the template pre-opens <think>): the output starts as reasoning;
    a <think> there is swallowed, </think> ends it, and so does <tool_call> (the
    tag goes to the answer). In the answer — from the start when thinking is off —
    a </think> is dropped and a <think> is plain text. No </think> at all =
    everything is reasoning. Tags are single tokens: a chunk holds a whole tag.
    """

    def __init__(self, thinking: bool):
        self.in_thinking = thinking
        self.gen_tokens = 0
        self.reasoning_tokens = 0

    def feed(self, text: str) -> tuple[str, str]:
        """Split a chunk into (thinking_part, answer_part)."""
        think = []
        while self.in_thinking and text:
            ends = [(text.find(t), t) for t in (THINK_OPEN, THINK_CLOSE, TOOL_OPEN)]
            i, tag = min(((i, t) for i, t in ends if i >= 0), default=(-1, ""))
            if i < 0:
                think.append(text)
                text = ""
                break
            think.append(text[:i])
            if tag == THINK_OPEN:
                text = text[i + len(tag):]
            elif tag == THINK_CLOSE:
                self.in_thinking = False
                text = text[i + len(tag):]
            else:                                 # <tool_call>: answer from here on
                self.in_thinking = False
                text = text[i:]
        return "".join(think), text.replace(THINK_CLOSE, "")

    def token(self, tok, think_close, tool_open):
        """Count a generated token: reasoning until the token that ends it."""
        self.gen_tokens += 1
        if self.counting:
            if tok in (think_close, tool_open):
                self.counting = False
            else:
                self.reasoning_tokens += 1


def _done(req_id, finish, state, t0, ttft_ms, prompt_tokens, tool_calls=None, tps=0.0):
    usage = pb.Usage(
        prompt_tokens=prompt_tokens,
        gen_tokens=state.gen_tokens,
        reasoning_tokens=state.reasoning_tokens,
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
        p = request.params
        log.info("Generate id=%s images=%d max_tokens=%s template_kwargs=%s budget=%s",
                 request.id, len(request.images),
                 p.max_tokens if p.HasField("max_tokens") else "-",
                 p.chat_template_kwargs_json or "-",
                 p.thinking_budget if p.HasField("thinking_budget") else "-")
        yield from self.engine.generate(request)

    def Cancel(self, request, context):
        self.engine.cancel(request.id)
        return pb.Ack()

    def Health(self, request, context):
        return pb.HealthReply(ready=self.engine.ready,
                              model=self.engine.model_id,
                              load_ms=self.engine.load_ms,
                              model_path=self.engine.model_path)

    def Check(self, request, context):
        # Runs beside a generation (own pool thread): template, tokenizer and
        # grammar checks, nothing that touches MLX.
        error, internal = self.engine.check(request)
        return pb.CheckReply(error=error, internal=internal, model=self.engine.model_id)

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

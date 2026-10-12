#!/usr/bin/env python3
"""Thin gRPC MLX worker for the Qwen3.x-family VL models (mlx-vlm).

Loads the model once, then serves the `Worker` service from worker.proto:
  - Generate: server-streaming Start/Token/Done/Error
  - Cancel:   cooperative stop of an in-flight generation
  - Health:   ready / model / load time
  - Shutdown: graceful stop

Generations are batched: one scheduler thread owns MLX and decodes every
running request in one forward per step (continuous batching, QWEN_MAX_BATCH);
the gRPC handler threads only render, tokenize and turn token ids into events.

Run:
  uv run --group proto bash gen_proto.sh   # once, to codegen
  uv run python worker.py                   # serve on 127.0.0.1:50051
"""
import argparse
import base64
import bisect
import collections
import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
import queue
import re
import signal
import sys
import tempfile
import threading
import time
from concurrent import futures

import grpc
import mlx.core as mx
import numpy as np
from mlx_vlm.tokenizer_utils import (BPEStreamingDetokenizer, _remove_space,
                                     make_streaming_detokenizer)

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

       Returns (paths, temp_files) — temp_files must be unlinked by the caller;
       on an error (bad base64) the files written so far are removed here."""
    out, tmp = [], []
    try:
        for img in images:
            if img.startswith("data:"):
                header, _, b64 = img.partition(",")
                ext = "png" if "png" in header.lower() else "jpg"
                fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix="qwen_img_")
                tmp.append(path)
                with os.fdopen(fd, "wb") as f:
                    f.write(base64.b64decode(b64))
                out.append(path)
            elif img.startswith("file://"):
                out.append(img[len("file://"):])
            else:
                out.append(img)
    except BaseException:
        for f in tmp:
            try:
                os.unlink(f)
            except OSError:
                pass
        raise
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
#     the balanced match for 3-bit weights; ~5.6 KiB/token (10 full-attention
#     layers x 2 KV heads x head_dim 256), ~0.7 GiB at 128K.
#   - 9B (8-bit weights, ~10 GB, tons of headroom): fp16 KV (QWEN_KV_BITS=0, see
#     below); q8 would cost ~17 KiB/token, ~2.1 GiB at 128K.
#   - 27B (5-bit weights, ~18 GB): kv_bits=8 — ~34 KiB/token (16 full-attention
#     layers x 4 KV heads x head_dim 256), ~4.25 GiB at 128K; fits with the wired bump.
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
# gain. The slab is shrunk automatically at long context (see _step) so that
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

    The result is a `_Sampler`: called on one row's logits it is the per-row
    sampler; `sample_rows` draws for several rows with equal settings at once.
    """
    return _Sampler(sampling, seed)


class _Sampler:
    """See make_sampler. `group`: rows whose samplers have equal groups can be
    filtered together (`sample_rows`); None for nothing to share."""

    def __init__(self, sampling, seed=None):
        temp = sampling["temperature"]
        self.greedy = temp <= 0             # vLLM: 0 = greedy, (0, 0.01) is raised to 0.01
        if self.greedy:
            self.group = ("greedy",)
            return
        self.top_p, self.min_p, self.top_k = sampling["top_p"], sampling["min_p"], sampling["top_k"]
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")
        self.key = mx.random.key(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
        self.temp = max(temp, 0.01)
        self.group = (self.temp, self.min_p, self.top_k, self.top_p)

    def filtered(self, logits):
        """temperature, min_p, top_k, top_p on float32 logits (rows, V): row
        ops only, so a stack of rows gives each row what it alone would."""
        ninf = mx.array(-float("inf"), dtype=mx.float32)
        x = logits.astype(mx.float32) / self.temp
        if self.min_p > 0:
            p = mx.softmax(x, axis=-1)
            x = mx.where(p < p.max(axis=-1, keepdims=True) * self.min_p, ninf, x)
        if 0 < self.top_k < x.shape[-1]:
            kth = mx.topk(x, self.top_k, axis=-1).min(axis=-1, keepdims=True)
            x = mx.where(x < kth, ninf, x)
        if self.top_p < 1.0:
            idx = mx.argsort(x, axis=-1)
            xs = mx.take_along_axis(x, idx, axis=-1)
            drop = mx.cumsum(mx.softmax(xs, axis=-1), axis=-1) <= 1 - self.top_p
            drop[..., -1] = False
            x = mx.put_along_axis(x, idx, mx.where(drop, ninf, xs), axis=-1)
        return x

    def draw(self, x):
        self.key, sub = mx.random.split(self.key)
        return mx.random.categorical(x, key=sub)

    def __call__(self, logits):
        if self.greedy:
            return mx.argmax(logits - mx.logsumexp(logits, axis=-1, keepdims=True), axis=-1)
        return self.draw(self.filtered(logits))


def sample_rows(samplers, logits, on_error=None):
    """Next tokens of several rows -> list of (1,) arrays, each equal to what
    `samplers[i](logits[i])` returns. Rows with the same settings and logits
    dtype are filtered as one (G, V) stack: MLX's row ops (topk, argsort,
    softmax, cumsum, logsumexp, argmax) give each row of a stack bit-for-bit
    the result of that row alone (checked on Metal at the real vocabulary,
    ties included: tests/test_mrqmm.py), and the draw stays per row with the
    row's own key. At B=16 the card sampler (top_k 20, top_p 0.95) costs ~15 ms
    of GPU per step row by row, mostly 16 full-vocab topk + argsort launches.

    An exception while building a group's draw fails that group only: with
    `on_error`, on_error(i, e) is called for each of its rows and their
    entries stay None (the other groups are drawn); without, it is raised."""
    out = [None] * len(samplers)
    groups = {}
    for i, (s, x) in enumerate(zip(samplers, logits)):
        groups.setdefault((s.group, x.dtype), []).append(i)
    for idx in groups.values():
        try:
            ys = _sample_group([samplers[i] for i in idx], [logits[i] for i in idx])
        except Exception as e:  # noqa: BLE001 — this group's rows only
            if on_error is None:
                raise
            for i in idx:
                on_error(i, e)
            continue
        for i, y in zip(idx, ys):
            out[i] = y
    return out


def _sample_group(samplers, logits):
    """sample_rows for rows of one group (equal settings and dtype)."""
    s0 = samplers[0]
    if len(samplers) == 1:
        return [s0(logits[0])]
    stack = mx.concatenate(logits)
    if s0.greedy:
        toks = s0(stack)
        return [toks[j:j + 1] for j in range(len(samplers))]
    x = s0.filtered(stack)
    return [s.draw(x[j:j + 1]) for j, s in enumerate(samplers)]


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
# (QWEN_DRAFT_SAMPLING=1, exact "sample-match" acceptance, see _Scheduler._spec), but a
# round (draft ~15 ms + a 4..16-row verify, flat ~2.6x a 1-row step on M1: the
# 8-bit qmm is dequantization-bound from 6 to 32 rows) costs about 3 plain
# steps and acceptance drops with temperature: measured x1.1-1.2 on code and
# x0.8 on prose (accept ~1.1/round), so it is off by default. Falling back to
# plain decoding on low acceptance was tried and lost: a probe of 8 rounds
# costs ~0.5 s, and the drafter restarts cold (~130 ms rounds, lower accept).
DRAFT_SAMPLING = os.environ.get("QWEN_DRAFT_SAMPLING", "0") == "1"

# Multi-row 8-bit matmul for batched decode (QWEN_MRQMM, default 1, read by
# mrqmm.install at load; 0 = always quantized_matmul): MLX runs
# a per-row qmv below ~6 rows and a qmm flat at x2.6 a 1-row step above; the
# custom kernel reads each 8-bit weight once for up to 16 rows (simdgroup MMA,
# fp32 accumulation). Only nn.QuantizedLinear modules that are 8-bit affine
# g64 with fp16 scales (QWEN_FP16=1) are swapped, and a call takes it only
# when decode-shaped: (rows, 1, K) with 3 <= rows <= 16. B <= 2 stays on qmv
# (bit-identical to solo); prefill slabs, the DFlash verify / drafter (always
# (1, T, K)) and B > 16 stay on quantized_matmul. 9B decode step (M1 Max, ctx
# 1K, same process A/B): 56 / 86 / 93 -> 31 / 34 / 52 ms at B = 4 / 8 / 16.

# Prefill matmul (QWEN_PFQMM, default 1, read by pfqmm.install at load; 0 =
# always quantized_matmul): on M1/M2 MLX's qmm_t uses 32x32 output tiles and
# reaches ~8.1 TFLOPS on prefill slabs (fp16 GEMM: ~9.5). pfqmm runs the same
# kernel code (the wheel's headers) with 64-row tiles: x1.03-1.28 per matmul
# for 3..8-bit g64 (8-bit x1.10-1.14), bit-identical to quantized_matmul (same
# per-element K order), and only where MLX itself would run qmm_t (>= 128
# rows, not split-K), self-tested per (bits, group) at load. 9B
# 2048-token slab, in-process A/B: x1.12 at offset 0, x1.17 at 8K/32K.


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


_FIXED_SLAB = None    # frozen at load (Engine.load); None = derived from the knobs (tests)


def _fixed_slab():
    """fp16 KV + fp16 queries + the fused SDPA (probed at load): slabs past
    SDPA_FUSED_MIN_KV keys hold no score matrix."""
    return KV_KWARGS["kv_bits"] is None and SDPA_FUSED and FP16


def _step(pos):
    """Slab size at cache offset `pos`. Shrunk with context where the attention
    materializes its scores (slab x context x heads: a 2048 slab at 120K
    peaked at 26.7 GB on the 9B), so that slab x pos <= PREFILL_BUDGET: the
    quantized-KV SDPA, and the default head_dim-256 SDPA below
    SDPA_FUSED_MIN_KV keys. With the fused kernel (fp16 KV, QWEN_SDPA_FUSED)
    a slab that reaches the threshold (pos + PREFILL_STEP >= 32K keys) takes
    it and holds no score matrix, so it stays PREFILL_STEP at any context;
    below it the budget rule still holds (at the defaults, 2048 and 2048 x
    32K, it never shrinks there). 9B, slab 2048 vs the old shrunk slab (546
    at 120K): x1.28 per token at 120K. The policy is frozen at load
    (_FIXED_SLAB): _bounds must cut a prompt the same way for the process's
    life (prompt-cache hits vs cold runs), whatever happens to SDPA_FUSED."""
    fixed = _FIXED_SLAB if _FIXED_SLAB is not None else _fixed_slab()
    if fixed and pos + PREFILL_STEP >= SDPA_FUSED_MIN_KV:
        return max(PREFILL_MIN_STEP, PREFILL_STEP)
    return max(PREFILL_MIN_STEP, min(PREFILL_STEP, PREFILL_BUDGET // max(pos, 1)))


def _prefill_reserve(heads):
    """Bytes kept outside the batch budget for a prefill slab's activations:
    the unfused attention scores (PREFILL_BUDGET x q heads x 2 B; with the
    fixed fp16 slab, at least PREFILL_STEP x SDPA_FUSED_MIN_KV — a short slab
    just under the fused threshold) + ~0.4 GiB of MLP."""
    fixed = _FIXED_SLAB if _FIXED_SLAB is not None else _fixed_slab()
    scores = max(PREFILL_BUDGET, PREFILL_STEP * SDPA_FUSED_MIN_KV) if fixed else PREFILL_BUDGET
    return scores * heads * 2 + (400 << 20)


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


# --- continuous batching -----------------------------------------------------
# Every running request decodes in ONE forward of shape (B, 1) per step: the
# weights are read once for B tokens, and decode is bandwidth-bound (the 8-bit
# qmm costs x1.17 at 2 rows, x1.97 at 4 and a flat x2.6 from 6 to 32 rows on
# M1). Each row keeps exactly the cache objects a lone request has (a KVCache /
# QuantizedKVCache per full-attention layer, prefilled by the slab loop), so
# quantized KV, its 5000-token start and the prompt cache work unchanged: only
# the cache update + SDPA run per row (RowsKV, _rows_attention). The
# gated-delta-net state of all rows is stacked on axis 0 in one ArraysCache per
# linear layer. Positions go in explicitly per row (KV length + mRoPE delta):
# no decode step reads the model's _rope_deltas / _position_ids, which only
# prefill writes. A row decoding alone runs the plain library call on its own
# caches (explicit positions are bit-identical to the computed ones), so B=1 is
# the pre-batching output; at B>1 a row differs from its solo run only by the
# kernels' row-count-dependent rounding (B=2 is still bit-identical: qmv per
# row; 3..16 rows take mrqmm on 8-bit modules (QWEN_MRQMM), qmv/qmm on the
# others; MoE sorted gather at B*top_k >= 64) — the same class of noise as the
# DFlash verify.
# mlx-vlm's own batch path is not used: its quantized batch cache ignores left
# padding at decode (wrong logits with q8/q4 KV), its ragged SDPA kernel asks
# M1 for a 1024-thread threadgroup, and it has no hook for the prompt cache.
#
# QWEN_MAX_BATCH (16): rows decoded together; 1 = one generation at a time, as
#   before batching. Reported to the gateway (Health.max_batch: its in-flight
#   limit; 1 when the memory budget cannot hold two typical rows); memory
#   admission (_Scheduler._admit) caps it further per request. Measured on M1
#   Max (2026-10-11): MLX runs a quantized matmul per row (qmv) below 6 rows
#   for layers wider than 4096 and below 10 for 4096-wide ones, then one qmm
#   whose cost is flat to 32 rows. 9B decode step: 41 ms at B=1, 51 / 90 /
#   127 / 133 ms at B=2/4/8/16, so 16 rows cost about 8 (and 16 measured >= 8
#   at every concurrency). With mrqmm (QWEN_MRQMM, 3..16 rows) the 8-bit
#   linears cost ~1.1x a 1-row step up to 8 rows and ~1.5x at 16. The MoE
#   (35B) reads ~B*8 distinct routed experts until they overlap: with
#   distinct prompts batching pays only from ~16 rows.
# QWEN_BATCH_PREFILL_SHARE (0.5): prefill runs one request at a time, slab by
#   slab, interleaved with the running rows' decode. s is prefill's share of
#   the time when the batch is full; with B of max_batch rows decoding it is
#   s_B = 1 - (1-s)*B/max_batch, i.e. after a slab that took t seconds decode
#   gets t*(1-s_B)/s_B. 1 = a new prompt prefills straight through while the
#   running rows wait.
# QWEN_BATCH_MEM_GB (0 = derived, see _batch_budget): GiB the running
#   rows may reserve for their KV + recurrent state.
# QWEN_BATCH_RESERVE_TOKENS (4096): generation tokens a row reserves at
#   admission, at most (prompt + min(max_tokens, this)). A row that grows past
#   its reservation keeps going; the scheduler re-accounts every row's real
#   length each step, so a grown batch blocks new admissions until the budget
#   holds again. Without the cap a request with no max_tokens would reserve
#   the rest of the 128K context (~4 GiB on the 9B) and run alone. The cost:
#   rows admitted short can outgrow the budget together; what stops that from
#   becoming an OOM is the headroom kept outside the budget (prefill reserve +
#   1 GiB, see _batch_budget) and the live memory check at admission. Past that
#   headroom, a growing row is ended with "length" while other rows run
#   (_Scheduler._regrow); a lone row is never cut.
# QWEN_GEN_SLOTS (max(2*max_batch, 8)): Generate RPCs accepted at once (running or
#   waiting in the scheduler queue); one more is refused with
#   RESOURCE_EXHAUSTED (the gateway answers 503). Every streaming Generate
#   holds a gRPC thread for its whole life, so the pool is slots + 8 threads
#   for Check / Cancel / Health (see serve()).
MAX_BATCH = max(1, int(os.environ.get("QWEN_MAX_BATCH", "16")))
PREFILL_SHARE = min(1.0, max(0.05, float(os.environ.get("QWEN_BATCH_PREFILL_SHARE", "0.5"))))
BATCH_MEM_GB = float(os.environ.get("QWEN_BATCH_MEM_GB", "0") or "0")
RESERVE_TOKENS = max(1, int(os.environ.get("QWEN_BATCH_RESERVE_TOKENS", "4096")))
GEN_SLOTS = int(os.environ.get("QWEN_GEN_SLOTS", "0") or "0")
SCHED_LOG_S = 10.0
GIB = 1024**3             # every *_GB knob and the batching logs are in GiB; Usage.peak_gb is 1e9
SCHED_STALL_S = 300.0     # a tick running this long marks the scheduler wedged (Health: not ready)
SCHED_MAX_FAILURES = 3    # consecutive failed decode steps before the scheduler gives up
EVENT_POLL_S = 5.0        # a handler checks the scheduler is alive this often while it waits


class RowsKV:
    """The per-row KV caches of one full-attention layer in a batched decode
    step. No left_padding / make_mask / _idx: the library then builds no mask
    and takes none of its batch-cache branches."""

    def __init__(self, caches):
        self.caches = caches

    @property
    def offset(self):                 # read by LanguageModel; positions are explicit
        return max(c.offset for c in self.caches)


_Q35 = None                           # mlx_vlm.models.qwen3_5.language, once installed
_ATTN_CALL = None


def _rows_attention(self, x, mask=None, cache=None, position_ids=None,
                    position_embeddings=None, **kw):
    """Qwen3_5Attention.__call__ (mlx-vlm 0.7.6, language.py 857-898) for a
    RowsKV cache: projections, norms and RoPE batched (the library's
    _prepare_projected_qkv without a cache: positions are always explicit),
    cache update + SDPA per row with the same calls a lone row makes. Any
    other cache goes to the library method. Diff against the library's
    __call__ / _prepare_projected_qkv when bumping mlx-vlm."""
    if not isinstance(cache, RowsKV):
        return _ATTN_CALL(self, x, mask=mask, cache=cache, position_ids=position_ids,
                          position_embeddings=position_embeddings, **kw)
    if position_ids is None:
        raise ValueError("a RowsKV step needs explicit position_ids")
    B, L, _ = x.shape
    queries, keys, values, gate, _ = self._prepare_projected_qkv(
        self.q_proj(x), self.k_proj(x), self.v_proj(x), None, position_ids,
        position_embeddings, None)
    outs = []
    for i, c in enumerate(cache.caches):
        k, v = c.update_and_fetch(keys[i:i + 1], values[i:i + 1])
        outs.append(_Q35.scaled_dot_product_attention(queries[i:i + 1], k, v, cache=c,
                                                      scale=self.scale, mask=None))
    out = mx.concatenate(outs, axis=0).transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(out * mx.sigmoid(gate))


def _vision_arrays(vision):
    """Scheduler side of Engine._vision_inputs: mlx-vlm prepare_inputs'
    conversion of the processor output (numpy -> mx.array, the rest as is)."""
    return {k: mx.array(x) if isinstance(x, np.ndarray) else x for k, x in vision.items()}


def _batch_cap(kernel, wired_gb):
    """Memory the batch may use: QWEN_WIRED_LIMIT_GB clamped to the kernel's
    max, else the device's recommended working set (`kernel`), which is the
    iogpu.wired_limit_mb sysctl when raised, else macOS's default share of RAM."""
    return min(int(wired_gb * GIB), kernel) if wired_gb > 0 else kernel


def _exit_soon(code=70):
    """The scheduler gave up: exit after the error events went out, so launchd
    (KeepAlive) restarts the worker instead of a live process refusing
    everything."""
    log.error("exiting with %d in 1 s so the worker is restarted", code)
    threading.Timer(1.0, os._exit, (code,)).start()


def _batch_budget(cap, active, prefill_reserve):
    """Bytes the running rows may reserve: QWEN_BATCH_MEM_GB, else cap - what
    is resident (weights + drafter, `active` after load) - the prefill
    activations (the attention scratch PREFILL_BUDGET x q heads x 2 B: the
    head_dim-256 SDPA fallback materializes the scores, + ~0.4 GiB of MLP) -
    1 GiB slack (logits, KV growth steps) - the prompt cache - the buffer pool
    when it is kept across requests (QWEN_CLEAR_CACHE=0). The headroom outside
    the budget (prefill reserve + 1 GiB) is what absorbs rows that outgrow
    their QWEN_BATCH_RESERVE_TOKENS reservation."""
    if BATCH_MEM_GB > 0:
        return int(BATCH_MEM_GB * GIB)
    budget = (cap - active - prefill_reserve - GIB
              - (int(PROMPT_CACHE_GB * GIB) if PROMPT_CACHE else 0)
              - (0 if CLEAR_CACHE else int(CACHE_LIMIT_GB * GIB)))
    return max(0, budget)


def _install_rows_attention(lm):
    """Patch Qwen3_5Attention (also the MoE model's attention: the same class)
    once. -> False when `lm` is another architecture (no batching then)."""
    global _Q35, _ATTN_CALL
    from mlx_vlm.models.qwen3_5 import language as q35
    layers = getattr(getattr(lm, "model", None), "layers", None) or []
    if not layers or not all(hasattr(x, "is_linear") and (x.is_linear or isinstance(
            x.self_attn, q35.Qwen3_5Attention)) for x in layers):
        return False
    if _ATTN_CALL is None:
        _Q35, _ATTN_CALL = q35, q35.Qwen3_5Attention.__call__
        q35.Qwen3_5Attention.__call__ = _rows_attention
    return True


# QWEN_SDPA_FUSED (1): a prefill slab of an fp16 KV cache (KVCache, fp16
#   queries, causal) with q_len >= 512 over >= 32K keys takes MLX's fused SDPA
#   kernel (force_fused, mlx >= 0.32). MLX's default for head_dim 256 there is
#   the unfused graph, which materializes the scores (slab x keys x heads).
#   Per call, M1 Max, mlx 0.32.3, 9B layout, time fused / default (final
#   interleaved re-measure 2026-10-12, results/prefill_profile.md §3; 64K+ from
#   results/microbench.md §2): up to 28K keys the default path is faster
#   (fused x1.05-1.12, x1.00 at 2048 x 8K); from 32K fused wins: x0.89 at
#   2048 x 32K (x0.95 at 1024, x1.03 at 512), x0.83 at 2048 x 40K, x0.59-0.68
#   at 64K, x0.67-0.72 at 512 x 120K. The default path also jumps at 32K keys
#   (+34% for +14% keys from 28K), so the threshold sits there. Prefill peak
#   -1.3 GB end-to-end at 64K (-1.5..-1.8 GB in the microbench). At q_len <= 16
#   fused is 30-50% slower, so decode / verify never take it. Quantized KV
#   keeps the library's quantized SDPA. 0 = the library's dispatch.
#   A slab under 512 tokens takes it too when its scores would pass
#   PREFILL_BUDGET: with the kernel, _step keeps fp16 slabs at 2048 at any
#   context.
# QWEN_GDN_FAST (0): gated-delta recurrence of slabs with >= 16 tokens through
#   mx.fast.gated_delta_update (chunk-parallel, mlx >= 0.32.3): x2.2 on the
#   recurrence (142 -> 64 ms per 2048-token slab, 24 layers), but end-to-end
#   8K prefill x0.994 (in-process A/B); it rounds g (and beta) to fp16: KL vs
#   off 6e-5 (9B), 1e-5 (27B), 9e-3 (35B). Off. Re-measured 2026-10-12 with
#   pfqmm: x1.024 per 2048 slab, but teacher-forced on 8K tokens of real text
#   KL 0.013, top-1 97.0% on the 9B (the fp16 decay drifts over long prompts).
# QWEN_VERIFY_EXACT_MAX_T (8): DFlash verify blocks of up to this many tokens
#   run mlx-vlm 0.7's exact verify linears (rows bit-identical to a 1-token
#   decode, so DFlash = plain greedy; 8-bit multi-row qmv: x0.63-0.82 of
#   quantized_matmul at 4..8 rows on M1); longer blocks use quantized_matmul,
#   as mlx-vlm 0.6.2 did for one row: the exact kernel spills registers from
#   12 rows (x9-12 slower, results/microbench.md §1a). DFlash on the 9B coding
#   tasks, median tok/s: cap 0 35, 8 46-52, 10/12 ~52 but one task drops to 27,
#   16 39. 0 = never exact; 16 = every block of the adaptive size (the
#   drafter's block_size 16; a QWEN_DRAFT_BLOCK above 16 still has longer
#   blocks on quantized_matmul). An empty value = the default.
SDPA_FUSED = os.environ.get("QWEN_SDPA_FUSED", "1") != "0"
SDPA_FUSED_MIN_Q, SDPA_FUSED_MIN_KV = 512, 32768
GDN_FAST = os.environ.get("QWEN_GDN_FAST", "0") == "1"
GDN_FAST_MIN_T = 16
VERIFY_EXACT_MAX_T = int(os.environ.get("QWEN_VERIFY_EXACT_MAX_T") or "8")
_SDPA = None                          # the library's qwen3_5 scaled_dot_product_attention
_GDN_KERNEL = None                    # the library's gated_delta_kernel
_VERIFY_LINEAR = _VERIFY_LINEARS = None   # Qwen3_5BatchInvariantForward._linear / _linears


def _fused_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """qwen3_5.language.scaled_dot_product_attention with force_fused for
    long fp16 prefill slabs (QWEN_SDPA_FUSED); everything else unchanged. If
    the fused kernel fails at run time, the gate turns off; a slab whose
    unfused scores would pass PREFILL_BUDGET (_step sized it for the fused
    kernel) then fails its request instead of materializing them."""
    global SDPA_FUSED
    # cache type first: a quantized cache hands over (packed, scales, biases)
    # tuples as keys/values
    if (type(cache).__name__ == "KVCache" and isinstance(keys, mx.array)
            and sinks is None and isinstance(mask, str) and mask == "causal"
            and queries.dtype == mx.float16 and keys.shape[2] >= SDPA_FUSED_MIN_KV
            and (queries.shape[2] >= SDPA_FUSED_MIN_Q
                 # a short slab (message cut, last slab) whose scores would
                 # exceed the budget _step relies on
                 or (queries.shape[2] > 16 and queries.shape[2] * keys.shape[2] > PREFILL_BUDGET))):
        if SDPA_FUSED:
            try:
                return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale,
                                                            mask=mask, force_fused=True)
            except (ValueError, RuntimeError) as e:    # no fused kernel for this shape / device
                SDPA_FUSED = False
                log.warning("QWEN_SDPA_FUSED off: %s", e)
        q, k = queries.shape[2], keys.shape[2]
        if q * k > PREFILL_BUDGET:
            raise RuntimeError(
                f"fused SDPA unavailable for a {q} x {k} prefill slab; unfused it would hold "
                f"{q * k * queries.shape[1] * 2 / GIB:.1f} GiB of scores (> QWEN_PREFILL_BUDGET)")
    return _SDPA(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)


def _probe_fused_sdpa():
    """QWEN_SDPA_FUSED off at load unless the fused kernel runs a causal
    head_dim-256 fp16 slab here: _step sizes fp16 slabs for it, and the
    unfused fallback at a long context would materialize the scores."""
    global SDPA_FUSED
    try:
        q = mx.zeros((1, 4, 64, 256), dtype=mx.float16)
        k = mx.zeros((1, 1, 128, 256), dtype=mx.float16)
        mx.eval(mx.fast.scaled_dot_product_attention(q, k, k, scale=0.0625, mask="causal",
                                                     force_fused=True))
    except (ValueError, RuntimeError, TypeError) as e:
        SDPA_FUSED = False
        log.warning("QWEN_SDPA_FUSED off (no fused SDPA kernel): %s", e)


def _fast_gdn_kernel(q, k, v, g, beta, state, mask=None):
    """qwen3_5.gated_delta.gated_delta_kernel (the Metal path of
    gated_delta_update without a speculative history) through
    mx.fast.gated_delta_update for slabs of >= GDN_FAST_MIN_T tokens."""
    if mask is None and g.ndim == 3 and q.shape[1] >= GDN_FAST_MIN_T:
        y, s = mx.fast.gated_delta_update(q, k, v, g, beta, initial_state=state)
        return y, s.astype(state.dtype)
    return _GDN_KERNEL(q, k, v, g, beta, state, mask)


def _verify_linear(self, linear, x):
    if x.shape[1] > VERIFY_EXACT_MAX_T:
        return linear(x)
    return _VERIFY_LINEAR(self, linear, x)


def _verify_linears(self, linears, x):
    if x.shape[1] > VERIFY_EXACT_MAX_T:
        return tuple(linear(x) for linear in linears)
    return _VERIFY_LINEARS(self, linears, x)


# QWEN_GDN_FUSE (1): a decode step's four GDN in-projections (qkv, z, b, a)
#   as ONE quantized matmul (qmv for 1..2 rows, mrqmm for 3..16 rows on 8-bit
#   models, otherwise quantized_matmul's own qmv/qmm) over their weights
#   concatenated at load; the four modules keep row views of the concatenation
#   (no extra memory). Needs fp16 scales/biases: with QWEN_FP16=0 (bf16) it
#   fuses nothing (the load log says "0 layers"). mlx-vlm 0.6.2 did this (with a second copy of the
#   weights, +1.2 GB on the 9B); 0.7 runs four launches per GDN layer.
#   Per output row the same kernel and K order: bit-identical to the unfused
#   calls on qmv; with mrqmm the narrow b/a projections (N=32, else on qmv)
#   ride along in the multi-row kernel, as on the old stack.
GDN_FUSE = os.environ.get("QWEN_GDN_FUSE", "1") != "0"
_GDN_IN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_GDN_CALL = None                      # the library's Qwen3_5GatedDeltaNet.__call__


class _Given:
    """Stands in for an in-projection during one fused decode call: returns
    its precomputed output."""
    __slots__ = ("y",)

    def __init__(self, y):
        self.y = y

    def __call__(self, x):
        return self.y


def _gdn_call(self, inputs, mask=None, cache=None):
    """Qwen3_5GatedDeltaNet.__call__ with the fused decode in-projection: the
    four outputs are computed in one matmul and handed to the library call in
    place of the modules (restored right after; the scheduler thread is the
    only caller of the model)."""
    f = self.__dict__.get("_fused_in")
    if f is None or not GDN_FUSE or inputs.ndim != 3 or inputs.shape[1] != 1 or inputs.dtype != f[2].dtype:
        return _GDN_CALL(self, inputs, mask=mask, cache=cache)
    w, s, b, split, gs, bits, mode, mr = f
    if mr is not None and mr._use(inputs):
        y = mr.mrqmm(inputs, w, s, b)
    else:
        y = mx.quantized_matmul(inputs, w, scales=s, biases=b, transpose=True,
                                group_size=gs, bits=bits, mode=mode)
    mods = [self[n] for n in _GDN_IN]
    for n, o in zip(_GDN_IN, mx.split(y, split, axis=-1)):
        dict.__setitem__(self, n, _Given(o))
    try:
        return _GDN_CALL(self, inputs, mask=mask, cache=cache)
    finally:
        for n, m in zip(_GDN_IN, mods):
            dict.__setitem__(self, n, m)


def _install_gdn_fuse(lm):
    """QWEN_GDN_FUSE: concatenate the in-projections of every GDN layer whose
    four are quantized alike (fp16 scales/biases, no bias), point the modules
    at row views of it, route the class __call__. After the fp16 cast and
    mrqmm.install. -> number of layers fused."""
    global _GDN_CALL
    if not GDN_FUSE:
        return 0
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import language as q35
    mr = sys.modules.get("mrqmm")
    n, warm = 0, {}
    for _, m in lm.named_modules():
        if not isinstance(m, q35.Qwen3_5GatedDeltaNet) or "_fused_in" in m.__dict__:
            continue
        ls = [m[a] for a in _GDN_IN]
        f = ls[0]
        if not all(isinstance(l, nn.QuantizedLinear) and "bias" not in l and l.get("biases") is not None
                   and l.bits == f.bits and l.group_size == f.group_size
                   and getattr(l, "mode", "affine") == getattr(f, "mode", "affine")
                   and l.scales.dtype == mx.float16 and l.biases.dtype == mx.float16 for l in ls):
            continue
        w, s, b = (mx.concatenate([l[k] for l in ls], axis=0) for k in ("weight", "scales", "biases"))
        mx.eval(w, s, b)
        offs, o = [], 0
        for l in ls:
            offs.append((o, o + l.weight.shape[0]))
            o += l.weight.shape[0]
        for l, (lo, hi) in zip(ls, offs):     # row views: the old buffers are freed
            l.weight, l.scales, l.biases = w[lo:hi], s[lo:hi], b[lo:hi]
            mx.eval(l.weight, l.scales, l.biases)
        use_mr = (mr is not None and mr.enabled() and f.bits == mr.BITS and f.group_size == mr.GROUP
                  and mr.supported(w, s, b, f.bits, f.group_size, getattr(f, "mode", "affine")))
        m.__dict__["_fused_in"] = (w, s, b, [hi for _, hi in offs[:-1]], f.group_size, f.bits,
                                   getattr(f, "mode", "affine"), mr if use_mr else None)
        if use_mr:
            warm.setdefault(w.shape, (w, s, b))
        n += 1
    # the concatenated shapes (9B K=4096, 35B K=2048 -> N=12352) are new to
    # mrqmm: compile their 3..8 / 9..16-row kernels now, not in the first
    # batched decode step on the scheduler thread
    for ws in warm.values():
        mr.warm(*ws)
    warm.clear()
    if n and _GDN_CALL is None:
        _GDN_CALL, q35.Qwen3_5GatedDeltaNet.__call__ = q35.Qwen3_5GatedDeltaNet.__call__, _gdn_call
    if n:
        mx.clear_cache()
    return n


def _install_fast_paths():
    """QWEN_SDPA_FUSED / QWEN_GDN_FAST / QWEN_VERIFY_EXACT_MAX_T as overrides
    of the qwen3_5 module globals / verifier methods (the MoE model uses the
    same modules). Once; -> what is on."""
    global _SDPA, _GDN_KERNEL, _VERIFY_LINEAR, _VERIFY_LINEARS
    from mlx_vlm.models.qwen3_5 import gated_delta, language as q35, speculative_verifier
    if SDPA_FUSED and _SDPA is None:
        _probe_fused_sdpa()
    if SDPA_FUSED and _SDPA is None:
        _SDPA, q35.scaled_dot_product_attention = q35.scaled_dot_product_attention, _fused_sdpa
    if GDN_FAST and _GDN_KERNEL is None and hasattr(mx.fast, "gated_delta_update"):
        _GDN_KERNEL, gated_delta.gated_delta_kernel = gated_delta.gated_delta_kernel, _fast_gdn_kernel
    v = speculative_verifier.Qwen3_5BatchInvariantForward
    if _VERIFY_LINEAR is None:
        _VERIFY_LINEAR, _VERIFY_LINEARS = v._linear, v._linears
        v._linear, v._linears = _verify_linear, _verify_linears
    return {"sdpa_fused": _SDPA is not None, "gdn_fast": _GDN_KERNEL is not None,
            "verify_exact_max_t": VERIFY_EXACT_MAX_T}


def _kv_token_bytes(lm, bits):
    """Full-attention KV bytes per token of one row: fp16, or `bits`-bit values
    in uint32 words plus an fp16 scale and bias per 64-element group."""
    n_fa = sum(1 for x in lm.model.layers if not x.is_linear)
    elems = 2 * n_fa * lm.args.num_key_value_heads * lm.args.head_dim
    return elems * 2 if not bits else int(elems * (bits / 8 + 4 / 64))


def _gdn_row_bytes(lm):
    """Recurrent state of one row: conv state (fp16) + delta-rule state (fp32)."""
    n = 0
    for layer in lm.model.layers:
        if layer.is_linear:
            la = layer.linear_attn
            n += (la.conv_kernel_size - 1) * la.conv_dim * 2 \
                + la.num_v_heads * la.head_v_dim * la.head_k_dim * 4
    return n


def _gdn_zero(layer, dtype):
    """The state Qwen3_5GatedDeltaNet starts from (a 1-token prompt has none yet)."""
    la = layer.linear_attn
    return [mx.zeros((1, la.conv_kernel_size - 1, la.conv_dim), dtype=dtype),
            mx.zeros((1, la.num_v_heads, la.head_v_dim, la.head_k_dim), dtype=mx.float32)]


def _gdn_stack(lm, row_caches, dtype):
    """{layer: ArraysCache holding the rows' recurrent states on axis 0}. The
    rows' own ArraysCaches are emptied (the stacked one is the live state),
    but only once every layer is built: a failure leaves the rows intact."""
    from mlx_vlm.models.cache import ArraysCache
    out = {}
    for i, layer in enumerate(lm.model.layers):
        if not layer.is_linear:
            continue
        parts = [list(rc[i].cache) if rc[i].cache[0] is not None else _gdn_zero(layer, dtype)
                 for rc in row_caches]
        b = ArraysCache(size=2)
        b.cache = [mx.concatenate([p[j] for p in parts], axis=0) for j in range(2)]
        out[i] = b
    for i in out:
        for rc in row_caches:
            rc[i].cache = [None] * len(rc[i].cache)
    return out


def _gdn_take(gdn, keep):
    from mlx_vlm.models.cache import ArraysCache
    idx = mx.array(keep, dtype=mx.int32)
    out = {}
    for i, c in gdn.items():
        out[i] = ArraysCache(size=2)
        out[i].cache = [a[idx] for a in c.cache]
    return out


def _gdn_concat(a, b):
    from mlx_vlm.models.cache import ArraysCache
    out = {}
    for i in a:
        out[i] = ArraysCache(size=2)
        out[i].cache = [mx.concatenate([x, y], axis=0) for x, y in zip(a[i].cache, b[i].cache)]
    return out


def _gdn_split(gdn, row, row_cache):
    """Give row `row` its state back (the batch shrinks to it alone)."""
    for i, c in gdn.items():
        row_cache[i].cache = [a[row:row + 1] for a in c.cache]


@dataclasses.dataclass(eq=False)
class _Req:
    """A generation as the handler thread hands it to the scheduler (built
    without MLX). The scheduler answers on `out`: ("tok", id, shown) per output
    token (shown False: an ignored EOS, not text), then ("done", finish) or
    ("error", kind, msg). `cancel` (Cancel RPC, a dropped stream, the handler's
    stop-string hit) evicts the request wherever it is."""
    id: str
    p: object                                   # pb.Params
    ids: list                                   # prompt ids (image tokens included)
    max_tokens: int
    sampling: dict
    ctl: "_Control"
    guide: "_Guide | None" = None
    thinking: object = None                     # ThinkingBudgetCriteria
    formatted: str = ""
    images: list = dataclasses.field(default_factory=list)
    vision: dict | None = None                  # processor output, numpy (see Engine._vision_inputs)
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event)
    out: queue.SimpleQueue = dataclasses.field(default_factory=queue.SimpleQueue)
    t_arrival: float = dataclasses.field(default_factory=time.time)
    # written by the scheduler before the events that make them meaningful
    t0: float = 0.0                             # prefill start
    queue_ms: int = 0
    reused: int = 0                             # prompt-cache tokens
    steps: int = 0                              # decode steps (DFlash: rounds) taken
    rows_sum: int = 0                           # sum of the batch sizes of those steps
    # snapshot taken on the scheduler when the request ends (before anything
    # can reset the peak): bytes, and the mean batch size over its steps
    peak: int = 0
    active: int = 0
    pool: int = 0
    batch_mean: float = 0.0

    @property
    def prompt_tokens(self):
        return len(self.ids)


class _Row:
    """Scheduler-side state of an admitted request. `y` is the next input token
    (a lazy (1,) array); unless `first` it is also the newest output token,
    not read yet. `first`: y is input only — the last prompt token, or a token
    DFlash already emitted (no read, no penalty / min_tokens count)."""

    def __init__(self, req, tokens, reserved):
        self.req = req
        self.tokens = tokens                        # KV tokens reserved for
        self.reserved = reserved                    # ... and their bytes
        self.cache = None
        self.pos = 0                                # KV length = next position
        self.delta = 0                              # mRoPE delta (vision)
        self.y = None
        self.first = True
        self.n = 0                                  # tokens sent
        self.done = False
        self.sampler = self.pen = None

    @property
    def sync(self):
        # the mask for token n+1 needs token n on the host: no step-ahead
        return self.req.guide is not None or self.req.ctl.sync


class _Scheduler:
    """Continuous batching; after load, the one thread that runs MLX.

    Handlers submit _Req (FIFO, no overtaking). The head is admitted when it
    fits: fewer than max_batch rows and its reservation (KV for prompt +
    min(max_tokens, QWEN_BATCH_RESERVE_TOKENS), recurrent state) within the
    budget, plus a live check of the active memory; an empty batch always
    admits it (the old one-at-a-time behaviour: MAX_CONTEXT validation bounds
    a lone request). Rows are re-accounted at their real length every step.
    Prefill is one
    request at a time (the slab loop, prompt cache, checkpoints), a slab per
    tick, interleaved with decode (PREFILL_SHARE); then the row joins the
    batch. A decode tick is one forward for every row, then the per-row
    post-processing.

    DFlash (QWEN_DRAFT) runs only for an eligible request admitted with
    nothing else running or waiting; when another request arrives (and
    max_batch > 1) it is demoted at the next round boundary to a plain batch
    row, and stays one.

    Usage.peak_gb: the MLX peak since the batch was last empty, read when the
    request ends; with concurrent rows it is the process peak during the
    request, not the request's own share.

    Failures: a failed decode step fails the rows in it (not the prefilling
    request). SCHED_MAX_FAILURES consecutive failed steps with several rows
    (or with none: the scheduler itself) close it: everything queued or
    running gets Error(worker_fault), new requests are refused, Health
    reports not ready and on_fault() runs (Engine: exit the process so
    launchd restarts it). A step failing for a lone row is that request's
    own error and no strike. A tick running for SCHED_STALL_S only makes
    alive() false (Health not ready, waiting handlers give up) until it
    ends. stop() (Shutdown) ends the loop after the current tick and sends
    Error(shutdown) to everything left."""

    def __init__(self, eng, *, max_batch, budget, cap, prefill_reserve,
                 share=PREFILL_SHARE, kv=None, on_fault=None):
        from mlx_vlm.generate.common import maybe_quantize_kv_cache
        self.eng = eng
        self.model = eng.model
        self.lm = eng.model.language_model
        self.max_batch = max_batch
        self.budget, self.cap, self.prefill_reserve, self.share = budget, cap, prefill_reserve, share
        self.kv = kv or KV_KWARGS
        self.quantize = functools.partial(
            maybe_quantize_kv_cache, quantized_kv_start=self.kv["quantized_kv_start"],
            kv_group_size=self.kv["kv_group_size"], kv_bits=self.kv["kv_bits"])
        self.kv_fp16 = _kv_token_bytes(self.lm, 0)
        self.kv_q = _kv_token_bytes(self.lm, self.kv["kv_bits"])
        self.gdn_row = _gdn_row_bytes(self.lm)
        self.cv = threading.Condition()
        self.waiting = collections.deque()
        self.rows: list[_Row] = []
        self.gdn = None                 # {layer: stacked ArraysCache} while B >= 2
        self.task = None                # generator: the prefill / lone DFlash run in progress
        self.task_row = None
        self.resume = 0.0               # prefill waits until then (decode's share)
        self.reserved = 0
        self.steps = 0
        self.clear = False
        self.wired = None
        self.hdtype = None
        self.log_at, self.log_steps, self.log_rows = time.time(), 0, 0
        self.thread = None
        self.closed = None              # (kind, msg) once the scheduler stopped or died
        self.stopping = False
        self.failures = 0
        self.tick_t0 = 0.0              # start of the running tick, 0 between ticks
        self.on_fault = on_fault

    # --- threads -------------------------------------------------------------
    def start(self):
        self.thread = threading.Thread(target=self._run, name="scheduler", daemon=True)
        self.thread.start()

    def alive(self):
        """Running, not closed, and not stuck in one tick (Health: ready).
        Idle time never counts: only a tick that has been running too long."""
        t0 = self.tick_t0
        return (self.closed is None and self.thread is not None and self.thread.is_alive()
                and not (t0 and time.time() - t0 > SCHED_STALL_S))

    def submit(self, req):
        with self.cv:
            if self.closed is None:
                self.waiting.append(req)
                self.cv.notify()
                return
        req.out.put(("error", *self.closed))

    def stop(self, timeout=10.0):
        """Stop after the current tick; everything left gets Error(shutdown).
        -> False when the tick did not end within `timeout`."""
        with self.cv:
            self.stopping = True
            self.cv.notify()
            return self.cv.wait_for(lambda: self.closed is not None or self.thread is None
                                    or not self.thread.is_alive(), timeout)

    def _run(self):
        from mlx_vlm.generate.common import generation_stream
        closed = ("shutdown", "the worker is shutting down")
        try:
            with mx.stream(generation_stream):
                while True:
                    with self.cv:
                        while not (self.rows or self.task or self.waiting or self.stopping):
                            self._idle()
                            self.cv.wait()
                    if self.stopping:
                        break
                    rows = len(self.rows)
                    self.tick_t0 = time.time()
                    try:
                        self.tick()
                    except Exception as e:  # noqa: BLE001 — a failed step fails its rows, not the worker
                        log.exception("batch step failed")
                        self._fail_rows(e)
                        # a lone row's failure is its own request's error (a
                        # client retrying it must not take the worker down)
                        self.failures += rows != 1
                        if self.failures >= SCHED_MAX_FAILURES:
                            closed = ("worker_fault", f"scheduler failed {self.failures} steps "
                                                      f"in a row: {type(e).__name__}: {e}")
                            log.error("scheduler gives up: %s", closed[1])
                            break
                    finally:
                        self.tick_t0 = 0.0
        except BaseException as e:  # noqa: BLE001 — never leave handlers waiting on a dead thread
            closed = ("worker_fault", f"scheduler thread died: {type(e).__name__}: {e}")
            log.exception("scheduler thread died")
        finally:
            self._close(*closed)
            if closed[0] == "worker_fault" and self.on_fault is not None:
                self.on_fault()
            # The thread never returns: MLX tears down its thread-local
            # generation stream when the thread exits, which crashed the
            # process (segfault) while arrays from that stream were alive.
            with self.cv:
                while True:
                    self.cv.wait()

    def _close(self, kind, msg):
        """Refuse new requests and end everything queued or running."""
        with self.cv:
            self.closed = (kind, msg)
            left = list(self.waiting)
            self.waiting.clear()
            self.cv.notify_all()
        rows = self.rows + ([self.task_row] if self.task_row is not None else [])
        for r in rows:
            if not r.done:
                r.done = True
                r.req.out.put(("error", kind, msg))
        for req in left:
            req.out.put(("error", kind, msg))
        self.rows, self.gdn, self.task, self.task_row, self.reserved = [], None, None, None, 0
        try:
            from mlx_vlm.generate.common import generation_stream
            mx.synchronize(generation_stream)
            self._idle()
        except Exception:  # noqa: BLE001 — closing anyway
            log.exception("scheduler cleanup failed")

    def _busy(self):
        # wired_limit's exit synchronizes and restores the limit, so it wraps
        # the busy period, never a single row
        if self.wired is None and mx.default_device().type == mx.gpu:
            from mlx_vlm.generate.common import generation_stream, wired_limit
            self.wired = wired_limit(self.model, [generation_stream])
            self.wired.__enter__()

    def _idle(self):
        wired, self.wired = self.wired, None       # never retried if __exit__ raises
        if wired is not None:
            wired.__exit__(None, None, None)
        if CLEAR_CACHE:
            mx.clear_cache()

    # --- one scheduling decision ----------------------------------------------
    def tick(self):
        self._busy()
        with self.cv:                       # handlers append meanwhile
            gone = [r for r in self.waiting if r.cancel.is_set()]
            for r in gone:
                self.waiting.remove(r)
        for r in gone:
            r.out.put(("done", "cancelled"))
        if self.task is None and self.waiting:
            self._admit()
        if self.task is not None and (not self.rows or time.time() >= self.resume):
            self._advance()
        elif self.rows:
            self._iterate()
        if self.clear:
            # a finished long row's KV and prefill buffers go back to the pool
            if CLEAR_CACHE:
                mx.clear_cache()
            self.clear = False
        self._log()

    def _bytes(self, n):
        """Reservation for a row of n KV tokens."""
        start = self.kv["quantized_kv_start"]
        if self.kv["kv_bits"] and n >= start:
            kv = start * self.kv_fp16 + n * self.kv_q     # + the fp16 copy while quantizing
        else:
            kv = n * self.kv_fp16
        # +5%: a KVCache grows by concatenating 256-token steps; the recurrent
        # state twice: the queued step's output exists beside its input
        return int(kv * 1.05) + 2 * self.gdn_row + (8 << 20)

    def _tokens(self, req):
        return len(req.ids) + min(req.max_tokens, RESERVE_TOKENS)

    def _reserve(self, req):
        return self._bytes(self._tokens(req))

    def _regrow(self, row):
        """Re-account a row that outgrew its reservation (by 1024-token steps,
        up to its own limit): admissions wait while the batch is over budget.
        Growth past the headroom too (budget + prefill reserve; the last 1 GiB
        stays slack) would end in an OOM of every row, so then the growing row
        ends with "length" — only when other rows run: a lone row is bounded
        by MAX_CONTEXT alone, as before batching."""
        if row.pos < row.tokens:
            return
        tokens = min(row.pos + 1024, len(row.req.ids) + row.req.max_tokens + 1)
        need = self._bytes(tokens)
        if (self.reserved + need - row.reserved > self.budget + self.prefill_reserve
                and sum(not r.done for r in self.rows) > 1):
            log.warning("id=%s ended at %d tokens: the batch's KV would pass its memory "
                        "headroom (%.2f GiB reserved)", row.req.id, row.n, self.reserved / GIB)
            self._finish(row, "length")
            return
        row.tokens = tokens
        self.reserved += need - row.reserved
        row.reserved = need

    def _fits(self, req, rows):
        """Would `req` be admitted next to `rows` running rows?"""
        if rows == 0:
            return True
        need = self._reserve(req)
        live = mx.get_active_memory() + (0 if CLEAR_CACHE else mx.get_cache_memory())
        return (rows < self.max_batch and self.reserved + need <= self.budget
                and live + need + self.prefill_reserve <= self.cap)

    def _admit(self):
        req = self.waiting[0]
        if not self._fits(req, len(self.rows)):
            return
        if not self.rows:
            mx.reset_peak_memory()
        self.waiting.popleft()
        tokens = self._tokens(req)
        row = _Row(req, tokens, self._bytes(tokens))
        self.reserved += row.reserved
        req.t0 = time.time()
        req.queue_ms = int((req.t0 - req.t_arrival) * 1000)
        try:
            row.sampler = make_sampler(req.sampling, seed=req.sampling.get("seed"))
            row.pen = _Penalties(req.sampling, req.ids, self.eng.vocab)
            req.ctl.build()
            spec = self._spec_ok(row)
        except Exception as e:  # noqa: BLE001 — this request's options, not the batch's
            log.exception("request setup failed: %s", req.id)
            self._fail(row, e)
            return
        self.task_row = row
        self.task = self._spec(row) if spec else self._prefill(row)

    def _spec_ok(self, row):
        req = row.req
        greedy = req.sampling["temperature"] <= 0 and not row.pen.active
        if (self.eng.draft is None or req.images or req.thinking is not None
                or not (greedy or DRAFT_SAMPLING)):
            return False
        if req.guide is not None or req.ctl.active:
            # DFlash drafts are not grammar-masked: the verify rows would each
            # need a mask over the draft prefix. Plain decoding instead.
            log.info("grammar / token options: speculative decoding off for this request")
            return False
        return not self.rows and (self.max_batch == 1 or not self.waiting)

    def _advance(self):
        try:
            next(self.task)
        except StopIteration:
            self.task = self.task_row = None
        except Exception as e:  # noqa: BLE001 — a failed prefill fails its request only
            log.exception("prefill failed: %s", self.task_row.req.id)
            self._fail(self.task_row, e)
            self.task = self.task_row = None

    # --- prefill ----------------------------------------------------------------
    def _prompt_cache(self, row):
        """Prompt-cache lookup / restore and the checkpoint plan (text only) ->
        (input ids to prefill, cache, reused tokens, slab cuts, on_ckpt, on_prefilled)."""
        from mlx_vlm.models.cache import make_prompt_cache
        req, eng, lm = row.req, self.eng, self.lm
        ids = req.ids
        use_cache = PROMPT_CACHE and not req.images
        bounds = [] if req.images else _bounds(ids, eng._im_start)
        ends = set(bounds) | {len(ids) - 1}
        parent, reused, split = eng._pcache.lookup(ids, ends) if use_cache else (None, 0, 0)
        if parent is not None:
            cache = eng._pcache.restore(parent, reused)
            input_ids = mx.array([ids[reused:]])
            # Text-only mRoPE continues from the cache offset with a zero
            # delta; a preceding vision request leaves its own delta here.
            lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
            lm._position_ids = None
            log.info("id=%s prompt cache HIT: %d/%d tokens reused (common prefix %d), "
                     "prefilling %d", req.id, reused, len(ids), split, len(ids) - reused)
        else:
            cache = make_prompt_cache(lm)
            input_ids = _vision_arrays(req.vision)["input_ids"] if req.images else mx.array([ids])
            if use_cache:
                log.info("id=%s prompt cache MISS (common prefix %d, %d entries)",
                         req.id, split, len(eng._pcache.entries))
        req.reused = reused
        keep, ckpts = set(), {}
        if use_cache:
            end = len(ids) - 1
            want = set(range(PROMPT_CACHE_STRIDE, end, PROMPT_CACHE_STRIDE))
            want.add(split)
            # Message boundaries are where prompts naturally diverge: the
            # end of the system/tools block (another conversation), the
            # last two messages (regenerate / edited turn) and the reply
            # (the next turn appends to it).
            starts = [i for i, t in enumerate(ids) if t == eng._im_start]
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
            if not use_cache or any(getattr(c, "keys", 1) is None for c in cache):
                return
            e = eng._pcache.store(ids[:-1], cache, ckpts, parent, reused)
            log.info("id=%s prompt cache store: %d tokens, %d checkpoints, %.1f MB "
                     "(%d entries, %.1f MB total)", req.id, len(e.ids), len(e.ckpts),
                     e.nbytes / 1e6, len(eng._pcache.entries),
                     sum(x.nbytes for x in eng._pcache.entries) / 1e6)

        return input_ids, cache, reused, set(bounds), checkpoint, snapshot

    def _slab_done(self, t):
        # Decode's time grows with the rows it serves: prefill is compute-
        # bound and B decode rows cost about one row's step, so a lone row
        # would take half the GPU from a waiting prompt for one token per
        # step, and with long prompts the batch would never form.
        s = 1 - (1 - self.share) * len(self.rows) / self.max_batch
        self.resume = time.time() + t * (1 - s) / s

    def _prefill(self, row):
        """Everything but the last prompt token goes through the cache in
        slabs (a generator: one slab per tick); the last one is the row's
        first decode input. A trimmed-down generate_step from mlx-vlm 0.6.2,
        kept local so that the prompt-cache snapshot is taken where the cache
        holds exactly prompt[:-1], and pre-tokenized vision inputs go in
        directly. When bumping mlx-vlm, diff generate/ar.py::generate_step
        against this + _iterate."""
        req, lm = row.req, self.lm
        input_ids, cache, start, cuts, on_ckpt, on_prefilled = self._prompt_cache(row)
        pixel_values = mask = None
        kwargs = {}
        if req.images:
            v = _vision_arrays(req.vision)
            req.vision = None
            pixel_values, mask = v.get("pixel_values"), v.get("attention_mask")
            kwargs = {k: x for k, x in v.items()
                      if k not in ("input_ids", "pixel_values", "attention_mask")}
        row.cache = cache
        emb = self.model.get_input_embeddings(input_ids, pixel_values, mask=mask, **kwargs)
        inputs_embeds = emb.inputs_embeds
        kwargs.update({k: v for k, v in emb.to_dict().items()
                       if k != "inputs_embeds" and v is not None})
        # mlx-vlm 0.7 returns mRoPE position_ids / rope_deltas (counted from 0
        # over the ids it was given) instead of setting them on the model.
        # Text: none of it (a prompt-cache hit's suffix would be placed at 0);
        # positions continue from the cache offset, as 0.6.2 did. Vision: the
        # positions are sliced per slab below, the delta goes to the row.
        vision_pos = kwargs.pop("position_ids", None)
        deltas = kwargs.pop("rope_deltas", None)
        if pixel_values is None:
            vision_pos = deltas = None
            lm._position_ids = None
        pos = start if pixel_values is None else int(cache[lm.model.fa_idx].offset)
        while inputs_embeds.shape[1] > 1:
            if req.cancel.is_set():
                self._finish(row, "cancelled")
                return
            t = time.time()
            n = _slab(pos, inputs_embeds.shape[1] - 1, cuts)
            # An image prompt's mRoPE positions (get_input_embeddings) are
            # sliced per slab: past the first slab the library would give
            # every token offset + delta, right only after the last image.
            if vision_pos is not None:
                kwargs["position_ids"] = vision_pos[:, :, pos:pos + n]
            # skip_logits: no lm_head over the slab (248K vocab = ~12% of
            # the per-token FLOPs); only the cache matters here.
            lm(input_ids[:, :n], inputs_embeds=inputs_embeds[:, :n], cache=cache,
               skip_logits=True, **kwargs)
            self.quantize(cache)
            mx.eval([c.state for c in cache])
            pos += n
            if pos in cuts:
                on_ckpt(pos, cache)
            inputs_embeds = inputs_embeds[:, n:]
            input_ids = input_ids[:, n:]
            if CLEAR_CACHE:
                mx.clear_cache()
            if inputs_embeds.shape[1] > 1:      # the last slab joins at once
                self._slab_done(time.time() - t)
                yield
        on_prefilled(cache)
        row.pos = pos
        if deltas is not None:
            row.delta = int(deltas.reshape(-1)[0].item())
        row.y = mx.array(req.ids[-1:])
        self._join(row)

    def _spec(self, row):
        """Greedy text-only prefill + DFlash speculative decode for a lone row
        (a generator: one slab or one round per tick).

        Prefill goes through the inner Qwen3_5Model directly, capturing the
        hidden states of the drafter's target layers. Only the last
        sliding_window-1 prompt positions are kept: that is all the drafter's
        rotating cache looks at. Each round then drafts a block (greedy),
        verifies it with one target forward (mlx-vlm 0.7: speculative_verify,
        the batch-invariant rows = singleton decode, inside a cache
        transaction), keeps the longest matching prefix and commits the
        transaction to it (the rest of the block leaves every cache).

        Sampling is exact ("sample-match"): every verify row is sampled from the
        target distribution for that position - the penalties over the output
        plus the drafts before the row, then temperature/min-p/top-k/top-p -
        and drafts are kept while they equal the target's own sample. Each
        emitted token is therefore a target sample given an accepted prefix,
        i.e. the same distribution as the plain loop; temperature only lowers
        the acceptance rate.

        Demotion (another request waiting, max_batch > 1): after a round the
        target cache holds every token before the last emitted one, `b` —
        exactly a batch row whose next input is b, already emitted (`first`).
        """
        from mlx_vlm.speculative.cache_state import (abort_speculative_round,
                                                     commit_speculative_round)
        from mlx_vlm.speculative.common import (_dflash_block_total, _record_speculative_round,
                                                _speculative_walk)
        from mlx_vlm.speculative.dflash import _dflash_next_block_size
        req, lm, draft = row.req, self.lm, self.eng.draft
        input_ids, cache, pos, cuts, on_ckpt, on_prefilled = self._prompt_cache(row)
        row.cache = cache
        layers = list(draft.config.target_layer_ids)
        win = draft.config.sliding_window - 1
        argmax = lambda logits: mx.argmax(logits, axis=-1)  # noqa: E731
        sampling, sampler, pen, max_tokens = req.sampling, row.sampler, row.pen, req.max_tokens
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

        # The first lm() call below takes the delta branch (cache offset + 0);
        # with _rope_deltas None it would rotate the last prompt token as
        # position 0 (get_rope_index of a 1-token input).
        lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
        lm._position_ids = None
        ctx = input_ids[:, :-1]
        hidden = []
        while ctx.shape[1] > 0:
            if req.cancel.is_set():
                self._finish(row, "cancelled")
                return
            n = _slab(pos, ctx.shape[1], cuts)
            sink = []
            lm.model(ctx[:, :n], cache=cache, capture_layer_ids=layers, hidden_sink=sink)
            h = mx.concatenate(sink, axis=-1)
            hidden.append(h)
            while sum(t.shape[1] for t in hidden[:-1]) >= win:
                hidden.pop(0)
            mx.eval(h, [c.state for c in cache])
            pos += n
            if pos in cuts:
                on_ckpt(pos, cache)
            ctx = ctx[:, n:]
            if CLEAR_CACHE:
                mx.clear_cache()
            if ctx.shape[1] > 0:
                yield
        on_prefilled(cache)
        out = lm(input_ids[:, -1:], cache=cache, capture_layer_ids=layers)
        hidden.append(mx.concatenate(out.hidden_states, axis=-1))
        hidden = mx.concatenate(hidden, axis=1)[:, -win:]
        first = pick(out.logits[:, -1:, :], input_ids[0, -1:])
        mx.eval(first, hidden)
        first = first.item()
        req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1

        def emit(tok):
            if self.eng.stopping(tok):
                return False
            if pen.active:
                pen.add(mx.array([tok]))
            req.out.put(("tok", tok, True))
            return True

        finish, n, acc, drafted = "length", 0, 0, 0
        if max_tokens > 0 and not emit(first):
            finish = "stop"
        elif max_tokens > 0:
            n = 1
            # mlx-vlm 0.7.6 `_dflash_rounds`, with `pick` as the target sampler.
            block_total = _dflash_block_total(draft, DRAFT_BLOCK or None)
            draft_cache = draft.reset(self.model)
            b = first
            while n < max_tokens:
                yield
                if req.cancel.is_set():
                    finish = "cancelled"
                    break
                row.pos = cache[lm.model.fa_idx].offset
                self._regrow(row)
                head = self.waiting[0] if self.waiting else None
                if (head is not None and not head.cancel.is_set() and self.max_batch > 1
                        and self._fits(head, 1)):
                    # only for a request that is admitted next to this row:
                    # otherwise it waits anyway and DFlash's speed-up is lost
                    log.info("dflash: %d tokens, accept rate %.0f%% (%d/%d drafts); "
                             "demoted to a batch row (another request is waiting)",
                             n, 100.0 * acc / max(1, drafted), acc, drafted)
                    row.y, row.first, row.n = mx.array([b]), True, n
                    self._join(row)
                    return
                bs = _dflash_next_block_size(draft, block_total, max_tokens - n + 1)
                if bs <= 1:
                    break
                drafts = draft.draft_block(b, hidden, draft_cache, bs, argmax, dtype)
                mx.async_eval(drafts)
                fed = mx.concatenate([mx.array([[b]], dtype=dtype), drafts], axis=1)
                txn = None
                try:
                    # the verify forward opens a cache transaction (out.gdn_states)
                    # that must be committed every round, or aborted
                    out = lm(fed, cache=cache, capture_layer_ids=layers, speculative_verify=True)
                    txn = out.gdn_states
                    hidden = mx.concatenate(out.hidden_states, axis=-1)
                    targets = pick(out.logits, fed[0])
                    mx.async_eval(targets, hidden)
                    accepted, new = _speculative_walk(drafts, targets, max_tokens - n)
                    commit_speculative_round(lm, cache, txn, accepted, bs)
                except BaseException:
                    abort_speculative_round(txn)
                    raise
                _record_speculative_round(draft, accepted, bs - 1)
                acc, drafted = acc + accepted, drafted + bs - 1
                req.steps, req.rows_sum = req.steps + 1, req.rows_sum + 1
                stopped = False
                for tok in new:
                    if not emit(tok):
                        finish, stopped = "stop", True
                        break
                    n += 1
                if stopped or not new:
                    break
                b = new[-1]
                if accepted < bs - 1:
                    hidden = hidden[:, : accepted + 1, :]
                if n % 256 < len(new) and CLEAR_CACHE:
                    mx.clear_cache()
        if max_tokens > 1:
            log.info("dflash: %d tokens, accept rate %.0f%% (%d/%d drafts)",
                     n, 100.0 * acc / max(1, drafted), acc, drafted)
        self._finish(row, finish)

    def fit_max_batch(self):
        """max_batch, or 1 when the budget cannot hold two typical rows (a
        2K prompt + the reserved generation): then batching would only make
        admissions wait while gateways keep max_batch requests in flight."""
        typical = self._bytes(2048 + RESERVE_TOKENS)
        if self.max_batch > 1 and self.budget < 2 * typical:
            log.warning("batching off: KV budget %.2f GiB holds fewer than 2 rows of %.2f GiB "
                        "(raise QWEN_WIRED_LIMIT_GB / iogpu.wired_limit_mb or set QWEN_BATCH_MEM_GB)",
                        self.budget / GIB, typical / GIB)
            self.max_batch = 1
        return self.max_batch

    # --- the batch ----------------------------------------------------------------
    def _hidden_dtype(self):
        if self.hdtype is None:
            self.hdtype = self.lm.model.embed_tokens(mx.array([[0]])).dtype
        return self.hdtype

    def _join(self, row):
        """Add a prefilled row. The new stacked state is built first and
        swapped in only on success, so a failed join leaves the batch as it was."""
        self._compact()
        gdn = self.gdn
        if len(self.rows) == 1:
            gdn = _gdn_stack(self.lm, [self.rows[0].cache, row.cache], self._hidden_dtype())
        elif self.rows:
            gdn = _gdn_concat(self.gdn, _gdn_stack(self.lm, [row.cache], self._hidden_dtype()))
        self.gdn = gdn
        self.rows.append(row)
        self._regrow(row)

    def _compact(self):
        keep = [i for i, r in enumerate(self.rows) if not r.done]
        if len(keep) == len(self.rows):
            return
        if self.gdn is not None:
            if len(keep) == 1:                  # back to the plain library call
                _gdn_split(self.gdn, keep[0], self.rows[keep[0]].cache)
                self.gdn = None
            else:
                self.gdn = _gdn_take(self.gdn, keep) if keep else None
        self.rows = [self.rows[i] for i in keep]

    def _end(self, row, event):
        """The row's last event, with its stats snapshot taken here (the next
        admission may reset the peak before the handler reads it)."""
        row.done = True
        self.reserved -= row.reserved
        req = row.req
        req.peak, req.active, req.pool = (mx.get_peak_memory(), mx.get_active_memory(),
                                          mx.get_cache_memory())
        req.batch_mean = req.rows_sum / req.steps if req.steps else 0.0
        req.out.put(event)
        # give a long row's KV back to the OS; clearing after every short row
        # would only make the next step allocate fresh buffers
        if row.pos > 8192 or all(r.done for r in self.rows):
            self.clear = True

    def _finish(self, row, why):
        if not row.done:
            self._end(row, ("done", why))

    def _fail(self, row, e):
        if row is None or row.done:
            return
        log.error("generation failed: %s: %s: %s", row.req.id, type(e).__name__, e)
        self._end(row, ("error", type(e).__name__, str(e)))

    def _fail_rows(self, e):
        """A failed decode step: its rows fail, a prefill in progress goes on
        (its caches are its own)."""
        for r in self.rows:
            self._fail(r, e)
        self.rows, self.gdn = [], None
        self.reserved = self.task_row.reserved if self.task_row is not None else 0
        try:
            mx.clear_cache()
        except Exception:  # noqa: BLE001 — the failure is already reported
            log.exception("clear_cache after a failed step")

    def _step(self, rows):
        """Queue one decode forward for `rows` -> their next tokens (lazy (1,)
        arrays; None where the row's sampling failed)."""
        lm = self.lm
        B = len(rows)
        if B == 1:
            r = rows[0]
            out = lm(r.y[None], cache=r.cache,
                     position_ids=mx.array([[r.pos + r.delta]], dtype=mx.int32))
            logits = [out.logits[:, -1, :]]
        else:
            # Qwen3_5GatedDeltaNet silently restarts from zeros on a batch-size
            # mismatch: never let a stacking slip through
            if any(c.cache[1].shape[0] != B for c in self.gdn.values()):
                raise RuntimeError("stacked recurrent state out of step with the batch")
            cache = [self.gdn[i] if i in self.gdn else RowsKV([r.cache[i] for r in rows])
                     for i in range(len(lm.model.layers))]
            pos = mx.array([[r.pos + r.delta] for r in rows], dtype=mx.int32)
            out = lm(mx.concatenate([r.y for r in rows])[:, None], cache=cache, position_ids=pos)
            lg = out.logits[:, -1, :]
            logits = [lg[i:i + 1] for i in range(B)]
        nxt = []
        pend = []                                   # (index in nxt, row, logits): drawn together
        for r, x in zip(rows, logits):
            r.pos += 1
            self._regrow(r)
            if r.done:                              # ended by _regrow
                nxt.append(None)
                continue
            r.req.steps, r.req.rows_sum = r.req.steps + 1, r.req.rows_sum + B
            try:
                if B > 1 and r.req.guide is None:
                    pend.append((len(nxt), r, self._logits(r, x)))
                    nxt.append(None)
                else:
                    nxt.append(self._sample(r, x))
            except Exception as e:  # noqa: BLE001 — e.g. a grammar error: this row only
                self._fail(r, e)
                nxt.append(None)
        if pend:                                    # a drawing error fails its group's rows only
            ys = sample_rows([r.sampler for _, r, _ in pend], [x for _, _, x in pend],
                             on_error=lambda i, e: self._fail(pend[i][1], e))
            for (j, _, _), y in zip(pend, ys):
                nxt[j] = y
        mx.async_eval([y for y in nxt if y is not None])
        self.log_rows += B
        return nxt

    def _logits(self, row, logits):
        """One row's logits before the draw: penalties, token masks (in
        vLLM's order), then the KV quantization check."""
        req = row.req
        if row.pen.active:
            if not row.first:                       # y is a generated token, not the prompt
                row.pen.add(row.y)
            logits = row.pen(logits)
        if req.ctl.active:
            if not row.first:
                req.ctl.n += 1
            logits = req.ctl.mask(logits)
        self.quantize(row.cache)
        return logits

    def _sample(self, row, logits):
        """One row's logits -> its next token: _logits, then grammar, sampler.
        (In a batch, rows without a grammar are drawn together: sample_rows.)"""
        logits = self._logits(row, logits)
        if row.req.guide is not None:
            return row.req.guide.sample(logits, row.sampler)
        return row.sampler(logits)

    def _iterate(self):
        """One decode step for the batch. Step-ahead: the next forward is
        queued before this step's tokens are read, so the host work overlaps
        the GPU; a row with a grammar or a multi-token bad word needs each
        token before its next mask, so while one is in the batch the whole
        batch reads first and steps after."""
        for r in self.rows:
            if r.req.cancel.is_set():
                self._finish(r, "cancelled")
            elif r.n == r.req.max_tokens:          # the pending token is never read
                self._finish(r, "length")
        self._compact()
        if not self.rows:
            return
        rows = self.rows
        ahead = not any(r.sync for r in rows)
        nxt = self._step(rows) if ahead else None
        live = [r for r in rows if not r.first and not r.done]
        # item() per row: each token was evaluated by its step's async_eval,
        # so this waits for that step only. Any new op on them (a concatenate
        # + tolist) would be queued behind the step just submitted and wait
        # for it too: the GPU then idled while the host built the next step
        # (measured −5…12% step time at B = 8/16, step_profile).
        toks = [r.y.item() for r in live]
        for r, tok in zip(live, toks):
            req = r.req
            try:
                if req.thinking is not None:
                    req.thinking(tok)
                why = req.ctl.stop(tok) if req.ctl.active else (
                    "stop" if self.eng.stopping(tok) else None)
                if why == "stop":
                    self._finish(r, "stop")
                    continue
                if not ahead:
                    if req.guide is not None:
                        req.guide.consume(tok)
                    req.ctl.history.append(tok)
                req.out.put(("tok", tok, why != "skip"))   # an ignored EOS is not text
                r.n += 1
            except Exception as e:  # noqa: BLE001 — e.g. grammar rejected the token
                self._fail(r, e)
        if not ahead:
            self._compact()
            rows = self.rows
            nxt = self._step(rows) if rows else []
        for r, y in zip(rows, nxt):
            if not r.done:
                # Reasoning budget: forces \n</think> as the next input (the
                # KV cache stays consistent)
                forced = r.req.thinking.pop_forced_token_id() if r.req.thinking is not None else None
                r.y = y if forced is None else mx.array([forced], dtype=y.dtype)
                r.first = False
        self.steps += 1
        self.failures = 0                       # consecutive failed decode steps
        if self.steps % 256 == 0:
            self.clear = True
        self._compact()

    def _log(self):
        now = time.time()
        if now - self.log_at < SCHED_LOG_S:
            return
        steps = self.steps - self.log_steps
        if steps or self.rows or self.waiting or self.task:
            log.info("batch: rows=%d waiting=%d prefilling=%s steps/s=%.1f mean_rows=%.2f "
                     "reserved=%.2f/%.2fGiB active=%.2fGiB",
                     len(self.rows), len(self.waiting),
                     self.task_row.req.id if self.task_row is not None else "-",
                     steps / (now - self.log_at), self.log_rows / steps if steps else 0.0,
                     self.reserved / GIB, self.budget / GIB, mx.get_active_memory() / GIB)
        self.log_at, self.log_steps, self.log_rows = now, self.steps, 0


class Engine:
    """Owns the model; requests go through its _Scheduler."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        # the id clients see (/v1/models, "model" in responses): the weights dir name
        self.model_id = os.path.basename(os.path.normpath(model_path))
        self.ready = False
        self.load_ms = 0
        self.max_batch = MAX_BATCH
        self.sched = None
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
            # mlx-vlm >= 0.7 drops the weightless vision tower before its strict
            # load (vision_tower = None); image requests are refused in _render.
            log.warning("no vision_config: text-only checkpoint (no vision tower; "
                        "image requests will be refused)")
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
        self.stopping = crit
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
        lm = self.model.language_model
        global _FIXED_SLAB
        log.info("prefill kernels: %s", _install_fast_paths())
        _FIXED_SLAB = _fixed_slab()            # after the fused-SDPA probe; never changes after
        log.info("prefill slab: %s", f"{PREFILL_STEP} at any context (fp16 KV + fused SDPA)"
                 if _FIXED_SLAB else f"{PREFILL_STEP}, shrunk to slab x offset <= {PREFILL_BUDGET}")
        if not _install_rows_attention(lm):
            log.warning("%s is not a qwen3_5 / qwen3_5_moe model: no batching (QWEN_MAX_BATCH=1)",
                        type(lm).__module__)
            self.max_batch = 1
        import pfqmm                       # after the fp16 cast (fp16 scales only); LM only
        pfqmm.install(lm)
        if self.max_batch >= 3:
            # after the fp16 cast (the kernel takes fp16 scales only); the
            # language model only (never the vision tower or the drafter,
            # which shares lm_head but always calls it with (1, T, K)).
            # Logs its coverage, or "off" for QWEN_MRQMM=0.
            import mrqmm
            mrqmm.install(lm)
        log.info("GDN decode in-projections fused (QWEN_GDN_FUSE): %d layers", _install_gdn_fuse(lm))
        cap = _batch_cap(int(mx.device_info()["max_recommended_working_set_size"]), _WIRED_GB)
        reserve = _prefill_reserve(lm.args.num_attention_heads)
        budget = _batch_budget(cap, mx.get_active_memory(), reserve)   # weights (+ drafter) = active now
        self.sched = _Scheduler(self, max_batch=self.max_batch, budget=budget, cap=cap,
                                prefill_reserve=reserve, on_fault=_exit_soon)
        self.max_batch = self.sched.fit_max_batch()
        log.info("batching: max_batch %d, KV budget %.2f GiB (cap %.1f GiB, prefill reserve %.1f GiB), "
                 "KV %d B/token, recurrent state %.1f MiB/row, reserve <= %d new tokens/row, "
                 "prefill share %.2f",
                 self.max_batch, budget / GIB, cap / GIB, reserve / GIB, self.sched.kv_q,
                 self.sched.gdn_row / 2**20, RESERVE_TOKENS, PREFILL_SHARE)
        self.sched.start()
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

    def _unregister(self, req_id: str, ev: threading.Event):
        with self._cancels_lock:
            if self._cancels.get(req_id) is ev:     # a newer request may reuse the id
                del self._cancels[req_id]

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
        # the ids index vocab-sized masks, and MLX indexing has no bounds check
        for name in ("allowed_token_ids", "stop_token_ids"):
            if any(t >= self.vocab for t in getattr(p, name)):
                raise _Reject("invalid_request", f"{name} contains out-of-vocab token id "
                              f"(vocab size {self.vocab})")
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
        # mlx-vlm >= 0.6.3 passes enable_thinking=False when it is unset and the
        # template uses it (vlm#1316); unset means the template's default
        # (thinking), as in vLLM. The request's own value wins.
        if "enable_thinking" in self._template_vars:
            kwargs.setdefault("enable_thinking", True)
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
        """What Generate would reject before generating, without the model
        (text prompts are tokenized to check the length; image prompts are
        counted by Generate, before queueing, not here: Check stays cheap)."""
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
    def _vision_inputs(self, formatted, images):
        """Handler thread: an image prompt through mlx-vlm 0.6.2's
        prepare_inputs minus the MLX part (the path the qwen3-vl processor
        takes): images loaded (path / URL / data URI, RGB), then the processor
        tokenizes and preprocesses -> {name: numpy}. The scheduler wraps the
        arrays as prepare_inputs does (_vision_arrays), so a slow image URL or
        a large decode never stalls the decoding batch. The processor is numpy
        throughout, but its to_mlx wraps the results as constant MLX arrays on
        this thread (allocations only: no graph, no stream, no evaluation);
        they come back to numpy here. padding=False (one prompt, the same
        output): padding would switch the shared Rust tokenizer's padding
        state, racing other handlers' encode ("Already borrowed"). The id
        count is exact, image tokens included (a 4K image is ~10k tokens vs
        ~15 for the text placeholder): there is deliberately no text-only
        fallback, an undercount would let an oversized prompt through to an
        OOM."""
        from mlx_vlm.utils import process_image, process_inputs_with_fallback
        ip = getattr(self.processor, "image_processor", None)
        pil = [process_image(img, None, ip) for img in images]
        inputs = process_inputs_with_fallback(self.processor, prompts=formatted, images=pil,
                                              audio=None, add_special_tokens=True, padding=False)
        if "images" in inputs:
            inputs["pixel_values"] = inputs.pop("images")
        return {k: np.array(v) if isinstance(v, mx.array) else v for k, v in inputs.items()}

    # --- generation (gRPC handler thread) ----------------------------------------
    def _event(self, r):
        """The scheduler's next event for r. A dead or wedged scheduler ends
        the stream with an Error instead of hanging it."""
        while True:
            try:
                return r.out.get(timeout=EVENT_POLL_S)
            except queue.Empty:
                if not self.sched.alive():
                    return ("error", "worker_fault", "the scheduler is not running")

    def _drain(self, r):
        """Wait for r's last event after a cancel / stop hit (the row is
        evicted at the next step) so its stats are the scheduler's."""
        while self._event(r)[0] == "tok":
            pass

    def generate(self, req: pb.Request, context=None):
        """Yield pb.Event for one request. Rendering, tokenizing and the text
        work per token (detokenizer, reasoning split, stop strings, tool
        parsing) happen here, on the handler thread; the model runs on the
        scheduler thread, which streams token ids back."""
        p = req.params
        t_arrival = time.time()
        cancel_ev = self._register(req.id)
        if context is not None:
            context.add_callback(cancel_ev.set)       # a dropped stream frees its row
        tools = json.loads(p.tools_json) if p.tools_json else []
        tmp_files = []
        try:
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
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if cancel_ev.is_set():                    # gone already: no image work
                yield pb.Event(id=req.id, done=pb.Done(finish="cancelled", usage=pb.Usage()))
                return
            # Tokenize once (images preprocessed once), reject prompts that
            # don't fit, and clamp the generation to the remaining budget.
            try:
                vision = self._vision_inputs(formatted, images) if images else None
                ids = (vision["input_ids"].reshape(-1).tolist() if images
                       else tokenizer.encode(formatted))
            except Exception as e:  # noqa: BLE001 — never proceed un-measured
                log.exception("prompt preparation failed: %s", req.id)
                yield pb.Event(id=req.id, error=pb.Error(
                    kind="prompt_count_failed", msg=f"could not measure prompt length: {e}"))
                return
            log.info("id=%s prompt_tokens=%d (images=%d)", req.id, len(ids), len(images))
            try:
                max_tokens = self._limit(p, len(ids))
            except _Reject as e:
                yield pb.Event(id=req.id, error=pb.Error(kind=e.kind, msg=str(e)))
                return

            # Thinking = the prompt leaves <think> open (the template does that
            # unless enable_thinking is false).
            inside = next((t == self._think[0] for t in reversed(ids) if t in self._think), False)
            state = _GenState(inside)
            state.counting = inside

            # Reasoning budget: forces \n</think> into the token stream when
            # exceeded (the KV cache stays consistent). Same criteria object
            # mlx-vlm's stream_generate builds.
            thinking = None
            if inside and p.HasField("thinking_budget"):
                from mlx_vlm.utils import ThinkingBudgetCriteria
                thinking = ThinkingBudgetCriteria(
                    tokenizer=tokenizer, thinking_budget=p.thinking_budget,
                    thinking_end_token=THINK_CLOSE, thinking_start_token=THINK_OPEN,
                    enable_thinking=True, prompt_preopens_thinking=True)

            sampling = dict(SAMPLING_KWARGS)
            for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                      "frequency_penalty", "repetition_penalty", "seed"):
                if p.HasField(f):
                    sampling[f] = getattr(p, f)
            ctl = _Control(p, tokenizer.stopping_criteria.eos_token_ids, tokenizer, self.vocab)

            guide = None
            if grammar is not None:
                guide = _Guide(self._llg_tok, grammar, think_close=self._think[1],
                               active=not inside, ws=None if tools else self._not_ws,
                               tool_open=self._tool_tags[0] if tools and self._tool_tags else None)
                log.info("id=%s constrained: response_format=%s tool_choice=%s%s", req.id,
                         json.loads(p.response_format_json).get("type")
                         if p.response_format_json else "-", p.tool_choice or "auto",
                         "" if guide.active else " (after </think>)")

            if cancel_ev.is_set():                    # gone while preparing: never queued
                yield pb.Event(id=req.id, done=pb.Done(
                    finish="cancelled", usage=pb.Usage(prompt_tokens=len(ids))))
                return
            r = _Req(id=req.id, p=p, ids=ids, max_tokens=max_tokens, sampling=sampling, ctl=ctl,
                     guide=guide, thinking=thinking, formatted=formatted, images=images,
                     vision=vision, cancel=cancel_ev, t_arrival=t_arrival)
            self.sched.submit(r)

            detok = make_streaming_detokenizer(self.processor)
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

            ttft_ms = 0
            while True:
                ev = self._event(r)
                ended = ev[0] != "tok"
                if ev[0] == "error":
                    yield pb.Event(id=req.id, error=pb.Error(kind=ev[1], msg=ev[2]))
                    return
                if ev[0] == "done":
                    finish = ev[1]
                    if finish == "cancelled":
                        yield _done(r, "cancelled", state, ttft_ms)
                        return
                    detok.finalize()
                    # не last_segment: он придерживает текст, кончающийся на U+FFFD
                    tok, text = None, detok.text[detok.offset:]
                    if not text:
                        break
                else:
                    tok = ev[1]
                    if ev[2]:                         # an ignored EOS is not text
                        detok.add_token(tok)
                    text = detok.last_segment
                if cancel_ev.is_set():
                    if not ended:
                        self._drain(r)
                    yield _done(r, "cancelled", state, ttft_ms)
                    return
                if ttft_ms == 0:
                    ttft_ms = int((time.time() - r.t0) * 1000)
                if tok is not None:
                    state.token(tok, self._think[1], self._tool_tags[0] if self._tool_tags else -1)
                # vLLM: stop strings apply to the whole output, reasoning included,
                # and only once min_tokens are out
                yield from split(stop.feed(text) if state.gen_tokens > p.min_tokens else text)
                if stop.hit:
                    cancel_ev.set()                   # the scheduler evicts the row
                    if not ended:
                        self._drain(r)
                    finish = "stop"
                    break
                if ev[0] == "done":
                    break
            if not stop.hit:
                yield from split(stop.flush())

            gen_s = time.time() - r.t0 - ttft_ms / 1000
            tps = state.gen_tokens / gen_s if gen_s > 0 else 0.0
            log.info("id=%s finish=%s gen_tokens=%d ttft_ms=%d tps=%.1f cached=%d "
                     "peak=%.2fGB active=%.2fGB pool=%.2fGB queue_ms=%d batch=%.2f",
                     req.id, finish, state.gen_tokens, ttft_ms, tps, r.reused,
                     r.peak / 1e9, r.active / 1e9, r.pool / 1e9, r.queue_ms, r.batch_mean)

            if cut:
                log.info("id=%s no tools: <tool_call> in the answer, it and %d chars after "
                         "it not returned (as vLLM)", req.id, len("".join(cut)))
            calls = []
            if tools:
                answer = "".join(answer_buf)
                key = hashlib.sha256(f"{r.ids}|{sampling.get('seed')}|{answer}".encode()).hexdigest()
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
            yield _done(r, finish, state, ttft_ms, calls, tps)
        except Exception as e:  # noqa: BLE001 — map any failure to a wire Error
            log.exception("generation failed: %s", req.id)
            yield pb.Event(id=req.id, error=pb.Error(kind=type(e).__name__, msg=str(e)))
        finally:
            cancel_ev.set()          # ended early (client gone, error): free the row
            self._unregister(req.id, cancel_ev)
            for f in tmp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass


class _Control:
    """vLLM's token-level request options: allowed_token_ids, bad_words, min_tokens
    (EOS and stop_token_ids masked until then), stop_token_ids, ignore_eos.
    `sync`: a multi-token bad word needs the tokens so far when the mask is built,
    so the decode loop does not step ahead (as with a grammar).

    Two halves: __init__ on the handler thread (the bad words' tokenization,
    no MLX), build() on the scheduler (the vocab masks)."""

    def __init__(self, p, eos, tokenizer, vocab):
        self.eos = set(eos)
        self.stop_ids = set(p.stop_token_ids)
        self.ignore_eos = p.ignore_eos
        self.min_tokens = p.min_tokens
        self.vocab = vocab
        self.n = 0
        self.history = []
        self.allowed = list(p.allowed_token_ids)
        self.seqs = []                         # (prefix, last) of multi-token bad words
        self.single = set()
        for w in p.bad_words:                  # vLLM: the word and " word"
            for v in {w, " " + w.lstrip()}:
                ids = tokenizer.encode(v, add_special_tokens=False)
                if len(ids) == 1:
                    self.single.add(ids[0])
                elif ids:
                    self.seqs.append((ids[:-1], ids[-1]))
        self.base = self.early = None
        self.sync = bool(self.seqs)
        self.active = bool(self.allowed or self.single or self.min_tokens or self.sync
                           or self.stop_ids or self.ignore_eos)

    def build(self):
        base = None
        if self.allowed:
            base = mx.zeros((self.vocab,), dtype=mx.bool_)
            base[mx.array(self.allowed)] = True
        if self.single:
            if base is None:
                base = mx.ones((self.vocab,), dtype=mx.bool_)
            base[mx.array(sorted(self.single))] = False
        self.base = base
        if self.min_tokens:
            self.early = mx.ones((self.vocab,), dtype=mx.bool_)
            self.early[mx.array(sorted(self.eos | self.stop_ids))] = False

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



def _done(r, finish, state, ttft_ms, tool_calls=None, tps=0.0):
    usage = pb.Usage(
        prompt_tokens=r.prompt_tokens,
        gen_tokens=state.gen_tokens,
        reasoning_tokens=state.reasoning_tokens,
        tps=tps,
        ttft_ms=ttft_ms,
        peak_gb=r.peak / 1e9,                    # taken by the scheduler when the request ended
        queue_ms=r.queue_ms,
        batch_mean=r.batch_mean,
    )
    return pb.Event(id=r.id, done=pb.Done(
        finish=finish, usage=usage, tool_calls=tool_calls or []))


class WorkerServicer(pb_grpc.WorkerServicer):
    def __init__(self, engine: Engine, stop_event: threading.Event, slots: int):
        self.engine = engine
        self.stop_event = stop_event
        self.slots = slots
        self.free = threading.BoundedSemaphore(slots)

    def Generate(self, request, context):
        p = request.params
        if not self.free.acquire(blocking=False):
            log.warning("Generate id=%s refused: %d generations in flight", request.id, self.slots)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                          f"worker busy: {self.slots} generations in flight")
        try:
            log.info("Generate id=%s images=%d max_tokens=%s template_kwargs=%s budget=%s",
                     request.id, len(request.images),
                     p.max_tokens if p.HasField("max_tokens") else "-",
                     p.chat_template_kwargs_json or "-",
                     p.thinking_budget if p.HasField("thinking_budget") else "-")
            yield from self.engine.generate(request, context)
        finally:
            self.free.release()     # also when grpc closes the stream (client gone)

    def Cancel(self, request, context):
        self.engine.cancel(request.id)
        return pb.Ack()

    def Health(self, request, context):
        return pb.HealthReply(ready=self.engine.ready and self.engine.sched.alive(),
                              model=self.engine.model_id,
                              load_ms=self.engine.load_ms,
                              model_path=self.engine.model_path,
                              max_batch=self.engine.max_batch)

    def Check(self, request, context):
        # Runs beside the generations (own pool thread): template, tokenizer and
        # grammar checks, nothing that touches MLX.
        error, internal = self.engine.check(request)
        return pb.CheckReply(error=error, internal=internal, model=self.engine.model_id)

    def Shutdown(self, request, context):
        log.warning("shutdown requested")
        self.stop_event.set()
        return pb.Ack()


def serve(engine, addr, stop_event):
    """gRPC server for a loaded engine -> (server, servicer, port). Generate RPCs
    beyond QWEN_GEN_SLOTS (default max(2*max_batch, 8)) are refused with
    RESOURCE_EXHAUSTED; the pool has 8 threads more for Check / Cancel /
    Health. grpc itself refuses any RPC past slots + 64 queued (a backstop:
    only a burst of > 64 Checks at once could then refuse a Health poll).
    Vision prompts carry base64 image data URIs: a detailed map snapshot is
    several MB, past gRPC's 4 MB default receive cap, so the message limits
    are lifted."""
    slots = max(1, GEN_SLOTS or max(2 * engine.max_batch, 8))
    if slots < engine.max_batch:
        log.warning("QWEN_GEN_SLOTS %d < max_batch %d: the batch can never fill", slots,
                    engine.max_batch)
    max_msg = 64 * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=slots + 8),
        maximum_concurrent_rpcs=slots + 64,
        options=[
            ("grpc.max_receive_message_length", max_msg),
            ("grpc.max_send_message_length", max_msg),
        ],
    )
    servicer = WorkerServicer(engine, stop_event, slots)
    pb_grpc.add_WorkerServicer_to_server(servicer, server)
    port = server.add_insecure_port(addr)
    server.start()
    log.info("worker serving on %s (%d generation slots, %d threads)", addr, slots, slots + 8)
    return server, servicer, port


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
    # plain `kill` (launchd, README) stops like Shutdown / Ctrl-C
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    server, _, _ = serve(engine, args.addr, stop_event)
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        # rows and queued requests get Error(shutdown) before the RPCs close
        if not engine.sched.stop(timeout=10.0):
            # a tick still inside MLX: interpreter teardown under it can crash
            log.error("scheduler did not stop within 10 s; exiting now")
            server.stop(0)
            os._exit(1)
        server.stop(grace=2).wait()
        log.info("worker stopped")


if __name__ == "__main__":
    main()

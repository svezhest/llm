"""Structured output: response_format / tool-call grammars (llguidance)."""

import json
import os

import numpy as np

from .chat import _schema_types


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

    def sample(self, logits, sampler, pos=None):
        """The masked draw: `sampler` on the logits the grammar allows, its
        key the token's absolute position `pos`."""
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
        return sampler(llguidance.mlx.apply_token_bitmask(logits, mask), pos)


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

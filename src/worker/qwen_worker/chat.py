"""Request side: tool-call parsing (vLLM qwen3_xml), images, chat messages and the
prompt tokenization with QWEN_SPLIT_SPECIAL_TOKENS."""

import base64
import hashlib
import json
import math
import os
import re
import tempfile

import worker_pb2 as pb

from .config import log


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


class _PlainContent:
    """Tokenizes a rendered prompt with the added tokens that came from
    message text kept as plain text (QWEN_SPLIT_SPECIAL_TOKENS).

    The messages are rendered a second time with every added-token string in
    the protected text replaced by a placeholder (a private-use character
    absent from the prompt + one naming the token). Restoring the
    placeholders must give the normal rendering back, character for
    character (else the template looked at those strings: the caller falls
    back to the normal tokenization). The placeholder rendering is then cut
    at the added tokens left in it — the template's own — and the text
    between two of them (template text and message text together, the
    placeholders restored) is encoded by a copy of the tokenizer without
    its added vocabulary: what the tokenizer itself does between added
    tokens (split on them, then normalize / pre-tokenize / BPE each piece),
    so a prompt with nothing to protect gets the same ids."""

    _BASE = 0xF0000                       # plane-15 private use: one char per added token

    def __init__(self, tok):
        import tokenizers
        bt = tok.backend_tokenizer
        added = bt.get_added_tokens_decoder()
        odd = [a.content for a in added.values()
               if a.lstrip or a.rstrip or a.single_word or a.normalized]
        if odd:
            # matched differently from a plain leftmost-longest literal split
            raise ValueError(f"added tokens with strip / single_word / normalized flags: {odd}")
        self.tokens = sorted({a.content for a in added.values()})
        self.ids = {a.content: i for i, a in added.items()}
        self.index = {s: k for k, s in enumerate(self.tokens)}
        # leftmost, longest first (the tokenizer's own added-token match)
        self.match = re.compile("|".join(re.escape(s) for s in
                                         sorted(self.tokens, key=len, reverse=True)))
        if tok.encode(""):
            # ids the tokenizer adds around every text (a BOS): not reproduced here
            raise ValueError(f"the tokenizer adds {tok.encode('')} to every text")
        js = json.loads(bt.to_str())
        js["added_tokens"] = []
        self.plain = tokenizers.Tokenizer.from_str(json.dumps(js))

    def _sub(self, s, mark):
        return self.match.sub(lambda m: mark + chr(self._BASE + self.index[m.group()]), s)

    def _deep(self, x, mark):
        if isinstance(x, str):
            return self._sub(x, mark)
        if isinstance(x, dict):
            return {self._deep(k, mark): self._deep(v, mark) for k, v in x.items()}
        if isinstance(x, list):
            return [self._deep(v, mark) for v in x]
        return x

    def mask(self, messages, mark):
        """-> (messages with the protected strings' added tokens as
        placeholders, how many were replaced)."""
        n, out = 0, []
        for m in messages:
            m = dict(m)
            if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str):
                n += len(self.match.findall(m["content"]))
                m["content"] = self._sub(m["content"], mark)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    f = c.get("function") if isinstance(c, dict) else None
                    args = f.get("arguments") if isinstance(f, dict) else None
                    as_str = isinstance(args, str)
                    if as_str:
                        # parsed: a string may spell the tokens JSON-escaped
                        # (the template renders the parsed values)
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = None               # rendered as {} anyway
                    if isinstance(args, (dict, list)):
                        masked = self._deep(args, mark)
                        if masked != args:
                            n += 1
                            c = dict(c, function=dict(f, arguments=json.dumps(
                                masked, ensure_ascii=False) if as_str else masked))
                    calls.append(c)
                m["tool_calls"] = calls
            out.append(m)
        return out, n

    def mark_for(self, text):
        """A private-use character the prompt does not hold."""
        for c in range(0xE000, 0xF900):
            if chr(c) not in text:
                return chr(c)
        raise ValueError("no free private-use character")

    def encode(self, formatted, masked, mark):
        """-> the ids, or None when the placeholder rendering is not the
        normal one with placeholders (the template treated them differently)."""
        back = re.compile(re.escape(mark) + "(.)", re.S)

        def restore(s):
            return back.sub(lambda m: self.tokens[ord(m.group(1)) - self._BASE], s)
        if restore(masked) != formatted:
            return None
        ids, pos = [], 0
        for m in self.match.finditer(masked):
            if m.start() > pos:
                ids += self.plain.encode(restore(masked[pos:m.start()]),
                                         add_special_tokens=False).ids
            ids.append(self.ids[m.group()])
            pos = m.end()
        if pos < len(masked):
            ids += self.plain.encode(restore(masked[pos:]), add_special_tokens=False).ids
        return ids


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

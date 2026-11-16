"""Handler side of a generation: detokenizer, stop strings, request controls,
the reasoning / answer split and the Done event."""

import mlx.core as mx
from mlx_vlm.tokenizer_utils import BPEStreamingDetokenizer, _remove_space

import worker_pb2 as pb

from .chat import THINK_CLOSE, THINK_OPEN, TOOL_OPEN


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
        self.counting = thinking                  # token(): reasoning tokens still being counted
        self.gen_tokens = 0
        self.reasoning_tokens = 0

    @classmethod
    def for_prompt(cls, ids, think):
        """The state generation starts in: thinking = the prompt leaves <think>
        open (the last of `think` = [<think>, </think>] ids in it is the open
        one; the template pre-opens it unless enable_thinking is false).
        `think` None (ids unknown): never thinking."""
        return cls(bool(think) and next((t == think[0] for t in reversed(ids) if t in think),
                                        False))

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
        cached_tokens=r.reused,
    )
    return pb.Event(id=r.id, done=pb.Done(
        finish=finish, usage=usage, tool_calls=tool_calls or []))

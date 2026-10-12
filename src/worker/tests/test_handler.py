"""Handler-side pieces (no model): the per-token text work of Engine.generate
on a scripted scheduler stream, stop strings, _Control's two halves, and the
image temp-file cleanup."""
import base64
import os
import tempfile
import time
import types

import mlx.core as mx
import pytest

import worker
import worker_pb2 as pb

OPEN, CLOSE, EOS, TC_OPEN, TC_CLOSE = 1, 2, 3, 4, 5
SPECIAL = {OPEN: "<think>", CLOSE: "</think>", EOS: "", TC_OPEN: "<tool_call>",
           TC_CLOSE: "</tool_call>"}


def text_of(t):
    return SPECIAL.get(t, chr(t))


class Detok:
    def __init__(self):
        self.reset()

    def reset(self):
        self.text, self.offset = "", 0

    def add_token(self, t):
        self.text += text_of(t)

    @property
    def last_segment(self):
        s, self.offset = self.text[self.offset:], len(self.text)
        return s

    def finalize(self):
        pass


class Tok:
    stopping_criteria = types.SimpleNamespace(eos_token_ids=[EOS])

    def __init__(self, prompt):
        self.prompt = prompt

    def encode(self, s, add_special_tokens=True):
        return list(self.prompt) if add_special_tokens else [ord(c) for c in s]


class Sched:
    def __init__(self, script):
        self.script, self.req = script, None

    def alive(self):
        return True

    def submit(self, r):
        self.req = r
        r.t0 = time.time()
        for ev in self.script:
            r.out.put(ev)


def engine(script, prompt=(100, 101, OPEN)):
    eng = worker.Engine("/tmp/model-x")
    eng.processor = types.SimpleNamespace(tokenizer=Tok(prompt), detokenizer=Detok())
    eng._think = [OPEN, CLOSE]
    eng._tool_tags = [TC_OPEN, TC_CLOSE]
    eng.vocab, eng._llg_tok, eng._not_ws = 256, None, None
    eng._render = lambda req, images: ("prompt", None)
    eng.sched = Sched(script)
    return eng


def toks(s):
    return [("tok", ord(c), True) for c in s]


def events(eng, **params):
    req = pb.Request(id="q", params=pb.Params(**params))
    evs = list(eng.generate(req))
    think = "".join(e.token.text for e in evs if e.HasField("token") and e.token.thinking)
    answer = "".join(e.token.text for e in evs if e.HasField("token") and not e.token.thinking)
    end = evs[-1]
    return think, answer, end


def test_reasoning_split_and_usage():
    eng = engine(toks("ab") + [("tok", CLOSE, True)] + toks("xy") + [("done", "stop")])
    think, answer, end = events(eng)
    assert (think, answer) == ("ab", "xy")
    assert end.done.finish == "stop"
    u = end.done.usage
    assert (u.prompt_tokens, u.gen_tokens, u.reasoning_tokens) == (3, 5, 2)


def test_stop_string_across_tokens_evicts():
    eng = engine(toks("xyzw") + [("done", "length")], prompt=(100,))
    think, answer, end = events(eng, stop=["yz"])
    assert (think, answer, end.done.finish) == ("", "x", "stop")
    assert eng.sched.req.cancel.is_set()             # the scheduler drops the row


def test_ignored_eos_and_min_tokens():
    eng = engine(toks("ab") + [("tok", EOS, False)] + toks("cd") + [("done", "length")],
                 prompt=(100,))
    _, answer, end = events(eng, ignore_eos=True)
    assert answer == "abcd" and end.done.usage.gen_tokens == 5
    # stop strings only count once min_tokens are out
    eng = engine(toks("abab") + [("done", "length")], prompt=(100,))
    _, answer, end = events(eng, stop=["ab"], min_tokens=2)
    assert (answer, end.done.finish) == ("ab", "stop")


def test_cancel_and_error():
    eng = engine(toks("ab") + [("done", "cancelled")], prompt=(100,))
    _, _, end = events(eng)
    assert end.done.finish == "cancelled"
    eng = engine(toks("ab") + [("error", "RuntimeError", "boom")], prompt=(100,))
    _, answer, end = events(eng)
    assert answer == "ab" and end.error.kind == "RuntimeError" and end.error.msg == "boom"


def test_tool_calls():
    call = "\n<function=get>\n<parameter=n>\n3\n</parameter>\n</function>\n"
    script = toks("ok ") + [("tok", TC_OPEN, True)] + toks(call) + \
        [("tok", TC_CLOSE, True), ("done", "stop")]
    tools = '[{"type":"function","function":{"name":"get","parameters":' \
            '{"type":"object","properties":{"n":{"type":"integer"}}}}}]'
    eng = engine(script, prompt=(100,))
    _, answer, end = events(eng, tools_json=tools)
    assert answer == "ok" and end.done.finish == "tool_calls"
    (c,) = end.done.tool_calls
    assert (c.name, c.arguments) == ("get", '{"n": 3}')


def test_context_limit_rejected_before_queueing():
    eng = engine([], prompt=(100,) * 10)
    _, _, end = events(eng, max_tokens=worker.MAX_CONTEXT)
    assert end.error.kind == "context_length_exceeded" and eng.sched.req is None


def test_stop_filter():
    f = worker._StopFilter(["abc", "bd"])
    assert f.feed("xa") == "x" and f.feed("b") == "" and f.feed("d") == "a" and f.hit
    f = worker._StopFilter(["END"], keep=True)
    assert f.feed("..EN") == ".." and f.feed("D!") == "END" and f.hit


def test_control_halves():
    # bad word chr(6): the token 6 alone, and " "+chr(6) = [32, 6] (multi-token: sync)
    p = pb.Params(allowed_token_ids=[5, 6, 7], bad_words=[chr(6)], min_tokens=2,
                  stop_token_ids=[9])
    ctl = worker._Control(p, [EOS], Tok(()), 40)
    assert ctl.active and ctl.sync and ctl.base is None and ctl.early is None   # no MLX yet
    ctl.build()
    m = ctl.mask(mx.zeros((1, 40)))
    assert [i for i in range(40) if m[0, i].item() == 0] == [5, 7]
    ctl.n = 2
    assert ctl.stop(9) == "stop" and ctl.stop(EOS) == "stop" and ctl.stop(5) is None
    assert not worker._Control(pb.Params(), [EOS], Tok(()), 40).active


def test_materialize_images_cleans_up(monkeypatch):
    made = []
    real = tempfile.mkstemp

    def spy(*a, **k):
        fd, path = real(*a, **k)
        made.append(path)
        return fd, path
    monkeypatch.setattr(tempfile, "mkstemp", spy)
    ok = "data:image/png;base64," + base64.b64encode(b"png").decode()
    paths, tmp = worker.materialize_images([ok, "file:///a.jpg", "http://x/y.png"])
    assert paths[1:] == ["/a.jpg", "http://x/y.png"] and tmp == made
    for f in tmp:
        os.unlink(f)
    made.clear()
    with pytest.raises(Exception):
        worker.materialize_images([ok, "data:image/png;base64,@@not base64@@"])
    assert len(made) == 2 and not any(os.path.exists(f) for f in made)


def test_out_of_vocab_ids_rejected(monkeypatch):
    """Check and Generate refuse ids past the logits width (400, before queueing)."""
    import mlx_vlm.prompt_utils
    monkeypatch.setattr(mlx_vlm.prompt_utils, "apply_chat_template", lambda *a, **k: "prompt")
    for f in ("allowed_token_ids", "stop_token_ids"):
        eng = engine([("done", "length")], prompt=(100,))
        eng.config, eng._template_vars, eng._developer = {}, set(), True
        eng.grammar_for = lambda p: None
        del eng._render                               # the real one
        _, _, end = events(eng, **{f: [5, 256]})
        assert end.error.kind == "invalid_request" and f in end.error.msg
        assert eng.sched.req is None
        assert events(engine(toks("a") + [("done", "length")], prompt=(100,)),
                      **{f: [5, 255]})[2].done.finish == "length"

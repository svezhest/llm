"""Log-probabilities (docs/perf/2026-10-21/design/logprobs.md): logprobs /
top_logprobs of generated tokens (built in the decode step, read with the
token), prompt_logprobs (teacher forcing inside the prefill, lm_head over the
slab's final hidden states in chunks), max_tokens 0 (prefill only) and
prompt_cache "off" (no lookup) / "read" (no store). Tiny random CPU models; the span rule and the
history == generation-prompt + reply property also on the real Ornith
tokenizer / chat template (skipped when the weights dir is absent)."""
import json
import logging
import os
import threading
import time
import types

import mlx.core as mx
import pytest

import worker
import worker_pb2 as pb
from conftest import EOS, IM_START, V, FakeTok, TinyVLM, prompts, request, scheduler

ORNITH = os.path.join(worker._REPO_ROOT, "weights", "ornith15-9b-mlx-8bit")


@pytest.fixture
def small(monkeypatch):
    """Prompt cache on (reply KV too, the production default), slabs of 8,
    lm_head chunks of 3 rows (a slab's scoring runs several chunks)."""
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", True)
    monkeypatch.setattr(worker, "PREFILL_STEP", 8)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 4)
    monkeypatch.setattr(worker, "PROMPT_CACHE_STRIDE", 16)
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_TOKENS", 8)
    monkeypatch.setattr(worker, "PROMPT_CACHE_KEEP_WAIT_S", float("inf"))
    monkeypatch.setattr(worker, "LOGPROBS_CHUNK", 3)


@pytest.fixture(params=["dense", "dense_fused", "moe"])
def lm(request):
    return request.getfixturevalue(request.param)


def tol(lm):
    """Per-token tolerance, prefill (M rows) vs decode (1 row) kernels on the CPU."""
    return 1e-4 if lm.model.norm.weight.dtype == mx.float32 else 1e-2


def drive(sched, reqs, arrive=None, max_ticks=5000, on_tick=None):
    """conftest.run keeping the logprobs: {id: (toks, lps, end)} with lps[i]
    the 4th item of the i-th "tok" event (None without one)."""
    arrive = dict(arrive or {})
    later = {id(r) for rs in arrive.values() for r in rs}
    for r in reqs:
        if id(r) not in later:
            sched.submit(r)
    outs = {r.id: ([], [], None) for r in reqs}
    for t in range(max_ticks):
        for r in arrive.pop(t, []):
            sched.submit(r)
        if not (sched.rows or sched.task or sched.waiting) and not arrive:
            return outs
        if on_tick is not None:
            on_tick(t, sched)
        sched.tick()
        for r in reqs:
            while not r.out.empty():
                ev = r.out.get()
                toks, lps, _ = outs[r.id]
                if ev[0] == "tok":
                    toks.append(ev[1])
                    lps.append(ev[3] if len(ev) > 3 else None)
                    assert len(ev) in (3, 4)
                else:
                    outs[r.id] = (toks, lps, ev)
    raise AssertionError("scheduler did not finish")


def gen_req(ids, n, rid, **params):
    return request(ids, n, rid, presence_penalty=0.0, temperature=0.0, **params)


def tf_req(ids, rid, **params):
    """Teacher forcing: prefill only, every position scored."""
    return request(ids, 0, rid, prompt_logprobs=True, **params)


def scored_at(req, pos):
    first, lps = req.plp
    return lps[pos - first]


# --- (a) teacher forcing == generation ---------------------------------------
def test_teacher_forcing_equals_generation(lm, small):
    """T=0 generation with logprobs, then the prompt + that answer (+ EOS)
    teacher-forced with max_tokens 0: each generated token's log p is the
    same (prefill kernels vs decode kernels), and so is the sum."""
    P = [IM_START] + prompts([29], 3)[0]
    g = gen_req(P, 10, "g", logprobs=True, top_logprobs=3)
    (toks, lps, end), = drive(scheduler(lm), [g]).values()
    assert end == ("done", "length") and len(toks) == 10
    t = tf_req(P + toks + [EOS], "tf")
    out = drive(scheduler(lm), [t])["tf"]
    assert out[0] == [] and out[2] == ("done", "length")
    first, scored = t.plp
    assert first == 1 and len(scored) == len(P) + len(toks)
    tf = [scored_at(t, len(P) + i) for i in range(len(toks))]
    for a, (b, _) in zip(tf, lps):
        assert abs(a - b) <= tol(lm), (tf, lps)
    assert abs(sum(tf) - sum(x for x, _ in lps)) <= tol(lm) * len(tf)
    # the log p of the EOS appended after the answer is scored too
    assert scored[-1] < 0 and all(x <= 1e-6 for x in scored)


def test_teacher_forcing_forced_token(dense, small):
    """A reasoning budget forces \\n</think> into the stream: the forced
    tokens' log p (taken from the step's kept log-softmax row) is what the
    teacher forcing of the same sequence says."""
    from mlx_vlm.utils import ThinkingBudgetCriteria
    NL, CLOSE, OPEN = 11, 12, 13
    tok = FakeTok({"</think>": [CLOSE], "<think>": [OPEN], "\n": [NL]})
    crit = ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=4, thinking_end_token="</think>",
                                  thinking_start_token="<think>", enable_thinking=True,
                                  prompt_preopens_thinking=True)
    P = prompts([9], 8)[0] + [OPEN]
    g = gen_req(P, 10, "think", logprobs=True)
    g.thinking = crit
    toks, lps, _ = drive(scheduler(dense), [g])["think"]
    assert toks[5:7] == [NL, CLOSE], toks
    t = tf_req(P + toks + [EOS], "tf")
    drive(scheduler(dense), [t])
    for i, (lp, _) in enumerate(lps):
        assert abs(scored_at(t, len(P) + i) - lp) <= 1e-4, i


# --- (b) flags off: the graph is the old one ------------------------------------
def test_flags_off_unchanged(lm, small, monkeypatch):
    """Without the flags no logprobs op is built (no _token_logprobs / lm_head
    scoring call, the prefill forward is skip_logits without return_hidden)
    and the "tok" events are the old 3-tuples; with them the token ids are
    the same."""
    calls = {"tl": 0, "head": 0, "hidden": 0, "fwd": 0}
    real_tl, real_head, real_call = worker._token_logprobs, worker._lm_head, type(lm).__call__

    def tl(*a, **k):
        calls["tl"] += 1
        return real_tl(*a, **k)

    def head(*a, **k):
        calls["head"] += 1
        return real_head(*a, **k)

    def call(self, *a, **k):
        if k.get("cache") is not None and a and a[0].shape[1] > 1:   # a prefill slab
            calls["fwd"] += 1
            assert k.get("skip_logits") is True
            calls["hidden"] += bool(k.get("return_hidden"))
        return real_call(self, *a, **k)
    monkeypatch.setattr(worker, "_token_logprobs", tl)
    monkeypatch.setattr(worker, "_lm_head", head)
    monkeypatch.setattr(type(lm), "__call__", call)

    ps = prompts([21, 13], 5)
    plain = drive(scheduler(lm), [gen_req(p, 8, f"r{i}") for i, p in enumerate(ps)])
    assert calls == {"tl": 0, "head": 0, "hidden": 0, "fwd": calls["fwd"]} and calls["fwd"] >= 3
    assert all(x is None for o in plain.values() for x in o[1])
    reqs = [gen_req(p, 8, f"r{i}", logprobs=True, prompt_logprobs=True) for i, p in enumerate(ps)]
    flagged = drive(scheduler(lm), reqs)
    assert calls["tl"] > 0 and calls["head"] > 0 and calls["hidden"] > 0
    for k in plain:
        assert flagged[k][0] == plain[k][0] and flagged[k][2] == plain[k][2]
        assert all(x is not None for x in flagged[k][1])
    assert all(r.plp is not None for r in reqs)


# --- (c) prompt-cache hits ---------------------------------------------------------
def test_prompt_cache_hit_first(dense, small):
    """A repeat resumes from the cache: `first` = resume depth + 1 (at most
    len - 1: the last token is always scored), the scored values equal the
    cold run's bit for bit (resumed at a slab end: same slabs); a variant
    resumes before its divergence; prompt_cache "off" scores everything."""
    X = [IM_START] + prompts([37], 11)[0] + [IM_START] + prompts([6], 12)[0]
    sched = scheduler(dense)
    a = tf_req(X, "a", prompt_cache="off")
    drive(sched, [a])
    assert a.plp[0] == 1 and a.reused == 0 and len(a.plp[1]) == len(X) - 1
    assert len(sched.eng._pcache.entries) == 1         # it still stores its entry
    b = tf_req(X, "b")
    drive(sched, [b])
    assert 0 < b.reused <= len(X) - 2 and b.plp[0] == b.reused + 1
    assert b.plp[1] == a.plp[1][b.reused:]
    Y = X[:30] + prompts([12], 13)[0]
    c = tf_req(Y, "c")
    drive(sched, [c])
    assert 0 < c.reused <= 30 and c.plp[0] == c.reused + 1
    assert c.plp[1][:29 - c.reused] == a.plp[1][c.reused:29]     # position 30 differs
    d = tf_req(X, "d", prompt_cache="off")
    drive(sched, [d])
    assert d.plp == a.plp and d.reused == 0


# --- (d) spans ---------------------------------------------------------------------------
IMS, IME, AS, US, SY, TL, NL, NN, TH, THC = IM_START, EOS, 20, 21, 22, 25, 24, 23, 4, 5
HDR, SKIPS = [IMS, AS, NL], [[TH, NN, THC, NN], [TH, NL]]


def spans(ids, roles, gen=False):
    return worker._assistant_spans(ids, roles, HDR, SKIPS, IME, gen=gen)


def test_spans():
    sys_ = [IMS, SY, NL, 100, 101, IME, NL]
    user = [IMS, US, NL, 102, IME, NL]
    off = [IMS, AS, NL, TH, NN, THC, NN, 110, 111, IME, NL]             # empty reasoning
    on = [IMS, AS, NL, TH, NL, 120, 121, NL, THC, NN, 122, IME, NL]      # reasoning
    tool = [IMS, US, NL, 130, 131, IME, NL]
    ids = sys_ + user + off + tool + on + user
    roles = ["system", "user", "assistant", "tool", "assistant", "user"]
    s1 = len(sys_ + user)
    s2 = len(sys_ + user + off + tool)
    assert spans(ids, roles) == [(2, s1 + 7, s1 + 10), (4, s2 + 5, s2 + 12)]
    # the generation prompt's trailing header has no message: dropped
    gen = ids + [IMS, AS, NL, TH, NL]
    assert spans(gen, roles, gen=True) == spans(ids, roles)
    # no think block at all (another template): right after the header
    assert spans(user + [IMS, AS, NL, 140, IME, NL], ["user", "assistant"]) == [(1, 9, 11)]


def test_spans_mismatch_logged(caplog):
    ids = [IMS, US, NL, 102, IME, NL] + [IMS, AS, NL, TH, NN, THC, NN, 110, IME, NL]
    with caplog.at_level(logging.WARNING, logger="qwen.worker"):
        got = spans(ids, ["user", "assistant", "assistant"])
    assert got == [(2, 13, 15)] and "2 assistant messages: matched from the end" in caplog.text


def test_spans_echoed_markup():
    """A user message holding the literal <|im_start|>assistant markup (an
    echoed transcript: the tokenizer makes it the special ids) adds a block
    before the real turns: matched from the end, the real reply keeps its span;
    an unterminated last block is a continued final message unless `gen`."""
    user = [IMS, US, NL, 102, IMS, AS, NL, 103, 104, IME, NL]           # "...<|im_start|>assistant\n.."
    reply = [IMS, AS, NL, TH, NN, THC, NN, 110, 111, IME, NL]
    ids = user + reply + [IMS, US, NL, 105, IME, NL]
    roles = ["user", "assistant", "user"]
    assert spans(ids, roles) == [(1, len(user) + 7, len(user) + 10)]
    cont = user + [IMS, AS, NL, TH, NN, THC, NN, 110, 111]              # continue_final_message
    assert spans(cont, ["user", "assistant"]) == [(1, len(user) + 7, len(cont))]


# --- (e) max_tokens 0 --------------------------------------------------------------------
def test_prefill_only(dense, small, monkeypatch):
    """max_tokens 0: finish length before any decode step, no tokens, the
    entry a generation would store (prompt[:-1]), nothing added by the
    reply KV."""
    P = [IM_START] + prompts([25], 21)[0]
    sched = scheduler(dense)
    t = tf_req(P, "t")
    toks, _, end = drive(sched, [t])["t"]
    assert toks == [] and end == ("done", "length") and t.steps == 0
    (e,) = sched.eng._pcache.entries
    assert e.ids == P[:-1] and e.reply is None
    ref = scheduler(dense)
    monkeypatch.setattr(worker, "PROMPT_CACHE_REPLY", False)
    drive(ref, [gen_req(P, 3, "g")])
    (r,) = ref.eng._pcache.entries
    assert r.ids == e.ids and set(r.ckpts) == set(e.ckpts)


# --- (f) a batch -----------------------------------------------------------------------------
def test_batch_rows(lm, small):
    """Rows with and without logprobs decoded together: only theirs carry
    logprobs, every row draws what it draws without the flags (a sampled
    row too: the draw is on the penalized / tempered logits, untouched)."""
    ps = prompts([17, 11, 14], 31)

    def make(flags):
        return [gen_req(ps[0], 9, "lp", **({"logprobs": True, "top_logprobs": 4} if flags else {})),
                gen_req(ps[1], 9, "plain"),
                request(ps[2], 9, "sampled", temperature=0.8, seed=7, top_k=20,
                        **({"logprobs": True} if flags else {}))]
    ref = drive(scheduler(lm), make(False))
    got = drive(scheduler(lm), make(True))
    for k in ref:
        assert got[k][0] == ref[k][0] and got[k][2] == ref[k][2], k
    assert all(x is not None for x in got["lp"][1]) and all(x is not None for x in got["sampled"][1])
    assert all(x is None for x in got["plain"][1])
    # raw logprobs: not the tempered distribution, top empty without top_logprobs
    assert all(top == [] for _, top in got["sampled"][1])


# --- (g) top_logprobs ---------------------------------------------------------------------
def test_top_logprobs(dense, small):
    g = gen_req(prompts([15], 41)[0], 8, "g", logprobs=True, top_logprobs=5)
    toks, lps, _ = drive(scheduler(dense), [g])["g"]
    for tok, (lp, top) in zip(toks, lps):
        assert len(top) == 5
        vals = [v for _, v in top]
        assert vals == sorted(vals, reverse=True)
        assert top[0] == (tok, lp)                      # T=0: the drawn token is the best
        assert len({i for i, _ in top}) == 5 and all(0 <= i < V for i, _ in top)


def test_stop_token_logprob(dense, small):
    """The stop token a row ends on has no "tok" event: its log p comes in
    req.stop_lp (the handler sends it last)."""
    P = prompts([15], 43)[0]
    g = gen_req(P, 8, "g", logprobs=True, top_logprobs=2)
    toks, lps, _ = drive(scheduler(dense), [g])["g"]
    k = next(i for i in range(2, len(toks)) if toks[i] not in toks[:i])
    s = gen_req(P, 8, "s", logprobs=True, top_logprobs=2, stop_token_ids=[toks[k]])
    got, glps, end = drive(scheduler(dense), [s])["s"]
    assert end == ("done", "stop") and got == toks[:k] and glps == lps[:k]
    assert s.stop_lp == (toks[k], *lps[k])


def test_no_dflash_for_logprobs(dense):
    sched = scheduler(dense)
    sched.eng.draft = object()
    for params, ok in (({}, True), ({"logprobs": True}, False), ({"prompt_logprobs": True}, False)):
        req = gen_req(prompts([5], 1)[0], 4, "x", **params)
        row = worker._Row(req, 10, 0)
        row.pen = worker._Penalties(req.sampling, req.ids, V)
        assert sched._spec_ok(row) is ok, params


# --- the handler, end to end (tiny model, a char tokenizer, a Qwen-like template) -------
SPECIAL = {"<|im_start|>": IMS, "<|im_end|>": IME, "<think>": TH, "</think>": THC,
           "assistant": AS, "user": US, "system": SY, "\n\n": NN, "\n": NL}
INV = {v: k for k, v in SPECIAL.items()}
LETTERS = [100 + ord(c) - 32 for c in "abcdefghijklmnopqrstuvwxyz"]


def tok_str(i):
    if i in INV:
        return INV[i]
    return chr(i - 100 + 32) if 100 <= i < 195 else chr(0x2000 + i)


class ChatTok:
    """Every id round-trips: specials (longest match first), printable ASCII
    at 100 + ord - 32, any other id as one private character."""
    stopping_criteria = types.SimpleNamespace(eos_token_ids=[IME])

    def encode(self, s, add_special_tokens=True):
        out, i = [], 0
        keys = sorted(SPECIAL, key=len, reverse=True)
        while i < len(s):
            k = next((k for k in keys if s.startswith(k, i)), None)
            if k is not None:
                out.append(SPECIAL[k])
                i += len(k)
                continue
            o = ord(s[i])
            out.append(100 + o - 32 if 32 <= o < 127 else o - 0x2000)
            i += 1
        return out

    def decode(self, ids):
        return "".join(tok_str(i) for i in ids)

    def batch_decode(self, seqs):
        return [self.decode(x) for x in seqs]

    def convert_tokens_to_ids(self, t):
        return SPECIAL[t]


class Detok:
    def __init__(self):
        self.reset()

    def reset(self):
        self.text, self.offset = "", 0

    def add_token(self, t):
        self.text += tok_str(t)

    @property
    def last_segment(self):
        s, self.offset = self.text[self.offset:], len(self.text)
        return s

    def finalize(self):
        pass


def render(processor, config, messages, add_generation_prompt=True, num_images=0,
           enable_thinking=True, **_):
    """The Ornith template's shape (assistant history always with its think block)."""
    out = ""
    for m in messages:
        if m["role"] == "assistant":
            rc = (m.get("reasoning_content") or "").strip()
            out += f"<|im_start|>assistant\n<think>\n{rc}\n</think>\n\n{m['content'].strip()}<|im_end|>\n"
        elif m["role"] == "tool":
            out += f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n"
        else:
            out += f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False
                                            else "<think>\n")
    return out


@pytest.fixture
def chat(dense, small, monkeypatch):
    import mlx_vlm.prompt_utils
    monkeypatch.setattr(mlx_vlm.prompt_utils, "apply_chat_template", render)
    eng = worker.Engine("/tmp/tiny-chat")
    tok = ChatTok()
    eng.processor = types.SimpleNamespace(tokenizer=tok, detokenizer=Detok())
    eng.config, eng._template_vars, eng._developer = {}, {"enable_thinking"}, True
    eng._think, eng._tool_tags = [TH, THC], None
    eng.vocab, eng._llg_tok, eng._not_ws = V, None, None
    eng._im_start, eng._im_user = IMS, US
    eng._span_tokens = worker._span_tokens(tok, IMS)
    assert eng._span_tokens == (HDR, SKIPS, IME)
    eng.model = TinyVLM(dense)
    eng.stopping = lambda t: t == IME
    eng.sched = scheduler(dense, eng=eng)
    eng.sched.alive = lambda: True
    return eng


def generate(eng, messages, rid, **params):
    """Engine.generate on a handler thread, the scheduler ticked here."""
    req = pb.Request(id=rid, messages=[pb.Message(**m) for m in messages],
                     params=pb.Params(**params))
    evs = []
    t = threading.Thread(target=lambda: evs.extend(eng.generate(req)))
    t.start()
    s = eng.sched
    while t.is_alive():
        if s.rows or s.task or s.waiting:
            s.tick()
        else:
            time.sleep(0.001)
    return evs


def lp_stream(evs):
    return [x for e in evs if e.HasField("token") for x in e.token.lp]


MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "say something"}]


@pytest.mark.parametrize("thinking", [False, True])
def test_handler_teacher_forcing(chat, thinking, caplog):
    """Generate T=0 with logprobs through the handler (template, tokenizer,
    Token.lp), then teacher-force the same messages + that answer as the last
    assistant message (add_generation_prompt false, max_tokens 0): the span is
    the generated tokens + <|im_end|>, its values are the generation's, the
    PromptLogprobs event comes right before Done."""
    kw = dict(temperature=0.0, presence_penalty=0.0, allowed_token_ids=LETTERS,
              chat_template_kwargs_json=json.dumps({"enable_thinking": thinking}))
    evs = generate(chat, MSGS, "g0", max_tokens=8, logprobs=True, top_logprobs=2, **kw)
    assert evs[-1].done.finish == "length"
    lps = lp_stream(evs)
    g = [x.token_id for x in lps]
    assert len(g) == 8 and all(len(x.top) == 2 for x in lps)
    assert all(x.token == tok_str(x.token_id) for x in lps)
    assert not any(e.HasField("prompt_logprobs") for e in evs)
    # the stop token's log p comes last (here on an empty Token)
    k = next(i for i in range(2, len(g)) if g[i] not in g[:i])
    evs = generate(chat, MSGS, "g1", max_tokens=8, logprobs=True, top_logprobs=2,
                   stop_token_ids=[g[k]], **kw)
    assert evs[-1].done.finish == "stop" and evs[-1].done.usage.gen_tokens == k
    got = lp_stream(evs)
    assert [x.token_id for x in got] == g[:k + 1]
    assert [x.logprob for x in got] == [x.logprob for x in lps[:k + 1]]
    text = chat.processor.tokenizer.decode(g[:k])
    reply = ({"role": "assistant", "content": "", "reasoning": text} if thinking
             else {"role": "assistant", "content": text})
    caplog.set_level(logging.INFO, logger="qwen.worker")
    evs = generate(chat, MSGS + [reply], "tf", max_tokens=0, prompt_logprobs=True,
                   prompt_cache="off", add_generation_prompt=False, **kw)
    assert [e.WhichOneof("kind") for e in evs] == ["start", "prompt_logprobs", "done"]
    d = evs[-1].done
    assert d.finish == "length" and d.usage.gen_tokens == 0 and d.usage.cached_tokens == 0
    plp = evs[1].prompt_logprobs
    ids = [x.token_id for x in plp.tokens]
    assert [x.token for x in plp.tokens] == [tok_str(i) for i in ids]
    assert plp.first == 1 and len(plp.logprobs) == len(ids) - 1
    (sp,) = plp.spans
    assert sp.message == 2 and sp.end - sp.start == k + 1 + (3 if thinking else 0)
    assert ids[sp.start:sp.start + k] == g[:k] and ids[sp.end - 1] == IME
    # the generation prompt is the history's prefix up to the span
    prompt = chat.processor.tokenizer.encode(render(None, None, [dict(m) for m in MSGS],
                                                    enable_thinking=thinking))
    assert ids[:sp.start] == prompt
    tf = [plp.logprobs[p - plp.first] for p in range(sp.start, sp.start + k)]
    for a, b in zip(tf, got):
        assert abs(a - b.logprob) <= 1e-4
    assert f"id=tf prompt_logprobs: scored=[1, {len(ids)}) of {len(ids)} spans=1" in caplog.text


def test_handler_cache_hit_first(chat):
    """Through the handler: a repeat's `first` = cached_tokens + 1."""
    msgs = MSGS + [{"role": "assistant", "content": "hello there"},
                   {"role": "user", "content": "and now something longer than a slab"}]
    kw = dict(max_tokens=0, prompt_logprobs=True, add_generation_prompt=False)
    a = generate(chat, msgs, "a", **kw)
    b = generate(chat, msgs, "b", **kw)
    pa, pb_ = a[1].prompt_logprobs, b[1].prompt_logprobs
    cached = b[-1].done.usage.cached_tokens
    assert pa.first == 1 and cached > 0 and pb_.first == cached + 1
    assert list(pb_.logprobs) == list(pa.logprobs)[cached:]
    assert [s.message for s in pb_.spans] == [2] and pb_.spans == pa.spans


def test_check_rejects(chat):
    def check(images=(), **params):
        req = pb.Request(id="c", messages=[pb.Message(**m) for m in MSGS],
                         params=pb.Params(**params), images=list(images))
        return chat.check(req)
    chat.config = {"vision_config": {}}
    assert check(max_tokens=0, prompt_logprobs=True) == ("", False)
    assert "max_tokens" in check(max_tokens=0)[0]
    assert "image" in check(["x.png"], prompt_logprobs=True)[0]
    assert check(["x.png"], logprobs=True)[0] == ""
    assert "top_logprobs" in check(top_logprobs=2)[0]
    assert "top_logprobs" in check(logprobs=True, top_logprobs=21)[0]
    assert check(logprobs=True, top_logprobs=20) == ("", False)


# --- the real tokenizer + template (no model) ----------------------------------------------
@pytest.fixture(scope="module")
def ornith():
    if not os.path.exists(os.path.join(ORNITH, "tokenizer.json")):
        pytest.skip("Ornith tokenizer not present")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ORNITH)
    with open(os.path.join(ORNITH, "chat_template.jinja")) as f:
        tpl = f.read()

    def ids(messages, gen, thinking=True, tools=None):
        s = tok.apply_chat_template(messages, chat_template=tpl, tokenize=False, tools=tools,
                                    add_generation_prompt=gen, enable_thinking=thinking)
        return tok.encode(s, add_special_tokens=False)
    return tok, ids


def test_ornith_span_tokens(ornith):
    tok, _ = ornith
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    hdr, skips, im_end = worker._span_tokens(tok, im_start)
    t = tok.convert_tokens_to_ids
    nl, nn = tok.encode("\n", add_special_tokens=False), tok.encode("\n\n", add_special_tokens=False)
    assert len(nl) == len(nn) == 1
    assert hdr == [im_start, t("assistant"), nl[0]] and im_end == t("<|im_end|>")
    assert skips == [[t("<think>"), nn[0], t("</think>"), nn[0]], [t("<think>"), nl[0]]]


@pytest.mark.parametrize("thinking", [False, True])
def test_ornith_history_is_prompt_plus_reply(ornith, thinking):
    """The history rendering of an assistant message = the generation prompt
    + the reply's tokens as the model generates them (reasoning + \\n</think>
    \\n\\n + content when thinking, content alone when not) + <|im_end|>, and
    the span is exactly that reply; over several turns with tool messages."""
    tok, render_ids = ornith
    hdr_etc = worker._span_tokens(tok, tok.convert_tokens_to_ids("<|im_start|>"))
    tools = [{"type": "function", "function": {"name": "get", "description": "x",
                                               "parameters": {"type": "object", "properties": {
                                                   "n": {"type": "integer"}}}}}]
    msgs = [{"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": "What is 2+2? Use the tool."}]
    replies = [("Let me call the tool.", "",
                [{"type": "function", "function": {"name": "get", "arguments": {"n": 4}}}]),
               ("The tool said four.", "The answer is 4.", None)]
    for k, (reasoning, content, calls) in enumerate(replies):
        prompt = render_ids(msgs, True, thinking, tools)
        m = {"role": "assistant", "content": content}
        if thinking:
            m["reasoning_content"] = reasoning
        if calls:
            m["tool_calls"] = calls
        text = (f"{reasoning}\n</think>\n\n" if thinking else "") + content
        if calls:
            text += ("\n\n" if content else "") + "<tool_call>\n<function=get>\n<parameter=n>\n4\n" \
                    "</parameter>\n</function>\n</tool_call>"
        reply = tok.encode(text + "<|im_end|>", add_special_tokens=False)
        msgs = msgs + [m]
        hist = render_ids(msgs, False, thinking, tools)
        assert hist[:len(prompt) + len(reply)] == prompt + reply, k
        roles = [x["role"] for x in msgs]
        spans = worker._assistant_spans(hist, roles, *hdr_etc)
        assert spans[-1] == (len(msgs) - 1, len(prompt), len(prompt) + len(reply))
        if not thinking:
            # thinking off: earlier turns render the empty think block too
            assert all(hist[s - 4:s] == hdr_etc[1][0] for _, s, _ in spans)
        msgs = msgs + [{"role": "tool", "content": "4"}] if calls else msgs


@pytest.mark.parametrize("tie", [False, True])
def test_lm_head_is_the_models_head(tie):
    """hidden_states[-1] (return_hidden) through _lm_head = the model's own
    logits, bit for bit, tied embeddings (embed_tokens.as_linear) or not."""
    from conftest import COMMON, VCFG
    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.models.qwen3_5.config import TextConfig
    mx.random.seed(3)
    m = q35.LanguageModel(TextConfig(model_type="qwen3_5_text", intermediate_size=256,
                                     tie_word_embeddings=tie, **COMMON), VCFG)
    assert hasattr(m, "lm_head") is not tie
    x = mx.array([prompts([9], 4)[0]])
    out = m(x, return_hidden=True)
    assert mx.array_equal(worker._lm_head(m, out.hidden_states[-1]), out.logits).item()


# --- review round 1 ---------------------------------------------------------------------------
def test_prefill_only_never_joins(dense, small):
    """M2: a max_tokens 0 request arriving while rows decode ends inside its
    prefill: it never joins the batch (no GDN restack, no group), the other
    rows' streams are unchanged, its entry is stored."""
    ps = prompts([13, 15], 61)
    make = lambda: [gen_req(ps[0], 20, "a", logprobs=True), gen_req(ps[1], 20, "b")]  # noqa: E731
    ref = drive(scheduler(dense), make())
    sched = scheduler(dense)
    joined, real = [], sched._join
    sched._join = lambda row: (joined.append(row.req.id), real(row))[1]
    X = [IM_START] + prompts([30], 62)[0]
    t = tf_req(X, "t")
    rows = []
    got = drive(sched, make() + [t], arrive={4: [t]},
                on_tick=lambda i, s: rows.append([r.req.id for r in s.rows]))
    assert "t" not in joined and not any("t" in r for r in rows) and max(map(len, rows)) == 2
    assert not sched.groups and t.steps == 0 and t.plp[0] == 1
    assert got["t"][0] == [] and got["t"][2] == ("done", "length")
    for k in ref:
        assert got[k] == ref[k], k
    assert any(e.ids == X[:-1] for e in sched.eng._pcache.entries)


def test_prompt_cache_read(dense, small):
    """prompt_cache "read": resumes like any hit, stores nothing (no entry,
    no reply KV, no partial store on a cancel), and lets go of the entry it
    resumed from (evictable at once)."""
    import gc
    import weakref
    X = [IM_START] + prompts([37], 71)[0] + [IM_START] + prompts([6], 72)[0]
    sched = scheduler(dense)
    pc = sched.eng._pcache
    base = tf_req(X, "base")
    drive(sched, [base])
    (e,) = pc.entries
    snap = (pc.bytes, [x.ids for x in pc.entries], dict(e.ckpts))
    v = tf_req(X[:-4] + prompts([9], 73)[0], "v", prompt_cache="read")
    drive(sched, [v])
    assert v.reused > 0 and v.plp[0] == v.reused + 1
    g = gen_req(X + prompts([5], 74)[0], 4, "g", prompt_cache="read")    # a generation: no reply KV
    c = tf_req(X + prompts([40], 75)[0], "c", prompt_cache="read")       # cancelled: no partial
    drive(sched, [g, c], on_tick=lambda i, s: c.cancel.set() if s.task_row is not None
          and s.task_row.req is c and s.task_row.started else None)
    assert g.reused > 0 and c.plp is None
    assert (pc.bytes, [x.ids for x in pc.entries], dict(e.ckpts)) == snap
    # nothing holds the entry any more: evicted, it is gone
    ref = weakref.ref(e)
    del e
    while pc.evict():
        pass
    gc.collect()
    assert ref() is None and pc.bytes == 0
    # "off" still stores; the cold values are the hit's
    o = tf_req(X, "o", prompt_cache="off")
    drive(sched, [o])
    assert o.plp[0] == 1 and len(pc.entries) == 1 and o.plp[1] == base.plp[1]


def test_logprobs_are_raw(dense, small):
    """Penalties, temperature and the draw do not touch the reported log p:
    a sampled run with presence / frequency / repetition penalties gives the
    log-softmax of the model's own logits (teacher forcing, and an
    independent forward for the first token)."""
    P = [IM_START] + prompts([19], 81)[0]
    kw = dict(presence_penalty=1.5, frequency_penalty=0.5, repetition_penalty=1.3,
              temperature=0.7, seed=5, logprobs=True)
    g = request(P, 10, "g", **kw)
    toks, lps, _ = drive(scheduler(dense), [g])["g"]
    t = tf_req(P + toks + [EOS], "tf", prompt_cache="off")
    drive(scheduler(dense), [t])
    for i, (lp, _) in enumerate(lps):
        assert abs(scored_at(t, len(P) + i) - lp) <= 1e-4, i
    x = dense(mx.array([P])).logits[0, -1].astype(mx.float32)
    dense._position_ids = dense._rope_deltas = None
    want = (x - mx.logsumexp(x))[toks[0]].item()
    assert abs(want - lps[0][0]) <= 1e-5


def test_batch_two_equals_solo(dense_fused, small):
    """B = 2 is bit-exact (per-row qmv): the same log p as alone."""
    ps = prompts([12, 17], 91)
    reqs = lambda: [gen_req(p, 8, f"r{i}", logprobs=True, top_logprobs=3)  # noqa: E731
                    for i, p in enumerate(ps)]
    both = drive(scheduler(dense_fused), reqs())
    for r in reqs():
        assert drive(scheduler(dense_fused, max_batch=1), [r])[r.id] == both[r.id]


def test_sync_row_and_ignored_eos(dense, small):
    """A grammar row (sync mode: no step-ahead) and an ignored EOS ("skip":
    a tok event, not text) carry their log p like any token."""
    from test_scheduler import AllowAll
    P = prompts([14], 95)[0]
    other = lambda: gen_req(prompts([9], 96)[0], 9, "o")  # noqa: E731
    plain = drive(scheduler(dense), [gen_req(P, 9, "g", logprobs=True, top_logprobs=2),
                                     other()])["g"]
    g = gen_req(P, 9, "g", logprobs=True, top_logprobs=2)
    g.guide = AllowAll()
    got = drive(scheduler(dense), [g, other()])["g"]
    assert got[0] == plain[0] and got[1] == plain[1]
    plain = drive(scheduler(dense), [gen_req(P, 9, "p", logprobs=True, top_logprobs=2)])["p"]
    toks = plain[0]
    k = next(i for i in range(2, len(toks)) if toks[i] not in toks[:i])
    s = gen_req(P, 9, "s", logprobs=True, top_logprobs=2, ignore_eos=True)
    s.ctl = worker._Control(s.p, [toks[k]], FakeTok(), V)
    sched = scheduler(dense)
    s.out = type(s.out)()
    sched.submit(s)
    evs = []
    while sched.rows or sched.task or sched.waiting:
        sched.tick()
        while not s.out.empty():
            evs.append(s.out.get())
    tok_evs = [e for e in evs if e[0] == "tok"]
    assert [e[1] for e in tok_evs] == toks and [e[3] for e in tok_evs] == plain[1]
    assert tok_evs[k][2] is False and evs[-1] == ("done", "length")


def test_forced_token_in_batch(dense, small):
    """A reasoning-budget forced token at B >= 2: its log p (from the kept
    row) is still the teacher-forced one."""
    from mlx_vlm.utils import ThinkingBudgetCriteria
    NL_, CLOSE, OPEN = 11, 12, 13
    tok = FakeTok({"</think>": [CLOSE], "<think>": [OPEN], "\n": [NL_]})
    crit = ThinkingBudgetCriteria(tokenizer=tok, thinking_budget=4, thinking_end_token="</think>",
                                  thinking_start_token="<think>", enable_thinking=True,
                                  prompt_preopens_thinking=True)
    P = prompts([9], 8)[0] + [OPEN]
    g = gen_req(P, 10, "think", logprobs=True, top_logprobs=2)
    g.thinking = crit
    others = [gen_req(p, 12, f"o{i}", logprobs=bool(i)) for i, p in enumerate(prompts([7, 11], 9))]
    toks, lps, _ = drive(scheduler(dense), [g] + others)["think"]
    assert toks[5:7] == [NL_, CLOSE], toks
    assert g.rows_sum > g.steps                         # it decoded in a batch
    t = tf_req(P + toks + [EOS], "tf")
    drive(scheduler(dense), [t])
    for i, (lp, top) in enumerate(lps):
        assert abs(scored_at(t, len(P) + i) - lp) <= 1e-4, i
        assert len(top) == 2


def test_prompt_logprobs_cancel_and_score_failure(dense, small, monkeypatch):
    """Cancelled mid-prefill: no plp, the prefix stored (partial). A _score
    failure fails the request (error event, no plp) and stores the prefix
    the caches hold (the slab's eval had finished)."""
    X = [IM_START] + prompts([40], 101)[0]
    sched = scheduler(dense)
    c = tf_req(X, "c")
    out = drive(sched, [c], on_tick=lambda i, s: c.cancel.set() if i == 2 else None)
    assert out["c"][2] == ("done", "cancelled") and c.plp is None
    assert [len(e.ids) for e in sched.eng._pcache.entries if e.partial]
    real, n = worker._score, []

    def boom(*a, **k):
        n.append(1)
        if len(n) == 2:
            raise RuntimeError("lm_head failed")
        return real(*a, **k)
    monkeypatch.setattr(worker, "_score", boom)
    sched = scheduler(dense)
    f = tf_req(X, "f")
    out = drive(sched, [f])
    assert out["f"][2][:2] == ("error", "RuntimeError") and f.plp is None
    (e,) = sched.eng._pcache.entries
    assert e.partial and len(e.ids) == 16                # two slabs of 8


def test_handler_stop_string_and_tools(chat):
    """Through the handler: a stop string's hit reports the tokens up to the
    one completing it (its text cut); tools mode (answer buffered) sends
    every log p on empty Tokens before the content. Entries == gen_tokens."""
    kw = dict(temperature=0.0, presence_penalty=0.0, allowed_token_ids=LETTERS,
              chat_template_kwargs_json=json.dumps({"enable_thinking": False}),
              max_tokens=8, logprobs=True)
    evs = generate(chat, MSGS, "g", **kw)
    g = [x.token_id for x in lp_stream(evs)]
    text = chat.processor.tokenizer.decode(g)
    k = next(i for i in range(2, 7) if text.find(text[i:i + 2]) == i)
    evs = generate(chat, MSGS, "s", stop=[text[k:k + 2]], **kw)
    d = evs[-1].done
    got = lp_stream(evs)
    assert d.finish == "stop" and [x.token_id for x in got] == g[:k + 2]
    assert len(got) == d.usage.gen_tokens
    assert "".join(e.token.text for e in evs if e.HasField("token")) == text[:k]
    tools = json.dumps([{"type": "function", "function": {"name": "get", "parameters": {
        "type": "object", "properties": {"n": {"type": "integer"}}}}}])
    evs = generate(chat, MSGS, "t", tools_json=tools, **kw)
    toks = [e.token for e in evs if e.HasField("token")]
    assert [x.token_id for x in lp_stream(evs)] == g and len(g) == evs[-1].done.usage.gen_tokens
    assert all(t.text == "" for t in toks if t.lp) and toks[-1].text == text and not toks[-1].lp


def test_check_prompt_cache_values(chat):
    def check(v):
        req = pb.Request(id="c", messages=[pb.Message(**m) for m in MSGS],
                         params=pb.Params(prompt_cache=v))
        return chat.check(req)[0]
    assert check("") == check("off") == check("read") == ""
    assert "prompt_cache" in check("false")

"""Log-probabilities (logprobs / top_logprobs / prompt_logprobs) and the
assistant spans of a prompt."""

import os

import mlx.core as mx

from .config import log
from .chat import THINK_CLOSE, THINK_OPEN


# --- log-probabilities (Params.logprobs / top_logprobs / prompt_logprobs) ----
# Raw: the model's own logits, before penalties / temperature / masks (vLLM
# logprobs_mode=raw_logprobs), as fp32 logit - logsumexp. prompt_logprobs runs
# lm_head over a prefill slab's final hidden states in chunks of this many rows
# (the transient is chunk x vocab x (2 + 4) bytes: ~0.38 GB at 256 x 248K).
LOGPROBS_CHUNK = max(1, int(os.environ.get("QWEN_LOGPROBS_CHUNK", "256")))
MAX_TOP_LOGPROBS = 20                     # vLLM's default max_logprobs


def _lm_head(lm, h):
    """Logits of final hidden states `h`: LanguageModel.__call__'s own last
    step (mlx-vlm 0.7.6 qwen3_5 / qwen3_5_moe: `out` = the model's output after
    its final norm, which return_hidden appends as hidden_states[-1], goes
    through embed_tokens.as_linear when the embeddings are tied, else lm_head)."""
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(h)
    return lm.lm_head(h)


def _score(lm, h, targets, chunk=None):
    """Teacher forcing over a prefill slab: h (1, n, H) final hidden states,
    targets (1, n) the token after each position -> [log p(target_j)] (floats).
    lm_head in chunks of `chunk` rows, each evaluated before the next is built
    (the vocab-wide logits never exist for the whole slab)."""
    chunk = chunk or LOGPROBS_CHUNK
    n, out = h.shape[1], []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        lg = _lm_head(lm, h[:, a:b])
        lse = mx.logsumexp(lg.astype(mx.float32), axis=-1)
        tgt = mx.take_along_axis(lg, targets[:, a:b, None], axis=-1)[..., 0]
        lp = tgt.astype(mx.float32) - lse
        mx.eval(lp)
        out += lp[0].tolist()
    return out


def _token_logprobs(x, y, k, keep=False):
    """A decode step's raw logits x (1, V) and its token y (1,) -> lazy
    (log p of y (1, 1), top-k ids (1, k), their log p (1, k), the whole
    log-softmax row when `keep` (a forced token may replace y) else None);
    no top when k == 0. Built inside the step (the same async_eval as y)."""
    x = x.astype(mx.float32)
    lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
    got = mx.take_along_axis(lp, y.reshape(1, 1), axis=-1)
    idx = vals = None
    k = min(k, lp.shape[-1])
    if k > 0:
        idx = mx.argpartition(-lp, kth=k - 1, axis=-1)[:, :k]
        vals = mx.take_along_axis(lp, idx, axis=-1)
        # sorted in fp32; the greedy draw is an argmax in the model dtype
        # (fp16 on the GPU): on a near-tie the drawn token can be top[1]
        order = mx.argsort(-vals, axis=-1)
        idx = mx.take_along_axis(idx, order, axis=-1)
        vals = mx.take_along_axis(vals, order, axis=-1)
    return got, idx, vals, lp if keep else None


def _forced_logprobs(lp, tok):
    """The step's logprobs with a forced token (reasoning budget) instead of
    the drawn one: its log p from the kept row, queued now."""
    got = lp[3][:, tok:tok + 1]
    mx.async_eval(got)
    return got, lp[1], lp[2], None


def _read_logprobs(lp):
    """A row's evaluated step logprobs -> (log p, [(id, log p)] top, best first)."""
    got, idx, vals, _ = lp
    top = list(zip(idx[0].tolist(), vals[0].tolist())) if idx is not None else []
    return got.item(), top


def _wants(req, flag):
    """A pb.Params flag of the request (False without Params: tests' bare _Req)."""
    return bool(getattr(req.p, flag, False))


PROMPT_CACHE_MODES = ("", "off", "read")


def _cache_mode(req):
    """Params.prompt_cache: "" normal; "off": no lookup (a cold prefill, every
    prompt position scored), the entry is still stored; "read": the lookup as
    usual, nothing stored (no prefill entry, no partial store on a cancel or
    failure, no reply KV): attribution variants of one baseline leave the
    cache as they found it."""
    return getattr(req.p, "prompt_cache", "") or ""


def _assistant_spans(ids, roles, hdr, skips, im_end, gen=False):
    """The token spans the model generated for each assistant message ->
    [(message index, start, end)]. A block is `hdr` = <|im_start|> assistant
    \n; then the first of `skips` it starts with ([<think>, \n\n, </think>,
    \n\n]: empty reasoning = the generation prompt with thinking off; then
    [<think>, \n]: thinking on, the reasoning is generated) is the prompt's
    too; `end` = its <|im_end|> + 1 (unterminated: the end of the ids). The
    k-th block from the end is the k-th assistant message from the end of
    `roles`; `gen`: the prompt ends with the generation prompt's header (an
    unterminated last block with no message: dropped). A count mismatch
    (message text holding the literal <|im_start|>assistant markup: an
    echoed transcript, few-shot examples; usually before the real turns) is
    logged and matched from the end: the latest messages keep their spans."""
    blocks, n = [], len(ids)
    for i in range(n - 1):
        if ids[i] != hdr[0] or ids[i + 1] != hdr[1]:
            continue
        s = i + 2
        if ids[s:s + len(hdr) - 2] == list(hdr[2:]):
            s += len(hdr) - 2
        for sk in skips:
            if sk and ids[s:s + len(sk)] == list(sk):
                s += len(sk)
                break
        e = next((j + 1 for j in range(s, n) if ids[j] == im_end), None)
        blocks.append((s, e))
    msgs = [k for k, r in enumerate(roles) if r == "assistant"]
    if gen and blocks and blocks[-1][1] is None:
        blocks.pop()                            # the generation prompt's header
    if len(blocks) != len(msgs):
        log.warning("prompt_logprobs: %d assistant blocks in the prompt, %d assistant messages: "
                    "matched from the end", len(blocks), len(msgs))
    pairs = list(zip(reversed(msgs), reversed(blocks)))[::-1]
    return [(k, s, n if e is None else e) for k, (s, e) in pairs]


def _span_tokens(tok, im_start):
    """_assistant_spans' tokens for a tokenizer: the assistant header and the
    two forms of the generation prompt's <think> pre-open (thinking off: an
    empty reasoning block; on: <think>\n, the reasoning is generated), and
    <|im_end|>. Ornith / Qwen3.5: [<|im_start|>, assistant, \n], [<think>,
    \n\n, </think>, \n\n], [<think>, \n] (tests/test_logprobs.py checks the
    real tokenizer and template). None: no such header."""
    hdr = list(tok.encode("<|im_start|>assistant\n", add_special_tokens=False))
    if len(hdr) < 2 or hdr[0] != im_start:
        return None
    skips = [list(tok.encode(x, add_special_tokens=False))
             for x in (f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n", f"{THINK_OPEN}\n")]
    return hdr, skips, tok.convert_tokens_to_ids("<|im_end|>")

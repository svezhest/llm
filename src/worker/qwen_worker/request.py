"""A request (_Req), a running row (_Row) and a moved prompt-cache lease (_Own)."""

import dataclasses
import queue
import threading
import time

from .logprobs import _wants, MAX_TOP_LOGPROBS


@dataclasses.dataclass(eq=False)
class _Req:
    """A generation as the handler thread hands it to the scheduler (built
    without MLX). The scheduler answers on `out`: ("tok", id, shown) per output
    token (shown False: an ignored EOS, not text; with Params.logprobs a 4th
    item (log p, [(id, log p)] top)), then ("done", finish) or
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
    bounds: list | None = None                  # _bounds(ids) (memoized: _Scheduler._bounds)
    keep_since: float | None = None             # first blocked by the entry it would resume from
    peeked: tuple | None = None                 # (cache version, weakref to the entry, depth)
    disk: object = None                         # pcdisk.Want (QWEN_PROMPT_CACHE_DISK)
    steps: int = 0                              # decode steps (DFlash: rounds) taken
    rows_sum: int = 0                           # sum of the batch sizes of those steps
    # snapshot taken on the scheduler when the request ends (before anything
    # can reset the peak): bytes, and the mean batch size over its steps
    peak: int = 0
    active: int = 0
    pool: int = 0
    batch_mean: float = 0.0
    # Params.prompt_logprobs: (first scored position, [log p of ids[first:]]),
    # set when the prefill ends; Params.logprobs: (id, log p, top) of the
    # stop token a "stop" finish ended on (no "tok" event carries it)
    plp: tuple | None = None
    stop_lp: tuple | None = None
    # tools / tests only (not on the wire): the generated tokens to feed
    # instead of the drawn ones, token k as the k-th input after the prompt
    # (DeepInfer's golden_token_ids: decode teacher forcing, the logprobs then
    # those of the forced tokens); past its end the draw goes on as usual
    golden: list | None = None
    # QWEN_PREEMPT (_Scheduler._preempt): `ids` then holds the prompt + the
    # tokens generated before the preemption and `max_tokens` what is left;
    # `orig` = the prompt as the handler sent it, `carry` = (sampler,
    # penalties) the resumed row goes on with
    orig: list | None = None
    carry: tuple | None = None
    preempts: int = 0

    @property
    def prompt_ids(self):
        """The prompt as the handler built it (`ids` after a preemption)."""
        return self.ids if self.orig is None else self.orig

    @property
    def prompt_tokens(self):
        return len(self.prompt_ids)


class _Own:
    """QWEN_PROMPT_CACHE_MOVE: what a row that holds its conversation's only
    KV copy keeps to hand it to the prompt cache when it ends
    (_Scheduler._settle / _settle_prefix). `husk`: the entry it resumed from,
    moved out of the cache (ids + states, no KV; None: a miss, or a hit on
    an entry that stays: a sibling's); `at`: where it resumed; `lo`: its
    store's own segment starts there (`at` when the entry it resumed from
    stays and its pieces are inherited, else 0); `L`: the prompt entry's
    length (len(prompt) - 1); `end`:
    (the recurrent states at L, the drafter's context there) once the prefill
    completed; `store`: _prompt_cache's store for this prompt (checkpoints,
    policies), taking the segment to keep."""
    __slots__ = ("husk", "at", "lo", "L", "end", "store")

    def __init__(self, husk, at, lo, L):
        self.husk, self.at, self.lo, self.L = husk, at, lo, L
        self.end = self.store = None

    def close(self):
        """When the row's KV went to the cache (or was lost): `store` is a
        closure over this object and the row — a reference cycle that would
        keep the dead row's caches, the saved states and the husk (0.4-1.3 GiB
        a turn on the 27B) until Python's cyclic collector happens to run."""
        self.husk = self.end = self.store = None


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
        self.parent = None                          # the entry the running prefill resumed from
        self.own = None                             # _Own: its KV is the cache's only copy (MOVE)
        self.started = False                        # its prefill / DFlash run began
        self.cached = False                         # resumed from the prompt cache or stored an entry
        self.spec = False                           # ran DFlash (_spec): never grouped
        self.group = None                           # its _PrefixGroup (QWEN_CASCADE)
        # reply KV (QWEN_PROMPT_CACHE_REPLY; _Scheduler._store_reply)
        self.fed = None                             # generated tokens as read, in order (list)
        self.entry = None                           # weakref to the entry its prefill stored
        self.plan = None                            # (the prompt's slab ends, protected ckpts)
        self.why = None                             # finish reason (_finish)
        # Params.logprobs: top_logprobs (0..20), else None (no logprobs op in the step)
        self.lpk = (min(int(req.p.top_logprobs), MAX_TOP_LOGPROBS)
                    if _wants(req, "logprobs") else None)
        self.lp = None                              # logprobs of y (lazy, _token_logprobs)
        self.lp_next = None                         # ... of the step just queued
        # QWEN_PREFILL_LANE: the model's mRoPE state (_rope_deltas,
        # _position_ids) as this row's prefill left it at its last slab;
        # put back before its next one (the other prefill wrote its own)
        self.rope = None

    @property
    def sync(self):
        # the mask for token n+1 needs token n on the host: no step-ahead
        return self.req.guide is not None or self.req.ctl.sync

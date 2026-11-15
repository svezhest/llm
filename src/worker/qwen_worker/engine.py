"""Engine: model load, request checks / rendering, generate() for the handlers."""

import hashlib
import json
import os
import queue
import threading
import time

import mlx.core as mx
import numpy as np
from mlx_vlm.tokenizer_utils import BPEStreamingDetokenizer, make_streaming_detokenizer

import cascade_mma
import pcdisk
import worker_pb2 as pb

from . import config as cfg
from . import sizing
from .config import (_WIRED_GB, CACHE_LIMIT_GB, CLEAR_CACHE, DRAFT, DRAFT_BITS, DRAFT_MTP,
                     EVENT_POLL_S, FP16,
                     GDN_FAST, GDN_FUSE, GIB, KV_KWARGS, log, MAX_BATCH, MAX_CONTEXT,
                     PREFILL_BUDGET, PREFILL_MIN_STEP, PREFILL_SHARE, PREFILL_STEP, PROMPT_CACHE,
                     PROMPT_CACHE_GB, PROMPT_CACHE_REPLY, RESERVE_TOKENS, SAMPLING_KWARGS,
                     SPLIT_SPECIAL_TOKENS, TEXT_ONLY, VERIFY_MR_HEAD, VERIFY_MRQMM)
from .chat import (_developer_to_system, _PlainContent, _rebuild_messages, materialize_images,
                   parse_tool_calls, THINK_CLOSE, THINK_OPEN, TOOL_OPEN)
from .pcache import _Entry, _PrefixCache, _Seg
from .sizing import _batch_budget, _batch_cap, _fixed_slab, _prefill_reserve
from .stream import _BPEDetokenizer, _Control, _done, _eos_ids, _GenState, _Reject, _StopFilter
from .grammar import _grammar, _Guide, _response_schema, _tool_choice, _ws_masks, JSON_MAX_WS
from .rows import _install_rows_attention
from .kernels import (_draft_wide, _install_fast_paths, _install_gdn_fuse, _invariant_check,
                      _log_invariant, _probe_verify_attention, _setup_verify_mrqmm, _setup_verify_qmv,
                      _widen_drafter)
from .mtp import load_mtp, mtp_dir
from .logprobs import (_assistant_spans, _span_tokens, LOGPROBS_CHUNK, MAX_TOP_LOGPROBS,
                       PROMPT_CACHE_MODES)
from .request import _Req
from .scheduler import _Scheduler


def _load_model(model_path, text_only=False):
    """mlx_vlm.load -> (model, processor, skipped): `skipped` = (tensors, bytes)
    of the vision tower left out with `text_only` (None: nothing left out).

    text_only: the model is loaded lazily (mx.load reads a tensor's bytes only
    when it is evaluated), its `vision_tower` is dropped -- the same state
    mlx-vlm >= 0.7 leaves a checkpoint without vision weights in
    (vision_tower = None) -- and only then are the parameters evaluated, so
    the tower's tensors are never read from disk nor resident. mlx-vlm's
    strict load_weights still checks them against the module (shapes, names).
    Dropping vision_config before the build is not possible from outside:
    load_model re-reads config.json and builds qwen3_5's VisionModel from
    defaults even without one. The tower's random init is lazy too, and dies
    unevaluated with it. Text: the same language-model bits as a full load."""
    from mlx.utils import tree_flatten
    from mlx_vlm import load
    model, processor = load(model_path, lazy=text_only)
    skipped = None
    if text_only and getattr(model, "vision_tower", None) is not None:
        params = tree_flatten(model.vision_tower.parameters())
        skipped = (len(params), sum(a.nbytes for _, a in params))
        del params
        model.vision_tower = None
    if text_only:
        mx.eval(model.parameters())               # what load(lazy=False) does
    return model, processor, skipped


def _exit_soon(code=70):
    """The scheduler gave up: exit after the error events went out, so launchd
    (KeepAlive) restarts the worker instead of a live process refusing
    everything."""
    log.error("exiting with %d in 1 s so the worker is restarted", code)
    threading.Timer(1.0, os._exit, (code,)).start()


class Engine:
    """Owns the model; requests go through its _Scheduler."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        # the id clients see (/v1/models, "model" in responses): the weights dir name
        self.model_id = os.path.basename(os.path.normpath(model_path))
        self.text_only = False          # QWEN_TEXT_ONLY=1 on a VL checkpoint: no vision tower
        self.ready = False
        self.load_ms = 0
        self.max_batch = MAX_BATCH
        self.sched = None
        self._cancels: dict[str, threading.Event] = {}
        self._cancels_lock = threading.Lock()
        self._pcache = _PrefixCache()
        self.draft = None
        self.draft_wide = False                 # QWEN_DRAFT_FP32: fp32 drafter activations
        self.mtp = None                         # the MTP head drafter (QWEN_DRAFT / QWEN_DRAFT_MTP)
        self._span_tokens = None        # (header, think pre-opens, <|im_end|>): _assistant_spans
        self._tok_str = {}              # id -> tokenizer.decode([id]) (logprobs; vocab-bounded)
        self._plain = None              # _PlainContent (QWEN_SPLIT_SPECIAL_TOKENS)

    def load(self):
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
        self.model, self.processor, skipped = _load_model(
            self.model_path, TEXT_ONLY and "vision_config" in self.config)
        if TEXT_ONLY and "vision_config" in self.config:
            self.text_only = True
            if skipped is not None:
                log.info("QWEN_TEXT_ONLY=1: vision tower not loaded (%d tensors, %.2f GiB of "
                         "checkpoint weights never read); image requests will be refused",
                         skipped[0], skipped[1] / GIB)
            else:
                log.warning("QWEN_TEXT_ONLY=1: %s has no vision tower to skip; image requests "
                            "will be refused", type(self.model).__module__)
        if FP16:
            lm = self.model.language_model
            lm.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
            mx.eval(lm.parameters())
            log.info("language model cast to float16 (QWEN_FP16=1)")
        if DRAFT and cfg.BATCH_INVARIANT:
            log.warning("QWEN_DRAFT ignored: QWEN_BATCH_INVARIANT=1 (the verify rounds "
                        "differently from the batch's decode)")
        elif DRAFT and KV_KWARGS["kv_bits"] is not None:
            # The spec loop never re-quantizes the cache and rollback trims a
            # quantized cache wrongly; the drafter is only worth it on the 9B,
            # which runs an fp16 cache anyway.
            log.warning("QWEN_DRAFT ignored: speculative decoding needs QWEN_KV_BITS=0")
        elif DRAFT and mtp_dir(DRAFT):            # the target's own MTP head (mtp.py)
            self.mtp = load_mtp(DRAFT, self.model.language_model, FP16)
        elif DRAFT:
            from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
            draft, kind = load_drafter(DRAFT)
            if kind != "dflash":
                raise RuntimeError(f"unsupported drafter kind {kind!r} (only dflash)")
            validate_drafter_compatibility(self.model, draft, kind)
            wide = _draft_wide(draft)
            if FP16 and not wide:
                draft.apply(lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a)
            if DRAFT_BITS:
                import mlx.nn as nn
                nn.quantize(draft, group_size=64, bits=DRAFT_BITS,
                            class_predicate=lambda _p, m: isinstance(m, nn.Linear)
                            and m.weight.shape[1] % 64 == 0)
            if wide:
                _widen_drafter(draft, bool(DRAFT_BITS))
            mx.eval(draft.parameters())
            if cfg.DRAFT_BLOCK:
                draft.prefer_requested_block_size = True
            self.draft, self.draft_wide = draft, wide
            log.info("dflash drafter loaded: %s (%s %s, %s, block %s, target layers %s)", DRAFT,
                     type(draft).__name__, f"{DRAFT_BITS}-bit" if DRAFT_BITS else
                     ("bf16" if wide or not FP16 else "fp16"),
                     "fp32 activations (QWEN_DRAFT_FP32)" if wide else "activations as the weights",
                     cfg.DRAFT_BLOCK or "adaptive", list(draft.config.target_layer_ids))
            if wide and not DRAFT_BITS:
                log.warning("QWEN_DRAFT_FP32 with QWEN_DRAFT_BITS=0: the drafter's weights stay "
                            "bf16 (slow on M1/M2: no native bf16)")
        if DRAFT_MTP and self.draft is not None:  # beside DFlash: QWEN_DRAFT_HYBRID
            self.mtp = load_mtp(DRAFT_MTP, self.model.language_model, FP16)
        elif DRAFT_MTP:
            log.warning("QWEN_DRAFT_MTP ignored: it goes beside a DFlash QWEN_DRAFT (an MTP head "
                        "alone is QWEN_DRAFT itself)")
        _tok = getattr(self.processor, "tokenizer", self.processor)
        # mlx-vlm стопит только на верхнем eos_token_id; у qwen их два
        # (<|im_end|> и <|endoftext|>), стоп на всех — как в HF/vLLM
        crit = _tok.stopping_criteria
        crit.eos_token_ids = sorted(set(crit.eos_token_ids or []) | _eos_ids(self.config))
        self.stopping = crit
        log.info("eos token ids: %s", crit.eos_token_ids)
        self._im_start = _tok.convert_tokens_to_ids("<|im_start|>")
        # the user role right after <|im_start|> (the prompt cache finds the
        # first user message by it); None: unknown, assume a system message
        role = list(_tok.encode("<|im_start|>user", add_special_tokens=False))
        self._im_user = role[1] if len(role) == 2 and role[0] == self._im_start else None
        self._span_tokens = _span_tokens(_tok, self._im_start)
        if self._span_tokens is None:
            log.warning("no <|im_start|>assistant header in the vocab: prompt_logprobs "
                        "without spans")
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
        if SPLIT_SPECIAL_TOKENS:
            try:
                t1 = time.time()
                self._plain = _PlainContent(_tok)
                log.info("special-token strings in user / tool text and tool-call arguments: "
                         "tokenized as plain text (%d added tokens, %.2f s)",
                         len(self._plain.tokens), time.time() - t1)
            except Exception:  # noqa: BLE001 — the normal tokenization stays
                log.exception("QWEN_SPLIT_SPECIAL_TOKENS: off (the tokenizer does not allow it)")
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
        log.info("prefill kernels: %s", _install_fast_paths())
        sizing._FIXED_SLAB = _fixed_slab()            # after the fused-SDPA probe; never changes after
        log.info("prefill slab: %s", f"{PREFILL_STEP} at any context (fp16 KV + fused SDPA)"
                 if sizing._FIXED_SLAB else f"{PREFILL_STEP}, shrunk to slab x offset <= {PREFILL_BUDGET}")
        if not _install_rows_attention(lm):
            log.warning("%s is not a qwen3_5 / qwen3_5_moe model: no batching (QWEN_MAX_BATCH=1)",
                        type(lm).__module__)
            self.max_batch = 1
        import pfqmm                       # after the fp16 cast (fp16 scales only); LM only
        pfqmm.install(lm)
        spec = self.draft is not None or self.mtp is not None
        if self.max_batch >= 3 or (spec and VERIFY_MR_HEAD):
            # after the fp16 cast (the kernel takes fp16 scales only); the
            # language model only (never the vision tower or the drafter,
            # which shares lm_head and calls it with (1, T, K): mrqmm only
            # with QWEN_VERIFY_MRQMM). Logs its coverage, or "off" for QWEN_MRQMM=0.
            import mrqmm
            if cfg.BATCH_INVARIANT:
                mrqmm.set_invariant(True)
                if self.max_batch > mrqmm.MAX_ROWS:
                    log.warning("QWEN_BATCH_INVARIANT=1: max_batch %d -> %d (mrqmm's rows)",
                                self.max_batch, mrqmm.MAX_ROWS)
                    self.max_batch = mrqmm.MAX_ROWS
            mrqmm.install(lm)
        log.info("GDN decode in-projections fused (QWEN_GDN_FUSE): %d layers", _install_gdn_fuse(lm))
        if cfg.BATCH_INVARIANT:
            self.max_batch = _invariant_check(lm, self.max_batch)
        if VERIFY_MRQMM not in ("0", "1", "head"):
            log.warning("QWEN_VERIFY_MRQMM=%r: not 0 / 1 / head, off", VERIFY_MRQMM)
        if spec and VERIFY_MR_HEAD:
            # never with QWEN_BATCH_INVARIANT (it loads no drafter)
            how = _setup_verify_mrqmm(lm, self.draft)
            log.info("DFlash through mrqmm (QWEN_VERIFY_MRQMM=%s): %s", VERIFY_MRQMM, how)
        if spec and cfg.VERIFY_QMV:
            log.info("verify through mrqmv (QWEN_VERIFY_QMV=%d): %s", cfg.VERIFY_QMV, _setup_verify_qmv(lm))
        if spec:
            nq = _probe_verify_attention(lm)
            log.info("DFlash verify attention (QWEN_VERIFY_ATT): %s",
                     f"{nq} queries per call (vector kernel, bit-equal to the one-query calls)" if nq
                     else "the library's one-query calls")
        cap = _batch_cap(int(mx.device_info()["max_recommended_working_set_size"]), _WIRED_GB)
        reserve = _prefill_reserve(lm.args.num_attention_heads)
        budget = _batch_budget(cap, mx.get_active_memory(), reserve)   # weights (+ drafter) = active now
        self.sched = _Scheduler(self, max_batch=self.max_batch, budget=budget, cap=cap,
                                prefill_reserve=reserve, on_fault=_exit_soon)
        self.max_batch = self.sched.fit_max_batch()
        if cfg.BATCH_INVARIANT:
            _log_invariant(lm, self.max_batch)
        log.info("batching: max_batch %d, KV budget %.2f GiB (cap %.1f GiB, prefill reserve %.1f GiB), "
                 "KV %d B/token, recurrent state %.1f MiB/row, reserve <= %d new tokens/row, "
                 "prefill share %.2f, prompt cache %s",
                 self.max_batch, budget / GIB, cap / GIB, reserve / GIB, self.sched.kv_q,
                 self.sched.gdn_row / 2**20, RESERVE_TOKENS, PREFILL_SHARE,
                 f"{self._pcache.bytes / GIB:.2f} GiB live inside the budget (soft cap "
                 f"{PROMPT_CACHE_GB:.1f} GiB, evicted on demand)" if PROMPT_CACHE else "off")
        log.info("cascade attention (QWEN_CASCADE): %s", self.sched.cascade_state())
        log.info("prompt cache move (QWEN_PROMPT_CACHE_MOVE): %s", self.sched.move_state())
        if self.sched.cascade_kernel:   # also when cascade is off now: set_cascade(True) later
            lm = self.model.language_model
            a = lm.args
            cascade_mma.warm(a.head_dim, a.num_key_value_heads, a.num_attention_heads,
                             lm.model.layers[self.sched.fa_layers[0]].self_attn.scale)
        if PROMPT_CACHE and pcdisk.DIR and pcdisk.ENV_ERRORS:
            log.error("prompt cache disk: off (QWEN_PROMPT_CACHE_DISK=%s): unparsable %s",
                      pcdisk.DIR, ", ".join(pcdisk.ENV_ERRORS))
        elif PROMPT_CACHE and pcdisk.DIR:
            try:
                disk = pcdisk.DiskTier(pcdisk.DIR, pcdisk.GB * GIB, self._disk_key(),
                                       seg_cls=_Seg, entry_cls=_Entry, keep=pcdisk.KEEP)
                disk.on_drained = self.sched.wake
                self._pcache.disk = disk
                log.info("prompt cache disk (QWEN_PROMPT_CACHE_DISK): %s, %.1f GiB, %s, entries "
                         ">= %d tokens, writer queue %.1f GiB, head waits <= %.0f s, F_NOCACHE %s",
                         pcdisk.DIR, pcdisk.GB, "kept across runs when the key matches"
                         if pcdisk.KEEP else "cleared at start", disk.min_tokens,
                         disk.queue_bytes / GIB, pcdisk.WAIT_S, pcdisk.NOCACHE)
            except Exception:  # noqa: BLE001 — the RAM cache works without it
                log.exception("prompt cache disk: off (%s unusable)", pcdisk.DIR)
        self.sched.start()
        self.ready = True
        log.info("model loaded in %d ms", self.load_ms)

    def _disk_key(self):
        """What the cached bits depend on (pcdisk validity key): the weights,
        the numerics knobs as resolved, every QWEN_* knob but the sizing /
        prompt-cache ones, the library versions and this code. A disk
        directory whose key differs is cleared at start."""
        import importlib.metadata
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/worker
        pkg = sorted(os.path.join("qwen_worker", n) for n in os.listdir(os.path.join(here, "qwen_worker"))
                     if n.endswith(".py"))
        src = hashlib.sha256()
        for f in ("worker.py", *pkg, "pfqmm.py", "mrqmm.py", "cascade.py", "cascade_mma.py", "pcdisk.py"):
            p = os.path.join(here, f)
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    src.update(fh.read())
        # weights: name, size and a digest of what identifies the tensors
        # without reading 10 GB: a .safetensors file's header (names, dtypes,
        # shapes, offsets: the 8-byte length + its JSON), a .json file whole
        weights = []
        for n in sorted(os.listdir(self.model_path)):
            p = os.path.join(self.model_path, n)
            if not n.endswith((".safetensors", ".json")) or not os.path.isfile(p):
                continue
            with open(p, "rb") as fh:
                if n.endswith(".json"):
                    head = fh.read()
                else:
                    raw = fh.read(8)
                    k = int.from_bytes(raw, "little") if len(raw) == 8 else 0
                    head = raw + fh.read(min(k, 1 << 28))
            weights.append((n, os.path.getsize(p), hashlib.sha256(head).hexdigest()))
        # the sizing / eviction-order knobs move no bit; QWEN_PROMPT_CACHE_REPLY
        # does (reply entries hold decode-kernel KV), so it is in `reply` below
        free = ("QWEN_PROMPT_CACHE", "QWEN_BATCH_MEM_GB", "QWEN_GEN_SLOTS", "QWEN_WIRED_LIMIT_GB",
                "QWEN_CACHE_LIMIT_GB", "QWEN_BATCH_RESERVE_TOKENS", "QWEN_BATCH_PREFILL_SHARE")
        env = {k: v for k, v in os.environ.items()
               if k.startswith("QWEN_") and not k.startswith(free)}
        return pcdisk.key_of(
            format=pcdisk.FORMAT, model=self.model_id, weights=weights, kv=KV_KWARGS, fp16=FP16,
            slab=[PREFILL_STEP, PREFILL_MIN_STEP, PREFILL_BUDGET, sizing._FIXED_SLAB],
            sdpa_fused=cfg.SDPA_FUSED, gdn_fast=GDN_FAST, gdn_fuse=GDN_FUSE, max_batch=self.max_batch,
            reply=PROMPT_CACHE_REPLY, env=env, mlx=mx.__version__,
            mlx_vlm=importlib.metadata.version("mlx-vlm"), src=src.hexdigest())

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
        if images and self.text_only:
            raise _Reject("invalid_request", f"{self.model_id} is loaded text-only "
                          "(QWEN_TEXT_ONLY=1): image inputs are not supported")
        if p.prompt_logprobs and images:
            raise _Reject("invalid_request", "prompt_logprobs is not supported with image inputs")
        if p.HasField("max_tokens") and p.max_tokens == 0 and not p.prompt_logprobs:
            raise _Reject("invalid_request", "max_tokens must be at least 1, got 0 "
                          "(0 = prefill only, with prompt_logprobs)")
        if p.prompt_cache not in PROMPT_CACHE_MODES:
            raise _Reject("invalid_request", f"prompt_cache must be one of \"\", \"off\", "
                          f"\"read\", got {p.prompt_cache!r}")
        if p.top_logprobs and not p.logprobs:
            raise _Reject("invalid_request", "top_logprobs needs logprobs")
        if p.top_logprobs > MAX_TOP_LOGPROBS:
            raise _Reject("invalid_request", f"top_logprobs must be at most {MAX_TOP_LOGPROBS}, "
                          f"got {p.top_logprobs}")
        # the ids index vocab-sized masks, and MLX indexing has no bounds check
        for name in ("allowed_token_ids", "stop_token_ids"):
            if any(t >= self.vocab for t in getattr(p, name)):
                raise _Reject("invalid_request", f"{name} contains out-of-vocab token id "
                              f"(vocab size {self.vocab})")
        try:
            grammar = self.grammar_for(p)
        except ValueError as e:
            raise _Reject("invalid_request", str(e)) from None
        messages, kwargs, add_gen = self._chat_inputs(req)
        try:
            formatted = apply_chat_template(self.processor, self.config, messages,
                                            add_generation_prompt=add_gen,
                                            num_images=len(images), **kwargs)
        except Exception as e:  # noqa: BLE001 — the template's own raise_exception, bad kwargs
            raise _Reject("invalid_request", f"chat template: {e}") from None
        return formatted, grammar

    def _chat_inputs(self, req, quiet=False):
        """The template's input -> (messages, kwargs, add_generation_prompt);
        raises _Reject. `quiet`: the second call for one request (no log)."""
        p = req.params
        kwargs = json.loads(p.chat_template_kwargs_json) if p.chat_template_kwargs_json else {}
        # vLLM's resolve_chat_template_kwargs: these two are refused, the rest is
        # kept only if the template uses it (the request's own fields win)
        bad = {"chat_template", "tokenize"} & kwargs.keys()
        if bad:
            raise _Reject("invalid_request",
                          f"Found unexpected chat template kwargs from request: {bad}")
        dropped = [k for k in kwargs if k not in self._template_vars]
        if dropped and not quiet:
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
        return messages, kwargs, add_gen

    def _encode_prompt(self, req, formatted):
        """A text prompt's ids: tokenizer.encode of the rendering, or with
        QWEN_SPLIT_SPECIAL_TOKENS the added-token strings of user / tool text
        and tool-call arguments as plain text (_PlainContent; the normal
        encode when there are none, or when the template treats the
        placeholders differently: logged)."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        plain = getattr(self, "_plain", None)
        if plain is None:
            return tokenizer.encode(formatted)
        from mlx_vlm.prompt_utils import apply_chat_template
        messages, kwargs, add_gen = self._chat_inputs(req, quiet=True)
        mark = plain.mark_for(formatted)
        masked, n = plain.mask(messages, mark)
        if not n:
            return tokenizer.encode(formatted)
        ids = None
        try:
            rendered = apply_chat_template(self.processor, self.config, masked,
                                           add_generation_prompt=add_gen, num_images=0, **kwargs)
            ids = plain.encode(formatted, rendered, mark)
        except Exception:  # noqa: BLE001 — the normal rendering worked: keep it
            log.exception("id=%s special-token strings: the placeholder rendering failed", req.id)
        if ids is None:
            log.warning("id=%s special-token strings in %d message texts: tokenized as control "
                        "tokens (the template renders them differently from placeholders)",
                        req.id, n)
            return tokenizer.encode(formatted)
        log.info("id=%s special-token strings in %d message texts: tokenized as plain text",
                 req.id, n)
        return ids

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
                self._limit(req.params, len(self._encode_prompt(req, formatted)))
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

    # --- log-probabilities (handler thread) ---------------------------------------
    def _token_strs(self, ids):
        """tokenizer.decode([id]) for each id (memoized)."""
        memo = self._tok_str
        miss = list({i for i in ids if i not in memo})
        if miss:
            tok = getattr(self.processor, "tokenizer", self.processor)
            bd = getattr(tok, "batch_decode", None)
            strs = bd([[i] for i in miss]) if bd is not None else [tok.decode([i]) for i in miss]
            memo.update(zip(miss, strs))
        return [memo[i] for i in ids]

    def _token_logprob(self, tok, lp, top):
        """A generated token's log p (+ top) -> pb.TokenLogprob."""
        strs = self._token_strs([tok] + [t for t, _ in top])
        return pb.TokenLogprob(token_id=tok, token=strs[0], logprob=lp,
                               top=[pb.TopLogprob(token_id=t, token=x, logprob=v)
                                    for (t, v), x in zip(top, strs[1:])])

    def _prompt_logprobs(self, req, r):
        """The PromptLogprobs event: every prompt token, the log p of
        positions first..len-1 (the scheduler's prefill) and the assistant
        messages' spans."""
        first, lps = r.plp
        ids = r.prompt_ids
        p = req.params
        gen = ((not p.HasField("add_generation_prompt") or p.add_generation_prompt)
               and not p.continue_final_message)
        spans = (_assistant_spans(ids, [m.role for m in req.messages], *self._span_tokens, gen=gen)
                 if self._span_tokens else [])
        log.info("id=%s prompt_logprobs: scored=[%d, %d) of %d spans=%d chunk=%d", req.id,
                 first, len(ids), len(ids), len(spans), LOGPROBS_CHUNK)
        return pb.Event(id=req.id, prompt_logprobs=pb.PromptLogprobs(
            first=first,
            tokens=[pb.TokenLogprob(token_id=i, token=x)
                    for i, x in zip(ids, self._token_strs(ids))],
            logprobs=lps,
            spans=[pb.Span(message=k, start=a, end=b) for k, a, b in spans]))

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
                       else self._encode_prompt(req, formatted))
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
            state = _GenState.for_prompt(ids, self._think)
            inside = state.in_thinking

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
            lp_pend = []      # logprobs: pb.TokenLogprob read since the last Token event

            def token(text, thinking):
                # the next Token event carries the logprobs read so far (not
                # aligned with its text; possibly with an empty text)
                ev = pb.Event(id=req.id, token=pb.Token(text=text, thinking=thinking, lp=lp_pend))
                lp_pend.clear()
                return ev

            def split(text):
                think_part, ans_part = state.feed(text)
                if think_part:
                    yield token(think_part, True)
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
                        yield token(ans_part, False)

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
                    if len(ev) > 3:                   # Params.logprobs
                        lp_pend.append(self._token_logprob(tok, *ev[3]))
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
                if lp_pend:
                    # no text went out for it (held back, buffered, an ignored
                    # EOS): the logprobs go now, on an empty Token
                    yield token("", state.in_thinking)
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
                if finish == "stop" and r.stop_lp is not None:
                    # the EOS / stop token the row ended on (a stop string's
                    # hit cuts what follows it, this token included)
                    lp_pend.append(self._token_logprob(*r.stop_lp))

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
                key = hashlib.sha256(f"{r.prompt_ids}|{sampling.get('seed')}|{answer}".encode()).hexdigest()
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
                    yield token(content, False)
                # vLLM: auto -> "tool_calls" whenever a call was parsed (even at the
                # length limit); required -> only instead of "stop"; named keeps it
                if calls and (choice == "auto" or (choice == "required" and finish == "stop")):
                    finish = "tool_calls"
            if lp_pend:
                yield token("", False)
            if p.prompt_logprobs and r.plp is not None:
                yield self._prompt_logprobs(req, r)
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

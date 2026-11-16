"""Env knobs (QWEN_*) and process-wide constants, read once at import.

Feature-local knobs stay with their code: QWEN_JSON_MAX_WS (grammar.py),
QWEN_LOGPROBS_CHUNK (logprobs.py). SDPA_FUSED, DRAFT_BLOCK and BATCH_INVARIANT
are also changed at load (kernels.py): other modules read them as `cfg.NAME`."""

import logging
import os

import cascade


log = logging.getLogger("qwen.worker")

# Repo-relative default: <repo>/weights/<name>. This file lives at
# src/worker/qwen_worker/, so the repo root is three directories up. Override
# with --model or QWEN_MODEL.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MODEL = os.environ.get(
    "QWEN_MODEL",
    os.path.join(_REPO_ROOT, "weights", "qwen36-35b-a3b-mlx-ud3"),
)

# Text-only mode (QWEN_TEXT_ONLY=1): a VL checkpoint is loaded without its
# vision tower, as a checkpoint without vision_config is (Ornith-1.5-9B): its
# `vision_tower.*` tensors (27B 0.86 GiB, 35B 0.83, 9B 0.85, bf16) are never
# read from disk, image requests are refused with the text-only checkpoint's
# 400. Text is unaffected (same language model, processor, template).
# Engine.load / _load_model.
TEXT_ONLY = os.environ.get("QWEN_TEXT_ONLY", "0") == "1"


# Special-token strings inside message TEXT (QWEN_SPLIT_SPECIAL_TOKENS, default
# 0 = as vLLM v0.30: the rendered prompt goes through tokenizer.encode, which
# matches every added token anywhere in it, so a tool output or a user message
# holding "<|im_end|>\n<|im_start|>system\n..." becomes real turn boundaries,
# "<tool_call>" / "<think>" real control tokens; HF's split_special_tokens does
# not help: it covers only special=True added tokens, Qwen's <tool_call>,
# <think>, <tool_response> are special=False). 1: the strings inside user and
# tool message content and inside tool-call arguments (names and values) are
# tokenized as plain text, the template's own control tokens by id, as
# DeepSeek Recipe does (DeepInfer doc/inference_deployment_qa.md:300). Ids
# change only for requests whose such text holds an added token's string;
# the rest are tokenized exactly as before (no extra work but a scan).
# Not for image prompts (the processor tokenizes them) nor assistant content
# / reasoning (the model's own output, the template splits <think> there).
# docs/perf/2026-10-24/design/special_tokens.md.
SPLIT_SPECIAL_TOKENS = os.environ.get("QWEN_SPLIT_SPECIAL_TOKENS", "0") == "1"

# Context ceiling (prompt + generation): bounded by KV-cache memory on 32 GB, not by the model.
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


# QWEN_SAMPLER_TOPK_FAST (0): with top_k > 0, min_p / top_p and the draw run
#   over the k survivors of one stable vocab argsort (_Sampler.survivors)
#   instead of topk + argsort + put_along_axis + a V-wide softmax / cumsum /
#   Gumbel draw: the same distribution (ties of the k-th value included),
#   another random stream for a given seed (a draw over k + 1 values), and
#   boundary rounding over k terms instead of V. Off = the vLLM-order path
#   above bit for bit. (DeepInfer finds the thresholds by radix select,
#   tilelang_ops/cuda/fused_sample.py.)
SAMPLER_TOPK_FAST = os.environ.get("QWEN_SAMPLER_TOPK_FAST", "0") == "1"

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
# hit. An entry whose ids are a prefix of a newer one (the previous turn of a
# conversation) is absorbed by it: its checkpoints (and its end state where
# that is a slab end of the new prompt) move to the new entry and it is dropped
# (not on quantized KV when only the old one is unquantized: it is the one
# resume point below quantized_kv_start). Past
# QWEN_PROMPT_CACHE_CKPTS an entry's checkpoints are thinned, never the end of
# the system block or the last three message starts (on quantized KV none
# below quantized_kv_start: never resumable there). An entry that absorbs an
# older one (a chain: a conversation / agent episode continuing turn by turn)
# keeps fewer: the end of the system block, the last
# QWEN_PROMPT_CACHE_CHAIN_MARKS message starts, the stride checkpoints in
# its last QWEN_PROMPT_CACHE_CHAIN_TAIL tokens (regenerate / an edited last
# turn) and those inside the first user message (a shared document: another
# conversation over it diverges anywhere in it; unprotected, thinned by the
# cap), plus its end state. The next turn resumes at the entry's last message
# starts, so older turns' boundaries and strides deep in the history were dead
# weight: ~60 % of an 11K agentic entry (9.7 x 49 MB of GDN states beside
# 355 MB of KV; its first user message is short, so the document strides cost
# nothing there; an 8K one keeps ~2-4 more states, ~100-200 MB). Fresh entries
# (nothing absorbed: a new conversation, a MISS) keep the cap-12 policy below
# QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS (16384; 0: always the cap policy): a sibling
# conversation over a short shared prompt still finds deep strides. From there
# on they take the chain policy too, plus the divergence point from the
# closest other entry and the strides inside the system block (as the first
# user message's: a sibling session diverges inside either): a fresh 50K
# entry after a MISS kept 12 states, ~0.6 GB of its ~2.3 GB (now 7); the
# episode's next turn resumes at its last message starts anyway. An entry
# stored for a cancelled prefill's prefix (below) does not count as absorbed:
# the retry that absorbs it takes the policy its own prompt would take cold.
# A text prefill cancelled after at least one slab (a client timeout on a
# long prompt), or failing outside the forward (the caches still hold exactly
# the prefix), stores the prefix it prefilled: ids = prompt[:pos], pos the last
# slab end (a _bounds cut: its KV is the cold prefill's bits there), the row's
# state at pos as the end state, the checkpoints taken so far, through the same
# store (budget, absorb, chain policy). A retry of that prompt resumes at pos
# (the 2026-10-17 AppWorld run lost 2 x 8 min re-prefilling a 114K prompt the
# client had cancelled). DFlash prefills too (since 2026-11-11), not vision
# (no store there either); quantized KV as a full store (resumable from
# quantized_kv_start on).
# Entries share their KV: an entry is a chain of immutable segments (buffers of
# their own, per full-attention layer); a store copies only the tokens past
# the resume point out of the row and reuses the parent's segments below it,
# so siblings over one system prompt hold it once and a chat turn adds its own
# tokens only. A restore concatenates the pieces once into the buffer the
# row's first update would have built (same shape, same bits); nothing ever
# holds a view of a row's buffer or of a segment. Bytes count every buffer
# once; a segment is freed with the last entry holding it.
# QWEN_PROMPT_CACHE_GB is a soft cap: past it entries are evicted at store
# (the newest stays, alone if need be); every eviction goes in
# _PrefixCache.victim's order (the entries the next requests would resume
# from last; stale ones LRU first — no hit / store within the last round of
# text lookups + stores, or idle for QWEN_PROMPT_CACHE_STALE_S; else the most
# recently used); the cache's
# live bytes count in the
# batch budget: a store evicts down to the batch's headroom (budget + prefill
# reserve - the rows' reservations; the new entry is dropped, the cache left as
# it was, if it alone is over), an admission evicts entries on demand
# (sparing the one it would resume from a full slab deep, and nothing when that
# would not admit it), a growing row too (never the parent of the running
# prefill). When only the entry the head of the queue would resume from keeps
# it out, the head waits for a row to end if that saves at least
# QWEN_PROMPT_CACHE_KEEP_TOKENS of prefill, for at most
# QWEN_PROMPT_CACHE_KEEP_WAIT_S; else (or then) the entry goes and the head is
# admitted cold. 8192 / 10 s: the 9B prefills 8K in ~14 s (35B ~13 s, 27B
# longer), so a hit that deep saves more than the longest wait costs; one slab
# (2048 tokens, ~3 s; 512 on the 35B recipe) saves less than waiting for a row
# that may decode for minutes. The library's own trim is never used (broken
# for quantized tuples).
PROMPT_CACHE = os.environ.get("QWEN_PROMPT_CACHE", "0") == "1"
PROMPT_CACHE_GB = float(os.environ.get("QWEN_PROMPT_CACHE_GB", "3"))
PROMPT_CACHE_STRIDE = int(os.environ.get("QWEN_PROMPT_CACHE_STRIDE", "2048"))
PROMPT_CACHE_CKPTS = int(os.environ.get("QWEN_PROMPT_CACHE_CKPTS", "12"))
PROMPT_CACHE_CHAIN_MARKS = int(os.environ.get("QWEN_PROMPT_CACHE_CHAIN_MARKS", "3"))  # < 0: off
PROMPT_CACHE_CHAIN_TAIL = int(os.environ.get("QWEN_PROMPT_CACHE_CHAIN_TAIL", "4096"))
# fresh entries of at least this many tokens take the chain policy (0: none)
PROMPT_CACHE_CHAIN_FRESH_TOKENS = int(os.environ.get("QWEN_PROMPT_CACHE_CHAIN_FRESH_TOKENS",
                                                     "16384"))
PROMPT_CACHE_STALE_S = float(os.environ.get("QWEN_PROMPT_CACHE_STALE_S", "300"))  # 0: off
PROMPT_CACHE_KEEP_TOKENS = int(os.environ.get("QWEN_PROMPT_CACHE_KEEP_TOKENS", "8192"))
PROMPT_CACHE_KEEP_WAIT_S = float(os.environ.get("QWEN_PROMPT_CACHE_KEEP_WAIT_S", "10"))
# Reply KV (QWEN_PROMPT_CACHE_REPLY, default 1 since the 2026-10-17 GPU gate,
# docs/perf/2026-10-17/reply_kv_gpu.md; 0 = off): when a text request ends
# (stop / length; not cancel, stop strings or errors), its entry is extended
# by the KV of the tokens the model consumed while answering (the prompt's
# last token + the generated ones it fed back: the step-ahead feeds the EOS
# it stops on, a sync row does not; a length row fed its last generated
# token too, the entry = prompt + completion) and the recurrent state after them as the
# new end state, so the next turn (prompt + reply + a new message: chats,
# agent loops) resumes after the reply instead of at the last message start.
# AppWorld traces: the template re-renders the reply to the same ids on 100 %
# of 1344 calls; 20.8 % of the prefilled tokens saved (148 / call;
# docs/perf/2026-10-17/design/reply_kv.md). Numerics: the reply's KV comes
# from decode kernels (qmv; mrqmm at B >= 3; cascade-released rows) and the
# next turn's first slab starts at the reply end, so a hit through a reply
# is not bit-identical to a cold prefill (the batched-decode numerics class:
# 9B gate, hit through a reply vs cold, top-1 99.88 %, KL 5.8e-5); with it on,
# a greedy answer can depend on what the cache holds (0: the bit-exact path).
# Skipped: DFlash rows, vision, rows still in a cascade group (logged), rows
# whose prefill entry was evicted / absorbed meanwhile, replies that generated
# a reasoning section (thinking on, as _GenState splits it: clients rarely
# send it back; a think id inside the content with thinking off is content,
# stored). A stop string usually ends the row as a
# cancel (nothing stored), but its cancel can come after the row already
# finished with stop / length: then the reply is stored with the tokens past
# the string and the next turn falls back (harmless). A regenerate of the same prompt resumes at
# the last message start (the prompt-end state is dropped with the extension).
PROMPT_CACHE_REPLY = os.environ.get("QWEN_PROMPT_CACHE_REPLY", "1") == "1"
# QWEN_PROMPT_CACHE_MOVE=1 (single stream: QWEN_MAX_BATCH=1, fp16 KV): one copy
# of a conversation's KV instead of two (the row's + the entry's). A hit moves
# the entry's KV into the row, layer by layer (the entry leaves the cache);
# nothing is copied at the prefill's end; when the row ends its own buffers
# become the entry (prompt, or prompt + reply). KV fp16 is 64 KiB/token on the
# 27B: 64K tokens = 4 GiB once instead of 8 (the cache refused such entries:
# "not kept"). Same bits as the copies. The price: while a request runs, the
# cache holds nothing of its conversation (a concurrent sibling / a regenerate
# sent before it ends is a miss), and a request that fails mid-forward loses
# the entry. Off: today's code path, untouched.
PROMPT_CACHE_MOVE = os.environ.get("QWEN_PROMPT_CACHE_MOVE", "0") == "1"

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
# only used when the request is greedy (temperature 0, no presence / frequency /
# repetition penalty: _Penalties.active) and text-only; otherwise the plain loop
# runs. The default QWEN_PRESENCE_PENALTY is 1.5, so set it to 0 (or the request
# must) for DFlash to run at all. A thinking budget is honoured
# inside the rounds (2026-11-11: the block is cut where the budget forces the
# next token, \n</think> go through plain steps; _Scheduler._spec).
# Output is identical to the plain greedy loop up to fp16 noise.
DRAFT = os.environ.get("QWEN_DRAFT", "")
# QWEN_DRAFT_BLOCK: tokens per verify round (the last accepted token + N - 1
# drafts), fixed. 0 = automatic: VERIFY_MR_BLOCK (8) when QWEN_VERIFY_MRQMM=1 and
# mrqmm covers the target (kernels._setup_verify_mrqmm sets it at load), else
# mlx-vlm's adaptive block (up to the drafter's trained size).
DRAFT_BLOCK = int(os.environ.get("QWEN_DRAFT_BLOCK", "0"))
# Drafter weight bits (0 = keep fp16). The target verifies every draft, so the
# drafter's precision only moves acceptance, not output: measured 4-bit ≈ 8-bit
# ≈ fp16 acceptance (~60%), and the round gets cheaper (fp16 ~100 ms → 4-bit
# ~85 ms): decode ×1.16–1.52 (fp16) → ×1.43–1.64 (4-bit).
DRAFT_BITS = int(os.environ.get("QWEN_DRAFT_BITS", "4"))
# QWEN_DRAFT_FP32 ("" = auto: on for DFlash2 drafters; 1 / 0 force): the
# drafter's activations in fp32 instead of fp16 (its weights stay quantized;
# scales, norms and kernels fp32, the selector codebooks fp16). The fp16 cast
# of QWEN_FP16 breaks z-lab/Qwen3.8-27B-DFlash2: its residual stream reaches
# 5.7e4 after the first layer and overflows fp16 (max 65504) in the next ones,
# every logit is NaN and no draft is ever accepted (measured 2026-11-10: 0/420
# drafts; in fp32 3.16 of 7 accepted at 4 bits vs 3.28 unquantized, the block
# 36 ms vs 32 in fp16). The 9B's DFlash drafter is fine in fp16 (~60 %).
# With QWEN_DRAFT_BITS=0 the weights stay as loaded (bf16: correct, but ~10x
# slower per block on M1/M2; fp32 weights would be 2x the memory).
DRAFT_FP32 = os.environ.get("QWEN_DRAFT_FP32", "")
# QWEN_DRAFT_MRQMM (1; 0 = off): the fp32 drafter's quantized linears (QWEN_DRAFT_FP32
# with QWEN_DRAFT_BITS 4 / 5 / 8) through mrqmm's wide variant for 1..16 rows:
# each row prescaled by an exact power of two to max |x| in [2^10, 2^11), cast
# to fp16, the kernel's fp32 sums written in fp32 times the inverse scale
# (kernels._drafter_linear), the scales / biases held in fp16 (checked exact).
# Instead of quantized_matmul in fp32 (dequantization-bound at 8 rows). Only
# the drafter: the target verifies, so acceptance is all it can move. 27B 4-bit
# + DFlash2 (2026-11-16, docs/perf/2026-11-16/drafter.md): drafter block 31.3 ->
# 17.4 ms, the same drafts on 240 probes, DFlash decode x1.117 in-run (code /
# en / ru 2K prompts x1.13 / x1.12 / x1.11). Read at call time (cfg.DRAFT_MRQMM:
# tools flip it; installed at load only when on).
DRAFT_MRQMM = os.environ.get("QWEN_DRAFT_MRQMM", "1") == "1"
# QWEN_DRAFT_COMPILE (0 / 1): the DFlash2 drafter's grouped dynamic causal
# convolution (2 per layer and call, ~8 elementwise ops each) as one compiled
# function. Block -0.65 ms, decode +0.3 % (under the 3 % gate): off.
# Read at call time (cfg.DRAFT_COMPILE).
DRAFT_COMPILE = os.environ.get("QWEN_DRAFT_COMPILE", "0") == "1"
# QWEN_DRAFT_TAIL: with the prompt cache, every checkpoint a DFlash row
# stores (the entry's end: QWEN_DRAFT_TAIL_END) carries the drafter's context there:
# the target's hidden states (the drafter's layers) of the last N positions.
# A row that resumes from the state starts its drafter with them; without, the
# drafter saw only the positions the hit left to prefill (1 on an exact
# repeat: 2.0 instead of 7.0 drafts accepted in the first rounds). 27B /
# DFlash2, 4 answers of 192 tokens replayed per context length
# (scripts/perf/dflash2/ctx_len.py): rounds 245 / 226 / 217 / 218 / 221 and
# drafts accepted in the first 8 rounds 2.3 / 3.5 / 5.3 / 5.3 / 5.4 at N = 1 /
# 16 / 32 / 128 / 2047 (the drafter's whole window = a cold run): 32 positions
# are a cold start, 128 is the margin. N x layers x hidden x 2 B per state
# (27B: 51 KiB per position, 6.6 MB at 128 beside its recurrent state). 0 = off.
DRAFT_TAIL = int(os.environ.get("QWEN_DRAFT_TAIL", "128"))
# QWEN_DRAFT_TAIL_END: the same for an entry's END state (the prompt's end: an
# exact repeat / regenerate; a reply's end: the next turn), where what comes
# next often copies from what is right before it (edit the previous answer):
# 27B, the next turn resuming at the reply's end with 128 positions there
# accepted 3.9 drafts per round where the cold run accepts 5.7 (52 rounds vs
# 38 for the same 256 tokens: the answer re-emits the previous one's code,
# half of it out of the drafter's sight); with the window kept, the cold
# run's rounds exactly (raw/cache_hit_reply*.txt). 27B: <= 105 MB.
# At most the drafter's window - 1 positions are kept (all it looks at: then
# a hit at the end starts the drafter exactly as a cold run). One end state
# per entry. 0 = off.
DRAFT_TAIL_END = int(os.environ.get("QWEN_DRAFT_TAIL_END", "2048"))
# Sampling / presence-penalty requests can take the speculative path too
# (QWEN_DRAFT_SAMPLING=1, exact "sample-match" acceptance, see _Scheduler._spec), but a
# round (draft ~15 ms + a 4..16-row verify, flat ~2.6x a 1-row step on M1: the
# 8-bit qmm is dequantization-bound from 6 to 32 rows) costs about 3 plain
# steps and acceptance drops with temperature: measured x1.1-1.2 on code and
# x0.8 on prose (accept ~1.1/round), so it is off by default. Falling back to
# plain decoding on low acceptance was tried and lost: a probe of 8 rounds
# costs ~0.5 s, and the drafter restarts cold (~130 ms rounds, lower accept).
DRAFT_SAMPLING = os.environ.get("QWEN_DRAFT_SAMPLING", "0") == "1"
# QWEN_LOOKUP: prompt lookup decoding (lookup.py, _Scheduler._spec): a round's
# draft is the continuation of the latest earlier occurrence of the text's
# last QWEN_LOOKUP_N..QWEN_LOOKUP_NMAX tokens (prompt + output; no MLX op),
# verified by the target as a drafter's block (same verify path, same
# exactness class). Used when the match is >= QWEN_LOOKUP_MIN tokens and
# leaves >= 2 drafts (a verify block of >= 3 rows), at most QWEN_LOOKUP_K
# drafts (<= 15: mrqmm's 16 rows).
#   "0" (default) off.
#   "draft": rows the drafter takes only (below); plain rows stay the plain loop.
#   "1": with a drafter (QWEN_DRAFT), a lookup round when a match exists, else
#        the drafter's block (its context = every position since its last
#        block); without a drafter (a lone greedy row, fp16 KV, no thinking
#        budget: _spec_ok), lookup rounds between plain steps (step-ahead bursts
#        of <= QWEN_LOOKUP_BURST steps, the plain loop's sampler: bit-identical
#        to it between rounds; the rounds' verify rows are not: DFlash's class).
#   "splice": with a drafter, the drafter's block every round; where a match
#        exists and its continuation agrees with the drafter on the first
#        token, the (longer) continuation instead (a host read of the drafts).
LOOKUP = os.environ.get("QWEN_LOOKUP", "0") or "0"
LOOKUP_N = int(os.environ.get("QWEN_LOOKUP_N", "3"))
LOOKUP_NMAX = int(os.environ.get("QWEN_LOOKUP_NMAX", "32"))
LOOKUP_MIN = int(os.environ.get("QWEN_LOOKUP_MIN", "4"))
LOOKUP_K = min(15, int(os.environ.get("QWEN_LOOKUP_K", "15")))
# QWEN_LOOKUP_K0: adaptive drafts per lookup round: start at K0, 2k+1 after a
# round whose drafts were all accepted (up to QWEN_LOOKUP_K), back to K0 after
# one that was not (a match into text that does not copy costs a short block).
# 0 = always QWEN_LOOKUP_K. 27B 4-bit verify (mrqmm buckets): 4 rows 99 ms,
# 8 rows 106, 12 rows 155, 16 rows 159 (a plain step 55): so 7, then 15.
LOOKUP_K0 = int(os.environ.get("QWEN_LOOKUP_K0", "7"))
# QWEN_LOOKUP_BACKOFF: a lookup round that kept fewer than half its drafts
# doubles the match length the next one needs (up to QWEN_LOOKUP_NMAX), a fully
# accepted one resets it to QWEN_LOOKUP_MIN (_Gate). 0 = off.
LOOKUP_BACKOFF = os.environ.get("QWEN_LOOKUP_BACKOFF", "1") == "1"
LOOKUP_BURST = max(1, int(os.environ.get("QWEN_LOOKUP_BURST", "16")))
# MTP drafting (qwen_worker/mtp.py): QWEN_DRAFT may also name the target's own
# MTP head split into a drafter dir (scripts/perf/mtp/split_mtp.py; the kind is
# detected from its config), and QWEN_DRAFT_MTP loads one beside a DFlash
# QWEN_DRAFT. QWEN_MTP_K: drafts per MTP round (chained: k head steps + k
# lm_heads, then a (k + 1)-token verify). QWEN_MTP_REFEED=1: every accepted
# position's head entry is recomputed from the target's hidden state (default:
# the chained steps' entries are kept, as mlx-vlm does). QWEN_DRAFT_HYBRID with
# both drafters: "1" = per round the one with the larger expected tokens per
# second (mtp.Chooser), "dflash" / "mtp" = always that one.
DRAFT_MTP = os.environ.get("QWEN_DRAFT_MTP", "")
MTP_K = int(os.environ.get("QWEN_MTP_K") or "0")
# QWEN_MTP_K=0: adaptive (mtp.MTPDraft.next_k), QWEN_MTP_COST = one draft's cost
# as a fraction of a round's verify
MTP_COST = float(os.environ.get("QWEN_MTP_COST") or "0.06")
# QWEN_MTP_KMAX: the adaptive draft count's ceiling (a verify block of KMAX + 1
# rows; mrqmm's cost is flat over 3..8 rows on the 27B)
MTP_KMAX = max(1, min(15, int(os.environ.get("QWEN_MTP_KMAX") or "4")))
MTP_REFEED = os.environ.get("QWEN_MTP_REFEED", "0") == "1"
DRAFT_HYBRID = os.environ.get("QWEN_DRAFT_HYBRID") or "1"
# QWEN_SPEC_PLAIN=1: plain steps are an arm of the per-round choice (mtp.Chooser)
# beside the loaded drafter(s): a request whose drafts are rarely accepted
# (short Russian prose on the 27B: DFlash below plain) decodes mostly plain,
# re-probing the drafters every mtp.Chooser.EXPLORE tokens.
SPEC_PLAIN = os.environ.get("QWEN_SPEC_PLAIN", "0") == "1"
# QWEN_VERIFY_QMV=N (2..3; 0 = off): verify blocks of 2..N tokens (MTP rounds:
# QWEN_MTP_K + 1) on 4 / 5 / 6 / 8-bit g64 modules through mrqmv.py, a multi-row qmv whose
# every row is MLX's solo qmv bit for bit (self-tested per shape at load):
# the weights read once for the T rows, instead of mrqmm's MMA (rows padded
# to 8, a flat ~x1.6 of a solo row) or mlx-vlm's exact kernels (T unpacks).
VERIFY_QMV = int(os.environ.get("QWEN_VERIFY_QMV") or "0")

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
#
# QWEN_MRQMM5 (default 0, read by mrqmm at load): the same kernel for 5-bit
# affine g64 modules (the 27B: every linear incl. lm_head and the fused GDN
# in-projection), 3..16 rows; with it the batched decode, the fused GDN
# in-projection (_gdn_call) and, with a drafter, the DFlash verify block and the
# drafter's lm_head (QWEN_VERIFY_MRQMM) take the kernel on 5-bit models exactly
# as they do on 8-bit ones. 0 = 5-bit calls stay on quantized_matmul, bit for
# bit as before. Ignored with QWEN_BATCH_INVARIANT=1 (5-bit modules stay
# uncovered there: max_batch 2). Kernel numbers: scripts/perf/dflash2/.
#
# QWEN_MRQMM4 (default 0, read by mrqmm at load): the same for 4-bit affine g64
# modules (the lmstudio-community 4-bit 27B, weights/qwen38-27b-mlx-4bit: every
# linear 4-bit), everywhere QWEN_MRQMM5 takes 5-bit ones. A knob of its own:
# gated on the 27B 4-bit (docs/perf/2026-11-16/profile_27b.md §5: DFlash x1.75,
# KL <= 1e-5, identity 12/16 at near-ties; default stays 0), and a model mixing widths (the 35B's 4-bit shared expert)
# must not be pulled in by QWEN_MRQMM5. Self-tested on the GPU at load before
# its first dispatch (mrqmm._selftest: the unpack exactly + real-scale rows vs
# an fp32 reference); a failure leaves every 4-bit call on quantized_matmul
# (load log "self-test FAILED"). Ignored with QWEN_BATCH_INVARIANT=1. Gate:
# scripts/perf/dflash2/gate_mrqmm4.sh.
#
# QWEN_MRQMM6 (default 0, read by mrqmm at load): the same for 6-bit affine g64
# modules (the mixed-precision oQ4e 27B, weights/qwen38-27b-mlx-oq4e: its
# lm_head and the last layer's k / v). Above all the DFlash verify block's head
# and the drafter's head: mlx-vlm's exact verify kernels take only 4 / 5 / 8
# bits, so without it a 6-bit head is read once per block token. Self-tested
# like 4 bits. Ignored with QWEN_BATCH_INVARIANT=1. Gate:
# scripts/perf/dflash2/gate_oq4e.sh (docs/perf/2026-11-15/design/oq4e_speed.md).

# Prefill matmul (QWEN_PFQMM, default 1, read by pfqmm.install at load; 0 =
# always quantized_matmul): on M1/M2 MLX's qmm_t uses 32x32 output tiles and
# reaches ~8.1 TFLOPS on prefill slabs (fp16 GEMM: ~9.5). pfqmm runs the same
# kernel code (the wheel's headers) with 64-row tiles: x1.03-1.28 per matmul
# for 3..8-bit g64 (8-bit x1.10-1.14), bit-identical to quantized_matmul (same
# per-element K order), and only where MLX itself would run qmm_t (>= 64
# rows, not split-K; the last tile is MLX's own 32-row one when <= 32 rows are
# left), self-tested per (bits, group) at load. 9B
# 2048-token slab, in-process A/B: x1.12 at offset 0, x1.17 at 8K/32K.

# Batch-invariant decode (QWEN_BATCH_INVARIANT, default 0; docs/perf/2026-10-24/
#   design/batch_invariant.md): a row's logits are the same bits whatever the
#   batch around it (1..16 rows, any lengths and mix), so a T=0 answer under
#   load is the answer the request gets alone, without QWEN_MAX_BATCH=1/2.
#   What depends on B by default, and what the mode does about it:
#   * mrqmm splits K over 8 simdgroups at <= 8 rows and 4 at 9..16 (another
#     sum order), and B <= 2 runs MLX's qmv (its own): mrqmm.set_invariant
#     fixes the split per K and takes 1..16 rows and every 8-bit module (lm_head,
#     the fused GDN in-projection and narrow layers too); QWEN_MAX_BATCH is
#     capped at 16 (mrqmm.MAX_ROWS). The cost is mrqmm vs qmv at B = 1 (~3 % of
#     the step's linears, mrqmm.md) and KS 8 vs 4 at 9..16 rows (within 3 %).
#   * cascade groups (another attention composition): off.
#   * DFlash (its verify kernels are not the decode's): not loaded.
#   Everything else was checked to be per-row already: the B = 1 library call
#   vs the RowsKV step (same per-row update_and_fetch + SDPA, positions explicit
#   both ways), norms, RoPE, the stacked GDN conv + recurrence (one threadgroup
#   per row and head), the logits slice, sample_rows (grouped draw == solo) and
#   the logprobs (tests/test_batch_invariant*.py). Not covered (logged at load,
#   mrqmm.uncovered): non-8-bit layers (27B, 35B experts / shared expert /
#   lm_head; MLX qmv per row up to its threshold, then qmm), MoE expert sorting
#   from B * top_k >= 64. Prefill was already batch-independent (one request at
#   a time, slab cuts fixed by _bounds); a 1-token slab now takes mrqmm too.
#   Not numerics but still run-dependent: a hit through a reply's KV
#   (QWEN_PROMPT_CACHE_REPLY=1) is not a cold prefill's bits and whether it hits
#   depends on evictions, so reproducible runs still want REPLY=0; a row that
#   outgrows the budget ends with "length" (_regrow) depending on its neighbours.
BATCH_INVARIANT = os.environ.get("QWEN_BATCH_INVARIANT", "0") == "1"
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
# QWEN_PREFILL_LANE (2048 since 2026-10-24; 0 = off): a second, priority prefill for short TEXT
#   requests. While a prefill runs (the main task: one request, slab by slab,
#   e.g. a 110K MISS for minutes), the first waiting text request with at most
#   this many tokens to prefill from where it would resume (its prompt-cache
#   hit, else 0) is admitted beside it (_admit_lane) and prefilled before the
#   main prefill's next slab: a 300-token suffix after a 20K hit takes ~1 s
#   instead of a wait for the whole long prefill (the 600 s client timeouts of
#   the 110K episode, 2026-10-17). Tokens, not slabs: a next turn resumed after
#   its reply is cut at the new message starts (_bounds), so 300 tokens are
#   often 2-3 slabs, the first a few tokens long. It may overtake waiting
#   requests that do not qualify, only when the budget holds it AND the head
#   of the queue; the others keep their FIFO order. Same admission rules as
#   any request (budget, live check, prompt-cache eviction) except it never
#   gives up its hit (without it, it would be a long one). Slab ends are each
#   prompt's own (_bounds): the main prefill is cut exactly as without the
#   lane. Recommended: PREFILL_STEP (2048): at most a slab's work per lane
#   request.
# QWEN_PREFILL_LANE_BURST (4): lane requests admitted in a row before the
#   main prefill gets its next slab (a lane request, once admitted, runs to
#   its end first: <= QWEN_PREFILL_LANE tokens): the lane never starves it.
PREFILL_LANE = max(0, int(os.environ.get("QWEN_PREFILL_LANE", "2048") or "0"))
PREFILL_LANE_BURST = max(1, int(os.environ.get("QWEN_PREFILL_LANE_BURST", "4") or "4"))
# QWEN_PREEMPT (2 since 2026-10-24; 0 = off): a row that outgrows the batch's headroom while
#   other rows run is preempted instead of ended with "length" (a cut tool
#   call for an agent), at most this many times per request: its KV (prompt +
#   the tokens it consumed) goes to the prompt cache as a reply-KV entry, the
#   request goes back to the HEAD of the queue with ids = prompt + generated
#   and resumes from that entry (nothing to prefill: the entry ends where the
#   next decode input goes). The victim is the growing row itself (the one
#   cut today: the other rows are untouched). The client sees one stream: the
#   handler's detokenizer / stop filter / reasoning split never notice, the
#   scheduler carries the sampler (its key: the seeded stream continues), the
#   penalty counts, _Control's counters, the grammar matcher and the thinking
#   budget (on the request). Not for vision, DFlash, cascade-grouped rows,
#   prompt_logprobs or prompt_cache "read" / "off" requests (they are cut as
#   before); see _preempt / _preemptable.
PREEMPT = max(0, int(os.environ.get("QWEN_PREEMPT", "2") or "0"))
MAX_BATCH = max(1, int(os.environ.get("QWEN_MAX_BATCH", "16")))
PREFILL_SHARE = min(1.0, max(0.05, float(os.environ.get("QWEN_BATCH_PREFILL_SHARE", "0.5"))))
BATCH_MEM_GB = float(os.environ.get("QWEN_BATCH_MEM_GB", "0") or "0")
# QWEN_PREFILL_RESERVE_GB: the memory kept outside the batch budget for a
# prefill slab's activations, instead of the estimate (_prefill_reserve). The
# estimate counts the attention scores + ~0.4 GiB of MLP; a DFlash prefill of
# the 27B (the drafter's hidden states, 5120-wide MLP at 2048 rows) peaked
# 2.0-2.7 GiB above its KV against an estimate of 1.15 (stand log 2026-11-14).
PREFILL_RESERVE_GB = float(os.environ.get("QWEN_PREFILL_RESERVE_GB", "0") or "0")
RESERVE_TOKENS = max(1, int(os.environ.get("QWEN_BATCH_RESERVE_TOKENS", "4096")))
GEN_SLOTS = int(os.environ.get("QWEN_GEN_SLOTS", "0") or "0")
SCHED_LOG_S = 10.0
# QWEN_DEBUG_NO_SYNC=1 (debug / tests; DeepInfer's forbid_cuda_sync): while a
#   decode step is built (_Scheduler._step, up to its async_eval) every host
#   wait on the GPU raises -- mx.eval / mx.synchronize, mx.array .item() /
#   .tolist() / int() / float() / bool() / operator.index(), np.asarray /
#   np.array of an mx.array. The step-ahead lives on that: one read of a
#   step's outputs before the next step is queued (the old concatenate +
#   tolist, found by hand: step_profile.md) and the GPU idles while the host
#   builds the next one. Only the scheduler thread is checked (handler
#   threads read numpy from MLX constants); off = no wrapper installed at all.
DEBUG_NO_SYNC = os.environ.get("QWEN_DEBUG_NO_SYNC", "0") == "1"


GIB = 1024**3             # every *_GB knob and the batching logs are in GiB; Usage.peak_gb is 1e9
SCHED_STALL_S = 300.0     # a tick running this long marks the scheduler wedged (Health: not ready)
SCHED_MAX_FAILURES = 3    # consecutive failed decode steps before the scheduler gives up
EVENT_POLL_S = 5.0        # a handler checks the scheduler is alive this often while it waits

# QWEN_CASCADE (1; GPU gate B4 passed: docs/perf/2026-10-12/cascade_gpu.md):
#   cascade decode attention (cascade.py; design: docs/perf/2026-10-12/
#   design/cascade_plan.md, Part B) for batch rows sharing a prompt prefix.
#   When >= QWEN_CASCADE_MIN_ROWS (8; 1..2 count as 3, 0 = off) running rows
#   share prompt ids [0, P) with P >= QWEN_CASCADE_MIN_PREFIX (4096; 0 = off)
#   and P a slab end of every one of them (then their KV [0, P) are the same
#   bits: _bounds, the prompt cache's hit == cold), the scheduler forms a
#   group: the prefix's KV once (fp32), the members' own tokens in one fp32
#   stack per layer, one attention for the group per layer; the members'
#   contiguous KV caches are freed. Memory rule (_Scheduler._group_wins):
#   the group's arrays at the members' REAL lengths (+ its transients, and
#   never less than what turning back into 2 contiguous rows takes) take no
#   more than their own fp16 caches would; the charge stays reservation-
#   based (_group_bytes) and a formation / join that would raise the
#   reservations is taken only while they + the prompt cache's live bytes
#   stay within the budget. Only with fp16 KV (QWEN_KV_BITS=0),
#   QWEN_PROMPT_CACHE=1 and max_batch >= 3 (else off whatever the switch),
#   for text-only rows that resumed from the prompt cache or stored an
#   entry; never vision or DFlash rows. QWEN_CASCADE=0 is the per-row path
#   bit for bit.
#   B <= 2 is never grouped (today's bit-exact path); ungrouped rows of a
#   mixed batch take today's per-row calls (bit-identical). Grouped rows are
#   NOT bit-equal to their per-row attention (fp32 statistics, another
#   summation order: rms ~2.1e-4 vs ~2.8e-4 for the per-row kernel against
#   an fp32 reference on the 9B layout), so with it on a B >= 3 greedy
#   answer also depends on the prompt cache's contents and arrival order.
# QWEN_CASCADE_CHECK (0): check each new member's KV [0, P) bitwise against
#   the group's prefix (a host sync per join; tests / validation runs). A
#   mismatch is logged and the row stays ungrouped.
def _cascade_knobs(env):
    """(on, min rows, min prefix, check) from QWEN_CASCADE*. Switches are on
    only for "1", as QWEN_PROMPT_CACHE / QWEN_DRAFT_SAMPLING; QWEN_CASCADE
    unset or empty = "1" (the default), anything else ("0", "false") off."""
    on = (env.get("QWEN_CASCADE") or "1") == "1"
    rows = int(env.get("QWEN_CASCADE_MIN_ROWS") or "8")
    prefix = int(env.get("QWEN_CASCADE_MIN_PREFIX") or "4096")
    return on, (max(3, rows) if rows > 0 else 0), max(0, prefix), \
        env.get("QWEN_CASCADE_CHECK", "0") == "1"


CASCADE, CASCADE_MIN_ROWS, CASCADE_MIN_PREFIX, CASCADE_CHECK = _cascade_knobs(os.environ)
# QWEN_CASCADE_KERNEL (0): a group's prefix stays fp16 and is attended by
#   cascade_mma (simdgroup MMA, split over the prefix; design:
#   docs/perf/2026-10-24/design/cascade_mma.md) instead of an fp32 copy with
#   materialized scores. The prefix is the donor row's own KV buffer, aliased
#   (cascade.PrefixGroup.build(kernel=True): no formation copy) when at most
#   CASCADE_ALIAS_SLACK slots lie past P, else a P-slot fp16 copy; the charge
#   (_group_bytes) counts P + CASCADE_ALIAS_SLACK fp16 tokens -- half of the
#   fp32 prefix, so groups also win at fewer rows (_group_wins). Numerics: the
#   same products (fp16 q and K/V, exact in fp32), another summation order, no
#   rounding below fp32 before the output: the class of today's cascade. fp16
#   KV only (the scheduler leaves it off otherwise, the load log says why);
#   "1" only, anything else = today's fp32 path bit for bit.
def _cascade_kernel_knob(env):
    return env.get("QWEN_CASCADE_KERNEL", "0") == "1"


CASCADE_KERNEL = _cascade_kernel_knob(os.environ)
CASCADE_ALIAS_SLACK = cascade.ALIAS_SLACK   # kernel mode (tests: 16)
CASCADE_CHUNK = cascade.CHUNK         # prefix chunk (tests: 16)
CASCADE_STEP = cascade.STEP           # suffix stack growth (tests: 16)
CASCADE_SLACK = 64 << 20              # fixed part of a group's step transient (tests: 0)


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
# QWEN_SDPA_FUSED_MIN_KV (32768): the fused kernel from this many keys on. Below
#   it the default path is 5-12 % faster per call but materializes slab x keys
#   x heads scores (2 B each): 2048 x 31K x 24 heads = 2.8 GiB on the 27B, the
#   prefill's memory hump (with the drafter, fp16 KV and the prompt cache a
#   60K prefill ran out of memory there). 8192 with QWEN_PREFILL_BUDGET =
#   2048 x 8192 = 16777216 caps the scores at 0.75 GiB and the prefill reserve
#   at 1.15 GiB instead of 3.4. Changes the slab kernels' bits past the
#   threshold (KL fused vs unfused 7e-6), not the slab cuts at the defaults.
SDPA_FUSED_MIN_Q = 512
SDPA_FUSED_MIN_KV = int(os.environ.get("QWEN_SDPA_FUSED_MIN_KV") or "32768")
# QWEN_KVQ_PREFILL_FUSED (0): quantized KV (QWEN_KV_BITS 8 / 4) past
#   quantized_kv_start: a prefill slab that the fp16 path would send to the
#   fused kernel (fp16 queries, causal, >= SDPA_FUSED_MIN_KV keys, >= 512
#   queries or scores over PREFILL_BUDGET) dequantizes the layer's cached
#   keys / values to fp16 (mx.dequantize: the exact values the quantized
#   matmuls use) and runs MLX's fused SDPA over them, instead of the
#   library's quantized SDPA (two quantized matmuls over a materialized fp16
#   slab x keys x heads score matrix). The stored cache stays quantized. With
#   it _step keeps quantized slabs at PREFILL_STEP past the threshold, as for
#   fp16 KV (frozen at load: _FIXED_SLAB), and the prefill reserve holds one
#   layer's dequantized K + V at QWEN_MAX_CONTEXT keys x 1.6, the measured
#   slab transient (27B: 2 x 128K x 4 x 256 x 2 B = 512 MiB -> 819 MiB). Numerics: attention over the dequantized KV with
#   fp32 scores inside the kernel -- closer to the fp32 reference over the
#   same dequantized KV than the quantized SDPA, whose scores and
#   probabilities are rounded to fp16 (tests/test_kvq8.py); not the old
#   path's bits, hence off until the GPU gate (docs/perf/2026-11-15/design/
#   kv_q8.md). Decode / DFlash verify keep the library's quantized calls.
KVQ_PREFILL_FUSED = os.environ.get("QWEN_KVQ_PREFILL_FUSED", "0") == "1"
GDN_FAST = os.environ.get("QWEN_GDN_FAST", "0") == "1"
GDN_FAST_MIN_T = 16
VERIFY_EXACT_MAX_T = int(os.environ.get("QWEN_VERIFY_EXACT_MAX_T") or "8")
# QWEN_VERIFY_ATT: a DFlash verify block's attention through MLX's "vector"
#   attention kernel, several queries per call. mlx-vlm verifies a block of T
#   tokens with T one-query calls per full-attention layer (each bit-equal to
#   a 1-token decode), so a round reads the row's whole KV T times: at 60K keys
#   (27B) 25 ms a layer for a block of 8 where a decode step's one query takes
#   2.9. The vector kernel (mx.fast SDPA, mask "causal", queries x GQA factor
#   <= 32: 5 queries on the 27B, 8 on the 9B) gives the same bits as the
#   one-query calls (checked at load on random data, _probe_verify_attention:
#   else off) and reads the KV once: 3.4 / 4.6 ms for 3 / 5 queries. A longer
#   block goes in chunks. fp16 KV, a lone row. 0 = the library's calls.
#   "2": the block's G x T query rows per kv head attend the KV together in one
#   pass (vatt.py: steel-attention MMA loop over key chunks with the block's
#   causal tail, fp32 partials + merge), the KV read once a round whatever T
#   (<= 16, head_dim 256) -- not the per-query calls' bits (the fused-attention
#   numerics class: fp32 throughout, error vs an fp32 reference <= MLX's own
#   kernel's); self-tested at load against an fp32 reference for every block
#   size (_probe_verify_attention), else "1". docs/perf/2026-11-16/verify_att2.md
#   Default "2" since 2026-11-16: DFlash x1.35 at 32K / x1.48 at 60K on the 27B,
#   equal at short context, tokens == plain on every long run; on 16 short tasks
#   12/16 == plain greedy, exactly as "1" on the 4-bit 27B (the mrqmm verify
#   decides those near-ties, gaps <= 0.016, not the attention). "1" = the
#   library's bits per query.
_VERIFY_ATT_MODE = os.environ.get("QWEN_VERIFY_ATT", "2")
VERIFY_ATT = _VERIFY_ATT_MODE in ("1", "2")
VERIFY_ATT2 = _VERIFY_ATT_MODE == "2"
# QWEN_DECODE_ATT (0 = off): a decode step's attention over >= this many fp16
#   keys through the same one-pass kernel (vatt.py, one query: the G query
#   heads of a kv head as one 8-row MMA block, the KV read once, fp32 partials)
#   instead of MLX's 2-pass vector SDPA, which reads the KV at ~90 GB/s on the
#   27B layout (G 6: one simdgroup per query head over strided keys): one layer
#   at 60K keys 2.85 -> 0.99 ms, at 8K equal (microbench, verify_att2.md). Every
#   decode row (a lone row's step and each row of a batch), fp16 KVCache only;
#   not MLX's bits (fp32 partials; error vs fp32 <= MLX's). Self-tested at load
#   (_probe_decode_att), else off.
DECODE_ATT_MIN_KV = int(os.environ.get("QWEN_DECODE_ATT") or "0")
# QWEN_VATT_Q8 (1; "0" = off): the one-pass kernel (vatt.py) over an 8-bit QuantizedKVCache
#   (QWEN_KV_BITS=8, group 64, fp16 scales; head_dim 256): the cache's own
#   (packed, scales, biases) buffers are read in place and dequantized in
#   registers into the kernel's fp16 key / value tiles, the rest of the kernel
#   is the fp16 one. Replaces the library's quantized SDPA (two quantized
#   matmuls + a softmax per query: a verify block of 8 = 8 calls, the KV read
#   8 times). "1" = DFlash verify blocks (T 2..16) and decode steps (one query:
#   a lone row's, or each row of a batch), "verify" / "decode" = one of them.
#   Before the cache is quantized (quantized_kv_start) it is a plain KVCache
#   and QWEN_VERIFY_ATT / QWEN_DECODE_ATT apply. Numerics: the fp16 kernel's
#   class over the dequantized KV (fp32 partials; not the library's bits).
#   Self-tested at load against an fp32 reference on dequantized KV
#   (_probe_vatt_q8), else off. Default on since 2026-11-16 (27B oQ4e, 32K:
#   KL vs the library 2.15e-5 / top-1 99.80 % verify, 1.67e-5 / 100 % decode;
#   plain x1.42, DFlash x1.54 vs the library): docs/perf/2026-11-16/vatt_q8.md
_VATT_Q8_MODE = os.environ.get("QWEN_VATT_Q8", "1")
VATT_Q8_VERIFY = _VATT_Q8_MODE in ("1", "verify")
VATT_Q8_DECODE = _VATT_Q8_MODE in ("1", "decode")


# QWEN_VERIFY_MRQMM (1): DFlash rounds through mrqmm, the batched-decode
#   kernel. "1": a verify block of 3..16 tokens (1, T, K) runs every linear
#   mrqmm.install covered (8-bit affine g64 fp16, N >= 512: on the 9B all of
#   them incl. lm_head; the GDN in-projections as their fused concatenation,
#   b/a riding along as in decode) as T decode rows (T, 1, K) through the
#   module's own __call__ (whatever row rules / configs mrqmm has); 1..2-token
#   blocks, uncovered modules and T > 16 keep the path above
#   (QWEN_VERIFY_EXACT_MAX_T then governs only them), and the drafter's lm_head
#   (the target's 8-bit head, (1, bs - 1, K)) too. "head": the drafter's
#   lm_head only. M1 Max, 9B linears of one forward (docs/perf/2026-10-12/
#   microbench.md 1a, mrqmm.md): exact qmv 63-73 ms at T = 8, quantized_matmul
#   85-90 at 8..16, mrqmm 33 at 8 rows and 46 at 16 (lm_head alone 9 -> 3.1 /
#   4.6 ms: the drafter's head is ~9 of its ~15 ms block). Cost of "1":
#   a verify row is no longer bit-identical to a 1-token qmv decode, so DFlash
#   output != plain greedy byte for byte (the batched-decode numerics class,
#   9B teacher-forced top-1 >= 99.7 %); "head" keeps DFlash == greedy (the
#   target verifies; the drafts only move acceptance and the adaptive block).
#   Default 1 (GPU 2026-10-24: DFlash x1.47 with the fixed block 8 below, 15/16
#   answers == plain greedy vs 13/16 today); without a drafter nothing changes.
#   Installs mrqmm also at max_batch < 3 when a drafter is loaded (decode at
#   B <= 2 never takes it). docs/perf/2026-10-24/design/verify_mrqmm.md.
VERIFY_MRQMM = os.environ.get("QWEN_VERIFY_MRQMM", "1") or "1"
VERIFY_MR = VERIFY_MRQMM == "1"
VERIFY_MR_HEAD = VERIFY_MRQMM in ("1", "head")
# The DFlash block with "1" when QWEN_DRAFT_BLOCK is unset / 0 and mrqmm covers
# something: fixed 8. GPU 2026-10-24 (9B, paired vs today's adaptive DFlash):
# fixed 8 x1.46-1.48, adaptive x1.32, adaptive capped at 8 x1.40, 16 x1.18 —
# mrqmm runs 3..8 rows as one bucket (~33 ms of linears), 9..16 as the next
# (~46 ms), and acceptance falls with depth. An explicit QWEN_DRAFT_BLOCK wins.
VERIFY_MR_BLOCK = 8


# QWEN_GDN_FUSE (1): a decode step's four GDN in-projections (qkv, z, b, a)
#   as ONE quantized matmul (qmv for 1..2 rows, mrqmm for 3..16 rows on 8-bit
#   models and, with QWEN_MRQMM5=1 / QWEN_MRQMM4=1, 5- / 4-bit ones, otherwise quantized_matmul's
#   own qmv/qmm) over their weights
#   concatenated at load; the four modules keep row views of the concatenation
#   (no extra memory). Needs fp16 scales/biases: with QWEN_FP16=0 (bf16) it
#   fuses nothing (the load log says "0 layers"). mlx-vlm 0.6.2 did this (with a second copy of the
#   weights, +1.2 GB on the 9B); 0.7 runs four launches per GDN layer.
#   Per output row the same kernel and K order: bit-identical to the unfused
#   calls on qmv; with mrqmm the narrow b/a projections (N=32, else on qmv)
#   ride along in the multi-row kernel, as on the old stack. A mixed-width
#   layer (oQ4e) is fused per (bits, group) group of >= 2 projections, each
#   group its own concatenation and matmul (mrqmm only with N >= MIN_N); a
#   projection alone at its width keeps its own call (kernels._gdn_groups).
GDN_FUSE = os.environ.get("QWEN_GDN_FUSE", "1") != "0"

# Map of the project's knowledge

What holds now in each topic and how firm it is. Numbers live in the topic files and in
the reports under [perf/](perf/); the next steps in [BACKLOG.md](../BACKLOG.md); the rules a
session must know before touching anything in [CLAUDE.md](../CLAUDE.md).

Firmness: **measured** = GPU-gated, in-process A/B; **tested** = CPU suite only;
**open** = known gap, not measured yet.

## Code layout — [layout.md](layout.md)

`worker.py` is a facade over `qwen_worker/`; patches through it reach every module that
reads the name (tested by `tests/test_facade.py`). Firm.

## Models, KV format, memory — [models.md](models.md)

Six checkpoints: 35B MoE 3-bit, Ornith 9B 8-bit (1.5 preferred over 1.0, empirically), 27B
as 5-bit / 4-bit / oQ4e (oQ4e closest to the 5-bit). KV per model: 9B fp16, 27B q8, 35B q4.
Long context on 32 GB needs the wired-limit dance. Measured; the mlx 0.32.3 / mlx-vlm 0.7.6
stack is checked for exactness against the old one.

## Prefill — [prefill.md](prefill.md)

Compute-bound: 2048-token slabs, pfqmm GEMMs (bit-identical), fused SDPA from 32K keys,
fp16 cast (the M1 has no fast bf16). Headline: 120K prefill ×1.27 with the fixed slab +
pfqmm. Measured. Open: long-context attention is the next lever.

## Decode and batching performance — [decode.md](decode.md)

Bandwidth-bound step; mrqmm makes 3..16 rows cheap, so the 9B batches well to 16 rows
(aggregate ~3.5–5× a single stream through the gateway). The step-ahead must stay intact.
Under agentic load prefill, not the decode step, sets throughput. Measured on the 9B; the
27B's batched decode (B ≥ 3) is not gated.

## Metal kernels — [kernels.md](kernels.md)

mrqmm (4 / 5 / 6 / 8 bits), mrqmv, pfqmm, fused GDN in-projection, vatt (one-pass attention,
fp16 and q8 KV). 8-bit paths, pfqmm, GDN fuse and the vatt verify are default on; 4 / 5 / 6
bits are opt-in recipe knobs for the 27B. The newer kernels self-test at load. Measured per
kernel and end to end; numerics within the batched-decode class.

## Speculative decoding — [drafters.md](drafters.md)

DFlash (9B) and DFlash2 (27B) for a lone greedy text row: ×1.4–2 on code / English, a loss
on short Russian prose, where the target's MTP head wins. Load one drafter (the hybrid
loses). Prompt lookup pays on copy-heavy turns, off by default. DFlash ≠ plain greedy
bitwise, deterministic run to run. Measured; open: an acceptance-based fallback to plain.

## Prompt cache — [prompt-cache.md](prompt-cache.md)

Segmented multi-entry prefix cache with GDN checkpoints, chain policy, wanted-last /
stale-LRU / MRU eviction, partial stores, reply KV (default on, −21 % prefill tokens on
AppWorld, not bit-exact), disk tier (×2.3 req/min on AppWorld, off by default), one-copy
MOVE for a single stream. Measured on AppWorld replays and the 27B at long context.

## Scheduler — [scheduler.md](scheduler.md)

One MLX thread, continuous batching to 16 rows, FIFO admission with reservation-based
memory, preemption instead of a `length` cut, short-request prefill lane, cascade groups
for shared prefixes (default on with fp16 KV). B ≤ 2 bit-exact; batch-invariant mode for
determinism. Measured; cascade's later floor / staged releases are tested only.

## API and gateway — [api.md](api.md)

vLLM v0.30 behaviour: tool calls (`qwen3_xml` port), reasoning parser, sampler order,
logprobs and prompt logprobs, llguidance grammars, request shaping; anything unsupported is a
400. Gateway: queue, Check, in-flight slots from the worker's `max_batch`, cancel on
disconnect. Firm. Open: multi-model routing (planned, not built).

## Env knobs — [knobs.md](knobs.md)

The full table of worker and gateway knobs with defaults. Defaults are the measured best for
the 9B; the 27B and 35B need recipe values listed there.

## Measuring — [measuring.md](measuring.md)

GPU drift ±20–30 % over minutes: only in-process interleaved A/B counts; no other GPU work,
no `BG_NICE`. Firm.

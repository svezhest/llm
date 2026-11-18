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
stack is checked for exactness against the old one. No KV format below 8 bits passes the KL gate on
the 9B at 32K (emulated, 2026-11-18); q8 + an fp16 tail is better than q8 but not built.

## Prefill — [prefill.md](prefill.md)

Compute-bound: 2048-token slabs, pfqmm GEMMs (bit-identical), fused SDPA from 32K keys,
fp16 cast (the M1 has no fast bf16). Headline: 120K prefill ×1.27 with the fixed slab +
pfqmm; 9B cold 2K ×1.09–1.11 of llama.cpp. Measured. Open: long-context attention is the next
lever; `auto`'s 6–10 % on the first token (the drafters' prefill) is mostly removed on a branch, not merged.

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
on Russian prose, where the target's MTP head wins (×1.2–1.3). `QWEN_DRAFT=auto` loads both
and picks per round: 27B greedy code ×1.94, English ×1.33, Russian ×1.20, short prose ×1.09–1.13
of plain (T 0.6: ×1.78 / ×1.23 / ×1.20 / ×1.07–1.09), above plain in every cell and
0.91–1.01 of the best single setting; the 0.97 gate holds in 1 of 5 greedy and 2 of 5
sampled cells at fp16 KV, 2 of 2 on q8, the floor 4 of 4. Prompt lookup pays on copy-heavy
turns, off by default. DFlash ≠ plain greedy bitwise, deterministic run to run. On the 9B
(2026-11-18): main's rule 0.755–1.033 of the best setting (1 of 6 cells), ×1.13 of llama.cpp +
MTP at the API, but behind the head alone on prose (a fixed DFlash start and shadow drafts per
request, slower head rounds); at 1024 tokens on a branch code 0.99–1.00, prose 0.935–0.965.
Open: the first request of a class on code (DFlash's fixed prior), tied arms under sampling; the
head is not sampled while DFlash runs; `auto`'s per-request cost on prose.

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
no `BG_NICE`. Firm. Across systems (`scripts/perf/bench27/`): cell by cell at the same
position of a session; 27B at 2K on 2026-11-17 — plain ×1.15 of llama.cpp and ×1.08 of
mlx-vlm, `auto` ×1.10 of llama.cpp + MTP ([perf/2026-11-17/decode_27b_bench.md](perf/2026-11-17/decode_27b_bench.md)).
9B on 2026-11-18: plain ties within 4 % (llama.cpp, oMLX, mlx-vlm), `auto` ×1.13 of llama.cpp +
MTP and ×1.19 of oMLX + MTP; no in-session warm-up on the 9B; compare passes before pooling them
([perf/2026-11-18/decode_9b_bench.md](perf/2026-11-18/decode_9b_bench.md)).

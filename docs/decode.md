# Decode and batching performance

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- A decode step is bandwidth-bound; batching pays because mrqmm streams the weights for up
  to 8 rows at one row's speed (16 rows are MMA-bound).
- The step-ahead must stay intact: never queue a new op on a step's outputs before the next
  step is submitted.
- Under agentic load prefill dominates the scheduler's busy time, so throughput moves with
  prompt-cache hits, not with the decode step.
- The 35B MoE batches poorly with distinct prompts (each row reads its own experts).

## Continuous batching

- **Continuous batching (measured 2026-10-11/12)**: 9B decode step 41 / 51 / 90 /
  127 / 133 ms at B = 1 / 2 / 4 / 8 / 16 (the design microbench and the worker agree
  — no scheduler overhead). The ceiling is the M1 qmm row cost: MLX runs a per-row
  qmv below 10 rows for 4096-wide projections (below 6 for wider), then a flat
  qmm_splitk, so 16 rows cost about 8 (hence `QWEN_MAX_BATCH=16`). 27B (5-bit)
  pays poorly: ×1.4 at B=2 … ×1.9 at B=8 (×3.7 at 16). 35B MoE: rows read ~B×8
  distinct routed experts, so with distinct prompts batching gains only at large B
  (aggregate 38 / 69 tok/s at 8 / 16 concurrent vs ~40 solo decode; a microbench
  that clones one prompt into every row overstates it). Prefill-heavy load gains
  little (prefill is compute-bound and serialized; arrivals in waves): the share
  of decode time scales with occupancy so batches form at all with 8K prompts.
  A prefill slab runs uninterrupted in one tick: with the fixed 2048 fp16 slab a
  120K-offset slab stalls every decode row ~9 s (see Prefill slab).
  End-to-end (2026-10-12, through the gateway, ~610-token prompts, `ignore_eos`, closed
  loop; agg tok/s / TTFT p50; serialized = `QWEN_MAX_BATCH=1` on the same code):

  | c | 256 serialized | 256 batched | 1024 batched |
  |---|---|---|---|
  | 1 | 31.9 / 1.2 s | 31.2 / 1.3 s | 34.2 / 1.7 s |
  | 2 | 31.6 / 9.3 s | 48.7 / 2.4 s | |
  | 4 | 31.4 / 25 s | 82.8 / 3.0 s | |
  | 8 | 31.8 / 58 s | 110.4 / 5.3 s | 142 / 7.0 s |
  | 16 | | 115.9 / 10.4 s | 170 / 12.9 s |

  Pre-change serialized (f2bcbdf, drifting machine): 22.5 tok/s at every c, TTFT p50 80 s
  at c=8 (256) / 315 s (1024). Per-request decode 37 / 30 / 29 / 21 / 12 tok/s at c = 1 /
  2 / 4 / 8 / 16. `QWEN_MRQMM=0`: c=8 64.6, c=16 87.5 tok/s. Prefill TTFT single stream:
  8K 22.1 → 17.2 s, 32K 72.5 s (old: ~82–89 s, in-process). Peak 10.4 → 11.7 GB (c=16,
  256) / 12.05 GB (1024), old 13.8 GB. TTFT p95 at c=16 is 24 s: arrivals in waves,
  prefill one prompt at a time. Raw: `docs/perf/2026-10-12/final_bench.md` (+ `raw/`).
  Shared-prefix waves with cascade groups (`QWEN_CASCADE=1`, `docs/perf/2026-10-12/
  cascade_gpu.md`): step 8K × 16 rows 66.4 → 53.7 ms (×1.24), 32K × 8 83.6 → 68.2 ms;
  16 × 32K all admitted (peak 25.3 → 23.2 GB); 16 × 8K active −3 GiB; formation
  transient +2.2 GB at 8 × 32K; 32K × 16 at 92 ms vs ~62 projected (open, profile it).

## Decode step profile

- **Decode step profile (2026-10-12, B ≥ 2)**: the worker's step was GPU-bound but
  the step-ahead was broken. The token read `mx.concatenate([r.y…]).tolist()` queued
  a new kernel *behind* the step just submitted, so the read waited for step k+1 too,
  and the GPU idled while the host built step k+2 (3–6 ms of Python under gRPC load).
  The fix is `r.y.item()` per row, which waits on the step's own event: in-run A/B −7…12% at
  B=8, −5…7% at B=16. Never put a new op on a step's outputs before the next step is
  queued. Per-row draws in a batch are grouped (`sample_rows`): rows with equal sampler
  settings are filtered as one (G,V) stack and drawn with their own keys; an error
  building a group's draw fails that group's rows only. MLX row ops are bit-identical
  per row in a stack (Metal, 248K vocab, ties included: `tests/test_mrqmm.py::
  test_sample_rows_full_vocab_stack_equals_solo`). The card
  sampler at B=16 goes from 14.8 to 7.8 ms of GPU, a −18% step. Host costs are small:
  graph build 2–4 ms CPU, encode 4–9 ms, greedy sampler 0.5 ms GPU, quantize /
  clear_cache / GDN gathers negligible. MLX encodes at most 10 command buffers ahead
  (50 ops / 50M elements each on M1 Max), so at B=16 host time still leaks into the
  step. `MLX_MAX_OPS/MB_PER_BUFFER=200` hides it but adds +2–3 GB of prefill peak
  (buffers live until their command buffer completes), so it is rejected. Per-row
  attention is 4–8% of the step at 1K and 16–27% at 8K (~2× its bandwidth floor),
  which makes a multi-row decode-attention kernel the next lever. This machine's GPU
  speed drifts ±30% over minutes (probably thermal), so A/B *within* a run
  (`scripts/perf/prof/step_prof.py --ab … --ab-steps N`). Details:
  `docs/perf/2026-10-12/step_profile.md`.

## DeepInfer port

- **DeepInfer port (2026-10-24, `docs/perf/2026-10-24/`, ideas: `deepinfer_ideas.md`)**: ideas
  from deepseek-ai/DeepInfer (clone at `~/Git/DeepInfer`), each behind a knob (Env knobs).
  (0) The `batch:` log line has `step_ms` (decode ticks' wall per step), `gbps` (bytes a
  step must read — weights + KV + 2× GDN, `_step_bytes` — over it), `decode_s` /
  `prefill_s` per window (`analyze_worker_log.py` prints them). The old "~200 ms/step at
  rows=6" was the log (rows/prefilling are a snapshot, steps/s a 10 s window with 4–9 s
  slabs in it): AppWorld replay decode steps are 36–60 ms at 2–7 rows (200–300 GB/s),
  and **prefill is 81 % of the scheduler's busy time** (`step_metric.md`) — req/min moves
  with prefill / cache misses, not the decode step. Cascade's "32K × 16 at 92 ms vs ~62
  projected" does not reproduce on today's code (62.5 ms; the 2026-10-12 run was ~18 %
  slow). `QWEN_PROMPT_CACHE_CHAIN_TAIL=0` saves 1.2–1.5 states per entry (−4…8 %) and
  costs +3–10K replayed tokens only when an older message is edited: default stays
  4096; for append-only agentic runs under cache pressure 0 is free (`design/chain_tail.md`).
  Special-token strings in message text become control tokens (as in vLLM: a tool
  output with `<|im_end|>\n<|im_start|>system` forges a turn; `split_special_tokens`
  does not help — `<tool_call>`, `<think>` are special=False): `QWEN_SPLIT_SPECIAL_TOKENS`.
  Disk tier of the prompt cache (`QWEN_PROMPT_CACHE_DISK`, `design/pcdisk.md`): the biggest
  AppWorld lever of the day (×2.3 req/min at a 3 GB RAM cache: evictions become 0.3 s loads
  instead of 50–200 s re-prefills); off by default only because it needs a directory.

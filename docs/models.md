# Models, KV format and memory

Current conclusions (the state of all topics: [MAP.md](MAP.md)):

- Six checkpoints in `weights/`: the 35B MoE (3-bit Unsloth Dynamic), Ornith 1.0 / 1.5 9B
  (8-bit; 1.5 is text-only and preferred), the 27B as 5-bit, uniform 4-bit and oQ4e.
- oQ4e is closer to the 5-bit than the uniform 4-bit is, for ~5 % more bytes; it needs the
  4 / 5 / 6-bit kernels to be fast ([kernels.md](kernels.md)).
- KV format per model: 9B fp16, 27B q8, 35B q4. The 35B and long-context 27B need the
  wired-limit dance on 32 GB.
- Below 8 bits nothing passes on the 9B (KL harness, emulated): 4 / 5-bit fail at 8K, 6-bit +
  rotation + fp16 tail passes at 8K and fails at 32K on code (×2.0 / ×2.4 of q8's KL); q8 with the
  newest 1024 tokens fp16 has 3–4× lower KL than q8, not built
  ([perf/2026-11-18/auto_gate_9b.md](perf/2026-11-18/auto_gate_9b.md#experiments-that-did-not-pay), `raw/kvq4/`).
- Model cards' benchmarks are not trusted: evaluate empirically.
- Downloads: aria2c with per-file sha256, Xet off, never hard-killed (recipe below).

## Models

- **`qwen36-35b-a3b-mlx-ud3`** — Qwen3.6-35B-A3B, VL MoE (256 experts, ~3B active).
  MLX affine g64, 3-bit **Unsloth Dynamic** (mixed per-layer: embed 6b, attn/gates
  8b, shared-expert 4b, MoE experts 3b). ~16 GB. Needs the wired-limit dance on 32 GB.
- **`ornith10-9b-mlx-8bit`** — Ornith-1.0-9B (deepreinforce-ai, agentic-coding RL
  finetune), dense VL, MLX 8-bit. ~10.4 GB. No wired-limit dance; 8-bit ≈ fp16 quality.
  Card benchmarks (SWE-bench 69%) are self-reported and implausibly high for the
  size — **evaluate empirically, don't trust the card**.
- **`ornith15-9b-mlx-8bit`** — Ornith-1.5-9B (`ornith-ai/Ornith-1.5-9B-MLX-8bit`, the
  official MLX quant). Same arch/quant as the 1.0 but the checkpoint is **text-only**:
  no `vision_config`, no `visual.*` weights, no `preprocessor_config.json`. mlx-vlm
  0.7.6 drops the weightless vision tower (`vision_tower = None`) and loads strictly
  (0.6.2 needed a `strict=False` monkeypatch and kept a random tower); image requests
  → 400. ~9.5 GB on disk = 9.5 GB loaded (was 13.0 with the random tower and 0.6.2's
  fused-GDN weight copy), ~37 tok/s.
  Empirically (5 coding tasks, temp 0) 1.5 ≥ 1.0: same correctness on 4, and 1.0
  leaked its thinking into the answer and hit the length cap on the SQL task while
  1.5 answered cleanly. 1.5 thinks longer on trivial prompts (2–5× more tokens).
- **`qwen38-27b-mlx-5bit`** — Qwen3.8-27B (lmstudio-community), dense hybrid VL,
  `model_type qwen3_5` like the 9B (64 layers, full attn every 4th, 4 KV heads,
  head_dim 256, native 256K). MLX 5-bit g64, ~18.1 GB. Run: `QWEN_KV_BITS=8` +
  wired-limit dance (~24 GB @128K; 200K via `QWEN_MAX_CONTEXT=204800`, ~26 GB).
- **`qwen38-27b-mlx-oq4e`** — the same 27B, Jundot's oQ4e (oQ/oMLX mixed precision, MTP weights
  dropped at load): affine g64, base 4 bits with per-module overrides in `config.json` (lm_head 6,
  embed 5, attention 5 (last layer's k/v 6), GDN out 5, z / b / a 5 in layers 0–18, b / a 8 or 6 in
  three late layers, MLP down 5 in layers 0–17, the rest 4). LM ~14.85 GiB vs ~14.09 for the uniform
  4-bit (from the headers, MTP excluded); teacher-forced KL vs the 5-bit (`kl_models.py`, 3 × 3072
  tokens) 0.0236 / top-1 93.6 % vs 0.0534 / 90.1 % for the uniform 4-bit. Fast paths for it:
  `QWEN_MRQMM4/5/6=1` + the width-grouped GDN fusion (see "oQ4e" below).

Download over a slow, throttled link — hard-won recipe:
- `HF_HUB_DISABLE_XET=1` always (Xet backend hangs silently at 0 MB/s here).
- Get each LFS file's sha256 from `…/api/models/<repo>/tree/main` (`lfs.oid`).
- `aria2c -c -x16 -s16 -k1M -j1 --max-tries=0 --check-integrity=true
  --checksum=sha-256=<oid> --header="Authorization: Bearer <token>" --dir=<weights/x> -o <file> <resolve-url>`.
  Multi-connection beats single curl ~12× (CDN throttles per-connection); the
  checksum defeats proxy-corrupted-but-full-size shards.
- **Never hard-kill** `hf download`/aria2 — orphans/duplicates partials. Token at
  `~/.cache/huggingface/token`. Runtime is **mlx-vlm** (VL models), not mlx-lm.

## Context and KV cache

- **Context = 128K** (`MAX_CONTEXT=131072`, prompt + generation). **KV cache is
  per-model** via `QWEN_KV_BITS` (default 4) — `stream_generate(..., kv_bits=N,
  kv_group_size=64, quantized_kv_start=5000)` (MLX analog of llama.cpp
  `--cache-type-k`; K and V uniform, fp16 warmup for the first 5000 tokens). Pick N
  by weight precision + budget: **35B (3-bit, ~18 GB) → q4** (5.6 KiB/token, ~0.7 GiB
  @ 128K; the balanced match for 3-bit weights, and the 35B is the tightest fit);
  **9B (8-bit, ~10 GB) → fp16** (`QWEN_KV_BITS=0`, 32 KiB/token, ~4 GiB @ 128K:
  faster than q8 here, see README; q8 would be 17 KiB/token, ~2.1 GiB); **27B
  (5-bit, ~18 GB) → q8** (34 KiB/token, ~4.25 GiB @128K, ~24 GB total). Only
  full-attention layers (every 4th) hold KV (`_kv_token_bytes`: bits/8 + an fp16
  scale and bias per 64 elements). Env sets sampler defaults;
  requests may override temperature/top_p/top_k/min_p/presence_penalty (proto
  `optional` fields; absent = env default, temperature -1 / top_p 0 likewise).

- **KV quantization costs speed** on this model: q8 vs fp16 cache at 32K is −12%
  prefill and −15% decode. The 9B runs fp16 KV (`QWEN_KV_BITS=0`, 32 KB/token,
  ~4 GB @128K). The 27B/35B keep q8/q4 for memory.

## Memory on 32 GB

- **OOM avoidance on 32 GB** (all three needed for the 35B at long context):
  `sudo sysctl iogpu.wired_limit_mb=28672` (kernel cap) + `QWEN_WIRED_LIMIT_GB=27`
  (MLX wired limit, clamped to cap) + `QWEN_PREFILL_STEP=512` (chunked prefill
  bounds the activation peak) + `mx.clear_cache()` after every generation (else
  request N's buffer pool carries into request N+1's prefill → the second-request
  OOM). Verified: 56K-token prompt × 2 back-to-back, no OOM.

- **Buffer pool**: `QWEN_CLEAR_CACHE` (1) / `QWEN_CACHE_LIMIT_GB` (3, via
  `mx.set_cache_limit` at load). See README "Context & memory" for the numbers.

## Library stack

- **Library bump (2026-10-12): mlx 0.32.3, mlx-vlm 0.7.6, mlx-lm dropped** (src/worker
  only; root tooling env untouched). Exactness, teacher-forced 3.4K tokens, old → new
  stack: 9B KL 1.5e-5 / top-1 99.76%; 27B 9.2e-5 / 99.97%; 35B 6.3e-3 / 97.7% — the
  3-bit MoE's rounding floor, not a regression: vs an fp32 reference fp16 is equally
  far on both stacks (new 8.3e-3 / 96.97%, old 7.5e-3 / 97.32%; bf16 is worse on
  both, 1.1–1.4e-2), old-fp32 vs new-fp32 1.6e-4. Greedy identity corpus vs old: 9B
  12/15 (near-ties at chars 221–518 + a seeded draw), 27B 14/15, 35B 6/15; within
  the new stack byte-identical. Speed, interleaved A/B: 9B decode step 26.8 / 42.0 /
  61.2 → 26.6 / 41.7 / 61.4 ms at B = 1 / 8 / 16 (equal); 9B prefill 8K/32K/64K
  equal (±2%, drift); 35B prefill 8K 15.0/13.8 → 13.1/12.8 s (−9%), decode 65 → 68
  tok/s; 27B decode 14.8 → 15.5 tok/s. Memory: 9B loaded 13.0 → 9.5 GB, 8 × 8K
  batch peak 16.8 → 13.3 GB. DFlash ×1.5 (see below). Raw: `docs/perf/2026-10-12/` (`bump_*`, `design/bump_apply.md`).

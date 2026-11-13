# Third-party code and credits

This project is MIT-licensed (`LICENSE`). Parts of it are derived from the
projects below; their notices are kept here as their licenses require.

## Code derived from other projects

### MLX — MIT License, Copyright © 2023 Apple Inc. (https://github.com/ml-explore/mlx)

- `src/worker/cascade_mma.py`: the Metal attention kernel is adapted from mlx 0.32.3
  `kernels/steel/attn/kernels/steel_attention.h`.
- `src/worker/pfqmm.py`: instantiates MLX's own quantized GEMM (`QuantizedBlockLoader`,
  steel `BlockMMA`) with other tile sizes. The headers are read at run time from the
  installed mlx wheel and are not copied into this repository.

### mlx-vlm — MIT License, Copyright © 2025 Prince Canuma (https://github.com/Blaizzy/mlx-vlm)

- `src/worker/worker.py`: the prefill / decode loop (`_Scheduler._prefill`, `_iterate`,
  `_sample`) is a trimmed copy of mlx-vlm 0.6.2 `generate_step` / `stream_generate`,
  re-checked against 0.7.6; the batched attention and the fused Gated DeltaNet decode path
  follow mlx-vlm's `Qwen3_5Attention` / `Qwen3_5GatedDeltaNet`. The worker patches mlx-vlm
  0.7.6 classes at run time.

Text of the MIT License (the same for both projects):

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### vLLM — Apache License 2.0, Copyright contributors to the vLLM project (https://github.com/vllm-project/vllm)

- `src/worker/worker.py`: `parse_tool_calls` is a port of vLLM v0.30's `qwen3_xml` tool
  parser (`vllm/parser/qwen3.py` and the parser engine's schema coercion); `_GenState`
  ports its `qwen3` reasoning parser; the request shaping (`developer` → `system`, merged
  system messages) follows vLLM's renderer. These portions remain under the Apache License
  2.0: https://www.apache.org/licenses/LICENSE-2.0. Modifications: rewritten for this
  worker's streaming loop and token-id handling.

## Ideas, no code copied

- **DeepInfer** (deepseek-ai, Apache License 2.0): the 2026-10-24 work (batch-invariant
  decode, the prompt cache's disk tier, the short-prefill lane, preemption into the prompt
  cache, the forbid-sync debug guard, the top-k sampler shortcut, special-token handling)
  started from reading its design; see `docs/perf/2026-10-24/deepinfer_ideas.md`.
- **DFlash** drafters (z-lab) and the speculative-decoding loop of mlx-vlm.
- **llguidance** and **xgrammar**: constrained decoding runs on llguidance; tool grammars
  were checked against xgrammar's `qwen_3_coder` structural tag.

## Test image

`test_free_description_and_high_res.jpg` is Vasily Polenov, *Overgrown Pond* (1879),
a public-domain painting.

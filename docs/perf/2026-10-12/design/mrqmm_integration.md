# mrqmm: integration into worker.py

Module: `src/worker/mrqmm.py` (new, self-contained). Tests: `src/worker/tests/test_mrqmm.py`.
Results: `scratchpad/results/mrqmm.md`.

## The 3 lines (end of `Engine.load`)

Put them right after the `if FP16:` cast block, or anywhere later in `Engine.load` before
`self.sched.start()`. The cast has to come first: the kernel only takes fp16 scales and biases.
Suggested spot: just before `kernel = int(mx.device_info()["max_recommended_working_set_size"])`,
so the ~1 s warm-up compile happens before the memory budget is read. The warm-up allocates only
transient zeros.

```python
        if os.environ.get("QWEN_MRQMM", "1") != "0":   # multi-row 8-bit matmul for batched decode
            import mrqmm
            mrqmm.install(self.model.language_model)
```

`install` checks `QWEN_MRQMM` and Metal itself, so the `if` only skips the import.
* Pass the **language model**, not the VLM, and never the DFlash drafter.
* It returns the number of modules covered; 0 means it was a no-op.
* It logs one INFO line per model, for example:
  * 9B: `mrqmm: 249 modules on the multi-row kernel for 3..16 decode rows (linear_attn.in_proj_a x24, …, lm_head x1, mlp.down_proj x32, …) + fused GDN in-projection; left on quantized_matmul: none`
  * 35B: `mrqmm: 230 modules … (linear_attn.* x30 each, mlp.gate x40, self_attn.* x10 each) … left on quantized_matmul: 6-bit g64 x1, 8-bit g64 N % 8 x40, 4-bit g64 x120`
  * 27B: `mrqmm: no supported module, all on quantized_matmul (5-bit g64 x497)`

Add the env row to CLAUDE.md / README:

| `QWEN_MRQMM` | worker | `1` (multi-row 8-bit g64 kernel for 3–16-row batched decode; `0` = always `quantized_matmul`) |

## What it changes (fallback rules)

`install` swaps the class of each supported `nn.QuantizedLinear` to `mrqmm.MRQuantizedLinear`. Parameters, names and state are untouched; it is a `__class__` swap. It also wraps mlx-vlm's module-level `qwen3_5.language._decode_quantized_linears_fused`, the GDN `in_proj_qkv/z/b/a` concatenated matmul used at decode. The wrapper reuses the library's own concatenated-weight cache.

A call takes the new kernel only if **all** of these hold. Otherwise the original `mx.quantized_matmul` runs unchanged:

| condition | otherwise |
|---|---|
| module is 8-bit, affine, group size 64, with biases | 3/4/5/6-bit (27B, 35B experts / shared expert / lm_head / embed) → not swapped at all |
| scales and biases are fp16 (`QWEN_FP16=1`) | bf16 model (`QWEN_FP16=0`) → not swapped |
| N % 8 == 0, K % 64 == 0 | e.g. 35B `shared_expert_gate` (N = 1) → not swapped |
| x is fp16 | bf16 x → original |
| x is **decode-shaped**: `x.shape[-2] == 1` | prefill slabs `(1, L, K)` and the DFlash verify `(1, T, K)` → original (prefill / prompt-cache / spec-decode numerics unchanged) |
| 3 ≤ rows (= x.size / K) ≤ 16 | B = 1, 2 → original (**B ≤ 2 stays bit-identical to solo**); B > 16 → original |
| the fused GDN path: all 4 in-projections swapped, no `bias` | → library function as before |

`QWEN_MAX_BATCH` > 16 still works: above 16 rows the kernel is not used.

## Numerics

* fp32 accumulation over fp16 operands; the dequantized weight is rounded to fp16, the same as MLX's qmm loader.
* Error against an fp32 dequant→matmul reference on real 9B weights:
  * Every shape, M = 3..16, with and without outlier activations: **rms rel 2.9e-4**, which is the fp16 output rounding itself.
  * Max |Δ| / max |ref| ≤ 6.4e-4.
  * The error is ≤ `quantized_matmul`'s own error everywhere.
* Versus `quantized_matmul`: max |Δ| ≤ 1 fp16 ulp of the output scale (e.g. 0.002–0.06 on outputs of magnitude 4–100).
* B = 3..5: today each row runs its own qmv, so rows are bit-identical to solo. With mrqmm they differ from solo by fp16 rounding.
  * This is the same class of noise that B ≥ 6 already has today (qmm), and the same as the DFlash verify. The batching comment block at worker.py ~1065 should say "kernels' row-count-dependent rounding (mrqmm at 3–16 rows, qmm above)".
  * Greedy runs can flip on near-ties.
* No NaN/Inf in any test, including ×30–×200 outlier activations and ×50 outlier weight rows.

## Measured (M1 Max)

See `results/mrqmm.md` for the tables. Short version: the 9B decode-step linears (8 rows) go 101.7 ms → 38.5 ms (×0.38), and lm_head benefits (×0.33 at 8 rows).

Per-call host overhead is ~4 µs: a dict lookup plus one `metal_kernel` call with the constants baked into the source (template arguments would cost ~6 µs more). About 200 calls per step means ~0.8 ms of graph building, which overlaps the GPU in the step-ahead loop.

## Other notes

* Warm-up at install: one Metal compile per (K, N, row bucket), ~60 ms each, ~16 kernels on the 9B, so ~1 s at load. `install(lm, warmup=False)` defers it to the first batch of each shape.
* Thread safety: the kernel and call caches are plain dicts and are only touched from the scheduler thread, which is the one that runs MLX.
* `mrqmm.uninstall(lm)` reverts the class swap. The fused-path wrapper stays and falls through.
* When bumping mlx-vlm, check that `qwen3_5/language.py::_decode_quantized_linears_fused` still exists with the same cache attribute (`_qwen3_5_fused_decode_linears`). If it doesn't, `install` still covers the per-module calls, and the GDN in-projection falls back to qmm.
* Pre-existing, independent of mrqmm: mlx-vlm's `_decode_quantized_linears_fused` keeps a **second copy** of the GDN in-projection weights (it concatenates them on the first decode and caches the result). On the 9B that is 24 × (12352 × 4096 B + scales) ≈ **1.3 GB** of extra active memory after the first decode step; on the 35B it is ≈ 0.8 GB. It already happens today with or without mrqmm, and it appears in active memory only after the first decode. The scheduler's KV budget is computed at load, before it exists.

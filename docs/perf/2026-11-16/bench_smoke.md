# 27B decode benchmark: harness smoke (Phase B)

15.11.2026, M1 Max 32 GB (32 GPU cores), AC power, `powermode 2`, `iogpu.wired_limit_mb=27648`.
One GPU lease, 08:46–09:00, nothing else on the GPU. Bandwidth probe before every session:
336.3–339.1 GB/s (1 GiB fp16 sum, median).

This is a **smoke**: each system ran once with the `smoke` plan (64 generated tokens per request,
one request per cell). The numbers below show that every system works and how the harness reads it.
They are **not** benchmark results: the final matrix runs with 256/1024 tokens and ≥ 3 ABAB sessions.

Harness: `scripts/perf/bench27/` (bench27 branch, commits 7ee30a5, f6788fb). Versions: mlx 0.32.3,
mlx-vlm 0.7.6, llama.cpp 1.0.0 build 12359 (commit b1b770d90, Homebrew), code under test = main 7437e62.

## Systems: all of them load and serve Qwen3.8-27B

| system | command (abridged) | result |
|---|---|---|
| ours-plain | worker (4-bit, `QWEN_TEXT_ONLY=1 QWEN_MRQMM4=1 QWEN_KV_BITS=0 QWEN_MAX_BATCH=1` + the 80K memory recipe) + gateway | ok, ready in 8 s |
| ours-dflash | + `QWEN_DRAFT=weights/qwen38-27b-dflash2 QWEN_DRAFT_SAMPLING=1` | ok; log: "DFlash through mrqmm … block 8" |
| llama-q4_0 | `llama-server -m Qwen3.8-27B-Q4_0.gguf -ngl 99 -fa on -c 8192 --jinja -np 1 --cache-ram 0` | ok: arch `qwen35` is supported |
| llama-q4_0-mtp | + `--spec-type draft-mtp -md mtp-Qwen3.8-27B-Q4_0.gguf` (n_max default 3) | ok |
| llama-q4_0-mtp, md=0 | + `--spec-type draft-mtp`, no draft file | ok: the unsloth GGUF has its own nextn layer (`qwen35.nextn_predict_layers = 1`, 65 blocks) |
| llama-q4_0-dflash | + `--spec-type draft-dflash -md dflash-Qwen3.8-27B-Q4_0.gguf` | ok: the ggml-org drafter (arch `dflash`, 5 blocks) pairs with the unsloth target |
| mlx-vlm | `python -m mlx_vlm.server --model weights/qwen38-27b-mlx-4bit` (worker venv) | ok (loads the vision tower too: stock has no text-only switch) |
| mlx-vlm-dflash | + `--draft-model weights/qwen38-27b-dflash2 --draft-kind dflash --draft-block-size 8` | runs, pathologically slow (below) |
| llama-bench-q4_0 | `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3 -d 0 -o json` | ok: pp512 149.9 ± 2.8, tg128 17.33 ± 0.18 tok/s |

## Prompts render to the same tokens everywhere

- `/apply-template` + `/tokenize` on llama-server against the manifest's count (HF chat template, the
  `tokenizer.json` our worker uses), thinking off: **5/5 equal** (code/en/ru at 2K, code/ru at 8K;
  2047 / 2048 / 2047 / 8192 / 8191 tokens). The rendered tails are identical (`…<|im_start|>assistant\n<think>\n\n</think>\n\n`).
- `usage.prompt_tokens` of every smoke request: ours, llama-server and mlx_vlm.server all report the
  manifest's count (2047 / 2047 / 2048; turn 2: 2141).

## Stream granularity: count tokens from usage, not chunks

| system | chunks for 64 tokens (code / ru / en) | first chunk |
|---|---|---|
| llama-server (plain, MTP, DFlash) | 64 / 64 / 64 | 1 token |
| mlx_vlm.server (plain, DFlash) | 64 / 64 / 64 | 1 token |
| ours (gateway) | 25 / 32 / 54 | 3–4 tokens (code, ru) |

Our gateway streams what the worker's streaming detokenizer releases, which groups tokens (code
indentation, Cyrillic byte pieces). Every count therefore comes from `usage.completion_tokens`. The
first chunk carrying 3–4 tokens made `(n − 1)/(t_last − t_first)` read 2–5 % high at 64 tokens
(≈ 1 % at 256). `decode_tps` now uses `(n − k)/span`, where k = the first chunk's tokens (HF
tokenizer, counted after the stream). `decode_tps_raw` keeps the old formula. A speculative server's
burst of accepted tokens arrives as separate chunks a few µs apart (gap p50 0.05 ms for llama MTP), so
`t_last`/`t_first` are unaffected.

## Smoke numbers (64 tokens, one request each, not a benchmark)

Decode = `(n − k)/(t_last − t_first)` measured by the client. TTFT / prefill are for the cold 2K code
prompt. Turn 2 = the follow-up after that answer (2141 tokens). Footprint = peak `footprint` of the
model process.

| system | code greedy tok/s | ru greedy tok/s | en sampling tok/s | TTFT s / prefill tok/s | turn-2 TTFT s | chunks code/ru/en | footprint GiB |
|---|---|---|---|---|---|---|---|
| ours-plain | 20.6 | 20.6 | 20.2 | 13.1 / 156 | 0.59 (2111 cached) | 25 / 32 / 54 | 18.3 |
| ours-dflash | 37.0 | **17.6** | 23.9 | 12.6 / 162 | 0.62 (2111 cached) | 25 / 32 / 54 | 19.7 |
| llama-q4_0 | 17.4 | 17.6 | 17.4 | 14.3 / 143 | 0.50 (2110 cached) | 64 / 64 / 64 | (1.7, see below) |
| llama-q4_0-mtp | 34.5 | 22.9 | 30.4 | 14.0 / 146 | 0.49 (2110 cached) | 64 / 64 / 64 | (2.2) |
| llama-q4_0-mtp, md=0 | 33.2 | 22.7 | 29.3 | 14.2 / 144 | 0.49 | 64 / 64 / 64 | n/a |
| llama-q4_0-dflash | 30.8 | 19.0 | 27.3 | 13.9 / 147 | 0.50 (2110 cached) | 64 / 64 / 64 | (2.5) |
| mlx-vlm | 18.9 | 18.9 | 18.7 | 18.4 / 111 | 19.35 (0 cached) | 64 / 64 / 64 | 18.9 |
| mlx-vlm-dflash | 4.9 | 5.0 | 9.6 | 18.6 / 110 | 19.48 (0 cached) | 64 / 64 / 64 | 25.9 |

The md=0 row is from the console log. Its session directory was deleted by mistake during the smoke,
so the row has no session.json.

Speculation stats from the server logs (now parsed into `session.json` → `server_stats`):

| system | code greedy | ru greedy | en sampling |
|---|---|---|---|
| ours-dflash (accepted / drafted) | 65 % (51/78) | **19 % (35/185)** | — (sampling rows log no dflash line) |
| llama MTP (acceptance, mean len) | 0.92, 3.65 | 0.46, 2.38 | 0.72, 3.15 |
| llama DFlash (acceptance, mean len) | 0.79, 3.32 | 0.34, 2.00 | 0.62, 2.86 |

Measured once at 64 tokens, these smoke runs already show three things for the final matrix (they
need the real runs to confirm):
- On Russian prose our DFlash accepts 19 % of its drafts, which makes it **slower than our own plain
  decode** (17.6 vs 20.6). llama.cpp's MTP still gains there (22.9 vs 17.6).
- With sampling, llama MTP (30.4) and llama DFlash (27.3) are ahead of our DFlash (23.9).
- Plain decode: ours 20.6, stock mlx-vlm 18.9, llama.cpp Q4_0 17.4 (llama-bench tg128 17.33 agrees
  with the server). Per-token gaps p50: 49.7 / 52.5 / 56.7 ms.

## Output identity

- Greedy, 64 tokens: our DFlash == our plain on both prompts (code, ru).
- Across frameworks the 64-token answers start identically and diverge at char 105 (llama.cpp, all
  three) / char 70 (stock mlx-vlm on ru). These are different quants and kernels, so only ours-vs-ours
  identity is a gate. `aggregate.py --identity ours-dflash,ours-plain` computes it on the final runs.

## Memory: footprint misses llama.cpp's weights

`footprint` sees MLX's Metal buffers (ours 18.3/19.7, mlx-vlm 18.9, mlx-vlm-dflash 25.9 GiB). It does
not see llama.cpp's weights, which are an mmap'd GGUF wrapped as Metal buffers (1.7–2.5 GiB reported).
Since the smoke, every session also samples the system's wired + active + compressed memory
(`vm_stat`) with a baseline before start. The llama.cpp memory column should use that delta. Its
default log level hides its buffer-size lines, so they could not be parsed.

## Stock mlx-vlm

- Its DFlash is pathological on this machine, as the plan predicted: 4.9 tok/s greedy (inter-token
  gaps up to 2.2 s), peak 25.9 GiB. The bf16 drafter is not cast and has no fp32 activations.
- Turn 2 is a cold prefill (0 cached tokens, 19.4 s). Its automatic prefix cache did not engage with
  the default flags.

## Perplexity tokens

The first `llama-tokenize` check of the perplexity corpora disagreed with the HF tokenizer: 55735 vs
55741 tokens (en), 47725 vs 47736 (ru). There were two causes:

1. llama.cpp processes `\n` escapes in prompt text: the docs' literal `"\n\n"` became real newlines.
2. The docs quote control tokens (`<|im_start|>`, `<think>`, `<tool_call>`, …, 13× in the ru corpus),
   which tokenizers turn into special ids depending on their parse-special setting.

Fixed in f6788fb: the corpus puts a space after the `<` of every added-token string, and
llama-perplexity / llama-tokenize run with `--no-escape` (tokenize also with `--no-parse-special`).
Neither the re-check nor a 2-chunk perplexity run of either scorer has been done yet (see below).

## Perplexity smoke

Not run yet: the queue for a second short lease was 6 deep, so I released my place to report the
system smoke first. The first perplexity sessions of the final matrix should run with `--set chunks=2`
first: `ppl-llama-q4_0` and `ppl-mlx-4bit`, about 2 min each. That also re-runs the id check:
`src/worker/.venv/bin/python scripts/perf/bench27/ppl.py tokens --gguf …Q4_0.gguf --corpus <ppl_en.txt>`.
The MLX scorer (`ppl.py mlx`) has not run on the GPU yet.

## Notes for the final matrix

- One command per session, with the worker venv's python so first chunks are token-counted:
  `src/worker/.venv/bin/python scripts/perf/bench27/session.py --system SYSTEM --plan PLAN --code <tree under test> --out $PERF_OUT/bench27 --lease NAME`
  (`--set n_max=K` for llama drafters, `--set md=0` = MTP from the target's own nextn layer,
  `--env K=V` for our worker). For ABAB:
  `… run_matrix.py --repeat 3 --lease NAME ours-dflash:headline llama-q4_0-mtp:headline mlx-vlm:headline ours-plain:headline llama-q4_0:headline …`.
  Then `aggregate.py --index <index json> --identity ours-dflash,ours-plain --json agg.json --md agg.md`
  and `uv run --offline --no-project --with matplotlib python scripts/perf/bench27/chart.py --agg agg.json --out docs/perf/2026-11-16/`.
- `--code` must point to a tree with `target/release/qwen-gateway` built (cargo build under the lease)
  and `src/worker/worker_pb2*.py`.
- `iogpu.wired_limit_mb` is 27648, not the plan's 28672. Our recipe uses `QWEN_WIRED_LIMIT_GB=27` =
  27648 MB, so it fits, but llama.cpp at 80K (`-c 90112`) has not been tried.
- llama-server's `-c` is sized per plan (smoke/headline 8192, ctx 36864, ctx80 86016). Our worker's
  `QWEN_MAX_CONTEXT` is 98304 in every plan.
- The llama.cpp MTP `--spec-draft-n-max` sweep (default 3): plan `sweep` with `--set n_max=1..4`.

Raw: `raw/smoke/*.json` (session.json per session, paths replaced by `$REPO` / `$WEIGHTS` / `$PERF_OUT`).

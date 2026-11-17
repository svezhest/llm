# Ornith-1.5-9B decode speed on an M1 Max: this server, llama.cpp, stock mlx-vlm

[[TODO: date]]. One machine: MacBook Pro M1 Max, 32 GB, 32 GPU cores, AC power, `powermode 2`,
`iogpu.wired_limit_mb=[[TODO]]`. mlx [[TODO]], mlx-vlm [[TODO]], llama.cpp build [[TODO]]
(Homebrew). Code under test: main `[[TODO]]` (every session; the table under "Sessions").
Harness: `scripts/perf/bench27/` (`run_matrix.py` → `session.py` → `bench_decode.py`), plan
`headline2`, the same prompts and cell order as the 27B run of
[../2026-11-17/decode_27b_bench.md](../2026-11-17/decode_27b_bench.md). Tables:
`scripts/perf/bench27/report.py` (commands at the end). Labels: [M] measured, [D] derived from
measured numbers.

## Systems

All five serve the same model through an OpenAI-compatible HTTP API with streaming, one
request at a time, thinking off.

| system | weights | how it runs |
|---|---|---|
| ours, plain | `ornith15-9b-mlx-8bit` (8 bit, group 64, 8.9 GB) | worker + gateway: `QWEN_KV_BITS=0` (fp16 KV) `QWEN_MAX_BATCH=1`, prompt cache 4 GB without reply KV (`session.py` `OURS9_ENV`) |
| ours, `auto` | + a DFlash drafter trained for Ornith 1.5 (2.4 GB) and the model's own MTP head (0.2 GB) | + `QWEN_DRAFT=auto` with `QWEN_DRAFT_DFLASH` / `QWEN_DRAFT_MTP` naming the two: picks DFlash / the head / neither per round, also under sampling |
| llama.cpp | `Ornith-1.5-9B-MTP-Q8_0.gguf` (9.1 GB, the MTP layers inside) | `llama-server -ngl 99 -fa on -c 8192 --jinja -np 1 --cache-ram 0` |
| llama.cpp + MTP | the same file | + `--spec-type draft-mtp` (the file's own MTP layers, no `-md`; its default draft length) |
| mlx-vlm | `ornith15-9b-mlx-8bit`, the same files as ours | `python -m mlx_vlm.server` (stock [[TODO: version]]) |

Ours and mlx-vlm load the same files; llama.cpp runs Q8_0, the GGUF of the same precision
(not the same bytes). Answer quality is not compared here. [[TODO: anything left out and why]]

## Method

- Plan `headline2`: 9 prompts of about 2K tokens (3 code, 3 English prose, 3 Russian prose),
  each greedy and sampled (T 0.6, top-p 0.95, top-k 20), 256 generated tokens; then one
  1024-token greedy code answer. 19 cells, the classes interleaved (code, English, Russian,
  code, …). The prompts were cut to 2047–2048 tokens of the 27B's tokenizer; the 9B's counts
  are in "Prompt tokens" below — every system must count the same.
- One session = one system in fresh processes: preflight (no other model process, nice 0,
  power mode), a 30 s memory-bandwidth probe ([[TODO: range]] GB/s in every session), start,
  one warm-up request, the 19 cells in the same order, stop. Sessions run back to back under
  one GPU lease, nothing else on the machine; two passes over the five systems (`a` / `b`).
- Decode tok/s = (completion tokens − tokens of the first chunk) / (last chunk − first
  chunk), timed by the client at the HTTP stream. Every greedy cell is a cold prefill
  (TTFT and prefill tok/s come from it).
- A session slows down as the machine warms up ([[TODO: % for this model]], below; the 27B
  lost 16–23 %), so systems are compared cell by cell at the same position of their
  sessions, and the per-class medians take one cool, one middle and one warm cell of every
  class.

## Results

Decode tok/s: each session's median over the class's three prompts, then the median of the two
sessions [M] (the chart shows the same numbers; whiskers = the two sessions):

![decode tok/s at 2K](decode_2k_light.png)

<!-- report.py medians -->
[[TODO: report.py fill]]
<!-- /report.py medians -->

Ratios cell by cell (the same prompt, mode and position in the session) [D]:

<!-- report.py ratios -->
[[TODO: report.py fill]]
<!-- /report.py ratios -->

- **Plain decode**: [[TODO: ours, plain / llama.cpp and / mlx-vlm: ×, cells ahead, cool vs warm]]
- **With speculation**: [[TODO: `auto` / llama.cpp + MTP over the 19 cells, by class, the cells behind]]
- [[TODO: `auto` / plain llama.cpp]]
- [[TODO: what speculation adds on top of our own plain decode (sessions apart, ±5 %); in-process
  A/B per drafter: [auto_gate_9b.md](auto_gate_9b.md)]]

Every cell, tok/s per session (`a / b` = first / second session of that system) [M]:

<!-- report.py cells -->
[[TODO: report.py fill]]
<!-- /report.py cells -->

### The warm-up inside a session

Greedy code cells by position, every session [M] (for `auto` the prompts differ in how well
they draft, so its column is not a warm-up measure):

<!-- report.py warmup -->
[[TODO: report.py fill]]
<!-- /report.py warmup -->

[[TODO: the drop per system, first → last; whether two sessions repeat each other]]

### Cold prefill at 2K

TTFT of the greedy cells = a cold 2K prefill [M]; the ratio = TTFT of the other system / ours
over the same cells at the same position, geometric mean [D]:

<!-- report.py prefill -->
[[TODO: report.py fill]]
<!-- /report.py prefill -->

[[TODO: one line: ours against llama.cpp and mlx-vlm]]

### Prompt tokens

`usage.prompt_tokens` of every cell as each server counted it, against the manifest's count
(the 27B tokenizer) [M]:

<!-- report.py tokens -->
[[TODO: report.py fill]]
<!-- /report.py tokens -->

## Long context (8K, 32K, 80K)

Plan `ctx`: code and Russian prompts of 8K and 32K tokens, greedy, 256 tokens, two rounds,
then a follow-up turn on the 32K conversation; plan `ctx80`: one 80K code prompt, two rounds
and a follow-up turn ([[TODO: which systems ran ctx80]]). One session per system and plan, the
same day. Only the first round is at a matching position (in ours the second round is a
prompt-cache hit, in the others' a second cold prefill, so the machine is warmer). [M]

<!-- report.py ctx-decode -->
[[TODO: report.py fill]]
<!-- /report.py ctx-decode -->

<!-- report.py ctx-ratios -->
[[TODO: report.py fill]]
<!-- /report.py ctx-ratios -->

<!-- report.py ctx-prefill -->
[[TODO: report.py fill]]
<!-- /report.py ctx-prefill -->

- [[TODO: `auto` / llama.cpp + MTP in the first round, by class and context; how decode falls from 8K to 32K / 80K]]
- [[TODO: cold prefill at 32K against llama.cpp and mlx-vlm; the follow-up turn]]

## How `auto` chooses here

[[TODO: summary, 2–3 lines.]] The in-process gate (plain / DFlash / the MTP head / `auto` in one
process, interleaved): [auto_gate_9b.md](auto_gate_9b.md).

## Limits of this comparison

- One machine, one day; two sessions per system.
- llama.cpp's `draft-mtp` ran with its default draft length; it was not tuned per class.
- llama.cpp runs Q8_0, ours and mlx-vlm the MLX 8-bit files: the same precision, not the same
  weights; speed per system as its ecosystem ships the model, not kernel against kernel.
- The prompts were cut for another tokenizer (the 27B's): about 2K, 8K, 32K, 80K tokens, not exact.
- Long context: one session per system.
- Peak memory is not in the table: the harness reads the process footprint, which does not
  count llama.cpp's Metal buffers.
- [[TODO: anything else]]

## Sessions

<!-- report.py setup -->
[[TODO: report.py fill]]
<!-- /report.py setup -->

Raw: `raw/bench9/<session>/session.json` (every request: tokens, TTFT, chunk arrival times, the
worker's per-request lines), `raw/bench9/index-*.json`; paths read `$CODE` (the tree under
test), `$WEIGHTS`, `$PERF_OUT`. Charts: `decode_2k_light.png` / `decode_2k_dark.png`.

## How to rebuild

The sessions (`$CODE` = the tree under test, `$WEIGHTS` = the weights dir, one GPU lease;
pass `b` = the same command with `--tag h2b`):

```sh
export PERF_OUT=... BENCH_WEIGHTS=$WEIGHTS
AUTO9=env.QWEN_DRAFT_DFLASH=$WEIGHTS/_staging/ornith15-9b-dflash,env.QWEN_DRAFT_MTP=$WEIGHTS/_staging/ornith15-9b-mtp
src/worker/.venv/bin/python scripts/perf/bench27/run_matrix.py --code $CODE --out $PERF_OUT/bench9 --repeat 1 \
    --tag h2a ours9-auto:headline2:$AUTO9 llama9-q8_0-mtp:headline2 mlx-vlm9:headline2 ours9-plain:headline2 \
    llama9-q8_0:headline2
src/worker/.venv/bin/python scripts/perf/bench27/run_matrix.py --code $CODE --out $PERF_OUT/bench9 --repeat 1 \
    --tag ctx ours9-auto:ctx:$AUTO9 llama9-q8_0-mtp:ctx mlx-vlm9:ctx ours9-plain:ctx llama9-q8_0:ctx \
    ours9-auto:ctx80:$AUTO9 ours9-plain:ctx80 llama9-q8_0-mtp:ctx80
```

This report from them (the tables between the `report.py` marks, the chart, the raw copy):

```sh
R=$PERF_OUT/bench9 D=docs/perf/2026-11-18
python3 scripts/perf/bench27/report.py fill $D/decode_9b_bench.md \
    --h2 $R/index-*-h2a.json $R/index-*-h2b.json --ctx $R/index-*-ctx.json
python3 scripts/perf/bench27/aggregate.py --index $R/index-*-h2a.json --index $R/index-*-h2b.json \
    --json $R/agg-h2.json --md $R/agg-h2.md
uv run --offline --no-project --with matplotlib python scripts/perf/bench27/chart.py --agg $R/agg-h2.json --out $D
python3 scripts/perf/bench27/report.py raw $D/raw/bench9 $R/index-*-h2a.json $R/index-*-h2b.json $R/index-*-ctx.json
```

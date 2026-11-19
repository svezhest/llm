# `QWEN_DRAFT=auto` on the 9B: the gate, why it trails the head on prose, what did not pay

Ornith-1.5-9B 8-bit (`weights/ornith15-9b-mlx-8bit`), drafters DFlash (`ornith15-9b-dflash`,
trained for Ornith 1.5, 4-bit at load) and the model's own MTP head (`ornith15-9b-mtp-q8`, split
from the original checkpoint, 8-bit), both named explicitly from `weights/_staging`. M1 Max
32 GB, mlx 0.32.3, mlx-vlm 0.7.6, fp16 KV unless stated. Labels: [M] measured, [D] derived.
Tool: `scripts/perf/mtp/ab_mtp.py` + `gate_table.py`: every variant in one process, interleaved
per prompt, 2 or 3 rounds, thinking off. The HTTP-level comparison with other systems is
[decode_9b_bench.md](decode_9b_bench.md); this report is the in-process gate behind it. Raw:
`raw/gate9b/`, `raw/dfl/`, `raw/dflash_causal9/`, `raw/round9b/`, `raw/mtp_head9*/`, `raw/kvq4/`,
`raw/spf/` (paths in them read `$PERF_OUT`, `$WEIGHTS`).

Findings first:

- With main's rule `auto` passes the gate in 1 of 6 cells at 256 tokens (0.755–1.033 of the
  best single setting) [M]. It is above plain decode in every cell (×1.14–1.54).
- At 1024 tokens (a branch, not merged) it holds on code (0.992, 0.999) and misses on prose by
  3–6 % (0.935–0.965) [M].
- On prose `auto` runs the head in 96–99 % of the rounds and still trails the head alone with
  one draft by 5.6–13.5 % of decode time (256 tokens, a branch run) [D]. The cost is a fixed part per request (DFlash's first
  rounds 2.3–3.5 %, shadow drafts 1.1–2.2 %) plus slower head rounds (up to 3.5 % each for 1-draft
  and 2–4-draft rounds).
- DFlash on this model gains ×1.69–1.89 over plain on code and loses on prose (×0.73–0.98 on
  main, ×0.62–0.92 on the branch run): it keeps 8–16 % of its drafts on prose against 39–46 %
  on code [M].
- Eight changes measured on branches did not pay (below); one did (spec-prefill: time to first
  token, not decode).

## The gate

`auto` ≥ 0.97 × the best of the other variants in every cell: plain decode, DFlash alone
(`dflash`), the head with one fixed draft (`mtp1+q`), with three (`mtp3+q`), and with its own
depth rule (`mtp0+q`, `QWEN_MTP_K=0`, what the benchmark calls "the MTP head"). Cells: code /
English / Russian at 2K, short English / Russian prose in some runs, greedy and sampling
(T 0.6, `@s`); 256 generated tokens unless stated. `auto / best` is the median over pairs of the
per-pair ratio (the tool's `gate_table.py`); "auto / head" below is a ratio of the cell medians
[D].

## As shipped on main

### The rule on main with the 8-bit head (`raw/gate9b/gate9b_q8head.*`)

Tok/s, medians of 2 requests [M]; `auto / best` from the tool [M]; `auto / head` [D]:

| cell | plain | DFlash | MTP, 1 draft | MTP, 3 drafts | MTP, own rule | `auto` | `auto` × plain | `auto` / best | `auto` / head |
|---|---|---|---|---|---|---|---|---|---|
| code, T 0.6 | 23.1 | 43.5 | 33.7 | 37.9 | 37.2 | 32.9 | 1.42 | 0.755 of DFlash | 0.88 |
| English, T 0.6 | 22.1 | 20.6 | 26.6 | 24.7 | 26.3 | 27.5 | 1.24 | 1.033 pass | 1.05 |
| Russian, T 0.6 | 22.0 | 16.0 | 29.6 | 23.9 | 27.4 | 26.5 | 1.20 | 0.896 | 0.97 |
| code | 23.9 | 40.2 | 34.4 | 38.4 | 36.1 | 37.0 | 1.54 | 0.914 of DFlash | 1.02 |
| English | 24.1 | 23.7 | 31.5 | 26.9 | 29.7 | 27.6 | 1.14 | 0.875 | 0.93 |
| Russian | 25.2 | 19.0 | 33.7 | 29.8 | 31.2 | 30.0 | 1.19 | 0.888 | 0.96 |

1 of 6 cells at 0.97. Plain ran at 22–25 tok/s in this run against 32–37 in the later gate runs
and in the benchmark: the machine was slow during it. The ratios are in-process and
interleaved, so they compare like with like, but the absolute numbers are low. Rounds of
`auto`: on prose 96–99 % on the head, 1–3 % on DFlash; on code 43 % (greedy) and 10 % (T 0.6)
on DFlash; 4–6 shadow drafts per request [M]. On sampled code `auto` stayed on the head (88 % of
rounds, 8 switches) where DFlash alone was ×1.89 of plain.

Other runs of main's rule [M]:

- The 4-bit head that ships inside a third-party 4-bit build of the model, 10 cells
  (`gate9b_auto.*`): 1 of 10 at 0.97 (0.775–0.982). That head drafts worse (below).
- q8 KV (`gate9b_q8kv.*`): 0 of 4 (0.855–0.924).
- 32K code at T 0.6 (`gate9b_32k.*`): 0.975 of DFlash, pass; `auto` ×1.50 of plain.
- The older DFlash drafter made for base Qwen3.5-9B, no head (`gate9b_old.*`): 3 of 10
  (0.905–0.978); that drafter is ×1.74–1.76 of plain on code and ×0.66–1.02 on prose, about the
  same as the Ornith one.

### DFlash on prose (`raw/gate9b/gate9b_q8head.txt`, `raw/dfl/93b_probe9.txt`, `raw/dflash_causal9/`)

- Share of DFlash's drafts kept, main, first round of the run [M]: code 39–46 %, English
  15–16 %, Russian 8–9 %; tokens per round 3.70–4.18 on code, 2.06–2.11 on English, 1.55–1.62
  on Russian. A DFlash round took 88–98 ms there, a plain step about 43 ms (23 tok/s) and a
  1-draft head round 53–60 ms for 1.54–1.85 tokens: DFlash needs about 2 tokens a round to
  match plain decode and about 3 to match the head [D], so it loses to the head on all prose
  and to plain decode on Russian.
- First-draft acceptance at T 0.6 (argmax rule, [M]): DFlash 0.822 code, 0.583 English,
  0.402 Russian, 0.488 / 0.303 short English / Russian; the head 0.868, 0.686, 0.576, 0.517,
  0.524.
- A non-causal mask inside the drafter's block was suspected; turning it causal changes
  nothing: greedy tokens per round code 3.64 against 3.70, English 2.06 = 2.06, Russian
  1.62 = 1.62 (`causal.txt` / `noncausal.txt`) [M].

### The head: which weights (`raw/mtp_head9/tables.md`, `raw/mtp_head9b/tables.md`)

Teacher-forced on the target's own trajectories, 1585 positions [M]:

| head | first draft kept, code greedy / sampled | drafts kept of 4, code greedy | paired against q8, mean kept of 4 (± s.e.) | draft k=1 cost |
|---|---|---|---|---|
| q8, split from the original (shipped) | 0.855 / 0.852 | 2.284 | – | 4.72 ms |
| fp16, the original's | 0.855 / 0.849 | 2.290 | +0.0069 ± 0.0055 (fp16 − q8) | 5.17 ms |
| q4, our own from the original (q4o) | 0.836 / – | 2.177 | −0.0662 ± 0.0154 | – |
| q4, from the third-party build (q4x) | 0.732 / 0.678 | 1.571 | −0.1987 ± 0.0236 | 4.64 ms |

Costs from `mtp_head9` (plain step 32.9 ms in that run). The third-party 4-bit head loses 0.20
drafts of 4 against q8: 0.07 of it is the bit width (q4o − q8) and 0.13 the build (q4x − q4o).
On prose the heads are close (English first draft 0.595–0.612, Russian 0.601–0.625). The q8
head is the shipped one: it costs 0.08 ms more per draft than q4 and drafts as well as fp16.

## Measured on a branch, not merged

### The 1024-token gate (`raw/gate9b/gate9b_int2_1k.*`)

Branch integ2 (bc553a7: the Chooser fix of branch auto-9b, the head's per-class depth of
branch mtp-depth pass 1, the DFlash changes of branch dflash-borrow, round overlap), 3
prompts × 3 rounds × 1024 tokens. Tok/s [M]:

| cell | plain | DFlash | MTP, 1 draft | MTP, 3 drafts | MTP, own rule | `auto` | `auto` / best | `auto` / head [D] |
|---|---|---|---|---|---|---|---|---|
| code, T 0.6 | 31.9 | 53.5 | 45.2 | 48.9 | 52.3 | 55.5 | 0.992 pass | 1.06 |
| English, T 0.6 | 32.5 | 27.7 | 40.1 | 36.2 | 40.2 | 38.9 | 0.965 | 0.97 |
| Russian, T 0.6 | 32.0 | 22.1 | 41.0 | 35.6 | 40.7 | 38.1 | 0.935 | 0.94 |
| code | 32.6 | 56.0 | 46.8 | 51.2 | 52.1 | 55.3 | 0.999 pass | 1.06 |
| English | 32.5 | 30.6 | 41.3 | 37.2 | 40.9 | 39.5 | 0.957 | 0.97 |
| Russian | 32.7 | 23.1 | 43.0 | 37.6 | 39.3 | 40.6 | 0.940 | 1.03 |

2 of 6 at 0.97. The head's own depth rule against the best fixed depth: 5 of 6 at 0.98
(0.993–1.008), Russian greedy 0.912 [M]. `auto` on prose: 96–99 % of the rounds on the head,
1–4 % on DFlash, 17–26 shadow drafts per request; on code 51–53 % of the rounds on DFlash [M].

### Why `auto` trails the head on prose (`raw/dfl/93c_ab9_trace.json`, `93c_auto_tax.txt`)

Branch dflash-borrow (5d6b23b), 3 rounds × 256 tokens, `auto` against the head with one fixed
draft (`mtp1+q`) in the same process; `scripts/perf/mtp/auto_tax.py` charges each of `auto`'s
rounds its time minus what the head alone would take for the same tokens (the head's ms per
token in that cell) [D]:

| share of the head's decode time | English | English, T 0.6 | Russian | Russian, T 0.6 | short Russian | short Russian, T 0.6 |
|---|---|---|---|---|---|---|
| `auto` against the head alone | +10.9 % | +9.4 % | +11.4 % | +13.5 % | +5.6 % | +8.1 % |
| DFlash rounds at the request's start | +3.5 | +2.9 | +2.7 | +3.1 | +2.4 | +2.3 |
| head rounds with a shadow DFlash draft | +2.1 | +2.1 | +2.2 | +1.4 | +1.1 | +1.5 |
| head rounds, 1 draft | +0.3 | +2.5 | +3.5 | +3.3 | +0.9 | +1.8 |
| head rounds, 2–4 drafts | +2.4 | −0.2 | +1.9 | +3.5 | +0.8 | +2.0 |
| plain steps (the head's own bursts) | +1.0 | +1.5 | – | +0.5 | – | +0.2 |
| DFlash later, prompt lookup | +0.8 | – | +0.6 | +1.1 | – | −0.2 |

The rows sum to within 0.4–0.8 % of the total (the remainder is time outside the rounds). On code
`auto` is faster than the head alone (−6.2 % greedy, −10.1 % at T 0.6), mostly through DFlash
rounds [D]. So on prose: about 2.3–3.5 % is the fixed start on DFlash (it shrinks with longer
answers, which is why 1024 tokens is closer), 1–2 % the shadow drafts, and the rest is head
rounds that keep fewer tokens or run longer under `auto` than under the fixed 1-draft head (`auto`
chooses 2–4 drafts in 4–15 % of its prose rounds, where the fixed head never does). In that run
`auto` / `mtp1+q` over all 24 pairs is 0.929 (0.860–1.186) [M].

### Experiments that did not pay

Each measured in one process against its own baseline [M]:

| change (branch) | what was measured | result | raw |
|---|---|---|---|
| DFlash depth from the drafter's confidence (dflash-borrow 5d6b23b) | `dflash` / `dflash+full` (always the full block), `auto` / `auto+full`, 24 pairs | 0.992 (0.905–1.003) and 0.999: no gain; the worker's request lines report 11.5–13.9 ms of GPU idle for the confidence read | `dfl/93c_ab9.txt` |
| The whole prompt for the drafter's full-attention layer (dflash-borrow 7ef7a22) | `dflash` / `dflash+win` (the old window) at 8K and 32K | 0.985 at 8K (tokens per round unchanged), `auto` 0.974; 32K 0.934 / 0.918 (one pair each): slower; prompt-cache hit and cold paths agree (4 / 4 PASS) | `dfl/93d_long9.txt` |
| Sampled (coupled) drafts and rejection sampling, probe only (dflash-borrow) | first-draft acceptance at T 0.6 against the argmax rule | DFlash −0.023…+0.055, head −0.006…+0.076 (short English 0.517 → 0.593 at most); the speed was not measured and rejection sampling was not built | `dfl/93b_probe9.txt` |
| Round overlap: the next round's host work overlaps the verify (round-host fc4304d) | on / sequential, 12 pairs each | `auto` ×0.995, `mtp0+q` ×1.023, `mtp1+q` ×1.046, `mtp3+q` ×1.025; per pair 0.84–1.16: within noise | `round9b/round9_pairs.txt` |
| The Chooser's burst rule `need()` (auto-9b 7d34b67) | the 10-cell gate | 0 of 10 at 0.97 (0.661–0.961); plain ran 27.4–32.2 tok/s across the cells of that run | `gate9b/gate9b_need.*` |
| The Chooser fix: priors, a margin of one standard error, DFlash's start arm (auto-9b c383c7b) | the 10-cell gate | 2 of 10 (0.810–1.018); on the six cells it shares with main's run above, 0.842–0.936 against 0.755–1.033 (separate runs): no clear gain | `gate9b/gate9b_choose.*` |
| The head's per-class depth, second pass (mtp-depth 5dbadce + ccce364) | the head's rule against the best fixed depth, 8 cells | 3 of 8 at 0.98, 0.905–1.015 (first pass: 6 of 8, 0.876–1.059); `auto`'s floor fell to 0.784 (English greedy; first pass 0.887); separate runs: rejected, the first pass went into integ2 | `gate9b/gate9b_depth.*`, `gate9b_depth2.*` |
| KV below 8 bits (kv-tail, emulated in a KL harness; q8 is the baseline) | KL and top-1 against the reference at 8K and 32K, 60K needles | every 4- and 5-bit format fails at 8K (mean KL ×4.8 of q8 and more); 6-bit + rotation + an fp16 tail of 1024 passes at 8K on code and English (×1.05–1.13 mean, ×1.32–1.34 p99) but fails at 32K on code (×2.02 / ×2.43, top-1 99.40 %); 60K needles 8 / 8 for fp16, q8 and both 6-bit candidates | `kvq4/` |

On the KV side two formats beat q8 on KL at the same or smaller size, emulated only: q8 with the
newest 1024 tokens kept in fp16 (×0.23–0.34 of q8's KL at 8K and 32K, 17 KiB per token) and
K 8-bit / V 6-bit + rotation + the fp16 tail (×0.49–0.74, 15 KiB against 17). Neither is built.
On Russian at 32K only q8 + tail reaches the 99.5 % top-1 bar (99.59 %); q8 itself is at 99.22 %,
K8 / V6 + tail at 99.29 %.

### What paid: spec-prefill (`raw/spf/spf9.txt`, `diff9.txt`)

Branch spec-prefill (0918548 / 31ac3d8): the drafters' work over the prompt's last window is
deferred behind the first token. Time to first token × plain, before → after [M]:

| | 2K | 8K | 32K |
|---|---|---|---|
| `auto`, cold | 1.064 → 0.996 | 1.073 → 1.025 | 1.085 → 1.061 |
| `auto`, prompt-cache hit | 1.359 → 0.981 | 1.406 → 0.973 | 1.366 → 1.011 |
| the head alone, cold | 1.077 → 1.000 | 1.042 → 1.031 | – |

Total time of a request on / old: 0.983–1.000. Tokens identical; the stored cache entries are
equal component by component (277 for the head, 279 with DFlash, 0 unequal) when each variant
starts from an empty cache.
In the run order that keeps the cache between variants one entry differs in its checkpoint
positions (the old path adds one at 2048), 230 equal and 50 not [M]. This A/B ran with reply KV
on, so its hit times are not the benchmark's.

## Open

- `auto` below the head on prose: 3–6 % at 1024 tokens, 5.6–13.5 % at 256. The fixed part
  (DFlash's start rounds, shadow drafts) wants a class prior that skips DFlash at the start of a
  prose request; the per-round part needs a look at why `auto`'s head rounds keep fewer tokens
  per unit time than the fixed head's.
- Sampled code under main's rule stays on the head (0.755 of DFlash); the branch fixes it (0.992)
  together with other changes: merge integ2's parts that paid, with the 27B gate.
- spec-prefill: merge candidate (cold and hit time to first token at plain's level, exact).
- The repeated 32K code cell under `auto` missed the prompt cache in the benchmark (84 s) while
  plain hit: not investigated.
- Rejection sampling with sampled drafts: the probe shows at most +0.076 first-draft acceptance;
  not built, speed not measured.
- KV: q8 + fp16 tail and K8 / V6 + rotation + tail are emulated only; a real kernel and a speed
  measurement come before any change of format.

# `QWEN_DRAFT=auto` on the 27B: the gate, what failed it, what was fixed

Qwen3.8-27B oQ4e (`weights/qwen38-27b-mlx-oq4e`, `QWEN_MRQMM4/5/6=1`), drafters DFlash2
(`weights/qwen38-27b-dflash2`) and the target's own MTP head (`weights/qwen38-27b-mtp`).
M1 Max 32 GB, mlx 0.32.3, mlx-vlm 0.7.6. Labels: [M] measured, [D] derived, [E] estimated.
The head is a function of the checkpoint: `scripts/perf/mtp/split_mtp.py` run again on the
oQ4e dir writes the same `model.safetensors` (sha256 `0b763aef…56adc4`) and config [M].
Design of the head and of the chooser: [../2026-11-16/mtp_draft.md](../2026-11-16/mtp_draft.md);
this report is the acceptance work on top of it. Tool: `scripts/perf/mtp/gate_auto.sh`
(`ab_mtp.py` + `gate_table.py`): every variant in one process, interleaved per prompt,
2 rounds × 256 tokens, 2K context, fp16 KV, thinking off.

## The gate

`auto` ≥ 0.97 × the best of the other variants in every cell — plain decode, DFlash alone
(`dflash`), the head with one fixed draft (`mtp1+q`), the head with its own gate
(`mtp0+q`). Cells: code / English / Russian at 2K + short English / Russian prose, greedy
and sampling (T 0.6). The floor: the head's machinery with no draft taken (`mtpz`) and the
plain path under speculation (`pplain`) ≥ 0.97 × plain.

## Result

Greedy, the rule as shipped (run 8, [M], tok/s, medians of 2 requests):

| cell | plain | DFlash | MTP, 1 draft | MTP, gated | `auto` | `auto` × plain | `auto` / best |
|---|---|---|---|---|---|---|---|
| code 2K | 16.5 | 34.6 | 21.9 | 27.9 | 31.8 | 1.94 | 0.926 |
| English 2K | 15.9 | 21.6 | 20.2 | 21.2 | 21.2 | 1.33 | 0.982 pass |
| Russian 2K | 16.0 | 13.4 | 20.0 | 19.8 | 19.2 | 1.20 | 0.959 |
| short English | 16.2 | 17.2 | 19.8 | 18.9 | 18.3 | 1.13 | 0.928 |
| short Russian | 16.2 | 12.0 | 19.2 | 18.5 | 17.6 | 1.09 | 0.914 |

1 of 5 cells at 0.97 (run 7: 2 of 5, code 0.973); every cell ≥ 0.91, and `auto` is above
plain in every cell (×1.09–1.94) where DFlash alone is ×0.74–0.84 on Russian. What the
gaps are made of [D]:

- code: only the first of the two requests (31.0 against DFlash's 36.3; the second 32.6
  against 32.9, the same rounds). A request starts on DFlash, whose prior is 3 tokens a
  round; when the head's arm in the class was last measured on code (here: by the `mtp0+q`
  variant just before) its rate reads higher, the head takes 20 rounds, and DFlash's shadow
  drafts take that long to show 4.8 tokens a round. In a session of `auto` requests only
  (the benchmark matrix, [decode_27b_bench.md](decode_27b_bench.md)) five of six code
  requests gave the head 0–6 rounds; the sixth stayed on it where it drafted as well as
  DFlash (3.4 tokens a round both, the head's round shorter).
- Russian 2K: 6 % of the rounds are DFlash's at the start of the request (its prior
  again), ~1 % shadow drafts.
- short prose: the gated head itself is 4–5 % under one fixed draft (probe rounds of all
  four drafts, 4 per request at +45 ms; on short Russian 2–3 bursts of plain steps per
  request at 16.2 tok/s against the head's 19.2), and `auto` 3–5 % under the gated head
  (DFlash's first rounds, shadow drafts, one more burst).

Sampling (T 0.6), the rule as shipped (run 8, [M], tok/s, medians of 2 requests):

| cell | plain | DFlash | MTP, 1 draft | MTP, gated | `auto` | `auto` × plain | `auto` / best |
|---|---|---|---|---|---|---|---|
| code 2K | 15.2 | 26.6 | 20.1 | 24.5 | 27.0 | 1.78 | 1.014 pass |
| English 2K | 15.4 | 19.9 | 18.9 | 19.9 | 18.9 | 1.23 | 0.952 |
| Russian 2K | 15.4 | 12.7 | 18.6 | 18.6 | 18.5 | 1.20 | 0.992 pass |
| short English | 15.6 | 16.7 | 18.4 | 18.3 | 17.0 | 1.09 | 0.922 |
| short Russian | 15.6 | 11.1 | 17.7 | 17.6 | 16.8 | 1.07 | 0.946 |

2 of 5 at 0.97, every cell ≥ 0.92 and above plain (×1.07–1.78). English 2K is a tie of
three settings (DFlash 19.9, the gated head 19.9, one draft 18.9): `auto` splits the
request between the arms (26 % of the rounds DFlash's, 7 and 3 switches in the two
requests) and lands on the lowest of them. Short prose: 5–8 % of the tokens went out in bursts of plain
steps (the gated head alone: 2–5 %), plus DFlash's first rounds. Two requests per cell do
not resolve 3 % under sampling: one variant's two requests of a cell differ by up to 27 %
(DFlash on short English 18.7 / 14.7; another trajectory, another text).

The floor, the rule as shipped (run 8, [M]): the head's machinery with no draft taken
(`mtpz`) 0.983–0.991 of plain, the plain path under speculation (`pplain`) 0.984–0.996,
English 2K and short Russian, greedy and sampling — 4 of 4 cells for both (run 2: 3 of 4,
the miss 0.969).

q8 KV (`QWEN_KV_BITS=8`, the 27B's serving format), greedy, the rule as shipped (run 8,
[M], tok/s; the fixed-draft variant here takes two drafts):

| cell | plain | DFlash | MTP, 2 drafts | MTP, gated | `auto` | `auto` × plain | `auto` / best |
|---|---|---|---|---|---|---|---|
| Russian 8K | 15.5 | 13.8 | 18.0 | 18.9 | 19.0 | 1.23 | 1.006 pass |
| English 2K | 15.8 | 21.7 | 19.0 | 20.4 | 21.6 | 1.37 | 0.996 pass |

2 of 2: on q8 `auto` takes the head on Russian (96 % of the rounds) and DFlash on English
(98 %), both requests of a cell within 1 % of each other. Longer contexts (8K and 32K on
q8, a whole session of `auto`) are in [decode_27b_bench.md](decode_27b_bench.md).

## The defects, in the order the gate found them

Each row is one commit and the gate run that measured the one before it. `auto / best` per
cell: code, English, Russian, short English, short Russian (greedy). The machine's speed
differs between runs (plain 11.4 … 17.4 tok/s: thermal state), so only ratios inside one
run compare.

| run | what the run showed | `auto / best` | fix |
|---|---|---|---|
| — | (the block prices below) 3-row verify blocks through the exact multi-row qmv cost ×1.80 of a step, through mrqmm ×1.48 | — | `auto`: `QWEN_VERIFY_QMV` 3 → 2 |
| 2 | (fixed variant: 2 drafts) the head's `lm_head` over a block's rows went through the library's `quantized_matmul` (a 1-draft round ×1.70 of a step); the depth rule read "seed confidence > 0.8 → 4 drafts": chained confidences come one draft late and are overconfident, 3 + drafts were kept in 1 round of 10 on prose | 0.91 0.98 1.01 0.99 0.97 | `kernels._rows_linear` (the block's own kernels); depth from measured keep rates per depth (`Clock.rate`), a probe round keeps them measured |
| 3 | 10–20 % of prose rounds took no draft: the seed's top-1 probability understates its keep rate under greedy (a seed of 0.4 is kept far more often) | 0.85 0.94 0.93 0.94 0.86 | the seed's keep rate measured per confidence bin (`Clock.seed`, 5 bins, optimistic start) |
| 4 | still 25 % of the tokens in no-draft rounds: a no-draft round was priced as a plain step of the plain loop (Clock 77.8 ms), the head's own cost ~15 % more | 0.91 0.89 0.91 0.86 0.85 | no-draft rounds timed as their own kind `("mtp", 0)`; a request starts on DFlash once the head's arm is known in its class (`Clock.arm`) |
| 5 | the chooser is right in steady state (code and English all DFlash from the 2nd request); bursts of 2 / 4 / 8 plain steps after one unlikely seed skip the sure tokens after it (short Russian: 53 burst rounds) | 0.92 0.99 0.92 0.89 0.82 | bursts 1, 1, 1, 2, 4, 8 |
| 6 | a single plain step between two rounds costs about a 1-draft round (no step-ahead, the head's catch-up); probes of all four drafts every 8th round cost 6 % | 0.91 0.98 0.91 0.91 0.93 | the rule picks among 1..`KMAX` only; plain steps only as a burst of 8 / 16 / 32 after 4 rounds in a row that kept nothing; probe interval doubles (8 … 64) while probes keep ≤ 1 draft |
| 7 | the gated head ≥ one fixed draft on code (+29 %) and English (+4 %), 0.99 / 0.965 / 0.95 on Russian / short prose; `auto` stuck on DFlash on short English | 0.97 1.01 0.94 0.89 0.91 | the head's arm keeps its class rate with the weight of 8 rounds, aged from the request's first sample; probes reset only at ≥ 3 kept |
| 8 | short English back on the head (97 % of the rounds); steady-state code and English equal DFlash alone within 1 %; the first code request of a class still detours through the head | 0.93 0.98 0.96 0.93 0.91 | — (open: a class prior for DFlash's tokens per round) |

Per-run tables (speeds, arms' shares, rounds by k and by accepted drafts, the Clock):
`raw/gate<N>_auto_greedy.{txt,json}`, `raw/gate2_auto_sampling.*`, `raw/gate2_floor_auto.*`.

The head with its own gate (`mtp0+q`) against one fixed draft (`mtp1+q`), per run — the
quantity the fixes of runs 3–6 moved [M]:

| run | code | English | Russian | short English | short Russian |
|---|---|---|---|---|---|
| 3 | 1.25 | 1.01 | 0.96 | 0.95 | 0.90 |
| 4 | 1.29 | 0.97 | 0.93 | 0.91 | 0.86 |
| 5 | 1.30 | 1.02 | 0.96 | 0.91 | 0.87 |
| 6 | 1.27 | 1.03 | 0.97 | 0.96 | 0.91 |
| 7 | 1.29 | 1.04 | 0.99 | 0.965 | 0.95 |
| 8 | 1.27 | 1.05 | 0.99 | 0.955 | 0.96 |

## What a verify block costs (the physics under the rule)

One quiet run, 15 iterations per row count after a 300 s cool-down, code prompt at 2K
(`raw/verify_block_anatomy.json`; measured on the `qmv5` branch, whose "cur" and "mm"
routes are main's exact multi-row qmv and mrqmm) [M]:

| rows T | exact multi-row qmv ("cur") | mrqmm ("mm") | what `auto` takes | × a 1-row step (59.2 ms) |
|---|---|---|---|---|
| 2 | 69.1 ms | 87.9 ms | qmv | 1.17 |
| 3 | 106.5 ms | 87.6 ms | mrqmm | 1.48 |
| 4 | 369 ms | 88.5 ms | mrqmm | 1.49 |
| 5 | 473 ms | 91.8 ms | mrqmm | 1.55 |

A second row rides in the slack of the scalar kernel (+16 %); from three rows on the cost
is the matrix kernel's fixed work per weight, flat in T (≈ ×1.5 for 3..8 rows). So with a
keep rate p1 for the first draft and p2.. for the chained ones, depth 1 pays when
(1 + p1) / 1.17 > 1, i.e. always on text the head can draft at all, and depths 2–4 cost the
same, so the choice is 1 or `KMAX`: all four pay only when
p1·p2·(1 + p3·(1 + p4)) > 0.5·(1 + p1) [D]. Code meets that (69 of 139 gated rounds keep
all four), Russian prose does not (run 2, where the old rule drafted four in 152 rounds:
77 kept two or more, 21 three or more — about 2.5 tokens a round [E] at ×1.5 the time of a
1-draft round's 1.8).

The same numbers bound what the head can reach on this machine: a 1-draft round is ×1.5 of
a plain step end to end (89 ms against 59: the block, the head, the sync), so ×1.24–1.29
on prose is (1 + p1) / 1.5 with p1 ≈ 0.75–0.8 kept, and the targets set for the head
(English ×1.6, code ×2.1) need blocks of 3–5 rows at ×1.20–1.35 of a step, below the
matrix kernel's measured ×1.48 [D]. On code the same class of gain comes from DFlash
inside `auto` (×2.07–2.13).

## Rules as shipped

- `MTPDraft.depth`: k = argmax over 1..`KMAX` of E(k) / T(k); E from `Clock.seed` (the
  seed's keep rate per confidence bin) and `Clock.rate` (the keep rate of depth i given
  i − 1 kept), both EMAs per class of requests; T from the Clock (`("mtp", k)`).
- Probe: all the drafts there is room for, every 8th drafting round, twice as rarely (up
  to 64) after each probe that kept ≤ 2.
- Plain steps: only after 4 drafting rounds in a row that kept nothing — a burst of 8,
  then one drafting round; 16, 32 while that one keeps nothing; a kept draft ends them.
- `Chooser`: a class's first request starts on the head; later ones on DFlash with the
  head's arm seeded from the class's last request (weight of 8 rounds); DFlash is
  shadow-drafted while the head runs; a switch needs a 2 % margin.

## Open

- Under sampling a request whose arms tie (English 2K: DFlash = the gated head) is split
  between them and pays the switches (0.95); short prose under sampling loses 5–8 % of its
  tokens to bursts of plain steps. More requests per cell are needed before tuning either:
  two do not resolve 3 % at T 0.6.
- 80K context for `auto` is not measured (8K and 32K are, in the benchmark report).
- DFlash's tokens per round start every request at a fixed prior (3.0): a class prior, as
  the head's arm has, would spare the first code request its detour and Russian prose its
  first DFlash rounds.
- The head is not sampled while DFlash runs; a shadow of its first draft would need the
  head's catch-up every round (not free). Until then the class prior carries it.
- Requests of one class mix code and prose: the per-class keep rates follow the last
  requests, a per-request start costs a few probe rounds.

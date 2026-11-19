# `QWEN_DRAFT=auto`: three versions of the arm rule against the gate (9B and 27B)

The arm rule is the part of `mtp.Chooser` that decides, round by round, whether a lone speculative
row drafts with DFlash or with the target's MTP head. This report gates three versions of it: the one
on this branch (from integ3) and two revisions measured on branches auto-arms and auto-arms2, which
are **not merged**.

Models: Ornith-1.5-9B 8-bit (`weights/ornith15-9b-mlx-8bit`) with DFlash (`ornith15-9b-dflash`, 4-bit at
load) and the 8-bit MTP head (`ornith15-9b-mtp-q8`); Qwen3.8-27B oQ4e (`weights/qwen38-27b-mlx-oq4e`)
with DFlash2 (`qwen38-27b-dflash2`) and its MTP head (`qwen38-27b-mtp`). fp16 KV, 2K context, thinking
off. M1 Max 32 GB, mlx 0.32.3, mlx-vlm 0.7.6. Labels: [M] measured, [D] derived. Tool:
`scripts/perf/mtp/ab_mtp.py` + `gate_table.py` (the 27B through `gate_auto.sh`): every variant in one
process, interleaved per prompt. All runs 2026-11-19. Raw: `raw/` (below; paths in them read
`$PERF_OUT`).

Findings first:

- Neither revision clearly passes the gate. On the 27B the cells at 0.97 went from 2 + 2 of 10
  (greedy + sampling, integ3) to 2 + 3 (auto-arms) and 2 + 4 (auto-arms2); on the 9B at 1024 tokens
  from 1 of 6 (integ3) to 4 of 6 (auto-arms, shared class state) and 3 of 6 for both revisions with a
  class state per variant; on short 9B requests 0 and 1 of 6 [M]. The offline replay of all three
  rules on the same recorded rounds puts them at 9, 12 and 11 of 28 cells [D].
- These are separate runs: plain decode itself moved 30.9–36.5 tok/s on the 9B and 15.3–17.5 tok/s on
  the 27B between them, and a cell holds 2–4 requests. Differences of a few hundredths between rules
  are not resolved. This branch keeps integ3's rule; the two revisions stay on their branches.
- On 27B code `auto` misses under every rule: 0.85–0.94 of DFlash alone [M]. The replay of the arm
  choices predicts 0.95–1.00 on the greedy code cell, so most of that miss is outside the choice [D].
- On greedy 9B prose the head's own depth rule (`mtp0+q`, `auto`'s head arm) was 0.903–0.905 of one
  fixed draft in the last run [M]: there no choice of arms can reach 0.97. Over all replayed runs the
  arm rule's ceiling is below 0.97 in 6 of 28 cells, all 9B prose [D].
- The speed ratio of the two arms over 16-token windows swings 0.6–2.0 on code with no
  autocorrelation, and the head's keep rate separates code from prose only over ~64 tokens: there is
  no short-window signal to switch on [D].

## The gate

`auto` ≥ 0.97 × the best of the other variants in every cell. The other variants: plain decode,
DFlash alone (`dflash`), the head with fixed drafts (9B: one, `mtp1+q`, and three, `mtp3+q`, the
latter not in the short set; 27B: two, `mtp2+q`) and the head with its own depth rule (`mtp0+q`, `QWEN_MTP_K=0`). `auto / best` is the median
over pairs of the per-pair ratio (`gate_table.py`) [M].

| run | cells | tokens | requests per cell |
|---|---|---|---|
| 27B gate (`gate_auto.sh gate`, `GATE_MODES` greedy / sampling) | code, English, Russian at 2K, short English, short Russian | 256 | 2 (auto-arms2: 4) |
| 9B gate at 1024 tokens | code, English, Russian at 2K, greedy and sampling (`@s`) | 1024 | 3 |
| 9B short | short English, short Russian, code at 2K, greedy and sampling | 256 | 3 |

Sampling is T 0.6, top_p 0.95, top_k 20.

**Class state.** `auto`'s round times and keep rates (`mtp.CLOCK`) are per class of requests and
process-wide. By default `ab_mtp.py` keeps one state for the whole process, so the fixed variants
that run before `auto` in a cell (the order changes from cell to cell and round to round) feed the
state `auto` decides from.
A server running `auto` alone has no such help. Branch auto-arms adds `ab_mtp.py --own-state`
(d2101ed, not on this branch): each variant gets its own state. The integ3 runs and the 27B runs use
the shared state; the 9B 1024-token gate was run both ways for auto-arms and with `--own-state` for
auto-arms2.

## The three rules

- **integ3 (this branch; the Chooser of branch auto-9b c383c7b, with the head's per-class depth rule
  of mtp-depth pass 1).** The head is the default arm and a class's first request starts on it;
  DFlash is scored by shadow drafts while the head runs (each priced at an assumed 0.10 of a DFlash
  round) and taken when its tokens per ms, less one standard error, beat the head's by 2 %. A
  class's later requests start with 3 rounds of DFlash and the head's rate guessed from the class's
  last request; a head not timed in the request is probed once (4 rounds) after 8 rounds of DFlash
  and kept on that probe's mean.
- **Iteration 1 (branch auto-arms, f35a645, measured, not merged).** Shadow drafts are charged at
  their measured cost (on the 9B 15–20 ms on a 36 ms head round, about 3× the assumed cost). A
  drafter that looks better but is not yet shown gets a short trial of real rounds instead of a
  shadow draft every round, where a real round costs less than a shadow draft; a failed trial
  doubles the wait before the next. The head's probe is kept only when it wins by one standard
  error, and a drafter below 0.8 of the head's rate is retired for the rest of the request.
- **Iteration 2 (branch auto-arms2, 2e1b3f7, docs f6c865e, measured, not merged).** On top of
  iteration 1, the head is probed again and again while a drafter runs (after 8, 16, 32 .. rounds),
  and the probes are judged together against the drafter's rate over the whole request, with a wide
  standard error per probe because the two arms ran on different text. Probes stop once the drafter
  is shown ahead.

## Results

`auto / best` per cell [M]; the best variant in brackets where it is not obvious (D = DFlash, h =
`mtp0+q`, h1 = `mtp1+q`); bold = at least 0.97.

### 27B, 256 tokens, shared state (`raw/gate27_int3/`, `raw/arms/gate27/`, `raw/arms2/gate27/`)

| rule | plain tok/s | code | English | Russian | short English | short Russian | cells ≥ 0.97 |
|---|---|---|---|---|---|---|---|
| greedy: integ3 | 17.1–17.5 | 0.936 (D) | 0.957 (D) | **0.992** (h) | **0.998** (h) | 0.953 (h) | 2 of 5 |
| greedy: iteration 1 | 16.6–16.9 | 0.901 (D) | 0.957 (D) | **1.027** (h) | **0.990** (h) | 0.959 (h) | 2 of 5 |
| greedy: iteration 2 | 16.1–16.5 | 0.877 (D) | 0.916 (D) | **0.972** (h) | **0.985** (h) | 0.951 (h) | 2 of 5 |
| T 0.6: integ3 | 16.9–17.4 | 0.854 (D) | 0.943 (h) | **0.977** (h) | 0.968 (h) | **0.977** (h) | 2 of 5 |
| T 0.6: iteration 1 | 16.3–16.6 | 0.878 (D) | **0.997** (h) | 0.959 (h) | **0.979** (h) | **0.977** (h) | 3 of 5 |
| T 0.6: iteration 2 | 15.3–15.9 | 0.853 (D) | **0.982** (D) | **0.976** (h) | **0.977** (h) | **0.979** (h) | 4 of 5 |

`auto` on greedy code: 32.3 / 30.5 / 29.4 tok/s against DFlash alone 34.5 / 33.8 / 32.4 (×1.87 / ×1.83
/ ×1.75 of plain) [M]. DFlash's share of `auto`'s rounds on greedy code: 66 %, 69 %, 36 % [M].

### 9B, 1024 tokens (`raw/gate9b/gate9b_int3_1k.*`, `raw/arms/gate9b/`, `raw/arms2/gate9b/`)

| rule, class state | plain tok/s | code @s | English @s | Russian @s | code | English | Russian | cells ≥ 0.97 |
|---|---|---|---|---|---|---|---|---|
| integ3, shared | 34.4–35.1 | **0.979** (D) | 0.950 (h1) | 0.966 (h1) | 0.927 (D) | 0.927 (h1) | 0.967 (h1) | 1 of 6 |
| iteration 1, shared | 35.0–36.0 | **0.990** (D) | **0.980** (h1) | **0.974** (h1) | **0.985** (D) | 0.956 (h1) | 0.951 (h1) | 4 of 6 |
| iteration 1, per variant | 32.9–33.7 | **0.978** (D) | 0.964 (h1) | **0.974** (h1) | 0.895 (D) | **0.973** (h) | 0.899 (h1) | 3 of 6 |
| iteration 2, per variant | 31.9–32.4 | 0.922 (D) | **0.978** (h1) | **0.986** (h) | 0.959 (D) | **1.004** (h1) | 0.960 (h1) | 3 of 6 |

The same rule (iteration 1) with the state shared and per variant: 4 and 3 of 6, greedy code 0.985
and 0.895 (separate runs) [M].

### 9B short requests, 256 tokens, shared state (`raw/arms/short9b/`, `raw/arms2/short9b/`)

| rule | plain tok/s | short English @s | short Russian @s | code @s | short English | short Russian | code | cells ≥ 0.97 |
|---|---|---|---|---|---|---|---|---|
| iteration 1 | 34.3–36.5 | 0.946 (h1) | 0.933 (h1) | 0.942 (D) | 0.957 (h1) | 0.959 (h1) | 0.958 (D) | 0 of 6 |
| iteration 2 | 30.9–33.1 | 0.969 (h1) | 0.952 (h1) | 0.888 (D) | 0.957 (h1) | **0.975** (h1) | 0.961 (D) | 1 of 6 |

integ3 was not run on this set.

### Offline replay (`raw/arms2/replay_final.txt`)

`scripts/perf/mtp/arms_sim.py` (branch auto-arms2, not on this branch) drives each version's real
`mtp.Chooser` through the calls the scheduler makes, on the rounds recorded with `--trace` in
iteration 1's five GPU runs (28 cells, 74 requests): each arm's rounds along the text, round times at
each kind's median over the run (the drift out), shadow and switch costs fitted from the run's `auto`
requests. "Ceiling" is the faster of DFlash alone and the head arm alone (`mtp0+q`) against the best
fixed variant: what no arm rule can beat. "Measured" is the run's own `auto`, computed by the
replay tool (it gives 11 cells where `gate_table.py` gives 12) [D]:

| | ceiling | measured (iteration 1) | integ3 | iteration 1 | iteration 2 |
|---|---|---|---|---|---|
| cells ≥ 0.97, of 28 | 22 | 11 | 9 | 12 | 11 |
| median cell | 1.000 | 0.966 | 0.948 | 0.962 | 0.965 |
| worst cell | 0.926 | 0.893 | 0.874 | 0.882 | 0.916 |
| requests: geometric mean, of 74 | – | – | 0.938 | 0.950 | 0.954 |
| requests below 0.90 | – | – | 11 | 7 | 6 |

Iteration 2 was tuned on two of these five runs (9B per-variant state, 9B short) and held out on the
other three. The tool's docstring puts a replayed cell within ±0.03 of the GPU typically, up to 0.08.
The replays at the end of each iteration-2 run's `.txt` (that run's rounds; integ3 / iteration 1 /
iteration 2) give 0 / 0 / 1 of 6 (9B per variant), 1 / 2 / 0 of 6 (9B short), 3 / 3 / 3 of 5 (27B
greedy) and 1 / 2 / 2 of 5 (27B T 0.6) cells at 0.97 [D].

## What the traces show

From the per-round traces of iteration 1's runs (the analysis behind iteration 2, f6c865e's
`drafters.md`) and the tables above:

- **Short windows are noise.** On code the ratio DFlash / head over 16-token windows swings
  0.6–2.0 with a lag-1 autocorrelation of about 0 (medians −0.01 / −0.03 / 0.07 at 16 / 32 / 64
  tokens). Following the last window's winner gives 0.972–0.974 of DFlash alone; knowing each
  window's winner in advance would give 1.017–1.105 [D]. On the 9B's 2K prompts both arms also run
  faster over a reply's first 16–32 tokens than over the rest, so rates measured on different
  stretches are biased [D].
- **The head's keep rate tells code from prose only over ~64 tokens.** Over its first 64 tokens the
  head kept ≥ 0.81 of its drafts on 9B code and ≥ 0.88 on 27B code (where DFlash / head is 0.98–1.23)
  against ≤ 0.79 everywhere else (DFlash / head ≤ 1.01); at 16–32 tokens the classes overlap, and the
  threshold differs by model [D]. A rule that starts on the head and decides from this table replayed
  worse on the tuned runs (0.932 geomean against 0.944) and was dropped.
- **The head's depth rule caps `auto` on greedy 9B prose.** `auto`'s head arm is the head with its own
  depth rule. In iteration 2's 9B run that rule was 0.905 (English) and 0.903 (Russian) of one fixed
  draft (`gate9b_arms_1k_own.txt`, the depth table) [M], and the replay's ceiling there is 0.890 and
  0.898 [D]: no arm choice reaches 0.97 in those cells. The same comparison in the other 1024-token
  runs: 0.983 / 0.959 (iteration 1, shared), 1.025 / 0.940 (iteration 1, per variant); in integ3's run
  the ratio of medians is 0.998 / 0.937 [D]. The depth rule is the same code in all of them, so its
  result on prose varies from run to run.
- **27B code: `auto` misses under every rule (0.853–0.936 of DFlash alone)** [M]. The replays of the
  greedy code cell give 0.951–0.998 for the three rules on iteration 1's run and 0.984–0.986 on
  iteration 2's, against 0.928 and 0.935 measured [D]: the choice of arms explains little of the miss.
  In iteration 2's run DFlash got 36 % of `auto`'s rounds (43 % of its tokens) on greedy code, where
  DFlash alone was ×1.19 of the head alone (×1.16 at T 0.6: 34 % of rounds) [M]. In iteration 1's traces
  `auto`'s DFlash rounds on 27B code took 147.8–154.7 ms against 136.4–148.6 ms for DFlash alone [M];
  on the 9B the arms run at their alone round times inside `auto`, and a switch costs 2–7 ms (the
  head's first round after DFlash) or nothing (DFlash after the head) [M].

## Not known

- Why `auto` misses on 27B code: the slower DFlash rounds inside `auto` (148–155 ms against 136–149
  ms alone) are unexplained, and a slower stretch of the GPU during those requests is not ruled out.
- Why the head's depth rule on greedy 9B prose lands at 0.90 in one run and 0.94–1.03 in others.
- Whether either revision is better than integ3's rule at all: the measured differences are within
  what separate runs and 2–4 requests per cell move, and the replay's per-cell error (±0.03, up to
  0.08) is as large as the differences between rules.

## Raw data

Under `raw/`, the layout of the runs' `$PERF_OUT` [M]:

- `gate9b/gate9b_int3_1k.{json,txt}`, `gate27_int3/mtp/gate_auto_{greedy,sampling}.{json,txt}`: integ3.
- `arms/gate9b/gate9b_arms_1k{,_own}.{json,txt}`, `arms/short9b/short9b_arms.*`,
  `arms/gate27/mtp/gate_auto_{greedy,sampling}.*`: iteration 1 (`_own` = per-variant state).
- `arms2/gate9b/gate9b_arms_1k_own.*`, `arms2/short9b/short9b_arms.*`,
  `arms2/gate27/mtp/gate_auto_{greedy,sampling}.*`: iteration 2; `arms2/replay_final.txt`: the replay
  above.
- The JSONs of iterations 1 and 2 (`arms/`, `arms2/`) are published without their per-round traces
  (`speed[*].trace`) and per-token arrival times (`speed[*].times`): with them the nine files take
  31 MB. Everything else in them is kept, and `gate_table.py` rebuilds the tables above from them;
  the replay and the trace analysis cannot be rerun from these files. integ3's JSONs are whole. The
  worker logs and the `*_fixed.json` inputs of the replay are left out. Each `.txt` is the tool's
  output; the iteration runs end with their replay table.

# Prompt cache on disk (`QWEN_PROMPT_CACHE_DISK`), 2026-10-24

Item 2 of `../deepinfer_ideas.md`. Motivation: `../../2026-10-17/REPORT-appworld-run-1017.md`.

- **8 processes.** The 6 GB cache holds 7 histories. The 8th is evicted every round, and each of
  its MISSes re-prefills 15–36K tokens (52–82 s).
- **The tail.** Histories of 27–58K tokens, 3–4 entries in RAM. MISSes on 54.6K / 58.4K tokens
  took 181 / 203 s.
- **Swap.** It stays at 5–6.7 GB with the cache in RAM, so a bigger RAM cache is not the answer.

An eviction used to be a deletion. Now an evicted entry is written to the SSD. A waiting request
whose deepest hit is on disk gets that entry loaded back while it waits. Its lookup then resumes
from it exactly as from a RAM entry.

CPU-only work so far. Nothing has been measured on the GPU. The GPU plan is at the end.

## What was done

- **`src/worker/pcdisk.py`** (new): `DiskTier`, the index, the blob files, and the writer and
  loader threads.
- **`src/worker/worker.py`**, small hooks:
  - `_PrefixCache.disk`, and `evict()` → `disk.offer(e)`;
  - `_PrefixCache.insert()`, which puts a loaded entry back through the store's budget rules;
  - `_Req.disk`, the per-request memo;
  - `_Scheduler.tick`: `_disk_tick` before admission, and `disk.flush()` at the end of the tick;
  - `_Scheduler._disk_tick` / `_disk_want` / `_disk_insert`, a new section next to the reply KV;
  - `_Scheduler._idle`, which drops finished loads that nobody waits for;
  - `_Scheduler.wake`;
  - `Engine.load`, which creates the tier, and `Engine._disk_key`.
- **`src/worker/tests/test_pcdisk.py`**: 19 CPU tests.
- **`scripts/perf/pcdisk/`**: `io_bench.py`, `exact_check.py`, `run_exact.sh`,
  `run_replay_disk.sh`, `disk_log.py`, `run_all.sh`.

## Knobs

| var | default | |
|---|---|---|
| `QWEN_PROMPT_CACHE_DISK` | unset = off | Directory for the cache files. Only our file names in it are ever touched. Needs `QWEN_PROMPT_CACHE=1`. |
| `QWEN_PROMPT_CACHE_DISK_GB` | `32` | Disk limit. The least recently used entries go first (used = written or loaded). |
| `QWEN_PROMPT_CACHE_DISK_KEEP` | `0` | `0`: our files are deleted at start. `1`: a previous run's files are kept when the validity key matches, else cleared. |
| `QWEN_PROMPT_CACHE_DISK_QUEUE_GB` | `4` | RAM of evicted entries waiting for the writer. Their MLX buffers stay alive until written and are charged in every budget check (`held_bytes`). An eviction past this cap is a plain deletion, as without the tier. There is no "one entry always" exception. `4` fits one 50K-token entry (~2.5 GB) or two 1 GB AppWorld entries. |
| `QWEN_PROMPT_CACHE_DISK_MIN_TOKENS` | `4096` | Shorter entries are not written. Their hit saves under ~2 slabs. |
| `QWEN_PROMPT_CACHE_DISK_WAIT_S` | `10` | How long the head of the queue waits for its own load. After that it is admitted without it. |
| `QWEN_PROMPT_CACHE_DISK_NOCACHE` | `1` | `F_NOCACHE` on every file, so the cache's GBs never sit in the macOS unified buffer cache next to the model. `0`: the OS caches as usual. |

## Design

### When an entry is written

- **On eviction, not on store.** Every eviction goes through `_PrefixCache.evict`: the store's
  soft cap and headroom, `_fits` (including "gives up its hit"), `_regrow`, `pc.insert`, and the
  reply store. Each of these calls `disk.offer(e)`.
- **Absorbed entries are not written.** An entry absorbed by a newer turn is removed from
  `entries`, not evicted, so it is not offered. The same holds for checkpoints thinned away
  before an eviction. A file therefore holds only what the RAM cache still had: one chain per
  conversation.
- **"Not kept" entries are not written.** An entry that does not fit the batch's headroom even
  alone would need its `[at, len)` copied out of the row. The batch has no room for that copy,
  and the row's own KV buffers are in use (decode writes into them).
- **Partial (cancelled-prefill) and reply entries are written like any other.**
- **Shutdown does not flush the RAM cache.** It is optional, and the `KEEP` use case only.
  `_Scheduler.stop()` returns as soon as `_close` sets `closed`, so a flush there would race the
  exit. This needs `stop()` to wait for a drain. Not done.
- **Shutdown drains the tier.** `main()` calls `disk.close(10)` after `sched.stop()`:
  1. nothing new is taken, and the offers not flushed are dropped;
  2. it waits until the writer and the loader are idle, so neither is inside a file write over
     an MLX buffer or an `mx.array` when the interpreter tears MLX down;
  3. if they are still busy after 10 s, `os._exit(1)`, as for a stuck scheduler tick.
- **One worker per directory.** `flock` on `pcdisk.lock`. A second worker on the same directory
  gets a `RuntimeError` at start, and its tier is off (logged).

### How an entry is written without blocking the scheduler

- **`offer`** appends the entry to a list. It runs inside `evict`, without MLX and without I/O.
- **`flush`**, at the end of the tick, on the scheduler thread:
  1. It takes the entry's segments (`_Seg`) and recurrent states, and finds the ones not already
     on disk (dedup, below).
  2. It sizes the files, then plans the disk LRU (`_plan`) before anything changes. The plan
     simulates the refcounts and never counts the blobs this entry shares as freed. When
     deleting every LRU entry would still leave no room, nothing is deleted and the entry is
     skipped.
  3. It registers the entry as `pending` and takes its blob references first. Only then are the
     planned entries dropped, so a blob it shares (a system segment whose last other holder was
     the LRU entry) can never go. The old order raised a KeyError there and left `pending`
     blobs behind (review #1).
  4. It takes **zero-copy numpy views** of the arrays: `mx.eval` (a no-op wait; they are
     evaluated), then `np.asarray(a)`. Checked on the GPU device: the numpy array shares the MLX
     buffer, with no memcpy and no stall. bfloat16 has no buffer format, so it goes as an MLX
     `uint16` view.
- **The writer thread** does plain file I/O over the views:
  - each file is written to `*.tmp` and then renamed;
  - then it drops the views. The MLX buffers are freed there: constant, evaluated arrays,
    through the allocator's lock. Handler threads already free the image processor's `to_mlx`
    arrays the same way.
- **Unlinks are queued under the index lock.** The writer's own drops (covered entries, LRU)
  queue their unlink while the index already says the files are gone. A re-offer of a live
  segment registered after that is therefore written after the unlink (FIFO), and a late unlink
  can never delete the fresh file (review #5).
- **Memory: an eviction to disk does not free RAM until the write ends** (~0.3–0.8 s/GB, not
  measured). This was review #2: the class of OOM on the 27B / 35B.
  - **What is held.** `DiskTier.held_bytes()` is the RAM the tier holds outside `pc.bytes`:
    - an evicted entry's freed bytes (`evict` passes `bytes before − after`), from the offer
      until its views are dropped after the write;
    - a load's `need`, from its start (before its arrays exist) until it is inserted (then it
      is `pc.bytes`) or dropped.
  - **What is charged.** `pc.charged() = pc.bytes + held` in every budget check: `_fits`,
    `_regrow`, the `limit` checks of `store` / `insert`, `_charge_fits`, the cascade tidy and
    `_disk_want`. The soft cap `QWEN_PROMPT_CACHE_GB` stays on `pc.bytes` alone.
  - **`evict(spill=…)`.** A spill (a write to disk) does not change `charged`, so a caller that
    needs the memory now passes `spill=False` and the entry is freed at once, as without the
    tier:
    - the `limit` loops of `store` / `insert` (the new segment is copied right after);
    - `_regrow`.
  - **The soft-cap loops spill only while it fits.** They spill while `charged + extra ≤ limit`
    (the held RAM still fits the headroom); else the entry is freed.
  - **`_fits` waits.** It evicts (with spill) until `reserved + need + pc.bytes ≤ budget`, as
    before. It then admits only when `reserved + need + charged ≤ budget`: a request whose
    room went to disk waits for the writer (≤ ~1–2 s), and nothing more is evicted meanwhile.
    Its live check sees the held buffers in `mx.get_active_memory()`.
  - **The queue cap is honest.** `held + freed ≤ QUEUE_GB`, else the eviction is a plain
    deletion.
  - **With the tier off**, `charged == bytes` and `spill` is unused, so every check is the old
    one bit for bit. The new `_fits` return can never trigger then: after its loop the bytes are
    within the budget, or only `keep` is left, which `blocked()` already ruled out.
- **Wake-up.** When the writer's and loader's queues drain, `on_drained` → `_Scheduler.wake()`
  wakes an idle loop. Its `_idle()` then clears the buffer pool and frees any load nobody waits
  for.

### Format and dedup

```
seg-<uid>.bin   one _Seg: KV of [a, b) of every full-attention layer
st-<uid>.bin    one recurrent state (a checkpoint or an end state)
ent-<uid>.bin   one entry: metadata header + its ids (int32)
pcdisk.json     {format, key}
```

- **Blob layout.** Each blob file is `MAGIC`, a u64 header length, then a JSON header (dtype,
  shape, offset and bf16 flag per array, plus the structure). The arrays follow in C order, at
  4096-aligned offsets. Its size is checked on every read, which catches torn files.
- **Entry metadata.** The chain as `(seg uid, lo, hi)`, the segments' `(a, b)`, `ckpts`
  (pos → state uid), `end`, the layout (class module and qualname, group size, bits, container),
  `quantized`, `low`, `reply`, `partial`, `chained`, `diverge`, `used`.
- **Why per-blob files and not one file per entry.** Segments and states are shared objects
  between RAM entries:
  - a system block's segment, shared by all episodes;
  - inherited checkpoints;
  - the pieces of a loaded entry, which the next turn absorbs.

  Blobs are therefore deduplicated by object identity:
  - `_Seg` → uid in a `WeakKeyDictionary`;
  - a state → uid keyed by its first array (held weakly), with every array's id compared.

  Blobs are reference-counted by the disk entries. A blob already on disk is not written again,
  and a blob whose object is still alive in RAM is reused at load, neither read nor duplicated.
  CPU tests show it:
  - two conversations over one system block share the segment file;
  - a turn built on a loaded entry writes only its new segment and states ("5 blobs shared");
  - a loaded entry evicted again unchanged is only touched, not rewritten;
  - an exact repeat writes 0.0 MB new.
- **Covered disk entries are dropped.** Once a written entry is complete, disk entries whose ids
  are a prefix of its ids are dropped, as the RAM store absorbs them. The same quantization
  exception applies: an unquantized entry stays under a quantized one.

### Index and lookup

- **The index lives in RAM.** `DiskEntry` keeps the ids as int32 numpy (4 B/token), the
  checkpoint positions, the flags and the blob uids.
- **`DiskEntry.best` is `_Entry.best` over numpy.** The LCP is one vector compare
  (`flatnonzero(a != b)`), so a scan is ~µs per entry. A test checks that the two agree on every
  entry and probe.
- **No prefix hashes.** At 32 GB and ~1 GB per entry there are about 30–60 entries, and a linear
  numpy scan is cheap. Hash chains at `<|im_start|>` / cut boundaries would only pay at thousands
  of entries.

### Load

- **The decision.** It is taken on the scheduler thread, in `_disk_tick`, every tick, for the
  first `max_batch` waiting requests:
  - the disk peek is memoized on `req.disk` by `(disk.version, pc.version)`;
  - the RAM depth comes from the existing `_peek` memo.
  - A load starts when the disk hit is ≥ RAM hit + `PREFILL_STEP`.
  - It also has to fit: the bytes not alive in RAM must be ≤ budget + prefill reserve −
    reserved (what a store may take), and live + that + prefill reserve ≤ cap (`_fits`'s live
    check).
  - One load at a time (`busy()`): the SSD is the bottleneck.
- **The loader thread** reads each array of a blob into numpy (`readinto`, GIL released) and
  turns it into `mx.array(numpy)` at once, before reading the next one (`_read_blob(each=…)`).
  The result is constant arrays: allocation + memcpy, no graph, no stream, the same as the
  processor's `to_mlx` on a handler thread. The numpy transient is one array, ≤ a layer's K or
  V (~100 MB at 50K tokens). Before review #3 it was the whole blob, i.e. 2× a 1.6 GB first
  segment; a test bounds it under 2 arrays.
- **Insert**, on the scheduler thread, at the next tick (`_disk_insert`). If the request still
  waits:
  1. `finish()` builds the `_Seg` / `_Entry` objects, reusing the live ones, and applies the
     bf16 views and `mx.eval`;
  2. then `pc.insert()`: it is not kept when it alone exceeds the headroom (beside the running
     prefill's parent). Otherwise other entries are evicted in `victim` order down to
     `QWEN_PROMPT_CACHE_GB` and the headroom, with `wanted` last and the running prefill's
     parent spared.
  3. `pc.version` bumps, so every peek memo (`req.peeked`) is recomputed. The head's `_fits`
     keeps the entry (≥ a slab deep), and `_wanted` protects it.

  The lookup in `_prompt_cache` is unchanged: it finds the entry as any RAM entry. A load whose
  request was admitted or cancelled meanwhile is dropped, at the next tick or at the idle pass.
- **Hold.** While the head's load runs (≤ `WAIT_S`), `tick` does not admit, so FIFO order is
  kept. Decode rows keep stepping. With nothing to do, the scheduler thread waits 5 ms on the
  load per tick instead of spinning.
- **Failures.** If the entry left the disk between the peek and the load (the writer's LRU),
  `load()` returns None and the scheduler peeks again. A failed read (a torn file, EIO) is a
  miss: logged, the request is prefilled as without the tier.

### Validity

`Engine._disk_key()` is a sha256 over:

- the model id; for each weights file its name, size and a sha256: of the header of a
  `*.safetensors` (the 8-byte length + the JSON of names, dtypes, shapes and offsets; the 10 GB
  of data are not read), of the whole of a `*.json`;
- the resolved numerics settings: `KV_KWARGS`, `FP16`, the slab policy (`PREFILL_STEP`,
  `PREFILL_MIN_STEP`, `PREFILL_BUDGET`, `_FIXED_SLAB`), `SDPA_FUSED`, `GDN_FAST`, `GDN_FUSE`,
  `max_batch` (mrqmm install, which matters for reply KV), and `PROMPT_CACHE_REPLY`. Reply
  entries hold decode-kernel KV, and the `QWEN_PROMPT_CACHE*` env prefix is otherwise excluded
  (review #4);
- every `QWEN_*` env var except the sizing and prompt-cache ones;
- the mlx and mlx-vlm versions;
- the sha256 of `worker.py`, `pfqmm.py`, `mrqmm.py`, `cascade.py` and `pcdisk.py`;
- the format version.

Clearing at start is the default. `KEEP=1` reuses files only under the same key. Being that
conservative is deliberate: any code change invalidates.

With `KEEP=1` every file is also `fsync`ed before its rename. Otherwise a power loss could leave
a full-size file of zeros under its name, which the size check would not catch. Without `KEEP`
the directory is cleared at start anyway, so the writes skip the `fsync`.

### Errors

Disk errors are never a request's failure:

| failure | effect |
|---|---|
| write failed | the entry is lost, as before the tier; its `.tmp` files are removed |
| read or validation failed | a miss |
| flush / insert / tick exception | logged; the queue goes on; a load whose insert raised is dropped (`_disk_take`: unpinned, hold released, so `busy()` cannot stay true) |

The writer's and loader's loops never die. A later entry that borrows a blob whose write failed
fails its own check and drops itself.

### Metrics

There are no proto changes: Health stays as is. Adding `prompt_cache_disk_*` fields 13–17 would
mean a gateway change as well; that is left to the integrator.

- **Log lines**, parsed by `scripts/perf/pcdisk/disk_log.py`:
  - `prompt cache disk: wrote an entry of N tokens (X MB new, k blobs shared) in S s; n entries, G GiB on disk`
  - `id=… prompt cache disk: loading an entry of N tokens (X MB to read) for a D-token hit (R in RAM)`
  - `id=… prompt cache disk: entry of N tokens loaded (X MB read in T ms; …)`
  - `… not written: <why>`, `… failed`, `… dropped: its request no longer waits`, `… not kept`
- **Counters.** `DiskTier.stats()` returns entries, bytes, writes, write_bytes, write_s, skipped,
  errors, deleted, loads, load_bytes, hits, discards, read_ms, pending and held.

## Invariants (self-review)

- **MLX on the scheduler thread only**, apart from constant-array allocation and free, as the
  image processor already does on handler threads:
  - views, `eval`, bf16 views, `_Entry` / `_Seg` construction and insert run on the scheduler;
  - the writer reads raw memory;
  - the loader calls `mx.array(numpy)`.
- **Memory accounting.**
  - Evicted entries not yet written and loads not yet inserted are `held` and charged with the
    cache in every budget check (`pc.charged()`). Evictions that need the RAM now free it at
    once (`spill=False`).
  - **Loaded entries** are inserted through the same headroom as a store. A load starts only when
    its non-live bytes fit beside what is already held, by budget and by live memory.
  - **Pending writes** are bounded by `QUEUE_GB`, honestly, with no exception. Their MLX
    buffers count in active memory until written.
- **Bits.**
  - **The bytes are exact.** fp16, fp32, packed uint32 and bf16 (as uint16) are written and read
    back as raw bytes. The tests compare every restore position of the loaded entry with the
    evicted one, byte by byte.
  - **A disk hit equals a RAM hit.** The tests check fp32, fp16 (8-bit model) and quantized KV
    (kv8, `kv_start` 24) against an all-RAM run: the same answers and the same resume point.
    `check_cold` on the resulting entries confirms that the restored bits equal a cold prefill.
- **Off is unchanged.** `QWEN_PROMPT_CACHE_DISK` unset means `pc.disk` is None. `tick` then
  makes one attribute read and one `is not None` test, `evict` / `_idle` the same, and nothing
  else changes. A test also runs with the tier on but `min_tokens` huge: the answers, resume
  points, entries and eviction counts equal those with it off.

## Tests (`tests/test_pcdisk.py`, CPU, ~35 s)

| test | what it checks |
|---|---|
| `test_disk_hit_equals_ram_hit_fp32` / `_fp16` / `_quantized` | a disk hit gives the same answers and resume point as an all-RAM run, `check_cold` passes, and an off run MISSes |
| `test_loaded_entry_bits` | metadata and every restore position byte-equal after write → load |
| `test_best_matches_entry_best` | `DiskEntry.best` agrees with `_Entry.best` |
| `test_shared_blobs_written_once_and_reused_live` | dedup, live reuse, no rewrite of an unchanged entry |
| `test_next_turn_on_a_loaded_entry` | the chain through the disk: borrowed blobs, the covered entry dropped, the next turn served |
| `test_lru_limit` | the limit, LRU order, no orphan files, bytes equal file sizes, an over-limit entry not written |
| `test_write_errors_never_fail_a_request` | ENOSPC: nothing on disk, nothing pending |
| `test_load_errors_are_misses` | a torn file and EIO both fall back to a miss |
| `test_off_changes_nothing` | off and on-without-writes give identical results |
| `test_keep_across_runs` | the index is rebuilt and loads; another key or KEEP off clears it; a second worker on the directory is refused (flock); `close()` |
| `test_blob_roundtrip` | dtypes, strided arrays, empty arrays, a truncated file |
| `test_insert_respects_the_headroom` | `pc.insert` respects the headroom |
| `test_disk_key` | the key is stable and moves with weights / numerics knobs, not with sizing knobs |
| `test_reply_entry_through_disk` | a reply entry goes through the disk and the next turn resumes after the reply, as with RAM |
| `test_load_while_rows_decode` | the head waits for its load while a row decodes |
| `test_wait_bound_and_abandoned_load` | the `WAIT_S` bound, and the idle pass frees an abandoned load |
| `test_lru_never_drops_a_blob_the_new_entry_shares` | review #1: an LRU entry that alone holds the shared segment is not dropped for nothing; the index equals the files; later entries are written and load |
| `test_late_unlink_never_deletes_a_rewritten_blob` | review #5: a re-offer after the writer's drop keeps its file and loads |
| `test_eviction_ram_is_charged_until_written` | review #2: `charged()` counts an evicted entry until its write ends; past the queue cap it is freed at once |
| `test_headroom_evictions_free_at_once` | review #2: the `limit` loops of `insert` / `store` free at once, the soft cap spills only within the headroom |
| `test_admission_waits_for_the_writer` | review #2: `_fits` admits only within the budget, the writer's hold included |
| `test_load_is_charged_until_inserted` | review #2: a load is held from its start until inserted |
| `test_load_reads_one_array_at_a_time` | review #3: the numpy peak is under 2 arrays, not the blob |
| `test_insert_failure_drops_the_load` | review #6: a raising insert drops the load; `busy()` false, nothing held |

The test file takes ~25 s.

Full suite (`pytest -q --ignore tests/test_mrqmm.py`, 2026-10-24): 359 passed, 49 skipped,
9.5 min (before the review fixes; after them see the commit). A light Metal correctness check, `io_bench.py --sizes-gb 0.4 --rounds 1` on the GPU
device, came back bit-exact (zero-copy views and loader-thread `mx.array` on Metal); it was not
a speed measurement.

## GPU plan

One command, run with the GPU lease held and nothing else on the GPU:

```zsh
setopt NO_BG_NICE
PERF_OUT=/tmp/perf scripts/perf/pcdisk/run_all.sh 10 8 3     # ~65 min
```

It writes `$PERF_OUT/pcdisk/summary.md` and copies the raw files to
`docs/perf/2026-10-24/raw/pcdisk/`. The worker comes from this worktree's `src/worker`, run with
the main tree's venv python. Never `uv`.

### 1. I/O microbench (`io_bench.py`, ~3 min, no model)

- **Setup.** Entries shaped like the 9B's: 8 FA layers × fp16 K/V (1, 4, T, 256) in two segments,
  plus 5 states of 24 GDN layers (~50 MiB each). Sizes 1 and 2.5 GB, 3 interleaved rounds,
  medians.
- **Measured.**
  - `flush_ms`: the scheduler's stall per write;
  - writer GB/s;
  - loader `read_ms` and GB/s;
  - `finish_ms`: the scheduler's stall per load;
  - the MLX peak;
  - a bit-exact check against arrays regenerated from their seeds.
- **Peak memory** ≈ the entry + one array: ≤ 3 GB.
- **Expected:**
  - flush < 5 ms;
  - write 1.5–3 GB/s, i.e. 0.4–0.7 s / 1 GB and 1–1.7 s / 2.5 GB;
  - read with `F_NOCACHE` 2–5 GB/s including the memcpy into MLX, i.e. 0.3–0.6 s / 1 GB and
    0.6–1.3 s / 2.5 GB;
  - finish < 20 ms.
- **Pass:** bit-exact in every round; flush_ms < 20 and finish_ms < 50 (the scheduler stalls
  less than one decode step); a 2.5 GB read < 2 s.

### 2. Exactness on the real 9B (`run_exact.sh 3`, ~8 min)

- **Worker.** `QWEN_MAX_BATCH=1`, `QWEN_PROMPT_CACHE_REPLY=0`, `QWEN_PROMPT_CACHE_GB=0.3` (each
  store keeps only the newest), disk on.
- **Pairs.** 3 pairs of AppWorld last calls (6–30K tokens). Per pair:
  1. prime A, evict it with B;
  2. teacher-forced `prompt_logprobs` of A with `prompt_cache: "read"` (a disk hit);
  3. the same with `prompt_cache: false` (cold), and compare: the scored values after the resume
     point must be `==`;
  4. evict again, then 64 greedy tokens of A + "Continue." as a hit and as cold: the content must
     be identical.

  The worker log must show exactly one disk load per hit.
- **Pass:** 3/3 pairs. Peak ~10–12 GB.

### 3. AppWorld replay, disk off vs on (`run_replay_disk.sh` × 4, ~50 min)

- **Load.** `replay.py --own-replies --ignore-eos off --episodes 8 --duration-min 10`, through
  :8090. The gateway is started if it is not running (thinking off).
- **Worker env.** The 17.10 knobs with a small cache, so evictions are frequent:

  ```
  QWEN_KV_BITS=0 QWEN_PROMPT_CACHE=1 QWEN_PROMPT_CACHE_GB=3 QWEN_BATCH_RESERVE_TOKENS=1024
  QWEN_PRESENCE_PENALTY=0 QWEN_PROMPT_CACHE_REPLY=1 [QWEN_PROMPT_CACHE_DISK=$PERF_OUT/pcdisk/disk_<run>]
  ```

- **Order.** `off_a → on_a → off_b → on_b`. Interleaving controls for drift: each pair is compared
  on its own, and both must agree.
- **Per run:**
  - memsampler (swap, CMPRS, pageins);
  - `sched_timer` (prefill / decode split);
  - the warm 8K probe after the replay (drift control);
  - `disk_log.py`: MISS / RAM-hit / disk-hit shares, TTFT and arrival → first token by class,
    MISS ≥ 10K, and the writer's and loader's MB, seconds and failures.
  - `compare.py` puts the four runs side by side.
- **Expected:**
  - **off:** many MISSes (8 histories vs ~3–4 entries of 0.65–1 GB).
  - **on:** MISSes drop to cold starts of new tasks. Disk-hit TTFT ≈ RAM-hit TTFT + 0.3–1 s
    (load). req/min +15–40 %. Peak and swap within ±1 GB of off: the tier holds nothing in RAM
    beyond the write queue.
- **Pass:**
  1. req/min(on) > req/min(off) in both pairs, with a mean gain ≥ +10 %;
  2. MISS share (on) < (off) in both pairs;
  3. no write or load failures, no request errors;
  4. worker peak (on) ≤ peak (off) + 1 GB, swap max (on) ≤ (off) + 1 GB;
  5. load p50 < 1.5 s;
  6. warm probes within ±15 % of each other (else repeat the pair that disagrees).
- **On fail:**
  - load latency too high → raise the prefetch depth (loads start for all `max_batch` waiting
    requests already), or check `NOCACHE=0`;
  - memory up → lower `QUEUE_GB`;
  - few disk hits → check `MIN_TOKENS` and the log's "not written" reasons.

### Risks to watch

- Write bandwidth competes with nothing on the GPU, but the SSD is shared with swap. Watch
  pageins and swapins against off.
- `_fits` charges an evicted entry until its write ends (`held`), so a request whose room went to
  disk waits for the writer, ≤ ~1–2 s per GB. Look for `queue_ms` growth in "on" at equal
  req/min. With `QUEUE_GB` at 4 the budget can shrink by up to 4 GB for that time.
- The `limit` loops of `store` / `insert` and `_regrow` free at once (`spill=False`), so under
  memory pressure entries are deleted, not written, exactly as without the tier. Count the "not
  written" log lines against the writes.
- Disk space: ≤ `QWEN_PROMPT_CACHE_DISK_GB` (32), deleted at the end of each run. 179 GB free on
  2026-10-24.

## Left open

- Shutdown / SIGTERM flush of the RAM cache (only for `KEEP=1`; needs `stop()` to wait for a
  drain).
- Health proto fields (`prompt_cache_disk_entries`, `_bytes`, `_writes`, `_loads`, `_read_ms`)
  plus the gateway metrics.
- Writing "not kept" entries: it would need the row's KV copied while the batch has no room. A
  zero-copy view of the row's live KV would keep decode from writing in place.
- A disk TTL (DeepInfer: 3 days). Today only the LRU and the size limit apply.
- `KEEP=1` across code changes: the key hashes the sources, so any edit clears the directory. A
  finer key (numerics-relevant code only) is possible, but riskier.

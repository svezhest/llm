# Qwen3.6-35B-A3B (MLX, 3-bit) — local setup

Running `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit` on Apple Silicon.

- **Model**: ~17.4 GB, 4 safetensors shards (5.35 / 5.32 / 5.29 / 1.41 GB), MoE with 3B active params.
- **Runtime**: `mlx-vlm` — it's a **vision-language** model, so use `mlx-vlm`, not `mlx-lm`.
- **Tested on**: M1 Max, 32 GB. Comfortable fit (model + context well under unified memory).

## Quick use (once weights are present)

```zsh
uv run python -c "
from mlx_vlm import load, generate
model, processor = load('~/models/Qwen3.6-35B-A3B-UD-MLX-3bit')
print(generate(model, processor, 'Привет! Кратко расскажи о себе.', max_tokens=128))
"
```

(Or pass the HF repo id `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit` if you let `hf` manage the cache.)

---

## Downloading the weights over a slow / flaky link

On a clean fast connection, `hf download unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit` is enough.
This section is the **hard-won recipe** for a degraded link (e.g. a throttling proxy
in front of everything), where the naive path fails.

### Pitfalls (why the simple commands fail)

| Symptom | Cause | Fix |
|---|---|---|
| `hf download` sits at 0 MB/s | Xet backend hangs on this network | `HF_HUB_DISABLE_XET=1` |
| Cache balloons with duplicate partials, never finishes | A `kill -9` stall-watchdog orphans the temp file every restart | Never hard-kill; graceful-exit resume only |
| Big shards never accumulate; `.incomplete` keeps vanishing | The proxy drops the CDN SSL mid-file; `hf` deletes the bad partial | Bypass `hf`, use a tool that **keeps** partials (aria2/curl `-c`) |
| File reaches full size but is **corrupt** | Proxy wrote block-page bytes to hit the size | **Verify SHA256** — size match ≠ integrity |
| Single-stream download capped at ~55 KB/s | CDN throttles **per connection** | aria2 `-x16` (got ~640 KiB/s, ~12×) |

### The approach that works

Download each shard with **aria2c**, straight into a local dir, **with its expected
SHA256** so aria2 self-verifies and re-fetches corrupt pieces — it won't report success
until the hash matches.

1. Get each shard's expected sha256 (`lfs.oid`) from the HF API:

   ```zsh
   curl -sS https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit/tree/main \
     | python3 -c "import sys,json;[print(x['path'],x.get('lfs',{}).get('oid','-')) for x in json.load(sys.stdin)]"
   ```

2. Fetch each shard (and the small json/tokenizer files) with integrity checking:

   ```zsh
   DIR=~/models/Qwen3.6-35B-A3B-UD-MLX-3bit
   BASE=https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit/resolve/main
   aria2c -c -x16 -s16 -k1M -j1 \
     --max-tries=0 --retry-wait=5 --connect-timeout=30 --timeout=60 \
     --check-integrity=true --checksum=sha-256=<oid-for-this-shard> \
     --dir="$DIR" -o model-00001-of-00004.safetensors \
     "$BASE/model-00001-of-00004.safetensors"
   ```

   Repeat for shards 2–4 plus: `model.safetensors.index.json`, `config.json`,
   `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json`.

### Notes

- aria2 **preallocates** the file, so `stat`/`ls` size jumps to full immediately.
  Track real progress in aria2's own log line (`<got>/<total> ... DL:<speed>`), not file size.
- Completed blobs from a prior partial `hf download` live under
  `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-UD-MLX-3bit/` and are
  content-addressed — safe to copy out and reuse (skip re-downloading what's intact).
- Auth: a global token at `~/.cache/huggingface/token` is picked up automatically; the
  shards themselves are public, so plain `aria2c`/`curl` need no token.
- Verify any file independently with: `shasum -a 256 <file>` vs the `lfs.oid` from step 1.

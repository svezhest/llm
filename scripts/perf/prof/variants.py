"""Decode-step GPU-time variants, interleaved in ONE process (the GPU's speed
drifts by +-30% over minutes on this machine, so only same-process ABAB
comparisons mean anything). Uses the worker's own code: worker._rows_attention
(installed as in Engine.load), RowsKV, _gdn_stack, mrqmm.install, the
worker's per-row greedy sampler. Rows are clones of one prefilled prompt cut
by r*37 tokens (ragged, like real traffic).

Variants (V_VARIANTS):
  base     the worker's step today (per-row attention, per-row greedy sampler)
  bsamp    one greedy argmax over the (B, V) logits
  noattn   per-row update+SDPA skipped (queries used as the attention output):
           an upper bound on what any attention rewrite can gain
  noupd    SDPA per row, cache update skipped (cost of the 2 SliceUpdates)

  cd src/worker; QWEN_MODEL=../../weights/ornith15-9b-mlx-8bit QWEN_KV_BITS=0 PYTHONPATH=. \
    V_CTX=1024 V_B=8,16 V_ROUNDS=4 .venv/bin/python $SP/prof/variants.py
"""
import gc, json, os, statistics, sys, time
S = os.environ.get("PERF_OUT", "/tmp/perf") + ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "microbench")); sys.path.insert(0, os.getcwd())
os.environ.setdefault("MB_CTX", os.environ.get("V_CTX", "1024"))
import mlx.core as mx
import microbench_decode_batch as mb
import worker as W
import mrqmm
from mlx_vlm.generate.common import generation_stream

CTX = int(os.environ.get("V_CTX", "1024"))
Bs = [int(b) for b in os.environ.get("V_B", "8,16").split(",")]
R = int(os.environ.get("V_ROUNDS", "4"))
STEPS = int(os.environ.get("V_STEPS", "40"))
VARS = os.environ.get("V_VARIANTS", "base,bsamp,noattn,noupd").split(",")

model = mb.load_like_worker(mb.MODEL)
lm = model.language_model
assert W._install_rows_attention(lm)
mrqmm.install(lm)
q35 = W._Q35
ids = [int(x) for x in mx.random.randint(1000, 150000, (CTX,)).tolist()]
with mx.stream(generation_stream):
    base_cache, last = mb.prefill_row(model, ids)

ORIG = q35.Qwen3_5Attention.__call__
VARIANT = {"v": "base"}


def attn(self, x, mask=None, cache=None, position_ids=None, position_embeddings=None,
         target_verify=False):
    v = VARIANT["v"]
    if not isinstance(cache, W.RowsKV) or v not in ("noattn", "noupd"):
        return ORIG(self, x, mask=mask, cache=cache, position_ids=position_ids,
                    position_embeddings=position_embeddings, target_verify=target_verify)
    B, L, _ = x.shape
    q_out, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    queries, gate = mx.split(q_out.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    queries, keys = self.rotary_emb.apply_rotary(queries, keys, position_ids, unsqueeze_dim=1)
    if v == "noattn":
        out = queries + 0 * (keys.sum() + values.sum())
    else:
        outs = []
        for i, c in enumerate(cache.caches):
            k, vv = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
            outs.append(q35.scaled_dot_product_attention(queries[i:i + 1], k, vv, cache=c,
                                                          scale=self.scale, mask=None))
        out = mx.concatenate(outs, axis=0)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(out * mx.sigmoid(gate))


q35.Qwen3_5Attention.__call__ = attn


def greedy_rows(lg):
    return mx.concatenate([mx.argmax(lg[i:i + 1] - mx.logsumexp(lg[i:i + 1], axis=-1, keepdims=True),
                                     axis=-1) for i in range(lg.shape[0])])


def greedy_batch(lg):
    return mx.argmax(lg - mx.logsumexp(lg, axis=-1, keepdims=True), axis=-1)


def run(B, v):
    VARIANT["v"] = v
    rows = [mb.clone_row(base_cache, cut=37 * r) for r in range(B)]
    gdn = W._gdn_stack(lm, rows, mx.float16)
    pos = [c[3].offset for c in rows]
    pick = greedy_batch if v == "bsamp" else greedy_rows
    with mx.stream(generation_stream):
        def step(y):
            cache = [gdn[i] if i in gdn else W.RowsKV([r[i] for r in rows])
                     for i in range(len(lm.model.layers))]
            p = mx.array([[q] for q in pos], dtype=mx.int32)
            out = lm(y[:, None], cache=cache, position_ids=p)
            for j in range(B):
                pos[j] += 1
            return pick(out.logits[:, -1, :])
        y = mx.array([last] * B, dtype=mx.int32)
        nxt = step(y); mx.async_eval(nxt)
        for k in range(4):
            cur = nxt; nxt = step(cur); mx.async_eval(nxt); cur.tolist()
        mx.eval(nxt)
        t0 = time.perf_counter()
        for k in range(STEPS):
            cur = nxt; nxt = step(cur); mx.async_eval(nxt); cur.tolist()
        mx.eval(nxt)
        dt = (time.perf_counter() - t0) / STEPS
    del rows, gdn
    gc.collect(); mx.clear_cache()
    return dt * 1e3


res = {}
for r in range(R):
    for B in Bs:
        order = VARS if r % 2 == 0 else VARS[::-1]
        for v in order:
            ms = run(B, v)
            res.setdefault(f"{B}_{v}", []).append(ms)
            print(f"round {r} ctx {CTX} B {B:2d} {v:7s} {ms:6.1f} ms/step", flush=True)
print(f"\n| ctx | B | " + " | ".join(VARS) + " |")
for B in Bs:
    b = statistics.median(res[f"{B}_base"])
    print(f"| {CTX} | {B} | " + " | ".join(
        f"{statistics.median(res[f'{B}_{v}']):.1f} ({statistics.median(res[f'{B}_{v}']) / b:.3f})" for v in VARS) + " |")
json.dump(res, open(os.environ.get("V_JSON", "/dev/null"), "w"))

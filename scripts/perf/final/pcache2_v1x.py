#!/usr/bin/env python3
"""V1 extra: hits that resume mid-prompt and prefill a multi-slab suffix.
  off --out F : cache-OFF references (P2, P3, P4, sysA)
  on  --out F : cache-ON worker, order P2 (cold) -> P3 (hit at system end, ~5K suffix) ->
                P4 (hit at a stride checkpoint inside the system prompt, multi-slab suffix) ->
                sysA (hit at P2's system-end checkpoint) -> P2 again -> P3 again
  cmp ON OFF"""
import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcache2_v1 as v1
w = v1.w
S = v1.SYS7K
U = lambda t, seed: t + "\n\nAttached context:\n" + w.filler(17000, seed=seed)
P = {
    "P2": [{"role": "system", "content": S}, {"role": "user", "content": U(w.TASKS[9], "u2")}],
    "P3": [{"role": "system", "content": S}, {"role": "user", "content": U(w.TASKS[10], "u3")}],
    "P4": [{"role": "system", "content": S[:18000] + "\n" + w.filler(9000, seed="p4tail")},
           {"role": "user", "content": w.TASKS[11]}],
    "sysA": [{"role": "system", "content": S}, {"role": "user", "content": w.TASKS[0]}],
}
mode, out = sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None
if mode == "cmp":
    sys.exit(0 if v1.cmp(sys.argv[2], sys.argv[3]) else 1)
order = ([("P2", "P2"), ("P3", "P3"), ("P4", "P4"), ("sysA", "sysA")] if mode == "off" else
         [("P2.cold", "P2"), ("P3.hit", "P3"), ("P4.hit", "P4"), ("sysA.hit", "sysA"),
          ("P2.hit", "P2"), ("P3.hit2", "P3")])
res = {}
for key, k in order:
    v1.run(res, key, P[k], 128, False)
json.dump({"results": res, "health": {"end": v1.health()}}, open(out, "w"), indent=1)

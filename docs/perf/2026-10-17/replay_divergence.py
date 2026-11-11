import json, os, urllib.request
O = os.path.expanduser("~/Projects/appworld/experiments/outputs")
def rec(run, task, k):
    L = [json.loads(l) for l in open(f"{O}/{run}/tasks/{task}/logs/lm_calls.jsonl")]
    return L[k]["input"], L[k]["output"]["choices"][0]["message"]["content"]
def call(i):
    body = {k: v for k, v in i.items() if k in ("messages", "temperature", "seed")}
    body["model"] = "ornith15-9b-mlx-8bit"
    req = urllib.request.Request("http://localhost:8090/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]
for task, k in [("530b157_3", 0), ("50e1ac9_2", 1), ("37a8675_2", 1)]:
    i, old = rec("ornith9b_react_dev.bak-1016", task, k)
    i2, new = rec("ornith9b_react_dev", task, k)
    assert i["messages"] == i2["messages"]
    outs = [call(i) for _ in range(3)]
    print(task, f"step {k+1}:", [("old" if o == old else "new" if o == new else "other") for o in outs])

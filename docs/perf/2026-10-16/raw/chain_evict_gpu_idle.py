import json, time, sys
sys.path.insert(0, sys.argv[1]); import drive
C = json.load(open(sys.argv[1] + "/chat_cold.json")); d = {r["tag"]: r for r in C}
time.sleep(320)
r = drive.call(d["turn7"]["messages"], "idle_turn7")
print("same as cold turn7:", r["content"] == d["turn7"]["content"] and r["completion_tokens"] == d["turn7"]["completion_tokens"])
json.dump([r], open(sys.argv[1] + "/idle.json", "w"), indent=1)

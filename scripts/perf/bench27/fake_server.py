#!/usr/bin/env python3
"""A stand-in OpenAI-compatible streaming server for testing the harness without a GPU
(session.py --system fake --probe-s 0 --cooldown 0): one token per chunk every --ms ms,
a role chunk first, usage at the end. Stdlib only."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
MS = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.reply({"data": [{"id": "fake-model"}]} if self.path.startswith("/v1/models") else {"status": "ok"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        print(json.dumps({k: v for k, v in req.items() if k != "messages"}), flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
            self.wfile.flush()
        send({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
        nprompt = sum(len(m["content"]) for m in req["messages"]) // 3
        time.sleep(nprompt / 50000)
        for i in range(req["max_tokens"]):
            send({"choices": [{"delta": {"content": f"tok{i} "}}]})
            time.sleep(MS / 1e3)
        send({"choices": [{"delta": {}, "finish_reason": "length"}]})
        send({"choices": [], "usage": {"prompt_tokens": nprompt, "completion_tokens": req["max_tokens"],
                                       "prompt_tokens_details": {"cached_tokens": 0}}})
        self.wfile.write(b"data: [DONE]\n\n")


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

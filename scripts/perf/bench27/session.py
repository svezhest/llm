#!/usr/bin/env python3
"""One benchmark session = one system in its own fresh processes, start to stop, one command:

  [GPU lease] -> preflight (no other model process, nice 0, wired limit, power mode)
  -> cooldown -> bandwidth probe (own process) -> start the system -> wait ready
  -> the plan through bench_decode.py (HTTP systems) or the CLI tool (llama-bench,
  perplexity) -> stop -> session.json (+ logs/, probe.json) -> release the lease.

  python3 scripts/perf/bench27/session.py --system ours-dflash --plan headline \\
      --out $PERF_OUT/bench27 [--code DIR] [--lease NAME] [--env K=V ...] [--set K=V ...]

Systems (`--list`): ours-plain / ours-dflash (worker + gateway of --code, the 4-bit),
ours-oq4e-plain / -dflash, mlx-vlm / mlx-vlm-dflash (stock `python -m mlx_vlm.server`
from the worker venv), llama-q4_0 / llama-q4kxl / llama-q4_0-mtp / llama-q4_0-dflash
(llama-server), llama-bench-q4_0 / -q4kxl, ppl-mlx-4bit / ppl-mlx-oq4e / ppl-llama-q4_0 /
ppl-llama-q4kxl (perplexity). --env adds/overrides process env (ours: the worker's);
--set K=V system parameters (n_max=3 for the drafters, ctx=..., model=<weights dir name>).
Weights: $BENCH_WEIGHTS or <main checkout>/weights. The lease: --lease NAME with
$BENCH_LEASE = the lease script (acquire/release NAME).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_decode as B  # noqa: E402
import prompts as P  # noqa: E402

CODE_DEFAULT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GGUF_DIR = "gguf-qwen38-27b"
GGUF = {"q4_0": "Qwen3.8-27B-Q4_0.gguf", "q4kxl": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "mtp": "mtp-Qwen3.8-27B-Q4_0.gguf", "dflash": "dflash-Qwen3.8-27B-Q4_0.gguf"}
# the 27B worker recipe (CLAUDE.md "DFlash2 on the 27B", long context; 80K fits under a 27 GiB wired limit)
OURS_ENV = {
    "QWEN_TEXT_ONLY": "1", "QWEN_MRQMM4": "1", "QWEN_KV_BITS": "0", "QWEN_MAX_BATCH": "1",
    "QWEN_MAX_CONTEXT": "98304", "QWEN_PROMPT_CACHE": "1", "QWEN_PROMPT_CACHE_MOVE": "1",
    "QWEN_PROMPT_CACHE_GB": "6", "QWEN_PROMPT_CACHE_CKPTS": "4", "QWEN_PREFILL_RESERVE_GB": "2.5",
    "QWEN_WIRED_LIMIT_GB": "27", "QWEN_CACHE_LIMIT_GB": "1", "QWEN_SDPA_FUSED_MIN_KV": "8192",
    "QWEN_PREFILL_BUDGET": "16777216", "QWEN_PRESENCE_PENALTY": "0", "LOG_LEVEL": "INFO",
}
MODEL_PROCS = ["python worker.py", "worker.py --addr", "llama-server", "llama-bench", "llama-perplexity",
               "mlx_vlm.server", "mlx_vlm.generate"]


def sh(argv, timeout=30):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<{argv[0]}: {e}>"


def ctx_for(plan_name):
    """llama-server -c: the plan's longest prompt + its generation, rounded up to 4096."""
    man = P.load_manifest()
    plan = B.load_plan(plan_name)
    need = 0
    for _, cell, pr in B.expand(plan, man):
        need = max(need, pr["ctx"] + cell.get("max_tokens", 256) + 2 * cell.get("turn2_max_tokens", 0) + 512)
    return max(8192, -(-need // 4096) * 4096)


# --------------------------------------------------------------------------- systems
def ours(ctx, model_dir, draft):
    w = ctx["weights"]
    env = dict(OURS_ENV, QWEN_MODEL=os.path.join(w, ctx["set"].get("model", model_dir)))
    if draft:
        env.update(QWEN_DRAFT=os.path.join(w, "qwen38-27b-dflash2"), QWEN_DRAFT_SAMPLING="1")
    env.update(ctx["env"])
    wdir = os.path.join(ctx["code"], "src", "worker")
    py = os.path.join(wdir, ".venv", "bin", "python")
    gw = os.path.join(ctx["code"], "target", "release", "qwen-gateway")
    if not os.path.exists(gw):
        raise SystemExit(f"no gateway binary at {gw}: `cargo build --release` in {ctx['code']} first "
                         "(while no GPU measurement runs)")
    if not os.path.exists(os.path.join(wdir, "worker_pb2.py")):
        raise SystemExit(f"no worker_pb2.py in {wdir}: copy it or run gen_proto.sh")
    waddr = f"127.0.0.1:{ctx['port'] + 1000}"
    return {
        "procs": [
            {"name": "worker", "argv": [py, "worker.py", "--addr", waddr, "--model", env["QWEN_MODEL"]],
             "env": env, "cwd": wdir, "mem": True},
            {"name": "gateway", "argv": [gw], "cwd": ctx["code"],
             "env": {"WORKER_ADDR": waddr, "GATEWAY_ADDR": f"127.0.0.1:{ctx['port']}", "RUST_LOG": "info"}},
        ],
        "ready": "/health", "model": None, "extra": {"presence_penalty": 0.0, "frequency_penalty": 0.0},
        "drop": [],
    }


def mlx_vlm(ctx, draft):
    w = ctx["weights"]
    model = os.path.join(w, ctx["set"].get("model", "qwen38-27b-mlx-4bit"))
    py = os.path.join(ctx["code"], "src", "worker", ".venv", "bin", "python")
    argv = [py, "-m", "mlx_vlm.server", "--model", model, "--host", "127.0.0.1", "--port", str(ctx["port"])]
    if draft:
        argv += ["--draft-model", os.path.join(w, "qwen38-27b-dflash2"), "--draft-kind", "dflash",
                 "--draft-block-size", str(ctx["set"].get("block", 8))]
    argv += ctx["set"].get("args", "").split()
    return {"procs": [{"name": "mlx_vlm", "argv": argv, "env": dict(ctx["env"]), "cwd": ctx["dir"], "mem": True}],
            "ready": "/health", "model": model, "extra": {}, "drop": []}


def llama(ctx, quant, spec=None):
    g = os.path.join(ctx["weights"], GGUF_DIR)
    gguf = os.path.join(g, GGUF[quant])
    c = int(ctx["set"].get("ctx", ctx_for(ctx["plan"])))
    argv = ["llama-server", "-m", gguf, "-ngl", "99", "-fa", "on", "-c", str(c), "--jinja", "-np", "1",
            "--host", "127.0.0.1", "--port", str(ctx["port"]), "--cache-ram", str(ctx["set"].get("cache_ram", 0))]
    if spec:
        argv += ["--spec-type", f"draft-{spec}", "-md", os.path.join(g, GGUF[spec])]
        if "n_max" in ctx["set"]:
            argv += ["--spec-draft-n-max", str(ctx["set"]["n_max"])]
    argv += ctx["set"].get("args", "").split()
    for f in [gguf] + ([os.path.join(g, GGUF[spec])] if spec else []):
        if not os.path.exists(f) or os.path.exists(f + ".aria2"):
            raise SystemExit(f"{f}: missing or still downloading")
    return {"procs": [{"name": "llama-server", "argv": argv, "env": dict(ctx["env"]), "cwd": ctx["dir"],
                       "mem": True}],
            "ready": "/health", "model": "qwen38-27b-" + quant,
            "extra": {"cache_prompt": True, "repeat_penalty": 1.0, "presence_penalty": 0.0,
                      "frequency_penalty": 0.0},
            "drop": [], "template_check": True}


def llama_bench(ctx, quant):
    gguf = os.path.join(ctx["weights"], GGUF_DIR, GGUF[quant])
    depths = ctx["set"].get("depths", "0,8192,32768")
    cmds = [["llama-bench", "-m", gguf, "-ngl", "99", "-fa", "1", "-p", "512", "-n", "128", "-r", "3",
             "-d", depths, "-o", "json"]]
    if ctx["set"].get("d80k", "0") == "1":
        cmds.append(["llama-bench", "-m", gguf, "-ngl", "99", "-fa", "1", "-p", "512", "-n", "128", "-r", "1",
                     "-d", "81920", "-o", "json"])
    return {"cli": cmds, "parse": "json"}


def ppl_cli(ctx, kind, what):
    corpus = os.path.join(ctx["dir"], f"{ctx['set'].get('corpus', 'ppl_en')}.txt")
    pyp = os.path.join(HERE, "ppl.py")
    chunks = ["--chunks", str(ctx["set"]["chunks"])] if "chunks" in ctx["set"] else []
    pre = [[sys.executable, pyp, "corpus", "--name", ctx["set"].get("corpus", "ppl_en"), "--out", corpus]]
    if kind == "llama":
        gguf = os.path.join(ctx["weights"], GGUF_DIR, GGUF[what])
        return {"cli": pre + [[sys.executable, pyp, "llama", "--gguf", gguf, "--corpus", corpus, "--json",
                               os.path.join(ctx["dir"], "ppl.json"), "--log",
                               os.path.join(ctx["dir"], "logs", "llama-perplexity.log")] + chunks],
                "parse": "ppl"}
    wdir = os.path.join(ctx["code"], "src", "worker")
    env = {"QWEN_MODEL": os.path.join(ctx["weights"], what), "QWEN_KV_BITS": "0", "PYTHONPATH": ".",
           "QWEN_TEXT_ONLY": "1", **ctx["env"]}
    return {"cli": pre + [[os.path.join(wdir, ".venv", "bin", "python"), pyp, "mlx", "--corpus", corpus, "--json",
                           os.path.join(ctx["dir"], "ppl.json")] + chunks],
            "cli_env": env, "cli_cwd": wdir, "parse": "ppl"}


def fake(ctx):
    return {"procs": [{"name": "fake", "argv": [sys.executable, os.path.join(HERE, "fake_server.py"),
                                                str(ctx["port"]), str(ctx["set"].get("ms", 10))],
                       "cwd": ctx["dir"], "mem": True}],
            "ready": "/health", "model": None, "extra": {}, "drop": []}


SYSTEMS = {
    "fake": fake,
    "ours-plain": lambda c: ours(c, "qwen38-27b-mlx-4bit", False),
    "ours-dflash": lambda c: ours(c, "qwen38-27b-mlx-4bit", True),
    "ours-oq4e-plain": lambda c: ours(c, "qwen38-27b-mlx-oq4e", False),
    "ours-oq4e-dflash": lambda c: ours(c, "qwen38-27b-mlx-oq4e", True),
    "mlx-vlm": lambda c: mlx_vlm(c, False),
    "mlx-vlm-dflash": lambda c: mlx_vlm(c, True),
    "llama-q4_0": lambda c: llama(c, "q4_0"),
    "llama-q4kxl": lambda c: llama(c, "q4kxl"),
    "llama-q4_0-mtp": lambda c: llama(c, "q4_0", "mtp"),
    "llama-q4_0-dflash": lambda c: llama(c, "q4_0", "dflash"),
    "llama-bench-q4_0": lambda c: llama_bench(c, "q4_0"),
    "llama-bench-q4kxl": lambda c: llama_bench(c, "q4kxl"),
    "ppl-mlx-4bit": lambda c: ppl_cli(c, "mlx", "qwen38-27b-mlx-4bit"),
    "ppl-mlx-oq4e": lambda c: ppl_cli(c, "mlx", "qwen38-27b-mlx-oq4e"),
    "ppl-mlx-5bit": lambda c: ppl_cli(c, "mlx", "qwen38-27b-mlx-5bit"),
    "ppl-llama-q4_0": lambda c: ppl_cli(c, "llama", "q4_0"),
    "ppl-llama-q4kxl": lambda c: ppl_cli(c, "llama", "q4kxl"),
}


# --------------------------------------------------------------------------- helpers
def preflight(force):
    out = {}
    ps = sh(["ps", "-axo", "pid,nice,command"])
    alive = [l for l in ps.splitlines() if any(p in l for p in MODEL_PROCS) and "session.py" not in l
             and "grep" not in l]
    out["model_processes"] = alive
    if alive and not force:
        raise SystemExit("other model processes are alive (one model on the GPU at a time):\n" + "\n".join(alive))
    out["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
    if out["nice"] > 0 and not force:
        raise SystemExit(f"this process runs at nice {out['nice']} (zsh BG_NICE?): `setopt NO_BG_NICE` "
                         "or run in the foreground")
    out["wired_limit_mb"] = sh(["sysctl", "-n", "iogpu.wired_limit_mb"])
    out["powermode"] = [l.strip() for l in sh(["pmset", "-g"]).splitlines() if "powermode" in l.lower()]
    out["power_source"] = sh(["pmset", "-g", "batt"]).splitlines()[:1]
    out["memory_pressure"] = [l for l in sh(["memory_pressure", "-Q"]).splitlines() if "free" in l.lower()]
    out["vm_stat"] = sh(["vm_stat"]).splitlines()[:6]
    out["loadavg"] = os.getloadavg()
    return out


def versions(code):
    py = os.path.join(code, "src", "worker", ".venv", "bin", "python")
    v = {"mlx/mlx_vlm": sh([py, "-c", "import mlx.core as m, importlib.metadata as im;"
                                      "print(m.__version__, im.version('mlx-vlm'))"]),
         "llama.cpp": sh(["llama-server", "--version"]).splitlines()[-2:] if shutil.which("llama-server") else None,
         "code_commit": sh(["git", "-C", code, "rev-parse", "HEAD"]),
         "code_dirty": bool(sh(["git", "-C", code, "status", "--porcelain", "--untracked-files=no"])),
         "bench_commit": sh(["git", "-C", HERE, "rev-parse", "HEAD"]),
         "gpu_cores": sh(["sh", "-c", "ioreg -l | grep -m1 -o '\"gpu-core-count\" = [0-9]*'"]),
         "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"])}
    return v


class MemSampler(threading.Thread):
    """footprint of one pid every `every` s: peak + series (GiB). MLX / Metal buffers are counted."""

    def __init__(self, pid, every=5.0):
        super().__init__(daemon=True)
        self.pid, self.every, self.series, self.stop_ev = pid, every, [], threading.Event()
        self.t0 = time.time()

    def run(self):
        while not self.stop_ev.is_set():
            o = sh(["footprint", "-p", str(self.pid), "--noCategories", "-f", "bytes"], timeout=20)
            for line in o.splitlines():
                if "Footprint:" in line:
                    try:
                        b = int(line.split("Footprint:")[1].split()[0])
                        self.series.append((round(time.time() - self.t0, 1), round(b / 2**30, 3)))
                    except (ValueError, IndexError):
                        pass
                    break
            self.stop_ev.wait(self.every)

    def result(self):
        return {"peak_gib": max((g for _, g in self.series), default=None), "samples": len(self.series),
                "series": self.series[::max(1, len(self.series) // 400)]}


def wait_ready(url, procs, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for p in procs:
            if p["popen"].poll() is not None:
                raise RuntimeError(f"{p['name']} exited with {p['popen'].returncode} before ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    try:
                        body = json.loads(r.read())
                    except ValueError:
                        body = {}
                    # mlx_vlm.server: up before its model is; llama-server: 503 while loading
                    if not (isinstance(body, dict) and "loaded_model" in body and not body["loaded_model"]):
                        return round(time.time() - t0, 1)
        except Exception:  # noqa: BLE001 -- not up yet
            pass
        time.sleep(2)
    raise RuntimeError(f"not ready within {timeout} s")


def start(procs, logdir):
    for p in procs:
        env = dict(os.environ)
        env.update(p.get("env") or {})
        f = open(os.path.join(logdir, p["name"] + ".log"), "w")
        p["popen"] = subprocess.Popen(p["argv"], cwd=p.get("cwd"), env=env, stdout=f, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        p["logf"] = f
        p["pid"] = p["popen"].pid


def stop(procs, grace=40):
    for p in reversed(procs):
        if p.get("popen") and p["popen"].poll() is None:
            try:
                os.killpg(p["popen"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p["popen"].wait(grace)
            except subprocess.TimeoutExpired:
                os.killpg(p["popen"].pid, signal.SIGKILL)
                p["popen"].wait(10)
                p["killed"] = True
        if p.get("logf"):
            p["logf"].close()
        p["returncode"] = p["popen"].returncode if p.get("popen") else None


def lease(cmd, name):
    path = os.environ.get("BENCH_LEASE")
    if not path:
        raise SystemExit("--lease needs $BENCH_LEASE = the lease script")
    argv = [sys.executable, path, cmd, name] + (["--wait", "14400"] if cmd == "acquire" else [])
    r = subprocess.run(argv)
    if r.returncode != 0:
        raise SystemExit(f"lease {cmd} failed")


# --------------------------------------------------------------------------- session
def run_session(system, plan, out_root, code=CODE_DEFAULT, env=None, sets=None, port=8095, cooldown=60,
                probe_s=30, lease_name="", force=False, ready_timeout=1500, tag="", log=print):
    sid = time.strftime("%Y%m%d-%H%M%S") + f"-{system}-{plan}" + (f"-{tag}" if tag else "")
    d = os.path.join(out_root, sid)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    ctx = {"code": os.path.abspath(code), "weights": P.weights_dir(), "env": env or {}, "set": sets or {},
           "port": port, "plan": plan, "dir": d}
    spec = SYSTEMS[system](ctx)
    meta = {"session": sid, "system": system, "plan": plan, "tag": tag, "set": ctx["set"], "env_overrides": ctx["env"],
            "code": ctx["code"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "started"}
    procs = spec.get("procs", [])
    held = False
    try:
        if lease_name:
            lease("acquire", lease_name)
            held = True
        meta["preflight"] = preflight(force)
        meta["versions"] = versions(ctx["code"])
        if cooldown:
            log(f"[{sid}] cooldown {cooldown} s")
            time.sleep(cooldown)
        if probe_s:
            py = os.path.join(ctx["code"], "src", "worker", ".venv", "bin", "python")
            pj = os.path.join(d, "probe.json")
            r = subprocess.run([py, os.path.join(HERE, "bwprobe.py"), "--seconds", str(probe_s), "--json", pj],
                               capture_output=True, text=True)
            meta["probe"] = json.load(open(pj)) if os.path.exists(pj) else {"error": r.stderr[-1000:]}
            log(f"[{sid}] probe {meta['probe']}")
        if "cli" in spec:
            meta["cli"] = []
            for argv in spec["cli"]:
                env_c = dict(os.environ)
                env_c.update(spec.get("cli_env", {}))
                t = time.time()
                r = subprocess.run(argv, cwd=spec.get("cli_cwd", d), env=env_c, capture_output=True, text=True)
                name = os.path.basename(argv[1] if argv[0] == sys.executable or argv[0].endswith("python")
                                        else argv[0])
                with open(os.path.join(d, "logs", f"{name}-{len(meta['cli'])}.log"), "w") as f:
                    f.write(r.stderr)
                ent = {"argv": argv, "returncode": r.returncode, "seconds": round(time.time() - t, 1)}
                if spec["parse"] == "json" and r.returncode == 0:
                    try:
                        ent["result"] = json.loads(r.stdout)
                    except json.JSONDecodeError:
                        ent["stdout"] = r.stdout[-4000:]
                else:
                    ent["stdout_tail"] = r.stdout[-4000:]
                meta["cli"].append(ent)
                log(f"[{sid}] {name}: rc {r.returncode} in {ent['seconds']} s")
            if os.path.exists(os.path.join(d, "ppl.json")):
                meta["ppl"] = json.load(open(os.path.join(d, "ppl.json")))
            meta["status"] = "ok" if all(c["returncode"] == 0 for c in meta["cli"]) else "failed"
        else:
            start(procs, os.path.join(d, "logs"))
            meta["procs"] = [{"name": p["name"], "argv": p["argv"], "env": p.get("env"), "pid": p["pid"]}
                             for p in procs]
            mem = [MemSampler(p["pid"]) for p in procs if p.get("mem")]
            for m in mem:
                m.start()
            url = f"http://127.0.0.1:{port}"
            meta["ready_s"] = wait_ready(url + spec["ready"], procs, ready_timeout)
            log(f"[{sid}] ready in {meta['ready_s']} s")
            if spec.get("template_check"):
                try:
                    meta["template_check"] = B.template_check(url)
                except Exception as e:  # noqa: BLE001 -- a check, never the session's outcome
                    meta["template_check"] = [{"error": f"{type(e).__name__}: {e}"}]
                log(f"[{sid}] template check: {meta['template_check']}")
            res = B.run_plan(url, plan, system, spec["model"], spec["extra"], log=lambda m: log(f"[{sid}] {m}"),
                             drop=spec["drop"])
            meta.update(res)
            meta["summary"] = B.table(res["results"])
            for m in mem:
                m.stop_ev.set()
            meta["memory"] = [m.result() for m in mem]
            meta["status"] = "ok" if not any(r.get("error") for r in res["results"]) else "errors"
    except BaseException as e:  # noqa: BLE001 -- the session record always gets written
        meta["status"] = "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        log(f"[{sid}] FAILED: {meta['error']}")
        if isinstance(e, KeyboardInterrupt):
            raise
    finally:
        stop(procs)
        meta["proc_exit"] = {p["name"]: p.get("returncode") for p in procs}
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.join(d, "session.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        log(f"[{sid}] {meta['status']}; wrote {os.path.join(d, 'session.json')}")
        if held:
            lease("release", lease_name)
    return meta


def kv(items):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system")
    ap.add_argument("--plan", default="smoke")
    ap.add_argument("--out", default=os.path.join(os.environ.get("PERF_OUT", "/tmp/perf"), "bench27"))
    ap.add_argument("--code", default=CODE_DEFAULT, help="the tree whose worker + gateway are 'ours'")
    ap.add_argument("--env", action="append", help="K=V process env (ours: the worker)")
    ap.add_argument("--set", action="append", help="K=V system parameter")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--probe-s", type=int, default=30)
    ap.add_argument("--lease", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true", help="skip the alive-process / nice checks")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.system:
        print("systems:", ", ".join(SYSTEMS))
        print("plans:", ", ".join(k for k in json.load(open(B.PLANS)) if not k.startswith("_")))
        return
    meta = run_session(a.system, a.plan, a.out, a.code, kv(a.env), kv(a.set), a.port, a.cooldown, a.probe_s,
                       a.lease, a.force, tag=a.tag)
    for row in meta.get("summary", []):
        print(row)
    sys.exit(0 if meta["status"] == "ok" else 1)


if __name__ == "__main__":
    main()

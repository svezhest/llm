"""Lifecycle and capacity: failed decode steps, a scheduler that gives up,
Shutdown, the gRPC generation slots, the memory budget and the per-row
reservation, and the handler talking to a real scheduler thread."""
import os
import queue
import threading
import time
import types

import grpc
import mlx.core as mx
import numpy as np
import pytest

import worker
import worker_pb2 as pb
import worker_pb2_grpc as pb_grpc
from conftest import prompts, request, run, scheduler


def collect(req, timeout=60):
    """Read a request's events until its terminal one (threaded scheduler)."""
    toks = []
    while True:
        ev = req.out.get(timeout=timeout)
        if ev[0] == "tok":
            toks.append(ev[1])
        else:
            return toks, ev


def wait_for(cond, timeout=30):
    t = time.time()
    while not cond():
        assert time.time() - t < timeout, "condition not reached"
        time.sleep(0.005)


def test_failed_step_fails_its_rows_only(dense, monkeypatch):
    """A decode step that raises fails the rows in it; the request being
    prefilled at that moment, later requests and the scheduler go on (the
    loop below is _run's: tick, and _fail_rows on an exception)."""
    monkeypatch.setattr(worker, "PREFILL_STEP", 4)
    monkeypatch.setattr(worker, "PREFILL_MIN_STEP", 2)
    ps = prompts([12, 12, 60, 12], 40)
    solo_c = run(scheduler(dense, max_batch=1), [request(ps[2], 10, "c")])["c"]
    solo_d = run(scheduler(dense, max_batch=1), [request(ps[3], 10, "d")])["d"]
    sched = scheduler(dense, share=0.3)
    a, b, c, d = (request(ps[0], 40, "a"), request(ps[1], 40, "b"),
                  request(ps[2], 10, "c"), request(ps[3], 10, "d"))
    step, hit = sched._step, []

    def failing(rows):
        if not hit and len(rows) >= 2 and sched.task_row is not None:
            hit.append(sched.task_row.req.id)
            raise RuntimeError("injected step failure")
        return step(rows)
    sched._step = failing
    for r in (a, b):
        sched.submit(r)
    d_sent = False
    for t in range(5000):
        if t == 3:
            sched.submit(c)
        if hit and not d_sent:
            sched.submit(d)
            d_sent = True
        if d_sent and not (sched.rows or sched.task or sched.waiting):
            break
        try:
            sched.tick()
        except Exception as e:  # noqa: BLE001 — as _Scheduler._run
            sched._fail_rows(e)
    else:
        raise AssertionError("scheduler did not finish")
    assert hit == ["c"]
    for r in (a, b):
        toks, end = collect(r, 1)
        assert end[0] == "error" and "injected" in end[2]
    assert collect(c, 1) == (solo_c[0], solo_c[2])
    assert collect(d, 1) == (solo_d[0], solo_d[2])
    assert sched.reserved == 0 and sched.rows == [] and sched.task is None


def test_scheduler_gives_up_and_refuses(dense):
    """Consecutive failed steps of several rows: everything running or queued
    gets Error(worker_fault), new requests are refused at once, alive() is
    False and on_fault runs (the Engine exits so launchd restarts it)."""
    faults = []
    sched = scheduler(dense, max_batch=2, on_fault=lambda: faults.append(1))
    sched._step = lambda rows: (_ for _ in ()).throw(RuntimeError("GPU gone"))
    reqs = [request(p, 20, f"r{i}") for i, p in enumerate(prompts([8] * 10, 41))]
    late = request(prompts([8], 42)[0], 5, "late")
    # (MLX is never used here while the scheduler thread runs: one thread only)
    sched.start()
    for r in reqs:
        sched.submit(r)
    ends = [collect(r, 30)[1] for r in reqs]
    assert all(e[0] == "error" for e in ends)
    assert any(e[1] == "worker_fault" for e in ends)
    # on_fault runs after _close has answered everyone and cleaned up: this
    # thread can see `closed` first (deterministic on mlx 0.32.3)
    wait_for(lambda: sched.closed is not None and faults)
    assert not sched.alive() and faults == [1]
    sched.submit(late)
    assert collect(late, 1)[1][:2] == ("error", "worker_fault")


def test_lone_row_failures_are_no_strikes(dense):
    """A step failing with one row is that request's error: a client retrying
    it any number of times does not close the scheduler."""
    faults = []
    sched = scheduler(dense, max_batch=1, on_fault=lambda: faults.append(1))
    sched._step = lambda rows: (_ for _ in ()).throw(RuntimeError("bad request"))
    reqs = [request(p, 5, f"x{i}") for i, p in enumerate(prompts([8] * 6, 52))]
    sched.start()
    for r in reqs:                                       # one after the other, like retries
        sched.submit(r)
        assert collect(r, 30)[1][:2] == ("error", "RuntimeError")
    assert sched.alive() and sched.closed is None and not faults
    sched.stop()


def test_stall_detector(dense, monkeypatch):
    """Idle time never counts as a stall (the first request after a long idle
    is served); only a tick running longer than SCHED_STALL_S does."""
    import worker as w
    monkeypatch.setattr(w, "SCHED_STALL_S", 0.3)
    sched = scheduler(dense)
    slow = {"n": 0}
    iterate = sched._iterate

    def slow_iterate():
        slow["n"] += 1
        if slow["n"] == 3:
            time.sleep(0.8)                              # one stuck tick
        iterate()
    sched._iterate = slow_iterate
    req = request(prompts([8], 53)[0], 8, "s")
    sched.start()
    time.sleep(0.6)                                      # idle for 2x the limit
    assert sched.alive()
    seen, end = [], None
    sched.submit(req)
    while end is None:
        seen.append(sched.alive())
        try:
            ev = req.out.get(timeout=0.02)
        except queue.Empty:
            continue
        end = ev if ev[0] != "tok" else None
    assert seen[0] and False in seen                     # alive at once, not during the stuck tick
    assert sched.alive() and end == ("done", "length")   # and alive again after it
    sched.stop()

def test_shutdown(dense):
    """stop(): rows and queued requests get Error(shutdown); the thread ends."""
    sched = scheduler(dense, max_batch=2)
    reqs = [request(p, 2000, f"s{i}") for i, p in enumerate(prompts([8, 8, 8], 43))]
    late = request(prompts([8], 44)[0], 5, "late")
    sched.start()
    for r in reqs:
        sched.submit(r)
    wait_for(lambda: len(sched.rows) == 2)
    sched.stop()
    assert sched.closed[0] == "shutdown" and not sched.alive()
    for r in reqs:
        assert collect(r, 1)[1][:2] == ("error", "shutdown")
    sched.submit(late)
    assert collect(late, 1)[1][:2] == ("error", "shutdown")


class BlockingEngine:
    """Engine stand-in for the gRPC layer: Generate streams until released."""

    def __init__(self, max_batch):
        self.max_batch, self.ready, self.model_id = max_batch, True, "m"
        self.load_ms, self.model_path = 0, "/m"
        self.sched = types.SimpleNamespace(alive=lambda: True)
        self.release = threading.Event()
        self.started = 0

    def generate(self, req, context=None):
        # like Engine.generate: blocked on the scheduler, it notices a dropped
        # stream (context callback / cancel) and returns
        self.started += 1
        yield pb.Event(id=req.id, start=pb.Start(role="assistant", model="m"))
        while not self.release.wait(0.02):
            if context is not None and not context.is_active():
                return
        yield pb.Event(id=req.id, done=pb.Done(finish="stop", usage=pb.Usage()))

    def check(self, req):
        return "", False

    def cancel(self, req_id):
        return False


def test_grpc_generation_slots(monkeypatch):
    """Generate beyond the slots -> RESOURCE_EXHAUSTED (the gateway's 503);
    Check / Health stay fast while every slot streams; slots come back."""
    monkeypatch.setattr(worker, "GEN_SLOTS", 3)
    eng = BlockingEngine(max_batch=2)
    server, servicer, port = worker.serve(eng, "127.0.0.1:0", threading.Event())
    try:
        stub = pb_grpc.WorkerStub(grpc.insecure_channel(f"127.0.0.1:{port}"))
        streams = [stub.Generate(pb.Request(id=f"g{i}")) for i in range(3)]
        firsts = [next(s) for s in streams]                      # all three are running
        assert all(f.HasField("start") for f in firsts)
        with pytest.raises(grpc.RpcError) as e:
            next(stub.Generate(pb.Request(id="over")))
        assert e.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        t = time.time()
        assert stub.Check(pb.Request(id="c")).error == ""
        assert stub.Health(pb.Empty()).ready
        assert time.time() - t < 1.0
        eng.release.set()
        assert all(list(s)[-1].HasField("done") for s in streams)
        wait_for(lambda: servicer.free._value == 3)              # slots released
        assert [e.done.finish for e in stub.Generate(pb.Request(id="again"))
                if e.HasField("done")] == ["stop"]
    finally:
        server.stop(0)


def test_cap_derivation():
    gib = worker.GIB
    assert worker._batch_cap(28 * gib, 0) == 28 * gib                # the device working set
    assert worker._batch_cap(28 * gib, 27) == 27 * gib               # QWEN_WIRED_LIMIT_GB
    assert worker._batch_cap(21 * gib, 27) == 21 * gib               # clamped to the kernel max


def test_budget_derivation(monkeypatch, dense):
    gib = worker.GIB
    monkeypatch.setattr(worker, "BATCH_MEM_GB", 0.0)
    monkeypatch.setattr(worker, "PROMPT_CACHE", False)
    monkeypatch.setattr(worker, "CLEAR_CACHE", True)
    assert worker._batch_budget(30 * gib, 12 * gib, 3 * gib) == 14 * gib
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)              # prompt cache: 3 GiB
    assert worker._batch_budget(30 * gib, 12 * gib, 3 * gib) == 11 * gib
    monkeypatch.setattr(worker, "CLEAR_CACHE", False)              # the pool stays: 3 GiB
    assert worker._batch_budget(30 * gib, 12 * gib, 3 * gib) == 8 * gib
    assert worker._batch_budget(20 * gib, 18 * gib, 3 * gib) == 0  # never negative
    monkeypatch.setattr(worker, "BATCH_MEM_GB", 1.5)
    assert worker._batch_budget(20 * gib, 18 * gib, 3 * gib) == int(1.5 * gib)
    # a budget that cannot hold two typical rows turns batching off
    s = scheduler(dense, max_batch=16, budget=0)
    typical = s._bytes(2048 + worker.RESERVE_TOKENS)
    assert scheduler(dense, max_batch=16, budget=2 * typical - 1).fit_max_batch() == 1
    assert scheduler(dense, max_batch=16, budget=2 * typical).fit_max_batch() == 16


def test_reservation_grows_and_blocks_admission(dense, monkeypatch):
    """Rows reserve prompt + RESERVE_TOKENS; growing past it is re-accounted
    every step, and a request that would fit next to the initial reservations
    waits once the rows have grown."""
    monkeypatch.setattr(worker, "RESERVE_TOKENS", 2)
    ps = prompts([8, 8, 8], 45)
    probe = scheduler(dense)
    short, grown = probe._bytes(8 + 2), probe._bytes(8 + 40 + 1)
    budget = 3 * short + (grown - short) // 2                      # 3 short rows fit, 2 grown + 1 not
    sched = scheduler(dense, budget=budget, prefill_reserve=4 * grown)
    a, b, c = request(ps[0], 40, "a"), request(ps[1], 40, "b"), request(ps[2], 10, "c")
    seen = []
    run(sched, [a, b, c], arrive={12: [c]}, on_tick=lambda t, s: seen.append((len(s.rows), s.reserved)))
    assert max(res for _, res in seen) >= 2 * grown               # a and b were re-accounted
    assert max(r for r, _ in seen) == 2                            # c never joined both
    assert c.t0 > 0 and sched.reserved == 0
    # without a cap a request with no max_tokens would reserve its whole context
    big = request(ps[0], 120000, "big")
    assert probe._reserve(big) == probe._bytes(8 + 2)


def test_rows_past_the_headroom_are_cut(dense, monkeypatch):
    """Growth past budget + prefill reserve ends the growing row ("length")
    while other rows run; with enough headroom nobody is cut."""
    monkeypatch.setattr(worker, "RESERVE_TOKENS", 2)
    ps = prompts([8, 8], 54)
    probe = scheduler(dense)
    short, grown = probe._bytes(8 + 2), probe._bytes(8 + 40 + 1)
    for headroom, cut in ((0, True), (4 * grown, False)):
        sched = scheduler(dense, budget=2 * short, prefill_reserve=headroom)
        a, b = request(ps[0], 40, "a"), request(ps[1], 40, "b")
        got = run(sched, [a, b])
        lens = sorted(len(got[k][0]) for k in "ab")
        assert all(got[k][2] == ("done", "length") for k in "ab")
        assert (lens[0] < 40) == cut and lens[1] == 40             # a lone row is never cut
        assert sched.reserved == 0


def test_peak_snapshot_belongs_to_the_request(dense):
    """Usage.peak is taken on the scheduler when the request ends, before the
    next admission (to an empty batch) resets the peak counter: a big request
    followed by a small one keeps its own, larger peak."""
    sched = scheduler(dense, max_batch=1)
    a, b = request(prompts([1500], 46)[0], 4, "a"), request(prompts([8], 47)[0], 4, "b")
    run(sched, [a, b])
    assert a.peak > b.peak > 0 and a.batch_mean == 1.0


def test_handler_with_scheduler_thread(dense):
    """Engine.generate against a real scheduler thread: tokens stream, a stop
    string ends the answer, and the row is evicted with its stats."""
    import test_handler as th
    prompt = tuple(prompts([10], 48)[0])
    lone = run(scheduler(dense, max_batch=1), [request(list(prompt), 12, "probe")])["probe"][0]
    sched = scheduler(dense)
    sched.start()                       # from here on MLX runs on the scheduler thread only
    eng = th.engine([], prompt=prompt)
    eng.sched = sched
    eng.vocab = 512
    text = "".join(th.text_of(t) for t in lone)
    stop = text[4:6]
    think, answer, end = th.events(eng, max_tokens=12, temperature=0.0, presence_penalty=0.0,
                                   chat_template_kwargs_json='{"enable_thinking": false}',
                                   stop=[stop])
    assert end.done.finish == "stop" and stop not in answer
    assert (think + answer).startswith(text[:text.index(stop)])
    assert end.done.usage.peak_gb > 0
    wait_for(lambda: not sched.rows and sched.reserved == 0)
    try:
        _, answer, end = th.events(eng, max_tokens=12, temperature=0.0, presence_penalty=0.0)
        assert end.done.finish == "length" and end.done.usage.gen_tokens == 12
    finally:
        sched.stop()


WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "weights", "ornith10-9b-mlx-8bit")


@pytest.mark.skipif(not os.path.isdir(WEIGHTS), reason="needs the ornith10 processor files")
def test_vision_inputs_equal_prepare_inputs(tmp_path):
    """The handler-side image path (padding off) + the scheduler's conversion
    give the arrays (values, dtypes, shapes) mlx-vlm's prepare_inputs gives
    (processor files only, no weights)."""
    from pathlib import Path

    import mlx_vlm.models.qwen3_5.qwen3_5  # noqa: F401 — registers the processor
    from PIL import Image
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config, load_image_processor, load_processor, prepare_inputs
    path = Path(WEIGHTS)
    proc = load_processor(path, True)
    ip = load_image_processor(path)
    if ip is not None:
        proc.image_processor = ip
    img = tmp_path / "x.png"
    Image.fromarray((np.random.default_rng(0).random((200, 300, 3)) * 255).astype("uint8")).save(img)
    fmt = apply_chat_template(proc, load_config(path), [{"role": "user", "content": "Describe."}],
                              add_generation_prompt=True, num_images=1)
    ref = prepare_inputs(proc, images=[str(img)], prompts=fmt, image_token_index=None,
                         add_special_tokens=True, padding=False)
    eng = worker.Engine(str(path))
    eng.processor = proc
    got = eng._vision_inputs(fmt, [str(img)])
    assert set(got) == set(ref)
    assert all(isinstance(v, np.ndarray) for v in got.values())
    arrays = worker._vision_arrays(got)                     # what _prefill feeds the model
    for k, v in ref.items():
        assert arrays[k].dtype == v.dtype and arrays[k].shape == v.shape, k
        assert mx.array_equal(arrays[k], v).item(), k


def test_handler_survives_a_dead_scheduler(monkeypatch):
    """A stream waiting on a scheduler that died ends with Error(worker_fault)."""
    import test_handler as th
    monkeypatch.setattr(worker, "EVENT_POLL_S", 0.02)
    eng = th.engine(th.toks("ab"), prompt=(100,))        # two tokens, then silence
    eng.sched.alive = lambda: False
    _, answer, end = th.events(eng)
    assert answer == "ab" and end.error.kind == "worker_fault"


def test_failed_join_leaves_the_batch_intact(dense, monkeypatch):
    """A join that fails while stacking the recurrent state fails the new row
    only; the running rows keep their state (their answers equal solo)."""
    from test_scheduler import mixed, serial
    make = mixed(51)
    ref = serial(dense, make)
    reqs = make()
    concat, calls = worker._gdn_concat, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("injected stacking failure")
        return concat(a, b)
    monkeypatch.setattr(worker, "_gdn_concat", flaky)
    got = run(scheduler(dense), reqs, arrive={6: reqs[2:]})
    failed = [k for k, v in got.items() if v[2][0] == "error"]
    assert len(failed) == 1 and "injected" in got[failed[0]][2][2]
    for k in ref:
        if k not in failed:
            assert got[k] == ref[k], k


def test_grpc_slot_freed_by_a_client_cancel(monkeypatch):
    """A client that drops its stream mid-generation gives its slot back."""
    monkeypatch.setattr(worker, "GEN_SLOTS", 2)
    eng = BlockingEngine(max_batch=1)
    server, servicer, port = worker.serve(eng, "127.0.0.1:0", threading.Event())
    try:
        stub = pb_grpc.WorkerStub(grpc.insecure_channel(f"127.0.0.1:{port}"))
        a, b = stub.Generate(pb.Request(id="a")), stub.Generate(pb.Request(id="b"))
        next(a), next(b)
        assert servicer.free._value == 0
        a.cancel()                                        # the client goes away
        wait_for(lambda: servicer.free._value == 1)
        c = stub.Generate(pb.Request(id="c"))
        assert next(c).HasField("start")                  # the freed slot is usable
        eng.release.set()
        list(b), list(c)
        wait_for(lambda: servicer.free._value == 2)
    finally:
        server.stop(0)


def test_failed_first_join_keeps_the_running_row(dense, monkeypatch):
    """A 1->2 join that fails half-way through stacking (second linear layer)
    leaves the running row's recurrent state intact: its answer equals solo."""
    ps = prompts([10, 12], 55)
    solo = run(scheduler(dense, max_batch=1), [request(ps[0], 20, "a")])["a"]
    stack, armed = worker._gdn_stack, {"on": False, "n": 0}
    concat = worker.mx.concatenate

    def flaky_concat(xs, axis=0):
        if armed["on"]:
            armed["n"] += 1
            if armed["n"] == 3:                           # layer 2's conv state
                raise ValueError("injected concatenate failure")
        return concat(xs, axis=axis)

    def armed_stack(lm, row_caches, dtype):
        armed["on"] = len(row_caches) == 2
        try:
            return stack(lm, row_caches, dtype)
        finally:
            armed["on"] = False
    monkeypatch.setattr(worker.mx, "concatenate", flaky_concat)
    monkeypatch.setattr(worker, "_gdn_stack", armed_stack)
    a, b = request(ps[0], 20, "a"), request(ps[1], 20, "b")
    got = run(scheduler(dense), [a, b], arrive={3: [b]})
    assert armed["n"] >= 3
    assert got["b"][2][0] == "error" and "injected" in got["b"][2][2]
    assert got["a"] == solo


def test_cancelled_request_is_never_queued():
    """A request cancelled while it is being prepared (before the image work
    / before submit) ends as cancelled without reaching the scheduler."""
    import test_handler as th
    eng = th.engine(th.toks("ab") + [("done", "length")], prompt=(100,))
    render = eng._render

    def render_then_cancel(req, images):
        eng.cancel(req.id)                                # e.g. the client went away meanwhile
        return render(req, images)
    eng._render = render_then_cancel
    _, answer, end = th.events(eng)
    assert end.done.finish == "cancelled" and answer == "" and eng.sched.req is None

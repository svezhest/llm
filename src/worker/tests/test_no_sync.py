"""QWEN_DEBUG_NO_SYNC: no host wait while a decode step is built (the
step-ahead: _Scheduler._step queues step k+1 before step k's tokens are
read). The scheduler's paths run with the guard on and must answer as with
it off; an injected wait must fail the row with SyncInStep.

(The whole CPU suite passes with QWEN_DEBUG_NO_SYNC=1 except the tests whose
spies read logits inside the step on purpose, e.g. test_cascade_sched's.)"""
import mlx.core as mx
import pytest

import worker
from conftest import prompts, request, run, scheduler
from test_scheduler import AllowAll, mixed


@pytest.fixture
def nosync(monkeypatch):
    monkeypatch.setattr(worker, "DEBUG_NO_SYNC", True)


@pytest.fixture(params=["dense", "dense_fused"])
def lm(request):
    return request.getfixturevalue(request.param)


def both(lm, make, monkeypatch, **kw):
    with monkeypatch.context() as m:
        m.setattr(worker, "DEBUG_NO_SYNC", False)
        ref = run(scheduler(lm, **kw), make())
    with monkeypatch.context() as m:
        m.setattr(worker, "DEBUG_NO_SYNC", True)
        got = run(scheduler(lm, **kw), make())
    return ref, got


@pytest.mark.parametrize("max_batch", [1, 4, 8])
def test_steps_build_without_a_host_wait(lm, monkeypatch, max_batch):
    """B = 1 (the library call), B >= 3 (stacked GDN, mrqmm's place on the
    CPU, grouped sampling), seeded sampling, penalties: same answers."""
    ref, got = both(lm, mixed(71), monkeypatch, max_batch=max_batch)
    assert got == ref
    assert all(ev[0] == "done" for _, _, ev in got.values())


def test_logprobs_sync_mode_and_prompt_cache(dense, monkeypatch):
    monkeypatch.setattr(worker, "PROMPT_CACHE", True)
    ps = prompts([12, 9, 15, 7], 72)

    def make():
        return [request(ps[0], 10, "lp", logprobs=True, top_logprobs=3),
                request(ps[1], 10, "lp-pen", logprobs=True, presence_penalty=0.5, temperature=0.7,
                        seed=1),
                request(ps[2], 10, "guided", guide=AllowAll()),         # the batch in sync mode
                request(ps[3], 10, "bad", bad_words=["ab"], min_tokens=4)]
    ref, got = both(dense, make, monkeypatch, max_batch=4)
    assert got == ref
    assert all(ev[0] == "done" for _, _, ev in got.values())


@pytest.mark.parametrize("where", ["tolist", "item", "eval", "np"])
def test_a_wait_in_the_step_fails_the_row(dense, nosync, monkeypatch, where):
    """The old regression, re-injected: a read of the step's output while it
    is being built."""
    orig = worker._token_logprobs

    def bad(x, y, k, keep=False):
        out = orig(x, y, k, keep)
        if where == "tolist":
            y.tolist()
        elif where == "item":
            y.item()
        elif where == "eval":
            mx.eval(y)
        else:
            worker.np.asarray(y)
        return out
    monkeypatch.setattr(worker, "_token_logprobs", bad)
    with pytest.raises(worker.SyncInStep, match="QWEN_DEBUG_NO_SYNC"):
        # outside the step the wrappers pass through; inside they raise
        with worker._no_sync():
            mx.array([1]).item()
    # the step raises (the scheduler thread's loop fails the step's rows: _fail_rows)
    with pytest.raises(worker.SyncInStep, match=r"_Scheduler\._step"):
        run(scheduler(dense, max_batch=2), [request(prompts([8], 73)[0], 5, "x", logprobs=True),
                                            request(prompts([8], 74)[0], 5, "y")])
    assert not getattr(worker._NO_SYNC, "on", False)
    assert mx.array([2]).item() == 2 and worker.np.asarray(mx.array([3]))[0] == 3


def test_off_installs_nothing(monkeypatch):
    monkeypatch.setattr(worker, "DEBUG_NO_SYNC", False)
    assert worker._no_sync() is worker._NULL_CTX

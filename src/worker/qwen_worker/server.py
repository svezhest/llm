"""gRPC servicer, serve() and main()."""

import argparse
import logging
import os
import signal
import threading
from concurrent import futures

import grpc

import worker_pb2 as pb
import worker_pb2_grpc as pb_grpc

from .config import DEFAULT_MODEL, GEN_SLOTS, log
from .engine import Engine


class WorkerServicer(pb_grpc.WorkerServicer):
    def __init__(self, engine: Engine, stop_event: threading.Event, slots: int):
        self.engine = engine
        self.stop_event = stop_event
        self.slots = slots
        self.free = threading.BoundedSemaphore(slots)

    def Generate(self, request, context):
        p = request.params
        if not self.free.acquire(blocking=False):
            log.warning("Generate id=%s refused: %d generations in flight", request.id, self.slots)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                          f"worker busy: {self.slots} generations in flight")
        try:
            log.info("Generate id=%s images=%d max_tokens=%s template_kwargs=%s budget=%s",
                     request.id, len(request.images),
                     p.max_tokens if p.HasField("max_tokens") else "-",
                     p.chat_template_kwargs_json or "-",
                     p.thinking_budget if p.HasField("thinking_budget") else "-")
            yield from self.engine.generate(request, context)
        finally:
            self.free.release()     # also when grpc closes the stream (client gone)

    def Cancel(self, request, context):
        self.engine.cancel(request.id)
        return pb.Ack()

    def Health(self, request, context):
        pc = self.engine._pcache        # counters only (plain ints the scheduler keeps current)
        sched = self.engine.sched
        return pb.HealthReply(ready=self.engine.ready and sched.alive(),
                              model=self.engine.model_id,
                              load_ms=self.engine.load_ms,
                              model_path=self.engine.model_path,
                              max_batch=self.engine.max_batch,
                              prompt_cache_entries=len(pc.entries),
                              prompt_cache_bytes=pc.bytes,
                              prompt_cache_hits=pc.hits,
                              prompt_cache_misses=pc.misses,
                              prompt_cache_evictions=pc.evictions,
                              cascade_groups=getattr(sched, "n_groups", 0),
                              cascade_rows=getattr(sched, "n_grouped", 0))

    def Check(self, request, context):
        # Runs beside the generations (own pool thread): template, tokenizer and
        # grammar checks, nothing that touches MLX.
        error, internal = self.engine.check(request)
        return pb.CheckReply(error=error, internal=internal, model=self.engine.model_id)

    def Shutdown(self, request, context):
        log.warning("shutdown requested")
        self.stop_event.set()
        return pb.Ack()


def serve(engine, addr, stop_event):
    """gRPC server for a loaded engine -> (server, servicer, port). Generate RPCs
    beyond QWEN_GEN_SLOTS (default max(2*max_batch, 8)) are refused with
    RESOURCE_EXHAUSTED; the pool has 8 threads more for Check / Cancel /
    Health. grpc itself refuses any RPC past slots + 64 queued (a backstop:
    only a burst of > 64 Checks at once could then refuse a Health poll).
    Vision prompts carry base64 image data URIs: a detailed map snapshot is
    several MB, past gRPC's 4 MB default receive cap, so the message limits
    are lifted."""
    slots = max(1, GEN_SLOTS or max(2 * engine.max_batch, 8))
    if slots < engine.max_batch:
        log.warning("QWEN_GEN_SLOTS %d < max_batch %d: the batch can never fill", slots,
                    engine.max_batch)
    max_msg = 64 * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=slots + 8),
        maximum_concurrent_rpcs=slots + 64,
        options=[
            ("grpc.max_receive_message_length", max_msg),
            ("grpc.max_send_message_length", max_msg),
        ],
    )
    servicer = WorkerServicer(engine, stop_event, slots)
    pb_grpc.add_WorkerServicer_to_server(servicer, server)
    port = server.add_insecure_port(addr)
    server.start()
    log.info("worker serving on %s (%d generation slots, %d threads)", addr, slots, slots + 8)
    return server, servicer, port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:50051")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )

    engine = Engine(args.model)
    engine.load()

    stop_event = threading.Event()
    # plain `kill` (launchd, README) stops like Shutdown / Ctrl-C
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    server, _, _ = serve(engine, args.addr, stop_event)
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        # rows and queued requests get Error(shutdown) before the RPCs close
        if not engine.sched.stop(timeout=10.0):
            # a tick still inside MLX: interpreter teardown under it can crash
            log.error("scheduler did not stop within 10 s; exiting now")
            server.stop(0)
            os._exit(1)
        disk = engine._pcache.disk
        if disk is not None and not disk.close(timeout=10.0):
            # a file write over an MLX buffer / an mx.array of a load still
            # running: interpreter teardown under it can crash, as above
            log.error("prompt cache disk: writer / loader still busy after 10 s; exiting now")
            server.stop(0)
            os._exit(1)
        server.stop(grace=2).wait()
        log.info("worker stopped")

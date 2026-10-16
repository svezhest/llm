import grpc, sys, worker_pb2 as pb, worker_pb2_grpc as g
s = g.WorkerStub(grpc.insecure_channel('127.0.0.1:50051'))
h = s.Health(pb.Empty(), timeout=10)
print(" ".join(f"{f.name}={v}" for f, v in h.ListFields()))

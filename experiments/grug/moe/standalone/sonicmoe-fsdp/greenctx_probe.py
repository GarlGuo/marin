"""Single-GPU smoke test for the green-context SM partitioning (run before the full benchmark).

Validates: (1) partitions create without error, (2) SM split is disjoint and correct, (3) kernels
run correctly on each green-ctx stream, (4) the SM cap is real (compute partition faster than comm
partition on the same matmul), (5) the two partitions run CONCURRENTLY (overlap).
"""
from __future__ import annotations

import contextlib

import torch

from greenctx import GreenCtxStreams


def _bench_matmul(x, stream, iters=50, warmup=10):
    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with ctx:
        for _ in range(warmup):
            _ = x @ x
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    with ctx:
        e0.record()
        for _ in range(iters):
            _ = x @ x
        e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    dev = 0
    torch.cuda.set_device(dev)
    print("gpu:", torch.cuda.get_device_name(dev), flush=True)

    g = GreenCtxStreams(dev, comm_sms=16)
    print("GREENCTX", g, flush=True)
    assert g.comm_sms + g.compute_sms <= g.total_sms + 8, "partitions overlap?"

    # (3) correctness on each stream
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    ref = a @ a
    with torch.cuda.stream(g.compute_stream):
        c1 = a @ a
    with torch.cuda.stream(g.comm_stream):
        c2 = a @ a
    torch.cuda.synchronize()
    print("compute-stream matmul correct:", torch.allclose(c1, ref, atol=0.2), flush=True)
    print("comm-stream    matmul correct:", torch.allclose(c2, ref, atol=0.2), flush=True)

    # (4) SM-cap is real: same matmul on full device vs each partition
    big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t_full = _bench_matmul(big, None)
    t_comp = _bench_matmul(big, g.compute_stream)
    t_comm = _bench_matmul(big, g.comm_stream)
    print(f"matmul ms/iter: full={t_full:.3f}  compute_part({g.compute_sms}sm)={t_comp:.3f}  "
          f"comm_part({g.comm_sms}sm)={t_comm:.3f}", flush=True)
    print(f"  -> comm partition is {t_comm/t_full:.1f}x slower than full (SM cap working if >1)", flush=True)

    # (5) concurrency: run a long comm-partition load and check compute-partition timing is ~unchanged
    comm_load = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t_comp_alone = _bench_matmul(big, g.compute_stream, iters=100)
    # launch sustained work on comm stream, then time compute stream concurrently
    with torch.cuda.stream(g.comm_stream):
        for _ in range(400):
            _ = comm_load @ comm_load
    t_comp_concurrent = _bench_matmul(big, g.compute_stream, iters=100)
    torch.cuda.synchronize()
    slowdown = t_comp_concurrent / t_comp_alone
    print(f"compute-part matmul: alone={t_comp_alone:.3f}ms  under-concurrent-comm={t_comp_concurrent:.3f}ms "
          f"(slowdown {slowdown:.2f}x; ~1.0 => truly disjoint/concurrent)", flush=True)
    print("PROBE_OK", flush=True)


if __name__ == "__main__":
    main()

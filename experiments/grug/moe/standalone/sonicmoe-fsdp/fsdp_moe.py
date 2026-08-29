"""Manual-FSDP sonicmoe MoE stack: measure NCCL all-gather bandwidth and comm/compute overlap.

Lever 1 (bandwidth): --bucket flat gathers a whole layer's params in ONE large all-gather (vs
per_param = 3).
Lever 2 (overlap): --arm overlap_greenctx runs the MoE compute on a green-context compute partition
while the NEXT layer's all-gather runs on a disjoint green-context comm partition (DeepEP schedule).

Forward-only for now (all-gather path); backward (reduce-scatter grads) is a follow-up. Run via
run.slurm with torchrun --nproc_per_node=4.
"""
from __future__ import annotations

import argparse
import ctypes
import os

import torch
import torch.distributed as dist

from greenctx import GreenCtxStreams

BF16 = torch.bfloat16


def _load_nccl():
    for cand in (
        "libnccl.so.2",
        os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "nccl", "lib", "libnccl.so.2"),
    ):
        try:
            return ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    raise OSError("libnccl.so.2 not found")


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


class RawNCCL:
    """Raw ctypes NCCL comm so collectives run on an EXPLICIT stream.

    PyTorch's ProcessGroupNCCL runs collectives on its own internal stream, so a PG all-gather
    inside ``with torch.cuda.stream(green_comm)`` is NOT confined to the green partition.
    ``all_gather`` here calls ncclAllGather directly on the given stream, so a green-context comm
    stream really caps NCCL's SMs. (torch.cuda.nccl.init_rank is broken in torch 2.9.1 —
    SystemError PY_SSIZE_T_CLEAN — hence ctypes.) NCCL env is read at ncclCommInitRank.
    """

    _NCCL_BF16 = 9  # ncclDataType_t: ncclBfloat16

    def __init__(self, world, rank):
        self._n = _load_nccl()
        self._n.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _NcclUniqueId, ctypes.c_int]
        self._n.ncclAllGather.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p]
        uid = _NcclUniqueId()
        if rank == 0:
            assert self._n.ncclGetUniqueId(ctypes.byref(uid)) == 0
        raw = ctypes.string_at(ctypes.addressof(uid), 128) if rank == 0 else b"\x00" * 128
        obj = [raw]
        dist.broadcast_object_list(obj, src=0)  # share the raw 128-byte id
        ctypes.memmove(ctypes.addressof(uid), obj[0], 128)
        self.comm = ctypes.c_void_p()
        rc = self._n.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank)
        assert rc == 0, f"ncclCommInitRank rc={rc}"

    def all_gather(self, shard, full, stream):
        rc = self._n.ncclAllGather(
            ctypes.c_void_p(shard.data_ptr()), ctypes.c_void_p(full.data_ptr()),
            shard.numel(), self._NCCL_BF16, self.comm, ctypes.c_void_p(stream.cuda_stream))
        assert rc == 0, f"ncclAllGather rc={rc}"


# ---------------------------------------------------------------- symm-mem all-gather stack ------
class SymmMemStack:
    """FSDP shards mirrored into symmetric-memory buffers; all-gather via sonicmoe's Triton NVLink
    AG (unicast P2P reads). The kernel runs on the current stream, so on a green comm stream it is
    SM-confined -> the low-SM high-BW comm that makes green-ctx overlap win. Params are fixed here,
    so one barrier after writing the symm shards suffices (real FSDP would barrier per step)."""

    def __init__(self, shards, group, dev):
        import torch.distributed._symmetric_memory as symm_mem
        from sonicmoe.functional.distributed.collectives import all_gather_triton, rendezvous
        self._ag = all_gather_triton
        self.group = group
        self.symm_shards, self.peer_bufs = [], []
        for s in shards:
            buf = symm_mem.empty(s.numel(), dtype=BF16, device=dev)
            buf.copy_(s)
            hdl = rendezvous(buf, group)
            W = hdl.world_size
            self.symm_shards.append(buf)
            self.peer_bufs.append(tuple(hdl.get_buffer(r, (s.numel(),), BF16) for r in range(W)))
        dist.barrier()  # all peers' symm buffers written + P2P-mapped before any AG reads them

    def all_gather(self, u, full, stream):
        with torch.cuda.stream(stream):
            self._ag(self.symm_shards[u], self.group, out=full, peer_bufs=self.peer_bufs[u])


# ---------------------------------------------------------------- sharded MoE stack --------------
class ShardedMoEStack:
    """MoE layers grouped into buckets of ``bucket_layers``; each bucket's params flattened to one
    buffer, sharded [P/W] over the data PG. One all-gather per bucket -> larger messages amortize
    NCCL per-message overhead (per-layer 589 -> ~640 GB/s at bucket=all). ``units`` = #all-gathers."""

    def __init__(self, n_layers, H, I, E, K, world, rank, dev, bucket_layers=1):
        assert n_layers % bucket_layers == 0, "n_layers must be divisible by bucket_layers"
        self.N, self.H, self.I, self.E, self.K = n_layers, H, I, E, K
        self.G = bucket_layers
        self.units = n_layers // bucket_layers
        self.world, self.rank, self.dev = world, rank, dev
        self.layer_numels = [E * H, E * 2 * I * H, E * H * I]  # router, c_fc, c_proj (per layer)
        self.layer_P = sum(self.layer_numels)
        self.P = self.layer_P * self.G  # params per bucket
        self.Ppad = ((self.P + world - 1) // world) * world
        self.shard_numel = self.Ppad // world
        self.shards = []  # one per bucket, this rank's slice; random init (perf-representative)
        for _ in range(self.units):
            full = torch.zeros(self.Ppad, dtype=BF16, device=dev)
            full[: self.P].normal_(0, 0.02)
            self.shards.append(full.view(world, self.shard_numel)[rank].clone())
        self.ag_bytes_per_unit = self.Ppad * 2 * (world - 1) / world  # busbw wire bytes / rank

    def views(self, full, g):
        """the g-th layer's (router_w, w1, w2) within a gathered bucket."""
        base = g * self.layer_P
        n0, n1, n2 = self.layer_numels
        E, H, I = self.E, self.H, self.I
        router_w = full[base : base + n0].view(E, H)
        w1 = full[base + n0 : base + n0 + n1].view(E, 2 * I, H).permute(1, 2, 0)  # [2I,H,E]
        w2 = full[base + n0 + n1 : base + n0 + n1 + n2].view(E, H, I).permute(1, 2, 0)  # [H,I,E]
        return router_w, w1, w2

    def compute(self, x, full):
        from sonicmoe.enums import ActivationType
        from sonicmoe.functional import moe_TC_softmax_topk_layer

        for g in range(self.G):  # compute all G layers packed in this gathered bucket
            router_w, w1, w2 = self.views(full, g)
            x, _, _ = moe_TC_softmax_topk_layer(
                x, router_w, w1, None, w2, None, self.K, None,
                ActivationType.SWIGLU, is_inference_mode_enabled=True,
            )
        return x


# ---------------------------------------------------------------- forward pipelines --------------
def _new_full(stack):
    return torch.empty(stack.Ppad, dtype=BF16, device=stack.dev)


def forward_serial(stack, x, compute_stream, gather):
    with torch.cuda.stream(compute_stream):
        for u in range(stack.units):
            full = _new_full(stack)
            gather(u, full, compute_stream)
            x = stack.compute(x, full)
    return x


def forward_overlap(stack, x, compute_stream, comm_stream, gather):
    N = stack.units
    fulls, evts = [None] * N, [None] * N

    def issue_ag(L):
        with torch.cuda.stream(comm_stream):
            full = _new_full(stack)  # created on comm_stream -> allocator reuse-fence honors it
            gather(L, full, comm_stream)
            e = torch.cuda.Event()
            e.record(comm_stream)
        fulls[L], evts[L] = full, e

    # make comm/compute streams see the persistent shards + input
    start = torch.cuda.Event()
    start.record(torch.cuda.current_stream())
    comm_stream.wait_event(start)
    compute_stream.wait_event(start)

    issue_ag(0)
    with torch.cuda.stream(compute_stream):
        for L in range(N):
            if L + 1 < N:
                issue_ag(L + 1)  # prefetch next layer's all-gather on the comm partition
            compute_stream.wait_event(evts[L])  # wait for THIS layer's gather
            fulls[L].record_stream(compute_stream)
            x = stack.compute(x, fulls[L])
            fulls[L] = None
    return x


# ---------------------------------------------------------------- timing -------------------------
def bench(fn, compute_stream, warmup, iters, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    samples = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(device)
        dist.barrier()
        e0.record(compute_stream)
        fn()
        e1.record(compute_stream)
        e1.synchronize()
        t = torch.tensor([e0.elapsed_time(e1)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank defines the sample
        samples.append(t.item())
    samples.sort()
    return samples[len(samples) // 2]


def measure_ag_busbw(stack, comm_stream, iters, device, gather):
    """Isolated all-gather bandwidth: N gathers timed, busbw = wire_bytes/time.

    Uses a fixed 2-buffer pool (no per-iter alloc overhead) and the given gather backend.
    """
    pool = [_new_full(stack), _new_full(stack)]

    def ag_only():
        with torch.cuda.stream(comm_stream):
            for u in range(stack.units):
                gather(u, pool[u % 2], comm_stream)
    t_ms = bench(ag_only, comm_stream, warmup=3, iters=iters, device=device)
    total_wire = stack.ag_bytes_per_unit * stack.units
    return t_ms, total_wire / (t_ms / 1e3) / 1e9  # GB/s


# ---------------------------------------------------------------- main ---------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--tokens", type=int, default=8192)  # per rank
    p.add_argument("--hidden", type=int, default=2560)
    p.add_argument("--inter", type=int, default=1280)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--bucket-layers", type=int, default=1)  # layers per all-gather bucket (>1 = coarser)
    p.add_argument("--bucket-sweep", default=None)  # comma list, e.g. 1,2,4,8,16 -> BW-only sweep
    p.add_argument("--symm", action="store_true")  # also test sonicmoe symm-mem NVLink all-gather
    p.add_argument("--arms", default="serial,overlap_plain,overlap_greenctx")
    p.add_argument("--comm-sms", default="16")  # comma list, e.g. 8,16,32,48
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    return p.parse_args()


def main():
    a = parse()
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    is0 = rank == 0

    stack = ShardedMoEStack(a.layers, a.hidden, a.inter, a.experts, a.topk, world, rank, dev,
                            bucket_layers=a.bucket_layers)
    x0 = torch.randn(a.tokens, a.hidden, device=dev, dtype=BF16) * 0.2
    if is0:
        print(f"[cfg] world={world} layers={a.layers} bucket_layers={stack.G} units={stack.units} "
              f"tokens/rank={a.tokens} H={a.hidden} I={a.inter} "
              f"E={a.experts} K={a.topk} P/bucket={stack.P/1e6:.0f}M shard={stack.shard_numel*2/1e6:.0f}MB "
              f"AG_wire/bucket={stack.ag_bytes_per_unit/1e6:.0f}MB", flush=True)

    raw = RawNCCL(world, rank)  # raw comm so collectives run on an explicit (green) stream

    def pg_gather(L, out, stream):
        dist.all_gather_into_tensor(out, stack.shards[L])  # PG: internal stream, full SMs

    def raw_gather(L, out, stream):
        raw.all_gather(stack.shards[L], out, stream)  # runs on `stream` -> green ctx caps SMs

    symm_stack = SymmMemStack(stack.shards, dist.group.WORLD, dev) if a.symm else None

    def symm_gather(u, out, stream):
        symm_stack.all_gather(u, out, stream)  # sonicmoe Triton NVLink AG on `stream`

    default_stream = torch.cuda.current_stream(dev)
    plain_comm = torch.cuda.Stream(dev, priority=-1)
    comm_sms_list = [int(x) for x in str(a.comm_sms).split(",")]

    # warm up QuACK autotune/compile once (untimed)
    _ = forward_serial(stack, x0, default_stream, pg_gather)
    torch.cuda.synchronize(dev)

    # AG bandwidth: PG (unconfined, full SMs) — 16 separate per-layer gathers
    pg_ms, pg_gbs = measure_ag_busbw(stack, plain_comm, a.iters, dev, pg_gather)
    if is0:
        print(f"[bw] all-gather PG per-bucket(x{stack.units}, {stack.G}L/bucket): {pg_ms:.2f} ms  busbw={pg_gbs:.0f} GB/s ({pg_gbs/900*100:.0f}%)", flush=True)
    if symm_stack is not None:  # sonicmoe Triton NVLink AG, full SMs -> the symm-mem BW ceiling
        sm_ms, sm_gbs = measure_ag_busbw(stack, plain_comm, a.iters, dev, symm_gather)
        if is0:
            print(f"[bw] all-gather SYMM(sonicmoe full-SM): {sm_ms:.2f} ms  busbw={sm_gbs:.0f} GB/s ({sm_gbs/900*100:.0f}%)", flush=True)
    # single GIANT all-gather (all layers in one message) -> per-message-overhead-free ceiling
    giant_shard = torch.empty(stack.units * stack.shard_numel, dtype=BF16, device=dev)
    giant_full = torch.empty(stack.units * stack.Ppad, dtype=BF16, device=dev)

    def giant_ag():
        with torch.cuda.stream(plain_comm):
            dist.all_gather_into_tensor(giant_full, giant_shard)

    gm = bench(giant_ag, plain_comm, 3, a.iters, dev)
    gbytes = giant_full.numel() * 2 * (stack.world - 1) / stack.world  # actual output size
    if is0:
        print(f"[bw] all-gather GIANT(1 msg {gbytes/1e9:.0f}GB): {gm:.2f} ms  busbw={gbytes/(gm/1e3)/1e9:.0f} GB/s "
              f"({gbytes/(gm/1e3)/1e9/900*100:.0f}%)", flush=True)
    del giant_shard, giant_full
    torch.cuda.empty_cache()

    # bucket-size BW sweep: coarser buckets (more layers/AG) -> larger messages -> higher busbw
    if a.bucket_sweep is not None:
        for g in [int(x) for x in a.bucket_sweep.split(",")]:
            if a.layers % g != 0:
                continue
            s = ShardedMoEStack(a.layers, a.hidden, a.inter, a.experts, a.topk, world, rank, dev,
                                bucket_layers=g)

            def _g(u, out, st, ss=s):
                dist.all_gather_into_tensor(out, ss.shards[u])

            ms, gbs = measure_ag_busbw(s, plain_comm, a.iters, dev, _g)
            if is0:
                print(f"[bw] bucket={g:2d}L units={s.units:2d} msg={s.Ppad * 2 / 1e6:.0f}MB: "
                      f"{ms:.2f} ms  busbw={gbs:.0f} GB/s ({gbs / 900 * 100:.0f}%)", flush=True)
            del s
            torch.cuda.empty_cache()

    results = {}
    if "serial" in a.arms:
        results["serial"] = bench(lambda: forward_serial(stack, x0, default_stream, pg_gather),
                                  default_stream, a.warmup, a.iters, dev)
    if "overlap_plain" in a.arms:
        results["overlap_plain"] = bench(lambda: forward_overlap(stack, x0, default_stream, plain_comm, pg_gather),
                                         default_stream, a.warmup, a.iters, dev)
    if "overlap_greenctx" in a.arms:
        for csms in comm_sms_list:
            g = GreenCtxStreams(local, comm_sms=csms)
            raw_ms, raw_gbs = measure_ag_busbw(stack, g.comm_stream, a.iters, dev, raw_gather)
            t = bench(lambda gg=g: forward_overlap(stack, x0, gg.compute_stream, gg.comm_stream, raw_gather),
                      g.compute_stream, a.warmup, a.iters, dev)
            results[f"greenctx_c{g.comm_sms}"] = t
            if is0:
                print(f"[bw] raw(green comm={g.comm_sms}SM compute={g.compute_sms}SM): {raw_ms:.2f} ms "
                      f"busbw={raw_gbs:.0f} GB/s ({raw_gbs/900*100:.0f}%)", flush=True)
            if symm_stack is not None:  # sonicmoe AG confined to the green comm partition
                s_ms, s_gbs = measure_ag_busbw(stack, g.comm_stream, a.iters, dev, symm_gather)
                ts = bench(lambda gg=g: forward_overlap(stack, x0, gg.compute_stream, gg.comm_stream, symm_gather),
                           g.compute_stream, a.warmup, a.iters, dev)
                results[f"greenctx_symm_c{g.comm_sms}"] = ts
                if is0:
                    print(f"[bw] symm(green comm={g.comm_sms}SM): {s_ms:.2f} ms busbw={s_gbs:.0f} GB/s ({s_gbs/900*100:.0f}%)", flush=True)
            g.close()  # release the SM partition before the next sweep point re-splits the device

    if is0:
        base = results.get("serial")
        print("[summary]", flush=True)
        for arm, t in results.items():
            sp = f"  speedup={base/t:.3f}x" if base else ""
            print(f"[arm] {arm:20s} {t:7.2f} ms{sp}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

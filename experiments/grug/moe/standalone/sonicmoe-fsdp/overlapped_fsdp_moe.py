"""Best implementation: FSDP-sharded sonicmoe MoE stack with green-context comm/compute overlap.

The winning configuration (see RESULTS.md): on 4xGB200 it reaches ~1180 TFLOP/s/GPU (52-53% of B200
bf16 peak), ~1.5x the exposed-comm baseline (785), by hiding the FSDP all-gather behind MoE compute on
a disjoint SM partition. Optimal split: comm=16 SMs / compute=136 SMs, one all-gather per layer.

Two ideas, composed:
  1. GREEN CONTEXTS — two DISJOINT SM partitions (compute: most SMs, comm: few) via the raw CUDA driver
     (greenctx.py); anything launched on a green-context stream is confined to that SM set. The overlap
     is compute-bound, so give comm the MINIMUM SMs that keep it hidden (16) and compute the rest (136,
     only +3-5% tax vs the full device).
  2. SYMM-MEM ALL-GATHER — sonicmoe's Triton NVLink P2P all-gather over torch symmetric memory
     (`all_gather_triton`). Unlike NCCL Ring (which needs ~32 SMs for full BW and collapses below), it
     holds near-peak BW on FEW SMs (668 GB/s full / 661 @32 / 473 @16) — which is what lets the comm
     partition be tiny. The next layer's gather runs on the comm partition while the current layer's MoE
     compute runs on the compute partition; a prefetch-by-one CUDA-event schedule keeps them concurrent.

`bucket_layers` groups G layers into one flat gather; coarsening beyond 1 does NOT help the overlap
(RESULTS.md), so the default is 1. Forward (inference) only — the validated path; backward (grad
reduce-scatter || dgrad) is the symmetric extension (RESULTS.md TODO).

Run:  sbatch run.slurm overlapped_fsdp_moe.py 4    # default demo (comm=16, 1 layer/gather): verify + time
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from greenctx import GreenCtxStreams

BF16 = torch.bfloat16


class OverlappedFSDPMoE:
    """A stack of sonicmoe MoE layers, FSDP-sharded, with green-context overlapped symm-mem all-gather.

    `forward(x)` is the overlapped path; `forward_reference(x)` is a plain non-overlapped path
    (NCCL all-gather, single stream) with identical math, for correctness checking.
    """

    def __init__(self, n_layers, hidden, inter, experts, topk, bucket_layers,
                 world, rank, group, device, comm_sms=32, seed=0):
        assert n_layers % bucket_layers == 0, "n_layers must be divisible by bucket_layers"
        self.H, self.I, self.E, self.K = hidden, inter, experts, topk
        self.G = bucket_layers
        self.units = n_layers // bucket_layers
        self.world, self.rank, self.group, self.dev = world, rank, group, device

        # --- FSDP sharding: one flat bucket per `G` layers ---
        self.layer_numels = [experts * hidden,                      # router      [E,H]
                             experts * 2 * inter * hidden,          # c_fc (w13)  [E,2I,H]
                             experts * hidden * inter]              # c_proj (w2) [E,H,I]
        self.layer_P = sum(self.layer_numels)
        self.P = self.layer_P * self.G                              # params per bucket
        self.Ppad = ((self.P + world - 1) // world) * world
        self.shard_numel = self.Ppad // world

        gen = torch.Generator(device=device).manual_seed(seed)
        self.shards = []           # this rank's [P/W] slice of each bucket (the persistent FSDP state)
        for u in range(self.units):
            full = torch.zeros(self.Ppad, dtype=BF16, device=device)
            full[: self.P].normal_(0, 0.02, generator=gen)
            self.shards.append(full.view(world, self.shard_numel)[rank].clone())

        # --- green contexts: comm (few SMs) + compute (the rest), disjoint ---
        self.green = GreenCtxStreams(device.index, comm_sms=comm_sms)

        # --- symm-mem all-gather over the shards (comm on the green comm partition) ---
        import torch.distributed._symmetric_memory as symm_mem
        from sonicmoe.functional.distributed.collectives import all_gather_triton, rendezvous
        self._ag = all_gather_triton
        self._symm_shards, self._peer_bufs = [], []
        for s in self.shards:
            buf = symm_mem.empty(s.numel(), dtype=BF16, device=device)
            buf.copy_(s)
            hdl = rendezvous(buf, group)
            self._symm_shards.append(buf)
            self._peer_bufs.append(tuple(hdl.get_buffer(r, (s.numel(),), BF16) for r in range(world)))
        dist.barrier(group=group)  # every rank's symm buffer written + P2P-mapped before any peer reads it

    # ---- per-bucket helpers ----------------------------------------------------------------
    def _views(self, full, g):
        """the g-th layer's (router_w, w1, w2) inside a gathered bucket buffer."""
        base = g * self.layer_P
        n0, n1, n2 = self.layer_numels
        E, H, I = self.E, self.H, self.I
        router_w = full[base : base + n0].view(E, H)
        w1 = full[base + n0 : base + n0 + n1].view(E, 2 * I, H).permute(1, 2, 0)  # [2I,H,E]
        w2 = full[base + n0 + n1 : base + n0 + n1 + n2].view(E, H, I).permute(1, 2, 0)  # [H,I,E]
        return router_w, w1, w2

    def _compute_bucket(self, x, full):
        from sonicmoe.enums import ActivationType
        from sonicmoe.functional import moe_TC_softmax_topk_layer
        for g in range(self.G):
            router_w, w1, w2 = self._views(full, g)
            x, _, _ = moe_TC_softmax_topk_layer(
                x, router_w, w1, None, w2, None, self.K, None,
                ActivationType.SWIGLU, is_inference_mode_enabled=True,
            )
        return x

    def _empty_bucket(self):
        return torch.empty(self.Ppad, dtype=BF16, device=self.dev)

    # ---- forward: overlapped (the winning path) -------------------------------------------
    def forward(self, x):
        comm, compute = self.green.comm_stream, self.green.compute_stream
        fulls, gathered = [None] * self.units, [None] * self.units

        def issue_ag(u):
            with torch.cuda.stream(comm):
                full = self._empty_bucket()          # created on comm stream (allocator reuse-fence)
                self._ag(self._symm_shards[u], self.group, out=full, peer_bufs=self._peer_bufs[u])
                ev = torch.cuda.Event(); ev.record(comm)
            fulls[u], gathered[u] = full, ev

        # both green streams must see the current stream's state (input x, params) first
        start = torch.cuda.Event(); start.record(torch.cuda.current_stream())
        comm.wait_event(start); compute.wait_event(start)

        issue_ag(0)
        with torch.cuda.stream(compute):
            for u in range(self.units):
                if u + 1 < self.units:
                    issue_ag(u + 1)                  # prefetch next bucket on the comm partition
                compute.wait_event(gathered[u])      # this bucket's gather must be done
                fulls[u].record_stream(compute)      # cross-stream allocator safety
                x = self._compute_bucket(x, fulls[u])
                fulls[u] = None
        done = torch.cuda.Event(); done.record(compute)
        torch.cuda.current_stream().wait_event(done)  # join back to the caller's stream
        return x

    # ---- forward: reference (non-overlapped, NCCL) — same math, for correctness -----------
    def forward_reference(self, x):
        for u in range(self.units):
            full = self._empty_bucket()
            dist.all_gather_into_tensor(full, self.shards[u], group=self.group)
            x = self._compute_bucket(x, full)
        return x


# ============================================================ demo: verify + time ==============
def _median_ms(fn, dev, group, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    dist.barrier(group=group)
    samples = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(dev); dist.barrier(group=group)
        e0.record(); fn(); e1.record(); e1.synchronize()
        t = torch.tensor([e0.elapsed_time(e1)], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)  # slowest rank defines the sample
        samples.append(t.item())
    return sorted(samples)[len(samples) // 2]


def _run_on(stream, body):
    """run `body()` on `stream`, joined to the caller's stream in and out (for clean timing)."""
    start = torch.cuda.Event(); start.record(torch.cuda.current_stream()); stream.wait_event(start)
    with torch.cuda.stream(stream):
        body()
    done = torch.cuda.Event(); done.record(stream); torch.cuda.current_stream().wait_event(done)


def _comm_only(model, stream):
    def body():
        for u in range(model.units):
            full = model._empty_bucket()
            model._ag(model._symm_shards[u], model.group, out=full, peer_bufs=model._peer_bufs[u])
    _run_on(stream, body)


def _comp_only(model, x, resident, stream):
    def body():
        xx = x
        for u in range(model.units):
            xx = model._compute_bucket(xx, resident[u])
    _run_on(stream, body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--tokens", type=int, default=32768)     # per rank
    p.add_argument("--hidden", type=int, default=2560)
    p.add_argument("--inter", type=int, default=1280)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--bucket-layers", type=int, default=1)  # 1 layer/gather (coarsening doesn't help)
    p.add_argument("--comm-sms", type=int, default=16)      # optimal: fewest comm SMs that still hide the AG
    a = p.parse_args()

    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    is0 = rank == 0

    model = OverlappedFSDPMoE(a.layers, a.hidden, a.inter, a.experts, a.topk, a.bucket_layers,
                              world, rank, dist.group.WORLD, dev, comm_sms=a.comm_sms)
    x = torch.randn(a.tokens, a.hidden, device=dev, dtype=BF16) * 0.2
    if is0:
        print(f"[cfg] world={world} layers={a.layers} bucket_layers={a.bucket_layers} units={model.units} "
              f"tokens/rank={a.tokens} comm_sms={model.green.comm_sms} compute_sms={model.green.compute_sms}",
              flush=True)

    # warm up (QuACK autotune + symm rendezvous), then CORRECTNESS: overlapped == reference
    _ = model.forward_reference(x); _ = model.forward(x)
    torch.cuda.synchronize(dev)
    y_ref = model.forward_reference(x)
    y_ovl = model.forward(x)
    torch.cuda.synchronize(dev)
    # all-gather is data movement (no reduction) and both paths run the identical GEMM, so agreement
    # should be tight; 2e-3 covers GEMM tile-order jitter from the differing SM-partition sizes.
    ok = torch.allclose(y_ref, y_ovl, atol=2e-3, rtol=2e-3)
    max_err = (y_ref - y_ovl).abs().max().item()
    if is0:
        print(f"[correctness] overlapped == reference: {ok}  (max_abs_err={max_err:.2e})", flush=True)
    assert ok, "overlapped forward does not match the reference!"

    # THROUGHPUT
    flop = a.layers * 6 * a.tokens * a.inter * a.hidden * a.topk  # sonicmoe GLU fwd
    ref_ms = _median_ms(lambda: model.forward_reference(x), dev, model.group)
    ovl_ms = _median_ms(lambda: model.forward(x), dev, model.group)
    if is0:
        def tf(ms): return flop / (ms / 1e3) / 1e12
        print(f"[perf] reference (exposed comm): {ref_ms:6.2f} ms  {tf(ref_ms):5.0f} TFLOP/s/GPU "
              f"({tf(ref_ms)/2250*100:.0f}% B200)", flush=True)
        print(f"[perf] overlapped (green+symm):  {ovl_ms:6.2f} ms  {tf(ovl_ms):5.0f} TFLOP/s/GPU "
              f"({tf(ovl_ms)/2250*100:.0f}% B200)  speedup={ref_ms/ovl_ms:.3f}x", flush=True)

    # --- SM-partition tax on compute + overlap efficiency (best-case reference) ---
    resident = [model._empty_bucket() for _ in range(model.units)]  # pre-gathered params for comp-only timing
    for u in range(model.units):
        dist.all_gather_into_tensor(resident[u], model.shards[u], group=model.group)
    torch.cuda.synchronize(dev)
    default_stream = torch.cuda.current_stream(dev)
    comm_full = _median_ms(lambda: _comm_only(model, default_stream), dev, model.group)             # T_comm: best-case comm (full SMs)
    comm_part = _median_ms(lambda: _comm_only(model, model.green.comm_stream), dev, model.group)     # comm on the green comm partition
    comp_full = _median_ms(lambda: _comp_only(model, x, resident, default_stream), dev, model.group)             # T_comp: MoE @152 SM
    comp_part = _median_ms(lambda: _comp_only(model, x, resident, model.green.compute_stream), dev, model.group)  # MoE @compute partition
    if is0:
        g = model.green
        tax = (comp_part / comp_full - 1) * 100
        print(f"[comp] MoE compute-only: {g.compute_sms}SM={comp_part:.2f}ms  full({g.total_sms})SM={comp_full:.2f}ms  "
              f"partition_tax=+{tax:.0f}% (cost of giving {g.comm_sms} SMs to comm)", flush=True)
        print(f"[comm] all-gather: {g.comm_sms}SM={comm_part:.2f}ms  full({g.total_sms})SM={comm_full:.2f}ms", flush=True)
        # Overlap efficiency vs the GLOBAL ideal: best-case comm (T_comm = full-SM AG, ~min) fully hidden behind
        # full-device compute (T_comp = MoE @152). <100% always (comm always steals SMs from compute); peaks at
        # the split that minimizes the overlapped step (megamoe).
        T_comm, T_comp = comm_full, comp_full
        serial = T_comm + T_comp
        eff = (serial - ovl_ms) / (serial - max(T_comm, T_comp))
        print(f"[eff] T_comm(min,{g.total_sms}SM)={T_comm:.2f}ms  T_comp(152SM)={T_comp:.2f}ms  "
              f"megamoe={ovl_ms:.2f}ms  serial={serial:.2f}ms", flush=True)
        print(f"[eff] overlap_efficiency = (serial-megamoe)/(serial-max(T_comm,T_comp)) = "
              f"({serial:.1f}-{ovl_ms:.1f})/{min(T_comm, T_comp):.1f} = {eff*100:.0f}%", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# sonicmoe + FSDP: maximize NCCL bandwidth and comm/compute overlap

**Status**: this is the original plan. Lever 2 (green-context overlap) is what delivered the
measured win (see `RESULTS.md`/`OVERLAP_REPORT.md`). The NVLS/NCCL-env half of Lever 1 below
(`NCCL_NVLS_ENABLE`, buffer registration, channel/bufsize tuning) was never wired up — the shipped
comm path is sonicmoe's Triton P2P all-gather over symmetric memory (`all_gather_triton`,
`overlapped_fsdp_moe.py`), which never calls NCCL and reaches "few SMs, near-peak BW" by a different
mechanism than the one described below.

Standalone PyTorch benchmark (env `smoe`, torch 2.9.1+cu130) that takes the FSDP bottleneck found
in the JAX grug MoE MFU run — **FSDP comm = 53.5% of GPU time, RING_LL, ~4% of NVLink5 peak, fully
exposed** — and attacks it with two independent levers on a stack of **sonicmoe** MoE layers.

## The workload

A stack of `N` `sonicmoe.MoE` layers (H=2560, I=1280, E=64, K=4, SWIGLU) — the same MoE shape as
grug, ~629M params/layer (router `[64,2560]` + w13 `[64,2560,2560]` + w2 `[64,2560,1280]`).
**Manual FSDP** over the data process group (W ranks): each layer's params are flattened into one
buffer and sharded `[P/W]` per rank. Manual (not `torch.distributed.fsdp`) because both levers need
control stock FSDP hides — bucket size/protocol (lever 1) and per-stream green-context placement
(lever 2). sonicmoe's grouped GEMMs honor `torch.cuda.current_stream()` at call time, and the
functional entry `moe_TC_softmax_topk_layer(x, router_w, w1, w2, ...)` takes the gathered weights as
explicit args — so a gathered flat buffer views straight into `(router_w, w1, w2)` and computes on
whatever stream we set.

## Lever 1 — maximize network bandwidth

Small per-parameter all-gathers make NCCL 2.31 pick **RING_LL** (0.6 µs latency, hard-capped at
~282 GB/s bus BW on Blackwell-N1, and latency-bound below that). Fixes, in order of increasing win:
1. **Bucket** all of a layer's params into ONE flat all-gather (~1.26 GB bf16 full) → crosses NCCL's
   LL→LL128→Simple thresholds (Simple ~70 GB/s/channel vs LL128 40).
2. **NVLS + NVLink-SHARP** (the intended FSDP fast path): `NCCL_CTA_POLICY=EFFICIENCY` +
   `NCCL_NVLS_ENABLE=1` + **buffer registration** (`ncclMemAlloc`+`ncclCommWindowRegister`, or torch
   symmetric memory) → NVLS AG/RS at ~80-96 GB/s/channel using **only 6 SMs** (which also frees SMs
   for overlap). Tune `NCCL_MIN_NCHANNELS` (16-32), `NCCL_BUFFSIZE` (8-16 MiB), `NCCL_NET_PLUGIN=none`.
   Arm order: `per_param` (reproduce LL) → `flat` bucket → `flat`+NVLS.

## Lever 2 — maximize overlap (green contexts)

The MFU identity held tightly *because* comm was exposed; hiding it is the headroom. We overlap the
**next** layer's all-gather with the **current** layer's MoE compute using the DeepEP-overlap event
schedule, but partition SMs with **CUDA green contexts** (the user's choice — sonicmoe's QuACK GEMM
and NCCL don't expose an SM-budget arg like the DeepEP cuDNN path, so we isolate at the stream level):

- Two **disjoint** SM partitions via the raw `cuda.bindings.driver` sequence (torch-native
  `GreenContext` isn't compiled into torch 2.9.1): `cuDeviceGetDevResource(SM)` →
  `cuDevSmResourceSplitByCount(nbGroups=1, minCount=comm_sms)` → `result[0]` (comm) + `remainder`
  (compute) → `cuDevResourceGenerateDesc` → `cuGreenCtxCreate` → `cuGreenCtxStreamCreate` →
  `torch.cuda.ExternalStream(int(cu_stream))`. **One split + remainder** (not two `create()` calls,
  which can overlap). SM count is a **multiple of 8** on SM100; comm partition e.g. 8-16 SMs (NVLS
  needs only 6), compute gets the rest (~132).
- Schedule (per the DeepEP `overlap_schedule/forward.py`): issue `AllGather(L+1)` on the comm stream
  one step ahead; `compute_stream.wait_event(ag_done[L])`; `full[L].record_stream(compute_stream)`;
  compute L on the compute stream; join at the end. Backward is symmetric (reduce-scatter grads on
  the comm stream overlapping dgrad).

## Arms (apples-to-apples, ABBA-paired like the DeepEP harness)

| arm | bucket | comm stream | SMs |
|---|---|---|---|
| `serial` (baseline/control) | per_param or flat | compute stream (no overlap) | all |
| `overlap_plain` | flat | separate plain stream | shared (contended) |
| `overlap_greenctx` | flat | green-ctx comm stream | disjoint (comm N, compute rest) |

Metrics: step time (CUDA-event bracket on compute stream, rank-MAX all-reduce, warmup, median),
achieved AllGather **busbw** (bytes/time), comm wall %, and the overlap speedup vs `serial`.

## Steps

1. **Baseline** — manual FSDP forward pipeline, `serial`, per_param vs flat → measure BW + exposed comm.
2. **Lever 1** — flat bucket + NCCL NVLS/CTA_POLICY env → measure BW toward peak.
3. **Lever 2** — green-ctx `overlap_greenctx` → measure the MFU/throughput gain from hiding comm.
4. Add backward (reduce-scatter grads) for the full FSDP picture.

Env `smoe`; SLURM 1 node / 4×GB200; `torchrun --standalone --nproc_per_node=4`.

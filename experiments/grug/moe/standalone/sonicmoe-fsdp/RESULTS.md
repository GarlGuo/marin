# sonicmoe + FSDP: results (4×GB200, forward pass, 16-layer MoE stack)

Env `smoe-t371` (torch 2.9.1+cu130, **nvidia-cutlass-dsl 4.5.2**, quack 0.5.0), `H=2560 I=1280 E=64
K=4 SWIGLU`, ~629M params/layer, manual FSDP over W=4 (each layer's params flat-gathered, 944 MB
AllGather wire/layer). Measurement: CUDA-event bracket on the compute stream, per-iter sync+barrier,
**rank-MAX** all-reduce, median of 20. GB200 = **152 SMs**; green contexts via the raw
`cuda.bindings.driver` split (verified disjoint + concurrent), NCCL run on the green comm stream via a
**ctypes** comm (torch.cuda.nccl.init_rank is broken in 2.9.1; PyTorch PG runs NCCL on its own internal
stream, so it can't be green-confined).

> **Env note (2026-07-17):** `smoe` had `nvidia-cutlass-dsl` bumped to **4.6.1**, which removed
> `cute.core.ThrMma` that quack 0.5.0 references → the whole `sonicmoe` import chain breaks. Use
> `smoe-t371` (identical torch 2.9.1+cu130, cutlass-dsl **4.5.2**) until `smoe` is repinned.

> **Correction (2026-07-17, jobs 5454414/5454539 @bucket=1, 5454415/5454540 @bucket=2):** the win is
> **green-context overlap + symm-mem all-gather**, measured best at **1 layer/gather (bucket=1)**.
> Multi-layer bucket *coarsening is not needed and does not help the overlap* — the earlier "bucketing
> lifts the overlap / bucket=2 sweet spot" conclusion was an artifact of comparing an *early untuned*
> bucket=1 run against a *later tuned* bucket=2 run. Measured same-session with the final code, green +
> symm-mem wins at bucket=1 (1.41×) and bucket=1 ≥ bucket=2.

## Lever 1 — bandwidth on FEW SMs (the actual enabler)

The green comm partition only gets 32 SMs, so what matters is **BW-at-32-SMs**, and that's exactly
where sonicmoe's symm-mem all-gather beats NCCL Ring:

| all-gather | full SMs | green 32 SM | green 16 SM |
|---|---|---|---|
| NCCL Ring (raw) | 587–607 (65–67%) | ~586 (65%) | 335 (starved) |
| **sonicmoe symm-mem** | **668 GB/s (74%)** | **661–664 (73%)** | 473 |

(vs the JAX XLA FSDP baseline ~28 GB/s / 4%, RING_LL small msgs.) symm-mem is a Triton NVLink **P2P**
read (`all_gather_triton` over `torch.distributed._symmetric_memory` buffers), so it holds near-peak BW
on 32 SMs where NCCL Ring — which needs ~32 SMs just to fill — collapses. That few-SM bandwidth is the
whole reason the green-context partition pays off. NCCL full-device BW rises slightly with coarser
buckets (587→607 GB/s at 1→2 layers/gather), but the overlap doesn't use it.

## Lever 2 — comm/compute overlap (green + symm-mem wins, best at bucket=1)

All arms, same session, 32768 tok/rank, comm partition = 32 SM. FLOPs = 6·T·I·H·K/layer × 16 =
41.2 TFLOP/step/rank (forward-MoE-only, not full-model MFU).

**bucket=1 (1 layer/gather) — job 5454414:**

| arm | step | TFLOP/s/GPU | %B200 | speedup |
|---|---|---|---|---|
| serial (control, comm exposed) | 52.59 ms | 783 | 35% | 1.00× |
| overlap_plain (2 streams, shared SMs) | 37.45 ms | 1100 | 49% | 1.404× |
| greenctx raw-NCCL c32 | 37.66 ms | 1094 | 49% | 1.396× |
| **greenctx + symm-mem c32** | **37.12 ms** | **1110** | **49%** | **1.417× ← best arm @comm=32** |

**bucket=2 (2 layers/gather) — job 5454415:**

| arm | step | TFLOP/s/GPU | %B200 | speedup |
|---|---|---|---|---|
| serial | 51.94 ms | 793 | 35% | 1.00× |
| overlap_plain | 39.77 ms | 1036 | 46% | 1.306× |
| greenctx raw-NCCL c32 | 39.96 ms | 1031 | 46% | 1.300× |
| **greenctx + symm-mem c32** | **39.24 ms** | **1050** | **47%** | **1.324×** |

**green + symm-mem wins at both buckets** — the disjoint SM partition + few-SM BW beats plain's
shared-SM contention, and even greenctx raw-NCCL is now competitive with plain (not losing). **bucket=1
≥ bucket=2**: finer pipelining (16 vs 8 stages, smaller unhideable first gather) hides more comm. The
self-consistent `overlapped_fsdp_moe.py` runs give bucket=1 **37.27 ms** / bucket=2 **37.82 ms** — a
smaller gap than the harness's 37.1/39.2 (the harness's larger bucket=2 penalty was partly node
variance). Robust conclusion: green+symm wins, and coarser buckets don't help.

## before → after

**~785 → 1180 TFLOP/s/GPU (35% → 52–53% of B200), ~1.5×**, from green-context + symm-mem overlap at
bucket=1, **comm=16 SM** split (§ next). Correctness: the overlapped forward is **bit-exact** vs the
NCCL-serial reference (`max_abs_err = 0.00e+00`). The compute-only ceiling (comp @152 SM, comm free)
≈ 29 ms = ~1420 TFLOP/s (63%); the overlap's realistic floor is comp @136 SM = 29.7 ms (compute lives
on the 136-SM partition, +3–5% tax), and the winner sits ~5 ms above it.

## Tuning the comm/compute SM split — comm=16 / compute=136 is optimal

Sweeping the green comm partition `C` (compute gets `152−C`), green + symm-mem, bucket=1, fresh
autotune cache each run, all bit-exact:

| comm SM | comp SM | comm ms | comp ms | tax | overlap ms | TFLOP/s | %B200 | speedup | efficiency |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 144 | 56.93 | 29.45 | +2% | 61.42 | 671 | 30% | 0.865× | −44% |
| **16** | **136** | 27.1 | 29.7 | +3–5% | **34.8** | **1180** | **52–53%** | **1.49–1.53×** | **74%** |
| 24 | 128 | 23.78 | 31.33 | +5% | 35.78 | 1152 | 51% | 1.463× | 73% |
| 32 | 120 | 22.72 | 32.47 | +9% | 37.38 | 1103 | 49% | 1.401× | 66% |
| 40 | 112 | 22.64 | 33.99 | +15% | 39.06 | 1056 | 47% | 1.341× | 57% |
| 64 | 88 | 22.62 | 41.53 | +41% | 46.19 | 893 | 40% | 1.134× | 26% |
<br>(efficiency uses the best-case reference below — `T_comm`=min comm 22.62 ms, `T_comp`=comp@152, not the per-split comm/comp columns.)

The overlap is **compute-bound at every split ≥16 SMs** (comp > comm), so the floor is `comp@(152−C)`
and the step tracks it — minimize the step by **maximizing compute SMs = minimizing comm SMs**, as far
as comm still hides. symm-mem's comm saturates ~32 SMs (22.6 ms); at 16 SMs it slows to 27.1 ms but is
**still < comp@136 (29.7), so fully hidden**, and compute gets 136 SMs (+3–5% tax vs +9% at 120). Net:
**comm=16 optimal, ~34.8 ms, ~1.5×**. Below it is a *cliff*: `comm=12` rounds up to 16 (8-SM
granularity), and **comm=8** over-starves the AG to 56.93 ms (double the compute) → comm-bound → 0.865×,
**slower than serial**. comm=16 is the smallest achievable partition that still hides the all-gather.

## Overlap efficiency (best-case reference)

`efficiency = (serial − megamoe) / (serial − max(T_comm, T_comp))` with **serial = T_comm + T_comp**,
where `T_comm` = the **minimum** comm time (best case, @64 SM = **22.62 ms**), `T_comp` = **pure compute
@152 SM** (~29 ms), `megamoe` = the measured overlap step. Since T_comp > T_comm, the denominator is
`T_comm = 22.62 ms`. This measures how close the overlap gets to the **global ideal** — best-case comm
fully hidden behind full-device compute — so it peaks at the optimal split and penalizes both extremes:

| comm SM | 8 | **16** | 24 | 32 | 40 | 64 |
|---|--:|--:|--:|--:|--:|--:|
| `T_comp` (comp@152) | 28.86 | 29.17 | 29.76 | 29.66 | 29.44 | 29.55 |
| `megamoe` (overlap) | 61.42 | 35.09 | 35.78 | 37.38 | 39.06 | 46.19 |
| **efficiency** | **−44%** | **74%** | 73% | 66% | 57% | 26% |

At the optimum **comm=16**: `(22.62 + 29.17 − 35.09) / 22.62 = 16.70 / 22.62 = ` **74%**. No split reaches
100% (comm always steals SMs from full-device compute); comm=16 gets closest. Efficiency falls off as
the excess `megamoe − comp@152` (= compute-partition tax + pipeline fill) grows — 26% at comm=64
(compute starved to 88 SMs), **−44% at comm=8** (comm-bound, worse than the ideal serial). Using the
*minimum* comm as T_comm (not the starved comm@C) is what makes comm=8 correctly negative.

## Copy-engine all-gather — investigated (parity, HBM-contention-bound)

sonicmoe's `all_gather_copy_engine_async` (peer reads via `cuMemcpyAsync` on the DMA/**copy engines**,
**0 SMs**) was benchmarked as an alternative comm, to free ALL 152 SMs for compute (no green partition,
no SM tax). The standalone test scripts were removed after the investigation; the findings:

- **As a primitive it wins.** On 4×GB200 it hits **703 GB/s (78% of NVLink5) on ZERO SMs** at
  `num_streams ≥ 4` — *beating* the SM-based symm-mem Triton AG (669 GB/s on 152 SMs), bit-exact. It
  also lets you delete the entire green-context stack (no raw-driver SM split, no ctypes NCCL). So this
  refutes the "SM kernels beat DMA for NVLink" folklore on GB200.
- **In the overlap it does NOT beat green-symm — it's memory-contention-bound.** Best CE overlap
  (prefetch-1) = **36.5 ms / 1.43×** vs green-symm **35.3 ms / 1.49×** — parity. Every schedule tried
  tied or regressed (prefetch-1/2/4/all, a decoupled comm stream, per-bucket completion events: 36.5–48
  ms; the *simplest* prefetch-1 was best).
- **nsys (rank-0) pinned the cause.** The GEMMs run only **+2%** slower under concurrent CE traffic (so
  compute isn't the victim), but the **CE copies are throttled +22%** when concurrent with the GEMM
  (551 → 671 µs/copy): the copy engine's HBM *writes* (gathered weights, ~15 GB) contend with the GEMM's
  HBM *reads* (weights, ~20 GB). HBM BW isn't saturated (~1.4 of 8 TB/s), so it's arbitration / L2
  eviction, not raw BW — but the throttle is real. So the effective comm under compute (~32 ms) *exceeds*
  the compute (~27.6 ms) → the overlap goes comm-bound → ~36 ms. No scheduling or CE-API change fixes it
  (comm and compute share HBM either way; green-symm has the same contention, just via SMs).

**Conclusion:** adopt CE for **equal performance at much lower complexity** (0 SMs, no green context),
but the overlap ceiling here is **~1.5×, memory-contention-bound**. The only path to the ~1.9×
compute-only ceiling is to remove the gather→HBM→read-back round-trip — see the TMA-fusion TODO.

## TODO
- Backward pass (grad reduce-scatter overlapped with dgrad) — symmetric to the forward AG path; the
  compute-elasticity sweep shows backward pays the same ~10% partition tax, so it should overlap as well.
- **TMA-fused grouped GEMM** — the real lever past ~1.5×: load each expert's weights directly from peer
  HBM over NVLink into shared memory (no gather → HBM → read-back), removing the comm/compute HBM
  contention that pins *both* the green-ctx and copy-engine overlaps. Reaches toward the comp@152 ceiling
  (~1.9×); needs a custom grouped GEMM + tile/expert-aligned sharding. (Deeper prefetch does NOT help —
  the overlap is HBM-contention-bound, not scheduling-bound; see the CE section above.)
- Try `all_gather_multimem_triton` (NVLink-SHARP multicast) for even higher few-SM BW.
- Repin `smoe` to cutlass-dsl 4.5.2 (or update quack for 4.6.1) so the canonical env imports sonicmoe.

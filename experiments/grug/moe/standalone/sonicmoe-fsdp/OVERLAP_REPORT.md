# sonicmoe + FSDP: comm/compute overlap on 4×GB200 — approach report

## 1. Problem & setup

Grug **row-13** MoE (H2560 / I1280 / E64 / K4, SwiGLU) trained with FSDP: the all-gather of the
sharded parameters sits on the critical path of every layer. In the JAX/XLA baseline the FSDP comm was
**53.5 % of GPU time** — NCCL **RING_LL at ~28 GB/s (~4 % of NVLink5), fully exposed** (serialized
before compute). Goal: **hide the parameter all-gather behind the MoE compute.**

**Setup.** 4×GB200 (152 SMs, 192 GB each), 16-layer sonicmoe MoE stack, **32768 tokens/rank**, bf16,
manual FSDP over the W=4 data group. Each layer's params are flattened into **one** all-gather (1
layer/gather; coarsening more layers into a gather does *not* help the overlap — see §6). `smoe-t371`
env (torch 2.9.1+cu130, nvidia-cutlass-dsl 4.5.2, quack 0.5.0). Compute FLOPs = `6·T·I·H·K` per layer
× 16 = **41.2 TFLOP/step/rank**; TFLOP/s = 41.2 / step_time.

All overlap arms use a **prefetch-by-one** schedule: the next layer's all-gather is issued *before* the
current layer's compute is awaited, so gather(u+1) runs concurrently with compute(u); CUDA events
enforce "gather done before compute consumes," and the buffer is `record_stream`-protected across
streams.

## 2. The four approaches (mechanism in detail)

### 2.1 Serial (control)
Per layer: `all_gather_into_tensor(full, shard)` → run the MoE compute on `full` → free. Everything is
on **one** stream, so the GPU is idle-on-compute during the gather and idle-on-comm during compute.
The exposed-comm baseline.

### 2.2 `overlap_plain` — two streams, shared SMs
Two CUDA streams: the compute stream, and PyTorch `ProcessGroupNCCL`'s **own internal** comm stream (a
PG collective always runs on its internal stream, not the caller's — this is what makes plain overlap
possible with no partitioning). Prefetch-by-one issues the next gather on the PG stream while compute
runs on the compute stream. **Both streams run on the full 152 SMs**, so the NCCL kernels and the QuACK
GEMM **contend** — the SM scheduler time-slices between them and neither runs at full speed. That
contention is the ceiling on this arm.

### 2.3 `greenctx + NCCL` — disjoint SM partitions, NCCL comm
Two **physically disjoint** SM partitions via CUDA green contexts (raw `cuda.bindings.driver`: a single
`cuDevSmResourceSplitByCount(nbGroups=1, minCount=C)` → a C-SM **comm** partition + a (152−C)-SM
**compute** partition, guaranteed non-overlapping; C is a multiple of the 8-SM coscheduled alignment).
The all-gather runs on the comm partition, the MoE on the compute partition — **no SM contention**. The
all-gather uses a **raw NCCL comm via a ctypes wrapper over `libnccl.so.2`** (torch's `init_rank` is
broken in 2.9.1; a PG collective can't be confined to a green stream). Works, but NCCL Ring is
BW-limited on few SMs (~586 @32 vs 668 GB/s full).

### 2.4 `greenctx + symm-mem` — disjoint partitions, symmetric-memory comm (winner)
Same disjoint green partitions, but the comm is **sonicmoe's `all_gather_triton`** — a Triton NVLink
**P2P** all-gather over `torch.distributed._symmetric_memory` buffers. Because it's a **direct P2P
read**, not a ring that needs SM-parallelism to fill, it holds near-peak BW on few SMs where NCCL Ring
collapses (668 full / 661 @32 / 473 @16 GB/s, vs NCCL 587 / 586 / 335). So the small comm partition
delivers high bandwidth while the compute partition runs the MoE, fully disjoint. This is the winner.

## 3. Results — the four arms (comm=32 SM split, bucket=1, same session)

| approach | step (ms) | TFLOP/s/GPU | % B200 | speedup |
|---|---|---|---|---|
| Serial (control, comm exposed) | 52.59 | 783 | 35 % | 1.00× |
| `overlap_plain` (2 streams, shared SMs) | 37.45 | 1100 | 49 % | 1.404× |
| `greenctx + NCCL` c32 (disjoint) | 37.66 | 1094 | 49 % | 1.396× |
| `greenctx + symm-mem` c32 | 37.12 | 1110 | 49 % | 1.417× |

All green arms beat plain (disjoint partition avoids plain's shared-SM contention); symm-mem edges NCCL
because it holds BW on few SMs. **But comm=32 is not the best split** — tuning the comm/compute SM
allocation (§3.1) takes the symm-mem arm to **1.5×**.

## 3.1 Tuning the comm/compute split — comm=16 / compute=136 is optimal

Sweeping the comm partition `C` (compute gets `152−C`), green + symm-mem, bucket=1, fresh
compile/autotune cache each run, all bit-exact:

| comm SM | comp SM | comm ms | comp ms | tax | **overlap ms** | TFLOP/s | %B200 | **speedup** | regime |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 8 | 144 | 56.93 | 29.45 | +2% | 61.42 | 671 | 30% | 0.865× | comm-bound — *slower than serial* |
| **16** | **136** | 27.1 | 29.7 | +3–5% | **34.8** | **1180** | **52–53%** | **1.49–1.53×** | **optimal (AG just hidden)** |
| 24 | 128 | 23.78 | 31.33 | +5% | 35.78 | 1152 | 51% | 1.463× | compute-bound |
| 32 | 120 | 22.72 | 32.47 | +9% | 37.38 | 1103 | 49% | 1.401× | compute-bound |
| 40 | 112 | 22.64 | 33.99 | +15% | 39.06 | 1056 | 47% | 1.341× | compute-bound |
| 64 | 88 | 22.62 | 41.53 | +41% | 46.19 | 893 | 40% | 1.134× | compute-starved |

**The overlap is compute-bound at every split ≥16 SMs** (comp > comm), so the floor is `comp@(152−C)`
and the step tracks it. Minimizing the step therefore means **maximizing compute SMs = minimizing comm
SMs** — as far as comm can go while still hiding. symm-mem's comm saturates at ~32 SMs (22.6 ms); at 16
SMs it slows to 27.1 ms but is **still < comp@136 (29.7 ms), so fully hidden** — and compute now runs on
136 SMs (+3–5% tax vs +9% at 120). That trade (a hidden comm slowdown for an exposed compute speedup)
makes **comm=16 the optimum: ~34.8 ms, ~1180 TF/s (52–53%), ~1.5×**.

**Below 16 it's a cliff, not a step:** `comm=12` rounds up to 16 (8-SM granularity), and **comm=8** over-
starves the all-gather — it collapses to **56.93 ms** (8 SMs can't drive NVLink), *double* the compute,
so the overlap goes comm-bound and balloons to 61.42 ms → **0.865×, slower than serial.** So comm=16 is
the smallest achievable partition that still hides the AG.

## 4. Overlap efficiency (best-case reference)

Efficiency against the **global ideal** — best-case comm fully hidden behind full-device compute —

  **efficiency = (serial − megamoe) / (serial − max(T_comm, T_comp))**,  **serial = T_comm + T_comp**

with `T_comm` = the **minimum** comm time (best case, @64 SM = **22.62 ms**), `T_comp` = **pure compute
@152 SM** (~29 ms), and `megamoe` = the measured overlap step. Since T_comp > T_comm the denominator is
`T_comm = 22.62 ms`, i.e. `eff = (22.62 + comp@152 − megamoe) / 22.62`:

| comm SM | 8 | **16** | 24 | 32 | 40 | 64 |
|---|--:|--:|--:|--:|--:|--:|
| comp@152 | 28.86 | 29.17 | 29.76 | 29.66 | 29.44 | 29.55 |
| megamoe | 61.42 | 35.09 | 35.78 | 37.38 | 39.06 | 46.19 |
| **efficiency** | **−44%** | **74%** | 73% | 66% | 57% | 26% |

At the optimum **comm=16**: `(22.62 + 29.17 − 35.09) / 22.62 = ` **74%**. No split reaches 100% (comm
always steals SMs from full-device compute); comm=16 gets closest, and efficiency falls off as the
excess over ideal `= megamoe − comp@152` (= compute-partition tax + pipeline fill) grows — **26%** at
comm=64 (compute starved to 88 SMs), **−44%** at comm=8 (comm-bound, *worse than the ideal serial*).

**Why minimum comm:** using `T_comm` = the *minimum* comm time (not the starved comm@C) is what makes
comm=8 correctly **negative** — the reference serial (22.62 + comp) is sensible, and megamoe exceeding it
signals a bad (comm-bound) split. The +3–5% partition tax at 136 SMs matches the `moe_sm_tflops.py` sweep.

## 5. Why the partition is cheap — the compute-elasticity data

The green partition works because **the compute barely misses the SMs it gives up**. The standalone
sweep (`~/sonic-moe/benchmarks/moe_sm_tflops.py`, single-GB200, 5 MoE shapes × {152…8 SMs}) shows MoE
compute is **occupancy-bound at full device** (each SM only ~52 % utilized), so it keeps ~95 % of
throughput at 136 SMs (comm=16) and ~90 % at 120 (comm=32) — matching the **+3–5 % / +9 % tax** measured
here. It stays elastic to ~88 SMs (comm=64, +41 % — the knee). Meanwhile the comm needs only BW, which
symm-mem delivers on 16–32 SMs. So the optimum hands compute *almost all* the SMs and comm just enough
to stay hidden. (Full compute-elasticity tables in `moe_sm_tflops_RESULTS.md`.)

## 6. On bucketing (multi-layer gather coarsening) — not needed

Coarsening several layers' params into one all-gather raises NCCL full-device BW slightly (587→607 GB/s
at 1→2 layers/gather) but **does not improve the overlap** — green + symm-mem is best at **1 layer/gather**
(bucket=1 vs bucket=2: 37.27 vs 37.82 ms / 78 % vs 74 % eff at comm=32). Finer pipelining (16 vs 8
stages, smaller unhideable first gather) hides more comm, and symm-mem already holds 661 GB/s on 32 SMs
at 1 layer/gather. (An earlier "bucketing lifts the overlap" claim was an artifact of comparing an early
untuned bucket=1 run against a later tuned bucket=2 run — corrected here.)

## 7. Takeaways & remaining work

- **Bandwidth:** the lever is few-SM BW — symm-mem holds **668 GB/s full / 661 @32 / 473 @16 (74/73/53 %
  of NVLink5)**, where NCCL Ring collapses (587 / 586 / 335). That is what lets the comm partition be tiny.
- **Optimal split:** **comm=16 / compute=136** (not 32/120). Since the overlap is compute-bound, push
  comm as *low* as it stays hidden and give the rest to compute. Result: **~785 → 1180 TF/s/GPU
  (35 → 52–53 % of B200), ~1.5×, 74 % overlap efficiency (best-case ref), bit-exact.** comm=8 is a cliff (comm-bound,
  0.87×); comm=32 leaves ~6 % on the table.
- **Remaining:** the **backward pass** (gradient reduce-scatter ∥ dgrad) — the compute-elasticity sweep
  shows backward pays the same small tax, so it should overlap as well; deeper prefetch could push
  toward the compute-only ceiling (comp@152 ≈ 29 ms = ~1420 TF/s, 63 %).

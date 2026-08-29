# GB200 run record — grug_moe_mfu.py (NVIDIA internal SLURM)

Internal record of building the env and running/profiling this benchmark on an NVIDIA GB200/NVL72
aarch64 SLURM cluster. Cluster-ops detail (module names, lustre paths, profiling) lives in the
`marin-moe-mfu` skill under `~/megatron-lm/skills/`. This file is a portable-README companion, not
part of the upstream benchmark.

## Environment (conda env `marin`, CUDA 13, aarch64, JAX 0.10.1)

The benchmark imports only `levanter.grug.*` + `levanter.optim` + haliax/eqx/jax — no `marin`
package. **Use the frozen `uv.lock` pins**: a fresh resolve pulls newer FA4/cutlass builds
(`flash-attn-4 4.0.0b19`, `nvidia-cutlass-dsl 4.5.3`) whose API drifted from levanter's call sites
and crashes the FA4 attention **backward** compile with
`FlashAttentionBackwardPreprocess.__call__() missing ... 'mScaleP','softmax_scale'`. Validated pins:
`flash-attn-4==4.0.0b16`, `nvidia-cutlass-dsl(+libs)==4.5.2`, `quack-kernels==0.5.0`,
`jax/jaxlib==0.10.1`.

```bash
conda create -y -n marin python=3.12 && conda activate marin
cd /home/wentaog/marin
uv sync --frozen --package marin-levanter --extra gpu --active --inexact   # -> ./.venv (correct pins)
uv pip freeze --python .venv/bin/python > /tmp/reqs.txt
uv pip install --python "$(conda run -n marin which python)" -r /tmp/reqs.txt
```

## Run (single-process multi-GPU JAX, 1 node / 4 GPU — no torchrun)

```bash
# sbatch launcher: .../ep_bench/srun/marin_moe_mfu/run_marin_moe_mfu.slurm  (job name "marin")
sbatch run_marin_moe_mfu.slurm                            # sonic_cute EP1 (default), --num-gpus 4
sbatch run_marin_moe_mfu.slurm ring_cute_ep4 ring_cute 4  # ring_cute EP4
```
Cold compile ~10 min (FA4 CuTeDSL + QuACK); persist `JAX_COMPILATION_CACHE_DIR` /
`CUTE_DSL_CACHE_DIR` / `TRITON_CACHE_DIR` to amortize. `--num-gpus` only sets the MFU denominator and
must equal the visible device count.

## Results (4xGB200-192GB, 2026-07-16, steady median steps 8-19)

| backend | MFU (B200) | aggregate TFLOP/s | per-GPU | tok/s | s/step |
|---|---|---|---|---|---|
| `sonic_cute` EP1 | 17.4% | 1565 | 391 | 91.8k | 5.71 |
| `ring_cute` EP4  | 16.4% | 1475 | 369 | 86.5k | 6.07 |

Identical loss trajectory 11.806->10.455 across both backends (cross-validates them). Higher than the
README's 8-GPU refs (more compute-saturated per GPU on 4 GPUs, less dispatch/comm).

## MFU accounting

`steady_median_achieved_tflops` is an **aggregate over all GPUs**; MFU = achieved / (num_gpus x
2.25e15 B200-dense-bf16). So 1565 TFLOP/s is the 4-GPU total → 391/GPU → 17.4% (not 1565/2250). MFU
counts only analytic model FLOPs (3x fwd+bwd, active experts); wall-clock also pays full-remat
recompute (HFU≈MFU*4/3), MuonH Newton-Schulz matmuls, attention/router/norms/casts, and EP comm.

## Time breakdown (nsys) — why sonic_cute is 17% MFU

nsys profile of a warm sonic_cute EP1 step (4xGB200; perturbation minimal, 5.85 vs 5.71 s/step).
GPU kernel-time share (relative — the trace span includes JAX init/compile, so don't read absolute
per-step sums):

| category | % of GPU kernel time |
|---|---|
| **FSDP communication** (NCCL AllGather 33% + ReduceScatter 14% + AllReduce ~6%) | **53.5%** |
| **GEMM compute** (QuACK gated 19% + cuBLAS nvjet 6.5% + dot) | **25.9%** |
| elementwise/fusion (mul/add/slice/norm/cast) | 7.7% |
| MoE dispatch (gather/scatter/concat/transpose) | 7.3% |
| router sort (cub radix) | 2.9% |
| D2D copy | 2.4% |

The QuACK gated GEMM is efficient: **4.66 ms/launch, 6.87e12 FLOP → 1473 TF/s = 65% of B200 peak.**
The MFU is diluted, not GEMM-bound: **MFU ≈ GEMM-share x GEMM-eff = 0.26 x 0.65 = 17.1%** (≈ measured
17.2%). The GPU is ~fully busy but only ~¼ of the time runs FLOP-producing GEMMs; **~54% is FSDP
communication, and it's *exposed*** (the identity holds → comm is not overlapped with compute).

**ncu (single-GPU microbench — the distributed 4-GPU CuTeDSL GEMM won't replay; 1 GPU/4 layers/batch 32
gives the same 4.70 ms GEMM as production):** `quackgemm_actGemmGatedSm100` is **Compute (SM) 90.4%,
Tensor Core the over-utilized bottleneck pipe, DRAM only 24.8%, L2 hit 79%** → tensor-core-bound and
near-optimal; achieved occupancy 10.9% is expected for Blackwell tcgen05 UTC-MMA. No GEMM headroom.

**Biggest optimization lever:** overlap / bucket the per-layer FSDP AllGather+ReduceScatter (RING_LL
small-message latency-bound at d2560, ~320 all-gathers/step). Then trim full-remat recompute
(HFU≈MFU*4/3) and fuse the router sort/dispatch. The GEMM needs no work.

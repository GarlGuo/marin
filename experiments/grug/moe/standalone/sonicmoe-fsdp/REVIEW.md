# How to review `overlapped_fsdp_moe.py` (the best impl)

The winning configuration, extracted clean from the benchmark harness (`fsdp_moe.py`) into one
~190-line module + `greenctx.py`. It is **self-verifying** (checks the overlapped forward is
numerically equal to a plain reference) and **self-timing**.

## Read in this order

1. **Module docstring** (`overlapped_fsdp_moe.py` top) — the three composed ideas (bucketing, green
   contexts, symm-mem all-gather) and why each is needed.
2. **`greenctx.py`** — the SM-partition primitive. Check: a **single**
   `cuDevSmResourceSplitByCount(nbGroups=1, minCount=comm_sms)` → `result[0]` (comm) + `remainder`
   (compute), which guarantees the two partitions are **disjoint** (two `create()` calls would not);
   each CUstream wrapped as `torch.cuda.ExternalStream`.
3. **`OverlappedFSDPMoE.__init__`** — FSDP sharding (flat bucket of `G` layers, `[P/W]` per rank),
   green-context creation, and the symm-mem setup (shards copied into `symm_mem.empty` buffers,
   `rendezvous`, one `dist.barrier()`).
4. **`forward()`** — THE core: the prefetch-by-one overlap schedule.
5. **`forward_reference()`** — the plain NCCL, single-stream baseline (same math).
6. **`main()`** — warmup → correctness assert → timing.

## The 5 things to scrutinize (correctness-critical)

1. **Overlap event schedule** (`forward`): `issue_ag(u+1)` is issued *before* `compute.wait_event(
   gathered[u])`, so the next bucket's gather (comm partition) overlaps the current bucket's compute
   (compute partition). `fulls[u].record_stream(compute)` protects the buffer across streams; the
   `start`/`done` events join the caller's stream in and out. **This is where a race would hide** —
   the correctness assert is the guard.
2. **Sharding ↔ views** (`_views`): the flat gathered bucket is sliced back into
   `(router_w [E,H], w1 [2I,H,E], w2 [H,I,E])` with per-layer offsets `base = g*layer_P` and the same
   `.permute(1,2,0)` sonicmoe's `moe_TC_softmax_topk_layer` expects (cross-check with
   `sonic-moe/benchmarks/moe-cute.py`).
3. **Green-ctx disjointness** (greenctx.py) — single split + remainder, SM count a multiple of 8.
4. **Symm-mem contract**: shards written → `rendezvous` (P2P map) → `dist.barrier()` before any peer
   read; `all_gather_triton` is a unicast P2P read of `peer_bufs`. Params are fixed here so one
   barrier suffices — **real FSDP needs a barrier per step** (noted in code).
5. **The correctness check itself** — `assert torch.allclose(forward(x), forward_reference(x))`. If
   this passes, the symm-mem all-gather + green-ctx overlap are numerically correct. It is the
   strongest single review artifact.

## Run it (smoe env, 4×GB200)

```bash
cd experiments/grug/moe/standalone/sonicmoe-fsdp
sbatch run.slurm overlapped_fsdp_moe.py 4     # logs -> .../ep_bench/srun/sonicmoe_fsdp/logs/
```
Expected: `[correctness] overlapped == reference: True (max_abs_err ~1e-3)` and
`[perf] ... speedup≈1.5x` (reference ~785 → overlapped ~1180 TFLOP/s/GPU).

Knobs: `--bucket-layers` (default 1, coarsening doesn't help), `--comm-sms` (default 16, the optimal
split), `--tokens`.

## Context / provenance
- `RESULTS.md` — the full measured sweeps this impl distills (BW curves, arm tables, TFLOPS before→after).
- `fsdp_moe.py` — the benchmark harness with all arms (serial / plain / green-ctx raw / green-ctx symm)
  and the bucket / comm-SM / NCCL-env sweeps; use it to reproduce any row of RESULTS.md.
- `DESIGN.md` — the plan.

## Known limitations (call these out in review)
- **Forward/inference only.** Backward (grad reduce-scatter ∥ dgrad) is the symmetric TODO.
- Random weights (perf-representative; not a trained model).
- One setup barrier assumes fixed params; a training loop must re-barrier after each optimizer step.

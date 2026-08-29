"""Two DISJOINT SM partitions exposed as torch CUDA streams, via CUDA green contexts.

torch-native ``torch.cuda.green_contexts.GreenContext`` is not compiled into torch 2.9.1, so we drive
``cuda.bindings.driver`` directly and wrap each green-context CUstream with ``torch.cuda.ExternalStream``.

Key correctness point: a SINGLE ``cuDevSmResourceSplitByCount(nbGroups=1, minCount=comm_sms)``
yields one group of >= comm_sms SMs plus a ``remainder`` of the rest — those two are guaranteed
DISJOINT. Two independent green-context creations would each split from the full device and could
land on the same physical SMs.
"""
from __future__ import annotations

import torch
from cuda.bindings import driver as cu


def _ck(ret):
    """cuda-python returns (CUresult, *outs); assert success and return the outs."""
    err = ret[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err!r}")
    outs = ret[1:]
    return outs[0] if len(outs) == 1 else outs


class GreenCtxStreams:
    """Create a comm partition (>= comm_sms SMs) and a disjoint compute partition (the rest).

    Attributes ``comm_stream`` / ``compute_stream`` are ``torch.cuda.ExternalStream`` usable by
    NCCL and sonicmoe/QuACK kernels (both honor the current stream). ``comm_sms`` / ``compute_sms``
    report the actual (alignment-rounded) SM counts.
    """

    def __init__(self, device_index: int, comm_sms: int):
        _ck(cu.cuInit(0))
        dev = _ck(cu.cuDeviceGet(device_index))
        res = _ck(cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        self.total_sms = int(res.sm.smCount)
        align = int(res.sm.smCoscheduledAlignment) or 8
        comm_sms = max(align, ((comm_sms + align - 1) // align) * align)

        groups, _nb, remainder = _ck(cu.cuDevSmResourceSplitByCount(1, res, 0, comm_sms))
        comm_res, compute_res = groups[0], remainder
        self.comm_sms = int(comm_res.sm.smCount)
        self.compute_sms = int(compute_res.sm.smCount)

        comm_desc = _ck(cu.cuDevResourceGenerateDesc([comm_res], 1))
        compute_desc = _ck(cu.cuDevResourceGenerateDesc([compute_res], 1))
        flag = cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        self.comm_gctx = _ck(cu.cuGreenCtxCreate(comm_desc, dev, flag))
        self.compute_gctx = _ck(cu.cuGreenCtxCreate(compute_desc, dev, flag))

        nb = cu.CUstream_flags.CU_STREAM_NON_BLOCKING
        comm_cu = _ck(cu.cuGreenCtxStreamCreate(self.comm_gctx, nb, 0))
        compute_cu = _ck(cu.cuGreenCtxStreamCreate(self.compute_gctx, nb, 0))
        dev_t = torch.device("cuda", device_index)
        self.comm_stream = torch.cuda.ExternalStream(int(comm_cu), device=dev_t)
        self.compute_stream = torch.cuda.ExternalStream(int(compute_cu), device=dev_t)

    def close(self):
        """Destroy both streams and green contexts. Not automatic: ``torch.cuda.ExternalStream``
        does not own the underlying CUstream, and ``cuGreenCtxCreate`` retains the primary context
        until its green context is destroyed — call this before re-creating a partition."""
        _ck(cu.cuStreamDestroy(self.comm_stream.cuda_stream))
        _ck(cu.cuStreamDestroy(self.compute_stream.cuda_stream))
        _ck(cu.cuGreenCtxDestroy(self.comm_gctx))
        _ck(cu.cuGreenCtxDestroy(self.compute_gctx))

    def __repr__(self):
        return (f"GreenCtxStreams(total={self.total_sms} sms, "
                f"comm={self.comm_sms}, compute={self.compute_sms})")

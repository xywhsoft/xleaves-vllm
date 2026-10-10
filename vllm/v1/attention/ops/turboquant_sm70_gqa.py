# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental SM70 K4/V4 grouped-query decode with optional native FP16 tensor cores.

The Triton fallback uses SIMT on SM70; the native WMMA route uses tensor cores.
Eight query heads share one unpack of their KV tile. Accumulation, softmax,
norm correction, and partition merging remain FP32. Q/K/V and probabilities
enter tensor-core products in FP16; this route needs numerical/quality gates.
"""

import importlib.util
import math
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton


@dataclass
class _GQADecodeArena:
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    capacity: int
    captured: bool = False
    previous: "_GQADecodeArena | None" = None


_gqa_decode_arenas: dict[tuple, _GQADecodeArena] = {}


def _get_gqa_decode_workspace(q_rot, kv_cache, block_table, num_kv_splits):
    from flash_attn_v100.flash_attn_interface import (
        VALID_DECODE_PARTITION_SIZES,
        _can_cache_workspace,
        _cuda_graph_capture_active,
        _get_turboquant_decode_workspace,
        _workspace_stream_id,
    )

    if q_rot.device.type != "cuda" or not _can_cache_workspace(q_rot):
        return _get_turboquant_decode_workspace(
            q_rot, kv_cache, block_table, num_kv_splits
        )

    batch = block_table.shape[0]
    heads, dim = q_rot.shape[1:]
    max_seq = block_table.shape[1] * kv_cache.shape[1]
    per_split = (max_seq + num_kv_splits - 1) // num_kv_splits
    partition = next(
        (size for size in VALID_DECODE_PARTITION_SIZES if per_split <= size),
        None,
    )
    if partition is None:
        raise ValueError(
            f"TurboQuant Flash-V100 decode cannot cover max_seq_capacity={max_seq} "
            f"with num_kv_splits={num_kv_splits}; "
            f"largest split tile is {VALID_DECODE_PARTITION_SIZES[-1]}"
        )

    # V1 workers serialize model execution. Isolate independent CUDA streams;
    # graphs captured on one stream share its arena and must replay serially.
    stream = _workspace_stream_id(q_rot.device)
    key = (q_rot.device.index, stream, heads, dim, num_kv_splits, partition)
    arena = _gqa_decode_arenas.get(key)
    if arena is None or arena.capacity < batch:
        previous = arena
        arena = _GQADecodeArena(
            workspace=(
                torch.empty(
                    (batch, heads, num_kv_splits, dim),
                    dtype=torch.float32,
                    device=q_rot.device,
                ),
                torch.empty(
                    (batch, heads, num_kv_splits),
                    dtype=torch.float32,
                    device=q_rot.device,
                ),
                torch.empty(
                    (batch, heads, num_kv_splits),
                    dtype=torch.float32,
                    device=q_rot.device,
                ),
            ),
            capacity=batch,
        )
        if previous is not None:
            # A captured graph holds raw addresses. Keep its old arena alive
            # when a later larger batch needs different storage.
            arena.previous = previous if previous.captured else previous.previous
        _gqa_decode_arenas[key] = arena
    if _cuda_graph_capture_active():
        arena.captured = True
    return tuple(tensor[:batch] for tensor in arena.workspace), partition


@cache
def _load_gqa_native(path_string):
    path = Path(path_string)
    name = path.stem
    if name not in {
        "ornith_tq_gqa_cuda_sm70",
        "ornith_tq_gqa_wmma_sm70",
        "ornith_tq_gqa_wmma8_sm70",
        "ornith_tq_gqa_wmma8p_sm70",
        "ornith_tq_gqa_wmma8c_sm70",
        "ornith_tq_gqa_wmma8u_sm70",
        "ornith_tq_gqa_wmma8halfpre_sm70",
        "ornith_tq_gqa_wmma8aliasvalue80_sm70",
    }:
        raise ValueError("Unsupported GQA extension module name")
    if not path.is_file():
        raise ValueError(f"Missing GQA extension: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the native GQA extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def populate_cached_keyscale(kv_cache, slot_mapping, centroids):
    """Append the canonical FP32 correction after the existing KV store."""
    library = os.environ.get("VLLM_SM70_TURBOQUANT_GQA_LIBRARY", "")
    if not library:
        raise ValueError("Cached key scale requires a qualified native library")
    module = _load_gqa_native(library)
    if not hasattr(module, "populate"):
        raise ValueError("Selected GQA library cannot populate cached key scales")
    module.populate(kv_cache, slot_mapping, centroids)


@triton.jit
def _gqa_stage1(
    Q,
    KV,
    BT,
    LEN,
    CENT,
    MID,
    MAX,
    SUM,
    SQB: tl.constexpr,
    SQH: tl.constexpr,
    SCB: tl.constexpr,
    SCT: tl.constexpr,
    SCH: tl.constexpr,
    SBT: tl.constexpr,
    SMB: tl.constexpr,
    SMH: tl.constexpr,
    SMS: tl.constexpr,
    SSB: tl.constexpr,
    SSH: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLITS: tl.constexpr,
    MIN_SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
    GROUP: tl.constexpr = 8,
):
    batch, kvhead, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(LEN + batch)
    active = tl.minimum(SPLITS, tl.maximum(MIN_SPLITS, tl.cdiv(length, 1024)))
    span = tl.cdiv(length, active)
    begin = split * span
    end = tl.minimum(begin + span, length)
    if begin >= end:
        return
    head = tl.arange(0, 16)
    dim = tl.arange(0, 256)
    token = tl.arange(0, TILE)
    queries = tl.load(
        Q + batch * SQB + (kvhead * GROUP + head[:, None]) * SQH + dim[None, :],
        mask=head[:, None] < GROUP,
        other=0,
    ).to(tl.float16)
    maxima = tl.full((16,), -float("inf"), tl.float32)
    sums = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, 256), tl.float32)
    for start in range(begin, end, TILE):
        position = start + token
        valid = position < end
        blocks = tl.load(BT + batch * SBT + position // BLOCK, mask=valid, other=0).to(
            tl.int64
        )
        slot = blocks * SCB + (position % BLOCK).to(tl.int64) * SCT + kvhead * SCH
        packed = tl.load(
            KV + slot[:, None] + dim[None, :] // 2, mask=valid[:, None], other=0
        ).to(tl.int32)
        index = (packed >> ((dim[None, :] % 2) * 4)) & 15
        keys = tl.load(CENT + index)
        inverse_norm = tl.rsqrt(tl.sum(keys * keys, axis=1) + 1.0e-16)
        norm_lo = tl.load(KV + slot + 128, mask=valid, other=0).to(tl.uint16)
        norm_hi = tl.load(KV + slot + 129, mask=valid, other=0).to(tl.uint16)
        norm = (norm_lo | (norm_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        keys = (keys * (norm * inverse_norm)[:, None]).to(tl.float16)
        scores = tl.dot(queries, tl.trans(keys)) * SCALE
        scores = tl.where(valid[None, :], scores, -float("inf"))
        new_max = tl.maximum(maxima, tl.max(scores, axis=1))
        alpha = tl.exp(maxima - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        packed_values = tl.load(
            KV + slot[:, None] + 130 + dim[None, :] // 2, mask=valid[:, None], other=0
        ).to(tl.int32)
        value_index = (packed_values >> ((dim[None, :] % 2) * 4)) & 15
        scale_lo = tl.load(KV + slot + 258, mask=valid, other=0).to(tl.uint16)
        scale_hi = tl.load(KV + slot + 259, mask=valid, other=0).to(tl.uint16)
        value_scale = (
            (scale_lo | (scale_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        )
        zero_lo = tl.load(KV + slot + 260, mask=valid, other=0).to(tl.uint16)
        zero_hi = tl.load(KV + slot + 261, mask=valid, other=0).to(tl.uint16)
        value_zero = (
            (zero_lo | (zero_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        )
        values = (
            value_index.to(tl.float32) * value_scale[:, None] + value_zero[:, None]
        ).to(tl.float16)
        sums = sums * alpha + tl.sum(probabilities, axis=1)
        acc = acc * alpha[:, None] + tl.dot(probabilities.to(tl.float16), values)
        maxima = new_max
    head_global = kvhead * GROUP + head
    tl.store(
        MID + batch * SMB + head_global[:, None] * SMH + split * SMS + dim[None, :],
        acc / sums[:, None],
        mask=head[:, None] < GROUP,
    )
    stats = batch * SSB + head_global * SSH + split
    tl.store(MAX + stats, maxima, mask=head < GROUP)
    tl.store(SUM + stats, sums, mask=head < GROUP)


@triton.jit
def _gqa_stage2(
    MID,
    MAX,
    SUM,
    LEN,
    OUT,
    SMB: tl.constexpr,
    SMH: tl.constexpr,
    SMS: tl.constexpr,
    SSB: tl.constexpr,
    SSH: tl.constexpr,
    SOB: tl.constexpr,
    SOH: tl.constexpr,
    SPLITS: tl.constexpr,
    MIN_SPLITS: tl.constexpr,
):
    batch, head, dimtile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(LEN + batch)
    dim = dimtile * 32 + tl.arange(0, 32)
    if length <= 0:
        tl.store(OUT + batch * SOB + head * SOH + dim, 0.0)
        return
    active = tl.minimum(SPLITS, tl.maximum(MIN_SPLITS, tl.cdiv(length, 1024)))
    span = tl.cdiv(length, active)
    count = tl.cdiv(length, span)
    parts = tl.arange(0, SPLITS)
    valid = parts < count
    maxima = tl.load(
        MAX + batch * SSB + head * SSH + parts, mask=valid, other=-float("inf")
    )
    sums = tl.load(SUM + batch * SSB + head * SSH + parts, mask=valid, other=0.0)
    weights = tl.exp(maxima - tl.max(maxima, axis=0)) * sums
    weights = weights / tl.sum(weights, axis=0)
    values = tl.load(
        MID + batch * SMB + head * SMH + parts[:, None] * SMS + dim[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    result = tl.sum(weights[:, None] * values, axis=0)
    tl.store(OUT + batch * SOB + head * SOH + dim, result)


def gqa_turboquant_decode(
    *,
    q_rot,
    kv_cache,
    output,
    block_table,
    seq_lens,
    centroids,
    softmax_scale,
    num_kv_splits,
    tile=16,
    warps=8,
):
    if q_rot.shape[-1] != 256 or q_rot.shape[1] // kv_cache.shape[2] != 8:
        raise ValueError(
            "Experimental GQA kernel requires D256 and eight queries per KV head"
        )
    workspace, partition_size = _get_gqa_decode_workspace(
        q_rot, kv_cache, block_table, num_kv_splits
    )
    library = os.environ.get("VLLM_SM70_TURBOQUANT_GQA_LIBRARY")
    if library:
        return _load_gqa_native(library).decode(
            q_rot,
            kv_cache,
            output,
            block_table,
            seq_lens,
            *workspace,
            centroids,
            softmax_scale,
            partition_size,
            4,
            4,
            True,
        )
    mid, maxima, sums = workspace
    if output is None:
        output = torch.empty_like(q_rot, dtype=torch.float16)
    batch, heads, _ = q_rot.shape
    kvheads = kv_cache.shape[2]
    minimum = max(
        1,
        math.ceil(
            2
            * torch.cuda.get_device_properties(q_rot.device).multi_processor_count
            / (batch * kvheads)
        ),
    )
    _gqa_stage1[(batch, kvheads, num_kv_splits)](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid,
        maxima,
        sums,
        *q_rot.stride()[:2],
        *kv_cache.stride()[:3],
        block_table.stride(0),
        *mid.stride()[:3],
        *maxima.stride()[:2],
        kv_cache.shape[1],
        num_kv_splits,
        minimum,
        softmax_scale,
        tile,
        num_warps=warps,
        num_stages=1,
    )
    _gqa_stage2[(batch, heads, 8)](
        mid,
        maxima,
        sums,
        seq_lens,
        output,
        *mid.stride()[:3],
        *maxima.stride()[:2],
        *output.stride()[:2],
        num_kv_splits,
        minimum,
        num_warps=4,
    )
    return output

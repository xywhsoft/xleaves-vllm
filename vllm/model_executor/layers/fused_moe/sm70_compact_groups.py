# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental bounded compaction of grouped-GEMM metadata on the GPU."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compact_offsets(OFFSETS, IDS, NEW, EXPERTS: tl.constexpr, CAP: tl.constexpr):
    expert = tl.arange(0, EXPERTS)
    first = tl.load(OFFSETS + expert)
    last = tl.load(OFFSETS + expert + 1)
    present = first < last
    rank = tl.cumsum(present.to(tl.int32)) - 1
    count = tl.sum(present.to(tl.int32))
    total = tl.load(OFFSETS + EXPERTS)
    tl.store(IDS + rank, expert, mask=present)
    tl.store(NEW + rank, first, mask=present)
    # These stores do not overlap the active scatter destinations.
    slots = tl.arange(0, triton.next_power_of_2(CAP + 1))
    tl.store(NEW + slots, total, mask=(slots >= count) & (slots <= CAP))
    tl.store(IDS + slots, 0, mask=(slots >= count) & (slots < CAP))


@triton.jit
def _gather_ptrs(PTR, IDS, OUT, CAP: tl.constexpr, EXPERTS: tl.constexpr):
    group = tl.program_id(0) * 32 + tl.arange(0, 32)
    bank = tl.program_id(1)
    byte = tl.arange(0, 16)
    source = tl.load(IDS + group, mask=group < CAP, other=0)
    pointer = tl.load(
        PTR + bank * EXPERTS * 16 + source[:, None] * 16 + byte[None, :],
        mask=group[:, None] < CAP,
        other=0,
    )
    tl.store(
        OUT + bank * CAP * 16 + group[:, None] * 16 + byte[None, :],
        pointer,
        mask=group[:, None] < CAP,
    )


def compact_groups(offsets, pointer_banks, rows):
    experts = offsets.numel() - 1
    capacity = min(experts, rows)
    if capacity == experts:
        return offsets, pointer_banks
    if experts != 256 or rows <= 0 or pointer_banks.shape != (6, experts, 16):
        raise ValueError("Unqualified compaction descriptor")
    indices = torch.empty(capacity, device=offsets.device, dtype=torch.int32)
    compressed = torch.empty(capacity + 1, device=offsets.device, dtype=torch.int32)
    pointers = torch.empty(6, capacity, 16, device=offsets.device, dtype=torch.uint8)
    _compact_offsets[(1,)](offsets, indices, compressed, experts, capacity, num_warps=4)
    _gather_ptrs[(triton.cdiv(capacity, 32), 6)](
        pointer_banks, indices, pointers, capacity, experts, num_warps=4
    )
    return compressed, pointers


def compact_affine_projection(bank, value, offsets, pointers, index):
    output = torch.empty(value.shape[0], bank.n, device=value.device, dtype=value.dtype)
    torch.ops._C.gguf_affine_grouped_gemm_sm70_out(
        output,
        value,
        offsets,
        pointers[index].flatten(),
        pointers[index + 1].flatten(),
        bank.decoder,
        pointers.shape[1],
        bank.group,
    )
    return output

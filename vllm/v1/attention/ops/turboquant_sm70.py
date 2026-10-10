# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in native SM70 TurboQuant decode with device-side split selection."""

import importlib.util
import os
from functools import cache
from pathlib import Path


@cache
def load_adaptive_turboquant():
    path = Path(os.environ["VLLM_SM70_TURBOQUANT_ADAPTIVE_LIBRARY"])
    if not path.is_file():
        raise ValueError(f"Adaptive TurboQuant extension is missing: {path}")
    spec = importlib.util.spec_from_file_location("ornith_tq_adaptive_sm70", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the native adaptive extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adaptive_turboquant_decode(
    *,
    q_rot,
    kv_cache,
    output,
    block_table,
    seq_lens,
    centroids,
    softmax_scale,
    num_kv_splits,
):
    from flash_attn_v100.flash_attn_interface import (
        _get_turboquant_decode_workspace,
    )

    workspace, partition_size = _get_turboquant_decode_workspace(
        q_rot, kv_cache, block_table, num_kv_splits
    )
    return load_adaptive_turboquant().decode(
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

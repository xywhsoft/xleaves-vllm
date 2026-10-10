# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the qualified eight-head SM70 continuation kernel and reserve its scratch."""

import importlib.util
import os
from functools import cache
from pathlib import Path


class _ChunkedH8:
    def __init__(self, module):
        self.module = module
        self.query_step = int(getattr(module, "max_query_length", 8192))
        if self.query_step not in (4096, 8192):
            raise ValueError("H8 extension has an unsupported query tile limit")

    def reserve_workspace(self, like):
        return self.module.reserve_workspace(like)

    def prefill(self, query, key, value, output, scale, causal):
        length = query.size(1)
        if length <= self.query_step:
            return self.module.prefill(query, key, value, output, scale, causal)
        if length > 16384 or length % 64 or key.size(1) <= length:
            raise ValueError("Chunked H8 requires Q<=16384 aligned to 64 and past KV")
        # Each segment's last query sees the same last KV position as before.
        prefix = key.size(1) - length
        for begin in range(0, length, self.query_step):
            end = min(begin + self.query_step, length)
            self.module.prefill(
                query[:, begin:end],
                key[:, : prefix + end],
                value[:, : prefix + end],
                output[:, begin:end],
                scale,
                causal,
            )


@cache
def _load(path_string):
    path = Path(path_string)
    if (
        path.stem
        not in {
            "ornith_prefill_h8_sm70",
            "ornith_prefill_h8k4_sm70",
            "ornith_prefill_h8k2_sm70",
        }
        or not path.is_file()
    ):
        raise ValueError(f"Invalid Ornith H8 prefill extension: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the native H8 prefill extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def h8_prefill_module():
    if os.getenv("VLLM_SM70_ORNITH_H8_PREFILL", "0") != "1":
        return None
    path = os.getenv("VLLM_SM70_ORNITH_H8_PREFILL_LIBRARY", "")
    if not path:
        raise ValueError("The H8 prefill route requires its qualified native library")
    return _ChunkedH8(_load(path))


def reserve_h8_prefill_workspace(like):
    module = h8_prefill_module()
    if module is not None:
        module.reserve_workspace(like)

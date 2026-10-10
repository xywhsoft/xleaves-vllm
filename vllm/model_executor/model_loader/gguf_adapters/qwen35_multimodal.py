# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the Qwen3.5 vision tower without altering the native GGUF text path."""

import gguf
import numpy as np
import regex as re
import torch

from .qwen35 import Qwen35Adapter
from .qwen35_moe import Qwen35MoeAdapter

_GLOBALS = {
    "v.patch_embd.weight": "patch_embed.proj.weight",
    "v.patch_embd.weight.1": "patch_embed.proj.weight",
    "v.patch_embd.bias": "patch_embed.proj.bias",
    "v.position_embd.weight": "pos_embed.weight",
    "v.post_ln.weight": "merger.norm.weight",
    "v.post_ln.bias": "merger.norm.bias",
    "mm.0.weight": "merger.linear_fc1.weight",
    "mm.0.bias": "merger.linear_fc1.bias",
    "mm.2.weight": "merger.linear_fc2.weight",
    "mm.2.bias": "merger.linear_fc2.bias",
}
_BLOCKS = {
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ln1": "norm1",
    "ln2": "norm2",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
}
_PATCH_PARTS = ("v.patch_embd.weight", "v.patch_embd.weight.1")


def _dense_vision(tensor, dtype):
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        raw = torch.from_numpy(tensor.data.view(np.uint16).copy())
        value = raw.view(torch.bfloat16)
    elif tensor.tensor_type in (
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.F32,
    ):
        value = torch.from_numpy(tensor.data.copy())
    else:
        raise ValueError("Qwen3.5 vision GGUF currently requires floating weights")
    value = value.reshape(tuple(int(dim) for dim in tensor.shape[::-1]))
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite vision weight: {tensor.name}")
    converted = value.to(dtype)
    if not torch.isfinite(converted).all():
        raise ValueError(f"Vision weight overflows {dtype}: {tensor.name}")
    return converted


class _MultimodalAdapter(Qwen35Adapter):
    requires_mmproj = True

    def __init__(self, config, tp_size=1):
        self.vision_config = config.vision_config
        if getattr(self.vision_config, "deepstack_visual_indexes", []):
            raise ValueError("Qwen3.5 GGUF vision deepstack weights are not supported")
        if self.vision_config.temporal_patch_size != 2:
            raise ValueError("Split vision patch embedding requires temporal size 2")
        super().__init__(config.get_text_config(), tp_size=tp_size)

    def build_name_map(self, tensors):
        text = {
            raw: tensor
            for raw, tensor in tensors.items()
            if not raw.startswith(("v.", "mm."))
        }
        result = {
            raw: "language_model." + name
            for raw, name in super().build_name_map(text).items()
        }
        seen = set()
        for raw in tensors.keys() - text.keys():
            if raw in _GLOBALS:
                name = _GLOBALS[raw]
            else:
                match = re.fullmatch(r"v\.blk\.(\d+)\.(.+)\.(weight|bias)", raw)
                if (
                    match is None
                    or int(match[1]) >= self.vision_config.depth
                    or match[2] not in _BLOCKS
                ):
                    raise ValueError(f"Unmapped Qwen3.5 vision GGUF tensor: {raw}")
                name = f"blocks.{match[1]}.{_BLOCKS[match[2]]}.{match[3]}"
            result[raw] = "visual." + name
            seen.add(raw)
        required = set(_GLOBALS)
        for layer in range(self.vision_config.depth):
            for module in _BLOCKS:
                required.update(
                    f"v.blk.{layer}.{module}.{suffix}" for suffix in ("weight", "bias")
                )
        if missing := required - seen:
            raise ValueError(f"Missing Qwen3.5 vision GGUF tensors: {sorted(missing)}")
        return result

    @staticmethod
    def is_linear(name):
        return not name.startswith("visual.") and Qwen35Adapter.is_linear(name)

    def needs_dense_fallback(self, name, tensor):
        return not name.startswith("visual.") and super().needs_dense_fallback(
            name, tensor
        )

    def weights(self, tensors, name_map, dtype):
        text_map = {
            raw: name
            for raw, name in name_map.items()
            if not name.startswith("visual.")
        }
        yield from super().weights(tensors, text_map, dtype)
        patch_emitted = False
        for raw, name in name_map.items():
            if not name.startswith("visual."):
                continue
            if raw in _PATCH_PARTS:
                if patch_emitted:
                    continue
                if not all(part in name_map for part in _PATCH_PARTS):
                    raise ValueError("Both temporal patch slices are required")
                slices = [_dense_vision(tensors[part], dtype) for part in _PATCH_PARTS]
                expected = (
                    self.vision_config.hidden_size,
                    self.vision_config.in_channels,
                    self.vision_config.patch_size,
                    self.vision_config.patch_size,
                )
                if any(tuple(part.shape) != expected for part in slices):
                    raise ValueError(f"Invalid vision patch shape; expected {expected}")
                weight = torch.stack(slices, dim=2)
                patch_emitted = True
            else:
                # Vision LayerNorm uses the stored scale directly; text RMSNorm
                # restoration subtracts one and must never touch these weights.
                weight = _dense_vision(tensors[raw], dtype)
            yield name, weight


class Qwen35MultimodalAdapter(_MultimodalAdapter):
    pass


class Qwen35MoeMultimodalAdapter(_MultimodalAdapter, Qwen35MoeAdapter):
    @staticmethod
    def is_linear(name):
        return not name.startswith("visual.") and Qwen35MoeAdapter.is_linear(name)

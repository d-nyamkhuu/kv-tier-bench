"""Per-token KV bytes and per-prefix recurrent-state bytes, computed from a model's HF config.json.

These are computed values, not measurements. The first GPU run checks them against vLLM's own
`kv_offload_config_info` gauge and the startup log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "fp8": 1}


@dataclass(frozen=True)
class ModelShape:
    name: str
    num_layers: int
    full_attn_layers: int
    kv_heads: int
    head_dim: int
    # Gated DeltaNet (linear attention) layers; 0 for standard models
    linear_layers: int = 0
    lin_k_heads: int = 0
    lin_v_heads: int = 0
    lin_k_dim: int = 0
    lin_v_dim: int = 0
    conv_kernel: int = 0

    @property
    def is_hybrid(self) -> bool:
        return self.linear_layers > 0

    def kv_bytes_per_token(self, kv_dtype: str = "bfloat16", tp: int = 1) -> int:
        """KV bytes for one token across the whole instance, including replication when tp > kv_heads.

        vLLM gives each rank max(1, kv_heads // tp) heads, so with tp > kv_heads the heads are copied.
        """
        heads_stored = max(1, self.kv_heads // tp) * tp
        return self.full_attn_layers * heads_stored * self.head_dim * 2 * DTYPE_BYTES[kv_dtype]

    def kv_replication(self, tp: int) -> int:
        return max(1, self.kv_heads // tp) * tp // self.kv_heads

    def state_bytes_per_prefix(self, ssm_dtype: str = "bfloat16", conv_dtype: str = "bfloat16") -> int:
        """One linear-attention state snapshot (recurrent + conv), which every cached prefix carries."""
        if not self.is_hybrid:
            return 0
        recurrent = self.lin_v_heads * self.lin_k_dim * self.lin_v_dim * DTYPE_BYTES[ssm_dtype]
        conv_channels = 2 * self.lin_k_heads * self.lin_k_dim + self.lin_v_heads * self.lin_v_dim
        conv = conv_channels * (self.conv_kernel - 1) * DTYPE_BYTES[conv_dtype]
        return self.linear_layers * (recurrent + conv)

    def prefix_bytes(self, tokens: int, kv_dtype: str = "bfloat16", ssm_dtype: str = "bfloat16", tp: int = 1) -> int:
        return tokens * self.kv_bytes_per_token(kv_dtype, tp) + self.state_bytes_per_prefix(ssm_dtype)


def shape_from_config(name: str, config: dict) -> ModelShape:
    t = config.get("text_config", config)
    layers = t["num_hidden_layers"]
    types = t.get("layer_types") or []
    full = types.count("full_attention") if types else layers
    linear = types.count("linear_attention") if types else 0
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    return ModelShape(
        name=name,
        num_layers=layers,
        full_attn_layers=full,
        kv_heads=t["num_key_value_heads"],
        head_dim=head_dim,
        linear_layers=linear,
        lin_k_heads=t.get("linear_num_key_heads", 0) if linear else 0,
        lin_v_heads=t.get("linear_num_value_heads", 0) if linear else 0,
        lin_k_dim=t.get("linear_key_head_dim", 0) if linear else 0,
        lin_v_dim=t.get("linear_value_head_dim", 0) if linear else 0,
        conv_kernel=t.get("linear_conv_kernel_dim", 0) if linear else 0,
    )


def load_shape(name: str, config_path: str | Path) -> ModelShape:
    return shape_from_config(name, json.loads(Path(config_path).read_text()))

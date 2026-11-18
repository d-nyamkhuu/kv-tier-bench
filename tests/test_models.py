from pathlib import Path

import pytest

from kv_tier_bench.models import load_shape

FIX = Path(__file__).parents[1] / "configs" / "models"


@pytest.fixture
def hybrid():
    return load_shape("qwen3.8-27b", FIX / "qwen3.8-27b.config.json")


@pytest.fixture
def standard():
    return load_shape("qwen3-vl-32b", FIX / "qwen3-vl-32b-instruct.config.json")


def test_layer_split(hybrid, standard):
    assert (hybrid.num_layers, hybrid.full_attn_layers, hybrid.linear_layers) == (64, 16, 48)
    assert (standard.num_layers, standard.full_attn_layers, standard.linear_layers) == (64, 64, 0)


def test_kv_bytes_per_token(hybrid, standard):
    # 16 layers x 4 heads x 256 dim x K,V x bf16, and 64 x 8 x 128 x 2 x 2
    assert hybrid.kv_bytes_per_token() == 64 * 1024
    assert standard.kv_bytes_per_token() == 256 * 1024
    # same per-layer KV width, so the ratio is exactly the full-attention layer ratio
    assert standard.kv_bytes_per_token() // hybrid.kv_bytes_per_token() == 4


def test_state_snapshot(hybrid, standard):
    # per GDN layer: recurrent 48 x 128 x 128, conv (2*16*128 + 48*128) channels x (4-1)
    per_layer_bf16 = 48 * 128 * 128 * 2 + (2 * 16 * 128 + 48 * 128) * 3 * 2
    assert hybrid.state_bytes_per_prefix() == 48 * per_layer_bf16 == 78_446_592
    assert hybrid.state_bytes_per_prefix(ssm_dtype="float32") == 153_944_064  # the HF default dtype
    assert standard.state_bytes_per_prefix() == 0


def test_replication_only_above_kv_heads(hybrid, standard):
    assert hybrid.kv_replication(4) == 1
    assert hybrid.kv_replication(8) == 2  # 4 KV heads at TP=8: each head stored twice
    assert standard.kv_replication(8) == 1
    assert hybrid.kv_bytes_per_token(tp=8) == 2 * hybrid.kv_bytes_per_token()


def test_prefix_bytes(hybrid):
    assert hybrid.prefix_bytes(1000) == 1000 * 64 * 1024 + 78_446_592

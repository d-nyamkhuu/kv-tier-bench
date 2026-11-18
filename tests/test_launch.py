"""The `docker run` command for each deployment, and the vLLM args in it."""

import json
import shlex
from pathlib import Path

import pytest

from kv_tier_bench.cli import check_endpoints
from kv_tier_bench.grid import load_experiment
from kv_tier_bench.launch import CONTAINER, VLLM_IMAGE, docker_run, max_model_len
from kv_tier_bench.models import load_shape
from kv_tier_bench.tierconfig import DEPLOYMENT_FOR_TIER, kv_transfer_config

ROOT = Path(__file__).parents[1]
EXP = load_experiment(ROOT / "configs" / "experiment.yaml")
GIB = 1024**3
IP = "10.0.0.5"


def _cmd(model="qwen3.8-27b", dep="fs", **kw) -> list[str]:
    return docker_run(EXP, "s1", model, dep, IP, s3_env="/home/u/kvtb-s3.env", **kw)


def _opt(cmd: list[str], flag: str) -> list[str]:
    """Every value given to `flag` before the image name (docker's own options)."""
    head = cmd[: cmd.index(VLLM_IMAGE)]
    return [head[i + 1] for i, a in enumerate(head) if a == flag]


def _vllm(cmd: list[str]) -> list[str]:
    return cmd[cmd.index(VLLM_IMAGE) + 1 :]


def _kv_config(cmd: list[str]) -> dict:
    args = _vllm(cmd)
    return json.loads(args[args.index("--kv-transfer-config") + 1])


def test_container_settings_carried_over_from_the_kubernetes_base():
    cmd = _cmd()
    env = dict(e.split("=", 1) for e in _opt(cmd, "-e"))
    assert env["PYTHONHASHSEED"] == "0" and env["VLLM_SERVER_DEV_MODE"] == "1" and env["HF_HUB_OFFLINE"] == "1"
    assert env["VLLM_P2P_SIDE_CHANNEL_HOST"] == IP == env["VLLM_NIXL_SIDE_CHANNEL_HOST"]
    assert env["VLLM_CACHE_ROOT"] == "/models/vllm-cache" and env["HF_HOME"] == "/models"
    assert VLLM_IMAGE == "docker.io/vllm/vllm-openai:v0.33.0"
    assert _opt(cmd, "--entrypoint") == ["vllm"] and _vllm(cmd)[0] == "serve"  # not the image's default
    assert _opt(cmd, "--name") == [CONTAINER] and _opt(cmd, "--network") == ["host"]
    assert f"--host={IP}" in _vllm(cmd)  # bound to the private IP only: dev mode exposes the cache reset


def test_one_secondary_tier_per_deployment():
    st = EXP["stages"]["s1"]
    for dep in ("fs", "obj", "p2p"):
        tiers = kv_transfer_config(dep, st, "m", "b")["kv_connector_extra_config"]["secondary_tiers"]
        assert [t["type"] for t in tiers] == [dep]


def test_recompute_gpu_cpu_fs_share_the_fs_deployment():
    assert {DEPLOYMENT_FOR_TIER[t] for t in ("recompute", "gpu", "cpu", "fs")} == {"fs"}


def test_only_fs_mounts_the_disk_and_only_obj_gets_the_keys():
    assert _opt(_cmd(dep="fs"), "-v") == ["/models:/models", "/mnt/kv:/mnt/kv"]
    assert _opt(_cmd(dep="p2p"), "-v") == ["/models:/models"] and _opt(_cmd(dep="p2p"), "--env-file") == []
    obj = _cmd(dep="obj")
    assert _opt(obj, "--env-file") == ["/home/u/kvtb-s3.env"]
    store = _kv_config(obj)["kv_connector_extra_config"]["secondary_tiers"][0]["store_config"]
    assert store["endpoint_override"] == "storage.us-central1.nebius.cloud"
    with pytest.raises(ValueError):
        docker_run(EXP, "s1", "qwen3.8-27b", "obj", IP)  # no keys: fail before starting a GPU container


def test_hybrid_flags_only_on_the_hybrid_model():
    hy, vl = _vllm(_cmd("qwen3.8-27b")), _vllm(_cmd("qwen3-vl-32b"))
    assert "--mamba-ssm-cache-dtype=bfloat16" in hy and "--no-disable-hybrid-kv-cache-manager" in hy
    assert not any("mamba" in a for a in vl)


def test_cpu_tier_fits_inside_the_container_shm():
    for model in EXP["models"]:
        cmd = _cmd(model)
        cpu_bytes = _kv_config(cmd)["kv_connector_extra_config"]["cpu_bytes_to_use"]
        assert cpu_bytes < int(_opt(cmd, "--shm-size")[0].removesuffix("g")) * GIB


def test_command_survives_a_shell():
    cmd = _cmd(dep="obj")  # the JSON kv-transfer config and the {"image":0,...} limit both need quoting
    assert shlex.split(shlex.join(cmd)) == cmd


def test_one_max_length_sequence_fits_in_gpu_kv_memory():
    """vLLM refuses to start otherwise. Budget: usable HBM - bf16 weights - 3 GiB per GPU for activations."""
    for stage_name, st in EXP["stages"].items():
        gpus = st["gpus_per_instance"]
        usable = st["gpu_mem_gib"] * GIB * st["gpu_memory_utilization"] * gpus
        for model, m in EXP["models"].items():
            shape = load_shape(model, ROOT / m["config"])
            free = usable - m["params"] * 2 - 3 * GIB * gpus
            need = shape.prefix_bytes(max_model_len(st, model), tp=st["tp"])
            assert need < free, (stage_name, model, need / GIB, free / GIB)
            cap = st.get("max_prefix_tokens", {}).get(model, max(EXP["prefix_tokens"]))
            assert cap < max_model_len(st, model), (stage_name, model)  # the largest point must fit


def test_max_model_len_and_utilization_reach_the_args():
    args = _vllm(_cmd("qwen3-vl-32b"))
    assert "--max-model-len=66000" in args and "--gpu-memory-utilization=0.95" in args


def test_p2p_needs_two_distinct_routable_endpoints():
    ok = ["http://10.0.0.5:8000", "http://10.0.0.6:8000"]
    assert check_endpoints("p2p", ok) is None and check_endpoints("fs", ok[:1]) is None
    assert "needs 2" in check_endpoints("p2p", ok[:1])
    assert "distinct" in check_endpoints("p2p", ["http://localhost:8000", ok[1]])  # the reader pulls from it
    assert "distinct" in check_endpoints("p2p", [ok[0], ok[0]])
    assert "needs 1" in check_endpoints("obj", ok)

"""The `docker run` command that starts vLLM for one (stage, model, deployment) on one VM.

The vLLM args are built here from configs/experiment.yaml and tierconfig, so the sweep definition is the only
source of truth for them. scripts/sweep.sh runs the command on the writer VM, and on the peer VM over ssh for p2p.
"""

from __future__ import annotations

from .tierconfig import kv_transfer_config_json

VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.33.0"  # tag confirmed on Docker Hub 2026-11-15
CONTAINER = "kvtb-vllm"  # one per VM; sweep.sh force-removes it before every start
PORT = 8000


def max_model_len(stage: dict, model: str) -> int:
    mml = stage["max_model_len"]
    return mml.get(model, mml["default"])


def vllm_args(exp: dict, stage_name: str, model: str, deployment: str, host_ip: str) -> list[str]:
    st = exp["stages"][stage_name]
    m = exp["models"][model]
    return [
        m["hf_id"],
        f"--host={host_ip}",  # the private IP only: dev mode exposes /reset_prefix_cache
        f"--port={PORT}",
        f"--tensor-parallel-size={st['tp']}",
        f"--max-model-len={max_model_len(st, model)}",
        f"--gpu-memory-utilization={st['gpu_memory_utilization']}",
        "--kv-transfer-config",
        kv_transfer_config_json(deployment, st, model, exp["s3_bucket"]),
        *exp["common_args"],
        *m.get("extra_args", []),
    ]


def docker_run(
    exp: dict,
    stage_name: str,
    model: str,
    deployment: str,
    host_ip: str,
    s3_env: str | None = None,
    models_dir: str = "/models",
) -> list[str]:
    """argv of `docker run` for one vLLM instance. `host_ip` is the VM's private IP; `s3_env` is an env file
    with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, needed by the obj deployment only."""
    st = exp["stages"][stage_name]
    env = {
        "VLLM_SERVER_DEV_MODE": "1",  # enables POST /reset_prefix_cache
        "PYTHONHASHSEED": "0",  # identical on every peer, or no block hash matches across instances
        "HF_HOME": "/models",
        "HF_HUB_OFFLINE": "1",  # weights are pre-staged (scripts/stage-weights.sh); never download in a timed run
        "VLLM_P2P_SIDE_CHANNEL_HOST": host_ip,
        "VLLM_NIXL_SIDE_CHANNEL_HOST": host_ip,
        "UCX_PROTO_INFO": "y",  # logs the transport UCX picks for p2p
        "DO_NOT_TRACK": "1",
        # compile caches on the weights volume, so later deploys of the same model skip most of the compile
        "VLLM_CACHE_ROOT": "/models/vllm-cache",
        "TRITON_CACHE_DIR": "/models/triton-cache",
    }
    cmd = ["docker", "run", "-d", "--name", CONTAINER, "--gpus", "all", "--network", "host"]
    # the CPU tier lives in /dev/shm: give the container its own, larger than the tier (Docker's default is 64 MB)
    cmd += ["--shm-size", f"{st['cpu_tier_gib'] + 8}g", "-v", f"{models_dir}:/models"]
    if deployment == "fs":
        cmd += ["-v", f"{st['disk_mount']}:{st['disk_mount']}"]
    if deployment == "obj":
        if not s3_env:
            raise ValueError("the obj deployment needs --s3-env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        cmd += ["--env-file", s3_env]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["--entrypoint", "vllm", VLLM_IMAGE, "serve"]
    return cmd + vllm_args(exp, stage_name, model, deployment, host_ip)

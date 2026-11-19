"""The `docker run` command that starts vLLM for one (stage, model, deployment) on one VM.

The vLLM args are built here from configs/experiment.yaml and tierconfig, so the sweep definition is the only
source of truth for them. scripts/sweep.sh runs the command on the writer VM, and on the peer VM over ssh for p2p.
"""

from __future__ import annotations

from pathlib import Path

from .tierconfig import FS_MOUNT_KEY, OBJ_DEPLOYMENTS, STORE_DEPLOYMENTS, kv_transfer_config_json

# sitecustomize hook for NIXL settings vLLM does not expose; sweep.sh copies it to the same path on the peer VM
NIXL_PATCH_DIR = Path(__file__).resolve().parents[2] / "scripts" / "nixl_patch"

VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.34.0"  # released 2026-11-16; tag confirmed on Docker Hub 2026-11-18
CONTAINER = "kvtb-vllm"  # one per VM; sweep.sh force-removes it before every start
PORT = 8000
MOONCAKE_CONFIG = "/kvtb/mooncake.json"  # where the store deployment's Mooncake config is mounted in the container


def max_model_len(stage: dict, model: str) -> int:
    mml = stage["max_model_len"]
    return mml.get(model, mml["default"])


def bucket_for(exp: dict, deployment: str) -> str:
    """The bucket of an obj deployment: one per Object Storage class (a bucket's default class applies to every new
    object in it). Other deployments use none; the results backup uses s3_bucket."""
    return exp.get("obj_buckets", {}).get(deployment, exp["s3_bucket"])


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
        kv_transfer_config_json(deployment, st, model, bucket_for(exp, deployment)),
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
    mooncake_config: str | None = None,
) -> list[str]:
    """argv of `docker run` for one vLLM instance. `host_ip` is the VM's private IP; `s3_env` is an env file
    with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, needed by the obj deployments only. `mooncake_config` is the
    host path of the store deployment's Mooncake config (`kvtb mooncake-config` writes it)."""
    st = exp["stages"][stage_name]
    env = {
        "VLLM_SERVER_DEV_MODE": "1",  # enables POST /reset_prefix_cache
        "PYTHONHASHSEED": "0",  # identical on every peer, or no block hash matches across instances
        "HF_HOME": "/models",
        "HF_HUB_OFFLINE": "1",  # weights are pre-staged (scripts/stage-weights.sh); never download in a timed run
        "VLLM_P2P_SIDE_CHANNEL_HOST": host_ip,
        "VLLM_NIXL_SIDE_CHANNEL_HOST": host_ip,
        # p2p runs over UCX on TCP: pin it to the VPC NIC, so it never advertises docker0 or lo to the peer
        "UCX_NET_DEVICES": "eth0",
        "UCX_PROTO_INFO": "y",  # logs the transport UCX picks for p2p
        "DO_NOT_TRACK": "1",
        # compile caches on the weights volume, so later deploys of the same model skip most of the compile.
        # v0.34.0 runs qwen3.8-27b uncompiled (compilation mode NONE), so only its Triton kernels land here.
        # One directory per VM: in s3 /models is a shared filesystem, and A and B start the same model at once
        # (Nebius gives no full POSIX guarantees for a file open on two VMs).
        "VLLM_CACHE_ROOT": f"/models/cache/{host_ip}/vllm",
        "TRITON_CACHE_DIR": f"/models/cache/{host_ip}/triton",
        # FlashInfer's autotune winners (~33 s of tuning per start); the default is inside the container
        "FLASHINFER_AUTOTUNE_CACHE_DIR": f"/models/cache/{host_ip}/flashinfer-autotune",
    }
    cmd = ["docker", "run", "-d", "--name", CONTAINER, "--gpus", "all", "--network", "host"]
    # the CPU tier lives in /dev/shm: give the container its own, larger than the tier (Docker's default is 64 MB)
    cmd += ["--shm-size", f"{st['cpu_tier_gib'] + 8}g", "-v", f"{models_dir}:/models"]
    if deployment in FS_MOUNT_KEY:  # the fs tier's directory, on the VM's disk or the shared filesystem
        mount = st[FS_MOUNT_KEY[deployment]]
        cmd += ["-v", f"{mount}:{mount}"]
    if deployment in STORE_DEPLOYMENTS:
        if not mooncake_config:
            raise ValueError(f"the {deployment} deployment needs --mooncake-config (kvtb mooncake-config writes it)")
        cmd += ["-v", f"{mooncake_config}:{MOONCAKE_CONFIG}:ro"]
        env |= {
            "MOONCAKE_CONFIG_PATH": MOONCAKE_CONFIG,
            # the address the store VM reaches this requester at; with --network host it could pick docker0
            "MOONCAKE_REQUESTER_LOCAL_HOSTNAME": host_ip,
            "VLLM_MOONCAKE_STORE_TIER_LOG": "1",  # logs memory vs disk hits per batch
            "VLLM_MOONCAKE_LOAD_RECV_THREADS": str(st["store"].get("load_recv_threads", 1)),
        }
        env |= {k: str(v) for k, v in st["store"].get("env", {}).items()}  # Mooncake TCP tuning
    patch_env = {}  # settings for scripts/nixl_patch
    if deployment == "p2p" and (nixl := st.get("p2p_nixl")):
        patch_env["KVTB_NIXL_UCX_PARAMS"] = f"split_batch_size={nixl['split_batch_size']}"
        if nixl.get("skip_desc_merge"):
            patch_env["KVTB_NIXL_SKIP_DESC_MERGE"] = "1"
        if nixl.get("stripes", 1) > 1:
            patch_env["KVTB_NIXL_STRIPES"] = str(nixl["stripes"])
        if (delay := nixl.get("pthr_delay_us")) is not None:
            patch_env["KVTB_NIXL_PTHR_DELAY_US"] = str(delay)
    if deployment in OBJ_DEPLOYMENTS:
        if not s3_env:
            raise ValueError(f"the {deployment} deployment needs --s3-env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        cmd += ["--env-file", s3_env]
        if major := st.get("obj_nixl_cuda_major"):
            patch_env["KVTB_NIXL_CUDA_MAJOR"] = str(major)
            # that build's AWS SDK looks for CA certs at the Red Hat path; without them every request fails (curl 77)
            cmd += ["-v", "/etc/ssl/certs/ca-certificates.crt:/etc/pki/tls/certs/ca-bundle.crt:ro"]
    if patch_env:
        cmd += ["-v", f"{NIXL_PATCH_DIR}:/kvtb-patch:ro"]
        env |= {"PYTHONPATH": "/kvtb-patch"} | patch_env
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["--entrypoint", "vllm", VLLM_IMAGE, "serve"]
    return cmd + vllm_args(exp, stage_name, model, deployment, host_ip)

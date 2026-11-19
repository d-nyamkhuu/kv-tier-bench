"""The vLLM `--kv-transfer-config` for each deployment, using the v0.34.0 schema (docs/features/kv_offloading_usage.md).

Deployments per model: fs (measures recompute, gpu, cpu and fs, on the VM's network SSD disk), fs-shared (the fs
tier on a Nebius shared filesystem, s2 only), obj (Object Storage, Enhanced Throughput class), obj-std (Standard class, its own
bucket), p2p, and store (s3: MooncakeStoreConnector, with the KV in a CPU-only VM's memory). The storage deployments
are separate because all of them report cached tokens as source "disk".
"""

from __future__ import annotations

import json

from . import constants as C

DEPLOYMENT_FOR_TIER = {
    "recompute": "fs",
    "gpu": "fs",
    "cpu": "fs",
    "fs": "fs",
    "fs-shared": "fs-shared",
    "obj": "obj",
    "obj-std": "obj-std",
    "p2p": "p2p",
    "store": "store",
    "store-p2p": "store-p2p",
}
# Stages whose baseline tiers (recompute, gpu, cpu) are measured on another deployment than fs. s3 has no KV disk and
# runs on one GPU VM, so they run on `base`: the CPU tier alone, no secondary tier.
BASE_DEPLOYMENT = {"s3": "base"}
BASE_TIERS = ("recompute", "gpu", "cpu")
# vLLM's fs tier on two kinds of storage, one deployment each: fs = the VM's network SSD disk, fs-shared = a Nebius
# shared filesystem (virtiofs). The stage key holding each one's mount point:
FS_MOUNT_KEY = {"fs": "disk_mount", "fs-shared": "shared_fs_mount"}
FS_DEPLOYMENTS = tuple(FS_MOUNT_KEY)
# Object Storage classes, one deployment (and bucket) each: obj = Enhanced Throughput, obj-std = Standard
OBJ_DEPLOYMENTS = ("obj", "obj-std")
# tiers that persist in vLLM's secondary storage and survive a reset_prefix_cache(reset_external=True)
STORAGE_TIERS = (*FS_DEPLOYMENTS, *OBJ_DEPLOYMENTS)
# MooncakeStoreConnector, its own connector rather than a tier of the OffloadingConnector, so no CPU tier:
#   store      standalone-store: vLLM is a pure requester; the pool is a mooncake_client on the CPU-only store VM
#   store-p2p  Mooncake as p2p: the writer runs embedded and holds the pool in its own RAM, and a second vLLM
#              instance (a pure requester) reads from it, as the p2p tier pulls from the writer's CPU tier
STORE_DEPLOYMENTS = ("store", "store-p2p")
# deployments whose reads go to the second instance (VM B), which loads what the writer stored
READER_DEPLOYMENTS = ("p2p", "store-p2p")
DEPLOYMENTS = ("base", *FS_DEPLOYMENTS, *OBJ_DEPLOYMENTS, "p2p", *STORE_DEPLOYMENTS)
MOONCAKE_MASTER_PORT = 50051  # mooncake_master's default
GIB = 1024**3


def deployment_for(stage: str, tier: str) -> str:
    """The deployment that measures `tier` in `stage`."""
    if tier in BASE_TIERS and stage in BASE_DEPLOYMENT:
        return BASE_DEPLOYMENT[stage]
    return DEPLOYMENT_FOR_TIER[tier]


def mooncake_config(stage: dict, store_ip: str, holder: bool = False) -> dict:
    """The MOONCAKE_CONFIG_PATH file of a store deployment (vLLM docs, MooncakeStoreConnector usage). `store_ip`
    runs mooncake_master. `holder`: this vLLM instance holds the pool itself (embedded mode, store-p2p's writer);
    otherwise it is a pure requester (standalone-store) and the pool is elsewhere."""
    st = stage["store"]
    return {
        "mode": "embedded" if holder else "standalone-store",
        "metadata_server": st.get("metadata_server", "P2PHANDSHAKE"),
        "master_server_address": f"{store_ip}:{st.get('master_port', MOONCAKE_MASTER_PORT)}",
        # embedded: the pool in this process's RAM; standalone-store: none, the external mooncake_client owns it
        "global_segment_size": st["holder_pool_gib"] * GIB if holder else 0,
        "local_buffer_size": st["local_buffer_size"],
        "protocol": st.get("protocol", "tcp"),  # no InfiniBand on the RTX PRO 6000 VMs
        "device_name": st.get("device_name", ""),
        "enable_offload": False,
        "tenant_id": st.get("tenant_id", "default"),
    }


def secondary_tiers(deployment: str, stage: dict, model: str, bucket: str) -> list[dict]:
    if deployment == "base":
        return []
    if deployment in FS_DEPLOYMENTS:
        return [
            {
                "type": "fs",
                "root_dir": stage[FS_MOUNT_KEY[deployment]],
                "n_read_threads": 16,
                "n_write_threads": 16,
                "locality": "LOCAL",
            }
        ]
    if deployment in OBJ_DEPLOYMENTS:
        # credentials: empty here, so the NIXL OBJ plugin uses the AWS SDK chain (AWS_* env from a Secret)
        return [
            {
                "type": "obj",
                "store_config": {
                    "bucket": bucket,
                    "endpoint_override": stage["s3_endpoint"],
                    "scheme": "https",
                    "region": stage["s3_region"],
                },
                "prefix": f"{model}/",
                "io_threads": 16,
                "locality": "REMOTE",
            }
        ]
    if deployment == "p2p":
        # host comes from VLLM_P2P_SIDE_CHANNEL_HOST (the VM private IP), set in launch.docker_run
        tier = {"type": "p2p", "port": C.P2P_PORT}
        # 0 is meaningful (no dedicated UCX workers, the agent's own progress thread only): vLLM's default is 4
        if (threads := stage.get("p2p_nixl", {}).get("num_threads")) is not None:
            tier["num_threads"] = threads
        return [tier]
    raise ValueError(f"unknown deployment {deployment!r}")


def kv_transfer_config(deployment: str, stage: dict, model: str, bucket: str) -> dict:
    if deployment in STORE_DEPLOYMENTS:  # the rest of its settings are in the Mooncake config file
        return {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": int(stage["cpu_tier_gib"] * GIB),
            "secondary_tiers": secondary_tiers(deployment, stage, model, bucket),
        },
    }


def kv_transfer_config_json(*args, **kwargs) -> str:
    return json.dumps(kv_transfer_config(*args, **kwargs), separators=(",", ":"))

"""The vLLM `--kv-transfer-config` for each deployment, using the v0.34.0 schema (docs/features/kv_offloading_usage.md).

Deployments per model: fs (measures recompute, gpu, cpu and fs), obj, and p2p. fs and obj are
separate because both report cached tokens as source "disk".
"""

from __future__ import annotations

import json

from . import constants as C

DEPLOYMENT_FOR_TIER = {
    "recompute": "fs",
    "gpu": "fs",
    "cpu": "fs",
    "fs": "fs",
    "obj": "obj",
    "p2p": "p2p",
}

GIB = 1024**3


def secondary_tiers(deployment: str, stage: dict, model: str, bucket: str) -> list[dict]:
    if deployment == "fs":
        return [
            {
                "type": "fs",
                "root_dir": stage["disk_mount"],
                "n_read_threads": 16,
                "n_write_threads": 16,
                "locality": "LOCAL",
            }
        ]
    if deployment == "obj":
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
        if threads := stage.get("p2p_nixl", {}).get("num_threads"):
            tier["num_threads"] = threads
        return [tier]
    raise ValueError(f"unknown deployment {deployment!r}")


def kv_transfer_config(deployment: str, stage: dict, model: str, bucket: str) -> dict:
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

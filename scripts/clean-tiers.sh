#!/usr/bin/env bash
# vLLM's fs and obj tiers never evict. After each model, delete what that model wrote, so the disk and bucket stay
# bounded and the next model starts empty. Runs on the writer VM, which holds the disk tier.
#   ./scripts/clean-tiers.sh s1 qwen3.8-27b
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?stage}; MODEL=${2:?model}
S3_ENV=${S3_ENV:-$HOME/kvtb-s3.env}
read -r ENDPOINT HF_DIR BUCKETS < <(uv run -q python -c "
from kv_tier_bench.grid import load_experiment
from kv_tier_bench.launch import bucket_for
from kv_tier_bench.tierconfig import OBJ_DEPLOYMENTS
e = load_experiment('configs/experiment.yaml'); st = e['stages']['$STAGE']
buckets = sorted({bucket_for(e, d) for d in OBJ_DEPLOYMENTS})
print(st['s3_endpoint'], e['models']['$MODEL']['hf_id'].replace('/', '_'), *buckets)")
# every fs tier directory the stage uses: the VM's disk, and the shared filesystem if the stage has one
DIRS=$(uv run -q python -c "
from kv_tier_bench.grid import load_experiment
from kv_tier_bench.tierconfig import FS_MOUNT_KEY
st = load_experiment('configs/experiment.yaml')['stages']['$STAGE']
print(' '.join(st[k] for k in FS_MOUNT_KEY.values() if k in st))")

# fs tier: <root_dir>/<model path with / as _>_<digest>[_r<rank>]/ (vLLM fs-tier layout). The model path is the HF
# id (Qwen_Qwen3-8B_...) or, when vLLM resolves it to the pre-staged snapshot under HF_HOME=/models, that path
# (_models_hub_models--Qwen--Qwen3-8B_snapshots_<sha>_...; seen in s2, where the HF-id glob alone cleaned nothing).
# The vLLM container wrote these files as root, so a container deletes them too.
HUB_DIR="*models--${HF_DIR/_/--}_snapshots"
for DISK in $DIRS; do
  [[ -d $DISK ]] || { echo "warning: $DISK is not mounted; not cleaned" >&2; continue; }
  docker run --rm -v "$DISK:/mnt/kv" busybox:1.37 \
    sh -c "du -sh /mnt/kv/${HF_DIR}_* /mnt/kv/${HUB_DIR}_* 2>/dev/null; rm -rf /mnt/kv/${HF_DIR}_* /mnt/kv/${HUB_DIR}_*"
done

# obj tiers: everything under the model's key prefix (tierconfig: prefix "<model>/"), in every class's bucket
if [[ -f $S3_ENV ]]; then
  for BUCKET in $BUCKETS; do
    docker run --rm -e HOME=/tmp --env-file "$S3_ENV" amazon/aws-cli:2.15.0 s3 rm "s3://$BUCKET/$MODEL/" \
      --recursive --only-show-errors --endpoint-url "https://$ENDPOINT"
  done
else
  echo "warning: $S3_ENV not found; obj tier not cleaned" >&2
fi
echo "cleaned fs and obj tiers for $MODEL"
# the store (s3): sweep.sh restarts it, empty, before each model's store deployment; nothing to delete here

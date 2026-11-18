#!/usr/bin/env bash
# vLLM's fs and obj tiers never evict. After each model, delete what that model wrote, so the disk and bucket stay
# bounded and the next model starts empty. Runs on the writer VM, which holds the disk tier.
#   ./scripts/clean-tiers.sh s1 qwen3.8-27b
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?stage}; MODEL=${2:?model}
S3_ENV=${S3_ENV:-$HOME/kvtb-s3.env}
read -r BUCKET ENDPOINT DISK HF_DIR < <(uv run -q python -c "
from kv_tier_bench.grid import load_experiment
e = load_experiment('configs/experiment.yaml'); st = e['stages']['$STAGE']
print(e['s3_bucket'], st['s3_endpoint'], st['disk_mount'], e['models']['$MODEL']['hf_id'].replace('/', '_'))")

# fs tier: <root_dir>/<model with / as _>_<digest>[_r<rank>]/ (vLLM fs-tier layout). The vLLM container wrote these
# files as root, so a container deletes them too.
docker run --rm -v "$DISK:/mnt/kv" busybox:1.37 \
  sh -c "du -sh /mnt/kv/${HF_DIR}_* 2>/dev/null; rm -rf /mnt/kv/${HF_DIR}_*"

# obj tier: everything under the model's key prefix (tierconfig: prefix "<model>/")
if [[ -f $S3_ENV ]]; then
  docker run --rm -e HOME=/tmp --env-file "$S3_ENV" amazon/aws-cli:2.15.0 s3 rm "s3://$BUCKET/$MODEL/" --recursive \
    --only-show-errors --endpoint-url "https://$ENDPOINT"
else
  echo "warning: $S3_ENV not found; obj tier not cleaned" >&2
fi
echo "cleaned fs and obj tiers for $MODEL"

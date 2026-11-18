#!/usr/bin/env bash
# Runs a whole stage from the writer VM ("VM A"), one deployment at a time:
#   start vLLM in Docker -> wait until it serves -> `kvtb run` -> stop vLLM; clean the fs and obj tiers after each
#   model. p2p also starts vLLM on VM B over ssh.
# Run it inside tmux, so a dropped ssh session doesn't stop it. Rerun the same command after a failure or a
# preemption: finished deployments and points are skipped.
#
#   PEER=ubuntu@10.0.0.6 ./scripts/sweep.sh s1 [model ...]
#   DEPLOYS="fs obj" ./scripts/sweep.sh s1     # no second VM needed without p2p
#
# Before the first run: scripts/stage-weights.sh, the disk tier mounted at /mnt/kv, and ~/kvtb-s3.env (docs/plan.md).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
STAGE=${1:?stage (s1)}; shift
MODELS=${*:-"qwen3.8-27b qwen3-vl-32b"}
DEPLOYS=${DEPLOYS:-$(uv run -q kvtb deployments --stage "$STAGE")}
S3_ENV=${S3_ENV:-$HOME/kvtb-s3.env}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-2400}
RESULTS=results/$STAGE.jsonl
read -r BUCKET ENDPOINT < <(uv run -q python -c "
from kv_tier_bench.grid import load_experiment
e = load_experiment('configs/experiment.yaml'); print(e['s3_bucket'], e['stages']['$STAGE']['s3_endpoint'])")
# the results are copied to the bucket after every deployment, and restored from it on a fresh VM
BACKUP=s3://$BUCKET/results/$STAGE.jsonl

echo "self: $SELF_IP  peer: ${PEER:-none}  stage: $STAGE  models: $MODELS  deployments: $DEPLOYS  backup: $BACKUP"
if [[ " $DEPLOYS " == *" p2p "* && -z $PEER ]]; then
  echo "!! p2p needs the second VM: set PEER, or run DEPLOYS=\"fs obj\"" >&2; exit 1
fi
[[ -f $S3_ENV ]] || { echo "!! $S3_ENV not found (object-store keys: obj tier, cleanup, backup)" >&2; exit 1; }

aws() {  # the AWS CLI in a container, as this user, so files it writes into results/ stay appendable
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --env-file "$S3_ENV" -v "$PWD/results:/r" \
    amazon/aws-cli:2.15.0 "$@" --only-show-errors --endpoint-url "https://$ENDPOINT"
}

start() {  # start <self|peer> <model> <deploy> <ip>: replace whatever container is there, then start this one
  local cmd
  cmd=$(uv run -q kvtb serve-cmd --stage "$STAGE" --model "$2" --deploy "$3" --host-ip "$4" --s3-env "$S3_ENV")
  on "$1" "docker rm -f $CONTAINER >/dev/null 2>&1 || true; $cmd >/dev/null"
}

wait_ready() {  # wait_ready <self|peer> <ip>: until vLLM serves; fail fast if its container exits
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  until curl -sf --max-time 5 "http://$2:8000/v1/models" >/dev/null; do
    if [[ $(on "$1" "docker inspect -f '{{.State.Running}}' $CONTAINER" 2>/dev/null) != true ]]; then
      on "$1" "docker logs --tail 40 $CONTAINER" >&2 || true
      echo "!! vLLM exited on $1 ($2)" >&2; return 1
    fi
    (( SECONDS < deadline )) || { echo "!! vLLM not serving on $1 ($2) after ${READY_TIMEOUT_S}s" >&2; return 1; }
    sleep 10
  done
}

stop() { on "$1" "docker rm -f $CONTAINER >/dev/null"; }

backup() { [[ ! -f $RESULTS ]] || aws s3 cp "/r/$STAGE.jsonl" "$BACKUP" || echo "!! backup to $BACKUP failed" >&2; }

# A fresh VM (preempted and replaced) has no local results. Restore them first: otherwise the first backup would
# overwrite the bucket's copy with a shorter file. Only a confirmed "no such key" means this is a fresh run.
mkdir -p results
if [[ ! -f $RESULTS ]]; then
  if out=$(aws s3 cp "$BACKUP" "/r/$STAGE.jsonl" 2>&1); then
    echo "restored $RESULTS from $BACKUP ($(wc -l < "$RESULTS") rows)"
  elif [[ $out == *"(404)"* || $out == *"does not exist"* ]]; then
    echo "no results at $BACKUP yet: a fresh run"
  else
    echo "!! could not check $BACKUP for earlier results: $out" >&2; exit 1
  fi
fi

for model in $MODELS; do
  for dep in $DEPLOYS; do
    run="$STAGE/$model/$dep"
    left=$(uv run -q kvtb run --stage "$STAGE" --model "$model" --deploy "$dep" --results "$RESULTS" --dry-run 2>&1 || true)
    if [[ $left == "0 of "* ]]; then echo "== $run: already done"; continue; fi
    echo "== $run: start vLLM ($left)"
    endpoints="http://$SELF_IP:8000"
    start self "$model" "$dep" "$SELF_IP"
    if [[ $dep == p2p ]]; then
      start peer "$model" "$dep" "$PEER_IP"
      endpoints+=",http://$PEER_IP:8000"
    fi
    wait_ready self "$SELF_IP"
    if [[ $dep == p2p ]]; then wait_ready peer "$PEER_IP"; fi

    echo "== $run: measure"
    if ! uv run -q kvtb run --stage "$STAGE" --model "$model" --deploy "$dep" --results "$RESULTS" \
         --endpoints "$endpoints"; then
      backup
      echo "!! $run failed. vLLM is left running for debugging (docker logs $CONTAINER). Fix the cause, then" \
           "rerun the same command." >&2
      exit 1
    fi
    backup

    echo "== $run: stop vLLM"
    stop self
    if [[ $dep == p2p ]]; then stop peer; fi
  done
  ./scripts/clean-tiers.sh "$STAGE" "$model"
done

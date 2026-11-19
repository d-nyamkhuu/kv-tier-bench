#!/usr/bin/env bash
# Runs a whole stage from the writer VM ("VM A"), one deployment at a time:
#   start vLLM in Docker -> wait until it serves -> `kvtb run` -> stop vLLM; clean the fs and obj tiers after each
#   model. p2p also starts vLLM on VM B over ssh. p2p runs first, for every model, so VM B can be stopped as soon as
#   the script says so; the other deployments need VM A only.
# Run it inside tmux, so a dropped ssh session doesn't stop it. Rerun the same command after a failure or a
# preemption: finished deployments and points are skipped.
#
#   PEER=ubuntu@10.0.0.6 ./scripts/sweep.sh s1 [model ...]
#   DEPLOYS="fs obj" ./scripts/sweep.sh s1     # no second VM needed without p2p
#   DEPLOYS="base store" STORE=kvtb@10.0.0.7 ./scripts/sweep.sh s3 qwen3-8b   # s3 on VM A and the store VM
#   DEPLOYS="store-p2p p2p" PEER=kvtb@10.0.0.6 ./scripts/sweep.sh s3 qwen3-8b  # then both p2p kinds with VM B
#
# Before the first run: scripts/stage-weights.sh (or the weights filesystem from scripts/vms.sh), the disk tier mounted at
# /mnt/kv for fs, and ~/kvtb-s3.env for obj and the results backup (docs/plan.md). Without obj, the backup is
# optional: a preempted VM is stopped, not deleted, so its results survive on its boot disk.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
STAGE=${1:?stage (s1)}; shift
MODELS=${*:-$(uv run -q kvtb models --stage "$STAGE")}
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
if [[ " $DEPLOYS " == *" p2p "* || " $DEPLOYS " == *" store-p2p "* ]] && [[ -z $PEER ]]; then
  echo "!! p2p and store-p2p need the second VM: set PEER, or leave them out of DEPLOYS" >&2; exit 1
fi
two_vms() { [[ $1 == p2p || $1 == store-p2p ]]; }  # deployments that also run vLLM on VM B
STORE=${STORE:-}
STORE_IP=${STORE_IP:-${STORE#*@}}
if [[ " $DEPLOYS " == *" store "* && -z $STORE ]]; then
  echo "!! the store deployment needs the store VM: set STORE (ssh destination of VM C)" >&2; exit 1
fi
MOONCAKE_CONFIG=$HOME/kvtb-mooncake.json
if [[ ! -f $S3_ENV ]]; then
  if [[ " $DEPLOYS " == *" obj"* ]]; then
    echo "!! $S3_ENV not found (object-store keys: obj tier, cleanup, backup)" >&2; exit 1
  fi
  echo "warning: $S3_ENV not found; results are not backed up to the bucket" >&2
fi

aws() {  # the AWS CLI in a container, as this user, so files it writes into results/ stay appendable
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --env-file "$S3_ENV" -v "$PWD/results:/r" \
    amazon/aws-cli:2.15.0 "$@" --only-show-errors --endpoint-url "https://$ENDPOINT"
}

start() {  # start <self|peer> <model> <deploy> <ip>: replace whatever container is there, then start this one
  local cmd
  cmd=$(uv run -q kvtb serve-cmd --stage "$STAGE" --model "$2" --deploy "$3" --host-ip "$4" --s3-env "$S3_ENV" \
    --mooncake-config "$MOONCAKE_CONFIG")
  if [[ $1 == peer ]]; then  # the p2p NIXL patch is mounted from this repo's path, so the peer needs it there too
    peer "mkdir -p '$PWD/scripts'" && scp -q -o BatchMode=yes -r scripts/nixl_patch "$PEER:$PWD/scripts/"
  fi
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

backup() {
  [[ -f $S3_ENV && -f $RESULTS ]] || return 0
  aws s3 cp "/r/$STAGE.jsonl" "$BACKUP" || echo "!! backup to $BACKUP failed" >&2
}

# A fresh VM (preempted and replaced) has no local results. Restore them first: otherwise the first backup would
# overwrite the bucket's copy with a shorter file. Only a confirmed "no such key" means this is a fresh run.
mkdir -p results
if [[ ! -f $RESULTS && -f $S3_ENV ]]; then
  if out=$(aws s3 cp "$BACKUP" "/r/$STAGE.jsonl" 2>&1); then
    echo "restored $RESULTS from $BACKUP ($(wc -l < "$RESULTS") rows)"
  elif [[ $out == *"(404)"* || $out == *"does not exist"* ]]; then
    echo "no results at $BACKUP yet: a fresh run"
  else
    echo "!! could not check $BACKUP for earlier results: $out" >&2; exit 1
  fi
fi

run_deploy() {  # run_deploy <model> <deploy>: one deployment, from vLLM start to stop
  local model=$1 dep=$2 run="$STAGE/$1/$2" left endpoints
  left=$(uv run -q kvtb run --stage "$STAGE" --model "$model" --deploy "$dep" --results "$RESULTS" --dry-run 2>&1 || true)
  if [[ $left == "0 of "* ]]; then echo "== $run: already done"; return; fi
  echo "== $run: start vLLM ($left)"
  if [[ $dep == store ]]; then  # an empty pool for every model, and the config that points vLLM at it
    STORE=$STORE ./scripts/store-up.sh
    uv run -q kvtb mooncake-config --stage "$STAGE" --store-ip "$STORE_IP" > "$MOONCAKE_CONFIG"
  elif [[ $dep == store-p2p ]]; then  # the master on this VM; this vLLM holds the pool, VM B's only requests
    STORE=$USER@$SELF_IP MASTER_ONLY=1 ./scripts/store-up.sh
    uv run -q kvtb mooncake-config --stage "$STAGE" --store-ip "$SELF_IP" --holder > "$MOONCAKE_CONFIG"
    uv run -q kvtb mooncake-config --stage "$STAGE" --store-ip "$SELF_IP" | peer "cat > '$MOONCAKE_CONFIG'"
  fi
  endpoints="http://$SELF_IP:8000"
  start self "$model" "$dep" "$SELF_IP"
  if two_vms "$dep"; then
    start peer "$model" "$dep" "$PEER_IP"
    endpoints+=",http://$PEER_IP:8000"
  fi
  wait_ready self "$SELF_IP"
  if two_vms "$dep"; then wait_ready peer "$PEER_IP"; fi

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
  if two_vms "$dep"; then stop peer; fi
}

# p2p first, for every model: it is the only deployment that needs VM B
if [[ " $DEPLOYS " == *" p2p "* ]]; then
  for model in $MODELS; do run_deploy "$model" p2p; done
  echo "== p2p done for every model: VM B ($PEER) is no longer needed and can be stopped"
fi
rest=$(for d in $DEPLOYS; do [[ $d == p2p ]] || echo -n "$d "; done)
if [[ -n $rest ]]; then
  for model in $MODELS; do
    for dep in $rest; do run_deploy "$model" "$dep"; done
    ./scripts/clean-tiers.sh "$STAGE" "$model"  # p2p writes neither the disk nor a bucket
  done
fi

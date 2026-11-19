#!/usr/bin/env bash
# Starts the s3 store on the CPU-only VM ("VM C") over ssh: mooncake_master (metadata and eviction) and
# mooncake_client, which owns the memory pool. vLLM on VM A is a pure requester (MooncakeStoreConnector,
# standalone-store mode; tierconfig.mooncake_config). Runs on VM A, from the repo directory.
#   STORE=kvtb@10.0.0.7 ./scripts/store-up.sh          # install if needed, then (re)start both: an empty pool
#   STORE=kvtb@10.0.0.7 ./scripts/store-up.sh stop
#   STORE=kvtb@10.0.0.7 ./scripts/store-up.sh status
#   STORE=kvtb@<A> MASTER_ONLY=1 ./scripts/store-up.sh  # store-p2p: the master only; VM A's vLLM holds the pool
# The Mooncake version is the one inside the vLLM image, so the requester and the store speak the same protocol.
# Processes are stopped by their pid files, never by a pattern match.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${STAGE:-s3}
STORE=${STORE:?set STORE to the ssh destination of the store VM, e.g. kvtb@10.0.0.7}
STORE_IP=${STORE_IP:-${STORE#*@}}
ACTION=${1:-start}
MASTER_ONLY=${MASTER_ONLY:-0}
store() { ssh -o BatchMode=yes "$STORE" "$@"; }

read -r POOL_GIB LEASE_MS PROTOCOL METADATA TENANT PORT IMAGE MC_ENV < <(uv run -q python -c "
from kv_tier_bench.grid import load_experiment
from kv_tier_bench.launch import VLLM_IMAGE
from kv_tier_bench.tierconfig import MOONCAKE_MASTER_PORT
s = load_experiment('configs/experiment.yaml')['stages']['$STAGE']['store']
print(s['pool_gib'], s.get('lease_ttl_ms', 10000), s.get('protocol', 'tcp'), s.get('metadata_server', 'P2PHANDSHAKE'), s.get('tenant_id', 'default'),
      s.get('master_port', MOONCAKE_MASTER_PORT), VLLM_IMAGE, *(f'{k}={v}' for k, v in s.get('env', {}).items()))")

stop_all() {
  store 'for p in client master; do
    f=~/mooncake/$p.pid
    if [[ -f $f ]] && kill "$(cat "$f")" 2>/dev/null; then
      for _ in $(seq 20); do kill -0 "$(cat "$f")" 2>/dev/null || break; sleep 0.5; done
      echo "stopped $p"
    fi
    rm -f "$f"
  done'
}

case $ACTION in
  stop) stop_all; exit ;;
  status) store 'for p in master client; do f=~/mooncake/$p.pid; if [[ -f $f ]] && kill -0 "$(cat "$f")" 2>/dev/null;
                 then echo "$p: running"; else echo "$p: not running"; fi; done; tail -n 3 ~/mooncake/*.log'; exit ;;
  start) ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac

# the Mooncake package vLLM's image ships (day-1 check: if it has none, the store deployment cannot run as is)
if ! PKG=$(docker run --rm --entrypoint python3 "$IMAGE" -c '
import importlib.metadata as m
d = next(d for d in m.distributions() if (d.metadata["Name"] or "").lower().startswith("mooncake"))
print(d.metadata["Name"] + "==" + d.version)' 2>/dev/null); then
  echo "!! $IMAGE has no Mooncake package: install one in a derived image first (docs/plan.md, stage 3)" >&2
  exit 1
fi
# the image's build is CUDA-specific (mooncake-transfer-engine-cuda13). The store VM has no GPU, and even the plain
# build links libcudart and libcuda, so it gets the non-CUDA build of the same version.
PKG="${PKG%%-cuda*==*}-non-cuda==${PKG##*==}"
echo "store VM $STORE_IP: $PKG, pool ${POOL_GIB} GiB, $PROTOCOL, metadata $METADATA, tenant $TENANT"

store "set -e; mkdir -p ~/mooncake; [[ -d ~/mooncake/venv ]] || python3 -m venv ~/mooncake/venv
  ~/mooncake/venv/bin/pip freeze | grep -i '^mooncake' | grep -vxiF '$PKG' | xargs -r ~/mooncake/venv/bin/pip uninstall -q -y
  ~/mooncake/venv/bin/pip install -q '$PKG'"  # a no-op once that version is there
stop_all
# Eviction (the master's defaults: from 90% full, 5% at a time) keeps the pool from filling over a sweep; a stopped
# client drops its segment, so a restart starts empty.
store "cd ~/mooncake && B=~/mooncake/venv/bin
  export $MC_ENV
  nohup \$B/mooncake_master $([[ $PORT == 50051 ]] || echo "--port=$PORT") --default_kv_lease_ttl=$LEASE_MS < /dev/null > master.log 2>&1 & echo \$! > master.pid
  sleep 2
  [[ $MASTER_ONLY == 1 ]] && exit 0
  nohup \$B/mooncake_client --global_segment_size=${POOL_GIB}GB --master_server_address=$STORE_IP:$PORT \
    --metadata_server=$METADATA --protocol=$PROTOCOL --host=$STORE_IP --tenant_id=$TENANT < /dev/null > client.log 2>&1 &
  echo \$! > client.pid"
sleep 5
PROCS=master; [[ $MASTER_ONLY == 1 ]] || PROCS="master client"
store "for p in $PROCS; do kill -0 \"\$(cat ~/mooncake/\$p.pid)\" 2>/dev/null ||
       { echo \"!! mooncake_\$p exited:\"; tail -n 20 ~/mooncake/\$p.log; exit 1; }; done; echo \"store up ($PROCS)\""

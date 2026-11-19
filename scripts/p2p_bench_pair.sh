#!/usr/bin/env bash
# One p2p_bench.py variant: a receiver and a sender, each in the vLLM image.
#   two VMs:  PEER=kvtb@10.88.0.63 PEER_IP=10.88.0.63 scripts/p2p_bench_pair.sh "<agent args>" [UCX_X=y ...]
#   one VM:   scripts/p2p_bench_pair.sh "<agent args>"   (no PEER: both containers here, on a Docker bridge network,
#             talking over their container IPs)
# Agent args (--stripes/--threads/--split/--backend) go to both sides; extra KEY=VAL args are env for both.
# The last line is a SUMMARY json. Needs the repo's scripts/ at ~/kv-tier-bench/scripts (on both VMs for two).
set -euo pipefail
AGENT_ARGS=$1; shift
SEND_ARGS=${SEND_ARGS:-}
IMAGE=docker.io/vllm/vllm-openai:v0.34.0
NET=kvtb-p2pbench
# PYTHONPATH loads scripts/nixl_patch/sitecustomize.py, as in vLLM: KVTB_NIXL_* args below reach the harness too
env_flags="-e UCX_NET_DEVICES=eth0 -e PYTHONPATH=/kvtb-scripts/nixl_patch"  # on the host network: the VPC NIC; on the bridge: the container's veth
for kv in "$@"; do env_flags+=" -e $kv"; done
# DOCKER_OPTS: extra docker run flags for both containers (e.g. --security-opt apparmor=unconfined)
image_args="${DOCKER_OPTS:-} $env_flags -v \$HOME/kv-tier-bench/scripts:/kvtb-scripts:ro --entrypoint python3 $IMAGE /kvtb-scripts/p2p_bench.py"

if [[ -n ${PEER:-} ]]; then
  : "${PEER_IP:?}"
  on_recv() { ssh "$PEER" "$@"; }
  net="--network host"
else
  on_recv() { bash -c "$*"; }
  docker network inspect $NET >/dev/null 2>&1 || docker network create $NET >/dev/null
  net="--network $NET"
fi
on_recv "docker rm -f kvtb-p2pbench-recv >/dev/null 2>&1 || true"
docker rm -f kvtb-p2pbench-send >/dev/null 2>&1 || true
on_recv "docker run -d --name kvtb-p2pbench-recv --gpus all $net $image_args recv --port 7790 $AGENT_ARGS" >/dev/null
trap 'on_recv "docker rm -f kvtb-p2pbench-recv >/dev/null 2>&1 || true"' EXIT
[[ -n ${PEER:-} ]] || PEER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' kvtb-p2pbench-recv)
eval "docker run --rm --name kvtb-p2pbench-send --gpus all $net $image_args send --peer $PEER_IP:7790 $AGENT_ARGS \
  $SEND_ARGS --label \"$AGENT_ARGS $*\"" 2>&1 | grep -E '^\{|^SUMMARY|^IDLE|rror|Traceback|^\[' || true
on_recv "docker logs kvtb-p2pbench-recv 2>&1 | grep -E 'Traceback|rror' -A5 | tail -20" || true

#!/usr/bin/env bash
# Downloads both models into /models on this VM, and on VM B too if PEER is set, before any timed run. vLLM then
# runs with HF_HUB_OFFLINE=1. Run it before the GPU work starts; the two VMs download in parallel.
#   PEER=ubuntu@10.0.0.6 ./scripts/stage-weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
MODELS=${MODELS:-"Qwen/Qwen3.8-27B Qwen/Qwen3-VL-32B-Instruct"}

cmd="docker run --rm -v /models:/models -e HF_HOME=/models -e HF_XET_HIGH_PERFORMANCE=1 python:3.12-slim \
  sh -c 'pip install -q huggingface_hub && for m in $MODELS; do hf download \$m --cache-dir /models/hub; done'"
on self "$cmd" & self_pid=$!
if [[ -n $PEER ]]; then on peer "$cmd" & peer_pid=$!; fi
wait "$self_pid"
if [[ -n ${peer_pid:-} ]]; then wait "$peer_pid"; fi
echo "weights staged: $MODELS"

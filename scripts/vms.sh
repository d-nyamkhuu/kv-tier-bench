#!/usr/bin/env bash
# Creates the benchmark's Nebius VMs from the laptop, with the nebius CLI. Model weights go on one shared filesystem,
# staged by a cheap CPU VM, so no GPU-hour is spent downloading:
#   1. `weights` creates the filesystem kvtb-models, attaches it to a small CPU VM, downloads the stage's models onto
#      it, then deletes the CPU VM. The filesystem is not managed by the VM, so it survives.
#   2. `gpu` creates a GPU VM with the filesystem attached; cloud-init mounts it at /models (virtiofs). vLLM runs with
#      HF_HOME=/models and HF_HUB_OFFLINE=1, as after scripts/stage-weights.sh. Every GPU VM reads the same copy;
#      their compile caches go to a directory per VM (launch.py), since a file open on two VMs lacks full POSIX
#      guarantees (docs.nebius.com/compute/storage/types).
# s2 measured 1.77 GiB/s single-stream and 8.2 GiB/s 16-way reads on a 1 TiB filesystem (fio); performance is
# documented per 4 TiB, and that a 128 GiB one reads as fast is UNVERIFIED.
#
#   export PROJECT_ID=project-... SUBNET_ID=vpcsubnet-... REGION=us-central1
#   STAGE=s3 scripts/vms.sh weights             # kvtb-models, filled; the CPU VM is gone
#   scripts/vms.sh gpu kvtb-vm-a                # VM A, spot; SPOT=0 for on demand
#   scripts/vms.sh gpu kvtb-vm-b                # VM B, the p2p peer
#   scripts/vms.sh cpu kvtb-vm-c 16vcpu-64gb    # VM C, the s3 store (no GPU, no weights)
#   scripts/vms.sh ip kvtb-vm-a                 # public and private IP
#
# Env:
#   PROJECT_ID, SUBNET_ID  required, in REGION (default us-central1), which also picks GPU_PLATFORM
#   NEBIUS_PROFILE         passed as --profile when set
#   STAGE                  whose models `weights` downloads (default s3); MODELS overrides with HF ids
#   FS_GIB                 filesystem size (default 128: ~35 GB of s3 weights plus caches; $0.014/h at $0.08 per
#                          GiB-month)
#   STAGER_PRESET          the staging CPU VM (default 8vcpu-32gb, about $0.26/h, cpu-d3)
#   GPU_PLATFORM           default by REGION: gpu-rtx6000 in us-central1, gpu-rtx6000-a elsewhere (eu-south1)
#   GPU_PRESET             default 1gpu-24vcpu-218gb
#   SPOT                   1 (default) for a preemptible GPU VM that follows the spot price
#   SSH_PUBKEY             default ~/.ssh/id_ed25519.pub; the VMs' user is kvtb
set -euo pipefail
cd "$(dirname "$0")/.."

: "${PROJECT_ID:?set PROJECT_ID}" "${SUBNET_ID:?set SUBNET_ID}"
REGION=${REGION:-us-central1}
STAGE=${STAGE:-s3}
FS=kvtb-models  # the filesystem's name, and its virtiofs mount tag
FS_GIB=${FS_GIB:-128}
STAGER_PRESET=${STAGER_PRESET:-8vcpu-32gb}
if [[ $REGION == us-central1 ]]; then GPU_PLATFORM=${GPU_PLATFORM:-gpu-rtx6000}; fi
GPU_PLATFORM=${GPU_PLATFORM:-gpu-rtx6000-a}
GPU_PRESET=${GPU_PRESET:-1gpu-24vcpu-218gb}
SPOT=${SPOT:-1}
SSH_PUBKEY=${SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}
STAGER=kvtb-stager
TMP=$(mktemp -d)
STAGER_UP=0  # set while the staging VM exists, so a failure says that it still bills
trap 'rm -rf "$TMP"; if [[ $STAGER_UP == 1 ]]; then log "!! $STAGER still runs (and bills): re-run to resume, or delete it"; fi' EXIT

nb() { nebius ${NEBIUS_PROFILE:+--profile "$NEBIUS_PROFILE"} "$@"; }
log() { echo "[$(date -u +%H:%M:%S)] $*" >&2; }
# a known-hosts file of its own: a new VM may get a public IP whose old host key is still in ~/.ssh/known_hosts
ssh_vm() {
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$TMP/known_hosts" \
    -o ConnectTimeout=10 "kvtb@$1" "${@:2}"
}

# id_of <instance|filesystem> <name>: the resource id, or nothing if there is none
id_of() {
  nb compute "$1" get-by-name --parent-id "$PROJECT_ID" --name "$2" --format json 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["metadata"]["id"])' 2>/dev/null || true
}

# public_ip <instance name>: waits for the VM to run and prints its public address
public_ip() {
  local id; id=$(id_of instance "$1")
  for _ in $(seq 60); do
    ip=$(nb compute instance get --id "$id" --format json | python3 -c '
import json,sys
s = json.load(sys.stdin).get("status", {})
a = s.get("network_interfaces", [{}])[0].get("public_ip_address", {}).get("address", "")
print(a.split("/")[0] if s.get("state") == "RUNNING" else "")')
    if [[ -n $ip ]]; then echo "$ip"; return; fi
    sleep 10
  done
  log "$1 has no public IP after 10 min"; return 1
}

# wait_ready <ip>: until ssh works and cloud-init has finished
wait_ready() {
  for _ in $(seq 90); do
    if ssh_vm "$1" test -e /var/lib/kvtb-ready 2>/dev/null; then return; fi
    sleep 10
  done
  ssh_vm "$1" 'cloud-init status --long; sudo tail -n 20 /var/log/cloud-init-output.log' || true
  log "$1: cloud-init not done after 15 min"; return 1
}

# cloud_init <gpu|cpu|stager>: the user data. GPU VMs get Docker with the NVIDIA runtime; GPU VMs and the stager mount
# the weights filesystem at /models.
cloud_init() {
  cat <<EOF
#cloud-config
users:
  - name: kvtb
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "$SSH_PUBKEY")
package_update: true
packages: [iperf3, fio, tmux, rsync, jq, python3-venv]
runcmd:
  - |
    set -x
EOF
  if [[ $1 != stager ]]; then
    cat <<'EOF'
    command -v docker || (curl -fsSL https://get.docker.com | sh)
    usermod -aG docker kvtb
EOF
  fi
  if [[ $1 != cpu ]]; then
    cat <<'EOF'
    mkdir -p /models
    grep -q kvtb-models /etc/fstab || echo 'kvtb-models /models virtiofs defaults,nofail 0 0' >> /etc/fstab
    mount -a
    # nofail lets the VM boot without it; then /models is empty and vLLM would try to download. Not ready.
    mountpoint -q /models || { echo "kvtb: /models is not mounted" | tee /dev/console; exit 1; }
EOF
  fi
  if [[ $1 == gpu ]]; then
    cat <<'EOF'
    if ! command -v nvidia-ctk; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
      apt-get update && apt-get install -y nvidia-container-toolkit
    fi
    nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
EOF
  fi
  echo '    touch /var/lib/kvtb-ready'
}

# create_vm <name> <platform> <preset> <image family> <boot GiB> <spot 0|1> <cloud-init kind>
# Every kind but cpu gets the weights filesystem attached.
create_vm() {
  local name=$1 platform=$2 preset=$3 image=$4 boot=$5 spot=$6 kind=$7 fs_id=
  if [[ -n $(id_of instance "$name") ]]; then log "$name exists, reusing it"; return; fi
  if [[ $kind != cpu ]]; then
    fs_id=$(id_of filesystem "$FS")
    [[ -n $fs_id ]] || { log "no filesystem $FS: run \`vms.sh weights\` first"; return 1; }
  fi
  cloud_init "$kind" > "$TMP/ci.yaml"
  python3 - "$TMP/ci.yaml" "$name" "$platform" "$preset" "$image" "$boot" "$spot" "$FS" "$fs_id" > "$TMP/$name.json" <<'PY'
import json, sys
ci, name, platform, preset, image, boot, spot, tag, fs_id = sys.argv[1:]
spec = {
    "resources": {"platform": platform, "preset": preset},
    "network_interfaces": [{"subnet_id": "SUBNET", "name": "eth0", "ip_address": {}, "public_ip_address": {}}],
    "boot_disk": {
        "attach_mode": "READ_WRITE",
        "managed_disk": {
            "name": f"{name}-boot",
            "spec": {
                "size_gibibytes": boot,
                "block_size_bytes": "4096",
                "type": "NETWORK_SSD",
                "source_image_family": {"image_family": image},
            },
        },
    },
    "filesystems": (
        [{"attach_mode": "READ_WRITE", "mount_tag": tag, "existing_filesystem": {"id": fs_id}}] if fs_id else []
    ),
    "cloud_init_user_data": open(ci).read(),
    "recovery_policy": "FAIL",
}
if spot == "1":
    spec |= {"preemptible": {"on_preemption": "STOP"}, "follows_spot_price": {}}
print(json.dumps({"metadata": {"parent_id": "PROJECT", "name": name}, "spec": spec}))
PY
  sed -i "s/\"SUBNET\"/\"$SUBNET_ID\"/; s/\"PROJECT\"/\"$PROJECT_ID\"/" "$TMP/$name.json"
  log "creating $name ($platform $preset)"
  nb compute instance create --file "$TMP/$name.json" --format json > /dev/null
}

cmd_weights() {
  local models=${MODELS:-$(uv run -q python -c "
from kv_tier_bench.grid import load_experiment, stage_models
e = load_experiment('configs/experiment.yaml'); print(' '.join(e['models'][m]['hf_id'] for m in stage_models(e, '$STAGE')))")}
  if [[ -z $(id_of filesystem "$FS") ]]; then
    log "creating filesystem $FS ($FS_GIB GiB, network_ssd)"
    nb compute filesystem create --parent-id "$PROJECT_ID" --name "$FS" --type network_ssd \
      --size-gibibytes "$FS_GIB" --block-size-bytes 4096 --format json > /dev/null
  fi
  STAGER_UP=1
  create_vm "$STAGER" cpu-d3 "$STAGER_PRESET" ubuntu24.04-driverless 50 0 stager
  local ip; ip=$(public_ip "$STAGER"); wait_ready "$ip"
  log "$STAGER at $ip: downloading $models"
  # Runs on the stager. hf download skips files already there, so a re-run resumes.
  if ssh_vm "$ip" "MODELS='$models' bash -s" <<'REMOTE'
set -euo pipefail
python3 -m venv /tmp/hf && /tmp/hf/bin/pip install -q huggingface_hub hf_xet
for m in $MODELS; do
  sudo HF_XET_HIGH_PERFORMANCE=1 /tmp/hf/bin/hf download "$m" --cache-dir /models/hub > /dev/null
  echo "downloaded $m"
done
sync
sudo du -sh /models/hub
REMOTE
  then :; else exit 1; fi
  log "deleting $STAGER; the filesystem stays"
  nb compute instance delete --id "$(id_of instance "$STAGER")" > /dev/null
  STAGER_UP=0
  echo "$FS $(id_of filesystem "$FS")"
}

cmd_ip() {
  local id; id=$(id_of instance "$1")
  nb compute instance get --id "$id" --format json | python3 -c '
import json,sys
n = json.load(sys.stdin)["status"]["network_interfaces"][0]
print(n.get("public_ip_address", {}).get("address", "-").split("/")[0], n["ip_address"]["address"].split("/")[0])'
}

case ${1:-} in
  weights) cmd_weights ;;
  gpu) create_vm "${2:?name}" "$GPU_PLATFORM" "$GPU_PRESET" ubuntu24.04-cuda13.0 150 "$SPOT" gpu ;;
  cpu) create_vm "${2:?name}" cpu-d3 "${3:-16vcpu-64gb}" ubuntu24.04-driverless 100 0 cpu ;;
  ip) cmd_ip "${2:?name}" ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 2 ;;
esac

# Sourced by the other scripts. They run on the writer VM ("VM A"), from the repo directory.
#   PEER     ssh destination of the second VM ("VM B"), e.g. ubuntu@10.0.0.6. Only the p2p deployment needs it.
#   SELF_IP  this VM's private IP (default: the first address of `hostname -I`)
#   PEER_IP  VM B's private IP (default: the host part of PEER)
# vLLM binds to the private IPs only: its dev endpoints (cache reset) must not face the internet.
SELF_IP=${SELF_IP:-$(hostname -I | awk '{print $1}')}
PEER=${PEER:-}
PEER_IP=${PEER_IP:-${PEER#*@}}
CONTAINER=kvtb-vllm  # launch.CONTAINER

peer() { ssh -o BatchMode=yes "${PEER:?set PEER to the ssh destination of the second VM}" "$@"; }

# on <self|peer> <shell command>: the command string is parsed by a shell exactly once, here or on VM B
on() {
  if [[ $1 == self ]]; then bash -c "$2"; else peer "$2"; fi
}

#!/usr/bin/env bash
# Runs the day-1 checks from docs/plan.md against the vLLM container running on this VM. Records facts and changes
# nothing. Start one deployment by hand first (`kvtb serve-cmd ...`), then:
#   ./scripts/day1-checks.sh | tee results/day1-$(hostname).txt
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
x() { docker exec "$CONTAINER" "$@"; }  # run inside the vLLM container
logs() { docker logs "$CONTAINER" 2>&1; }
metrics() { curl -sf "http://$SELF_IP:8000/metrics"; }

echo "== vLLM version";             x python3 -c 'import vllm; print(vllm.__version__)'
echo "== NIXL plugins";             x python3 -c 'import nixl, os; d=os.path.dirname(nixl.__file__); print(d); print(sorted(os.listdir(d)))' 2>&1 | tail -3
echo "== attention block size";     logs | grep -m3 -iE "attention block size|block_size" || echo "(not found)"
echo "== HMA / hybrid manager";     logs | grep -m3 -iE "hybrid kv cache manager|HMA" || echo "(no mention)"
echo "== UCX transport (p2p)";      logs | grep -m5 -iE "ucx.*(tcp|sm|cuda_ipc|rc)" || echo "(no UCX proto info yet; run a p2p pull first)"
echo "== fs tier I/O mode";         logs | grep -m3 -iE "O_DIRECT" || echo "(no O_DIRECT warning: direct I/O in use, or not an fs deployment)"
echo "   GATE: without O_DIRECT every fs read can come from the page cache; fs numbers are not valid (plan.md, day-1)"
echo "== compile cache";            du -sh /models/vllm-cache /models/triton-cache 2>/dev/null || echo "(empty: first deploy)"
echo "== /dev/shm";                 x df -h /dev/shm
echo "== metric names";             metrics | grep -oE '^vllm:(kv_offload[a-z_]*|prompt_tokens_cached_by_source[a-z_]*)' | sort -u
echo "== cached_by_source labels";  metrics | grep -E '^vllm:prompt_tokens_cached_by_source' | head
echo "== offload config";           metrics | grep -E '^vllm:kv_offload_config_info' | head
echo "== tier labels";              metrics | grep -oE 'tier="[^"]*"' | sort -u
echo "== GDS";                      x sh -c 'command -v gdscheck && gdscheck -p || echo "gdscheck not in image"' | tail -5

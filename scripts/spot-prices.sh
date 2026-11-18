#!/usr/bin/env bash
# Live Nebius spot prices for the GPUs this benchmark uses, from the public API (no login needed).
# Run before launching a stage: prices change as often as every 15 minutes and are uncapped by default.
#   ./scripts/spot-prices.sh            # RTX PRO 6000
#   PLATFORMS=gpu-h200-sxm ./scripts/spot-prices.sh
set -euo pipefail
PLATFORMS=${PLATFORMS:-"gpu-rtx6000 gpu-rtx6000-a"}
curl -sf https://api.nebius.cloud/billing/v1alpha1/spot-price | PLATFORMS="$PLATFORMS" python3 -I -c '
import json, os, sys
want = os.environ["PLATFORMS"].split()
rows = [i for i in json.load(sys.stdin)["items"] if i["computeInstance"].get("platform") in want]
for i in sorted(rows, key=lambda i: (i["computeInstance"]["platform"], float(i["price"]["value"]))):
    platform, region, price = i["computeInstance"]["platform"], i["region"], float(i["price"]["value"])
    since = i["effectiveTime"]
    print(f"{platform:<15} {region:<12} ${price:.2f}/GPU-h  since {since}")
'

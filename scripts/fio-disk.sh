#!/usr/bin/env bash
# Day-1 disk check: the raw bandwidth and latency of the disk-tier volume (network SSD), with direct I/O, as a
# ceiling for the fs tier's numbers. Run on the writer VM with the volume mounted at $DIR.
# vLLM's fs tier writes one file per block, per KV group, per rank. Files range from ~4 MiB (Qwen3-VL-32B: 16-token blocks at 256 KiB/token) to tens of MB
# (Qwen3.8-27B: ~800-token blocks, plus its state group). So test MB-sized I/O at the tier's default thread
# counts (16 + 16).
#
#   DIR=/mnt/kv BS=8M ./scripts/fio-disk.sh > results/fio-kv-disk.json
set -euo pipefail
DIR=${DIR:?mount point of the disk-tier volume}
BS=${BS:-8M}
SIZE=${SIZE:-4G}
command -v fio >/dev/null || { echo "fio not installed (apt-get install -y fio)" >&2; exit 1; }

fio --output-format=json --directory="$DIR" --size="$SIZE" --bs="$BS" --ioengine=libaio --direct=1 \
  --group_reporting --time_based --runtime=60 \
  --name=write --rw=write --numjobs=16 --stonewall \
  --name=read --rw=randread --numjobs=16 --stonewall \
  --name=read-qd1 --rw=randread --numjobs=1 --iodepth=1 --stonewall

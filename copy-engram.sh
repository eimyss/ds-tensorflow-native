#!/usr/bin/env bash
# Copy the two 1017 Engram shards from spark-3's verified local copy to spark-1 and spark-2.
# Relay through spark-1 because spark-2 has no SSH key for spark-3. Low priority: the rank bakes
# are still running; ionice keeps the copies from starving them.
set -euo pipefail
ROOT=/home/eimantas/ds41-tp3
SRC=spark-3:/opt/dsv41/engram-1017
FILES=(model-00047-of-00048.safetensors model-00048-of-00048.safetensors)

mkdir -p "$ROOT/engram-1017"

echo "[$(date -Is)] pull spark-3 -> spark-1"
rsync -rlt --partial --info=progress2 \
  "$SRC/${FILES[0]}" "$SRC/${FILES[1]}" "$ROOT/engram-1017/"

echo "[$(date -Is)] push spark-1 -> spark-2"
ssh -o BatchMode=yes spark-2 "mkdir -p $ROOT/engram-1017"
rsync -rlt --partial --info=progress2 \
  "$ROOT/engram-1017/${FILES[0]}" "$ROOT/engram-1017/${FILES[1]}" "spark-2:$ROOT/engram-1017/"

echo "[$(date -Is)] verify sizes and head/tail hashes"
for f in "${FILES[@]}"; do
  echo "== $f"
  ref_size=$(ssh -o BatchMode=yes spark-3 "stat -c %s /opt/dsv41/engram-1017/$f")
  ref_head=$(ssh -o BatchMode=yes spark-3 "head -c 1048576 /opt/dsv41/engram-1017/$f | sha256sum" | cut -d' ' -f1)
  ref_tail=$(ssh -o BatchMode=yes spark-3 "tail -c 1048576 /opt/dsv41/engram-1017/$f | sha256sum" | cut -d' ' -f1)
  for h in spark-1 spark-2; do
    if [[ "$h" == spark-1 ]]; then p="$ROOT/engram-1017/$f"; else p=""; fi
    if [[ "$h" == spark-1 ]]; then
      sz=$(stat -c %s "$p"); hd=$(head -c 1048576 "$p" | sha256sum | cut -d' ' -f1); tl=$(tail -c 1048576 "$p" | sha256sum | cut -d' ' -f1)
    else
      read -r sz hd tl <<<"$(ssh -o BatchMode=yes "$h" "p=$ROOT/engram-1017/$f; echo \$(stat -c %s \$p) \$(head -c 1048576 \$p | sha256sum | cut -d' ' -f1) \$(tail -c 1048576 \$p | sha256sum | cut -d' ' -f1)")"
    fi
    if [[ "$sz" == "$ref_size" && "$hd" == "$ref_head" && "$tl" == "$ref_tail" ]]; then
      echo "  [ok] $h: $sz bytes, head/tail match"
    else
      echo "  [FAIL] $h: size $sz (want $ref_size), head $hd (want $ref_head), tail $tl (want $ref_tail)"
      exit 1
    fi
  done
done
echo "[$(date -Is)] done"

#!/usr/bin/env bash
# Wait for the three rank-cache bakes, then verify each one.
set -uo pipefail
ROOT=/home/eimantas/ds41-tp3
HOSTS=(spark-1 spark-2 spark-3)
FAILED=0
on() { local h=$1; shift; if [[ "$h" == "spark-1" ]]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$*"; fi; }
rank() { case "$1" in spark-1) echo 0;; spark-2) echo 1;; spark-3) echo 2;; esac; }

for i in $(seq 1 300); do
  active=0
  for h in "${HOSTS[@]}"; do
    r=$(rank "$h")
    st=$(on "$h" "systemctl --user is-active ds41-bake-r$r 2>/dev/null || true")
    case "$st" in
      active|activating) active=1 ;;
      failed) echo "[FAIL] bake r$r on $h failed"; on "$h" "tail -20 $ROOT/bake-$r.log 2>/dev/null"; FAILED=1 ;;
    esac
  done
  (( FAILED )) && exit 1
  (( active == 0 )) && break
  (( i % 10 == 0 )) && echo "[wait] bakes still running ($((i)) min)"
  sleep 60
done

echo
for h in "${HOSTS[@]}"; do
  r=$(rank "$h")
  echo "== $h (rank $r)"
  f=$(on "$h" "ls -1 $ROOT/rank-cache/rank$r/rank${r}of3-*.bin 2>/dev/null | head -1")
  if [[ -z "$f" ]]; then
    echo "  [FAIL] no final rank cache"; on "$h" "tail -20 $ROOT/bake-$r.log 2>/dev/null"; FAILED=1; continue
  fi
  sz=$(on "$h" "stat -c %s '$f'")
  echo "  file: $(basename "$f")  $((sz/1024/1024/1024)) GiB"
  if (( sz < 90000000000 )); then echo "  [FAIL] short"; FAILED=1; fi
  echo "  layout check:"
  on "$h" "$ROOT/out/bin/tf-dsv41-check $ROOT/model-tp3 $ROOT/rank-cache/rank$r $r 3 2>&1 | tail -3" || FAILED=$((FAILED+1))
  echo "  sampled bytes:"
  on "$h" "$ROOT/out/bin/tf-dsv41-check $ROOT/model-tp3 $ROOT/rank-cache/rank$r $r 3 --bytes 16 2>&1 | tail -3" || FAILED=$((FAILED+1))
done
echo
if (( FAILED == 0 )); then echo "[done] all three rank caches baked and verified"; else echo "[done] failures: $FAILED"; fi
exit $(( FAILED == 0 ? 0 : 1 ))

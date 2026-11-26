#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 native / TP3 startup on spark-1.
#
#   ./tp3-start.sh check    # read-only preflight (memory, ssh, kits, caches, ports)
#   ./tp3-start.sh start    # check, then start ranks 2 -> 1 -> 0 and wait for health
#   ./tp3-start.sh stop     # remove only our container on all three Sparks
#   ./tp3-start.sh status   # containers, health, memory
#   ./tp3-start.sh logs     # last 100 lines from each rank
#
# Overridable: MASTER_IP, NCCL_IFACE, NCCL_IB_DISABLE, PORT, TP_PORT, CONTEXT,
# POOL, PARALLEL, ROUND_ROWS, DRAFTS, THINKING, MIN_FREE_GIB, FORCE.
set -uo pipefail

# Optional tuning overlay (docs/TP3-TUNING.md). Sourced before the defaults below, so any
# variable it sets wins; absent file = exactly the previous behaviour.
_TUNE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$_TUNE_DIR/tp3-tune.env" ]]; then . "$_TUNE_DIR/tp3-tune.env"; fi

ROOT=${ROOT:-/home/eimantas/ds41-tp3}
IMAGE=${IMAGE:-tensorfold-glm53:v0.6.0}
NAME=${NAME:-tf-dsv41-tp3}
HOSTS=(spark-1 spark-2 spark-3)
MASTER_IP=${MASTER_IP:-192.168.1.6}      # spark-1 management LAN; override for the 10.10 fabric
NCCL_IFACE=${NCCL_IFACE:-enP7s7}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}    # 0: RoCE ring (libnccl 2.29.7 ring branch)
NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1}
PORT=${PORT:-18888}
TP_PORT=${TP_PORT:-29760}
CONTEXT=${CONTEXT:-262144}
POOL=${POOL:-1048576}
PARALLEL=${PARALLEL:-4}
ROUND_ROWS=${ROUND_ROWS:-48}
DRAFTS=${DRAFTS:-1}
DRAFT_MS=${DRAFT_MS:-25}        # TF_DS_DRAFT_MS: a round's ms past its forward, for draft-depth pricing
CLOCK_CAP=${CLOCK_CAP:-}        # e.g. 2100: nvidia-smi -lgc 300,$CLOCK_CAP on every rank before start
EXTRA_ENVS=${EXTRA_ENVS:-}      # extra "VAR=value" pairs passed to every rank (e.g. TF_DS_PREFILL_CHUNK=4096)
FIX_K=${FIX_K-}            # empty: adaptive 1..5 drafts (TF_DS_FIX_K); FIX_K=5 forces 5
THINKING=${THINKING:-1}
REASONING_EFFORT=${REASONING_EFFORT:-xhigh}
MIN_FREE_GIB=${MIN_FREE_GIB:-100}
FORCE=${FORCE:-0}
SERVED_K=${SERVED_K:-solo}
# the baked prefill-tail specialization (kit-extra/aot, see tools/bake_extra_aot.py): its cubin's name
TAIL_CUBIN=${TAIL_CUBIN:-75347c8c6589e4f9ac49908ec9ce660cce36f01ffbade8054079e5ae6832d338}

# Engram: spark-3 has a verified local copy; spark-1/2 prefer a local copy under $ROOT/engram-1017
# when it exists, else read the checkpoint over TrueNAS.
engram_local_on() { # host: both shards present locally?
  on "$1" "test -r $ROOT/engram-1017/model-00047-of-00048.safetensors && test -r $ROOT/engram-1017/model-00048-of-00048.safetensors"
}
engram_host_for() { # host rank
  case "$2" in
    2) echo /opt/dsv41/engram-1017 ;;
    *) if engram_local_on "$1"; then echo "$ROOT/engram-1017"
       else echo /mnt/truenas/models/deepseekv41-full; fi ;;
  esac
}
engram_ct_for() { # host rank
  case "$2" in
    2) echo /engram ;;
    *) if engram_local_on "$1"; then echo /work/engram-1017
       else echo /mnt/truenas/models/deepseekv41-full; fi ;;
  esac
}

c_red=$'\e[31m'; c_grn=$'\e[32m'; c_yel=$'\e[33m'; c_off=$'\e[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s[ok]%s   %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_yel" "$c_off" "$*"; }
bad()  { printf '%s[FAIL]%s %s\n' "$c_red" "$c_off" "$*"; FAILED=$((FAILED+1)); }
FAILED=0

on() { # host command
  local h=$1; shift
  if [[ "$h" == "spark-1" ]]; then
    bash -c "$*"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$*"
  fi
}

host_rank() { case "$1" in spark-1) echo 0;; spark-2) echo 1;; spark-3) echo 2;; esac; }

rss_gib() { # host
  on "$1" "awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null
}

# ------------------------------------------------------------------ checks
check_one() {
  local h=$1 r
  r=$(host_rank "$h")
  say ""
  say "== $h (rank $r) =="

  if ! on "$h" true 2>/dev/null; then bad "$h: ssh failed"; return; fi
  ok "$h: ssh"

  local avail
  avail=$(rss_gib "$h")
  if [[ -z "$avail" ]]; then bad "$h: cannot read MemAvailable"; else
    if (( avail >= MIN_FREE_GIB )); then ok "$h: ${avail} GiB available"
    else bad "$h: only ${avail} GiB available (need >= ${MIN_FREE_GIB}); stop the other stacks first"; fi
  fi

  local names
  names=$(on "$h" "docker ps --format '{{.Names}}'" 2>/dev/null | tr '\n' ' ')
  say "     containers: ${names:-none}"
  if [[ " $names " == *" $NAME "* ]]; then warn "$h: $NAME already running (start will replace it)"; fi

  # rank cache
  local matches cache size
  matches=$(on "$h" "ls -1 $ROOT/rank-cache/rank$r/rank${r}of3-*.bin 2>/dev/null | wc -l")
  if [[ "$matches" != "1" ]]; then
    bad "$h: rank cache has ${matches} final file(s) in $ROOT/rank-cache/rank$r (want exactly 1)"
  else
    cache=$(on "$h" "ls -1 $ROOT/rank-cache/rank$r/rank${r}of3-*.bin | head -1")
    size=$(on "$h" "stat -c %s '$cache'" 2>/dev/null || echo 0)
    if (( size > 90000000000 )); then ok "$h: $(basename "$cache") ($((size/1024/1024/1024)) GiB)"
    else bad "$h: $(basename "$cache") is only $((size/1024/1024/1024)) GiB (bake unfinished?)"; fi
  fi

  # binaries
  for b in "$ROOT/out/bin/tf-dsv41-lanes" "$ROOT/out/native/bin/tensorfold-native"; do
    if on "$h" "test -x '$b'" 2>/dev/null; then ok "$h: $(basename "$b")"; else bad "$h: missing $b"; fi
  done

  # kit: AOT, RoPE, token map, vision, grammar
  local n
  n=$(on "$h" "python3 -c 'import json;print(len(json.load(open(\"$ROOT/kit-tp3/aot/aot.json\"))[\"kernels\"]))'" 2>/dev/null || echo 0)
  if [[ "$n" == "643" ]]; then ok "$h: aot 643 kernels"; else bad "$h: aot has ${n:-?} kernels (want 643)"; fi
  # the prefill tail chunk's own specialization (a launch shape no shipped kit recorded: without it a
  # prompt whose length leaves 1..16 tokens of tail dies with MissingTritonVariant, see kit-extra/)
  if on "$h" "test -r '$ROOT/kit-tp3/aot/cubins/$TAIL_CUBIN.cubin'" 2>/dev/null; then
    ok "$h: prefill tail variant"; else bad "$h: kit lacks the prefill tail variant ($TAIL_CUBIN)"; fi
  local rope_ok=1 f sz
  for f in rope-plain-cos.f32 rope-plain-sin.f32 rope-compressed-cos.f32 rope-compressed-sin.f32; do
    sz=$(on "$h" "stat -c %s '$ROOT/kit-tp3/$f'" 2>/dev/null || echo 0)
    [[ "$sz" == "142606336" ]] || rope_ok=0
  done
  (( rope_ok )) && ok "$h: RoPE tables" || bad "$h: RoPE tables missing/short"
  on "$h" "test -s '$ROOT/kit-tp3/dsv41_token_map.json'" && ok "$h: token map" || bad "$h: token map missing"
  on "$h" "test -s '$ROOT/kit-tp3/vision/gate_bias_vl.safetensors' && test -s '$ROOT/kit-tp3/grammar/libtfgrammar.so'" \
    && ok "$h: vision/grammar" || bad "$h: vision/grammar missing"

  # model view + config
  on "$h" "test -r '$ROOT/model-tp3/model-00001-of-00048.safetensors'" \
    && ok "$h: model shards" || bad "$h: model shards not readable"
  if [[ "$h" == "spark-1" ]]; then
    python3 - "$ROOT/model-tp3/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); t=c.get("text_config",c); m=c.get("tensorfold_tp3",{})
ok = t.get("num_attention_heads")==72 and t.get("o_groups")==9 and t.get("vocab_size")==129408 \
     and m.get("world_size")==3 and m.get("logical_vocab_size")==129280
print("[ok]   padded config: 72 heads / 9 groups / 129408 vocab / TP3 marker" if ok else "[FAIL] config is not the padded TP3 config")
sys.exit(0 if ok else 1)
PY
    (( $? == 0 )) || FAILED=$((FAILED+1))
  fi

  # Engram
  local en; en=$(engram_host_for "$h" "$r")
  on "$h" "test -r '$en/model-00047-of-00048.safetensors' && test -r '$en/model-00048-of-00048.safetensors'" \
    && ok "$h: Engram shards ($en)" || bad "$h: Engram shards missing under $en"
}

check_all() {
  FAILED=0
  say "DeepSeek-V4.1-Flash-1017 · TP3 · native"
  say "master $MASTER_IP, port $PORT (TP $TP_PORT), context $CONTEXT, pool $POOL, parallel $PARALLEL, drafts $DRAFTS"
  for h in "${HOSTS[@]}"; do check_one "$h"; done
  say ""
  if on spark-1 "ss -ltn 'sport = :$PORT' 2>/dev/null | grep -q LISTEN"; then
    bad "spark-1: TCP $PORT is already in use"
  else ok "spark-1: TCP $PORT free"; fi
  say ""
  if (( FAILED == 0 )); then ok "preflight passed"; else bad "$FAILED check(s) failed"; fi
  return $(( FAILED == 0 ? 0 : 1 ))
}

# ------------------------------------------------------------------ launch
envs() { local x; for x in "$@"; do printf -- '-e %q ' "$x"; done; }

start_one() {
  local h=$1 r en; en=$(engram_ct_for "$h" "$2"); local extra_mount=""
  r=$(host_rank "$h")
  [[ "$h" == "spark-3" ]] && extra_mount="-v /opt/dsv41/engram-1017:/engram:ro "
  local common=(
    "TF_TP_WORLD=3" "TF_TP_MASTER=$MASTER_IP" "TF_TP_PORT=$TP_PORT"
    "TF_DS_RANK_CACHE=/cache" "TF_DS_KIT=/work/kit-tp3" "TENSORFOLD_CUDA_KERNELS=/work/kit-tp3/aot"
    "TF_DS_ENGRAM=$en" "TF_DS_TOKEN_MAP=/work/kit-tp3/dsv41_token_map.json"
    "TF_DS_WINDOW=$POOL" "TF_DS_ROUND_ROWS=$ROUND_ROWS" "TF_DS_STREAMS=$PARALLEL" "TF_DS_SERVED_K=$SERVED_K"
    "TF_DS_GRAPHS=1" "TF_DS_HC_SIDE=1" "TF_DS_L2_PREFETCH=1" "TF_DS_ENGRAM_AIO=1" "TF_DS_DRAFT_MS=$DRAFT_MS"
    "NCCL_SOCKET_IFNAME=$NCCL_IFACE" "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" "NCCL_DEBUG=WARN" "HF_HUB_OFFLINE=1"
    "NCCL_IB_HCA=$NCCL_IB_HCA" "NCCL_IB_SUBNET_AWARE_ROUTING=1" "NCCL_IB_MERGE_NICS=0"
    "NCCL_NET_PLUGIN=none" "NCCL_IB_ADDR_FAMILY=AF_INET" "LD_PRELOAD=/opt/nccl/lib/libnccl.so.2"
  )
  [[ -n "${FIX_K:-}" ]] && common+=("TF_DS_FIX_K=$FIX_K")
  if [[ -n "${EXTRA_ENVS:-}" ]]; then read -r -a _extra <<<"$EXTRA_ENVS"; common+=("${_extra[@]}"); fi

  local run=(docker run -d --name "$NAME" --gpus all --ipc=host --network host
    --ulimit memlock=-1 --ulimit stack=67108864
    -v "$ROOT:/work:ro" -v "$ROOT/rank-cache/rank$r:/cache:ro"
    -v /mnt/truenas:/mnt/truenas:ro -v /home/eimantas/nccl_spark_cluster/build/lib:/opt/nccl/lib:ro ${extra_mount})
  local envstr; envstr=$(envs "${common[@]}")

  if [[ "$r" == "0" ]]; then
    local serve=(serve /work/model-tp3 --backend cuda --host 0.0.0.0 --port "$PORT"
      --name DeepSeek-V4.1-Flash-1017-TP3 --context "$CONTEXT" --parallel "$PARALLEL"
      --max-tokens 32768 --temperature 1.0 --top-p 1.0 --top-k 20)
    [[ "$THINKING" == "1" ]] && serve+=(--thinking) || serve+=(--no-thinking)
    [[ -n "$REASONING_EFFORT" ]] && serve+=(--reasoning-effort "$REASONING_EFFORT")
    [[ "$DRAFTS" == "1" ]] || serve+=(--no-drafts)
    local cmd="docker run -d --name $NAME --gpus all --ipc=host --network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
      -v $ROOT:/work:ro -v $ROOT/rank-cache/rank0:/cache:ro -v /mnt/truenas:/mnt/truenas:ro -v /home/eimantas/nccl_spark_cluster/build/lib:/opt/nccl/lib:ro $envstr \
      --entrypoint /work/out/native/bin/tensorfold-native $IMAGE $(printf '%q ' "${serve[@]}")"
    on "$h" "docker rm -f $NAME >/dev/null 2>&1 || true; $cmd" >/dev/null || { bad "$h: docker run failed"; return 1; }
  else
    local lanes=(/work/model-tp3 /cache "$r" 3 "$MASTER_IP" "$TP_PORT" /work/kit-tp3
      --pool "$POOL" --context "$CONTEXT" --engram "$en" --token-map /work/kit-tp3/dsv41_token_map.json
      --drafts "$DRAFTS" --round-rows "$ROUND_ROWS" --streams "$PARALLEL" --graphs 1 --side 1 --prefetch 1 --aio 1)
    [[ -n "${FIX_K:-}" ]] && lanes+=(--fix-k "$FIX_K")
    local cmd="docker run -d --name $NAME --gpus all --ipc=host --network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
      -v $ROOT:/work:ro -v $ROOT/rank-cache/rank$r:/cache:ro -v /mnt/truenas:/mnt/truenas:ro -v /home/eimantas/nccl_spark_cluster/build/lib:/opt/nccl/lib:ro $extra_mount $envstr \
      --entrypoint /work/out/bin/tf-dsv41-lanes $IMAGE $(printf '%q ' "${lanes[@]}")"
    on "$h" "docker rm -f $NAME >/dev/null 2>&1 || true; $cmd" >/dev/null || { bad "$h: docker run failed"; return 1; }
  fi
  ok "$h: started rank $r"
}

wait_health() {
  local i running
  say "waiting for rank 0 health on http://127.0.0.1:$PORT ..."
  for i in $(seq 1 180); do
    if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      ok "rank 0 is serving (after ~$((i*10))s)"; return 0
    fi
    running=$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)
    if [[ "$running" != "true" ]]; then
      bad "rank 0 container exited"; docker logs --tail 80 "$NAME" 2>&1 | tail -60; return 1
    fi
    (( i % 3 == 0 )) && say "  ... still loading ($((i*10))s)"
    sleep 10
  done
  bad "rank 0 did not become healthy in 1800s"; docker logs --tail 80 "$NAME" 2>&1 | tail -60; return 1
}

smoke() {
  say ""
  say "smoke: one short request"
  local body
  body=$(curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d '{"model":"DeepSeek-V4.1-Flash-1017-TP3","messages":[{"role":"user","content":"Reply with the single word READY."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
  say "$body" | head -c 800; say ""
  [[ -n "$body" ]] && ok "server answered" || bad "no answer"
}

clock_cap() { # apply CLOCK_CAP on every rank (thermal stability; peers cap at 2100 MHz)
  [[ -n "${CLOCK_CAP:-}" ]] || return 0
  local h
  for h in "${HOSTS[@]}"; do
    if on "$h" "sudo -n nvidia-smi -lgc 300,$CLOCK_CAP >/dev/null 2>&1"; then
      ok "$h clock capped at ${CLOCK_CAP} MHz"
    else
      warn "$h could not cap the clock (needs passwordless sudo nvidia-smi)"
    fi
  done
}

cmd_start() {
  check_all || { say ""; bad "preflight failed; not starting (FORCE=1 overrides only the memory check)"; return 1; }
  say ""
  clock_cap
  say "starting ranks 2, 1, then 0"
  start_one spark-3 2 || return 1
  start_one spark-2 1 || return 1
  sleep 2
  start_one spark-1 0 || return 1
  wait_health || return 1
  smoke
}

cmd_stop() {
  local h
  for h in "${HOSTS[@]}"; do
    on "$h" "docker rm -f $NAME >/dev/null 2>&1 || true"
    ok "$h: $NAME removed (if it existed)"
  done
}

cmd_status() {
  local h r
  for h in "${HOSTS[@]}"; do
    r=$(host_rank "$h")
    say "== $h (rank $r)"
    on "$h" "docker ps --filter name=$NAME --format '  {{.Names}} {{.Status}}'" || true
    on "$h" "free -g | awk '/^Mem:/{printf \"  memory available: %d GiB\\n\", \$7}'" || true
  done
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    ok "health: serving on :$PORT"
  else
    warn "health: not serving on :$PORT"
  fi
}

cmd_logs() {
  local h
  for h in "${HOSTS[@]}"; do
    say "== $h"
    on "$h" "docker logs --tail 100 $NAME 2>&1" || true
  done
}

case "${1:-}" in
  check)  check_all ;;
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  *) say "usage: $0 {check|start|stop|status|logs}"; exit 2 ;;
esac

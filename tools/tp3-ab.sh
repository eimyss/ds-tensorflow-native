#!/usr/bin/env bash
# tools/tp3-ab.sh LABEL [N] — A/B one lever at a time on a running TP3 server.
#
# Runs a fixed prompt set N times against http://127.0.0.1:PORT, then writes the median
# decode tok/s and TTFT to results/ab-<LABEL>.json. It never restarts the cluster, so the
# discipline is: change one knob, `./tp3-start.sh stop && ./tp3-start.sh start`, run a label,
# change one knob, run the next label. Keep the unfavourable runs.
#
# Env: PORT (18888), MODEL (DeepSeek-V4.1-Flash-1017-TP3), N (9), MAX_TOKENS (512),
#      PROMPTS (tools/ab-prompts.txt), OUT_DIR (results)
set -uo pipefail
LABEL=${1:-base}
shift || true
N=${N:-${1:-9}}
MAX_TOKENS=${MAX_TOKENS:-512}
PORT=${PORT:-18888}
MODEL=${MODEL:-DeepSeek-V4.1-Flash-1017-TP3}
OUT_DIR=${OUT_DIR:-results}
PROMPTS=${PROMPTS:-$(dirname "$0")/ab-prompts.txt}

mkdir -p "$OUT_DIR"
if [[ ! -f "$PROMPTS" ]]; then
  cat >"$PROMPTS" <<'P'
Write a Python function that merges two sorted lists, then explain its complexity.
Write a Rust function that parses a semicolon-separated config file into a HashMap.
Explain, in three paragraphs, how a write-ahead log guarantees crash recovery.
P
  printf 'created a default prompt set: %s\n' "$PROMPTS"
fi
mapfile -t PROMPT_LIST <"$PROMPTS"

body_for() {
  python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[2],"messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[3]),"temperature":0,"stream":True}))' "$1" "$MODEL" "$MAX_TOKENS"
}

# one request -> "<tok_s> <ttft_s> <events>"
one() {
  local body tmp meta tf tt ev
  body=$(body_for "$1")
  tmp=$(mktemp)
  meta=$(curl -sN --max-time 1800 -o "$tmp" -w '%{time_starttransfer} %{time_total}' \
           -H 'Content-Type: application/json' -d "$body" \
           "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null) || true
  ev=$(grep -c '^data: ' "$tmp" 2>/dev/null || echo 0)
  rm -f "$tmp"
  python3 -c '
import sys
try:
    tf, tt = sys.argv[1].split()
    tf, tt, ev = float(tf), float(tt), int(sys.argv[2])
except Exception:
    print("0 0 0"); raise SystemExit
dt = tt - tf
tok = max(ev - 1, 1)          # the last event is [DONE]
print(f"{(tok / dt if dt > 0 else 0):.2f} {tf:.3f} {ev}")' "$meta" "$ev"
}

toks=(); ttfts=()
for i in $(seq 1 "$N"); do
  for p in "${PROMPT_LIST[@]}"; do
    read -r t f _ < <(one "$p")
    toks+=("$t"); ttfts+=("$f")
  done
  printf '\r%s: run %d/%d  (%d samples)' "$LABEL" "$i" "$N" "${#toks[@]}"
done
printf '\n'

{
  printf '%s\n' "${toks[@]}" >"$OUT_DIR/.toks.$LABEL"
  printf '%s\n' "${ttfts[@]}" >"$OUT_DIR/.ttft.$LABEL"
}
python3 - "$LABEL" "$OUT_DIR" <<'PY'
import json, os, statistics, sys
lab, out = sys.argv[1], sys.argv[2]
tok = [float(x) for x in open(f"{out}/.toks.{lab}") if x.strip()]
ttf = [float(x) for x in open(f"{out}/.ttft.{lab}") if x.strip()]
rec = {"label": lab, "samples": len(tok),
       "tok_s_median": round(statistics.median(tok), 2),
       "tok_s_min": round(min(tok), 2), "tok_s_max": round(max(tok), 2),
       "ttft_s_median": round(statistics.median(ttf), 3),
       "tok_s_samples": [round(t, 2) for t in tok]}
path = os.path.join(out, f"ab-{lab}.json")
open(path, "w").write(json.dumps(rec, indent=2) + "\n")
os.remove(f"{out}/.toks.{lab}"); os.remove(f"{out}/.ttft.{lab}")
print(f"{lab}: median {rec['tok_s_median']} tok/s "
      f"(min {rec['tok_s_min']}, max {rec['tok_s_max']}), "
      f"TTFT {rec['ttft_s_median']} s, {rec['samples']} samples -> {path}")
PY

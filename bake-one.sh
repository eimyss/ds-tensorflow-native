#!/bin/bash
set -euo pipefail
R=$1
cd "$HOME/ds41-tp3"
mkdir -p "rank-cache/rank$R"
exec nice -n 19 ionice -c2 -n7 ./out/bin/tf-dsv41-rank-cache model-tp3 "rank-cache/rank$R" "$R" 3

# TP3 tuning runbook — measure, then set

Peer repos disagree about defaults because they run different shapes (2, 3, 4 and 8 Sparks,
different pools, different weight packs). None of their numbers transfer as a default. What
transfers is the *order* of measurement, which is what this file fixes.

House rules (from `spark-ds41f`'s methodology, the most rigorous public one):
one lever per boot · medians of ≥9 runs · keep the unfavourable runs · never change two knobs
between two points you intend to compare.

## 0. Baseline

```sh
./tp3-start.sh start
tools/tp3-ab.sh base            # writes results/ab-base.json
```

## 1. Price the round overhead (`TF_DS_DRAFT_MS`) — do this first

`chooseKWith` maximises expected tokens per ms using a round cost table plus a fixed
overhead `dms`. `tp3-start.sh` currently passes `TF_DS_DRAFT_MS=25`. If the real overhead on
TP3 is ~10 ms, 25 makes the engine verify fewer drafts than it should — a silent loss in
every single-stream reply, with identical outputs.

```sh
# the engine prints per-round timings with clock marks on
EXTRA_ENVS="TF_DS_CLOCK_MARKS=1" ./tp3-start.sh start
./tp3-start.sh logs | grep -i round
```

Overhead ≈ (measured round ms) − (round-forward ms from the `ROUND_MS` table for those rows).
Set `DRAFT_MS` to the median, then A/B:

```sh
DRAFT_MS=25 tools/tp3-ab.sh dms25
DRAFT_MS=12 tools/tp3-ab.sh dms12
DRAFT_MS=8  tools/tp3-ab.sh dms8
```

Keep the best *median*; if two are within noise, keep the larger (it is the safer direction:
fewer wasted verifications).

## 2. Recalibrate the confidence fit (`TF_DS_CONF_CAL`)

`1.0,0.19` was fitted on the **four**-node lane. On three ranks the accept statistics differ.
Fit from `TF_DS_ACCEPT_LOG`-style logs if available, else sweep `b` in small steps:

```sh
EXTRA_ENVS="TF_DS_CONF_CAL=1.0,0.05" tools/tp3-ab.sh cal05
EXTRA_ENVS="TF_DS_CONF_CAL=1.0,0.30" tools/tp3-ab.sh cal30
```

Do not touch `a` until `b` is settled.

## 3. Clock cap

```sh
CLOCK_CAP=2100 ./tp3-start.sh start
```

GB10 throttling appears as variance, not as a clean slowdown, so compare the *spread* of
9 runs, not only the median. If thermals are fine, 2200 MHz is the other value seen in the wild.

## 4. Prefill chunk (`TF_DS_PREFILL_CHUNK`)

4K is the only TP3-qualified value published (LakyFX TP3; their TP4 uses 8K). Larger chunks
mean fewer, longer prefill passes — better TTFT on long prompts, higher peak memory.
Watch `MemAvailable` on the tightest rank.

## 5. L2 prefetch sizing

Sweep in this order — each is independent, one per boot:

```sh
EXTRA_ENVS="TF_DS_L2_MOE_MB=512" tools/tp3-ab.sh l2moe512
EXTRA_ENVS="TF_DS_L2_DN_MB=256"  tools/tp3-ab.sh l2dn256
EXTRA_ENVS="TF_DS_L2_DEC_MB=128" tools/tp3-ab.sh l2dec128
EXTRA_ENVS="TF_DS_L2_ATTN_MB=64" tools/tp3-ab.sh l2attn64
```

`TF_DS_L2_SKIP` (bitmask) is the negative control: if skipping prefetch entirely is *not*
slower, the sizing is not doing anything and you can stop tuning it.

## 6. Rounds (`TF_DS_ROUND_ROWS`)

48 is the four-node lane's value; LakyFX keeps 16 on TP3. Wide steps favour multi-stream
throughput, narrow ones favour single-stream latency. Since the goal here is single-stream
tok/s, A/B 16 / 32 / 48 and keep the median winner for `PARALLEL=1` and `PARALLEL=4`
separately — they need not agree.

## 7. Pool (`POOL` / `TF_DS_WINDOW`)

Bigger pool = more prefix reuse, less headroom for KV. Peers run 2.1M (2 Sparks), 3.3M and
8M (LakyFX TP3, 2 GiB snapshot). Our rank cache is 96.82 GiB/rank, so the binding constraint
is `MemAvailable` on the tightest Spark: raise in steps, and re-run the 128K needle after
each raise.

## Verification gates (any of these failing means revert the lever)

1. Drafted == undrafted token-for-token on the 320-prompt identity set.
2. Needle 3/3 at 32K and 128K.
3. `MemAvailable` never below ~8 GiB on any rank during the run.
4. No `MissingTritonVariant` in any log (a new shape was selected).

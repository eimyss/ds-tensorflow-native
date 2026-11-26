# Native DeepSeek-V4.1-Flash-1017 on TP3 — prep status

Host: `spark-1.eimantas.homelab` (GB10, 121 GiB RAM, 349 GB free on `/`)
Date: 2026-11-01
Constraint during prep: another LLM (`glm53-flash-tf`) holds ~112 GiB; all builds ran
with `-j1` and a 6 GiB Docker cgroup cap. Weights were **not** copied or baked.

## TL;DR

The engine is ported, built, and offline-validated for world=3. The full 3-rank
weight index over the real checkpoint is correct: **95,987 entries, 96.82 GiB per
rank, every split tensor partitioned exactly**. The three rank caches are baking
now on their Sparks (low priority); no GPU run yet.

## Workspace

```
~/ds41-tp3/
  src/mia/          MiaAI recipe (patches + scripts)
  src/laky/         LakyFx recipe (TP3 donor; kernels extracted below)
  src/kitrepo/      Bertholomus 4-node kit
  src/engine/       5940208 + all 43 Mia patches + our TP3 changes
  src/engine-baseline/  same, without our changes (for the diff)
  donor-kernels/    LakyFx tensorfold-kernels.tar.gz extracted (aot/ + cubins/)
  kit-tp3/          base kit + merged TP3 AOT manifest (642 kernels)
  model-tp3/        symlinks to /mnt/truenas/models/deepseekv41-full + padded config.json
  tools/zig/        Zig 0.17.0 (aarch64, SHA verified)
  tools/merge_aot.py, make_tp3_config.py, verify_index.py
  out/              built binaries (tensorfold-native, all tf-dsv41-*)
  rank0.index rank1.index rank2.index
  tp3-changes.patch (547 lines, 14 files — our changes only)
  build.log test.log
```

## What was done

1. Cloned the pinned engine (`bertholomus/TensorFold` @ `5940208`), applied all
   **43** Mia patches (numbers skip 0038, 0044–0045), cleanly.
2. Implemented the TP3 port (see `tp3-changes.patch`):
   - `config.zig`: `logical_vocab`, `tokenVocab()`, `maskHostLogits()`,
     `checkWorld()`, `tensorfold_tp3` marker parsing, padded `checkShapes`
     (72 heads / 9 groups / 129,408 vocab).
   - `model.zig`: world=3 accepted, `checkWorld`, logits paths use
     `tokenVocab`/mask.
   - `native.zig`, `prompt2d.zig`: world=3 = flat TP.
   - `rank_cache_main.zig`: `WORLD=3` bake mode + `checkWorld`.
   - `check_main.zig`, `load_main.zig`: `checkWorld`.
   - `checkpoint.zig`: **zero-fill past the source tensor** for padded native
     slices (heads 64–71, vocab tail, rank-2 `wo_b` K tail). The **9th `wo_a`
     group is a byte-copy of group 0** (HC normalization needs a real group; its
     output is silenced by `wo_b`'s zeroed input rows), with unit tests for both.
   - `weights.zig`: pad the replicated `attn_sink` vector 64→72 at load, and the
     Markov head/embed to the padded vocabulary for DSpark.
   - `ops.zig` + `dsv41_ops.cu`: `addN_bf16` (N-rank Engram sum, rank order,
     one fp32 add each, one bf16 round) replacing the two `world != 2`
     `notPorted` sites in `round.zig`/`prompt.zig`.
   - `draft.zig`: mask padded token ids in the DSpark Markov draft.
3. Merged the TP3 AOT variants: **642 kernels** total, 297 donor variants
   accepted. Verified present: `_sparse_attn_part` H=24/HB=8 (34),
   `_sparse_attn_merge` (1), `_sparse_attn_merge_rot2` (1), `_rope_heads`
   H=24/HB=8 (2), `_hc_mix_part` WORLD=3 (5), `_hc_mix_pf` (8),
   `_hc_post` (1), `_hc_post_only` (1), `_pick` (1).
4. Built the padded `model-tp3/config.json` (marker included) over symlinks to
   the real checkpoint at `/mnt/truenas/models/deepseekv41-full`.
5. Built with Zig 0.17.0 + CUDA 13.3 in the existing NGC container
   (`tensorfold-glm53:v0.6.0`): `tensorfold-native` + every `tf-dsv41-*` tool.
6. Ran the world-3 index-only bake for all three ranks and verified it
   independently.

## Verified

- `zig build test`: **128/129 pass** in the dsv41 unit tests. The one failure is
  pre-existing (`lanes` `chooseK`: `conf_cal` default `{1,0.19}` vs the test's
  `{1,0}`). The separate `direct_io` step fails on O_DIRECT in the container
  overlay (environmental). Both new padding tests pass.
- `tf-dsv41-rank-cache MODEL OUT 0|1|2 3 --index-only`: 95,987 entries each;
  **96.82 GiB each**; 0.2 s after the first scan.
- `tools/verify_index.py`: same tensor/part names on all ranks; **0 byte-count
  mismatches**; **94,985 split groups, 0 partition failures** (no gaps/overlaps).
- Padded shapes confirmed in the index:
  `wq_b [12288,1280]`, `wo_a` 9th group `[1024,4096]`, `wo_b [5120,3072]`,
  `head [43136,5120]`, expert `w1 [768,2560]` / `w2 [5120,384]`,
  Engram `[25600,2048]`.
- Container smoke test (no weights, 1 GiB cap): `tensorfold-native --help`,
  `tf-dsv41-lanes`, and `tensorfold-native info /work/model-tp3` run; the padded
  view reports 475.26 GiB / 510.3 GB of checkpoint weights (Engram included).

## Rank cache: size and placement

Measured from the actual world-3 index:

| | per rank | 3 ranks |
|---|---:|---:|
| with DSpark (default) | 96.82 GiB / 104.0 GB | 290.5 GiB / 311.9 GB |
| target only (`--no-dspark`) | 94.19 GiB / 101.1 GB | 282.6 GiB / 303.4 GB |
| DSpark blocks | 2.63 GiB / 2.8 GB | — |

You do **not** need 300 GB on one disk: each Spark only ever reads its own
`rank{r}of3-<hash>.bin`. The checkpoint (473 GB) stays on TrueNAS; the rank
caches are the only local copies.

Recommended placement (checked):
- spark-1: 345 GB free → rank 0 (104 GB) fits.
- spark-3: 173 GB free, `/mnt/truenas/models/deepseekv41-full` mounted → rank 2
  (104 GB) fits, ~69 GB left.
- spark-2: host is up, but its SSH host key is not in spark-1's known_hosts;
  add it before baking there. If its disk is tight, bake rank 1 on spark-1 and
  copy it over the fabric.

Either bake each rank on its own Spark from TrueNAS (parallel, 473 GB read
each), or bake on spark-1 one at a time and move it (peak 104 GB local, 1.4 TB
total read). Use `nice -n 19 ionice -c2 -n7`; the baker is low-RAM (~1–2 GB)
but writing 104 GB will use page cache, so a 2–4 GB cgroup/scope is prudent
while the other LLM runs. Bake with DSpark included (default);
`--no-dspark` saves only 2.8 GB/rank and would need a re-bake later.

## DSpark

Correction to the earlier note: **DSpark is not blocked by `_gemm`.**
`tri_markov.matmul` (`_gemm`) is only the EXL3 prefill path
(`exl3_prefill.zig`, `mmRows` with `e.mx == null`) and its conformance test;
the native path's draft linears go through `mx`/`groupedDecode`, and the Markov
loop uses `_bias/_score/_finish/_pick`. The merged kit already has `_pick`
WORLD=3 (and, as a bonus, 36 `_gemm` variants incl. the TP3 K/N shapes with
both `*bf16` and `*fp32` OUT).

What was added for DSpark:
- draft vocab mask (ids ≥129,280) in `draft.zig`;
- `_pick` WORLD=3 from the donor kit;
- Markov head/embed grown to the padded 129,408 rows (128 zero rows) in the
  loader — `draft.zig` converts `c.vocab * markov_rank` elements, so without
  this it would read past the 129,280-row source.

Geometry checks: `n_cols = 129408 / 3 = 43136`, a multiple of `bn=64`;
`s_tiles = cdiv(43136,1024) = 43`, `b_tiles = 169`, `_finish` `BP = 64`
(present). The padded vocabulary is what makes this legal; 129,280/3 is not an
integer.

So: bake with drafts (default), then set `DRAFTS=1` (and `--drafts 1` on the
lanes). Keep `DRAFTS=0` only for the very first target-model smoke, then turn it
on and check drafted == undrafted token-for-token.

## Startup orchestrator

`~/ds41-tp3/tp3-start.sh` (on spark-1):

```sh
./tp3-start.sh check    # read-only preflight
./tp3-start.sh start    # check, then ranks 2 -> 1 -> 0, wait for health, smoke
./tp3-start.sh status
./tp3-start.sh logs
./tp3-start.sh stop     # removes only tf-dsv41-tp3 on the three Sparks
```

`check` validates: SSH to spark-2/3; `MemAvailable >= 100 GiB`; exactly one final
`rank?of3-*.bin` (>90 GB) per Spark; both binaries; **642** AOT kernels; four
142,606,336-byte RoPE tables; the token map; vision bias + grammar library; model
shards; the padded config (72/9/129408 + marker); the Engram shards; and that
TCP 18888 is free. It prints the running containers and refuses to start while
another stack holds the memory.

`start` launches rank 2 on spark-3, rank 1 on spark-2, then rank 0 on spark-1
(the followers wait for rank 0's rendezvous), waits up to 30 min for
`/v1/models`, then sends one short smoke request. It mounts `/work` (model, kit,
binaries), the per-rank `/cache`, `/mnt/truenas` (model shards; Engram on
spark-1/2), and on spark-3 `/opt/dsv41/engram-1017` (its local Engram copy).

Defaults: master `192.168.1.6` (management LAN), NCCL over `enP7s7` with
`NCCL_IB_DISABLE=1`, API port 18888, TP port 29760, context 262144, pool
1048576, parallel 4, drafts 0. Override `MASTER_IP`/`NCCL_IFACE`/
`NCCL_IB_DISABLE` when the 10.10 fabric is wired, and `CONTEXT`/`POOL`/
`PARALLEL`/`DRAFTS`/`THINKING` as needed.

spark-3's existing **EXL3 2.9bpw** service (`tf-dsv41-rank0`, port 18891) is not
touched; its local `/opt/dsv41/engram-1017` and token map were reused.

## Known limitations / deferred

1. **AOT coverage must be proven at runtime.** The merged set is structurally
   compatible and covers the TP3 profile, but the engine's exact launch consts
   (NBLK/SPLITS/… for the pool/context chosen) are only checked when the first
   GPU round runs. Watch for “no captured Triton variant”.
2. **Rank caches are baking now.** Three low-priority systemd user units
   (`ds41-bake-r0/1/2`), one per Spark, capped at 3 GiB and `ionice` best-effort;
   ~2 h at the observed TrueNAS rate. `wait-bakes.sh` verifies each when done.
3. **No GPU run yet**: `glm53-flash-tf` on spark-1/2 and `tf-dsv41-rank0` on
   spark-3 hold the memory. `tp3-start.sh check` will fail until they are
   stopped; the script does not stop them.
4. **Fabric**: first run is TCP over the management LAN (1 GbE); the 10.10
   ConnectX links are point-to-point/ring and need their own address plan. RoCE
   and a 3-rank RDMA ring are later.
5. **Memory** 96.82 GiB/rank leaves ~15–20 GiB for KV/arena on a 128 GB Spark:
   the defaults are 262k context / 1M pool / 4 streams; measure before raising
   them.

## Next step

```sh
cd ~/ds41-tp3
./tp3-start.sh check          # will fail on memory + unfinished caches
# when the bakes finish and you are ready to free the Sparks:
#   spark-1/2: docker rm -f glm53-flash-tf
#   spark-3:   docker rm -f tf-dsv41-rank0
./tp3-start.sh start
curl -s localhost:18888/v1/models
```

For the first GPU smoke: `DRAFTS=0`, one greedy request, then
`tools/validate.py` (adapted 4→3) and a needle check at 32k/128k. Turn on
`DRAFTS=1` afterwards and compare drafted vs undrafted.


## Safety notes

- `glm53-flash-tf` on spark-1/2 and `tf-dsv41-rank0` (EXL3) on spark-3 were
  never touched; builds were `--memory=6g -j1`, bakes are 3 GiB-capped and
  `ionice` best-effort.
- The 473 GB checkpoint stays on TrueNAS; only the per-rank caches (~104 GB
  each) are written to local NVMe. `model-tp3/` is symlinks only.
- `out/` is 145 MB; `kit-tp3/` is ~690 MB (rope + merged AOT); `engine-baseline/`
  is 41 MB.

## Serving fix (2026-11-02): prefill tail chunk — MissingTritonVariant

**Symptom.** Any request whose prefill leaves a **1..16-row tail chunk** (`len % 2048`
in 1..16, or that many uncached tokens) died instantly with `MissingTritonVariant`
(`tokens=0`), at any context long enough to select all `index_topk` keys. Examples:
`92175 = 45*2048 + 15`, `92176` (tail 16), `2063` (tail 15). Clients just retried.

**Cause.** `prompt.zig` took the *round* path for such a chunk (`ring_mode = n <=
RING_EXTRA` with `RING_EXTRA = 16`) plus split attention (`rows <= decode_rows`), so it
launched `_sparse_attn_part` with `RING=1 SPLITS=8 FINAL=0 HAS_BASE=0 NBLK=20`. No
shipped kit recorded that shape: the base+LakyFx kit has `HAS_BASE=0` only at `NBLK=5`
(short contexts) and `NBLK=20` only with row bases or without ring mode. The H=32/H=16
lanes do have it — it was simply never merged for `H=24`/TP3. The engine has no JIT, so
the launch aborts.

**Fixes (independent of each other, both in place).**

1. `kit-extra/aot` — the missing shape baked from the donor source
   (`tools/bake_extra_aot.py`, Triton 3.7.1, same conventions as the packer: 4 warps,
   1 CTA, no scratch, PDL, ints not assumed divisible; deterministic on re-bake).
   `merge_aot.py` takes it as a 4th input and asserts the shape in its coverage check;
   the kit is now **643** kernels and `tp3-start.sh check` requires the cubin
   (`TAIL_CUBIN=75347c8c…`). `tools/check_dispatch.py` replays the engine's dispatch.
2. Engine (`tp3-changes.patch`) — prompt chunks always take the chunk path
   (`Attn.prompt`, no ring mode, one pass), so every chunk, tail included, launches the
   `RING=0 SPLITS=1 FINAL=1` family the kit bakes for every `NBLK` (5,6,8,12,20,36).
   Removed the now-dead `rslot`/`pm_`/`pl_`/`po_` chunk buffers (~6.3 MB) and, with
   ring mode gone, the replay `floor` (`model.zig:836`) is now honoured for ≤16-row
   post-`replayCut` chunks (ring mode masked only `wp >= 0`, so they could read
   pre-replay keys).

**Verified end to end** (stack restarted, log clean): 2052 / 2058 / 2063-token prompts
(tails 4 / 10 / 15), the exact former failure `92175`, and `198664` (tail 8, resumed
from a 90 112-token cache hit) — all served. The patched engine's launch shape selects an
*existing* cubin (`9244f503…`) in the old 642-kernel kit, i.e. fix 2 does not depend on
fix 1.

## Peer-engine integration (2026-11-15)

Reviewed every public DeepSeek-V4.1-Flash serving repo worth reading (pico-dgx-ds4,
jayleaton/deepseek-v41-tensorfold-spark, spark-ds41f, both MiaAI-Lab TensorFold kits,
bertholomus TP2, LakyFX TP3/TP4, upstream TensorFold) and sorted each mechanism into
*already have*, *port now*, *needs a kernel bake*, *not applicable to TP3*:
**[docs/UPSTREAM-INTEGRATION.md](docs/UPSTREAM-INTEGRATION.md)** has the table and the porting queue.

Two things that came out of it and are fixed/applied here:

1. `TF_DS_DRAFT_MS=25` was hardcoded in `tp3-start.sh`. That is the round-overhead term the
   draft-depth optimiser prices (`chooseKWith`); 25 ms is high for three ranks and quietly
   makes the engine verify fewer drafts. It is now `DRAFT_MS`, overridable, and
   [docs/TP3-TUNING.md](docs/TP3-TUNING.md) says to measure it first.
2. Tuning is now an overlay (`tp3-tune.env`, git-ignored, example committed) plus
   `CLOCK_CAP` (GB10 thermal cap, the value peers run) and `EXTRA_ENVS` for one-lever-per-boot
   sweeps, with `tools/tp3-ab.sh` writing medians to `results/ab-<label>.json`.

Already-in-tree, verified while reviewing (no change needed): the index skip when dense top-k
covers every entry (`round.zig:612`, jayleaton `0058`), the round-overhead term itself
(`chooseKWith`), and the relay-free RDMA all-gather (`rdma.zig` relays only unreachable peers,
i.e. pico's `RING_DIRECT` case is the default when the fabric routes every peer).

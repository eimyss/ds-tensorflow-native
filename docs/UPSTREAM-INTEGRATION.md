# Peer-engine integration review — what other DeepSeek-V4.1-Flash repos do, and what lands here

Date: 2026-11-15 · Branch: `integration/peer-tuning-2026-11-15`
Scope: this repo's engine (`src/engine` = TensorFold native Zig, bertholomus DeepSeek family
`zig/src/families/dsv41/`, + 43 Mia patches + our TP3 port) against every public DeepSeek-V4.1-Flash
serving repo that is worth reading.

Read this as a **triage**, not a wish list: every item below is marked with whether it is already in
our tree, portable as-is, gated on a kernel bake, or not applicable to a 3-rank TP3 deployment.

---

## 1. Our baseline (so the deltas mean something)

| | Value | Source |
|---|---|---|
| Engine | TensorFold native Zig, pinned `5940208`, + 43 Mia patches + `tp3-changes.patch` | `STATUS.md` |
| Family | bertholomus `dsv41` (own clean-room family, not upstream `deepseek_v41`) | `src/engine/ATTRIBUTION.md` |
| Knob namespace | `TF_DS_*` (83 `getenv` sites), ~60 runtime knobs | `grep -rhoE 'TF_DS_[A-Z0-9_]+'` |
| Launch profile | ctx 262,144 · pool 1,048,576 · 4 streams · round rows 48 · drafts adaptive · `DRAFT_MS=25` | `tp3-start.sh` |
| Kit | 643 AOT kernels, no JIT — an unbaked launch shape aborts (`MissingTritonVariant`) | `STATUS.md` |

Two structural facts drive everything below:

1. **Our family is not upstream's.** jayleaton's 88-patch series and upstream's `deepseek_v41` tree
   (`block_index.zig`, `forward_prefill.zig`, `idx_split.zig`, `dspark_*.zig`) do not apply mechanically:
   our family is `prompt.zig / round.zig / tri_*.zig / draft.zig / exl3_*.zig`. Their patches are
   **mechanism documents**, not mergeable diffs.
2. **No JIT.** Any port that changes a launch constellation needs cubins
   (`tools/bake_extra_aot.py` → `tools/merge_aot.py` → `tools/check_dispatch.py`).
   That is the single largest cost of every kernel-level item, and exactly what bit us with the
   prefill-tail chunk.

---

## 2. Sources surveyed

| Repo | Lineage | What it is worth to us |
|---|---|---|
| `Picom-code/pico-dgx-ds4` | fork of `blaesus/dsgx10` — C++/CUDA, **not** TensorFold | Best public *system-level* tuning log (slots, wide steps, draft budget, direct ring). Mechanisms transfer as ideas; code does not. |
| `jayleaton/deepseek-v41-tensorfold-spark` | upstream TensorFold `f212eeabd` + 88 patches, Zig | Closest to us in model+hardware, richest in DeepSeek-specific engine mechanisms (ROT_FOLD, KV_MIRROR, IDX_SPLIT, IDX_IOTA, depth overhead). Also the best *engine docs*. |
| `MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks` | source of our 43 patches | We already carry most of it. Worth re-reading for patch `0001-c7-indexer-split`. |
| `MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold` | TensorFold, 4-node switchless | TP4 counterpart of our TP3 port; launch/profile hygiene. |
| `bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10` | our family's origin (TP2) | Where ROT_FOLD / idx_all / depth-pricing came from; TP2 numbers as sanity bounds. |
| `christopherowen/spark-ds41f` | vLLM + TileLang + sparknet | Not portable code, but the best **methodology** (one lever per boot, unfavourable runs kept) and the NCCL/SM121 fabric notes. |
| `LakyFx` (vendored at `src/laky`) | bertholomus Zig base `6a1af414` — **same lineage as us** | The only other public **TP3** profile. S12/R32, 8M pool / 2 GiB snapshot, 4K prefill chunk, ctx 401,408. Closest tuning reference. |
| `tonyd2wild/...TP2-and-TP4`, `Elevatormusic/...EXL3-2x`, `raspy135`, `drowzeys` | various | Isolated mechanisms only (dual-rail prefill, adaptive draft depth, prefix cache). |
| `ashhart/TensorFold` (upstream) | the engine itself | 2,500+ commits newer than our pin; the real long-term answer (§5). |
| `Shoestring-dev/StrixFold`, `plotarmordev/drama-tensorfold`, `HawkBearPig/dgpp` | HIP / non-DeepSeek / other engine | Not applicable. |

---

## 3. Verdict table

Legend — **HAVE**: already in our tree · **PORT**: portable now, no bake · **BAKE**: needs cubins + GPU gate ·
**N/A**: not worth it here.

| # | Mechanism | Where it comes from | Reported effect (their words) | Verdict for TP3 |
|---|---|---|---|---|
| 1 | Index skip when dense top-k covers every entry (`IDX_IOTA`) | jayleaton `0058` (after bertholomus `idx_all/iotaVis`) | drops query-projection, rope and score launches on short windows, byte-identical top-k | **HAVE** — `round.zig:612` `idx_all` + `ops.zig iotaVis`. Verify it fires on TP3 (`TF_DS_IDX_RB`/bucket interplay). |
| 2 | Round-overhead term in draft-depth pricing | jayleaton `0056` (credits bertholomus) | moves depth optimum up when the modelled ms omits host time | **HAVE, but mis-set** — `chooseKWith(..., dms)` uses `TF_DS_DRAFT_MS`, which `tp3-start.sh` pins at **25 ms** for a *three*-node cluster. See §4.1: measure, don't guess. |
| 3 | No-relay collective (direct ring writes) | pico `DSGB10_RING_DIRECT`, jayleaton `stack5 #16` | 10 KB all-reduce 14.45 → 9.89 µs; step −0.21…−0.59 ms; +1.3…1.8 % c1 | **HAVE (implicitly)** — `rdma.zig` relays only peers no device routes. **Action:** assert `n_relayers == 0` on our fabric, i.e. verify we are in the direct case, not silently relaying. |
| 4 | RotOut fold: `wo_b` input rotation into `wo_a` epilogue | jayleaton `0059` (bertholomus) | one `seg_rot_in` launch removed per layer per rank per round | **PORT (bake)** — our `rot_in` lives in `exl3_experts.zig`/`exl3_prefill.zig`; needs a `dense3` variant with an extra bf16 store + cubins. Best single kernel-level win. |
| 5 | RotOut fold: `wo_a` rotation into the attention split-merge | jayleaton `0062` | second `rot_in` launch removed per layer per rank per round | **PORT (bake)** — `tri_attn.zig` split-merge variant; gated on 128-column group grid. Pairs with #4. |
| 6 | Replicated mirror of a slot's first compressed KV rows (`KV_MIRROR`) | jayleaton `0063` | short windows skip the split-KV exchange (`kx_dense`/`kx_union`) entirely | **PORT (bake)** — needs a second store into a per-slot mirror + memory; big win for short/decode-heavy rounds on 3 ranks. |
| 7 | Split long prefill index selections by rows (`IDX_SPLIT`) | jayleaton `0049`–`0053`; Mia `0001-c7-indexer-split` (idea: drowzeys); vLLM #54951 | each rank selects only its rows of an index layer, int32 rows all-gathered before counts | **PORT (bake)** — new split programs + ragged gather plan; pays on long prompts (>256 rows), i.e. exactly our 262K-context prefill. |
| 8 | Prefill index selection split by rows alone / agreed at TP bootstrap | jayleaton `0051` | makes the split independent of rank-local index budget | fold into #7. |
| 9 | Suppression of copies beside DSpark | jayleaton `0047` | fewer draft copies when DSpark already covers them | **PORT** (policy, `draft.zig`) but verify accepted length on TP3 first. |
| 10 | DSpark staging int64 alignment (>12 live slots) | jayleaton `0088` | panic on 16 slots, second staging region misaligned | **CHECK** — our `draft2d.zig`/`tri_markov.zig` staging has no `region_bytes` formula, so the bug as written is absent; re-audit if we go past 12 streams. |
| 11 | 16 slots, big/small slot units, smallest-fit admission | pico change #3 | c16 172 → 234–247 tok/s | **N/A for c1 goal**; interesting only if we ever serve >4 streams. Our `TF_DS_STREAMS=4` is memory-bound first (`STATUS.md` §limitations). |
| 12 | Wide batch step 17–64 rows (`WIDE=64`) | pico #7–#10 | c16 296 → 438 | **HAVE-equivalent** — `TF_DS_ROUND_ROWS=48` already batches rounds; pico's gate (`WIDE_COST`) is the same idea as our per-row `ROUND_MS` table. |
| 13 | One-pass drafter / cross-session draft budget | pico #6, #9 | c16 +6, agent +4.9 % | **PARTIAL** — our `passSteps()` derives batch depth from live streams; `TF_DS_DRAFT_LINK` is the join. No new code needed, tune instead. |
| 14 | Expert re-routing `OEA` (batch-aware MoE) | pico #5 | c16 253 → 286 | **N/A** — explicitly lossy (their own qeval gate). We keep bit-exact outputs. |
| 15 | 17-bit block index → 1M context | pico #1 | 512K → 1M context | **N/A as a port**; if we want >262K we must first prove the kit covers the extra `NBLK` shapes (our tail-chunk lesson). |
| 16 | Front-end body limit 2 MB → 64 MB | pico #2 | 413 above ~400K tokens | **CHECK** — cheap, and a prerequisite for any large-context experiment. |
| 17 | Certified / coarse-then-exact LM head (`CERT`) | jayleaton `stack4 #15` | head −0.18…−0.28 ms/step, bit-exact | **PORT (bake)** — attractive for single-stream decode; our head path is `draft.zig`/`model.zig`. |
| 18 | Confidence-MLP draft gate | jayleaton `stack4 #14` | +1.2 % chat | **N/A now** — needs offline training data; only after #2 is measured. |
| 19 | NCCL 2.30.7 rebuilt with the IB send-path fence fix for SM121 | spark-ds41f | avoids fabric stalls under load | **EVALUATE** — we pin 2.29.7 ring branch (`LD_PRELOAD`). |
| 20 | One-shot RoCE collectives (sparknet) | spark-ds41f | collective cost ↓ | **HAVE-equivalent** — our `rdma.zig` one-hop all-gather. |
| 21 | Methodology: one lever per boot, keep the unfavourable runs | spark-ds41f | — | **ADOPT** — `tools/tp3-ab.sh` in this branch. |
| 22 | GPU clock cap 2100 MHz | jayleaton | thermal headroom on GB10 | **ADOPT** — `CLOCK_CAP` in `tp3-start.sh`. |

---

## 4. What this branch actually changes

### 4.1 Runtime tuning (no rebuild, reversible)

| Knob | Now | Peer value | Why |
|---|---|---|---|
| `TF_DS_DRAFT_MS` | **25** | 6.6 (our own four-node default), jayleaton adds an equivalent term and *measures* it | 25 ms says "a round costs a lot past its forward", which makes `chooseKWith` verify fewer drafts. On TP3 that is likely 8–15 ms. A too-high value silently throws away the speculative gain — measure with `TF_DS_CLOCK_MARKS=1` + `TF_DS_PASS_MS`. |
| `TF_DS_CONF_CAL` | `1.0,0.19` (fit on the **four**-node lane) | recalibrated per cluster | TP3 has different accept statistics; a stale fit biases depth. |
| `CLOCK_CAP` | unset | 2100 MHz (jayleaton) | GB10 throttling shows up as decode variance, not as a clean slowdown. |
| `TF_DS_PREFILL_CHUNK` | engine default | 4K (LakyFX TP3), 8K (their TP4) | chunk size trades TTFT vs peak memory; 4K is the only TP3-qualified value in the wild. |
| `TF_DS_L2_MOE_MB` / `_DN_MB` / `_DEC_MB` / `_ATTN_MB` | engine defaults | — | L2 prefetch sizing is the cheapest knob to sweep on a 3-node ring; `TP3-TUNING.md` has the order. |
| `TF_DS_ROUND_ROWS` | 48 | LakyFX keeps **16** on TP3 | wide steps help multi-stream, and can hurt single-stream tail latency; A/B it. |

### 4.2 Porting queue (needs a bake + a GPU gate, in this order)

1. **#4 + #5 ROT_FOLD** — highest value/lowest risk: removes two launches per layer per rank per round,
   outputs bit-identical by construction (the fold reproduces `seg_rot_in_kernel`'s rounding exactly).
   Gate: `tf-dsv41-test rot-fold` / `rot-fold-attn` twins + 320-prompt identity run.
2. **#6 KV_MIRROR** — removes the split-KV exchange on short windows; needs mirror memory per slot
   (`positions` rounded up to page × world) and a boot byte count.
3. **#7 IDX_SPLIT** — long-prompt prefill scaling; the ragged gather plan is the risky part.
   Gate: unequal-index-budget ranks must issue identical collectives.
4. **#17 CERT head** — single-stream decode; largest measurement discipline required.

Each of these needs: cubins baked (`tools/bake_extra_aot.py`), `merge_aot.py` coverage assertion updated,
`tp3-start.sh check` cubin name added, and `tools/check_dispatch.py` replay before a GPU run.

### 4.3 Deliberately not done

- No engine source was modified in this branch. A fold/kernel port cannot be verified without a GPU, and
  an untested launch-const change aborts the engine outright (`MissingTritonVariant`) — the exact failure
  mode we already paid for once.
- No upstream rebase. Our family is not upstream's; that decision (§5) deserves its own branch.

---

## 5. The strategic option: rebase instead of port

jayleaton runs upstream `f212eeabd` + 88 patches: **2,517 commits newer** than our pin, with IDX_SPLIT,
ROT_FOLD, KV_MIRROR, prompt-cache retention and a far richer knob surface (`TF_DSV41_*`, ~250 knobs)
already implemented and qualified on the same weights.

Cost: our TP3 port (padding 72 heads / 9 groups / 129,408 vocab, `addN_bf16` Engram sum, the padded
rank cache) plus Mia's 43 patches would have to be re-expressed on the upstream family — a rewrite of
`tp3-changes.patch`, not a cherry-pick.

A middle path that is cheap to evaluate: stand up one **single-rank** upstream build on one Spark against
an EXL3 pack, and measure `TF_DSV41_*` defaults (pool, prefill chunk, graph bucket) with the same harness.
If upstream's single-stream decode beats our TP3 c1 at equal context, the rebase is justified; if not,
keep porting mechanisms into the bertholomus family (#4 first).

---

## 6. Verification checklist for anything from this document

1. `zig build test` — 128/129 is the known-good baseline (one pre-existing `lanes` failure).
2. `tp3-start.sh check` — 643 cubins, tail cubin hash, padded config marker.
3. `tools/check_dispatch.py` — replay the dispatch of the shapes you intend to run.
4. Drafted vs undrafted equality on 320 held-out prompts (identity set) — bit-for-bit.
5. Needle at 32K / 128K, `TF_DS_TOPK_FUSED_MAX` untouched.
6. Only then: the A/B harness, one lever per boot, medians of ≥9 runs, unfavourable runs kept.

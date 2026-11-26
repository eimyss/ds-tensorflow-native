#!/usr/bin/env python3
"""Merge the TP3 Triton AOT variants of a donor kit into a base kit.

Only donor kernels whose (function, parameter names/types, const names) signature
already exists in the base are taken: the engine's launch sites then accept them
(patch 0004's "general" pass relaxes nospec/div16 differences).  Constexpr values
(H=24, HB=8, WORLD=3) are not relaxed by that pass, which is exactly why these
variants must be present.

An optional fourth directory holds launch shapes neither shipped kit recorded (baked by
tools/bake_extra_aot.py); its entries are added unconditionally, as the launch sites
request exactly those constexpr values.

usage: merge_aot.py BASE_KIT_DIR DONOR_KIT_DIR OUT_KIT_DIR [EXTRA_KIT_DIR]
"""
import collections
import json
import os
import shutil
import sys


def sig(k):
    return (
        tuple((p["name"], p["type"]) for p in k["params"]),
        tuple(sorted(k["consts"].keys())),
    )


def cvals(k):
    return {n: v.get("int") for n, v in k["consts"].items()}


def main():
    base_dir, donor_dir, out_dir = sys.argv[1:4]
    extra_dir = sys.argv[4] if len(sys.argv) > 4 else None
    base = json.load(open(os.path.join(base_dir, "aot.json")))["kernels"]
    donor = json.load(open(os.path.join(donor_dir, "aot.json")))["kernels"]

    base_sigs = collections.defaultdict(set)
    for k in base:
        base_sigs[k["fn"]].add(sig(k))

    seen = {(k["fn"], json.dumps(cvals(k), sort_keys=True), k["hash"]) for k in base}
    out = list(base)
    added = skipped = 0
    for k in donor:
        if k["fn"] not in base_sigs or sig(k) not in base_sigs[k["fn"]]:
            skipped += 1
            continue
        key = (k["fn"], json.dumps(cvals(k), sort_keys=True), k["hash"])
        if key in seen:
            continue
        seen.add(key)
        out.append(k)
        added += 1

    # the extra kit: launch shapes no shipped kit recorded, added as baked (same signature
    # as the donor family they extend, so the launch sites accept them)
    extra = json.load(open(os.path.join(extra_dir, "aot.json")))["kernels"] if extra_dir else []
    for k in extra:
        key = (k["fn"], json.dumps(cvals(k), sort_keys=True), k["hash"])
        if key in seen:
            continue
        seen.add(key)
        out.append(k)

    os.makedirs(os.path.join(out_dir, "cubins"), exist_ok=True)
    missing = []
    for k in out:
        name = k["hash"] + ".cubin"
        dst = os.path.join(out_dir, "cubins", name)
        if os.path.exists(dst):
            continue
        for d in (base_dir, donor_dir) + ((extra_dir,) if extra_dir else ()):
            src = os.path.join(d, "cubins", name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                break
        else:
            missing.append(name)
    if missing:
        sys.exit("missing cubins: %s" % ", ".join(missing[:8]))

    doc = {
        "generator": "tensorfold TP3 merge (base kit + LakyFx TP3 variants + baked extras)",
        "kernels": out,
    }
    with open(os.path.join(out_dir, "aot.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")

    # What a TP3 run must be able to launch, and whether it can.
    need = [
        ("_sparse_attn_part", {"H": 24, "HB": 8}),
        # the prompt chunk a prefill's tail leaves (rows <= RING_EXTRA: ring mode, split pass) at a
        # context long enough to select all index_topk keys -- its own specialization, not the
        # rounds' (no per-row bases) and not a full chunk's (one split, no ring)
        ("_sparse_attn_part", {"H": 24, "HB": 8, "RING": 1, "SPLITS": 8, "FINAL": 0, "HAS_BASE": 0,
                               "HAS_COMP": 1, "PACKED": 1, "NBLK": 20}),
        ("_sparse_attn_merge", {"H": 24, "HB": 8}),
        ("_sparse_attn_merge_rot2", {"H": 24, "HB": 8}),
        ("_rope_heads", {"H": 24, "HB": 8}),
        ("_hc_mix_part", {"WORLD": 3}),
        ("_hc_mix_pf", {"WORLD": 3}),
        ("_hc_post", {"WORLD": 3}),
        ("_hc_post_only", {"WORLD": 3}),
        ("_pick", {"WORLD": 3}),
    ]
    ok = True
    for fn, req in need:
        n = 0
        for k in out:
            if k["fn"] != fn:
                continue
            c = cvals(k)
            if all(c.get(a) == b for a, b in req.items()):
                n += 1
        ok = ok and n > 0
        print("%-26s %-32s %d variant(s)" % (fn, req, n))
    print("merged: base %d + donor %d + extra %d = %d (donor skipped: %d)" % (
        len(base), added, len(extra), len(out), skipped))
    print("TP3 coverage:", "OK" if ok else "MISSING")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bake Triton AOT variants the merged kit is missing.

The native engine has no JIT: every launch must match a cubin in the kit's aot.json
(see src/engine/zig/src/cuda/aot.zig).  A launch shape the shipped kit never recorded
therefore aborts the request with MissingTritonVariant.  This compiles such a shape
from the donor kernels' own source and writes it as a small extra kit that
tools/merge_aot.py merges into kit-tp3.

The compile reproduces the donor packer's conventions (see
src/laky/tensorfold/prefill-128/{README,SOURCE}.json): the kernel source as shipped,
four warps, one CTA, no scratch, PDL on, and no divisibility assumption on the int
arguments (they are passed as values that are neither 1 nor multiples of 16, exactly
what the kit's `"div16": false, "nospec": true` entries record).

usage: bake_extra_aot.py OUT_KIT_DIR      (run inside the build image, on a GPU host)
"""
import hashlib
import importlib.util
import json
import os
import sys

SRC = "/work/src/laky/tensorfold/prefill-128/kernels.py"

# One entry per launch shape to bake: the constexprs of the kernel launch (the kit's
# `consts`).  Everything else (args, grid, specialization) is fixed below.
VARIANTS = [
    {
        "fn": "_sparse_attn_part",
        "consts": {
            "H": 24, "HD": 512, "HB": 8, "WIN": 128, "BN": 32,
            "RING": 1, "HAS_COMP": 1, "PACKED": 1, "SPLITS": 8, "NBLK": 20,
            "FINAL": 0, "HAS_BASE": 0, "PDL": 1, "WPF": 1,
        },
    },
]


def types_of(module):
    import torch
    t = torch
    return {
        "Q": (t.bfloat16, "*bf16"), "WSRC": (t.bfloat16, "*bf16"), "WLO": (t.int64, "*i64"),
        "COMP": (t.uint8, "*u8"), "CSC": (t.uint8, "*u8"), "IDX": (t.int64, "*i64"),
        "POS": (t.int64, "*i64"), "PM": (t.float32, "*fp32"), "PL": (t.float32, "*fp32"),
        "PO": (t.float32, "*fp32"), "SINK": (t.float32, "*fp32"), "OUT": (t.bfloat16, "*bf16"),
        "RBASE": (t.int64, "*i64"), "CBASE": (t.int64, "*i64"),
    }


def sparse_attn_part(module, c):
    """Compile and launch one `_sparse_attn_part` variant; returns the compiled kernel."""
    import torch
    R, H, HD, SPLITS, NIDX, NCOMP, RINGROWS = 15, c["H"], c["HD"], c["SPLITS"], 511, 8, 145
    HB = c["HB"]

    def buf(*shape, dtype):
        return torch.zeros(*shape, dtype=dtype, device="cuda")

    args = dict(
        Q=buf(R, H, HD, dtype=torch.bfloat16), WSRC=buf(RINGROWS, HD, dtype=torch.bfloat16),
        WLO=buf(1, dtype=torch.int64), COMP=buf(NCOMP, HD // 2, dtype=torch.uint8),
        CSC=buf(NCOMP, HD // 16, dtype=torch.uint8), IDX=buf(R, NIDX, dtype=torch.int64),
        POS=buf(R, dtype=torch.int64), PM=buf(16, H, SPLITS, dtype=torch.float32),
        PL=buf(16, H, SPLITS, dtype=torch.float32), PO=buf(16, H, SPLITS, HD, dtype=torch.float32),
        SINK=buf(H, dtype=torch.float32), OUT=buf(R, H, HD, dtype=torch.bfloat16),
        RBASE=buf(R, dtype=torch.int64), CBASE=buf(R, dtype=torch.int64),
    )
    return module._sparse_attn_part[(R, H // HB, SPLITS)](
        *(args[n] for n in ("Q", "WSRC", "WLO", "COMP", "CSC", "IDX", "POS", "PM", "PL", "PO",
                            "SINK", "OUT")),
        0.044194173,          # scale: never specialized
        RINGROWS,             # ring_size: 145, so Triton assumes no divisibility (as the packer did)
        NIDX,                 # n_idx: 511, ditto
        args["RBASE"], args["CBASE"],
        **{k: c[k] for k in ("H", "HD", "HB", "WIN", "BN", "RING", "HAS_COMP", "PACKED",
                             "SPLITS", "NBLK", "FINAL", "HAS_BASE", "PDL", "WPF")},
        launch_pdl=bool(c["PDL"]),
    )


def entry(fn, consts, kernel, spec_types):
    """The kit's aot.json entry for a compiled kernel (same shape as the merged ones)."""
    attrs = getattr(kernel.src, "attrs", {})
    params = []
    for i, name in enumerate(("Q", "WSRC", "WLO", "COMP", "CSC", "IDX", "POS", "PM", "PL", "PO",
                              "SINK", "OUT", "scale", "ring_size", "n_idx", "RBASE", "CBASE")):
        if name == "scale":
            params.append({"name": name, "type": "fp32", "div16": False, "nospec": True})
        elif name in ("ring_size", "n_idx"):
            params.append({"name": name, "type": "i32", "div16": False, "nospec": True})
        else:
            hints = attrs.get((i,), [])
            div16 = any(h[0] == "tt.divisibility" and h[1] == 16 for h in hints)
            params.append({"name": name, "type": spec_types[name][1], "div16": div16, "nospec": True})
    md = kernel.metadata
    return {
        "fn": fn, "hash": kernel.hash, "name": md.name,
        "num_warps": md.num_warps, "num_ctas": md.num_ctas, "shared": md.shared,
        "global_scratch": md.global_scratch_size, "global_align": md.global_scratch_align,
        "profile_scratch": md.profile_scratch_size, "pdl": bool(md.launch_pdl),
        "params": params,
        "consts": {k: {"int": v} for k, v in consts.items()},
    }


def main():
    out_dir = sys.argv[1]
    spec = importlib.util.spec_from_file_location("donor_kernels", SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    types = types_of(module)

    os.makedirs(os.path.join(out_dir, "cubins"), exist_ok=True)
    path = os.path.join(out_dir, "aot.json")
    existing = json.load(open(path))["kernels"] if os.path.exists(path) else []
    by_hash = {k["hash"]: k for k in existing}
    kernels = list(existing)

    for v in VARIANTS:
        kernel = sparse_attn_part(module, v["consts"])
        e = entry(v["fn"], v["consts"], kernel, types)
        if e["hash"] in by_hash and by_hash[e["hash"]] != e:
            sys.exit("hash collision with a different entry: %s" % e["hash"])
        cubin = kernel.asm["cubin"]
        with open(os.path.join(out_dir, "cubins", e["hash"] + ".cubin"), "wb") as f:
            f.write(cubin)
        if e["hash"] not in by_hash:
            kernels.append(e)
        print("%s %s -> %s.cubin (%d bytes, %d B shared, %d regs, %d spills)" % (
            e["fn"], json.dumps(v["consts"], sort_keys=True), e["hash"][:12], len(cubin),
            e["shared"], kernel.n_regs, kernel.n_spills))
        print("   cubin sha256: %s" % hashlib.sha256(cubin).hexdigest())

    with open(path, "w") as f:
        json.dump({"generator": "bake_extra_aot.py (launch shapes the merged kit lacks)",
                   "kernels": kernels}, f, indent=1)
        f.write("\n")
    print("wrote %s (%d variant(s))" % (path, len(kernels)))


if __name__ == "__main__":
    main()

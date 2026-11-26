#!/usr/bin/env python3
"""Which kit cubin would the engine launch for a `_sparse_attn_part` shape?

The native engine has no JIT: a launch aborts with MissingTritonVariant unless an entry of
the kit's aot.json matches its constexprs and its arguments' specialization exactly (the
`find`/`matches` pair in src/engine/zig/src/cuda/aot.zig).  This replays that dispatch over
a kit, so a kit can be checked without bringing the stack up -- e.g. after a rebuild, or to
tell whether a given launch shape is covered at all.

usage: check_dispatch.py KIT_AOT_DIR [CONSTS_JSON]

  KIT_AOT_DIR   a kit's aot directory (the one holding aot.json + cubins/)
  CONSTS_JSON   the launch's constexprs, e.g. '{"H":24,"HB":8,...}'; the default is the
                prefill tail chunk's (1..RING_EXTRA rows: ring mode, split pass, no bases)
                at a context long enough to select all index_topk keys (NBLK 20)

Exits 0 and prints the selected entry when one matches, 1 otherwise.
"""
import json
import sys

DEFAULT_CONSTS = {"H": 24, "HD": 512, "HB": 8, "WIN": 128, "BN": 32, "RING": 1, "HAS_COMP": 1,
                  "PACKED": 1, "SPLITS": 8, "NBLK": 20, "FINAL": 0, "HAS_BASE": 0, "PDL": 1, "WPF": 1}


def launch_args(consts):
    """sparseAttn's arguments for this shape (families/dsv41/tri_attn.zig), in the kernel's order."""
    ptr = lambda t: {"kind": "ptr", "ty": t, "addr": 0x1000}      # every served pointer is 16-aligned
    one_pass = consts.get("FINAL")
    return [
        ("Q", ptr("*bf16")), ("WSRC", ptr("*bf16")), ("WLO", ptr("*i64")), ("COMP", ptr("*u8")),
        ("CSC", ptr("*u8")), ("IDX", ptr("*i64")), ("POS", ptr("*i64")),
        # one pass writes OUT directly: the partials are bf16 (a split pass's are fp32)
        ("PM", ptr("*bf16" if one_pass else "*fp32")),
        ("PL", ptr("*bf16" if one_pass else "*fp32")),
        ("PO", ptr("*bf16" if one_pass else "*fp32")),
        ("SINK", ptr("*fp32")), ("OUT", ptr("*bf16")),
        ("scale", {"kind": "f32", "v": 0.044194173}),
        # ring launches pass the ring's size (window + RING_EXTRA), others 16 (unused)
        ("ring_size", {"kind": "i32", "v": 144 if consts.get("RING") else 16}),
        ("n_idx", {"kind": "i32", "v": 512}),                    # index_topk, once the context is long
        ("RBASE", ptr("*i64")), ("CBASE", ptr("*i64")),
    ]


def assumes(compiled, holds, general):
    return compiled == holds or (general and not compiled)


def matches(k, consts, args, general):
    for name, want in consts.items():
        if k["consts"].get(name, {}).get("int") != want:
            return False
    params = {p["name"]: p for p in k["params"]}
    runtime = 0
    for name, a in args:
        p = params.get(name)
        if a["kind"] == "i32":
            if p is None:                       # folded: only the value 1 is ever a constexpr
                if a["v"] != 1 or k["consts"].get(name, {}).get("int") != 1:
                    return False
                continue
            if p["type"] != "i32":
                return False
            if not general and not p["nospec"] and a["v"] == 1:
                return False
            if not assumes(p["div16"], (not p["nospec"]) and a["v"] % 16 == 0, general):
                return False
        elif a["kind"] == "ptr":
            if p is None or p["type"] != a["ty"] or not assumes(p["div16"], a["addr"] % 16 == 0, general):
                return False
        elif a["kind"] == "f32":
            if p is None or p["type"] != "fp32":
                return False
        runtime += 1
    return runtime == len(k["params"])


def main():
    kit = sys.argv[1]
    consts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CONSTS
    kernels = json.load(open(kit + "/aot.json"))["kernels"]
    args = launch_args(consts)
    for general in (False, True):
        for k in kernels:
            if k["fn"] != "_sparse_attn_part" or not matches(k, consts, args, general):
                continue
            print("selected in the %s pass: %s (%s, %d B shared, pdl=%s)" % (
                "general" if general else "exact", k["hash"][:12], k["name"], k["shared"], k["pdl"]))
            print("   consts:", {n: v["int"] for n, v in sorted(k["consts"].items())})
            return 0
    print("no variant matches this launch: the engine would raise MissingTritonVariant")
    print("   consts:", json.dumps(consts, sort_keys=True))
    return 1


if __name__ == "__main__":
    sys.exit(main())

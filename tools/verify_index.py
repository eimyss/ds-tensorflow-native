#!/usr/bin/env python3
"""Check that the three world-3 rank indexes partition every split tensor exactly.

For every (prefix, part) whose key carries a range, the union of the three
ranks' intervals must be one contiguous [0, n) with no overlap; whole tensors
(None/None) are replicated and skipped.  Also checks each entry's byte count.
"""
import collections
import json
import sys

SIZE = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "I64": 8, "U8": 1,
        "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1,
        "bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}


def load(r):
    return json.load(open("rank%d.index" % r))


def parse(key):
    f = key.split("|")
    if len(f) != 4:
        return f[0], None, None, None
    def rng(x):
        if x == "None":
            return None
        a, b = x[1:-1].split(", ")
        return (int(a), int(b))
    return f[0], rng(f[1]), rng(f[2]), f[3]


def main():
    idx = [load(r) for r in range(3)]
    def names(entries):
        out = set()
        for key, *_ in entries:
            prefix, cols, rows, part = parse(key)
            out.add((prefix, part))
        return out
    ns = [names(x) for x in idx]
    print("same tensor/part names on all three ranks:", ns[0] == ns[1] == ns[2], len(ns[0]))

    bad = 0
    for r, entries in enumerate(idx):
        for key, dtype, shape, off, nbytes in entries:
            want = 1
            for d in shape:
                want *= d
            want *= SIZE.get(dtype, 1)
            if want != nbytes:
                print("rank", r, "byte count", key, dtype, shape, nbytes, "!=", want)
                bad += 1
    print("byte-count mismatches:", bad)

    # prefix|part -> dimension -> list of intervals (one per rank that has it)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r, entries in enumerate(idx):
        for key, dtype, shape, off, nbytes in entries:
            prefix, cols, rows, part = parse(key)
            if part is None:
                continue
            if cols is not None:
                groups[(prefix, part, "cols")][r].append(cols)
            if rows is not None:
                groups[(prefix, part, "rows")][r].append(rows)

    checked = 0
    failures = 0
    for (prefix, part, dim), per_rank in sorted(groups.items()):
        ivals = []
        for r, rs in per_rank.items():
            ivals.extend(rs)
        ivals.sort()
        # merge and require one contiguous run from 0
        end = 0
        ok = True
        for a, b in ivals:
            if a != end:
                ok = False
                break
            end = b
        checked += 1
        if not ok:
            failures += 1
            if failures <= 20:
                print("NOT A PARTITION:", prefix, part, dim, ivals)
    print("split groups checked:", checked, "failures:", failures)
    if failures:
        sys.exit(1)

    for r in range(3):
        total = sum(e[4] for e in idx[r])
        print("rank %d: %d entries, %.2f GiB" % (r, len(idx[r]), total / 2**30))


if __name__ == "__main__":
    main()

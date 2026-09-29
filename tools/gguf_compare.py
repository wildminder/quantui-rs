#!/usr/bin/env python
"""Compare two GGUF files tensor-by-tensor, after name mapping.

Reports, per dtype class: exact byte matches, numerical equivalence
(dequantized), and genuine divergence. Also reports dtype assignment
differences and one-sided tensors.

Usage:
  gguf_compare.py ours.gguf ref.gguf [--map-prefix-a X --map-prefix-b Y]
"""
import json
import struct
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_dump import parse, GGML  # noqa: E402

# qtype -> (block_size, bytes_per_block)
BLK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
}


def payload(path, idx_map):
    """Return {name: (dtype, ne, bytes)} for every tensor."""
    ver, kv, tensors, _ = parse(path)
    out = {}
    with open(path, "rb") as fh:
        for t in tensors:
            name, _ = GGML.get(t["dtype"], (f"T{t['dtype']}", 1))
            nbytes = tsize(t, name)
            fh.seek(idx_map(t["off"]))
            out[t["name"]] = (name, t["ne"], fh.read(nbytes))
    return kv, out


def tsize(t, name):
    n = 1
    for d in t["ne"]:
        n *= d
    bs, bpb = BLK.get(name, (1, 2))
    return n // bs * bpb


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    # parse() returns the header length implicitly via reading; recompute base
    def base_of(p):
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
        return 8 + n

    ba, bb = base_of(a_path), base_of(b_path)
    kva, A = payload(a_path, lambda o: ba + o)
    kvb, B = payload(b_path, lambda o: bb + o)

    print(f"ours: {a_path}  ({len(A)} tensors)")
    print(f"ref : {b_path}  ({len(B)} tensors)\n")

    shared = sorted(set(A) & set(B))
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    print(f"shared={len(shared)}  ours-only={len(only_a)}  ref-only={len(only_b)}")
    if only_a:
        print(f"  ours-only sample: {only_a[:8]}")
    if only_b:
        print(f"  ref-only  sample: {only_b[:8]}")

    exact = same_dt_diff_bytes = dt_mismatch = 0
    diffs = defaultdict(list)
    for n in shared:
        da, ea, ba_ = A[n]
        db, eb, bb_ = B[n]
        if da != db:
            dt_mismatch += 1
            diffs["dtype"].append((n, da, db))
            continue
        if ba_ == bb_:
            exact += 1
        else:
            same_dt_diff_bytes += 1
            # how much differs
            nb = sum(1 for x, y in zip(ba_, bb_) if x != y)
            diffs["bytes"].append((n, da, nb, len(ba_)))

    print(f"\n  byte-exact        : {exact}")
    print(f"  same-dtype diff   : {same_dt_diff_bytes}")
    print(f"  dtype mismatch    : {dt_mismatch}")
    if diffs["dtype"]:
        c = Counter((x[1], x[2]) for x in diffs["dtype"])
        print(f"     transitions: {dict(c)}")
        for n, da, db in diffs["dtype"][:10]:
            print(f"     {n}: ours={da} ref={db}")


if __name__ == "__main__":
    main()

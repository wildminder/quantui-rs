#!/usr/bin/env python
"""Byte-level payload comparison between our GGUF and the reference GGUF,
after mapping our llama-style names onto the reference's native names.

Usage: gguf_bytes.py <ours.gguf> <ref.gguf>
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_dump import parse, GGML  # noqa: E402
from gguf_dequant_cmp import ours_to_ref, load  # noqa: E402


def main():
    A, _ = load(sys.argv[1])
    B, _ = load(sys.argv[2])
    print(f"ours={len(A)} ref={len(B)}")

    same_dt = same_bytes = diff_bytes = dt_mism = 0
    only_ref = only_ours = 0
    diffs = []
    for name, (dt, ne, raw, nel) in A.items():
        rname = ours_to_ref(name)
        if rname not in B:
            only_ours += 1
            continue
        rdt, rne, rraw, rnel = B[rname]
        if dt != rdt:
            dt_mism += 1
            continue
        same_dt += 1
        if raw == rraw:
            same_bytes += 1
        else:
            diff_bytes += 1
            nd = sum(1 for x, y in zip(raw, rraw) if x != y)
            diffs.append((nd, len(raw), name, dt))

    for name in B:
        if name not in {ours_to_ref(k) for k in A}:
            only_ref += 1

    print(f"\n  shared, same dtype      : {same_dt}")
    print(f"    byte-identical        : {same_bytes}")
    print(f"    differing             : {diff_bytes}")
    print(f"  dtype mismatch          : {dt_mism}")
    print(f"  ours-only names         : {only_ours}")
    print(f"  ref-only  names         : {only_ref}")
    if diffs:
        diffs.sort(reverse=True)
        print("\n  worst differing payloads:")
        for nd, tot, name, dt in diffs[:15]:
            print(f"    {name:46s} {dt:6s} {nd}/{tot} bytes differ "
                  f"({100*nd/tot:.4f}%)")


if __name__ == "__main__":
    main()

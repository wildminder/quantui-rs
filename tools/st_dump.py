#!/usr/bin/env python
"""Safetensors header inspector: dtype/shape census + comfy_quant blob dump.

Usage: st_dump.py <file.safetensors> [--blobs] [--cmp <other.safetensors>]
"""
import json
import struct
import sys
from collections import Counter


def read_header(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(n))
    meta = hdr.pop("__metadata__", None)
    return hdr, meta


def main():
    path = sys.argv[1]
    hdr, meta = read_header(path)
    print(f"file   : {path}")
    print(f"tensors: {len(hdr)}")
    if meta:
        print(f"metadata keys: {list(meta)}")
        for k, v in meta.items():
            if len(str(v)) < 400:
                print(f"  {k} = {v}")
            else:
                print(f"  {k} = <{len(str(v))} chars>")
    else:
        print("metadata: (none)")

    dt = Counter(v["dtype"] for v in hdr.values())
    print("\n--- dtype census ---")
    for k, v in sorted(dt.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} {v}")

    # comfy_quant blobs
    blobs = {k: v for k, v in hdr.items() if k.endswith(".comfy_quant")}
    print(f"\n--- comfy_quant blobs: {len(blobs)} ---")
    seen = Counter()
    for k, v in blobs.items():
        try:
            b = json.loads(v["data"]) if isinstance(v.get("data"), (str, bytes)) else v
        except Exception:
            b = v
        sig = json.dumps(b, sort_keys=True)
        seen[sig] += 1
    for sig, c in seen.most_common():
        print(f"  x{c}: {sig[:400]}")

    scales = [k for k in hdr if k.endswith(".weight_scale") or k.endswith(".weight_scale_2")]
    isc = [k for k in hdr if k.endswith(".input_scale")]
    print(f"\nweight_scale*: {len(scales)}  input_scale: {len(isc)}")

    if "--names" in sys.argv:
        print("\n--- all tensors ---")
        for k, v in hdr.items():
            print(f"  {k:64s} {str(v['shape']):22s} {v['dtype']}")

    if "--cmp" in sys.argv:
        other = sys.argv[sys.argv.index("--cmp") + 1]
        h2, m2 = read_header(other)
        print(f"\n=== COMPARE vs {other} ===")
        print(f"  tensors: {len(hdr)} vs {len(h2)}")
        only1 = sorted(set(hdr) - set(h2))
        only2 = sorted(set(h2) - set(hdr))
        print(f"  only in A: {len(only1)}")
        for k in only1[:25]:
            print(f"    {k:60s} {hdr[k]['shape']} {hdr[k]['dtype']}")
        print(f"  only in B: {len(only2)}")
        for k in only2[:25]:
            print(f"    {k:60s} {h2[k]['shape']} {h2[k]['dtype']}")
        diff = [k for k in set(hdr) & set(h2)
                if hdr[k]["shape"] != h2[k]["shape"] or hdr[k]["dtype"] != h2[k]["dtype"]]
        print(f"  shape/dtype differs: {len(diff)}")
        for k in sorted(diff)[:25]:
            print(f"    {k:60s} A={hdr[k]['shape']}/{hdr[k]['dtype']}  B={h2[k]['shape']}/{h2[k]['dtype']}")


if __name__ == "__main__":
    main()

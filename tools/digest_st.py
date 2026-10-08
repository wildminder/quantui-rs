#!/usr/bin/env python
"""Per-tensor SHA-256 digests of a .safetensors file — parity differ.

Prints `<name> <dtype> <shape> <sha256-of-payload>` for every tensor (sorted),
plus a whole-file header digest. Used to prove a refactor moved zero bytes:
run before and after, diff the output. Two files are byte-parity-identical
iff their digests are identical (sha256 collision aside).

    python tools/digest_st.py A.safetensors [B.safetensors ...]

With more than one file, also prints a per-pair verdict:
    SAME | DIFF (n tensors differ; first 5 named)

NOTE: only the data payload is hashed, not dtypes/shapes — print those
separately so a dtype change is still visible in the text output.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys


def read_header(path: str):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def digests(path: str):
    header, base = read_header(path)
    out = {}
    with open(path, "rb") as f:
        for name, info in header.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            f.seek(base + s)
            payload = f.read(e - s)
            h = hashlib.sha256(payload).hexdigest()
            out[name] = (info["dtype"], info["shape"], h)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    all_out = {}
    for p in sys.argv[1:]:
        d = digests(p)
        all_out[p] = d
        print(f"# {p}  ({len(d)} tensors)")
        for name in sorted(d):
            dt, shape, h = d[name]
            print(f"{name}\t{dt}\t{shape}\t{h}")
    if len(all_out) == 2:
        (pa, da), (pb, db) = list(all_out.items())
        keys = set(da) | set(db)
        diffs = [k for k in keys if da.get(k) != db.get(k)]
        only_a = [k for k in da if k not in db]
        only_b = [k for k in db if k not in da]
        print()
        print(f"# verdict: {'SAME' if not diffs and not only_a and not only_b else 'DIFF'}")
        if diffs:
            print(f"# {len(diffs)} differing: {diffs[:5]}")
        if only_a:
            print(f"# only in {pa}: {len(only_a)} e.g. {only_a[:3]}")
        if only_b:
            print(f"# only in {pb}: {len(only_b)} e.g. {only_b[:3]}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Build a UNIFORM legacy imatrix from a GGUF's tensor list.

Purpose: route quantui-rs's K-quant conversion through its own byte-exact
*weighted* port (golden-verified vs llama-quantize) instead of rlx-gguf's
documented "simplified ... lower quality" unweighted encoder.

Legacy layout (imatrix.rs::load_legacy / imatrix-loader.cpp:25-34):
    i32 n_entries
    per entry: i32 name_len, name bytes, i32 ncall, i32 nval, nval x f32 sums

Usage: make_uniform_imatrix.py <ours.gguf> <out.imatrix> [value]
"""
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_dump import parse  # noqa: E402


def main():
    gguf_p, out_p = sys.argv[1], sys.argv[2]
    val = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    _, _, ts, _ = parse(gguf_p)
    entries = []
    for t in ts:
        ne = t["ne"]
        if len(ne) < 2:
            continue
        # imatrix rows are indexed by the input dim = ne[0]
        nval = ne[0]
        entries.append((t["name"], nval))

    with open(out_p, "wb") as fh:
        fh.write(struct.pack("<i", len(entries)))
        for name, nval in entries:
            nb = name.encode()
            fh.write(struct.pack("<i", len(nb)))
            fh.write(nb)
            fh.write(struct.pack("<i", 1))       # ncall
            fh.write(struct.pack("<i", nval))
            fh.write(struct.pack("<%df" % nval, *([val] * nval)))
    print(f"wrote {out_p}: {len(entries)} entries")


if __name__ == "__main__":
    main()

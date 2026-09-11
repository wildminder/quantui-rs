"""Classify tensor-name conventions in a GGUF file.

Counts how many tensor names follow llama.cpp conventions (blk.N.*,
token_embd, output_norm, output.weight) vs HF pass-through (original
names kept unchanged — tokenizers, connectors, vision towers), and
prints the per-group breakdown with examples.

Usage: python classify_gguf_names.py <file.gguf>
"""

import mmap
import re
import struct
import sys
from collections import Counter
from pathlib import Path

LLAMA_STYLE = re.compile(
    r"^(blk\.\d+\.|token_embd|output_norm|output\.weight|token_embd_norm)"
)


def read_str(data, p):
    (n,) = struct.unpack_from("<Q", data, p)
    p += 8
    s = data[p : p + n].decode("utf-8")
    return s, p + n


def main():
    gguf = Path(sys.argv[1])
    f = open(gguf, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0
    magic, ver, n_tensors, n_kv = struct.unpack_from("<IIQQ", data, pos)
    assert magic == 0x46554747, "not a GGUF file"
    pos = 24

    for _ in range(n_kv):
        k, pos = read_str(data, pos)
        (t,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if t == 8:  # string
            _, pos = read_str(data, pos)
        elif t == 9:  # array
            (et,) = struct.unpack_from("<I", data, pos)
            pos += 4
            (n,) = struct.unpack_from("<Q", data, pos)
            pos += 8
            # Element sizes per GGUF v3: BOOL/8-bit=1B, 16-bit=2B,
            # 32-bit=4B, FLOAT64/INT64/UINT64=8B (STRING handled above).
            if et == 8:
                for _ in range(n):
                    _, pos = read_str(data, pos)
            else:
                pos += n * {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                            10: 8, 11: 8, 12: 8}[et]
        elif t in (0, 1, 7):   # uint8 / int8 / bool
            pos += 1
        elif t in (2, 3):      # uint16 / int16
            pos += 2
        elif t in (4, 5, 6):   # uint32 / int32 / float32
            pos += 4
        elif t in (10, 11, 12):  # float64 / int64 / uint64
            pos += 8
        else:
            raise ValueError(f"kv type {t}")

    names = []
    for _ in range(n_tensors):
        nm, pos = read_str(data, pos)
        (nd,) = struct.unpack_from("<I", data, pos)  # n_dims
        pos += 4
        pos += nd * 8  # ne: u64 per dim
        pos += 4  # ggml dtype (u32)
        pos += 8  # offset: u64 (relative to data segment)
        names.append(nm)

    print(f"file: {gguf}  tensors: {len(names)}")
    groups = Counter()
    pt_prefixes = Counter()
    llama_prefixes = Counter()
    for n in names:
        if LLAMA_STYLE.match(n):
            groups["llama.cpp convention"] += 1
            llama_prefixes[n.split(".")[0] if not n.startswith("blk.")
                           else "blk.N"] += 1
        else:
            groups["HF pass-through"] += 1
            pt_prefixes[".".join(n.split(".")[:3])] += 1
    for g, c in groups.most_common():
        print(f"  {g}: {c}")
    print("\ntop HF pass-through prefixes:")
    for p, c in pt_prefixes.most_common(12):
        print(f"  {c:5d}  {p}")
    llama_top = sorted(
        {n for n in names if LLAMA_STYLE.match(n) and not n.startswith("blk.")}
    )
    print("\nllama.cpp top-level names:", llama_top)
    print("\nllama.cpp group prefixes:")
    for p, c in llama_prefixes.most_common(10):
        print(f"  {c:5d}  {p}")


if __name__ == "__main__":
    main()

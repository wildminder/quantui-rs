#!/usr/bin/env python
"""Pure-python GGUF v3 header dumper (no `gguf` package needed).

Prints the KV metadata and a full per-tensor dtype census so a reference GGUF's
quantization *recipe* can be read off directly (which tensors stayed F16/BF16,
which became Q4_0, and in what row-width shape).
"""
import json
import struct
import sys
from collections import Counter, defaultdict

# GGUF value-type ids -> (name, parser)
# 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,8 string,9 array,10 u64,11 i64,12 f64
SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


def read_str(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    s = buf[off:off + n].decode("utf-8", "replace")
    return s, off + n


def read_value(buf, off, t):
    if t in SCALARS:
        fmt, size = SCALARS[t]
        return struct.unpack_from(fmt, buf, off)[0], off + size
    if t == 8:
        return read_str(buf, off)
    if t == 9:
        (arr_t,) = struct.unpack_from("<I", buf, off)
        off += 4
        (n,) = struct.unpack_from("<Q", buf, off)
        off += 8
        out = []
        for _ in range(n):
            v, off = read_value(buf, off, arr_t)
            out.append(v)
        return out, off
    raise ValueError(f"unhandled gguf type {t} at {off}")


def parse(path):
    with open(path, "rb") as fh:
        buf = fh.read(400 * 1024 * 1024)  # header is small; guard anyway
    assert buf[:4] == b"GGUF", "not a GGUF file"
    ver, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    off = 24
    kv = {}
    for _ in range(n_kv):
        key, off = read_str(buf, off)
        (t,) = struct.unpack_from("<I", buf, off)
        off += 4
        val, off = read_value(buf, off, t)
        kv[key] = val
    tensors = []
    for _ in range(n_tensors):
        name, off = read_str(buf, off)
        (n_dims,) = struct.unpack_from("<I", buf, off)
        off += 4
        dims = struct.unpack_from("<%dQ" % n_dims, buf, off)
        off += 8 * n_dims
        (dtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        (doff,) = struct.unpack_from("<Q", buf, off)
        off += 8
        tensors.append({"name": name, "ne": list(dims), "dtype": dtype, "off": doff})
    # GGUF tensor `off` values are relative to the DATA SECTION, which starts
    # at the header end rounded up to `general.alignment` (default 32).
    align = kv.get("general.alignment", 32) or 32
    data_base = (off + align - 1) // align * align
    return ver, kv, tensors, data_base


# ggml_type -> (name, block_size)
GGML = {
    0: ("F32", 1), 1: ("F16", 1), 2: ("Q4_0", 32), 3: ("Q4_1", 32),
    6: ("Q5_0", 32), 7: ("Q5_1", 32), 8: ("Q8_0", 32), 9: ("Q8_1", 32),
    10: ("Q2_K", 256), 11: ("Q3_K", 256), 12: ("Q4_K", 256), 13: ("Q5_K", 256),
    14: ("Q6_K", 256), 15: ("Q8_K", 256), 16: ("IQ2_XXS", 256),
    17: ("IQ2_XS", 256), 18: ("IQ3_XXS", 256), 19: ("IQ1_S", 256),
    20: ("IQ4_NL", 32), 21: ("IQ3_S", 256), 22: ("IQ2_S", 256),
    23: ("IQ4_XS", 256), 24: ("I8", 1), 25: ("I16", 1), 26: ("I32", 1),
    27: ("I64", 1), 28: ("F64", 1), 29: ("IQ1_M", 256), 30: ("BF16", 1),
    34: ("TQ1_0", 256), 35: ("TQ2_0", 256),
}


def main():
    path = sys.argv[1]
    ver, kv, tensors, _ = parse(path)
    print(f"file   : {path}")
    print(f"version: {ver}  tensors: {len(tensors)}  kv: {len(kv)}")
    print("\n--- metadata ---")
    for k, v in kv.items():
        if isinstance(v, list):
            s = f"[{len(v)} items]"
        elif isinstance(v, str):
            s = repr(v[:110] + ("..." if len(v) > 110 else ""))
        else:
            s = repr(v)
        print(f"  {k} = {s}")

    print("\n--- dtype census (element counts) ---")
    cen = Counter()
    for t in tensors:
        name, _ = GGML.get(t["dtype"], (f"T{t['dtype']}", 1))
        cen[name] += 1
    for k, v in sorted(cen.items(), key=lambda kv_: -kv_[1]):
        print(f"  {k:10s} {v}")

    if "--names" in sys.argv:
        print("\n--- full tensor list (name ne dtype) ---")
        for t in tensors:
            name, _ = GGML.get(t["dtype"], (f"T{t['dtype']}", 1))
            print(f"  {t['name']:60s} {str(t['ne']):22s} {name}")
    if "--shape" in sys.argv:
        print("\n--- distinct shapes per dtype ---")
        by = defaultdict(Counter)
        for t in tensors:
            name, _ = GGML.get(t["dtype"], (f"T{t['dtype']}", 1))
            by[name][tuple(t["ne"])] += 1
        for dt, shapes in by.items():
            print(f"  {dt}:")
            for sh, c in sorted(shapes.items(), key=lambda x: -x[1]):
                print(f"     {str(list(sh)):24s} x{c}")


if __name__ == "__main__":
    main()

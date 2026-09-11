"""Full structural diff of two GGUF files.

Parses both headers (tensor name, dtype, shape, offset), then:
  1. compares name sets (missing/extra);
  2. for common tensors compares dtype + shape;
  3. for same-dtype tensors byte-compares the payloads (mmap);
  4. sums per-dtype counts on each side.

Usage: python diff_gguf_full.py <a.gguf> <b.gguf>
"""

import mmap
import struct
import sys
from collections import Counter
from pathlib import Path

GGML_TYPE = {
    0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32),
    3: ("Q4_1", 20, 32), 6: ("Q5_0", 22, 32), 7: ("Q5_1", 24, 32),
    8: ("Q8_0", 34, 32), 10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256),
    12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256),
    16: ("BF16", 2, 1),
}


def parse(path):
    f = open(path, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ver, nt, nk = struct.unpack_from("<IIQQ", data, 0)
    assert magic == 0x46554747
    pos = 24

    def rstr(p):
        (n,) = struct.unpack_from("<Q", data, p)
        p += 8
        return data[p : p + n].decode("utf-8"), p + n

    meta = {}
    for _ in range(nk):
        k, pos = rstr(pos)
        (t,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if t == 8:  # string
            v, pos = rstr(pos)
            meta[k] = v
        elif t == 0:  # uint8
            meta[k] = struct.unpack_from("<B", data, pos)[0]
            pos += 1
        elif t == 1:  # int8
            meta[k] = struct.unpack_from("<b", data, pos)[0]
            pos += 1
        elif t == 2:  # uint16
            meta[k] = struct.unpack_from("<H", data, pos)[0]
            pos += 2
        elif t == 3:  # int16
            meta[k] = struct.unpack_from("<h", data, pos)[0]
            pos += 2
        elif t == 4:  # uint32
            meta[k] = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        elif t == 5:  # int32
            meta[k] = struct.unpack_from("<i", data, pos)[0]
            pos += 4
        elif t == 6:  # float32
            meta[k] = struct.unpack_from("<f", data, pos)[0]
            pos += 4
        elif t == 7:  # bool
            meta[k] = struct.unpack_from("<B", data, pos)[0]
            pos += 1
        elif t == 10:  # float64
            meta[k] = struct.unpack_from("<d", data, pos)[0]
            pos += 8
        elif t == 11:  # int64
            meta[k] = struct.unpack_from("<q", data, pos)[0]
            pos += 8
        elif t == 12:  # uint64
            meta[k] = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
        elif t == 9:  # array
            (et,) = struct.unpack_from("<I", data, pos)
            pos += 4
            (n,) = struct.unpack_from("<Q", data, pos)
            pos += 8
            if et == 8:  # array of strings
                arr = []
                for _ in range(n):
                    s, pos = rstr(pos)
                    arr.append(s)
                meta[k] = arr
            else:  # fixed-size elements: BOOL=1B, 16-bit=2B, 32-bit=4B,
                # FLOAT64/INT64/UINT64=8B (uint64 has no struct code for
                # arrays; handled by fixed stride below)
                sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                         10: 8, 11: 8, 12: 8}
                pos += n * sizes[et]
        else:
            raise ValueError(f"kv type {t}")

    tensors = {}
    order = []
    for _ in range(nt):
        nm, pos = rstr(pos)
        (nd,) = struct.unpack_from("<I", data, pos)
        pos += 4
        ne = []
        for _ in range(nd):
            (d,) = struct.unpack_from("<Q", data, pos)
            pos += 8
            ne.append(d)
        (dt,) = struct.unpack_from("<I", data, pos)
        pos += 4
        (off,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        tensors[nm] = (dt, ne, off)
        order.append(nm)

    # data segment starts at next alignment boundary
    alignment = meta.get("general.alignment", 32)
    pad = (alignment - (pos % alignment)) % alignment
    data_start = pos + pad
    return f, data, tensors, order, data_start, meta


def payload_len(dt, ne):
    name, tsz, blck = GGML_TYPE[dt]
    n = 1
    for d in ne:
        n *= d
    return (n // blck) * tsz


def main():
    pa, pb = Path(sys.argv[1]), Path(sys.argv[2])
    fa, da, ta, oa, sa, ma = parse(pa)
    fb, db, tb, ob, sb, mb = parse(pb)

    print(f"A (old): {pa.name}  tensors={len(ta)}")
    print(f"B (new): {pb.name}  tensors={len(tb)}")

    ca = Counter(GGML_TYPE[v[0]][0] for v in ta.values())
    cb = Counter(GGML_TYPE[v[0]][0] for v in tb.values())
    print("A dtypes:", dict(ca))
    print("B dtypes:", dict(cb))

    only_a = set(ta) - set(tb)
    only_b = set(tb) - set(ta)
    print(f"only in A: {len(only_a)}", sorted(only_a)[:5])
    print(f"only in B: {len(only_b)}", sorted(only_b)[:5])

    dtype_diffs = []
    shape_diffs = []
    byte_diffs = []
    common = set(ta) & set(tb)
    for nm in common:
        dta, nea, _ = ta[nm]
        dtb, neb, _ = tb[nm]
        if dta != dtb:
            dtype_diffs.append((nm, GGML_TYPE[dta][0], GGML_TYPE[dtb][0],
                                nea))
        elif nea != neb:
            shape_diffs.append(nm)
        else:
            la = payload_len(dta, nea)
            offa = ta[nm][2] + sa
            offb = tb[nm][2] + sb
            if da[offa : offa + la] != db[offb : offb + la]:
                byte_diffs.append(nm)

    print(f"\ndtype differs: {len(dtype_diffs)}")
    for nm, ta_, tb_, ne in dtype_diffs:
        n = 1
        for d in ne:
            n *= d
        print(f"  {nm}: {ta_} -> {tb_}  shape={ne} ({n} elems)")
    print(f"shape differs: {len(shape_diffs)}")
    print(f"payload bytes differ (same dtype): {len(byte_diffs)}")
    for nm in byte_diffs[:10]:
        print(f"  {nm}")

    # metadata diff
    keys = set(ma) | set(mb)
    mdiff = [(k, ma.get(k), mb.get(k)) for k in keys if ma.get(k) != mb.get(k)]
    print(f"\nmetadata keys: A={len(ma)} B={len(mb)} differing={len(mdiff)}")
    for k, va, vb in mdiff[:15]:
        print(f"  {k}: {va!r} vs {vb!r}")


if __name__ == "__main__":
    main()

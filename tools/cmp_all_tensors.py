"""Byte-compare ALL shared tensors between two GGUFs — including the
F32/BF16/F16 payloads that `--verify-against` deliberately skips (the 1-D
F32 convention). That skip masked the gemma norm bugs (name mapping +
the `1+w` pre-bake were invisible until a direct value probe); this tool
is the complement: it reports name-set alignment AND payload equality for
EVERY dtype.

Verdict buckets: shared byte-exact / byte-differing / dtype-mismatch /
dims-mismatch / ours-only / reference-only.
"""
import mmap
import struct
import sys


def gguf_all(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ver, n_t, n_kv = struct.unpack_from("<IIQQ", mm, 0)
    pos = 24

    def rstr(p):
        (n,) = struct.unpack_from("<Q", mm, p)
        p += 8
        return mm[p : p + n].decode(), p + n

    kv = {}
    for _ in range(n_kv):
        k, pos = rstr(pos)
        (vt,) = struct.unpack_from("<I", mm, pos)
        pos += 4
        if vt == 8:
            v, pos = rstr(pos)
        elif vt in (0, 1):
            (v,) = struct.unpack_from("<B", mm, pos)
            pos += 1
        elif vt in (2, 3):
            (v,) = struct.unpack_from("<H", mm, pos)
            pos += 2
        elif vt in (4, 5):
            (v,) = struct.unpack_from("<I" if vt == 4 else "<i", mm, pos)
            pos += 4
        elif vt == 6:
            (v,) = struct.unpack_from("<f", mm, pos)
            pos += 4
        elif vt == 7:
            (v,) = struct.unpack_from("<B", mm, pos)
            pos += 1
        elif vt in (10, 11):
            (v,) = struct.unpack_from("<Q" if vt == 10 else "<q", mm, pos)
            pos += 8
        elif vt == 12:
            (v,) = struct.unpack_from("<d", mm, pos)
            pos += 8
        elif vt == 9:
            (elt, n) = struct.unpack_from("<IQ", mm, pos)
            pos += 12
            if elt == 8:
                vals = []
                for _ in range(n):
                    sv, pos = rstr(pos)
                    vals.append(sv)
                v = vals
            else:
                sz = {0: 1, 1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[elt]
                pos += sz * n
                v = None
        else:
            raise SystemExit(f"kv type {vt}")
        kv[k] = v
    tensors = []
    for _ in range(n_t):
        name, pos = rstr(pos)
        (nd,) = struct.unpack_from("<I", mm, pos)
        pos += 4
        dims = struct.unpack_from("<" + "q" * nd, mm, pos)
        pos += 8 * nd
        (ty,) = struct.unpack_from("<I", mm, pos)
        pos += 4
        (off,) = struct.unpack_from("<Q", mm, pos)
        pos += 8
        tensors.append((name, ty, list(dims), off))
    align = kv.get("general.alignment", 32)
    data_start = (pos + align - 1) // align * align
    return kv, tensors, data_start, mm


# GGML type ids: gguf-py constants.py (GGMLQuantizationType), verified
# against the vendored reference (TQ1_0=34, TQ2_0=35 — NOT the MOSTLY_*
# ftype ids 36/37).
TYNAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4",
}
# elements per block
BLK = {2: 32, 3: 32, 6: 32, 7: 32, 8: 32, 20: 32, 39: 32, 40: 16,
       10: 256, 11: 256, 12: 256, 13: 256, 14: 256, 15: 256, 16: 256,
       17: 256, 18: 256, 19: 256, 21: 256, 22: 256, 23: 256, 29: 256,
       34: 256, 35: 256}
# bytes per block
BBLK = {2: 18, 3: 20, 6: 22, 7: 24, 8: 34, 20: 36, 39: 24, 40: 12,
        10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292, 16: 66,
        17: 88, 18: 110, 19: 90, 21: 110, 22: 94, 23: 134, 29: 92,
        34: 50, 35: 82}


def payload_bytes(ty, dims):
    n = 1
    for d in dims:
        n *= d
    if ty == 0:
        return n * 4
    if ty in (1, 30):
        return n * 2
    if ty == 24:
        return n
    if ty == 25:
        return n * 2
    if ty == 26:
        return n * 4
    if ty in (27, 28):
        return n * 8
    if ty in BLK:
        assert n % BLK[ty] == 0
        return (n // BLK[ty]) * BBLK[ty]
    raise SystemExit(f"size for type {ty} unknown")


def main(ours_path, theirs_path):
    kv_o, t_o, ds_o, mm_o = gguf_all(ours_path)
    kv_t, t_t, ds_t, mm_t = gguf_all(theirs_path)
    theirs = {n: (ty, dims, off) for n, ty, dims, off in t_t}
    ours = {n: (ty, dims, off) for n, ty, dims, off in t_o}

    exact, diff, dtype_mm, dims_mm = [], [], [], []
    for n in sorted(ours):
        if n not in theirs:
            continue
        ty_o, dims_o, off_o = ours[n]
        ty_t, dims_t, off_t = theirs[n]
        if ty_o != ty_t:
            dtype_mm.append((n, TYNAME.get(ty_o, ty_o), TYNAME.get(ty_t, ty_t)))
            continue
        if dims_o != dims_t:
            dims_mm.append((n, dims_o, dims_t))
            continue
        b = payload_bytes(ty_o, dims_o)
        a = bytes(mm_o[ds_o + off_o : ds_o + off_o + b])
        c = bytes(mm_t[ds_t + off_t : ds_t + off_t + b])
        (exact if a == c else diff).append((n, TYNAME.get(ty_o, ty_o)))

    ours_only = sorted(set(ours) - set(theirs))
    ref_only = sorted(set(theirs) - set(ours))

    print(f"shared byte-exact (ALL dtypes incl. F32): {len(exact)}")
    print(f"shared byte-differing:                    {len(diff)}")
    for n, t in diff[:12]:
        print("  !", n, t)
    if len(diff) > 12:
        print(f"  ... {len(diff) - 12} more")
    print(f"dtype mismatches: {len(dtype_mm)}")
    for d in dtype_mm[:12]:
        print("  !", d)
    print(f"dims mismatches: {len(dims_mm)}")
    for d in dims_mm[:12]:
        print("  !", d)
    print(f"ours-only: {len(ours_only)}")
    for n in ours_only[:12]:
        print("  ?", n, TYNAME.get(ours[n][0], ours[n][0]))
    if len(ours_only) > 12:
        print(f"  ... {len(ours_only) - 12} more")
    print(f"reference-only: {len(ref_only)}")
    for n in ref_only[:12]:
        print("  ?", n, TYNAME.get(theirs[n][0], theirs[n][0]))
    if len(ref_only) > 12:
        print(f"  ... {len(ref_only) - 12} more")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

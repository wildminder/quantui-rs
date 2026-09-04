"""Diff two GGUF files' tensor inventories (names, dtypes, shapes)."""
import json, mmap, struct, sys
from pathlib import Path

def parse(path):
    f = open(path, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ver, n_t, n_kv = struct.unpack_from("<IIQQ", data, 0)
    pos = 24
    def rstr(p):
        (n,) = struct.unpack_from("<Q", data, p); p += 8
        return data[p:p+n].decode(), p+n
    kv = {}
    for _ in range(n_kv):
        k, pos = rstr(pos)
        (vt,) = struct.unpack_from("<I", data, pos); pos += 4
        if vt == 8: v, pos = rstr(pos)
        elif vt == 4: (v,) = struct.unpack_from("<I", data, pos); pos += 4
        elif vt == 5: (v,) = struct.unpack_from("<i", data, pos); pos += 4
        elif vt == 6: (v,) = struct.unpack_from("<f", data, pos); pos += 4
        elif vt == 7: (v,) = struct.unpack_from("<B", data, pos); pos += 1
        elif vt == 10: (v,) = struct.unpack_from("<Q", data, pos); pos += 8
        elif vt == 11: (v,) = struct.unpack_from("<q", data, pos); pos += 8
        else: raise SystemExit(f"kv type {vt} unhandled for key {k}")
        kv[k] = v
    tensors = []
    for _ in range(n_t):
        name, pos = rstr(pos)
        (nd,) = struct.unpack_from("<I", data, pos); pos += 4
        dims = struct.unpack_from("<" + "q"*nd, data, pos); pos += 8*nd
        (ty,) = struct.unpack_from("<I", data, pos); pos += 4
        (off,) = struct.unpack_from("<Q", data, pos); pos += 8
        tensors.append((name, ty, dims))
    return kv, tensors

old_kv, old_t = parse(sys.argv[1])
new_kv, new_t = parse(sys.argv[2])
print(f"OLD: {sys.argv[1]}")
print(f"  general.name={old_kv.get('general.name')!r} quantized_by={old_kv.get('general.quantized_by')!r} tensors={len(old_t)}")
print(f"NEW: {sys.argv[2]}")
print(f"  general.name={new_kv.get('general.name')!r} quantized_by={new_kv.get('general.quantized_by')!r} tensors={len(new_t)}")
hist = {}
for _, ty, _ in old_t: hist[ty] = hist.get(ty, 0) + 1
print(f"OLD dtype histogram (type codes): {hist}")
old_map = {t[0]: t for t in old_t}
new_map = {t[0]: t for t in new_t}
only_old = sorted(set(old_map) - set(new_map))
only_new = sorted(set(new_map) - set(old_map))
print(f"\nin OLD only ({len(only_old)}):")
for n in only_old:
    _, ty, dims = old_map[n]
    print(f"  - {n}  type={ty} dims={list(dims)}")
print(f"\nin NEW only ({len(only_new)}):")
for n in only_new:
    _, ty, dims = new_map[n]
    print(f"  + {n}  type={ty} dims={list(dims)}")
mismatch = [(n, old_map[n][2], new_map[n][2]) for n in set(old_map) & set(new_map) if list(old_map[n][2]) != list(new_map[n][2])]
print(f"\nshared-name shape mismatches: {len(mismatch)}")
for n, a, b in mismatch[:10]:
    print(f"  ! {n} old={list(a)} new={list(b)}")

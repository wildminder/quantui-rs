"""Why do shortconv.in_proj payloads differ vs unsloth?

Dequantize blk.0.shortconv.in_proj.weight from both files and analyze:
identical rows / permuted rows / different rows.
"""
import mmap, struct, sys
import numpy as np

def gguf_all(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ver, n_t, n_kv = struct.unpack_from("<IIQQ", mm, 0)
    pos = 24
    def rstr(p):
        (n,) = struct.unpack_from("<Q", mm, p); p += 8
        return mm[p:p+n].decode(), p+n
    for _ in range(n_kv):
        k, pos = rstr(pos)
        (vt,) = struct.unpack_from("<I", mm, pos); pos += 4
        if vt == 8: _, pos = rstr(pos)
        elif vt in (0,1): pos += 1
        elif vt in (2,3): pos += 2
        elif vt in (4,5,6): pos += 4
        elif vt in (7,): pos += 1
        elif vt in (10,11,12): pos += 8
        elif vt == 9:
            (elt, n) = struct.unpack_from("<IQ", mm, pos); pos += 12
            sz = {0:1,1:1,2:1,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(elt)
            if elt == 8 or sz is None:
                for _ in range(n): _, pos = rstr(pos)
            else: pos += sz*n
    tensors = {}
    for _ in range(n_t):
        name, pos = rstr(pos)
        (nd,) = struct.unpack_from("<I", mm, pos); pos += 4
        dims = struct.unpack_from("<" + "q"*nd, mm, pos); pos += 8*nd
        (ty,) = struct.unpack_from("<I", mm, pos); pos += 4
        (off,) = struct.unpack_from("<Q", mm, pos); pos += 8
        tensors[name] = (ty, list(dims), off)
    align = 32
    data_start = (pos + align - 1) // align * align
    return tensors, data_start, mm

def deq_q8_0(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n // 32, 34)
    d = b[:, :2].copy().view("<f2").astype(np.float32).ravel()
    q = b[:, 2:].astype(np.int8).astype(np.float32).reshape(-1, 32)
    return (q * d[:, None]).reshape(-1)

import re
def map_name(n):
    m = re.match(r"model\.language_model\.layers\.(\d+)\.conv\.(\w+)_proj\.weight", n)
    if m: return f"blk.{m.group(1)}.shortconv.{m.group(2)}_proj.weight"
    return n

def main(ours, theirs, tname="blk.0.shortconv.in_proj.weight"):
    t_raw, ds_o, mm_o = gguf_all(ours)
    t_o = {map_name(n): v for n, v in t_raw.items()}
    t_t, ds_t, mm_t = gguf_all(theirs)
    ty, dims, off = t_o[tname]
    tty, tdims, toff = t_t[tname]
    n = int(np.prod(dims))
    nb = (n // 32) * 34
    a = deq_q8_0(bytes(mm_o[ds_o+off:ds_o+off+nb]), n)
    c = deq_q8_0(bytes(mm_t[ds_t+toff:ds_t+toff+nb]), n)
    ne0 = dims[0]  # 2048
    nrows = dims[1]  # 6144
    A = a.reshape(nrows, ne0)
    C = c.reshape(nrows, ne0)
    same_row = np.zeros(nrows, dtype=bool)
    for i in range(nrows):
        # find j where C[j] == A[i] within q8 tolerance
        d = np.abs(C - A[i]).max(axis=1)
        same_row[i] = (d < 0.02).any()
    print(f"{tname}: rows total={nrows}, rows whose values appear somewhere in theirs: {same_row.sum()}")
    # check simple chunk permutations: split into 3 chunks of 2048 rows
    for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        P = np.concatenate([C[perm[0]*2048:(perm[0]+1)*2048],
                            C[perm[1]*2048:(perm[1]+1)*2048],
                            C[perm[2]*2048:(perm[2]+1)*2048]])
        eq = np.abs(P - A).max()
        if eq < 0.02:
            print(f"  CHUNK PERMUTATION MATCH: rows reordered as {perm}")
            return
    # row-level: where does A[i] match C (as index)
    loc = []
    for i in range(0, nrows, 171):  # sample 36 rows
        d = np.abs(C - A[i]).max(axis=1)
        j = int(np.argmin(d))
        loc.append((i, j, float(d[j])))
    print("  sample row mapping (ours_idx, theirs_best_idx, maxdiff):")
    for i, j, d in loc[:12]: print(f"    {i} -> {j}  ({d:.4f})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "blk.0.shortconv.in_proj.weight")

"""Distribution of the q8_0 code mismatches: which blocks, what values."""
import json, mmap, struct
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
    data_start = (pos + 31) // 32 * 32
    return tensors, data_start, mm

ST = "<LOCAL-MODELS>lfm/LFM2.5-VL-3B-original.safetensors"
TNAME_HF = "model.language_model.layers.0.conv.in_proj.weight"
f = open(ST, "rb")
(hlen,) = struct.unpack("<Q", f.read(8))
hdr = json.loads(f.read(hlen))
f.close()
s0, e0 = hdr[TNAME_HF]["data_offsets"]
mm_src = np.memmap(ST, dtype=np.uint8, mode="r")
base = 8 + hlen + ((8 - hlen % 8) % 8)
src = np.frombuffer(mm_src[base+s0:base+e0].tobytes(), dtype="<u2")
src = (src.astype(np.uint32) << 16).view("<f4").astype(np.float32).ravel()
nb = src.size // 32
sb = src.reshape(nb, 32)

def load(path, tname):
    tensors, ds, mm = gguf_all(path)
    ty, dims, off = tensors[tname]
    nn = dims[0]*dims[1]
    raw = np.frombuffer(bytes(mm[ds+off:ds+off+(nn//32)*34]), dtype=np.uint8).reshape(nn//32, 34)
    d = raw[:, :2].copy().view("<f2").astype(np.float32).ravel()
    q = raw[:, 2:].astype(np.int8)
    return d, q

d_o, q_o = load("<LOCAL-MODELS>lfm/LFM2.5-VL-3B-Q8_0-ours.gguf", TNAME_HF)
d_u, q_u = load("<LOCAL-MODELS>lfm/LFM2.5-VL-3B-Q8_0-unsloth.gguf", "blk.0.shortconv.in_proj.weight")

# dequantized values
v_o = q_o.astype(np.float32) * d_o[:, None]
v_u = q_u.astype(np.float32) * d_u[:, None]
err_o = np.abs(v_o - sb).max(axis=1)
err_u = np.abs(v_u - sb).max(axis=1)
# count mismatching codes per block
mism = (q_o != q_u)
per_block = mism.sum(axis=1)
print(f"blocks with >=1 code mismatch: {(per_block > 0).sum()} / {nb}")
print(f"block error vs source: ours max-of-max={err_o.max():.6f}, unsloth max-of-max={err_u.max():.6f}")
print(f"ours better blocks:   {(err_o < err_u).sum()}")
print(f"unsloth better blocks: {(err_u < err_o).sum()}")
print(f"tie blocks:            {(err_o == err_u).sum()}")
# look at blocks where codes differ, what do the differing codes look like
idx = np.where(per_block > 0)[0]
i = idx[0]
print(f"\nfirst differing block #{i}: d_ours={d_o[i]:.8g} d_unsloth={d_u[i]:.8g}")
diff_pos = np.where(q_o[i] != q_u[i])[0]
print(f"positions differing: {diff_pos}")
for p in diff_pos[:10]:
    x = sb[i, p]
    print(f"  pos {p}: src={x:.8g}  ours q={q_o[i,p]:4d} -> {v_o[i,p]:.8g}   unsloth q={q_u[i,p]:4d} -> {v_u[i,p]:.8g}")

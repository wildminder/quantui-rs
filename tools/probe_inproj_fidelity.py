"""Which q8_0 is faithful to the source: ours or unsloth's?

Load the original BF16 tensor, dequantize both GGUF encodings, compare
per-block scales and reconstruction error vs source.
"""
import json, mmap, struct, sys
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

# load source
f = open(ST, "rb")
(hlen,) = struct.unpack("<Q", f.read(8))
hdr = json.loads(f.read(hlen))
f.close()
s0, e0 = hdr[TNAME_HF]["data_offsets"]
mm_src = np.memmap(ST, dtype=np.uint8, mode="r")
base = 8 + hlen + ((8 - hlen % 8) % 8)
src_bf16 = np.frombuffer(mm_src[base+s0:base+e0].tobytes(), dtype="<u2")
src = (src_bf16.astype(np.uint32) << 16).view("<f4").astype(np.float32)
src = src.reshape(6144, 2048)  # HF [out, in]

def load_q80(path, tname):
    tensors, ds, mm = gguf_all(path)
    ty, dims, off = tensors[tname]
    n = dims[0] * dims[1]
    raw = np.frombuffer(bytes(mm[ds+off:ds+off+(n//32)*34]), dtype=np.uint8).reshape(n//32, 34)
    d = raw[:, :2].copy().view("<f2").astype(np.float32).ravel()
    q = raw[:, 2:].astype(np.int8)
    return d, q, dims

d_o, q_o, dims_o = load_q80("<LOCAL-MODELS>lfm/LFM2.5-VL-3B-Q8_0-ours.gguf", "model.language_model.layers.0.conv.in_proj.weight")
d_u, q_u, _ = load_q80("<LOCAL-MODELS>lfm/LFM2.5-VL-3B-Q8_0-unsloth.gguf", "blk.0.shortconv.in_proj.weight")

nb = 6144 * 2048 // 32
print(f"blocks: {nb}")
print(f"d bytes identical: {np.array_equal(d_o.view(np.uint16), d_u.view(np.uint16))} ({(d_o != d_u).sum()}/{nb} blocks differ)")
dq = (q_o != q_u).sum()
print(f"q int8 identical: {dq == 0} ({dq}/{nb*32} values differ)")

# source amax per block — the ground-truth scale
src_blocks = src.reshape(nb, 32)
amax = np.abs(src_blocks).max(axis=1)
d_ref = (amax / 127.0).astype(np.float32)
d_ref_f16 = d_ref.astype("<f2").astype(np.float32)
match_o = (d_o.view(np.uint16) == d_ref_f16.view(np.uint16)).sum()
match_u = (d_u.view(np.uint16) == d_ref_f16.view(np.uint16)).sum()
print(f"d matches source-amax/127 (llama.cpp formula): ours {match_o}/{nb}, unsloth {match_u}/{nb}")

# reconstruction error vs source
deq_o = (q_o.astype(np.float32) * d_o[:, None]).reshape(6144, 2048)
deq_u = (q_u.astype(np.float32) * d_u[:, None]).reshape(6144, 2048)
err_o = np.abs(deq_o - src)
err_u = np.abs(deq_u - src)
print(f"reconstruction err vs source: ours mean={err_o.mean():.6f} max={err_o.max():.6f}; unsloth mean={err_u.mean():.6f} max={err_u.max():.6f}")

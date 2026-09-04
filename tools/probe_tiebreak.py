"""Arbiter: which tie-break rule does each encoder use?

llama.cpp quantize_row_q8_0_ref: q = roundf(x*id)  (half-away-from-zero)
numpy np.round:                   q = round-half-to-even
Compare both reconstructions against the actual bytes in each file.
"""
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
n = src.size
nb = n // 32
src_blocks = src.reshape(nb, 32)

# llama.cpp reference: d = amax/127 in f32; id = 1/d from the F32 d
# (f16 rounding happens ONLY at storage); q = roundf(x*id)
amax = np.abs(src_blocks).max(axis=1)
d_f32 = (amax / np.float32(127.0)).astype(np.float32)
idv = np.where(d_f32 > 0, np.float32(1.0) / np.where(d_f32 > 0, d_f32, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
prod = (src_blocks * idv[:, None]).astype(np.float32)
# rule A: half-away-from-zero (C roundf semantics)
qA = np.where(prod >= 0, np.floor(prod + 0.5), np.ceil(prod - 0.5)).astype(np.int8)
qA = np.clip(qA, -128, 127)
# rule B: numpy half-to-even
qB = np.clip(np.round(prod), -128, 127).astype(np.int8)

def q_from(path, tname):
    tensors, ds, mm = gguf_all(path)
    ty, dims, off = tensors[tname]
    nn = dims[0]*dims[1]
    raw = np.frombuffer(bytes(mm[ds+off:ds+off+(nn//32)*34]), dtype=np.uint8).reshape(nn//32, 34)
    return raw[:, 2:].astype(np.int8)

q_ours = q_from("<LOCAL-MODELS>lfm/LFM2.5-VL-3B-Q8_0-ours.gguf", TNAME_HF)
q_uns  = q_from("<LOCAL-MODELS>lfm/LFM2.5-VL-3B-Q8_0-unsloth.gguf", "blk.0.shortconv.in_proj.weight")

print(f"ties (prod ends exactly on .5): {(prod % 1 == 0.5).sum()} / {n}")
print(f"ours  == rule A (roundf, llama.cpp): {np.array_equal(q_ours, qA)}")
print(f"ours  == rule B (np.round):          {np.array_equal(q_ours, qB)}")
print(f"unsloth == rule A (roundf):          {np.array_equal(q_uns, qA)}")
print(f"unsloth == rule B (np.round):        {np.array_equal(q_uns, qB)}")

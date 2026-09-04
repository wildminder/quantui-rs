"""Why do Q4_0 payloads differ? Inspect block scale + nibbles vs source."""
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
TNAME_HF = "model.language_model.layers.0.feed_forward.w1.weight"  # ffn_gate
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

def load_q40(path, tname):
    tensors, ds, mm = gguf_all(path)
    ty, dims, off = tensors[tname]
    nn = dims[0]*dims[1]
    raw = np.frombuffer(bytes(mm[ds+off:ds+off+(nn//32)*18]), dtype=np.uint8).reshape(nn//32, 18)
    d = raw[:, :2].copy().view("<f2").astype(np.float32).ravel()
    qs = raw[:, 2:]
    q = np.empty((nn//32, 32), dtype=np.int8)
    q[:, :16] = (qs & 0x0F).astype(np.int8)
    q[:, 16:] = (qs >> 4).astype(np.int8)
    return d, q

d_o, q_o = load_q40("<LOCAL-MODELS>lfm/LFM2.5-VL-3B-Q4_0-ours.gguf", TNAME_HF)
d_u, q_u = load_q40("<LOCAL-MODELS>lfm/LFM2.5-VL-3B-Q4_0-unsloth.gguf", "blk.0.ffn_gate.weight")

print(f"blocks: {nb}")
dm = (d_o != d_u).sum()
print(f"d mismatch: {dm}/{nb}")
qm = (q_o != q_u).sum()
print(f"q mismatch: {qm}/{nb*32}")

# llama Q4_0 ref: amax & max tracking; d = max/-8; q = clamp(round(x*id+8),0,15)
amax = np.abs(sb).max(axis=1)
maxv = sb[np.arange(nb), np.abs(sb).argmax(axis=1)]
d_ref = (maxv / -8.0).astype(np.float32)
d_ref_f16 = d_ref.astype("<f2").astype(np.float32)
mo = (d_o.view(np.uint16) == d_ref_f16.view(np.uint16)).sum()
mu = (d_u.view(np.uint16) == d_ref_f16.view(np.uint16)).sum()
print(f"d matches llama ref (max/-8): ours {mo}/{nb}, unsloth {mu}/{nb}")

# alternative: symmetric amax/8
d_alt = (amax / 8.0).astype(np.float32)
# note sign: q codes map to [-8..7]*d  with offset; symmetric variant: d = amax/8 (positive), q = x*id+8
d_alt_f16 = d_alt.astype("<f2").astype(np.float32)
mo2 = (d_o.view(np.uint16) == d_alt_f16.view(np.uint16)).sum()
mu2 = (d_u.view(np.uint16) == d_alt_f16.view(np.uint16)).sum()
print(f"d matches amax/8 (alt): ours {mo2}/{nb}, unsloth {mu2}/{nb}")

# check sign convention via reconstruction
v_o = (q_o.astype(np.float32) - 8.0) * d_o[:, None]
err_o = np.abs(v_o - sb)
v_u = (q_u.astype(np.float32) - 8.0) * d_u[:, None]
err_u = np.abs(v_u - sb)
print(f"err: ours mean={err_o.mean():.6f} max={err_o.max():.6f} | unsloth mean={err_u.mean():.6f} max={err_u.max():.6f}")

# first differing block details
idx = np.where((d_o != d_u) | (q_o != q_u).any(axis=1))[0]
i = int(idx[0])
print(f"\nfirst diff block {i}: src sample {sb[i,:6]}")
print(f"  d_ours={d_o[i]:.8g}  d_unsloth={d_u[i]:.8g}  ref max/-8={d_ref[i]:.8g}  alt amax/8={d_alt[i]:.8g}")
print(f"  q_ours {q_o[i,:8]}  q_unsloth {q_u[i,:8]}")

"""Verify the 102 F16 conv tensors in the FIXED GGUF against the HF source.

Usage:
    python tools/check_fixed_f16.py <fixed.gguf> <hf_model_dir>
"""
import json, mmap, os, struct, sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 3:
    sys.exit("usage: check_fixed_f16.py <fixed.gguf> <hf_model_dir>")
GGUF = Path(sys.argv[1])
HF_DIR = Path(sys.argv[2])

# --- parse GGUF header (v3) ---
f = open(GGUF, "rb"); data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
pos = 24
def rstr(p):
    (n,) = struct.unpack_from("<Q", data, p); p += 8
    return data[p:p+n].decode(), p+n
kv = {}
for _ in range(struct.unpack_from("<Q", data, 16)[0]):
    k, pos = rstr(pos)
    (vt,) = struct.unpack_from("<I", data, pos); pos += 4
    if vt == 8: v, pos = rstr(pos)
    elif vt in (4,5): (v,) = struct.unpack_from("<I" if vt==4 else "<i", data, pos); pos += 4
    elif vt == 6: (v,) = struct.unpack_from("<f", data, pos); pos += 4
    elif vt == 10: (v,) = struct.unpack_from("<Q", data, pos); pos += 8
    else: raise SystemExit(f"kv type {vt} unhandled")
    kv[k] = v
tensors = []
n_t = kv.get("tensor_count", struct.unpack_from("<Q", data, 8)[0])
for _ in range(n_t):
    name, pos = rstr(pos)
    (nd,) = struct.unpack_from("<I", data, pos); pos += 4
    dims = struct.unpack_from("<" + "q"*nd, data, pos); pos += 8*nd
    (ty,) = struct.unpack_from("<I", data, pos); pos += 4
    (off,) = struct.unpack_from("<Q", data, pos); pos += 8
    tensors.append((name, ty, dims, off))
infos_end = pos
align = kv.get("general.alignment", 32)
data_start = (infos_end + align - 1) // align * align

# --- load HF shards ---
idx = json.loads((HF_DIR / "model.safetensors.index.json").read_text())
shards = {}
for hf_name, sf in idx["weight_map"].items():
    shards.setdefault(sf, []).append(hf_name)
hf_bytes = {}  # name -> (np array)
for sf, names in shards.items():
    mm = np.memmap(HF_DIR / sf, dtype=np.uint8, mode="r")
    (hlen,) = struct.unpack_from("<Q", mm, 0)
    hdr = json.loads(mm[8:8+hlen].tobytes())
    base = 8 + hlen + ((8 - hlen % 8) % 8)
    for n in names:
        s, e = hdr[n]["data_offsets"]
        dt = {"BF16": np.dtype("V2"), "F32": np.dtype("<f4")}[hdr[n]["dtype"]]
        a = np.frombuffer(mm[base+s:base+e].tobytes(), dtype=dt)
        hf_bytes[n] = a
    del mm

# --- check every F16 tensor in the GGUF ---
f16_tensors = [(n, d, o) for (n, t, d, o) in tensors if t == 1]
print(f"F16 tensors in GGUF: {len(f16_tensors)}")
worst = 0.0
checked = 0
for name, dims, off in f16_tensors:
    raw = np.frombuffer(bytes(data[data_start+off : data_start+off+2*int(np.prod(dims))]), dtype="<f2")
    got = raw.astype(np.float32)
    src = hf_bytes[name]
    if src.dtype == np.dtype("V2"):
        u = src.view(np.uint16).astype(np.uint32) << np.uint32(16)
        src = u.view("<f4") if src.nbytes % 4 == 0 else src  # bf16 -> f32
        if src.dtype != np.dtype("<f4"):
            src = np.frombuffer(src.tobytes(), dtype="<f4")
    src = np.asarray(src, dtype=np.float32).reshape(got.shape)
    err = float(np.max(np.abs(got - src))) if got.size else 0.0
    worst = max(worst, err)
    checked += 1
print(f"checked {checked} F16 tensors vs HF source; worst abs err = {worst:.3e} (f16 ulp near 1.0 ~ 1e-3)")
print("PASS" if worst < 5e-3 else "FAIL")

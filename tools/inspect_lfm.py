"""Inspect LFM2.5-VL tensors: original safetensors + unsloth GGUFs."""
import json, mmap, struct, sys
from pathlib import Path

def st_inventory(path):
    f = open(path, "rb")
    (hlen,) = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(hlen))
    f.close()
    out = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        out[name] = (info["dtype"], info["shape"])
    return out

def gguf_inventory(path):
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
        elif vt == 5: (v,) = struct.unpack_from("<i", data, pos); pos += 4
        elif vt == 7: (v,) = struct.unpack_from("<B", data, pos); pos += 1
        elif vt == 11: (v,) = struct.unpack_from("<q", data, pos); pos += 8
        elif vt == 12: (v,) = struct.unpack_from("<d", data, pos); pos += 8
        elif vt == 9:  # array
            (elt, n) = struct.unpack_from("<IQ", data, pos); pos += 12
            if elt == 8:
                vals = []
                for _ in range(n):
                    sv, pos = rstr(pos)
                    vals.append(sv)
                v = vals
            elif elt == 4:
                vals = struct.unpack_from("<" + "I"*n, data, pos); pos += 4*n; v = list(vals)
            elif elt == 5:
                vals = struct.unpack_from("<" + "i"*n, data, pos); pos += 4*n; v = list(vals)
            elif elt == 6:
                vals = struct.unpack_from("<" + "f"*n, data, pos); pos += 4*n; v = list(vals)
            elif elt == 10:
                vals = struct.unpack_from("<" + "Q"*n, data, pos); pos += 8*n; v = list(vals)
            else:
                raise SystemExit(f"array elt {elt} unhandled for {k}")
        else: v = f"?{vt}"
        kv[k] = v
    tensors = {}
    order = []
    for _ in range(n_t):
        name, pos = rstr(pos)
        (nd,) = struct.unpack_from("<I", data, pos); pos += 4
        dims = struct.unpack_from("<" + "q"*nd, data, pos); pos += 8*nd
        (ty,) = struct.unpack_from("<I", data, pos); pos += 4
        (off,) = struct.unpack_from("<Q", data, pos); pos += 8
        tensors[name] = (ty, list(dims))
        order.append(name)
    data.close(); f.close()
    return kv, tensors, order

TY = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",
      10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",
      17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",
      23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",
      30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4",40:"NVFP4"}
BLCK = {"F32":1,"F16":1,"BF16":1,"Q4_0":32,"Q4_1":32,"Q5_0":32,"Q5_1":32,
        "Q8_0":32,"Q2_K":256,"Q3_K":256,"Q4_K":256,"Q5_K":256,"Q6_K":256,
        "IQ2_XXS":256,"IQ2_XS":256,"IQ3_XXS":256,"IQ1_S":256,"IQ4_NL":32,
        "IQ3_S":256,"IQ2_S":256,"IQ4_XS":256,"IQ1_M":256,"TQ1_0":256,"TQ2_0":256}

print("=" * 70)
print("ORIGINAL:", sys.argv[1])
orig = st_inventory(sys.argv[1])
print(f"tensors: {len(orig)}")
from collections import Counter
print("dtypes:", dict(Counter(d for d, _ in orig.values())))
# find tensors whose last-dim (row width) is not divisible by 32 or 256
odd = []
for name, (dt, shape) in orig.items():
    if dt not in ("F32", "F16", "BF16", "I8"):
        continue
    ne0 = shape[-1] if shape else 0
    if ne0 and (ne0 % 32 != 0 or (len(shape) >= 2 and ne0 % 256 != 0 and ne0 % 32 == 0)):
        odd.append((name, dt, shape))
# simpler: report every >=2D tensor with ne0 % 32 != 0 (candidate problem rows)
odd32 = [(n, d, s) for n, (d, s) in orig.items() if len(s) >= 1 and s[-1] % 32 != 0]
print(f"\ntensors with last-dim % 32 != 0 ({len(odd32)}):")
for n, d, s in odd32[:40]:
    print(f"  {n}  {d} {s}")
if len(odd32) > 40: print(f"  ... and {len(odd32)-40} more")

for gg in sys.argv[2:]:
    print("=" * 70)
    print("GGUF:", gg)
    kv, tensors, order = gguf_inventory(gg)
    hist = Counter(TY.get(t, f"T{ty}") for ty, _ in tensors.values()
                   for t in [TY.get(ty, f"T{ty}")])
    print(f"arch={kv.get('general.architecture')} tensors={len(tensors)} dtypes={dict(hist)}")
    # spec violations in the unsloth file itself
    viol = []
    for n, (ty, dims) in tensors.items():
        t = TY.get(ty, f"T{ty}")
        bl = BLCK.get(t, 1)
        if bl > 1 and dims and dims[0] % bl != 0:
            viol.append((n, t, dims))
    print(f"spec violations (ne[0] % blck != 0): {len(viol)}")
    for n, t, d in viol[:10]:
        print(f"  ! {n} {t} {d}")

if len(sys.argv) > 4 and sys.argv[4] == "dump":
    import sys as _s
    print("FULL DUMP:", _s.argv[2])
    for n in order:
        ty, dims = tensors[n]
        print(f"  {n:60s} {TY.get(ty, f"T{ty}"):8s} {dims}")

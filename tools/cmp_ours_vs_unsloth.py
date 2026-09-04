"""Byte-compare our GGUF vs unsloth's on shared tensors (name-mapped).

GGUF tensor payloads are deterministic given identical source values and
identical row-major layout, so a byte-exact match on every shared
quantized tensor proves the quantizers agree.
"""
import mmap, struct, sys, re

def gguf_all(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ver, n_t, n_kv = struct.unpack_from("<IIQQ", mm, 0)
    pos = 24
    def rstr(p):
        (n,) = struct.unpack_from("<Q", mm, p); p += 8
        return mm[p:p+n].decode(), p+n
    kv = {}
    for _ in range(n_kv):
        k, pos = rstr(pos)
        (vt,) = struct.unpack_from("<I", mm, pos); pos += 4
        if vt == 8: v, pos = rstr(pos)
        elif vt in (0,1): (v,) = struct.unpack_from("<B", mm, pos); pos += 1
        elif vt in (2,3): (v,) = struct.unpack_from("<H", mm, pos); pos += 2
        elif vt in (4,5): (v,) = struct.unpack_from("<I" if vt==4 else "<i", mm, pos); pos += 4
        elif vt == 6: (v,) = struct.unpack_from("<f", mm, pos); pos += 4
        elif vt == 7: (v,) = struct.unpack_from("<B", mm, pos); pos += 1
        elif vt in (10,11): (v,) = struct.unpack_from("<Q" if vt==10 else "<q", mm, pos); pos += 8
        elif vt == 12: (v,) = struct.unpack_from("<d", mm, pos); pos += 8
        elif vt == 9:
            (elt, n) = struct.unpack_from("<IQ", mm, pos); pos += 12
            if elt == 8:
                vals = []
                for _ in range(n): sv, pos = rstr(pos); vals.append(sv)
                v = vals
            else:
                sz = {0:1,1:1,2:1,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[elt]
                pos += sz*n; v = None
        else: raise SystemExit(f"kv type {vt}")
        kv[k] = v
    tensors = []
    for _ in range(n_t):
        name, pos = rstr(pos)
        (nd,) = struct.unpack_from("<I", mm, pos); pos += 4
        dims = struct.unpack_from("<" + "q"*nd, mm, pos); pos += 8*nd
        (ty,) = struct.unpack_from("<I", mm, pos); pos += 4
        (off,) = struct.unpack_from("<Q", mm, pos); pos += 8
        tensors.append((name, ty, list(dims), off))
    align = kv.get("general.alignment", 32)
    data_start = (pos + align - 1) // align * align
    return kv, tensors, data_start, mm

TYNAME = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",
          10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",
          17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",
          23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",
          30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4",40:"NVFP4"}
# bytes per block for payload-size computation
BLK = {2:32,3:32,6:32,7:32,8:32,20:32,   # 32-elem blocks
       10:256,11:256,12:256,13:256,14:256,15:256,16:256,17:256,18:256,
       19:256,21:256,22:256,23:256,29:256,34:256,35:256,39:32,40:16}
BBLK = {2:18,3:20,6:22,7:24,8:34,20:36,10:84,11:110,12:144,13:176,14:210,15:292,
        16:66,17:88,18:110,19:90,21:110,22:94,23:134,29:92,34:50,35:82,39:24,40:12}

def payload_bytes(ty, dims):
    n = 1
    for d in dims: n *= d
    if ty in (0,): return n*4
    if ty in (1,30): return n*2
    if ty == 24: return n
    if ty == 25: return n*2
    if ty == 26: return n*4
    if ty == 27: return n*8
    if ty == 28: return n*8
    if ty in BLK:
        assert n % BLK[ty] == 0
        return (n // BLK[ty]) * BBLK[ty]
    raise SystemExit(f"size for type {ty} unknown")

def map_name(n):
    """our HF-ish name -> unsloth's llama.cpp name"""
    m = re.match(r"model\.language_model\.layers\.(\d+)\.conv\.conv\.weight", n)
    if m: return f"blk.{m.group(1)}.shortconv.conv.weight"
    m = re.match(r"model\.language_model\.layers\.(\d+)\.conv\.(\w+)_proj\.weight", n)
    if m: return f"blk.{m.group(1)}.shortconv.{m.group(2)}_proj.weight"
    m = re.match(r"model\.language_model\.layers\.(\d+)\.feed_forward\.(\w+)\.weight", n)
    if m:
        t = {"w1": "ffn_gate", "w2": "ffn_down", "w3": "ffn_up"}[m.group(2)]
        return f"blk.{m.group(1)}.{t}.weight"
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(\w+)_norm\.weight", n)
    if m:
        t = "attn_norm" if m.group(2) == "operator" else m.group(2)
        return f"blk.{m.group(1)}.{t}.weight"
    if n == "model.language_model.embed_tokens.weight": return "token_embd.weight"
    if n == "model.language_model.embedding_norm.weight": return "token_embd_norm.weight"
    if n == "model.lm_head.weight": return "output.weight"
    return n

def main(ours_path, theirs_path):
    kv_o, t_o, ds_o, mm_o = gguf_all(ours_path)
    kv_t, t_t, ds_t, mm_t = gguf_all(theirs_path)
    theirs = {n: (ty, dims, off) for n, ty, dims, off in t_t}
    ours = {}
    for n, ty, dims, off in t_o:
        ours[map_name(n)] = (ty, dims, off)
    shared_q = []
    diffs = []
    unmatched = []
    for n, (ty, dims, off) in sorted(ours.items()):
        if n not in theirs:
            unmatched.append(n); continue
        tty, tdims, toff = theirs[n]
        if ty == 0 or tty == 0:  # ours kept F32 or theirs F32
            continue
        if ty != tty:
            diffs.append((n, TYNAME.get(ty,ty), TYNAME.get(tty,tty), "dtype")); continue
        if dims != tdims:
            diffs.append((n, dims, tdims, "dims")); continue
        b = payload_bytes(ty, dims)
        a = bytes(mm_o[ds_o+off:ds_o+off+b])
        c = bytes(mm_t[ds_t+toff:ds_t+toff+b])
        if a == c:
            shared_q.append(n)
        else:
            nd = sum(1 for x, y in zip(a, c) if x != y)
            diffs.append((n, TYNAME.get(ty,ty), f"{nd}/{b} bytes differ", "payload"))
    print(f"byte-exact shared quantized tensors: {len(shared_q)}")
    print(f"differences: {len(diffs)}")
    for d in diffs[:25]: print("  !", d)
    if len(diffs) > 25: print(f"  ... {len(diffs)-25} more")
    print(f"ours-only (unmatched after mapping): {len(unmatched)}")
    for u in unmatched[:10]: print("  ?", u)
    return len(shared_q), len(diffs)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

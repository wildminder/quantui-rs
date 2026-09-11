"""Compare renamed LM tensors between two GGUFs across name conventions.

Typical case: one file keeps HF names (model.language_model.*), the
other uses llama.cpp names (blk.N.*). Maps old->new via the same table
the converter uses, then compares dtype, shape, and payload bytes.
Also reports which side has tensors the mapping cannot pair, and any
mapped pair whose dtype differs.
"""

import mmap
import struct
import sys
from pathlib import Path

GGML_TYPE = {
    0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32),
    3: ("Q4_1", 20, 32), 6: ("Q5_0", 22, 32), 7: ("Q5_1", 24, 32),
    8: ("Q8_0", 34, 32), 10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256),
    12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256),
    16: ("BF16", 2, 1),
}

# The dense mapping table (mirror of gguf_names.rs hf_to_gguf_name)
DENSE = {
    "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
    "o_proj": "attn_output", "out_proj": "attn_output",
    "q_layernorm": "attn_q_norm", "k_layernorm": "attn_k_norm",
    "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
}


def hf_to_gguf(name):
    n = name
    for p in ("model.language_model.", "model.model."):
        if n.startswith(p):
            n = n[len(p):]
            break
    if n in ("model.embed_tokens.weight", "embed_tokens.weight"):
        return "token_embd.weight"
    if n == "lm_head.weight":
        return "output.weight"
    if n in ("model.norm.weight", "norm.weight"):
        return "output_norm.weight"
    for p in ("model.layers.", "layers."):
        if n.startswith(p):
            rest = n[len(p):]
            idx, tail = rest.split(".", 1)
            core = tail
            suffix = ""
            for s in (".weight", ".bias"):
                if core.endswith(s):
                    core, suffix = core[: -len(s)], s
                    break
            path = core.replace("self_attn.", "").replace("mlp.", "")
            mapped = DENSE.get(path)
            if mapped:
                return f"blk.{idx}.{mapped}{suffix}"
            return None
    return None


def parse(path):
    f = open(path, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ver, nt, nk = struct.unpack_from("<IIQQ", data, 0)
    pos = 24

    def rstr(p):
        (n,) = struct.unpack_from("<Q", data, p)
        p += 8
        return data[p : p + n].decode("utf-8"), p + n

    for _ in range(nk):
        k, pos = rstr(pos)
        (t,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if t == 8:
            _, pos = rstr(pos)
        elif t == 9:  # array
            (et,) = struct.unpack_from("<I", data, pos)
            pos += 4
            (n,) = struct.unpack_from("<Q", data, pos)
            pos += 8
            if et == 8:  # array of strings
                for _ in range(n):
                    _, pos = rstr(pos)
            else:
                pos += n * {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                            10: 8, 11: 8, 12: 8}[et]
        elif t in (0, 1, 7):   # uint8 / int8 / bool
            pos += 1
        elif t in (2, 3):      # uint16 / int16
            pos += 2
        elif t in (4, 5, 6):   # uint32 / int32 / float32
            pos += 4
        elif t in (10, 11, 12):  # float64 / int64 / uint64
            pos += 8

    tensors = {}
    for _ in range(nt):
        nm, pos = rstr(pos)
        (nd,) = struct.unpack_from("<I", data, pos)
        pos += 4
        ne = []
        for _ in range(nd):
            (d,) = struct.unpack_from("<Q", data, pos)
            pos += 8
            ne.append(d)
        (dt,) = struct.unpack_from("<I", data, pos)
        pos += 4
        (off,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        tensors[nm] = (dt, ne, off)

    alignment = 32
    pad = (alignment - (pos % alignment)) % alignment
    return f, data, tensors, pos + pad


def plen(dt, ne):
    _, tsz, blck = GGML_TYPE[dt]
    n = 1
    for d in ne:
        n *= d
    return (n + blck - 1) // blck * tsz


def main():
    fa, da, ta, sa = parse(sys.argv[1])
    fb, db, tb, sb = parse(sys.argv[2])

    # Build old(HF) -> new(llama.cpp) pairs through the mapping.
    pairs = []
    unmapped = []
    for old_name in ta:
        new_name = hf_to_gguf(old_name)
        if new_name is None:
            continue
        if new_name in tb:
            pairs.append((old_name, new_name))
        else:
            unmapped.append((old_name, new_name))

    print(f"mapped pairs: {len(pairs)}  target-missing: {len(unmapped)}")
    for o, n in unmapped[:5]:
        print(f"  target missing: {o} -> {n}")

    dtype_diff = []
    shape_diff = []
    byte_diff = []
    for old_name, new_name in pairs:
        dta, nea, offa = ta[old_name]
        dtb, neb, offb = tb[new_name]
        if dta != dtb:
            dtype_diff.append((old_name, new_name, dta, dtb, nea, neb))
        elif nea != neb:
            shape_diff.append((old_name, new_name, nea, neb))
        else:
            la = plen(dta, nea)
            pa = sa + offa
            pb = sb + offb
            if da[pa : pa + la] != db[pb : pb + la]:
                byte_diff.append((old_name, new_name, la))

    print(f"dtype differs : {len(dtype_diff)}")
    for o, n, a, b, nea, neb in dtype_diff:
        print(f"  {o} -> {n}: {GGML_TYPE[a][0]} vs {GGML_TYPE[b][0]}"
             f"  ne_old={nea} ne_new={neb}")
    print(f"shape differs : {len(shape_diff)}")
    for o, n, a, b in shape_diff[:5]:
        print(f"  {o} -> {n}: {a} vs {b}")
    print(f"bytes differ   : {len(byte_diff)}")
    for o, n, l in byte_diff[:5]:
        print(f"  {o} -> {n} ({l} B)")

    # Which new-file blk.* names had no old HF counterpart?
    mapped_targets = {n for _, n in pairs}
    orphan_new = [n for n in tb if n not in mapped_targets
                  and (n.startswith("blk.") or n in
                       ("token_embd.weight", "output.weight",
                        "output_norm.weight"))]
    print(f"new-side llama names with no old counterpart: {len(orphan_new)}")
    for n in orphan_new[:5]:
        print(f"  {n}: dtype={GGML_TYPE[tb[n][0]][0]} ne={tb[n][1]}")


if __name__ == "__main__":
    main()

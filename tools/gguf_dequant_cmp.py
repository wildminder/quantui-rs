#!/usr/bin/env python
"""Dequantize two GGUFs (llama-style names vs audiocpp native names), align
them through a name map, and compare against the BF16 safetensors source.

Answers the real question: "does our GGUF quantize the same tensors the same
way, and is the result as accurate?"

Usage: gguf_dequant_cmp.py <ours.gguf> <ref.gguf> <source.safetensors>
"""
import json
import struct
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_dump import parse, GGML  # noqa: E402

BLK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
}


def dequant_q4_0(raw, n):
    """block_q4_0 = {fp16 d; u8 qs[16]} -> 32 elems.
    llama.cpp: y[j] = (qs[j]&0xF - 8)*d ; y[j+16] = (qs[j]>>4 - 8)*d
    (the two nibble planes are SPLIT, not interleaved)."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32).ravel()
    q = b[:, 2:].astype(np.int32)
    lo = (q & 0x0F) - 8
    hi = (q >> 4) - 8
    out = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (out * d[:, None]).ravel()[:n]


def dequant_f16(raw, n):
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)[:n]


def dequant_bf16(raw, n):
    u = np.frombuffer(raw, dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)[:n]


def dequant_f32(raw, n):
    return np.frombuffer(raw, dtype=np.float32)[:n]


def dequant_q2_k(raw, n):
    """block_q2_K = { u8 scales[16]; u8 qs[64]; fp16 d; fp16 dmin } (84 bytes, 256 elems).
    Layout order is scales, qs, d, dmin — NOT scales, d, dmin, qs.
    Port of llama.cpp dequantize_row_q2_K."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 84)
    nb = b.shape[0]
    out = np.empty((nb, 256), dtype=np.float32)
    for i in range(nb):
        blk = b[i]
        scales = blk[0:16]
        qs = blk[16:80]
        d = np.frombuffer(blk[80:82].tobytes(), dtype=np.float16)[0].astype(np.float32)
        dmin = np.frombuffer(blk[82:84].tobytes(), dtype=np.float16)[0].astype(np.float32)
        y = out[i]
        is_ = 0
        qoff = 0
        pos = 0
        for _ in range(2):  # n += 128
            shift = 0
            for _j in range(4):
                sc = int(scales[is_]); is_ += 1
                dl = d * (sc & 0x0F); ml = dmin * (sc >> 4)
                for l in range(16):
                    y[pos] = dl * float((qs[qoff + l] >> shift) & 3) - ml
                    pos += 1
                sc = int(scales[is_]); is_ += 1
                dl = d * (sc & 0x0F); ml = dmin * (sc >> 4)
                for l in range(16):
                    y[pos] = dl * float((qs[qoff + l + 16] >> shift) & 3) - ml
                    pos += 1
                shift += 2
            qoff += 32
    return out.ravel()[:n]


DEQ = {"Q4_0": dequant_q4_0, "F16": dequant_f16, "BF16": dequant_bf16,
       "F32": dequant_f32, "Q2_K": dequant_q2_k}


# ---- llama-style (ours)  <->  HF native (ref) -------------------------------
CORE = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
}


def ours_to_ref(name):
    """Map our llama-style GGUF name to the reference's `model_weights/<hf>`."""
    if name == "token_embd.weight":
        return "model_weights/model.embed_tokens.weight"
    if name == "output.weight":
        return "model_weights/lm_head.weight"
    if name == "output_norm.weight":
        return "model_weights/model.norm.weight"
    if name.startswith("blk."):
        _, idx, rest = name.split(".", 2)
        core = rest.rsplit(".", 1)[0]
        suf = rest.rsplit(".", 1)[1]
        if core in CORE:
            return f"model_weights/model.layers.{idx}.{CORE[core]}.{suf}"
    # unmapped names pass through with the prefix
    return "model_weights/" + name


def load(path):
    ver, kv, ts, base = parse(path)
    out = {}
    fh = open(path, "rb")
    for t in ts:
        name = GGML.get(t["dtype"], (f"?{t['dtype']}", 1))[0]
        n_el = int(np.prod(t["ne"]))
        bs, bpb = BLK.get(name, (1, 2))
        nbytes = n_el // bs * bpb
        fh.seek(base + t["off"])
        out[t["name"]] = (name, t["ne"], fh.read(nbytes), n_el)
    fh.close()
    return out, kv


def rel(got, ref):
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def main():
    ours_p, ref_p, src_p = sys.argv[1], sys.argv[2], sys.argv[3]
    A, kva = load(ours_p)
    B, kvb = load(ref_p)

    with open(src_p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        sh = json.loads(fh.read(n))
        sh.pop("__metadata__", None)
        sbase = 8 + n
    sfile = open(src_p, "rb")

    def src_f32(key):
        e = sh[key]
        a, z = e["data_offsets"]
        sfile.seek(sbase + a)
        raw = sfile.read(z - a)
        if e["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16)
            return (u.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])
        if e["dtype"] == "F16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(e["shape"])
        return np.frombuffer(raw, dtype=np.float32).reshape(e["shape"])

    # dtype assignment comparison
    print("=== dtype assignment ===")
    mism = []
    for name, (dt, ne, raw, nel) in A.items():
        rname = ours_to_ref(name)
        if rname not in B:
            continue
        rdt = B[rname][0]
        if dt != rdt:
            mism.append((name, rname, dt, rdt))
    print(f"  ours tensors        : {len(A)}")
    print(f"  ref  tensors        : {len(B)}")
    print(f"  dtype mismatches    : {len(mism)}")
    from collections import Counter
    print(f"    transitions       : {dict(Counter((m[2], m[3]) for m in mism))}")
    for name, rname, dt, rdt in mism[:12]:
        print(f"      {name:44s} ours={dt:6s} ref={rdt}")

    # numeric comparison vs source
    print("\n=== reconstruction error vs BF16 source ===")
    rows = []
    for name, (dt, ne, raw, nel) in A.items():
        rname = ours_to_ref(name)
        if rname not in B:
            continue
        ref_key = rname[len("model_weights/"):]
        if ref_key not in sh or len(sh[ref_key]["shape"]) != 2:
            continue
        src = src_f32(ref_key).astype(np.float32)
        if dt not in DEQ or B[rname][0] not in DEQ:
            continue
        try:
            a = DEQ[dt](raw, nel)
            b = DEQ[B[rname][0]](B[rname][2], B[rname][3])
        except Exception:
            continue
        if a.size != src.size or b.size != src.size:
            print(f"  [skip size] {name}: ours={a.size} ref={b.size} src={src.size}")
            continue
        rows.append((name, rel(a, src.ravel()), rel(b, src.ravel()), dt, B[rname][0]))
    if rows:
        aa = np.array([r[1] for r in rows])
        bb = np.array([r[2] for r in rows])
        print(f"  tensors compared    : {len(rows)}")
        print(f"  OURS mean rel-L2    : {aa.mean():.6f}   max {aa.max():.6f}")
        print(f"  REF  mean rel-L2    : {bb.mean():.6f}   max {bb.max():.6f}")
        print(f"  ours better on {int((aa < bb * 0.999).sum())}, "
              f"worse on {int((aa > bb * 1.001).sum())}, tie {int((abs(aa-bb)<=1e-9).sum())}")
        print("\n  top-8 where OURS is worse:")
        for n, x, y, dt, rdt in sorted(rows, key=lambda r: r[1] - r[2], reverse=True)[:8]:
            print(f"    {n:44s} ours={x:.5f}({dt}) ref={y:.5f}({rdt})")


if __name__ == "__main__":
    main()

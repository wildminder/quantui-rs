#!/usr/bin/env python
"""Check whether a GGUF tensor's payload really matches its declared dtype.

Specifically hunts for a *mislabel*: BF16 bytes stored under an F16 label (or
vice versa). Those are silent corruption — a runtime reading F16 sees garbage.

Compares the GGUF payload against the same tensor from a safetensors source,
under both interpretations (raw byte-identity vs numeric equality).
"""
import struct
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_dump import parse, GGML  # noqa: E402

GGUF_TO_ST = {  # gguf dtype name -> safetensors dtype name
    "F32": "F32", "F16": "F16", "BF16": "BF16", "Q4_0": "I8",
}


def np_dt(name):
    return {"F32": np.float32, "F16": np.float16, "BF16": None}[name]


def decode_bf16(raw):
    return np.frombuffer(raw, dtype=np.uint16).view(np.uint16)


def main():
    gguf_path, st_path = sys.argv[1], sys.argv[2]
    with open(gguf_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
    base = 8 + n
    ver, kv, tensors, _ = parse(gguf_path)

    st = safe_open(st_path, framework="np")
    st_keys = set(st.keys())

    print(f"gguf: {gguf_path}\nst  : {st_path}\n")
    print("--- unquantized-tensor dtype check (F16/BF16/F32) ---")
    hdr_names = [
        (t["name"], GGML.get(t["dtype"], ("?", 1))[0], t["ne"], t["off"])
        for t in tensors
        if GGML.get(t["dtype"], ("T?", 1))[0] in ("F16", "BF16", "F32")
    ]
    # map gguf name -> st name (strip the `model_weights/` prefix)
    checked = mismatch = 0
    for gname, dt, ne, off in hdr_names:
        cand = gname.split("/")[-1]
        if cand not in st_keys:
            continue
        with open(gguf_path, "rb") as fh:
            fh.seek(base + off)
            raw = fh.read(int(np.prod(ne)) * 2 if dt != "F32" else int(np.prod(ne)) * 4)
        st_dt = st.get_slice(cand).get_dtype()
        checked += 1

        # interpretation A: treat gguf bytes AS the declared dtype
        # interpretation B: treat gguf bytes as the SOURCE dtype (mislabel test)
        ref = st.get_tensor(cand)
        ref_bf16 = ref.view(np.uint16).tobytes() if st_dt == "BFLOAT16" else None
        ref_f16 = ref.view(np.uint16).tobytes() if st_dt == "FLOAT16" else None
        ref_f32 = ref.tobytes() if st_dt == "FLOAT32" else None

        as_declared = (ref_bf16 == raw) if dt == "BF16" else (
            (ref_f16 == raw) if dt == "F16" else (ref_f32 == raw))
        as_source = (ref_bf16 == raw) if st_dt == "BFLOAT16" else (
            (ref_f16 == raw) if st_dt == "FLOAT16" else (ref_f32 == raw))

        if not as_declared:
            mismatch += 1
            status = "MISLABELLED" if as_source else "genuinely-converted"
            print(f"  {gname}")
            print(f"     declared={dt}  source={st_dt}  bytes-as-declared={as_declared}"
                  f"  bytes-as-source={as_source}   -> {status}")
    print(f"\nchecked {checked} unquantized tensors, {mismatch} not byte-identical "
          f"under their declared dtype")


if __name__ == "__main__":
    main()

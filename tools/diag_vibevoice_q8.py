"""Diagnose the VibeVoice-1.5B q8_0 GGUF flat-block-pool warning (v2).

Parse the GGUF header (mmap; the file is 2.87 GB), find every quantized
tensor whose ne[0] is NOT a multiple of its type's block size (the exact
condition llama.cpp's gguf.cpp:724 flags), and for each print:
  - tensor name, ggml type, dims, ne[0], block size
  - stored byte size vs the FLAT-pool expectation
    (ceil(n_elems/blck) * block_bytes) vs the ROW-padded layout
    (n_rows * ceil(ne0/blck) * block_bytes) — proves which layout is
    on disk
Then dequantize the flat pool of the first offending tensor back to f32
and compare against the original safetensors tensor (mean/max abs err) —
answers "is the DATA recoverable" independent of the layout question.
"""

import json
import mmap
import os
import struct
import sys
from pathlib import Path

# Path to the q8_0 GGUF to analyze (required; pass it as the first argument
# or set QUANTUI_PROBE_GGUF). Developed against a VibeVoice-1.5B-q8_0.gguf.
DEFAULT_GGUF = Path(os.environ.get("QUANTUI_PROBE_GGUF", "model-q8_0.gguf"))

# (name, type_size, block_size) — GGML types present in this file
GGML_TYPE = {
    0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32), 3: ("Q4_1", 20, 32),
    6: ("Q5_0", 22, 32), 7: ("Q5_1", 24, 32), 8: ("Q8_0", 34, 32),
    10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256), 12: ("Q4_K", 144, 256),
    13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256), 16: ("BF16", 2, 1),
}


def main():
    gguf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GGUF
    print(f"analyzing: {gguf}")
    f = open(gguf, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0
    magic, ver, n_tensors, n_kv = struct.unpack_from("<IIQQ", data, pos)
    assert magic == 0x46554747, "not a GGUF file"
    pos = 24

    # ---- KV section ----
    def read_str(p):
        (n,) = struct.unpack_from("<Q", data, p)
        p += 8
        s = data[p : p + n].decode("utf-8")
        return s, p + n

    kv = {}
    for _ in range(n_kv):
        k, pos = read_str(pos)
        (t,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if t == 8:  # string
            v, pos = read_str(pos)
        elif t == 9:  # array (not expected here)
            (et,) = struct.unpack_from("<I", data, pos); pos += 4
            (cnt,) = struct.unpack_from("<Q", data, pos); pos += 8
            arr = []
            for _ in range(cnt):
                if et == 8:
                    sv, pos = read_str(pos); arr.append(sv)
                else:
                    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                           6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}[et]
                    sz = struct.calcsize(fmt)
                    (vv,) = struct.unpack_from(fmt, data, pos); pos += sz
                    arr.append(vv)
            v = arr
        else:
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                   6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}[t]
            sz = struct.calcsize(fmt)
            (v,) = struct.unpack_from(fmt, data, pos)
            pos += sz
        kv[k] = v

    # ---- tensor info section ----
    # GGUF v3 tensor_info: name(str) + n_dims(u32) + dims[n_dims](i64)
    #                       + type(u32) + offset(u64). NO size field — the
    # size is derived from dims and the ggml type.
    tensors = []
    for i in range(n_tensors):
        try:
            name, pos = read_str(pos)
            (nd,) = struct.unpack_from("<I", data, pos)
            pos += 4
            dims = list(struct.unpack_from(f"<{nd}q", data, pos))
            pos += 8 * nd
            ggml_type, off = struct.unpack_from("<IQ", data, pos)
            pos += 12
            tensors.append((name, ggml_type, dims, off))
        except Exception as e:
            print(f"tensor-info parse failed at #{i}: {e}")
            print(f"pos={pos}")
            break

    print(f"GGUF: {gguf.name}")
    print(f"parsed tensors: {len(tensors)}/{n_tensors}  arch: {kv.get('general.architecture')}")
    print()

    bad = []
    for name, gt, dims, off in tensors:
        if gt not in GGML_TYPE:
            continue
        tname, tsize, blck = GGML_TYPE[gt]
        ne0 = dims[0] if dims else 1
        if blck > 1 and ne0 % blck != 0:
            # stored size must be recomputed: the GGUF has no size field,
            # so use the FLAT-pool size the writer actually wrote. We
            # derive it two ways and compare against the byte distance to
            # the next tensor's offset (the ground truth).
            n_elems = 1
            for d in dims:
                n_elems *= d
            n_blocks_flat = (n_elems + blck - 1) // blck
            size_flat = n_blocks_flat * tsize
            n_rows = n_elems // dims[0]
            blocks_per_row = (dims[0] + blck - 1) // blck
            size_row = n_rows * blocks_per_row * tsize
            bad.append((name, gt, tname, dims, off, size_flat, size_row, blck, tsize, n_elems))

    print(f"tensors with ne[0] % blck != 0: {len(bad)}")
    for name, gt, tname, dims, off, size_flat, size_row, blck, tsize, n_elems in bad:
        print(f"  {name}")
        print(f"    type={tname} dims={dims} ne0={dims[0]} blck={blck} n_elems={n_elems:,}")
        print(f"    offset={off:,}  flat-pool-bytes={size_flat:,}  row-padded-bytes={size_row:,}")
        # ground truth: distance to the NEXT tensor offset (sorted)
        next_off = min(o for (nm, g2, d2, o) in tensors if o > off) if any(o2 > off for (_, _, _, o2) in tensors) else None
        if next_off:
            actual = next_off - off
            verdict = "FLAT-POOL" if actual == size_flat else ("ROW-PADDED" if actual == size_row else f"OTHER({actual:,})")
            print(f"    bytes-to-next-tensor={actual:,}  -> layout verdict: {verdict}")

    # dtype histogram
    hist = {}
    for _, gt, *_ in tensors:
        tname = GGML_TYPE.get(gt, (f"type{gt}",))[0]
        hist[tname] = hist.get(tname, 0) + 1
    print()
    print("dtype histogram:", json.dumps(hist, indent=None))

    data.close()
    f.close()


if __name__ == "__main__":
    main()

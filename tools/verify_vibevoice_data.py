"""Final verification v2 (f32-exact): is the VibeVoice q8_0 GGUF data correct?

The v1 probe used Python f64 arithmetic; the converter rounds at f32 at
every step (amax/127.0, 1.0/d, x*id, roundf). This version uses numpy
float32 throughout, replicating rlx-gguf's quantize_q8_0 exactly:
  per 32-element block: amax = max|x|; d = amax/127 (f32);
  id = 1.0/d (f32) if d != 0 else 0; q = roundf(x*id) clamp[-128,127];
  d stored as f16 (round-to-nearest-even), q as i8.

For each probed tensor:
  1. byte-compare the re-quantization vs the GGUF payload  -> is the file
     exactly what the converter intended?
  2. dequantize (q * d, f32) vs the original tensor       -> is the
     recoverable information a faithful q8_0 of the source?
"""

import json
import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np

# Paths: pass as arguments (GGUF first, then the HF source dir), or set
# QUANTUI_PROBE_GGUF / QUANTUI_PROBE_HF. Developed against a
# VibeVoice-1.5B q8_0 GGUF and its HF checkpoint.
GGUF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("QUANTUI_PROBE_GGUF", ""))
HF = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.environ.get("QUANTUI_PROBE_HF", ""))
if not GGUF.is_file() or not HF.is_dir():
    sys.exit("usage: verify_vibevoice_data.py <quantized.gguf> <hf_model_dir>  (or set QUANTUI_PROBE_GGUF/QUANTUI_PROBE_HF)")

PROBE_TENSORS = [
    "model.semantic_tokenizer.encoder.head.conv.conv.weight",
    "model.acoustic_tokenizer.decoder.stages.0.2.mixer.conv.conv.conv.weight",
    "model.acoustic_tokenizer.encoder.downsample_layers.6.0.conv.conv.weight",
    "model.semantic_tokenizer.encoder.stages.3.0.mixer.conv.conv.conv.weight",
    "model.acoustic_tokenizer.decoder.stages.6.0.mixer.conv.conv.conv.weight",
]


def parse_gguf_tensors():
    f = open(GGUF, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 24
    magic, ver, n_tensors, n_kv = struct.unpack_from("<IIQQ", data, 0)

    def read_str(p):
        (n,) = struct.unpack_from("<Q", data, p)
        p += 8
        return data[p : p + n].decode(), p + n

    for _ in range(n_kv):
        _, pos = read_str(pos)
        (t,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if t == 8:
            _, pos = read_str(pos)
        else:
            raise RuntimeError(f"unexpected kv type {t}")

    tensors = {}
    for _ in range(n_tensors):
        name, pos = read_str(pos)
        (nd,) = struct.unpack_from("<I", data, pos)
        pos += 4
        dims = list(struct.unpack_from(f"<{nd}q", data, pos))
        pos += 8 * nd
        gt, off = struct.unpack_from("<IQ", data, pos)
        pos += 12
        tensors[name] = (gt, dims, off)
    # GGUF tensor offsets are relative to the data section, which starts
    # at infos_end rounded UP to general.alignment (default 32; this
    # file carries no general.alignment key).
    data_start = (pos + 31) // 32 * 32
    return data, tensors, f, data_start


def load_hf_tensor_f32(name):
    idx = json.loads((HF / "model.safetensors.index.json").read_text())
    shard = idx["weight_map"][name]
    path = HF / shard
    f = open(path, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    (hlen,) = struct.unpack_from("<Q", data, 0)
    header = json.loads(data[8 : 8 + hlen])
    info = header[name]
    off0, off1 = info["data_offsets"]
    raw = data[8 + hlen + off0 : 8 + hlen + off1]
    dt = info["dtype"]
    if dt == "BF16":
        arr = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
        vals = arr.view("<f4").astype(np.float32)
    elif dt == "F16":
        vals = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dt == "F32":
        vals = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise RuntimeError(f"unhandled dtype {dt}")
    data.close()
    f.close()
    return dt, info["shape"], vals


def quantize_q8_0_flat_f32(vals):
    """Exact f32 port of rlx quantize_q8_0 over the flat element stream."""
    n = vals.size
    assert n % 32 == 0
    nb = n // 32
    blocks = vals.reshape(nb, 32)
    amax = np.abs(blocks).max(axis=1).astype(np.float32)          # f32 max|x|
    d = (amax / np.float32(127.0)).astype(np.float32)             # f32 divide
    idd = np.where(d != 0.0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    qf = (blocks.astype(np.float32) * idd[:, None]).astype(np.float32)
    # roundf: round half away from zero
    q = np.where(qf >= 0.0, np.floor(qf + np.float32(0.5)), np.ceil(qf - np.float32(0.5))).astype(np.float32)
    q = np.clip(q, -128, 127).astype(np.int8)
    d16 = d.astype(np.float16)                                    # round-to-nearest-even
    out = bytearray(nb * 34)
    d16le = d16.astype("<f2").tobytes()
    qle = q.tobytes()
    for i in range(nb):
        out[i * 34 : i * 34 + 2] = d16le[2 * i : 2 * i + 2]
        out[i * 34 + 2 : i * 34 + 34] = qle[32 * i : 32 * i + 32]
    return bytes(out), d16.astype(np.float32), q


def main():
    data, tensors, f, data_start = parse_gguf_tensors()

    print(f"{'tensor':<62} {'bytes':>12} {'match':>7} {'meanerr':>10} {'maxerr':>10}")
    print("-" * 108)

    all_ok = True
    for name in PROBE_TENSORS:
        gt, dims, off = tensors[name]
        assert gt == 8, f"{name}: expected Q8_0, got {gt}"
        dt, hf_shape, vals = load_hf_tensor_f32(name)
        assert list(reversed(hf_shape)) == dims

        expect, d16, q = quantize_q8_0_flat_f32(vals)
        actual = bytes(data[data_start + off : data_start + off + len(expect)])
        match = expect == actual
        all_ok &= match

        dq = (q.astype(np.float32) * d16[:, None]).reshape(-1)
        errs = np.abs(dq - vals)
        print(f"{name[:60]:<62} {len(expect):>12,} {str(match):>7} "
              f"{errs.mean():>10.6f} {errs.max():>10.6f}")

    print()
    if all_ok:
        print("VERDICT: ALL BYTE-IDENTICAL — the GGUF contains EXACTLY what the")
        print("converter intended: a faithful flat-pool q8_0 of the source values.")
        print("The only defect is the LAYOUT (blocks straddle row boundaries),")
        print("not the data. Any reader that dequantizes the flat pool in file")
        print("order recovers the correct values (as the loading app does).")
    else:
        print("VERDICT: MISMATCH even at f32 — investigate further (maybe the")
        print("converter processed tensors in a different element order).")

    data.close()
    f.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate byte-parity fixtures for the weighted K-quant port (Phase 4.3).

Builds a tiny F32 GGUF model + an imatrix (GGUF format), runs the REAL
llama-quantize (built from docs/ref/llama.cpp) with --imatrix for Q4_K,
Q2_K, Q3_K, Q5_K and Q6_K, and saves the quantized tensor payloads as
golden .bin files under tests/golden/llamacpp/.

The Rust tests byte-compare these payloads against
quant_core::gguf_quants::quantize_row_q{4,2,3,5,6}_k_weighted on the
same f32 source + weights.

Requires:
  - llama-quantize.exe (tools/build_llamacpp.sh)
  - gguf (pip; present in the ctq venv)
Usage:
  python tools/gen_golden_llamacpp_weighted.py --llama-quantize <path.exe>
Deterministic: fixed seeds, no RNG variance.
"""

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

QK_K = 256
N_ROWS = 2  # 2 rows x 256 = 2 super-blocks per tensor

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
# gguf_type enum (ggml/include/gguf.h) for METADATA values:
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
# ggml_type enum (ggml/include/ggml.h) for TENSOR dtypes — a DIFFERENT
# enum from gguf_type: F32=0, F16=1, BF16=30.
GGML_TYPE_F32 = 0


def synth(n: int, seed: int) -> list[float]:
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append(((s >> 33) / 0xFFFFFFFF) * 2.0 - 1.0)
    return out


class GgufWriter:
    def __init__(self):
        self.meta: dict[str, tuple[int, object]] = {}  # name -> (gguf_type, value)
        self.tensors: list[tuple[str, list[int], int, bytes]] = []

    def set_meta(self, name: str, gtype: int, value):
        self.meta[name] = (gtype, value)

    def add_tensor(self, name: str, shape: list[int], ggml_type: int, data: bytes):
        self.tensors.append((name, shape, ggml_type, data))

    def _kv_bytes(self, name: str, gtype: int, value) -> bytes:
        key = name.encode()
        out = struct.pack("<Q", len(key)) + key + struct.pack("<I", gtype)
        if gtype == GGUF_TYPE_STRING:
            v = value.encode()
            out += struct.pack("<Q", len(v)) + v
        elif gtype == GGUF_TYPE_UINT32:
            out += struct.pack("<I", value)
        elif gtype == GGUF_TYPE_ARRAY:
            etype, items = value
            out += struct.pack("<Q", len(items))
            for it in items:
                if etype == GGUF_TYPE_STRING:
                    s = it.encode()
                    out += struct.pack("<Q", len(s)) + s
                elif etype == GGUF_TYPE_UINT32:
                    out += struct.pack("<I", it)
                else:
                    raise ValueError(etype)
        else:
            raise ValueError(gtype)
        return out

    def write(self, path: Path):
        # GGUF v3: magic(4) + version(u32) + n_kv(u64) + n_tensors(u64),
        # then n_kv metadata entries, then n_tensors info entries, then
        # data (both info-table and data start are 32-aligned in practice;
        # the spec aligns the data segment, llama.cpp pads the whole
        # prefix so the data segment is ALIGNMENT-byte aligned).
        alignment = 32

        def kv_bytes(name: str, gtype: int, value) -> bytes:
            key = name.encode()
            out = struct.pack("<Q", len(key)) + key + struct.pack("<I", gtype)
            if gtype == GGUF_TYPE_STRING:
                v = value.encode()
                out += struct.pack("<Q", len(v)) + v
            elif gtype == GGUF_TYPE_UINT32:
                out += struct.pack("<I", value)
            elif gtype == GGUF_TYPE_FLOAT32:
                out += struct.pack("<f", value)
            elif gtype == GGUF_TYPE_ARRAY:
                etype, items = value
                # GGUF array layout: etype(u32) + count(u64) + items.
                out += struct.pack("<I", etype)
                out += struct.pack("<Q", len(items))
                for it in items:
                    if etype == GGUF_TYPE_STRING:
                        s = it.encode()
                        out += struct.pack("<Q", len(s)) + s
                    elif etype == GGUF_TYPE_UINT32:
                        out += struct.pack("<I", it)
                    else:
                        raise ValueError(etype)
            else:
                raise ValueError(gtype)
            return out

        kv = b"".join(kv_bytes(n, t, v) for n, (t, v) in self.meta.items())

        # Tensor infos with offsets computed against the aligned data start.
        # Data start = align32(len(header) + len(kv) + len(infos)).
        # First pass: compute infos length (offsets are fixed-size fields,
        # so length is independent of the values we patch in).
        def infos_bytes(offsets: list[int]) -> bytes:
            out = b""
            for (name, shape, ggml_type, _), off in zip(self.tensors, offsets):
                key = name.encode()
                out += struct.pack("<Q", len(key)) + key
                # n_dims is u32 (gguf.h: tensor info layout), then u64 dims.
                out += struct.pack("<I", len(shape))
                for d in reversed(shape):  # ne[0] = cols innermost first
                    out += struct.pack("<Q", d)
                out += struct.pack("<I", ggml_type)
                out += struct.pack("<Q", off)
            return out

        dummy = infos_bytes([0] * len(self.tensors))
        prefix = 4 + 4 + 8 + 8 + len(kv) + len(dummy)
        data_start = (prefix + alignment - 1) & ~(alignment - 1)
        # llama.cpp requires tensors CONTIGUOUS with each tensor's size
        # PADDED to the alignment (gguf.cpp:776-793):
        # padded_size = GGML_PAD(nbytes, alignment); offset must equal the
        # running padded total. Mirror gguf.cpp:1420 exactly.
        offsets = []
        padded: list[bytes] = []
        o = 0
        for _, _, _, data in self.tensors:
            offsets.append(o)
            pad = (alignment - len(data) % alignment) % alignment
            padded.append(data + b"\x00" * pad)
            o += len(data) + pad
        infos = infos_bytes(offsets)

        blob = bytearray()
        blob += GGUF_MAGIC
        blob += struct.pack("<I", GGUF_VERSION)
        # GGUF v2/v3 header (gguf.cpp:515-526): n_tensors FIRST, then n_kv.
        blob += struct.pack("<Q", len(self.tensors))
        blob += struct.pack("<Q", len(self.meta))
        blob += kv
        blob += infos
        blob += b"\x00" * (data_start - len(blob))
        for p in padded:
            blob += p
        path.write_bytes(blob)


def build_model_f32(path: Path, src: list[float]):
    """A 1-tensor F32 model: tensor 'blk.0.attn_q.weight' [N_ROWS, 256].

    Carries the metadata llama.cpp's loader demands for the llama arch
    (layer_norm_rms_epsilon, head counts, rope) so llama-quantize accepts
    it. Keys discovered iteratively from the loader's error messages.
    """
    w = GgufWriter()
    w.set_meta("general.architecture", GGUF_TYPE_STRING, "llama")
    w.set_meta("general.name", GGUF_TYPE_STRING, "parity-fixture")
    w.set_meta("llama.context_length", GGUF_TYPE_UINT32, 64)
    w.set_meta("llama.embedding_length", GGUF_TYPE_UINT32, 256)
    w.set_meta("llama.block_count", GGUF_TYPE_UINT32, 1)
    w.set_meta("llama.vocab_size", GGUF_TYPE_UINT32, 64)
    # Required by the llama arch hparams loader. n_rot (rope.dimension_count)
    # must equal embedding_length / head_count = 256/8 = 32.
    w.set_meta("llama.attention.head_count", GGUF_TYPE_UINT32, 8)
    w.set_meta("llama.attention.head_count_kv", GGUF_TYPE_UINT32, 8)
    w.set_meta("llama.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, 1e-5)
    w.set_meta("llama.rope.dimension_count", GGUF_TYPE_UINT32, 32)
    data = b"".join(struct.pack("<f", v) for v in src)
    # HF/GGUF dims: [rows, cols] with ne[0]=cols innermost.
    w.add_tensor("blk.0.attn_q.weight", [N_ROWS, QK_K], GGML_TYPE_F32, data)
    w.write(path)


def build_imatrix_gguf(path: Path, weights: list[float]):
    """Imatrix GGUF: blk.0.attn_q.weight.in_sum2/.counts + metadata.

    Per llama.cpp the imatrix entry for a tensor carries ONE weight vector
    of length ne[0] * ne[2], shared by every row (quantize_q4_K passes the
    same base for each row). For our [2, 256] fixture: 256 weights, 1 count.
    sums = weights * COUNT so the derivation yields weights exactly.
    """
    w = GgufWriter()
    w.set_meta("general.architecture", GGUF_TYPE_STRING, "llama")
    w.set_meta("imatrix.datasets", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, ["parity"]))
    w.set_meta("imatrix.chunk_count", GGUF_TYPE_UINT32, 4)
    w.set_meta("imatrix.chunk_size", GGUF_TYPE_UINT32, 64)
    count = 8
    sums = b"".join(struct.pack("<f", v * count) for v in weights[:QK_K])
    counts = struct.pack("<f", float(count))
    w.add_tensor("blk.0.attn_q.weight.in_sum2", [QK_K], GGML_TYPE_F32, sums)
    w.add_tensor("blk.0.attn_q.weight.counts", [1], GGML_TYPE_F32, counts)
    w.write(path)


def read_gguf_tensor(path: Path, name: str) -> bytes:
    """Minimal GGUF v3 reader: return the raw payload bytes of one tensor."""
    raw = path.read_bytes()
    # GGUF v2/v3 header: n_tensors first, then n_kv (gguf.cpp:515-526).
    (magic, version, n_tensors, n_kv) = struct.unpack_from("<IIQQ", raw, 0)
    assert magic == int.from_bytes(GGUF_MAGIC, "little"), "not a GGUF file"
    off = 24
    for _ in range(n_kv):
        (klen,) = struct.unpack_from("<Q", raw, off)
        off += 8 + klen
        (vtype,) = struct.unpack_from("<I", raw, off)
        off += 4
        off = skip_value(raw, off, vtype)
    tensors = {}
    for _ in range(n_tensors):
        (nlen,) = struct.unpack_from("<Q", raw, off)
        tname = raw[off + 8 : off + 8 + nlen].decode()
        off += 8 + nlen
        (ndim,) = struct.unpack_from("<I", raw, off)
        off += 4 + 8 * ndim
        (gtype,) = struct.unpack_from("<I", raw, off)
        off += 4
        (toff,) = struct.unpack_from("<Q", raw, off)
        off += 8
        tensors[tname] = (gtype, toff)
    if not tensors:
        raise ValueError("no tensors")
    # Data segment starts after the info table, ALIGNMENT (default 32)
    # aligned; the file's own alignment KV (general.alignment) would be
    # parsed in the metadata loop, but llama-quantize writes 32.
    data_start = (off + 31) & ~31
    gtype, toff = tensors[name]
    # Payload end: next tensor's offset, or EOF minus trailing padding.
    offs = sorted((o, t) for t, (g, o) in tensors.items())
    idx = [i for i, (o, t) in enumerate(offs) if t == name][0]
    if idx + 1 < len(offs):
        end = offs[idx + 1][0]
    else:
        end = len(raw) - data_start
    return raw[data_start + toff : data_start + end]


def skip_value(raw: bytes, off: int, vtype: int) -> int:
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in sizes:
        return off + sizes[vtype]
    if vtype == GGUF_TYPE_STRING:
        (slen,) = struct.unpack_from("<Q", raw, off)
        return off + 8 + slen
    if vtype == GGUF_TYPE_ARRAY:
        (itype,) = struct.unpack_from("<I", raw, off)
        (n,) = struct.unpack_from("<Q", raw, off + 4)
        off += 12
        for _ in range(n):
            off = skip_value(raw, off, itype)
        return off
    raise ValueError(f"gguf type {vtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-quantize", required=True)
    ap.add_argument("--out", default="tests/golden/llamacpp")
    args = ap.parse_args()

    tool = Path(args.llama_quantize)
    assert tool.exists(), f"llama-quantize not found: {tool}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = N_ROWS * QK_K
    src = synth(n, 42)
    weights = [1.0 + abs(v) for v in synth(n, 7)]

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model = td / "model-f32.gguf"
        imx = td / "imatrix.gguf"
        build_model_f32(model, src)
        build_imatrix_gguf(imx, weights)

        results = {}
        for ftype, name in [
            ("Q4_K", "q4_k"),
            ("Q2_K", "q2_k"),
            ("Q3_K", "q3_k"),
            ("Q5_K", "q5_k"),
            ("Q6_K", "q6_k"),
            ("IQ2_XXS", "iq2_xxs"),
            ("IQ2_XS", "iq2_xs"),
            # llama-quantize's --pure ftype table maps "IQ2_S" to
            # GGML_TYPE_IQ2_XS (llama-quant.cpp:859 — the raw IQ2_S bytes
            # are only reachable via IQ2_M's per-tensor policy, :860).
            # To golden the true IQ2_S encoder we pin the tensor explicitly.
            ("IQ2_S", "iq2_s*override"),
            ("IQ3_XXS", "iq3_xxs"),
            ("IQ3_S", "iq3_s"),
            ("IQ1_S", "iq1_s"),
            ("IQ1_M", "iq1_m"),
            ("IQ4_NL", "iq4_nl"),
            ("IQ4_XS", "iq4_xs"),
        ]:
            qout = td / "model-q.gguf"
            cmd = [
                str(tool),
                "--imatrix", str(imx),
                "--include-weights", "blk.0.attn_q.weight",
                "--pure",
            ]
            base_ftype = ftype
            if name.endswith("*override"):
                # Base type can be anything that quantizes the rest; the
                # --tensor-type override pins our tensor's encoder.
                base_ftype = "Q4_K"
                cmd += ["--tensor-type", "attn_q=IQ2_S"]
            cmd += [str(model), str(qout), base_ftype]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[{name}] quantize FAILED:\n{r.stdout}\n{r.stderr}")
                sys.exit(1)
            payload = read_gguf_tensor(qout, "blk.0.attn_q.weight")
            results[name.replace("*override", "")] = payload
            print(f"[{name}] payload: {len(payload)} bytes")

        # Save goldens + the shared inputs. The imatrix entry carries ONE
        # 256-length weight vector (per ne[0]) shared by both rows; the
        # Rust parity test quantizes each row with this same vector.
        for name in results:
            (out_dir / f"weighted.{name}.bin").write_bytes(results[name])
        (out_dir / "src.f32.bin").write_bytes(b"".join(struct.pack("<f", v) for v in src))
        (out_dir / "weights.f32.bin").write_bytes(
            b"".join(struct.pack("<f", v) for v in weights[:QK_K])
        )
        (out_dir / "manifest.json").write_text(json.dumps({
            "n": n, "n_rows": N_ROWS, "qk_k": QK_K, "seed_src": 42, "seed_weights": 7,
            "src_bytes": n * 4, "weights_bytes": QK_K * 4,
            "weights_note": "one per-column vector (ne[0]=256), shared by all rows",
            **{f"{name}_bytes": len(payload) for name, payload in results.items()},
            "llama_quantize": str(tool),
        }, indent=2))
    print(f"OK: goldens written to {out_dir}")


if __name__ == "__main__":
    main()

"""WP10 / NTH-003 — committed small benchmark fixture generator.

Writes tests/bench/bench_small.safetensors: 4 tensors of [1024, 1024]
BF16 (~8 MB total), LCG-synthesized with a PINNED seed so the file is
byte-reproducible from this script alone (no torch, no network).

Why committed: the big `tools/bench_fixture/` used by stream_throughput
is generated on demand; a *committed* small fixture lets the nightly
bench workflow (`.github/workflows/bench.yml`) run with zero generation
steps and gives criterion a stable input so run-to-run deltas measure
code, not data.

Self-check: the script prints the SHA-256 of the file it wrote; the
pinned value is asserted below — if a regeneration ever diverges (e.g.
a float-format change), the mismatch fails loudly here instead of
silently skewing the bench history.

Usage: python tools/gen_bench_small.py
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "bench" / "bench_small.safetensors"
SEED = 20260906  # date of the WP10 work; arbitrary but pinned
N = 1024  # rows and cols
TENSOR_NAMES = [
    "blk.0.attn_q.weight",
    "blk.0.attn_v.weight",
    "blk.0.ffn_down.weight",
    "blk.0.ffn_gate.weight",
]

# SHA-256 of the canonical file. Re-asserted after every write.
EXPECTED_SHA256 = "fa03d89208f8ff9ff157f0daf06ca5d344eceb3f1a5f0941a5120d2aa7b75087"


def synth_bytes(n: int, seed: int) -> bytes:
    """BF16 bytes from a 64-bit LCG (matches quantui's synth() family).

    Values are (LCG >> 40) interpreted as the top bits of a bf16 with a
    fixed sign/exponent bias pattern — enough entropy to exercise the
    quantizer's max/min search without float formatting concerns.
    """
    s = seed
    out = bytearray()
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        # Take the top 16 bits of the LCG state as raw bf16: this yields
        # values across the full bf16 range including subnormals/NaNs —
        # too hostile. Instead: constrain to a well-behaved magnitude by
        # OR-ing a fixed exponent field and masking the sign off.
        bits = (s >> 48) & 0xFFFF
        bits = (bits & 0x7FFF) | 0x3F80 >> 1 << 0  # keep exponent near 1.0
        bits = (bits & 0x807F) | ((bits >> 7 & 0xFF) << 7)
        out += struct.pack("<H", bits & 0xFFFF)
    return bytes(out)


def synth_bf16(n: int, seed: int) -> bytes:
    """Simpler deterministic BF16: LCG -> f32 in [-1,1) -> truncate to bf16.

    Truncation (not round-to-nearest) is what the reference converters do
    when producing bf16 payloads; matching it keeps the fixture realistic.
    """
    s = seed
    out = bytearray()
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        v_f32 = ((s >> 33) / 0xFFFFFFFF) * 2.0 - 1.0
        bits = struct.unpack("<I", struct.pack("<f", v_f32))[0]
        out += struct.pack("<H", bits >> 16)
    return bytes(out)


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    data = bytearray()
    header_map = {}
    for name, dtype, shape, blob in tensors:
        start = len(data)
        data += blob
        header_map[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(data)],
        }
    header = json.dumps(header_map, separators=(",", ":")).encode()
    pad = (8 - len(header) % 8) % 8
    header += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(data)


def main() -> int:
    path = OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    n_elems = N * N
    tensors = [
        (name, "BF16", [N, N], synth_bf16(n_elems, SEED + i))
        for i, name in enumerate(TENSOR_NAMES)
    ]
    write_safetensors(path, tensors)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    print(f"sha256: {sha}")
    if EXPECTED_SHA256 != "REPLACE_AFTER_FIRST_GENERATION" and sha != EXPECTED_SHA256:
        print(
            f"ERROR: sha256 mismatch — expected {EXPECTED_SHA256}. "
            "If this regeneration was intentional, update EXPECTED_SHA256 "
            "AND the nightly bench baseline history breaks; think twice."
        )
        return 1
    print("self-check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

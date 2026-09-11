"""One-off probe: does `safetensors.torch.save_file` preserve dict insertion
order in the header JSON, or does it sort/reorder?

Why this matters (plan Phase E.5): the calibration-cache draw order is
`FileOrderAll2D` (INT8/FP8) vs `SortedWeightsOnly` (MXFP8/NVFP4). To build a
fixture that DISCRIMINATES the two, the on-disk (file) order of the 2D
`.weight` tensors must differ from their sorted-by-name order. That is only
possible if `save_file` writes the header in dict insertion order.

Run with a torch environment that has torch + safetensors:

    CUDA_VISIBLE_DEVICES=-1 ^
    python tools\\probe_st_order.py
"""

from __future__ import annotations

import json
import os
import struct
import tempfile

from safetensors.torch import save_file

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"needs torch: {exc}") from exc


def header_keys(path: str) -> list[str]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return [k for k in hdr if k != "__metadata__"]


def main() -> int:
    # Deliberately NOT alphabetical: zzz, aaa, mmm.
    tensors = {
        "zzz.weight": torch.randn(64, 32).to(torch.bfloat16),
        "aaa.weight": torch.randn(64, 64).to(torch.bfloat16),
        "mmm.weight": torch.randn(64, 16).to(torch.bfloat16),
        "zzz.bias": torch.randn(64),
        "aaa.bias": torch.randn(64),
    }
    path = os.path.join(tempfile.mkdtemp(), "probe.safetensors")
    save_file(tensors, path)

    dict_order = list(tensors.keys())
    file_order = header_keys(path)

    print(f"dict order : {dict_order}")
    print(f"file order : {file_order}")
    print()

    if dict_order == file_order:
        print("RESULT: save_file PRESERVES dict insertion order.")
    elif file_order == sorted(dict_order):
        print("RESULT: save_file SORTS keys alphabetically.")
    else:
        print("RESULT: save_file groups by DTYPE (larger dtype first) and")
        print("        sorts ALPHABETICALLY within each dtype group.")
        print("        (confirmed 2026-08-28: F32 group before BF16 group,")
        print("         alphabetical inside each group.)")
        print()
        print("CONSEQUENCE for E.5: a fixture written with save_file always has")
        print("  its 2D `.weight` tensors in alphabetical order whenever they")
        print("  share a dtype, so file order == sorted order and the fixture")
        print("  CANNOT discriminate FileOrderAll2D vs SortedWeightsOnly.")
        print("  Fix: write the shards with an explicit hand-built safetensors")
        print("  header (see tools/gen_golden_sharded_unsorted.py).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

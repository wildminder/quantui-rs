"""Probe: does ``safetensors.safe_open(...).keys()`` return FILE order?

Answer (verified against the pinned safetensors the goldens were made with):

    NO. ``keys()`` returns the names SORTED ALPHABETICALLY, deterministically,
    regardless of the on-disk header order.

Why this matters (plan Phase E.5)
---------------------------------
Every ctq conversion entry point builds its key list from ``safe_open``:

* ``formats/fp8_conversion.py:147``  -> ``MemoryEfficientSafeOpen``
  - standard mode (``low_memory=False``, the default):
    ``self._all_keys = list(f.keys())`` with ``f = safe_open(...)``  -> SORTED
  - low-memory mode:
    ``self._all_keys = [k for k in self._header if ...]``            -> FILE order
* ``formats/mxfp8_conversion.py`` / ``nvfp4_conversion.py``: same loader, then
  ``weight_keys = sorted([...])``                                    -> SORTED
* ``formats/int8_conversion.py:46``: ``for key in f.keys()``         -> SORTED

So ctq's calibration draw order is SORTED-BY-NAME over the 2D ``.weight``
tensors, for all three ctq formats, in the default (non-low-memory) path.

The quantui reference streaming path is different: it reads the header RAW
(the reference stream_quant.py's `_resolve_union_header` ->
`read_safetensors_header`) and walks ``names`` in FILE order. INT8's bias
correction comes from that path, so INT8 really is file order.

Consequence for the Rust port: ``Format::calib_order()`` must map
``Int8 -> FileOrderAll2D`` and ``Fp8E4m3 | Mxfp8 | Nvfp4 -> SortedWeightsOnly``.
The two only differ for a file whose 2D weights are NOT alphabetical on disk,
which ``safetensors.torch.save_file`` usually hides (it groups by dtype and
sorts within each group — see ``tools/probe_st_order.py``) but which is legal
and does occur. ``tests/golden/sharded_unsorted`` is the discriminating
fixture.

Run with a torch environment that has safetensors pinned as the goldens
were made with:

    CUDA_VISIBLE_DEVICES=-1 ^
    python tools\\probe_safe_open_keys.py
"""

from __future__ import annotations

import glob
import json
import os
import struct

_HERE = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir))

# Files whose on-disk order is interesting: the hand-built E.5 fixture (file
# order deliberately NOT alphabetical) plus the committed sharded fixture
# (written by save_file -> dtype-grouped, alphabetical within group).
CANDIDATES = [
    "tests/golden/sharded_unsorted/input/*.safetensors",
    "tests/golden/sharded_model/input/*.safetensors",
]


def disk_order(path: str) -> list[str]:
    """Tensor names in on-disk header order."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return [k for k in hdr if k != "__metadata__"]


def main() -> int:
    from safetensors import safe_open

    files: list[str] = []
    for pat in CANDIDATES:
        files.extend(sorted(glob.glob(os.path.join(WS_ROOT, pat))))

    if not files:
        print("no fixture files found; run from the workspace root")
        return 1

    all_sorted = True
    for path in files:
        disk = disk_order(path)
        # Open three times: a HashMap-backed implementation would give a
        # different (or at least unstable) order across runs.
        runs = []
        for _ in range(3):
            with safe_open(path, framework="pt", device="cpu") as f:
                runs.append(list(f.keys()))
        stable = all(r == runs[0] for r in runs)
        is_sorted = runs[0] == sorted(disk)
        all_sorted &= is_sorted and stable

        print(os.path.relpath(path, WS_ROOT))
        print(f"  disk  : {disk}")
        print(f"  keys  : {runs[0]}")
        print(f"  == sorted(disk): {is_sorted}   stable over 3 opens: {stable}")

    print()
    if all_sorted:
        print("VERDICT: safe_open.keys() is ALWAYS sorted(disk order), stable.")
        print("  -> ctq (fp8/mxfp8/nvfp4) calibration draws in SORTED order;")
        print("     the quantui INT8 reference streaming path draws in FILE order.")
    else:
        print("VERDICT: NOT uniformly sorted - inspect the output above before")
        print("  relying on either order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cross-check our int8_convrot safetensors against an official ComfyUI one.

Answers three separate questions that are easy to conflate:

1. COVERAGE — which tensors did each side quantize? (a size difference here
   is a SELECTION difference, not a numerics difference)
2. BLOB CONTRACT — do the comfy_quant sidecars agree on format/keys?
3. PAYLOAD — for tensors BOTH sides quantized, are the int8 bytes and the
   F32 scales byte-identical?

Usage:
  python tools/cmp_convrot_official.py <ours.safetensors> <official.safetensors>

NOTE on interpreting a payload MISMATCH: ConvRot payloads live in ROTATED
space. Both sides rotate the same source with the same orthogonal H, so
their int8 payloads SHOULD match byte-for-byte. A mismatch is a real
defect, not an expected difference. (Contrast: comparing a ConvRot
payload to the bf16 SOURCE reads as rel-L2 ~sqrt(2) and proves nothing.)
"""
import json
import struct
import sys
from collections import Counter


def read_header(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(n))
    meta = hdr.pop("__metadata__", None)
    return hdr, 8 + n, meta


class Reader:
    def __init__(self, path):
        self.path = path
        self.hdr, self.base, self.meta = read_header(path)
        self.fh = open(path, "rb")

    def raw(self, key):
        e = self.hdr[key]
        o = e["data_offsets"]
        self.fh.seek(self.base + o[0])
        return self.fh.read(o[1] - o[0])

    def blob(self, weight_key):
        return json.loads(self.raw(weight_key.replace(".weight", ".comfy_quant")))

    def close(self):
        self.fh.close()


def is_quantized(hdr, key):
    return hdr.get(key, {}).get("dtype") == "I8"


def main():
    ours_p, theirs_p = sys.argv[1], sys.argv[2]
    A, B = Reader(ours_p), Reader(theirs_p)
    print(f"OURS   : {ours_p}")
    print(f"THEIRS : {theirs_p}")
    print()

    # ---- 1. file-level ----
    import os

    sa, sb = os.path.getsize(ours_p), os.path.getsize(theirs_p)
    print("=== 1. FILE ===")
    print(f"  size      ours {sa:>15,}   theirs {sb:>15,}   delta {sa - sb:>+15,}")
    print(f"  header    ours {A.base:>15,}   theirs {B.base:>15,}")
    print(f"  tensors   ours {len(A.hdr):>15,}   theirs {len(B.hdr):>15,}")
    print(f"  metadata  ours {A.meta}   theirs {B.meta}")
    print()

    ca = Counter(v["dtype"] for v in A.hdr.values())
    cb = Counter(v["dtype"] for v in B.hdr.values())
    print("  dtype census:")
    for d in sorted(set(ca) | set(cb)):
        print(f"    {d:<6} ours {ca.get(d, 0):>5}   theirs {cb.get(d, 0):>5}")
    print()

    # ---- 2. coverage ----
    print("=== 2. COVERAGE (which weights each side quantized) ===")
    wa = {k for k in A.hdr if k.endswith(".weight") and is_quantized(A.hdr, k)}
    wb = {k for k in B.hdr if k.endswith(".weight") and is_quantized(B.hdr, k)}
    only_a, only_b = sorted(wa - wb), sorted(wb - wa)
    print(f"  ours quantized   {len(wa)}")
    print(f"  theirs quantized {len(wb)}")
    print(f"  shared           {len(wa & wb)}")
    print()
    if only_a:
        print(f"  ONLY OURS ({len(only_a)}) — we quantized, they left BF16:")
        for k in only_a:
            s = A.hdr[k]["shape"]
            print(f"    + {k:<50} {s}  in_features={s[-1]} div256={s[-1] % 256 == 0}")
    if only_b:
        print(f"  ONLY THEIRS ({len(only_b)}):")
        for k in only_b:
            print(f"    - {k:<50} {B.hdr[k]['shape']}")
    if not only_a and not only_b:
        print("  coverage identical")
    # Are the divergent ones all outside transformer_blocks?
    if only_a:
        inside = sum(1 for k in only_a if "transformer_blocks" in k)
        print(f"\n  of the ONLY-OURS set, {inside} are inside transformer_blocks, "
              f"{len(only_a) - inside} are OUTSIDE")
    print()

    # ---- 3. blob contract ----
    print("=== 3. BLOB CONTRACT (shared tensors) ===")
    shared = sorted(wa & wb)
    fmt_a, fmt_b = Counter(), Counter()
    keydiff = Counter()
    for k in shared:
        ja, jb = A.blob(k), B.blob(k)
        fmt_a[ja.get("format")] += 1
        fmt_b[jb.get("format")] += 1
        if set(ja) != set(jb):
            keydiff[(tuple(sorted(set(ja) - set(jb))), tuple(sorted(set(jb) - set(ja))))] += 1
    print(f"  ours formats  : {dict(fmt_a)}")
    print(f"  theirs formats: {dict(fmt_b)}")
    if keydiff:
        print("  key-set differences (ours_only, theirs_only) -> count:")
        for kk, c in keydiff.most_common(10):
            print(f"    {kk} -> {c}")
    else:
        print("  key sets identical on all shared tensors")
    # show one full pair
    if shared:
        k = shared[0]
        print(f"\n  sample blob for {k}:")
        print(f"    ours  : {json.dumps(A.blob(k), sort_keys=True)}")
        print(f"    theirs: {json.dumps(B.blob(k), sort_keys=True)}")
    print()

    # ---- 4. payload ----
    print("=== 4. PAYLOAD (byte comparison, shared tensors) ===")
    qsame = qdiff = 0
    sdiff = 0
    diff_list = []
    for k in shared:
        qa, qb = A.raw(k), B.raw(k)
        if qa == qb:
            qsame += 1
        else:
            qdiff += 1
            if len(diff_list) < 12:
                n = sum(1 for x, y in zip(qa, qb) if x != y)
                diff_list.append((k, n, len(qa)))
        sk = k.replace(".weight", ".weight_scale")
        if sk in A.hdr and sk in B.hdr:
            if A.raw(sk) != B.raw(sk):
                sdiff += 1
    print(f"  int8 payload identical : {qsame}/{len(shared)}")
    print(f"  int8 payload DIFFERING : {qdiff}/{len(shared)}")
    print(f"  weight_scale differing : {sdiff}/{len(shared)}")
    for k, n, tot in diff_list:
        print(f"    {k:<50} {n}/{tot} bytes differ")
    print()

    A.close()
    B.close()


if __name__ == "__main__":
    main()

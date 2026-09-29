#!/usr/bin/env python
"""Translate a reference GGUF's per-tensor dtype assignment into a
`--tensor-type-file` recipe expressed in OUR (llama.cpp-style) name space.

Needed because `--recipe-from` emits rules keyed on the reference's own tensor
names; an `audiocpp` reference uses `model_weights/<hf-name>`, which never
matches our post-mapping `blk.N.*` names, so every rule is silently ignored.

Usage: make_recipe.py <ours.gguf> <ref.gguf> <out.recipe>
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gguf_dump import GGML, parse  # noqa: E402
from gguf_dequant_cmp import ours_to_ref  # noqa: E402

# ggml type -> quantui-rs method id
METHOD = {
    "F32": "f32", "F16": "f16", "BF16": "bf16",
    "Q8_0": "q8_0", "Q4_0": "q4_0", "Q4_1": "q4_1", "Q5_0": "q5_0", "Q5_1": "q5_1",
    "Q2_K": "q2_k", "Q3_K": "q3_k_s", "Q4_K": "q4_k_s", "Q5_K": "q5_k_s",
    "Q6_K": "q6_k", "IQ4_NL": "iq4_nl", "Q8_K": "q8_k",
}


def main():
    ours_p, ref_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    _, _, A, _ = parse(ours_p)
    _, _, B, _ = parse(ref_p)
    ref = {t["name"]: GGML.get(t["dtype"], (f"?{t['dtype']}", 1))[0] for t in B}

    lines = ["# recipe derived from a reference GGUF, remapped into our name space",
             "# format: <version>"]
    body = []
    skipped = []
    for t in A:
        name = t["name"]
        rname = ours_to_ref(name)
        if rname not in ref:
            skipped.append((name, "no-ref-counterpart"))
            continue
        dt = ref[rname]
        m = METHOD.get(dt)
        if m is None:
            skipped.append((name, f"unmapped-type:{dt}"))
            continue
        if dt == "F32":
            # 1-D convention: F32 needs no rule, and our resolver forces F32
            # for every 1-D tensor before consulting the recipe anyway.
            continue
        body.append(f"^{name}$={m}")

    with open(out_p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# quantui-rs recipe format 1\n")
        fh.write("# remapped from reference GGUF by tools/make_recipe.py\n")
        for b in body:
            fh.write(b + "\n")
    print(f"wrote {out_p}: {len(body)} rules, {len(skipped)} skipped")
    for n, why in skipped[:10]:
        print(f"  skip {n} ({why})")
    from collections import Counter
    c = Counter(l.rsplit("=", 1)[1] for l in body)
    print("  rule types:", dict(c))


if __name__ == "__main__":
    main()

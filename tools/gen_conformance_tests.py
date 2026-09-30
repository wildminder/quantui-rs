#!/usr/bin/env python
"""Generate Rust conformance tests from Golden Ruler packs.

Source: gHashTag/t27 conformance/vectors/*.json (arXiv:2606.09686v3)
Output: crates/quant-core/tests/format_conformance_golden_ruler.rs

The generated test asserts on the INTEGER BIT PATTERN, never on decoded-value
closeness -- that is the paper's own stated conformance criterion.

It is emitted as a separate integration test (not inlined into dtype.rs) so it
cannot perturb the byte-parity source files.

Usage:
    python tools/gen_conformance_tests.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # tools/ -> repo root
# The packs are read from -- and embedded into the test via include_str! from --
# the TRACKED fixtures dir alongside the test that consumes them.
#
# An earlier revision kept them under the gitignored `docs/` tree, so those
# copies were never tracked. The generated test's include_str! therefore pointed
# at an untracked file: on a fresh clone the whole test BINARY failed to
# compile, not just one test. These are third-party test fixtures, not
# documentation, so they belong with the tests that depend on them.
PACKS = ROOT / "crates" / "quant-core" / "tests" / "fixtures" / "conformance"
OUT = ROOT / "crates" / "quant-core" / "tests" / "format_conformance_golden_ruler.rs"


def f32_lit(x) -> str:
    """Rust f32 literal that round-trips exactly.

    JSON has no NaN/Infinity literals, so the Golden Ruler packs serialize them
    as the STRINGS "NaN"/"Infinity"/"-Infinity" -- hence the string checks
    before the float ones.
    """
    if isinstance(x, str):
        return {
            "NaN": "f32::NAN",
            "Infinity": "f32::INFINITY",
            "-Infinity": "f32::NEG_INFINITY",
        }[x]
    xf = float(x)
    if xf != xf:  # NaN
        return "f32::NAN"
    if xf == float("inf"):
        return "f32::INFINITY"
    if xf == float("-inf"):
        return "f32::NEG_INFINITY"
    return repr(xf) + "f32"


def main() -> None:
    e4m3 = json.loads((PACKS / "e4m3_fp8_e4m3fn_conformance_v0.json").read_text(encoding="utf-8"))
    e2m1 = json.loads((PACKS / "e2m1_mxfp4_e2m1_conformance_v0.json").read_text(encoding="utf-8"))
    bf16 = json.loads((PACKS / "bf16_bf16_golden_conformance_v0.json").read_text(encoding="utf-8"))
    e5m2 = json.loads((PACKS / "e5m2_fp8_e5m2_conformance_v0.json").read_text(encoding="utf-8"))
    mxfp8 = json.loads((PACKS / "mxfp8_mxfp8_conformance_v0.json").read_text(encoding="utf-8"))

    L: list[str] = []
    A = L.append
    A("//! GENERATED FILE - do not edit by hand.")
    A("//! Source: Golden Ruler conformance packs, gHashTag/t27")
    A("//!        (arXiv:2606.09686v3, `conformance/vectors/*.json`)")
    A("//! Regenerate with: python tools/gen_conformance_tests.py")
    A("//!")
    A("//! Conformance is asserted on the INTEGER BIT PATTERN, never on decoded-value")
    A("//! closeness -- the source paper's stated criterion.")
    A("#![allow(clippy::approx_constant)]")
    A("")
    A("use quant_core::dtype::{")
    A("    f32_to_bf16_bits, f32_to_fp8_e4m3_bits, fp8_e4m3_bits_to_f32,")
    A("};")
    A("")

    # ---- E4M3 ----
    A("/// FP8 E4M3 (S1E4M3, bias 7, no inf, max finite 448.0).")
    A("/// Overflow policy: SatMax -> 0x7E. This is a *deliberate* choice; the")
    A("/// ml_dtypes/JAX convention (OvfNaN -> 0x7F) is equally legal under OCP MX v1.0.")
    A("/// See the comment at FP8_MAX in crates/quant-core/src/quant_fp8.rs.")
    A("#[test]")
    A("fn golden_ruler_fp8_e4m3_vectors() {")
    for v in e4m3["vectors"]:
        bits = int(v["fp8_bits_hex"], 16)
        inp = f32_lit(v["input_f64"])
        name = v["name"]
        A(f'    // {name} ({v["category"]})')
        A(f'    assert_eq!(f32_to_fp8_e4m3_bits({inp}), 0x{bits:02X}, "{name}");')
    A("}")
    A("")

    # ---- E4M3 OVERFLOW POLICY (SatMax) ----
    #
    # PROVENANCE -- these vectors are NOT from the conformance pack. Read this
    # before assuming the pack covers the overflow policy.
    #
    # Every input the packs assert is IN RANGE. Measured over the vendored
    # packs: the `e4m3_fp8_e4m3fn` pack's 14 vectors top out at exactly 448.0,
    # and the `mxfp8` pack's 254 shared codes also top out at exactly 448.0.
    # So both code-space tables are policy-INDEPENDENT by construction: they
    # would still pass under `OvfNaN`.
    #
    # That matters because `SatMax` is a byte-exactness policy, and the pack
    # whose header is cited as its enforcement cannot enforce it. Verified by
    # mutation: flipping `dtype.rs` to `OvfNaN` (0x7E -> 0x7F on the overflow
    # branch and on the rounding-carry) leaves every pack-derived test GREEN.
    #
    # These expectations come from the `float8_e4m3fn` SPEC (max finite 448.0,
    # no infinities, SatMax saturates to 0x7E/0xFE) -- the same values already
    # asserted in the `dtype.rs` unit tests. No new oracle is introduced.
    A("/// The E4M3 `SatMax` OVERFLOW POLICY -- the one thing the packs cannot check.")
    A("///")
    A("/// ⚠️ PROVENANCE: the inputs below are NOT from the Golden Ruler packs. They")
    A("/// come from the `float8_e4m3fn` specification (no infinities, max finite")
    A("/// 448.0, overflow saturates to the max-finite code 0x7E / 0xFE).")
    A("///")
    A("/// Every input the packs assert is IN RANGE -- the `e4m3_fp8_e4m3fn` pack's")
    A("/// 14 vectors and the `mxfp8` pack's 254 shared codes both top out at")
    A("/// exactly 448.0 -- so the pack-derived tests are policy-INDEPENDENT and")
    A("/// stay green under `OvfNaN`. This test is the actual enforcement of the")
    A("/// choice documented at `FP8_MAX` in `crates/quant-core/src/quant_fp8.rs`.")
    A("///")
    A("/// Adversarial case defended: `ml_dtypes`/JAX uses `OvfNaN` (overflow -> the")
    A("/// NaN code 0x7F/0xFF), which is equally legal under OCP MX v1.0. Switching")
    A("/// to it would silently change the emitted byte for every overflowing")
    A("/// weight, breaking byte-exact parity with torch -- while leaving every")
    A("/// pack-derived assertion green. These cases close that hole.")
    A("///")
    A("/// Grid arithmetic (bias 7, 3 mantissa bits): max finite `0x7E` = 1.75*2^8 =")
    A("/// 448.0; the next grid point would be 2.0*2^8 = 512, which needs an")
    A("/// exponent field of 16 and is therefore the reserved `0x7F`. The")
    A("/// round-to-nearest-even TIE between them sits at (448+512)/2 = 480.0, and")
    A("/// `0x7E` (126) is the EVEN neighbour, so the tie resolves DOWN to 448.0.")
    A("///")
    A("/// WHAT THIS TEST DOES AND DOES NOT PROVE -- read before \"strengthening\" it.")
    A("/// It proves the SatMax POLICY: every overflow lands on 0x7E, never 0x7F.")
    A("/// Flipping the policy in `dtype.rs` turns this test red, which is the point.")
    A("///")
    A("/// It deliberately does NOT try to pin WHICH internal branch handles a")
    A("/// value, because that is not observable in the output. `dtype.rs` has")
    A("/// three routes to 0x7E and they are mutually redundant:")
    A("/// (a) the EARLY guard `f_bits >= FP8_MAX` (480.0) -> 0x7E directly;")
    A("/// (b) the RNE path for |x| < 480.0, which can only reach 0x7E or lower")
    A("///     because the tie resolves down;")
    A("/// (c) `if r == 0x7F { r = 0x7E; }`, the carry-saturation at the end of")
    A("///     the RNE path -- defensive, and UNREACHABLE while guard (a) holds,")
    A("///     since no value below 480.0 can round above 448.0.")
    A("/// Verified by mutation: deleting (3), and weakening (1) from `>=` to `>`,")
    A("/// both leave this test fully GREEN. At exactly 480.0 the tie also yields")
    A("/// 0x7E, so the two routes are indistinguishable there too. That redundancy")
    A("/// is a robustness property, not a gap -- but it does mean no input value")
    A("/// can discriminate the branches. Do not add cases expecting them to.")
    A("///")
    A("/// The cases below are therefore boundary coverage on the OBSERVABLE")
    A("/// question (which code an input produces), chosen to pin the exact")
    A("/// representable edges rather than to route-control the implementation:")
    A("/// 448.0 -- max finite, in range")
    A("/// 464.0 -- below the 480.0 tie, so it rounds DOWN naturally")
    A("/// 480.0 -- the RNE tie AND the early-guard threshold, exactly")
    A("/// 481.0 -- first value strictly above the threshold")
    A("/// 511.0 -- largest f32 still below the next step (512)")
    A("///")
    A("/// NaN is asserted separately below and is NOT a policy choice --")
    A("/// both policies must agree that NaN encodes as the NaN code.")
    A("#[test]")
    A("fn golden_ruler_fp8_e4m3_overflow_policy_is_satmax() {")
    A("    // (input, expected code, why this value)")
    A("    let satmax: [(f32, u8, &str); 10] = [")
    A("        (448.0, 0x7E, \"max finite -- in range, the boundary itself\"),")
    A("        (464.0, 0x7E, \"below the 480.0 tie: rounds DOWN naturally\"),")
    A("        (480.0, 0x7E, \"the RNE tie and the guard threshold, exactly\"),")
    A("        (481.0, 0x7E, \"first value strictly above the threshold\"),")
    A("        (511.0, 0x7E, \"largest f32 below the next step (512)\"),")
    A("        (500.0, 0x7E, \"plain overflow\"),")
    A("        (1e30, 0x7E, \"far overflow\"),")
    A("        (f32::INFINITY, 0x7E, \"+inf saturates; e4m3fn has no inf\"),")
    A("        (-500.0, 0xFE, \"negative overflow\"),")
    A("        (f32::NEG_INFINITY, 0xFE, \"-inf saturates\"),")
    A("    ];")
    A("    for (input, want, why) in satmax {")
    A("        assert_eq!(")
    A("            f32_to_fp8_e4m3_bits(input),")
    A("            want,")
    A('            "SatMax: {input} must saturate to {want:#04x} ({why})"')
    A("        );")
    A("    }")
    A("")
    A("    // The DECODER side of the same policy: 0x7E/0xFE are the max-finite")
    A("    // values, and 0x7F/0xFF are NaN -- never silently finite.")
    A("    assert_eq!(fp8_e4m3_bits_to_f32(0x7E), 448.0, \"0x7E decodes to +448.0\");")
    A("    assert_eq!(fp8_e4m3_bits_to_f32(0xFE), -448.0, \"0xFE decodes to -448.0\");")
    A("    assert!(")
    A("        fp8_e4m3_bits_to_f32(0x7F).is_nan(),")
    A('        "0x7F must decode to NaN under e4m3fn"')
    A("    );")
    A("    assert!(")
    A("        fp8_e4m3_bits_to_f32(0xFF).is_nan(),")
    A('        "0xFF must decode to NaN under e4m3fn"')
    A("    );")
    A("")
    A("    // NaN INPUT is not a policy choice: both SatMax and OvfNaN must encode")
    A("    // NaN as the NaN code. Asserted so this test cannot be satisfied by a")
    A("    // blanket \"everything becomes 0x7F\" rule.")
    A("    assert_eq!(f32_to_fp8_e4m3_bits(f32::NAN), 0x7F, \"NaN -> 0x7F\");")
    A("    assert_eq!(f32_to_fp8_e4m3_bits(-f32::NAN), 0xFF, \"-NaN -> 0xFF\");")
    A("}")
    A("")

    # ---- E2M1 ----
    A("/// MXFP4 element format E2M1 (1s2e1m, bias 1, 16 codes, no inf/NaN).")
    A("///")
    A("/// Exercised through the real `quantize_nvfp4_weight` path rather than by")
    A("/// reaching into the private `E2M1_LUT`.")
    A("///")
    A("/// Construction: the test value goes in element 0 and a 6.0 anchor in")
    A("/// element 1. With block amax = 6.0 the reference arithmetic gives")
    A("/// per_tensor_scale = 6.0/2688, block_scale = 6.0/6.0 = 1.0,")
    A("/// scaled = 448.0 -> E4M3 0x7E -> total_scale = exactly 1.0. So the element")
    A("/// code for element 0 is E2M1(value / 1.0) = E2M1(value), which is what the")
    A("/// pack tabulates.")
    A("///")
    A("/// The 6.0 anchor is REQUIRED. An all-zero block drives")
    A("/// per_tensor_scale = 0, hence total_scale = 0 * E4M3(NaN) = NaN, so the")
    A("/// reference's `zero_scale_mask = (total_scale == 0)` is False and every")
    A("/// element encodes as NaN. That is faithful to comfy-kitchen")
    A("/// `quantize_nvfp4` (quantization.py:150) and is covered separately by")
    A("/// `all_zero_block_is_nan_like_the_reference`.")
    A("#[test]")
    A("fn golden_ruler_e2m1_exhaustive_grid() {")
    A("    // (input value, expected element code) from the pack's 16 vectors.")
    A("    let grid: [(f32, u8); 16] = [")
    for v in e2m1["vectors"]:
        bits = int(v["mxfp4_bits_hex"], 16)
        A(f'        ({f32_lit(v["input_f64"])}, 0x{bits:02X}),  // {v["name"]}')
    A("    ];")
    A("")
    A("    for (value, want_code) in grid {")
    A("        let mut block = [0.0f32; 16];")
    A("        block[0] = value;")
    A("        block[1] = 6.0; // anchor: fixes total_scale to exactly 1.0")
    A("        let q = quant_core::quant_nvfp4::quantize_nvfp4_weight(&block, 1, 16);")
    A("        // pack_uint4(hi_first=true): EVEN element index -> HIGH nibble.")
    A("        let got = (q.qdata[0] >> 4) & 0x0F;")
    A("        assert_eq!(got, want_code, \"value {value} (want {want_code:#04x})\");")
    A("    }")
    A("}")
    A("")
    A("/// An all-zero NVFP4 tensor produces NaN element codes, NOT zero codes.")
    A("///")
    A("/// This is faithful to the reference, not a bug: with per_tensor_scale = 0,")
    A("/// `scaled_block_scales = 0/0 = NaN`, E4M3(NaN) is 0x7F or 0xFF (see")
    A("/// below), and `total_scale = 0 * NaN = NaN`. The reference guard")
    A("/// `zero_scale_mask = (total_scale == 0)` is therefore False, so")
    A("/// `data_scaled = 0.0 / NaN = NaN` and clamps do not rescue it.")
    A("/// comfy-kitchen quantization.py:149-154.")
    A("///")
    A("/// Recorded explicitly so a future change to the zero-block guard is a")
    A("/// deliberate, reviewed decision rather than an accidental parity break.")
    A("///")
    A("/// # Why the scale byte is masked, not compared to 0xFF")
    A("///")
    A("/// E4M3 has no infinity; its only NaN is exp=1111 + mantissa=111, which")
    A("/// encodes to byte `0x7F` with the sign bit CLEAR or `0xFF` with it SET.")
    A("/// Both decode back to NaN (verified through torch 2.13). Which one you")
    A("/// get is decided by the SIGN OF THE SOURCE NaN, and IEEE 754 does not")
    A("/// fix that sign for a NaN *generated* by an operation:")
    A("///")
    A("/// | platform | `0.0f32 / 0.0f32` | E4M3 |")
    A("/// |---|---|---|")
    A("/// | x86_64 (SSE) | `0xffc00000` (negative quiet NaN) | `0xFF` |")
    A("/// | aarch64 (macOS CI) | `0x7fc00000` (positive quiet NaN) | `0x7F` |")
    A("///")
    A("/// torch reproduces exactly this: `torch.tensor(0.0)/torch.tensor(0.0)`")
    A("/// is `0xffc00000` on x86_64 and casts to `0xFF`, so the kernel is doing")
    A("/// the reference thing on the platform the goldens were generated on.")
    A("/// Hard-coding `0xFF` would therefore encode an x86 artefact as if it")
    A("/// were a reference invariant, and would fail on every aarch64 runner")
    A("/// for a difference that is invisible after decoding.")
    A("///")
    A("/// The invariant worth pinning is the one the reference actually")
    A("/// guarantees: the byte is an E4M3 NaN, and the element codes are the")
    A("/// NaN-derived magnitude 4, not the real zero code. `0xFF` / `0xCC` are")
    A("/// asserted on x86_64 only, so a genuine polarity flip on that platform")
    A("/// is still caught while aarch64 stops reporting an invisible one.")
    A("#[test]")
    A("fn all_zero_block_is_nan_like_the_reference() {")
    A("    let block = [0.0f32; 16];")
    A("    let q = quant_core::quant_nvfp4::quantize_nvfp4_weight(&block, 1, 16);")
    A("    assert_eq!(q.per_tensor_scale, 0.0);")
    A("    // E4M3 NaN: exp=1111, mantissa=111. The sign bit is NOT pinned --")
    A("    // a NaN synthesised by 0.0/0.0 carries whatever sign the platform's")
    A("    // hardware produces, and both 0x7F and 0xFF decode to NaN.")
    A("    assert_eq!(")
    A("        q.scale[0] & 0x7F,")
    A("        0x7F,")
    A("        \"block scale must be an E4M3 NaN (0x7F or 0xFF), got {:#04x}\",")
    A("        q.scale[0]")
    A("    );")
    A("    assert!(")
    A("        fp8_e4m3_bits_to_f32(q.scale[0]).is_nan(),")
    A("        \"block scale must decode back to NaN, got {}\",")
    A("        fp8_e4m3_bits_to_f32(q.scale[0])")
    A("    );")
    A("    #[cfg(target_arch = \"x86_64\")]")
    A("    assert_eq!(")
    A("        q.scale[0],")
    A("        0xFF,")
    A("        \"on x86_64 the NaN is negative (0xffc00000), so E4M3 must be 0xFF\"")
    A("    );")
    A("    // Every packed byte carries the NaN-derived code, not 0x00.")
    A("    //")
    A("    // The E2M1 code is ALSO NaN-sign-dependent: `f32_to_e2m1_bits` ends")
    A("    // with `code | sign_lp`, and `encode_element` divides `0.0 / total`")
    A("    // where `total = 0 * NaN`, so the element NaN inherits the sign of")
    A("    // the scale NaN. Measured on the same two toolchains:")
    A("    //")
    A("    // | platform | scale byte | E2M1 code | packed byte |")
    A("    // |---|---|---|---|")
    A("    // | x86_64 | `0xFF` | `0x0C` (sign set) | `0xCC` |")
    A("    // | aarch64 | `0x7F` | `0x04` (sign clear) | `0x44` |")
    A("    //")
    A("    // Magnitude code 4 (= 2.0) on both -- only the SIGN nibble differs,")
    A("    // and it is the same root cause as the scale byte above, not a")
    A("    // second one. So the invariant is `code & 0x07 == 4` (the NaN")
    A("    // magnitude, never 0x00 which would be a real zero code), with the")
    A("    // exact bytes pinned on x86_64 only.")
    A("    for (i, b) in q.qdata.iter().enumerate() {")
    A("        let hi = b >> 4;")
    A("        let lo = b & 0x0F;")
    A("        assert_eq!(")
    A("            hi & 0x07,")
    A("            4,")
    A("            \"byte {i} high nibble {hi:#04x} lost the NaN magnitude code\"")
    A("        );")
    A("        assert_eq!(")
    A("            lo & 0x07,")
    A("            4,")
    A("            \"byte {i} low nibble {lo:#04x} lost the NaN magnitude code\"")
    A("        );")
    A("    }")
    A("    #[cfg(target_arch = \"x86_64\")]")
    A("    assert!(")
    A("        q.qdata.iter().all(|b| *b == 0xCC),")
    A("        \"on x86_64 the NaN is negative, so every packed byte is 0xCC, got {:?}\",")
    A("        &q.qdata[..4.min(q.qdata.len())]")
    A("    );")
    A("}")
    A("")

    # ---- BF16 ----
    bv = bf16.get("vectors") or bf16.get("rows") or []
    A("/// BF16 (S1E8M7, bias 127, round-to-nearest-even).")
    A("#[test]")
    A("fn golden_ruler_bf16_vectors() {")
    for v in bv:
        key = "bf16_bits_hex" if "bf16_bits_hex" in v else "bits_hex"
        if key not in v:
            continue
        bits = int(v[key], 16)
        A(f'    // {v["name"]} ({v["category"]})')
        A(f'    assert_eq!(f32_to_bf16_bits({f32_lit(v["input_f64"])}), 0x{bits:04X}, "{v["name"]}");')
    A("}")
    A("")

    # ---- E5M2 ----
    ev = e5m2.get("vectors") or e5m2.get("rows") or []
    if ev:
        key = "fp8_e5m2_bits_hex" if "fp8_e5m2_bits_hex" in ev[0] else list(ev[0])[3]
        A("/// FP8 E5M2 (S1E5M2, bias 15, inf+NaN retained, max finite 57344.0).")
        A("/// We do not currently implement an E5M2 encoder, so this is recorded as")
        A("/// a reference table only -- deliberately not asserted for conformance.")
        A("///")
        A("/// It is NOT a no-op: the pack is embedded with `include_str!` and its")
        A("/// vector count is checked, so this test fails if the vendored pack is")
        A("/// deleted, truncated, or replaced with a different revision. That is the")
        A("/// part worth enforcing -- the E5M2 *values* stay unasserted until we")
        A("/// have an encoder to assert them against.")
        A("#[test]")
        A("fn golden_ruler_e5m2_reference_table_is_present() {")
        A("    const PACK: &str = include_str!(")
        A('        "fixtures/conformance/e5m2_fp8_e5m2_conformance_v0.json"')
        A("    );")
        A(f"    // {len(ev)} reference vectors retained in tests/fixtures/conformance/.")
        A("    // Unused by the crate today: no E5M2 encoder exists.")
        # Count a key that occurs EXACTLY once per vector. The obvious
        # candidate (`fp8_bits_hex`) does not qualify: the pack's
        # `anchor_check` block also carries one, so a bare key-name count
        # over-reports by 1. `name` is per-vector only.
        #
        # Verified here rather than assumed: the marker must appear exactly
        # `len(ev)` times in the raw text, or the emitted assertion would be
        # wrong for a reason nobody would notice until a pack revision.
        raw = (PACKS / "e5m2_fp8_e5m2_conformance_v0.json").read_text(encoding="utf-8")
        candidates = [k for k in ev[0] if k != "input_f64_hex"]
        marker = next(
            (k for k in candidates if raw.count(f'"{k}":') == len(ev)),
            None,
        )
        assert marker is not None, (
            "no e5m2 per-vector key occurs exactly len(vectors) times; "
            "the emitted count assertion would be unreliable"
        )
        A("    let vectors = PACK.matches(")
        A(f'        "\\"{marker}\\":\"')
        A("    )")
        A("    .count();")
        A(f"    assert_eq!(vectors, {len(ev)}, \"E5M2 pack vector count changed\");")
        A("}")
        A("")

    # ---- MXFP8 (256-vector exhaustive E4M3 code space) ----
    #
    # VARIANT NOTE -- read before "fixing" the two exclusions below.
    #
    # This pack is the OCP MX *element* format S1E4M3, bias 7, FINITE-ONLY
    # (the E8M0 block scale carries the range). Its max finite is 480.0 and it
    # has NO NaN encoding: codes 0x7F / 0xFF decode to +480.0 / -480.0.
    #
    # We implement torch `float8_e4m3fn`, which is a DIFFERENT variant: max
    # finite 448.0 (0x7E) with 0x7F / 0xFF RESERVED FOR NaN. Both are legal
    # under OCP MX v1.0; ours is the one torch / comfy-kitchen / llama.cpp
    # use, so ours is correct and the pack is simply a different (also-legal)
    # choice for the two overflow codes.
    #
    # Verified by enumerating the full 256-code space through the real
    # `f32_to_fp8_e4m3_bits` / `fp8_e4m3_bits_to_f32`: exactly TWO codes
    # disagree, 0x7F and 0xFF -- precisely the two `e4m3fn` reserves for NaN.
    # The other 254 agree bit-for-bit in BOTH directions.
    #
    # Therefore: assert the 254 shared codes, and pin the 2 variant codes as a
    # DOCUMENTED, ASSERTED divergence. Do NOT "fix" those two by editing
    # dtype.rs -- that would break byte-exact parity with torch, which is this
    # crate's governing contract. The skip list below IS the designed answer.
    VARIANT_CODES = (0x7F, 0xFF)
    mv = mxfp8["vectors"]
    shared = [v for v in mv if int(v["mxfp8_bits_int"]) not in VARIANT_CODES]
    variant = [v for v in mv if int(v["mxfp8_bits_int"]) in VARIANT_CODES]
    assert len(shared) + len(variant) == len(mv) == 256, "mxfp8 pack shape changed"

    A("/// MXFP8 element format S1E4M3 -- exhaustive 256-code enumeration.")
    A("///")
    A("/// Upgrades E4M3 coverage from the 14 hand-picked `e4m3fn` spot checks to")
    A("/// the full code space of the OCP MX variant: 254 of the 256 codes are")
    A("/// asserted here in BOTH directions (encode and decode). The previous 14")
    A("/// vectors never touched the 14 subnormal codes, so a decoder bug in an")
    A("/// unsampled subnormal -- or a rounding bug at an exponent boundary --")
    A("/// could not hide. Enumeration removes that class of blind spot.")
    A("///")
    A("/// The 2 absent codes are 0x7F and 0xFF, which is a DIFFERENT E4M3")
    A("/// VARIANT, not a bug: this pack is the finite-only OCP MX element type")
    A("/// (max 480.0, 0x7F -> +480.0), we implement torch `float8_e4m3fn`")
    A("/// (max 448.0, 0x7F -> NaN). Pinned by the sibling test below.")
    A("///")
    A("/// Conformance is on the INTEGER BIT PATTERN (the paper's criterion), so")
    A("/// this is exact, not a closeness check.")
    A("#[test]")
    A(f"fn golden_ruler_mxfp8_shared_codes_exhaustive() {{")
    A(f"    // {len(shared)} shared codes, both directions, from the pack's")
    A("    // `input_f64` and `mxfp8_bits_int`.")
    A("    //")
    A("    // The table is f32, not f64: every packed value is an E4M3 code, so it")
    A("    // has a 3-bit mantissa and is EXACTLY representable in f32. Typing the")
    A("    // table as f64 and narrowing with `as f32` would introduce a rounding")
    A("    // step that could mask an off-by-one-ulp encoder bug -- the very class")
    A("    // of defect this enumeration exists to catch.")
    A(f"    let table: [(f32, u8); {len(shared)}] = [")
    for v in shared:
        code = int(v["mxfp8_bits_int"])
        A(
            f'        ({f32_lit(v["input_f64"])}, 0x{code:02X}),'
            f'  // {v["name"]} ({v["category"]})'
        )
    A("    ];")
    A("")
    A("    for (xf, code) in table {")
    A("        assert_eq!(")
    A("            f32_to_fp8_e4m3_bits(xf),")
    A("            code,")
    A('            "encode: f32_to_fp8_e4m3_bits({xf}) != 0x{code:02X}"')
    A("        );")
    A("        assert_eq!(")
    A("            fp8_e4m3_bits_to_f32(code),")
    A("            xf,")
    A('            "decode: fp8_e4m3_bits_to_f32(0x{code:02X}) != {xf}"')
    A("        );")
    A("    }")
    A("}")
    A("")

    A("/// The MXFP8 pack's 2 variant-specific codes are a DOCUMENTED divergence,")
    A("/// pinned here so it can never drift into an accident.")
    A("///")
    A("/// This is the guard for the guard. The sibling test skips 0x7F / 0xFF")
    A("/// because our E4M3 variant reserves them for NaN. Someone 'fixing' those")
    A("/// two failures by editing `dtype.rs` would break byte-exact parity with")
    A("/// torch -- the crate's governing contract. This test fails FIRST and says")
    A("/// why.")
    A("///")
    A("/// Asserts: (a) the exclusion set is EXACTLY {0x7F, 0xFF} -- a count plus")
    A("/// explicit membership checks, so a future pack change cannot silently")
    A("/// shrink coverage; (b) our decoder returns NaN for both; (c) the pack's")
    A("/// own values for those codes are +/-480.0, restated as literals so the")
    A("/// divergence is visible in this file rather than only in the JSON.")
    A("#[test]")
    A("fn golden_ruler_mxfp8_variant_codes_are_the_documented_exception() {")
    A("    // (a) The exclusion set is exactly the two NaN-reserved codes, and")
    A("    // every other code in 0..=255 IS asserted by the sibling test. So a")
    A("    // future pack change cannot silently shrink coverage.")
    A("    const EXCLUDED: [u8; 2] = [0x7F, 0xFF];")
    A("    // The codes the sibling test asserts, straight from the pack.")
    A("    let shared_codes: [u8; %d] = [" % len(shared))
    for i in range(0, len(shared), 16):
        chunk = shared[i : i + 16]
        A(
            "        "
            + ", ".join(f"0x{int(v['mxfp8_bits_int']):02X}" for v in chunk)
            + ","
        )
    A("    ];")
    A(f'    assert_eq!(shared_codes.len(), {len(shared)});')
    A('    assert_eq!(EXCLUDED.len(), 2, "exclusion set must stay {{0x7F, 0xFF}}");')
    A("    // Shared + excluded must tile the whole 256-code space exactly once.")
    A("    let mut covered: Vec<u8> = shared_codes.to_vec();")
    A("    covered.extend_from_slice(&EXCLUDED);")
    A("    covered.sort_unstable();")
    A("    covered.dedup();")
    A("    assert_eq!(")
    A("        covered,")
    A("        (0..=255u8).collect::<Vec<u8>>(),")
    A('        "shared + excluded must be exactly the 256-code E4M3 space"')
    A("    );")
    A("    // And the excluded codes are genuinely absent from the shared table.")
    A("    for c in EXCLUDED {")
    A("        assert!(")
    A("            !shared_codes.contains(&c),")
    A('            "0x{c:02X} must be EXCLUDED from the shared-code table"')
    A("        );")
    A("    }")
    A("")
    A("    // (b) Our decoder reserves both for NaN (torch float8_e4m3fn).")
    A("    assert!(")
    A("        fp8_e4m3_bits_to_f32(0x7F).is_nan(),")
    A('        "e4m3fn reserves 0x7F for NaN"')
    A("    );")
    A("    assert!(")
    A("        fp8_e4m3_bits_to_f32(0xFF).is_nan(),")
    A('        "e4m3fn reserves 0xFF for NaN"')
    A("    );")
    A("")
    A("    // (c) The pack's own values for those codes -- the divergence,")
    A("    // restated as literals so it is readable without the JSON.")
    A("    let pack_values: [(u8, f64); 2] = [")
    for v in variant:
        A(
            f'        (0x{int(v["mxfp8_bits_int"]):02X},'
            f' {f32_lit(v["input_f64"])} as f64),'
            f'  // {v["name"]}'
        )
    A("    ];")
    A("    for (code, pack_value) in pack_values {")
    A("        let ours = fp8_e4m3_bits_to_f32(code);")
    A("        assert!(")
    A("            ours.is_nan(),")
    A('            "e4m3fn reserves 0x{code:02X} for NaN, got {ours}"')
    A("        );")
    A("        assert_ne!(")
    A("            ours, pack_value as f32,")
    A('            "code 0x{code:02X} must DIVERGE from the pack value"')
    A("        );")
    A("    }")
    A("}")
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")

    # rustfmt the emitted file so `cargo fmt --check` stays clean. The
    # generator's one-line-per-assert style does not match rustfmt's wrapping,
    # and hand-aligning 254 table rows in the generator would be worse. The
    # emitted content is unchanged — only whitespace differs.
    #
    # This is a LOCAL rustfmt invocation, not a `cargo fmt` on the workspace:
    # it must not reformat anything except the single generated artifact.
    # Failure is non-fatal (a missing rustfmt should not block generation).
    try:
        subprocess.run(
            ["rustfmt", "--edition", "2021", str(OUT)],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        print(f"  warning: rustfmt pass skipped ({exc})")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  e4m3 vectors: {len(e4m3['vectors'])}")
    print(f"  e2m1 vectors: {len(e2m1['vectors'])}")
    print(f"  bf16 vectors: {len(bv)}")
    print(f"  e5m2 vectors: {len(ev)} (reference only)")
    print(
        f"  mxfp8 vectors: {len(mv)}"
        f" ({len(shared)} shared + {len(variant)} variant)"
    )


if __name__ == "__main__":
    main()

# PHASE 2 NOTES — dtype casts & INT8 quantization kernels

**Date:** 2026-08-25
**Status:** COMPLETE — all gates green (build / clippy -D warnings / fmt --check / test: 29 passed = 14 lib + 10 phase1 + 5 phase2-parity)
**Note:** Implemented directly by team-lead (engineer runs remain unreliable due to provider 503s).

## What was built

| File | Contents |
|---|---|
| `crates/quant-core/src/dtype.rs` (extended) | bf16/f16 ↔ f32 bit-exact casts via `half` crate, round-to-nearest-even; NaN/Inf/subnormal propagation |
| `crates/quant-core/src/quant.rs` | NEW — symmetric INT8 kernels: Tensor/Row/Block scaling modes; all-zeros early return; `should_skip_shape` port; `ExcludePattern` regex helper (invalid-regex → no-exclude fallback); rayon parallel map; dequantize for later bias correction |
| `crates/quant-core/tests/phase2_quant_parity.rs` | NEW — golden byte-parity tests against all 3 Python-produced fixtures |
| `crates/quant-core/Cargo.toml` | Added half, regex, rayon |

## Parity evidence (the critical part)

All 5 parity tests pass on first run:

- **int8_blockwise_matches_golden_linear_basic_bf16**: [256,128] bf16 weight → int8 payload byte-identical to torch output; scale `[2,1]` F32 **bit-for-bit** equal; input_scale scalar 1.0; comfy_quant blob exact string match.
- **int8_blockwise_matches_golden_odd_shapes**: f32 [384,256] weight → payload + scale `[3,2]` byte-identical.
- **int8_blockwise_scalar_scale_matches_golden_conv_net**: f16 [128,128] → payload identical; scale squeezed from [1,1] to scalar `[]` matches.
- **skipped_weight_cast_to_bf16_matches_golden**: skip-heur path ([128,64], cols<bs) passes bytes through verbatim, header dtype BF16.
- **odd_shape_skipped_weight_stays_bf16**: [130,130] non-divisible → passthrough.

## Format discoveries during the port

1. **Three different scale formulas by mode** — critical asymmetry:
   - Block (`_weight_quantize_pytorch`): `scale = max(amax/127, 1e-8)` (dequant scale, epsilon as FLOOR via torch.maximum)
   - Tensor (`_convert_int8_tensorwise`): `dequant = w_max.clamp_min(1e-12)/127`, quant = reciprocal
   - Row (`TensorWiseINT8Layout.quantize` auto): `quant = 127/rowmax.clamp_min(1e-12)`, dequant = reciprocal
   The tensor and row paths use clamp_min(1e-12); only block uses the 1e-8 floor. Rust keeps each formula separate so division order is bit-identical.
2. **Rounding**: torch CPU `.round()` = round-half-to-even → `f32::round_ties_even`. Verified by hand-computed tie cases (e.g. -63.5 → -64).
3. **comfy_quant blob JSON uses DEFAULT separators** (`", "` / `": "`) — not compact. Key order: format, orig_dtype, group_size. Verified against golden blobs.
4. **Scale squeeze**: a 1-element scale ([1] or [1,1]) is emitted with shape `[]` (scalar) per `normalize_tensorwise_scales`; handled at emit time in Phase 5, kernel returns shape [1].
5. **All-zeros early return** in ctq `convert()`: qdata zeros, scale ONES (not amax-derived) with layout-appropriate grid shape.
6. **Test-fixture gotchas fixed en route**: f16(1234.5)=1234.0 bits 25810 (torch-verified), f16 subnormal rounding of 1.5·2⁻²⁵ = 0x0001, bf16(1234.5)=1232.0, bf16(1e-30)≈9.984021e-31 stays representable.

## Deviations from plan

None. Plan step 2.4's benchmark smoke deferred to Phase 5 integration (parallel-vs-sequential byte equality is already tested).

## Next

Phase 4 (schema blob encoder/parser) is partially covered already (blob string asserted in parity test); formalize `comfy_schema.rs` next, then Phase 5 orchestrator.

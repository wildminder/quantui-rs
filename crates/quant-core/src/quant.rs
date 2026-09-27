//! Symmetric INT8 quantization kernels — byte-parity port of convert_to_quant.
//!
//! Sources ported:
//!
//! - `convert_to_quant/comfy/quant_ops.py::BlockWiseINT8Layout._weight_quantize_pytorch`
//!   (block path: scale = max(amax/127, 1e-8))
//! - `convert_to_quant/comfy/quant_ops.py::TensorWiseINT8Layout.quantize` +
//!   `learned_rounding.py::_convert_int8_tensorwise`
//!
//!   Tensor path: dequant = `w_max.clamp_min(1e-12)/127`. Row path: quant_scale =
//!   `127/rowmax.clamp_min(1e-12)`, dequant = `1/quant_scale`.
//! - `learned_rounding.py::convert` all-zeros early return (scale = ones)
//! - `stream_quant.py::_should_skip_layer_for_performance` / `_should_skip_shape`
//!
//! Rounding is round-half-to-even (`f32::round_ties_even`) matching torch's
//! `.round()` on CPU. All math runs in f32 exactly like COMPUTE_DTYPE=f32.

use rayon::prelude::*;

/// Scaling mode for symmetric INT8, mirroring `QuantConfig.scaling_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMode {
    Tensor,
    Row,
    Block,
}

/// Output of one INT8 quantization: quantized data + dequant scales.
#[derive(Debug, Clone)]
pub struct Int8QuantResult {
    /// Quantized int8 data in row-major order, same element count as input.
    pub qdata: Vec<i8>,
    /// Scale shape depends on mode:
    /// - Tensor: `[1]` (or `[]` after scalar squeeze at emit time)
    /// - Row: `[M, 1]`
    /// - Block: `[M/bs, N/bs]`
    pub scale: Vec<f32>,
    pub scale_shape: Vec<u64>,
}

/// Quantize a 2D weight tensor (row-major f32) with symmetric INT8.
///
/// `w` must contain exactly `m * n` elements. Mirrors
/// `LearnedRoundingConverter.convert()` simple-mode routing:
/// - all-zeros tensor → qdata zeros, scale ones (per-layout shape)
/// - tensor → global amax clamp_min(1e-12)/127
/// - row → per-row amax, quant_scale=127/clamp_min(1e-12), dequant=recip
/// - block → per-block amax/127 then max(scale, 1e-8)
pub fn quantize_int8_weight(
    w: &[f32],
    m: usize,
    n: usize,
    mode: ScalingMode,
    block_size: usize,
) -> Int8QuantResult {
    debug_assert_eq!(w.len(), m * n);

    if w.iter().all(|&v| v == 0.0) {
        let (scale_shape, scale_len) = match mode {
            ScalingMode::Tensor => (vec![1], 1),
            ScalingMode::Row => (vec![m as u64, 1], m),
            ScalingMode::Block => {
                let bm = m / block_size;
                let bn = n / block_size;
                (vec![bm as u64, bn as u64], bm * bn)
            }
        };
        return Int8QuantResult {
            qdata: vec![0i8; m * n],
            scale: vec![1.0; scale_len],
            scale_shape,
        };
    }

    match mode {
        ScalingMode::Tensor => {
            // learned_rounding.py L906-912: w_max.clamp_min(1e-12)/127 as DEQUANT scale,
            // then TensorWiseINT8Layout.quantize(scale=Some): quant_scale = 1/scale.
            let w_max = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let dequant_scale = (w_max.max(1e-12)) / 127.0;
            let quant_scale = 1.0 / dequant_scale;
            let qdata = quantize_scaled(w, quant_scale);
            Int8QuantResult {
                qdata,
                scale: vec![dequant_scale],
                scale_shape: vec![1],
            }
        }
        ScalingMode::Row => {
            // TensorWiseINT8Layout auto path L908-916: row_max keepdim,
            // quant_scale = 127/row_max.clamp_min(1e-12), dequant = 1/quant_scale.
            //
            // FLOAT SEMANTICS (Phase 7.2, probe tools/probe_convrot_scale_diff.py
            // + tools/tmp/ulp_test3.rs): BOTH divisions are torch
            // `cpu-scalar / tensor` ops, which torch does NOT compute as IEEE
            // division — div_true_kernel takes the fast path
            // `scalar * reciprocal(tensor)` (two roundings):
            //   quant_scale = 127.0 * (1.0 / clamped_row_max)
            //   dequant     = 1.0  * (1.0 / quant_scale)
            // On row 3 of the conv_net fixture (row_max = 0.11627197265625):
            // IEEE `127.0/max` = 0x44888889 but torch emits 0x44888888 —
            // 1 ULP off correct rounding — and the golden dequant scale
            // 0x3a700001 follows. Plain IEEE division diverged on 29/128
            // rows (and cascaded into qdata ties: 3/16384). The FP8 module
            // documents the same quirk (`quant_fp8.rs` header); the INT8
            // row path never hit it before because no golden locked row
            // mode until the ConvRot fixtures.
            let mut scale = Vec::with_capacity(m);
            let mut qdata = Vec::with_capacity(m * n);
            let rows: Vec<&[f32]> = w.chunks_exact(n).collect();
            let out: Vec<(f32, Vec<i8>)> = rows
                .par_iter()
                .map(|row| {
                    let row_max = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                    let clamped = row_max.max(1e-12);
                    // torch: 127.0 / clamped  →  127.0 * reciprocal(clamped)
                    let quant_scale = 127.0f32 * (1.0f32 / clamped);
                    // torch: 1.0 / quant_scale  →  1.0 * reciprocal(quant_scale)
                    let dequant_scale = 1.0f32 / quant_scale;
                    (dequant_scale, quantize_scaled(row, quant_scale))
                })
                .collect();
            for (s, q) in out {
                scale.push(s);
                qdata.extend(q);
            }
            Int8QuantResult {
                qdata,
                scale,
                scale_shape: vec![m as u64, 1],
            }
        }
        ScalingMode::Block => {
            assert!(
                m.is_multiple_of(block_size) && n.is_multiple_of(block_size),
                "INT8 block-wise requires dims divisible by block_size={block_size}, got ({m}, {n})"
            );
            let bm = m / block_size;
            let bn = n / block_size;
            // Per-block amax over the (bs, bs) tile; scale = max(amax/127, 1e-8).
            // Parallelized over block-rows (Phase 11.1 perf). This is bit-safe:
            // each block's amax is a `max` over its own fixed element set
            // (order-independent), and every block writes a distinct `scale` slot,
            // so the result is identical to the sequential loop.
            let mut scale = vec![0.0f32; bm * bn];
            scale
                .par_chunks_mut(bn)
                .enumerate()
                .for_each(|(bi, scale_row)| {
                    for bj in 0..bn {
                        let mut amax = 0.0f32;
                        for i in bi * block_size..(bi + 1) * block_size {
                            let row = &w[i * n..(i + 1) * n];
                            for &v in &row[bj * block_size..(bj + 1) * block_size] {
                                amax = amax.max(v.abs());
                            }
                        }
                        scale_row[bj] = (amax / 127.0).max(1e-8);
                    }
                });
            // Divide by broadcast scale, clamp, round-ties-even, to i8.
            // Parallelized over output blocks (order-preserving).
            let n_blocks_row = bn;
            let q_chunks: Vec<Vec<i8>> = (0..bm)
                .into_par_iter()
                .map(|bi| {
                    let mut out = Vec::with_capacity(block_size * n);
                    for i in bi * block_size..(bi + 1) * block_size {
                        let row = &w[i * n..(i + 1) * n];
                        for bj in 0..bn {
                            let s = scale[bi * n_blocks_row + bj];
                            let tile = &row[bj * block_size..(bj + 1) * block_size];
                            out.extend(tile.iter().map(|&v| round_clamp_i8(v / s)));
                        }
                    }
                    out
                })
                .collect();
            let mut qdata = Vec::with_capacity(m * n);
            for chunk in q_chunks {
                qdata.extend(chunk);
            }
            Int8QuantResult {
                qdata,
                scale,
                scale_shape: vec![bm as u64, bn as u64],
            }
        }
    }
}

/// The clip ratio [`Quality::Int8Clip09`](crate::quality::Quality::Int8Clip09)
/// bakes into [`quantize_int8_weight_clipped`].
///
/// A named constant rather than a bare literal at the call site: the ratio is
/// simultaneously the algorithm's identity and part of the user-facing format
/// id (`int8_clip09`), so a magic `0.9` in three places is three chances to
/// drift. It is deliberately NOT tunable — see the function's docs for why a
/// ratio swept until the benchmark passed would not be a measurement.
pub const INT8_CLIP_RATIO: f32 = 0.9;

/// INT8 with **absmax clipping**: the quant scale is derived from
/// `clip_ratio * row_max` instead of `row_max`, so the top
/// `(1 - clip_ratio)` of the value range saturates and everything below it
/// gets a correspondingly finer step.
///
/// This is the Tier 2 S07 quality mode, selected by `--format int8_clip09`
/// ([`Quality::Int8Clip09`](crate::quality::Quality::Int8Clip09)). It is a NEW
/// function beside [`quantize_int8_weight`], not a parameter on it:
/// `quantize_int8_weight`'s bytes are pinned by `parity_fingerprint.rs` and by
/// the Python reference, so it must keep its exact current arithmetic for ever.
/// The two share the private helpers below, which is the only coupling.
///
/// # Why clipping at all
///
/// Absmax scaling spends the whole int8 range on the single largest element in
/// the row. Clipping deliberately gives that element up: it saturates, and the
/// freed range is redistributed as finer resolution across the many small
/// elements that actually carry most of the weight tensor's energy. It is the
/// standard MSE-vs-range trade and it is what QuaRot calls "clipping".
///
/// # The catch, stated plainly
///
/// The published justification for clipping is a **perplexity** improvement,
/// and this crate cannot compute perplexity — it quantizes weights and never
/// runs a model. Asserting a PPL number here would be asserting something the
/// crate does not measure. So the benefit has to be argued on the crate's own
/// metric, weight-space relative L2, through the shipped public dequantizer —
/// and that is a strictly harder bar, because this path does not quantize
/// activations at all, which is where clipping is usually said to help most.
///
/// Consequently the clip ratio is FIXED at [`INT8_CLIP_RATIO`] and is not
/// exposed as a flag. A ratio swept until `rel_l2` improved would be a
/// threshold fitted to the benchmark, not a measurement of a hypothesis; the
/// pre-registered decision rule in `benches/quality_error.rs` is only meaningful
/// if the configuration under test was fixed before the numbers were read.
///
/// # Float semantics
///
/// Identical to [`quantize_int8_weight`]: the row path keeps torch's
/// `scalar * reciprocal(tensor)` two-rounding division rather than IEEE
/// division (see the long note in `quantize_int8_weight`), because the pinned
/// row-mode goldens were produced that way. Clipping changes the *clamped
/// value*, never the order of the divisions, so a clipped row and an unclipped
/// row are computed by identical arithmetic on different amax values.
pub fn quantize_int8_weight_clipped(
    w: &[f32],
    m: usize,
    n: usize,
    mode: ScalingMode,
    block_size: usize,
    clip_ratio: f32,
) -> Int8QuantResult {
    debug_assert_eq!(w.len(), m * n);
    assert!(
        clip_ratio > 0.0 && clip_ratio <= 1.0,
        "INT8 clip ratio must be in (0, 1]; got {clip_ratio}"
    );

    // All-zeros early return, mirroring `quantize_int8_weight`. Clipping a
    // zero tensor is still zero, so this arm is unchanged by the ratio — which
    // is why it is a plain shared helper and not a ratio-dependent branch.
    if w.iter().all(|&v| v == 0.0) {
        return Int8QuantResult {
            qdata: vec![0i8; m * n],
            scale: vec![1.0; int8_zero_scale_len(mode, m, n, block_size)],
            scale_shape: int8_zero_scale_shape(mode, m, n, block_size),
        };
    }

    match mode {
        ScalingMode::Tensor => {
            let w_max = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            // Clipped absmax. The `clamp_min` stays for the same reason it does
            // in the base path: `clip_ratio` shrinks the value, so a tensor
            // whose absmax is below the floor would otherwise divide by ~0.
            let dequant_scale = (w_max * clip_ratio).max(1e-12) / 127.0;
            let quant_scale = 1.0 / dequant_scale;
            Int8QuantResult {
                qdata: quantize_scaled(w, quant_scale),
                scale: vec![dequant_scale],
                scale_shape: vec![1],
            }
        }
        ScalingMode::Row => {
            let mut scale = Vec::with_capacity(m);
            let mut qdata = Vec::with_capacity(m * n);
            let rows: Vec<&[f32]> = w.chunks_exact(n).collect();
            let out: Vec<(f32, Vec<i8>)> = rows
                .par_iter()
                .map(|row| {
                    let row_max = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                    let clipped = (row_max * clip_ratio).max(1e-12);
                    // torch: 127.0 / clipped  →  127.0 * reciprocal(clipped)
                    let quant_scale = 127.0f32 * (1.0f32 / clipped);
                    // torch: 1.0 / quant_scale  →  1.0 * reciprocal(quant_scale)
                    let dequant_scale = 1.0f32 / quant_scale;
                    (dequant_scale, quantize_scaled(row, quant_scale))
                })
                .collect();
            for (s, q) in out {
                scale.push(s);
                qdata.extend(q);
            }
            Int8QuantResult {
                qdata,
                scale,
                scale_shape: vec![m as u64, 1],
            }
        }
        ScalingMode::Block => {
            assert!(
                m.is_multiple_of(block_size) && n.is_multiple_of(block_size),
                "INT8 block-wise requires dims divisible by block_size={block_size}, got ({m}, {n})"
            );
            let bm = m / block_size;
            let bn = n / block_size;
            // Per-block amax over the (bs, bs) tile, clipped, then the same
            // `max(scale, 1e-8)` floor the base path applies.
            let mut scale = vec![0.0f32; bm * bn];
            scale
                .par_chunks_mut(bn)
                .enumerate()
                .for_each(|(bi, scale_row)| {
                    for bj in 0..bn {
                        let mut amax = 0.0f32;
                        for i in bi * block_size..(bi + 1) * block_size {
                            let row = &w[i * n..(i + 1) * n];
                            for &v in &row[bj * block_size..(bj + 1) * block_size] {
                                amax = amax.max(v.abs());
                            }
                        }
                        scale_row[bj] = (amax * clip_ratio / 127.0).max(1e-8);
                    }
                });
            let n_blocks_row = bn;
            let q_chunks: Vec<Vec<i8>> = (0..bm)
                .into_par_iter()
                .map(|bi| {
                    let mut out = Vec::with_capacity(block_size * n);
                    for i in bi * block_size..(bi + 1) * block_size {
                        let row = &w[i * n..(i + 1) * n];
                        for bj in 0..bn {
                            let s = scale[bi * n_blocks_row + bj];
                            let tile = &row[bj * block_size..(bj + 1) * block_size];
                            out.extend(tile.iter().map(|&v| round_clamp_i8(v / s)));
                        }
                    }
                    out
                })
                .collect();
            let mut qdata = Vec::with_capacity(m * n);
            for chunk in q_chunks {
                qdata.extend(chunk);
            }
            Int8QuantResult {
                qdata,
                scale,
                scale_shape: vec![bm as u64, bn as u64],
            }
        }
    }
}

/// Dequantize back to f32 (for bias correction later; exposed for tests now).
pub fn dequantize_int8(
    q: &[i8],
    scale: &[f32],
    m: usize,
    n: usize,
    mode: ScalingMode,
    block_size: usize,
) -> Vec<f32> {
    match mode {
        ScalingMode::Tensor => q.iter().map(|&v| v as f32 * scale[0]).collect(),
        ScalingMode::Row => q
            .iter()
            .enumerate()
            .map(|(idx, &v)| v as f32 * scale[idx / n])
            .collect(),
        ScalingMode::Block => {
            let bn = n / block_size;
            let mut out = vec![0.0f32; m * n];
            for bi in 0..(m / block_size) {
                for bj in 0..bn {
                    let s = scale[bi * bn + bj];
                    for i in bi * block_size..(bi + 1) * block_size {
                        let row_off = i * n;
                        for j in bj * block_size..(bj + 1) * block_size {
                            out[row_off + j] = q[row_off + j] as f32 * s;
                        }
                    }
                }
            }
            out
        }
    }
}

/// should_skip_layer_for_performance port (shape-only variant from stream_quant.py):
/// skip when not 2D, any dim < block_size, or any dim not divisible by block_size.
pub fn should_skip_shape(shape: &[u64], block_size: usize) -> bool {
    if shape.len() != 2 {
        return true;
    }
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    rows < block_size || cols < block_size || rows % block_size != 0 || cols % block_size != 0
}

/// Regex layer-exclusion mirroring `QuantConfig.excluded`: search semantics,
/// invalid pattern → never exclude (reference catches re.error and returns False).
pub struct ExcludePattern {
    re: Option<regex::Regex>,
}

impl ExcludePattern {
    pub fn new(pattern: Option<&str>) -> Self {
        Self {
            re: pattern.and_then(|p| regex::Regex::new(p).ok()),
        }
    }

    pub fn is_excluded(&self, name: &str) -> bool {
        self.re.as_ref().is_some_and(|r| r.is_match(name))
    }
}

// --- shared helpers -------------------------------------------------------- //

/// Scale-tensor SHAPE for the all-zeros early return, per scaling mode.
///
/// Extracted so [`quantize_int8_weight_clipped`] gets the same arm without
/// duplicating the match. `quantize_int8_weight` keeps its own inline copy of
/// this logic on purpose: it is a parity-critical function whose text is pinned
/// by review, and it should not be made to depend on a helper that a later edit
/// could move underneath it. The two are asserted equal by
/// `clipped_zero_tensor_matches_base_shapes`.
fn int8_zero_scale_shape(mode: ScalingMode, m: usize, n: usize, block_size: usize) -> Vec<u64> {
    match mode {
        ScalingMode::Tensor => vec![1],
        ScalingMode::Row => vec![m as u64, 1],
        ScalingMode::Block => {
            let bm = m / block_size;
            let bn = n / block_size;
            vec![bm as u64, bn as u64]
        }
    }
}

/// Element count of [`int8_zero_scale_shape`] — always the product of its dims.
fn int8_zero_scale_len(mode: ScalingMode, m: usize, n: usize, block_size: usize) -> usize {
    match mode {
        ScalingMode::Tensor => 1,
        ScalingMode::Row => m,
        ScalingMode::Block => (m / block_size) * (n / block_size),
    }
}

#[inline]
fn round_clamp_i8(scaled: f32) -> i8 {
    // torch.round on CPU = round-half-to-even; then clamp [-127, 127]; then to int8.
    let r = scaled.round_ties_even();
    let c = r.clamp(-127.0, 127.0);
    c as i8
}

#[inline]
fn quantize_scaled(data: &[f32], quant_scale: f32) -> Vec<i8> {
    data.par_iter()
        .map(|&v| round_clamp_i8(v * quant_scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensorwise_hand_computed() {
        // [2, 2]: values 4, -2, 1, 0 → amax 4 → dequant 4/127 ≈ 0.031496063
        // quant_scale = 127/4 exactly (since amax>1e-12, no clamp effect)
        let w = [4.0f32, -2.0, 1.0, 0.0];
        let r = quantize_int8_weight(&w, 2, 2, ScalingMode::Tensor, 128);
        // 4*(127/4)=127, -2*31.75=-63.5 → ties-even → -64? No: -63.5 rounds to -64
        // (nearest even of -63/-64 is -64), 1*31.75=31.5 → ties-even → 32.
        assert_eq!(r.qdata, vec![127, -64, 32, 0]);
        assert_eq!(r.scale_shape, vec![1]);
        let expected = 4.0f32 / 127.0;
        assert_eq!(r.scale[0], expected);
    }

    #[test]
    fn tensorwise_zero_tensor_ones_scale() {
        let w = vec![0.0f32; 256 * 128];
        let r = quantize_int8_weight(&w, 256, 128, ScalingMode::Tensor, 128);
        assert!(r.qdata.iter().all(|&v| v == 0));
        assert_eq!(r.scale, vec![1.0]);
        assert_eq!(r.scale_shape, vec![1]);
    }

    #[test]
    fn block_zero_tensor_ones_scale_grid() {
        let w = vec![0.0f32; 256 * 256];
        let r = quantize_int8_weight(&w, 256, 256, ScalingMode::Block, 128);
        assert_eq!(r.scale.len(), 4);
        assert!(r.scale.iter().all(|&s| s == 1.0));
        assert_eq!(r.scale_shape, vec![2, 2]);
    }

    #[test]
    fn rowwise_matches_reference_formula() {
        // Two rows with different amax; verify per-row scales are reciprocals of
        // quant_scale = 127/amax (NOT amax/127 directly — must be bit-equal to 1/x).
        let w = [8.0f32, -4.0, 1.0, 2.0, 100.0, 0.5];
        let r = quantize_int8_weight(&w, 3, 2, ScalingMode::Row, 128);
        let q0 = 127.0 / 8.0f32.max(1e-12); // row 0 amax=8
        assert_eq!(r.scale[0], 1.0 / q0);
        let q2 = 127.0 / 100.0f32.max(1e-12); // row 2 amax=100
        assert_eq!(r.scale[2], 1.0 / q2);
        assert_eq!(r.qdata[0], 127); // 8 * (127/8)
        assert_eq!(r.qdata[4], 127); // 100 * (127/100)
        assert_eq!(r.scale_shape, vec![3, 1]);
    }

    #[test]
    fn blockwise_hand_computed_with_epsilon_clamp() {
        // [2,2] with bs=1: four independent blocks.
        // Block (0,1) has value ~0: amax tiny → scale clamps to 1e-8 floor.
        let tiny = 1e-11f32;
        let w = [254.0, tiny, 1.0, -3.0];
        let r = quantize_int8_weight(&w, 2, 2, ScalingMode::Block, 1);
        assert_eq!(r.scale[0], (254.0f32 / 127.0)); // 2.0
        assert_eq!(r.scale[1], 1e-8); // epsilon clamp active
        assert_eq!(r.scale[2], 1.0f32 / 127.0);
        assert_eq!(r.scale[3], 3.0f32 / 127.0);
        assert_eq!(r.qdata, vec![127, 0, 127, -127]); // tiny/1e-8 = 1e-3 → 0
    }

    #[test]
    fn blockwise_dequant_roundtrip_close() {
        // Deterministic pseudo-random matrix.
        let (m, n, bs) = (256usize, 128usize, 128usize);
        let mut w = vec![0.0f32; m * n];
        let mut x: u64 = 0x243F6A8885A308D3;
        for v in &mut w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
        }
        let r = quantize_int8_weight(&w, m, n, ScalingMode::Block, bs);
        let dq = dequantize_int8(&r.qdata, &r.scale, m, n, ScalingMode::Block, bs);
        // Max abs error must be <= half step per block.
        for (i, (&d, &o)) in w.iter().zip(&dq).enumerate() {
            let bi = i / n / bs;
            let bj = i % n / bs;
            let s = r.scale[bi * (n / bs) + bj];
            assert!((d - o).abs() <= s / 2.0 + 1e-6, "idx {i}: {d} vs {o}");
        }
    }

    #[test]
    fn parallel_equals_sequential_blockwise() {
        // Byte-for-byte equality between rayon-parallel result and a sequential
        // recomputation (plan 2.4 requirement).
        let (m, n, bs) = (512usize, 256usize, 128usize);
        let mut w = vec![0.0f32; m * n];
        let mut x: u64 = 0xDEADBEEFCAFEF00D;
        for v in &mut w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = ((x % 40_000) as f32 / 20_000.0) - 1.0;
        }
        let par = quantize_int8_weight(&w, m, n, ScalingMode::Block, bs);

        // Sequential reference implementation.
        let bm = m / bs;
        let bn = n / bs;
        let mut seq_q = vec![0i8; m * n];
        let mut seq_s = vec![0.0f32; bm * bn];
        for bi in 0..bm {
            for bj in 0..bn {
                let mut amax = 0.0f32;
                for i in bi * bs..(bi + 1) * bs {
                    for j in bj * bs..(bj + 1) * bs {
                        amax = amax.max(w[i * n + j].abs());
                    }
                }
                let s = (amax / 127.0).max(1e-8);
                seq_s[bi * bn + bj] = s;
                for i in bi * bs..(bi + 1) * bs {
                    for j in bj * bs..(bj + 1) * bs {
                        seq_q[i * n + j] = round_clamp_i8(w[i * n + j] / s);
                    }
                }
            }
        }
        assert_eq!(par.qdata, seq_q);
        assert_eq!(par.scale, seq_s);
    }

    // --- quantize_int8_weight_clipped (Tier 2 S07) ------------------------- //

    /// The single most important property: at `clip_ratio = 1.0` the clipped
    /// kernel must be **bit-identical** to the base kernel, in every scaling
    /// mode. This is what makes "the base function is unchanged" a checkable
    /// claim rather than an assertion — the new code is a generalization of the
    /// old one, and `ratio = 1.0` is the specialization that must reduce to it.
    ///
    /// If a future edit perturbs the shared float semantics (the
    /// reciprocal-multiply quirk in particular), this fails immediately.
    #[test]
    fn clip_ratio_one_is_bit_identical_to_the_base_kernel() {
        let (m, n, bs) = (256usize, 128usize, 128usize);
        let mut w = vec![0.0f32; m * n];
        let mut x: u64 = 0x243F6A8885A308D3;
        for v in &mut w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
        }
        for mode in [ScalingMode::Tensor, ScalingMode::Row, ScalingMode::Block] {
            let base = quantize_int8_weight(&w, m, n, mode, bs);
            let clipped = quantize_int8_weight_clipped(&w, m, n, mode, bs, 1.0);
            assert_eq!(base.qdata, clipped.qdata, "qdata differs in {mode:?}");
            assert_eq!(base.scale, clipped.scale, "scale differs in {mode:?}");
            assert_eq!(
                base.scale_shape, clipped.scale_shape,
                "scale_shape differs in {mode:?}"
            );
        }
    }

    /// The zero-tensor early return must produce the SAME shape and scale as
    /// the base path. Guards the extracted helpers against drifting away from
    /// the inline match they were extracted from.
    #[test]
    fn clipped_zero_tensor_matches_base_shapes() {
        for (m, n, bs) in [(256usize, 128usize, 128usize), (4usize, 4usize, 1usize)] {
            let w = vec![0.0f32; m * n];
            for mode in [ScalingMode::Tensor, ScalingMode::Row, ScalingMode::Block] {
                let base = quantize_int8_weight(&w, m, n, mode, bs);
                let clipped = quantize_int8_weight_clipped(&w, m, n, mode, bs, INT8_CLIP_RATIO);
                assert_eq!(base.scale_shape, clipped.scale_shape, "{mode:?}");
                assert_eq!(base.scale, clipped.scale, "{mode:?}");
                assert!(clipped.qdata.iter().all(|&v| v == 0), "{mode:?}");
            }
        }
    }

    /// Hand-computed row-mode case: clipping must shrink the dequant scale by
    /// exactly the ratio, and the row's max element must then SATURATE.
    ///
    /// Row = [8, -4, 1, 2], row_max = 8, ratio 0.9 → clipped amax 7.2 →
    /// quant_scale = 127/7.2 → 8 maps to 8 * 17.638.. = 141.1 → clamps to 127.
    /// The saturation is the entire mechanism: without it this function would
    /// be a no-op rescale.
    #[test]
    fn clipped_row_saturates_the_max_element() {
        let w = [8.0f32, -4.0, 1.0, 2.0, 100.0, 0.5];
        let r = quantize_int8_weight_clipped(&w, 3, 2, ScalingMode::Row, 128, 0.9);

        // Row 0: unclipped would give qdata[0] == 127 exactly. Clipped, the
        // same element overflows and must clamp to the same 127 — which is why
        // "did the max element change?" is NOT a valid saturation check.
        assert_eq!(r.qdata[0], 127);
        // Row 2 (amax 100) is where clipping is visible in the DATA: unclipped
        // it also gives 127, but the small element beside it gets more codes.
        assert_eq!(r.qdata[4], 127, "row 2 max still saturates at 127");

        // The dequant scale is `clipped_amax / 127` (as a reciprocal chain).
        let qs = 127.0f32 * (1.0f32 / (100.0f32 * 0.9));
        assert_eq!(r.scale[2], 1.0f32 / qs, "scale must use the clipped amax");
        // And it is strictly SMALLER than the unclipped scale — clipping trades
        // range for precision, so the step must get finer, not coarser.
        let base = quantize_int8_weight(&w, 3, 2, ScalingMode::Row, 128);
        assert!(
            r.scale[2] < base.scale[2],
            "clipping must shrink the step: {} vs {}",
            r.scale[2],
            base.scale[2]
        );
    }

    /// A non-saturating element must get genuinely FINER resolution — the
    /// positive half of the trade, and the only reason clipping could ever be
    /// worth shipping.
    ///
    /// The mechanism, stated carefully because it is easy to get backwards: a
    /// smaller `row_max` means a SMALLER dequant step, so the same value maps
    /// to a LARGER integer code. So the assertion is on the RECONSTRUCTION
    /// (which is what "finer" means), not on the integer moving toward zero.
    ///
    /// Row [100, 0.5]: unclipped the step is 100/127 = 0.787, so `0.5` lands on
    /// code 1 and reconstructs to 0.787. Clipped the step is 90/127 = 0.709, so
    /// it still lands on code 1 but reconstructs to 0.709 — closer to 0.5.
    #[test]
    fn clipping_gives_small_elements_finer_resolution() {
        let w = [100.0f32, 0.5];
        let base = quantize_int8_weight(&w, 1, 2, ScalingMode::Row, 128);
        let clipped = quantize_int8_weight_clipped(&w, 1, 2, ScalingMode::Row, 128, 0.9);

        // Finer step: the dequant scale (which IS the step) shrinks by the ratio.
        assert!(
            clipped.scale[0] < base.scale[0],
            "clipping must shrink the step: {} vs {}",
            clipped.scale[0],
            base.scale[0]
        );

        // The claim that matters: the small element reconstructs MORE closely.
        let recon = |r: &Int8QuantResult| f32::from(r.qdata[1]) * r.scale[0];
        let err = |v: f32| (v - 0.5f32).abs();
        assert!(
            err(recon(&clipped)) < err(recon(&base)),
            "the unsaturated element must reconstruct more closely: clipped err {} vs base err {}",
            err(recon(&clipped)),
            err(recon(&base))
        );
        // The max still saturates at the top code in both arms — which is why
        // "did the max element change?" is not a valid check that clipping ran.
        assert_eq!(clipped.qdata[0], base.qdata[0]);
    }

    /// A ratio outside `(0, 1]` is a caller bug, not something to silently
    /// clamp. A ratio of 0 would divide by ~0; a ratio > 1 would *expand* the
    /// range and is the opposite of the intended trade, so both are rejected
    /// loudly rather than producing a plausible-looking artifact.
    #[test]
    #[should_panic(expected = "clip ratio must be in (0, 1]")]
    fn clip_ratio_out_of_range_is_rejected() {
        let w = [1.0f32, 2.0];
        quantize_int8_weight_clipped(&w, 1, 2, ScalingMode::Row, 128, 1.5);
    }

    #[test]
    #[should_panic(expected = "clip ratio must be in (0, 1]")]
    fn clip_ratio_zero_is_rejected() {
        let w = [1.0f32, 2.0];
        quantize_int8_weight_clipped(&w, 1, 2, ScalingMode::Row, 128, 0.0);
    }

    /// The all-zeros arm must fire BEFORE the ratio assert would matter, i.e.
    /// a valid zero tensor with any legal ratio is still a clean early return.
    #[test]
    fn clipped_zero_tensor_is_fine_at_the_shipped_ratio() {
        let w = vec![0.0f32; 8 * 4];
        let r = quantize_int8_weight_clipped(&w, 8, 4, ScalingMode::Row, 128, INT8_CLIP_RATIO);
        assert_eq!(r.scale, vec![1.0; 8]);
        assert_eq!(r.scale_shape, vec![8, 1]);
    }

    #[test]
    fn should_skip_shape_all_branches() {
        let b = 128;
        assert!(should_skip_shape(&[32, 16, 3, 3], b), "non-2D");
        assert!(should_skip_shape(&[128, 64], b), "cols < bs");
        assert!(should_skip_shape(&[64, 128], b), "rows < bs");
        assert!(should_skip_shape(&[130, 130], b), "not divisible");
        assert!(!should_skip_shape(&[384, 256], b), "quantizable");
        assert!(!should_skip_shape(&[128, 128], b), "exactly bs");
    }

    #[test]
    fn exclude_pattern_search_semantics_and_invalid_fallback() {
        let ex = ExcludePattern::new(Some("attn_norm|text_embed"));
        assert!(ex.is_excluded("blocks.0.attn_norm.weight"));
        assert!(ex.is_excluded("text_embed.out.weight"));
        assert!(!ex.is_excluded("blocks.1.ff.weight"));
        // Invalid regex → never excludes (mirrors reference except-clause).
        let bad = ExcludePattern::new(Some("[invalid"));
        assert!(!bad.is_excluded("attn_norm.weight"));
        // None → disabled.
        let none = ExcludePattern::new(None);
        assert!(!none.is_excluded("anything.weight"));
    }
}

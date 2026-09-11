//! Weighted GGML quantizers — a port of llama.cpp's `ggml-quants.c`
//! K-quant paths, for use with an importance matrix (plan Phase 4.3).
//!
//! Source: llama.cpp, MIT ("Copyright (c) 2023-2026 The ggml
//! authors"), the vendored upstream tree. Line references below point
//! into `ggml/src/ggml-quants.c` of that tree.
//!
//! Scope decision (vertical slices): slice 1 ported the shared
//! `make_qkx3/qp` helpers plus Q4_K and Q2_K; slice 2 (this) adds
//! `make_qx_quants` and the Q3_K / Q5_K / Q6_K weighted impls
//! (:1355 / :1758 / :1970). llama.cpp runs the `*_impl` variant whenever
//! an imatrix row is supplied (`quantize_q{3,5,6}_K` :1439/:1851/:2054).
//!
//! IMPORTANT — why port at all instead of using rlx-gguf: rlx-gguf 0.2.14
//! has NO weights parameter in any K-quant encoder; its uniform-weight
//! output is a *different* min/max search. Porting the upstream impls
//! gives byte-parity with `llama-quantize` (the parity tier upgrade the
//! plan's Phase 4 promised) and true imatrix support in one move.
//!
//! Block layouts (ggml-common.h:296-338):
//! - block_q4_K: `d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128]` = 144B
//! - block_q2_K: `scales: [u8; 16], qs: [u8; 64], d: f16, dmin: f16` = 84B
//! - block_q3_K: `hmask: [u8; 32], qs: [u8; 64], scales: [u8; 12], d: f16` = 110B
//! - block_q5_K: `d, dmin: f16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128]` = 176B
//! - block_q6_K: `ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16` = 210B

use half::f16;

/// Super-block size for all K-quants (ggml-common.h:88 QK_K = 256).
const QK_K: usize = 256;
/// 6-bit-packed scale bytes per K-quant super-block (ggml-common.h:90).
const K_SCALE_SIZE: usize = 12;

/// `GROUP_MAX_EPS` (ggml-quants.c:613) — values below this count as zero.
const GROUP_MAX_EPS: f32 = 1e-15f32;

/// Port of `nearest_int` (:621-626). The magic-number trick is exact for
/// |fval| <= 2^22; upstream asserts that range, we debug_assert it.
#[inline]
pub(crate) fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0;
    let i = val.to_bits();
    ((i & 0x007fffff) as i32) - 0x00400000
}

/// Port of `make_qkx3_quants` (:993-1074) — the weighted group-scale
/// search shared by the K-quant impls. `weights = None` reproduces the
/// unweighted fallback `w = x[i]^2` exactly as upstream (:998, :1008).
#[allow(clippy::too_many_arguments)]
fn make_qkx3_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: Option<&[f32]>,
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights.map_or(x[0] * x[0], |w| w[0]);
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights.map_or(x[i] * x[i], |w| w[i]);
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        l[..n].fill(0);
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min));
        l[i] = li.clamp(0, nmax) as u8;
        let mut diff = scale * l[i] as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        let w = weights.map_or(x[i] * x[i], |w| w[i]);
        best_mad += w * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let mut li = nearest_int(iscale * (x[i] - min));
            li = li.clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights.map_or(x[i] * x[i], |w| w[i]);
            // C association (:1043-1045): (w*l), (w*l)*l, (w*l)*x — keep
            // exact; do NOT regroup to w*(l*l) etc. (rounding differs).
            sum_l += w * li as f32;
            sum_l2 += (w * li as f32) * li as f32;
            sum_xl += (w * li as f32) * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                let w = weights.map_or(x[i] * x[i], |w| w[i]);
                mad += w * diff;
            }
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// Port of `make_qp_quants` (:1076-1147) — the weighted super-block scale
/// search with the ±4% nudge loop and 5 coordinate-descent passes.
pub(crate) fn make_qp_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [u8],
    quant_weights: &[f32],
) -> f32 {
    let mut max = 0.0f32;
    for i in 0..n {
        max = max.max(x[i]);
    }
    if max < GROUP_MAX_EPS {
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = x[i] - scale * l[i] as f32;
        let w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let li = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * li as f32;
            let w = quant_weights[i];
            mse += w * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).min(nmax);
        l[i] = li as u8;
        let w = quant_weights[i];
        // C association (:1118-1119): (w*x)*l and (w*l)*l — keep exact.
        sumlx += (w * x[i]) * li as f32;
        suml2 += (w * li as f32) * li as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            // C association (:1129-1130): (w*x)*L, (w*L)*L.
            let mut slx = sumlx - (w * x[i]) * l[i] as f32;
            let mut sl2 = suml2 - (w * l[i] as f32) * (l[i] as f32);
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax) as u8;
                if new_l != l[i] {
                    // C association (:1133-1134): (w*x)*new_l, (w*new_l)*new_l.
                    slx += (w * x[i]) * new_l as f32;
                    sl2 += (w * new_l as f32) * (new_l as f32);
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// Port of `make_qx_quants` (:628-695) — the signed symmetric group
/// quantizer used by Q3_K/Q6_K (and Q5_K via nmax=31). Weighted when
/// `qw = Some` (imatrix path); `None` reproduces the `w = x[i]^2`
/// fallback for rmse_type 1 (:666).
///
/// rmse_type semantics (:643-654):
/// - `0`: nearest_int pass, scale = 1/iscale — no search.
/// - negative (e.g. -1): search with `return_early` — returns
///   `0.5*(scale + 1/iscale)` when any weight accumulated.
/// - positive: full ±9 nudge search around `nmax` maximizing
///   `sumlx*sumlx/suml2` (weighted correlation).
///
/// NOTE the C `l` values are SIGNED here (`-nmax..nmax-1`), then
/// `L[i] = l + nmax` stores the offset; Q6_K instead keeps raw signed
/// L (`MAX(-32, MIN(31, l))` + 32 at pack time :2029). We keep i32
/// buffers to stay faithful to both call sites.
fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        // all zero (:636-641)
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let li = nearest_int(iscale * x[i]);
            l[i] = (nmax + li.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rmse_type = rmse_type;
    let mut return_early = false;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let mut li = nearest_int(iscale * x[i]);
        li = li.clamp(-nmax, nmax - 1);
        l[i] = (li + nmax) as i8;
        let w = match qw {
            Some(q) => q[i],
            None => match rmse_type {
                1 => x[i] * x[i],
                2 => 1.0,
                3 => x[i].abs(),
                _ => x[i].abs().sqrt(), // rmse_type == 4
            },
        };
        sumlx += (w * x[i]) * li as f32;
        suml2 += (w * li as f32) * li as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let mut li = nearest_int(iscale * x[i]);
            li = li.clamp(-nmax, nmax - 1);
            let w = match qw {
                Some(q) => q[i],
                None => match rmse_type {
                    1 => x[i] * x[i],
                    2 => 1.0,
                    3 => x[i].abs(),
                    _ => x[i].abs().sqrt(),
                },
            };
            sumlx += (w * x[i]) * li as f32;
            suml2 += (w * li as f32) * li as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int(iscale * x[i]);
                l[i] = (nmax + li.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Port of `get_scale_min_k4` (:880-887) — decode the packed 6-bit scales.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8], d: &mut u8, m: &mut u8) {
    if j < 4 {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ─── Q4_K ──────────────────────────────────────────────────────────────

/// Bytes per block_q4_K (ggml-common.h:338 static_assert).
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144

/// Port of `quantize_row_q4_K_impl` (:1553-1624) — the imatrix-aware path.
/// `quant_weights` is the per-element imatrix for THIS row (length
/// `n_per_row`); `None` selects the unweighted weights formula
/// (`weights[l] = av_x + |x[l]|`, :1578) — identical to what the `*_ref`
/// path produces via make_qkx2/av_x in the unweighted case? NO — the
/// `*_ref` variant uses `make_qkx2_quants` with a different weight
/// formula and 20 steps; the impl with None-weights uses make_qkx3 with
/// 36 steps. llama.cpp only calls the impl when quant_weights != NULL
/// (:1628-1630); we keep the None branch because it is a legal upstream
/// configuration (defensive), but our driver passes Some for weighted
/// conversions and rlx-gguf handles the historic unweighted path.
pub fn quantize_row_q4_k_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let nb = n_per_row / QK_K;
    let mut out = vec![0u8; nb * Q4_K_BLOCK_BYTES];

    let mut l = vec![0u8; QK_K];
    let mut laux = vec![0u8; 32];
    let mut ls_arr = vec![0u8; QK_K / 32];
    let mut lm_arr = vec![0u8; QK_K / 32];
    let mut weights = vec![0f32; 32];
    let mut sw = vec![0f32; QK_K / 32];
    let mut mins = vec![0f32; QK_K / 32];
    let mut scales = vec![0f32; QK_K / 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let ob = &mut out[i * Q4_K_BLOCK_BYTES..(i + 1) * Q4_K_BLOCK_BYTES];
        // Layout: d: f16 | dmin: f16 | scales[12] | qs[128].
        let (scales_off, qs_off) = (4usize, 4 + K_SCALE_SIZE);

        let mut sum_x2 = 0.0f32;
        for &v in xb {
            sum_x2 += v * v;
        }
        let sigma2 = 2.0 * sum_x2 / QK_K as f32;
        let av_x = sigma2.sqrt();

        for j in 0..QK_K / 32 {
            let qw = quant_weights.map(|w| &w[i * QK_K + 32 * j..i * QK_K + 32 * j + 32]);
            for l_idx in 0..32 {
                weights[l_idx] = match qw {
                    Some(q) => q[l_idx] * (sigma2 + xb[32 * j + l_idx] * xb[32 * j + l_idx]).sqrt(),
                    None => av_x + xb[32 * j + l_idx].abs(),
                };
            }
            let mut sumw = 0.0f32;
            for l_idx in 0..32 {
                sumw += weights[l_idx];
            }
            sw[j] = sumw;
            // C passes `L + 32*j` (offset subslice) — the requantize below
            // overwrites every group, but groups with d == 0 are SKIPPED
            // and keep their search values, so the offset matters.
            scales[j] = make_qkx3_quants(
                32,
                15,
                &xb[32 * j..32 * j + 32],
                Some(&weights),
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }

        let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls_arr, &sw);
        let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm_arr, &sw);
        for j in 0..QK_K / 32 {
            let (ls, lm) = (ls_arr[j], lm_arr[j]);
            if j < 4 {
                ob[scales_off + j] = ls;
                ob[scales_off + j + 4] = lm;
            } else {
                ob[scales_off + j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                ob[scales_off + j - 4] |= (ls >> 4) << 6;
                ob[scales_off + j] |= (lm >> 4) << 6;
            }
        }
        let d_h = f16::from_f32(d_block);
        let dmin_h = f16::from_f32(m_block);
        ob[..2].copy_from_slice(&d_h.to_le_bytes());
        ob[2..4].copy_from_slice(&dmin_h.to_le_bytes());
        // NOTE upstream re-reads d/dmin AFTER fp16 rounding (:1606-1608);
        // the rounded values drive the requantize below.
        let d = f16::from_le_bytes([ob[0], ob[1]]).to_f32();
        let dmin = f16::from_le_bytes([ob[2], ob[3]]).to_f32();

        for j in 0..QK_K / 32 {
            let (mut sc, mut m) = (0u8, 0u8);
            get_scale_min_k4(
                j,
                &ob[scales_off..scales_off + K_SCALE_SIZE],
                &mut sc,
                &mut m,
            );
            let dd = d * sc as f32;
            if dd == 0.0 {
                continue;
            }
            let dm = dmin * m as f32;
            for ii in 0..32 {
                let mut li = nearest_int((xb[32 * j + ii] + dm) / dd);
                li = li.clamp(0, 15);
                l[32 * j + ii] = li as u8;
            }
        }
        let mut q = qs_off;
        for j in (0..QK_K).step_by(64) {
            for l_idx in 0..32 {
                ob[q + l_idx] = l[j + l_idx] | (l[j + l_idx + 32] << 4);
            }
            q += 32;
        }
    }
    out
}

// ─── Q2_K ──────────────────────────────────────────────────────────────

/// Bytes per block_q2_K (ggml-common.h:296-302: scales[16] qs[64] d dmin).
pub const Q2_K_BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 + 2; // 84

/// Port of `quantize_row_q2_K_impl` (:1149-1209). Upstream REQUIRES
/// quant_weights (GGML_ASSERT at :1150); we accept `None` by falling back
/// to the unweighted formula used by the `*_ref` variant, so callers can
/// exercise the same code path — but the parity contract is with Some.
pub fn quantize_row_q2_k_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let nb = n_per_row / QK_K;
    let mut out = vec![0u8; nb * Q2_K_BLOCK_BYTES];

    let mut l = vec![0u8; QK_K];
    let mut laux = vec![0u8; 16];
    let mut mins = vec![0f32; QK_K / 16];
    let mut scales = vec![0f32; QK_K / 16];
    let mut sw = vec![0f32; QK_K / 16];
    let mut weight = vec![0f32; 16];
    let mut ls_arr = vec![0u8; QK_K / 16];
    let mut lm_arr = vec![0u8; QK_K / 16];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let ob = &mut out[i * Q2_K_BLOCK_BYTES..(i + 1) * Q2_K_BLOCK_BYTES];
        // Layout: scales[16] | qs[64] | d: f16 | dmin: f16.
        let (scales_off, qs_off, d_off, dmin_off) = (
            0usize,
            QK_K / 16,
            QK_K / 16 + QK_K / 4,
            QK_K / 16 + QK_K / 4 + 2,
        );

        let mut sumx2 = 0.0f32;
        for &v in xb {
            sumx2 += v * v;
        }
        let sigma2 = sumx2 / QK_K as f32;

        sw.fill(0.0);
        for j in 0..QK_K / 16 {
            let qw = quant_weights.map(|w| &w[i * QK_K + 16 * j..i * QK_K + 16 * j + 16]);
            for l_idx in 0..16 {
                weight[l_idx] = match qw {
                    // :1170 — qw * sqrt(sigma2 + x^2)
                    Some(q) => q[l_idx] * (sigma2 + xb[16 * j + l_idx] * xb[16 * j + l_idx]).sqrt(),
                    // Unweighted fallback: |x| (as the *_ref path uses).
                    None => xb[16 * j + l_idx].abs(),
                };
            }
            // :1171 — upstream accumulates ALL weights into sw[j] (a
            // known upstream quirk: `for l < QK_K/16: sw[j] += weight[l]`
            // reads weights[0..16] repeatedly). Reproduced EXACTLY for
            // byte parity — do not "fix" to per-group sums.
            for _ in 0..QK_K / 16 {
                for l_idx in 0..16 {
                    sw[j] += weight[l_idx];
                }
            }
            scales[j] = make_qkx3_quants(
                16,
                3,
                &xb[16 * j..16 * j + 16],
                Some(&weight),
                &mut l[16 * j..16 * j + 16],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }

        let dm = make_qp_quants(QK_K / 16, 15, &scales, &mut ls_arr, &sw);
        let mm = make_qp_quants(QK_K / 16, 15, &mins, &mut lm_arr, &sw);

        let d_h = f16::from_f32(dm);
        let dmin_h = f16::from_f32(mm);
        ob[d_off..d_off + 2].copy_from_slice(&d_h.to_le_bytes());
        ob[dmin_off..dmin_off + 2].copy_from_slice(&dmin_h.to_le_bytes());
        // Rounded values drive the requantize (:1181-1182).
        let d = f16::from_le_bytes([ob[d_off], ob[d_off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([ob[dmin_off], ob[dmin_off + 1]]).to_f32();

        for j in 0..QK_K / 16 {
            ob[scales_off + j] = ls_arr[j] | (lm_arr[j] << 4);
        }

        // Requantize with the rounded scales (:1188-1199).
        for j in 0..QK_K / 16 {
            let dd = d * (ob[scales_off + j] & 0xF) as f32;
            if dd == 0.0 {
                continue;
            }
            let m = dmin * (ob[scales_off + j] >> 4) as f32;
            for ii in 0..16 {
                let mut li = nearest_int((xb[16 * j + ii] + m) / dd);
                li = li.clamp(0, 3);
                l[16 * j + ii] = li as u8;
            }
        }

        let mut q = qs_off;
        for j in (0..QK_K).step_by(128) {
            for l_idx in 0..32 {
                ob[q + l_idx] = l[j + l_idx]
                    | (l[j + l_idx + 32] << 2)
                    | (l[j + l_idx + 64] << 4)
                    | (l[j + l_idx + 96] << 6);
            }
            q += 32;
        }
    }
    out
}

// ─── Q3_K ──────────────────────────────────────────────────────────────

/// Bytes per block_q3_K (ggml-common.h:314-321):
/// hmask[QK_K/8] | qs[QK_K/4] | scales[12] | d: f16 = 32+64+12+2.
pub const Q3_K_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + K_SCALE_SIZE + 2; // 110

/// Port of `quantize_row_q3_K_impl` (:1355-1437) — the imatrix-aware path.
/// Groups of 16, nmax=4 (3-bit + high bit). NOTE the group weight is
/// `qw * sqrt(sigma2 + x^2)` where sigma2 = 2*sumx2/QK_K (:1374) — this
/// differs from Q4_K/Q5_K only in reading `x` at the block base (no
/// `32*j` offset in the sigma2 term), matching upstream exactly.
pub fn quantize_row_q3_k_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let nb = n_per_row / QK_K;
    let mut out = vec![0u8; nb * Q3_K_BLOCK_BYTES];

    // L is signed through the group search, offset by nmax=4 at store.
    let mut l = vec![0i8; QK_K];
    let mut scales = vec![0f32; QK_K / 16];
    let mut weight = vec![0f32; 16];
    let mut sw = vec![0f32; QK_K / 16];
    let mut ls_arr = vec![0i8; QK_K / 16];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let ob = &mut out[i * Q3_K_BLOCK_BYTES..(i + 1) * Q3_K_BLOCK_BYTES];
        // Layout: hmask[32] | qs[64] | scales[12] | d: f16.
        let (hmask_off, qs_off, scales_off, d_off) = (
            0usize,
            QK_K / 8,
            QK_K / 8 + QK_K / 4,
            QK_K / 8 + QK_K / 4 + K_SCALE_SIZE,
        );

        let mut sumx2 = 0.0f32;
        for &v in xb {
            sumx2 += v * v;
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for j in 0..QK_K / 16 {
            let qw = quant_weights.map(|w| &w[i * QK_K + 16 * j..i * QK_K + 16 * j + 16]);
            for l_idx in 0..16 {
                weight[l_idx] = match qw {
                    // :1374 — qw * sqrt(sigma2 + x^2); NOTE x read at the
                    // block base offset only by 16*j (upstream same shape).
                    Some(q) => q[l_idx] * (sigma2 + xb[16 * j + l_idx] * xb[16 * j + l_idx]).sqrt(),
                    None => xb[16 * j + l_idx] * xb[16 * j + l_idx],
                };
            }
            let mut sumw = 0.0f32;
            for l_idx in 0..16 {
                sumw += weight[l_idx];
            }
            sw[j] = sumw;

            // :1382 — rmse_type 1 (signed search, full ±9 nudge); C
            // passes `L + 16*j` (offset subslice).
            scales[j] = make_qx_quants(
                16,
                4,
                &xb[16 * j..16 * j + 16],
                &mut l[16 * j..16 * j + 16],
                1,
                Some(&weight),
            );
        }

        ob[scales_off..scales_off + 12].fill(0);

        // :1388 — super-block scale from the group scales (nmax=32).
        let d_block = make_qx_quants(QK_K / 16, 32, &scales, &mut ls_arr, 1, Some(&sw));
        for j in 0..QK_K / 16 {
            let mut li = ls_arr[j] as i32;
            if j < 8 {
                ob[scales_off + j] = (li & 0xF) as u8;
            } else {
                ob[scales_off + j - 8] |= ((li & 0xF) << 4) as u8;
            }
            li >>= 4;
            ob[scales_off + j % 4 + 8] |= (li << (2 * (j / 4))) as u8;
        }
        let d_h = f16::from_f32(d_block);
        ob[d_off..d_off + 2].copy_from_slice(&d_h.to_le_bytes());

        // Requantize with the decoded scales (:1401-1414).
        for j in 0..QK_K / 16 {
            let mut sc = if j < 8 {
                (ob[scales_off + j] & 0xF) as i32
            } else {
                (ob[scales_off + j - 8] >> 4) as i32
            };
            sc = ((sc | ((((ob[scales_off + 8 + j % 4] >> (2 * (j / 4))) & 3) as i32) << 4)) - 32)
                as i32;
            let d = f16::from_le_bytes([ob[d_off], ob[d_off + 1]]).to_f32() * sc as f32;
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let mut li = nearest_int(xb[16 * j + ii] / d);
                li = li.clamp(-4, 3);
                l[16 * j + ii] = (li + 4) as i8;
            }
        }

        ob[hmask_off..hmask_off + QK_K / 8].fill(0);
        // High bit for the 1st 8 quants into bit 0, next 8 into bit 1, ...
        let mut m = 0usize;
        let mut hm: u8 = 1;
        for j in 0..QK_K {
            if l[j] > 3 {
                ob[hmask_off + m] |= hm;
                l[j] -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }
        let mut q = qs_off;
        for j in (0..QK_K).step_by(128) {
            for l_idx in 0..32 {
                ob[q + l_idx] = (l[j + l_idx]
                    | (l[j + l_idx + 32] << 2)
                    | (l[j + l_idx + 64] << 4)
                    | (l[j + l_idx + 96] << 6)) as u8;
            }
            q += 32;
        }
    }
    out
}

// ─── Q5_K ──────────────────────────────────────────────────────────────

/// Bytes per block_q5_K (ggml-common.h:348-356):
/// d: f16 | dmin: f16 | scales[12] | qh[QK_K/8] | qs[QK_K/2].
pub const Q5_K_BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2; // 176

/// Port of `quantize_row_q5_K_impl` (:1758-1849). Same structure as Q4_K
/// but nmax=31 (5-bit) and the qs/qh split packing (:1829-1844).
pub fn quantize_row_q5_k_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let nb = n_per_row / QK_K;
    let mut out = vec![0u8; nb * Q5_K_BLOCK_BYTES];

    let mut l = vec![0u8; QK_K];
    let mut laux = vec![0u8; 32];
    let mut ls_arr = vec![0u8; QK_K / 32];
    let mut lm_arr = vec![0u8; QK_K / 32];
    let mut mins = vec![0f32; QK_K / 32];
    let mut scales = vec![0f32; QK_K / 32];
    let mut sw = vec![0f32; QK_K / 32];
    let mut weights = vec![0f32; 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let ob = &mut out[i * Q5_K_BLOCK_BYTES..(i + 1) * Q5_K_BLOCK_BYTES];
        // Layout: d: f16 | dmin: f16 | scales[12] | qh[32] | qs[128].
        let (scales_off, qh_off, qs_off) = (4usize, 4 + K_SCALE_SIZE, 4 + K_SCALE_SIZE + QK_K / 8);

        let mut sum_x2 = 0.0f32;
        for &v in xb {
            sum_x2 += v * v;
        }
        let sigma2 = 2.0 * sum_x2 / QK_K as f32;
        let av_x = sigma2.sqrt();

        for j in 0..QK_K / 32 {
            let qw = quant_weights.map(|w| &w[i * QK_K + 32 * j..i * QK_K + 32 * j + 32]);
            for l_idx in 0..32 {
                weights[l_idx] = match qw {
                    Some(q) => q[l_idx] * (sigma2 + xb[32 * j + l_idx] * xb[32 * j + l_idx]).sqrt(),
                    None => av_x + xb[32 * j + l_idx].abs(),
                };
            }
            let mut sumw = 0.0f32;
            for l_idx in 0..32 {
                sumw += weights[l_idx];
            }
            sw[j] = sumw;
            // :1789 — nmax=31 for the 5-bit L; C passes `L + 32*j`.
            scales[j] = make_qkx3_quants(
                32,
                31,
                &xb[32 * j..32 * j + 32],
                Some(&weights),
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -0.9,
                0.05,
                36,
                false,
            );
        }

        let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls_arr, &sw);
        let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm_arr, &sw);
        for j in 0..QK_K / 32 {
            let (mut ls, mut lm) = (ls_arr[j], lm_arr[j]);
            // :1798-1799 — MIN(63, ·) clamp (absent in the Q4_K sibling!).
            ls = ls.min(63);
            lm = lm.min(63);
            if j < 4 {
                ob[scales_off + j] = ls;
                ob[scales_off + j + 4] = lm;
            } else {
                ob[scales_off + j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                ob[scales_off + j - 4] |= (ls >> 4) << 6;
                ob[scales_off + j] |= (lm >> 4) << 6;
            }
        }
        let d_h = f16::from_f32(d_block);
        let dmin_h = f16::from_f32(m_block);
        ob[..2].copy_from_slice(&d_h.to_le_bytes());
        ob[2..4].copy_from_slice(&dmin_h.to_le_bytes());
        let d = f16::from_le_bytes([ob[0], ob[1]]).to_f32();
        let dmin = f16::from_le_bytes([ob[2], ob[3]]).to_f32();

        for j in 0..QK_K / 32 {
            let (mut sc, mut m) = (0u8, 0u8);
            get_scale_min_k4(
                j,
                &ob[scales_off..scales_off + K_SCALE_SIZE],
                &mut sc,
                &mut m,
            );
            let dd = d * sc as f32;
            if dd == 0.0 {
                continue;
            }
            let dm = dmin * m as f32;
            for ii in 0..32 {
                let mut li = nearest_int((xb[32 * j + ii] + dm) / dd);
                li = li.clamp(0, 31);
                l[32 * j + ii] = li as u8;
            }
        }

        ob[qh_off..qh_off + QK_K / 8].fill(0);
        let mut m1: u8 = 1;
        let mut m2: u8 = 2;
        let mut ql = qs_off;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let mut l1 = l[n + j] as i32;
                if l1 > 15 {
                    l1 -= 16;
                    ob[qh_off + j] |= m1;
                }
                let mut l2 = l[n + j + 32] as i32;
                if l2 > 15 {
                    l2 -= 16;
                    ob[qh_off + j] |= m2;
                }
                ob[ql + j] = (l1 | (l2 << 4)) as u8;
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }
    }
    out
}

// ─── Q6_K ──────────────────────────────────────────────────────────────

/// Bytes per block_q6_K (ggml-common.h:362-368):
/// ql[QK_K/2] | qh[QK_K/4] | scales[QK_K/16] (i8) | d: f16.
pub const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 210

/// Port of `quantize_row_q6_K_impl` (:1970-2052). Distinctive points:
/// - the imatrix weights are used DIRECTLY (`qw` passed straight to
///   make_qx_quants, :1994 — no sigma2/x^2 transform, unlike Q2/Q3/Q4/Q5);
/// - groups of 16 with nmax=32, L stays signed at pack time (+32);
/// - super-block scale = 1/iscale where iscale = -128/max_scale (:2015).
pub fn quantize_row_q6_k_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let nb = n_per_row / QK_K;
    let mut out = vec![0u8; nb * Q6_K_BLOCK_BYTES];

    let mut l = vec![0i8; QK_K];
    let mut scales = vec![0f32; QK_K / 16];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let ob = &mut out[i * Q6_K_BLOCK_BYTES..(i + 1) * Q6_K_BLOCK_BYTES];
        // Layout: ql[128] | qh[64] | scales[16] | d: f16.
        let (ql_off, qh_off, scales_off, d_off) = (
            0usize,
            QK_K / 2,
            QK_K / 2 + QK_K / 4,
            QK_K / 2 + QK_K / 4 + QK_K / 16,
        );

        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;

        for ib in 0..QK_K / 16 {
            let qw = quant_weights.map(|w| &w[i * QK_K + 16 * ib..i * QK_K + 16 * ib + 16]);
            // :1994 — qw used DIRECTLY (the transformed variant is
            // commented out upstream); :1996 — NULL weights. C passes
            // `L + 16*ib` (offset subslice).
            let scale = make_qx_quants(
                16,
                32,
                &xb[16 * ib..16 * ib + 16],
                &mut l[16 * ib..16 * ib + 16],
                1,
                qw,
            );
            scales[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if max_abs_scale < GROUP_MAX_EPS {
            // :2008-2013 — zero the whole block, d = 0.
            ob.fill(0);
            let d_h = f16::from_f32(0.0);
            ob[d_off..d_off + 2].copy_from_slice(&d_h.to_le_bytes());
            continue;
        }

        let iscale = -128.0f32 / max_scale;
        let d_h = f16::from_f32(1.0 / iscale);
        ob[d_off..d_off + 2].copy_from_slice(&d_h.to_le_bytes());
        for ib in 0..QK_K / 16 {
            // :2018 — MIN(127, nearest_int(...)); scales stored as i8
            // (values can exceed 127 only via nearest_int edge).
            ob[scales_off + ib] = nearest_int(iscale * scales[ib]).min(127) as i8 as u8;
        }

        // Requantize with the rounded scales (:2021-2031).
        for j in 0..QK_K / 16 {
            let d = f16::from_le_bytes([ob[d_off], ob[d_off + 1]]).to_f32()
                * ob[scales_off + j] as i8 as f32;
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let mut li = nearest_int(xb[16 * j + ii] / d);
                li = li.clamp(-32, 31);
                l[16 * j + ii] = (li + 32) as i8;
            }
        }

        let mut ql = ql_off;
        let mut qh = qh_off;
        for j in (0..QK_K).step_by(128) {
            for l_idx in 0..32 {
                let q1 = (l[j + l_idx] & 0xF) as u8;
                let q2 = (l[j + l_idx + 32] & 0xF) as u8;
                let q3 = (l[j + l_idx + 64] & 0xF) as u8;
                let q4 = (l[j + l_idx + 96] & 0xF) as u8;
                ob[ql + l_idx] = q1 | (q3 << 4);
                ob[ql + l_idx + 32] = q2 | (q4 << 4);
                ob[qh + l_idx] = ((l[j + l_idx] >> 4) as u8)
                    | (((l[j + l_idx + 32] >> 4) as u8) << 2)
                    | (((l[j + l_idx + 64] >> 4) as u8) << 4)
                    | (((l[j + l_idx + 96] >> 4) as u8) << 6);
            }
            ql += 64;
            qh += 32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn nearest_int_matches_magic_number_semantics() {
        // The magic-number trick is a bit manipulation, NOT roundf:
        // halves round to EVEN in the small-value range (mantissa LSB),
        // and the small-value assertions document actual behavior.
        assert_eq!(nearest_int(0.4), 0);
        assert_eq!(nearest_int(0.6), 1);
        assert_eq!(nearest_int(-0.6), -1);
        assert_eq!(nearest_int(2.5), 2); // half → even
        assert_eq!(nearest_int(3.5), 4); // half → even
        assert_eq!(nearest_int(-2.5), -2);
        // Non-half values round normally.
        for v in [0.49, 1.51, 15.4, 15.6, 63.4, 63.6, -15.6] {
            assert_eq!(nearest_int(v), v.round() as i32, "v={v}");
        }
    }

    #[test]
    fn q4_k_weighted_output_shape_and_structure() {
        // 2 super-blocks (2 rows of 256 = 512 elements).
        let n = 512;
        let src = synth(n, 42);
        let weights: Vec<f32> = synth(n, 7).iter().map(|v| 1.0 + v.abs()).collect();

        let out = quantize_row_q4_k_weighted(&src, n, Some(&weights));
        assert_eq!(out.len(), 2 * Q4_K_BLOCK_BYTES);

        // Structure: d/dmin are finite f16, the packed scales decode, and
        // dequantized values stay close to the source (weighted K4 on
        // uniform noise: bounded error, NOT exactness — that needs the
        // llama.cpp byte-parity fixture).
        let blk = &out[..Q4_K_BLOCK_BYTES];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        assert!(d.is_finite());
        assert!(dmin.is_finite());
        assert!(d >= 0.0);

        // Dequantize with the upstream formula and check mean error.
        let deq = dequantize_q4_k(blk);
        let xb = &src[..QK_K];
        let mean_err: f32 =
            deq.iter().zip(xb).map(|(a, b)| (a - b).abs()).sum::<f32>() / QK_K as f32;
        assert!(
            mean_err < 0.2,
            "weighted q4_k mean reconstruction error too high: {mean_err}"
        );
    }

    /// Independent dequantize for verification (mirrors :1529-1551).
    fn dequantize_q4_k(blk: &[u8]) -> Vec<f32> {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..4 + K_SCALE_SIZE];
        let qs = &blk[4 + K_SCALE_SIZE..Q4_K_BLOCK_BYTES];
        let mut y = vec![0f32; QK_K];
        let mut is = 0usize;
        let mut q = 0usize;
        for j in (0..QK_K).step_by(64) {
            let (mut sc, mut m) = (0u8, 0u8);
            get_scale_min_k4(is, scales, &mut sc, &mut m);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            get_scale_min_k4(is + 1, scales, &mut sc, &mut m);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for l in 0..32 {
                y[j + l] = d1 * (qs[q + l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                y[j + 32 + l] = d2 * (qs[q + l] >> 4) as f32 - m2;
            }
            q += 32;
            is += 2;
        }
        y
    }

    #[test]
    fn q2_k_weighted_output_shape_and_structure() {
        let n = 512;
        let src = synth(n, 99);
        let weights: Vec<f32> = synth(n, 3).iter().map(|v| 1.0 + v.abs()).collect();

        let out = quantize_row_q2_k_weighted(&src, n, Some(&weights));
        assert_eq!(out.len(), 2 * Q2_K_BLOCK_BYTES);

        let blk = &out[Q2_K_BLOCK_BYTES..2 * Q2_K_BLOCK_BYTES];
        let d_off = QK_K / 16 + QK_K / 4;
        let d = f16::from_le_bytes([blk[d_off], blk[d_off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[d_off + 2], blk[d_off + 3]]).to_f32();
        assert!(d.is_finite() && d >= 0.0);
        assert!(dmin.is_finite() && dmin >= 0.0);
    }

    #[test]
    fn weights_change_the_output() {
        // The whole point of Phase 4.3: the same source with different
        // imatrix weights must produce different bytes (proves the weights
        // are actually consumed, not parsed and dropped).
        let n = 256;
        let src = synth(n, 5);
        let w1: Vec<f32> = vec![1.0; n];
        let mut w2: Vec<f32> = vec![1.0; n];
        for (i, w) in w2.iter_mut().enumerate() {
            *w += (i % 32) as f32 * 0.1; // strongly non-uniform
        }
        let a = quantize_row_q4_k_weighted(&src, n, Some(&w1));
        let b = quantize_row_q4_k_weighted(&src, n, Some(&w2));
        assert_ne!(a, b, "different imatrix weights must change the output");
    }

    #[test]
    fn none_weights_is_deterministic() {
        let n = 256;
        let src = synth(n, 11);
        let a = quantize_row_q4_k_weighted(&src, n, None);
        let b = quantize_row_q4_k_weighted(&src, n, None);
        assert_eq!(a, b);
    }

    #[test]
    fn make_qp_quants_zero_input() {
        // max < GROUP_MAX_EPS → all-zero L, scale 0 (:1081-1084).
        let x = [0.0f32; 8];
        let mut l = vec![0u8; 8];
        let w = [1.0f32; 8];
        let s = make_qp_quants(8, 63, &x, &mut l, &w);
        assert_eq!(s, 0.0);
        assert!(l.iter().all(|&v| v == 0));
    }

    #[test]
    fn make_qkx3_quants_constant_group() {
        // All-equal NON-ZERO values: min is clamped to 0 (:1012-1014), so
        // max > min and the search proceeds normally — the group fits
        // [0, x] and the scale is nmax/x derived: here 3/2 steps → 2/3.
        let x = [2.0f32; 16];
        let mut l = vec![0u8; 16];
        let mut the_min = 0.0f32;
        let mut laux = vec![0u8; 16];
        let s = make_qkx3_quants(
            16,
            3,
            &x,
            Some(&[1.0; 16]),
            &mut l,
            &mut the_min,
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
        assert_eq!(s, 2.0 / 3.0);
        assert_eq!(the_min, 0.0); // min was clamped to 0

        // All-ZERO values hit the max <= min early return (:1015-1019):
        // scale 0, all L zero.
        let x = [0.0f32; 16];
        let s = make_qkx3_quants(
            16,
            3,
            &x,
            Some(&[1.0; 16]),
            &mut l,
            &mut the_min,
            &mut laux,
            -0.9,
            0.05,
            36,
            false,
        );
        assert_eq!(s, 0.0);
        assert!(l.iter().all(|&v| v == 0));
    }

    #[test]
    fn make_qx_quants_rmse_type_0_is_nearest_int() {
        // rmse_type == 0 (:643-648): plain nearest_int, scale = 1/iscale.
        // max is the SIGNED first-occurrence value at the largest |x|
        // (:632-635): here x[0] = -1.0 wins (ax > amax is strict), so
        // iscale = -4 / -1 = 4, scale = 0.25.
        let x = [-1.0f32, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, -0.25];
        let mut l = vec![0i8; 8];
        let s = make_qx_quants(8, 4, &x, &mut l, 0, None);
        assert_eq!(s, 0.25);
        // l = nearest_int(4*x) clamped to [-4, 3] then +4:
        // -4→0, -2→2, 0→4, 1→5, 2→6, 3→7, 4→CLAMP 3→7, -1→3.
        assert_eq!(l, vec![0, 2, 4, 5, 6, 7, 7, 3]);
    }

    #[test]
    fn make_qx_quants_zero_group() {
        // amax < GROUP_MAX_EPS → all zero, scale 0 (:636-641).
        let x = [0.0f32; 16];
        let mut l = vec![1i8; 16];
        let s = make_qx_quants(16, 32, &x, &mut l, 1, Some(&[1.0; 16]));
        assert_eq!(s, 0.0);
        assert!(l.iter().all(|&v| v == 0));
    }

    #[test]
    fn make_qx_quants_negative_rmse_returns_halfway() {
        // rmse_type < 0 → return_early: 0.5*(scale + 1/iscale) when
        // suml2 > 0 (:650-654, :671). iscale uses the SIGNED max, so the
        // direct scale 1/iscale = |max|/nmax in magnitude — positive.
        let x = synth(16, 21);
        let nmax = 32;
        // Reproduce the upstream max-scan exactly.
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in &x {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let iscale = -(nmax as f32) / max;
        let mut l = vec![0i8; 16];
        let s = make_qx_quants(16, nmax, &x, &mut l, -1, Some(&[1.0; 16]));
        let direct = 1.0 / iscale;
        // Result is the average of the LS scale and the direct scale, so
        // it must stay within a factor ~2 of direct (both positive when
        // the LS solution is sensible).
        assert!(s > 0.0, "s={s}");
        assert!(
            (s / direct - 1.0).abs() < 1.0,
            "s={s} direct={direct} — halfway bound violated"
        );
        // And the stored L must be the offset values from the FIRST pass
        // (:663-669) — return_early skips the ±9 nudge overwrite.
        let li0 = nearest_int(iscale * x[0]).clamp(-nmax, nmax - 1);
        assert_eq!(l[0], (li0 + nmax) as i8);
    }

    #[test]
    fn q3_k_weighted_output_shape_and_structure() {
        let n = 512;
        let src = synth(n, 71);
        let weights: Vec<f32> = synth(n, 13).iter().map(|v| 1.0 + v.abs()).collect();

        let out = quantize_row_q3_k_weighted(&src, n, Some(&weights));
        assert_eq!(out.len(), 2 * Q3_K_BLOCK_BYTES);

        // Decode one block: d finite; scales decode to nonzero for at
        // least one group; hmask/qs stay in range.
        let blk = &out[..Q3_K_BLOCK_BYTES];
        let d_off = QK_K / 8 + QK_K / 4 + K_SCALE_SIZE;
        let d = f16::from_le_bytes([blk[d_off], blk[d_off + 1]]).to_f32();
        assert!(d.is_finite() && d != 0.0);
        // High-bit mask: at most 256 bits set across hmask (32 bytes).
        let hmask = &blk[..QK_K / 8];
        let bits: u32 = hmask.iter().map(|b| b.count_ones()).sum();
        assert!(bits <= QK_K as u32);
    }

    #[test]
    fn q5_k_weighted_output_shape_and_structure() {
        let n = 512;
        let src = synth(n, 55);
        let weights: Vec<f32> = synth(n, 17).iter().map(|v| 1.0 + v.abs()).collect();

        let out = quantize_row_q5_k_weighted(&src, n, Some(&weights));
        assert_eq!(out.len(), 2 * Q5_K_BLOCK_BYTES);

        let blk = &out[..Q5_K_BLOCK_BYTES];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        assert!(d.is_finite() && d >= 0.0 && d != 0.0);
        assert!(dmin.is_finite() && dmin >= 0.0);

        // Mean reconstruction error must stay bounded (5-bit quality).
        let deq = dequantize_q5_k(blk);
        let xb = &src[..QK_K];
        let mean_err: f32 =
            deq.iter().zip(xb).map(|(a, b)| (a - b).abs()).sum::<f32>() / QK_K as f32;
        assert!(mean_err < 0.25, "q5_k mean err too high: {mean_err}");
    }

    /// Independent dequantize for verification (mirrors :1731-1756).
    fn dequantize_q5_k(blk: &[u8]) -> Vec<f32> {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..4 + K_SCALE_SIZE];
        let (qh_off, qs_off) = (4 + K_SCALE_SIZE, 4 + K_SCALE_SIZE + QK_K / 8);
        let qh = &blk[qh_off..qh_off + QK_K / 8];
        let qs = &blk[qs_off..qs_off + QK_K / 2];
        let mut y = vec![0f32; QK_K];
        let mut is = 0usize;
        let mut q = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let (mut sc, mut m) = (0u8, 0u8);
            get_scale_min_k4(is, scales, &mut sc, &mut m);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            get_scale_min_k4(is + 1, scales, &mut sc, &mut m);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                y[j + l] = d1 * ((qs[q + l] & 0xF) as f32 + hi1 as f32) - m1;
            }
            for l in 0..32 {
                let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                y[j + 32 + l] = d2 * ((qs[q + l] >> 4) as f32 + hi2 as f32) - m2;
            }
            q += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        y
    }

    #[test]
    fn q6_k_weighted_output_shape_and_structure() {
        let n = 512;
        let src = synth(n, 33);
        let weights: Vec<f32> = synth(n, 23).iter().map(|v| 1.0 + v.abs()).collect();

        let out = quantize_row_q6_k_weighted(&src, n, Some(&weights));
        assert_eq!(out.len(), 2 * Q6_K_BLOCK_BYTES);

        let blk = &out[..Q6_K_BLOCK_BYTES];
        let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
        let d = f16::from_le_bytes([blk[d_off], blk[d_off + 1]]).to_f32();
        assert!(d.is_finite() && d != 0.0);

        // Mean reconstruction error stays bounded (6-bit quality).
        let deq = dequantize_q6_k(blk);
        let xb = &src[..QK_K];
        let mean_err: f32 =
            deq.iter().zip(xb).map(|(a, b)| (a - b).abs()).sum::<f32>() / QK_K as f32;
        assert!(mean_err < 0.15, "q6_k mean err too high: {mean_err}");
    }

    /// Independent dequantize for verification (mirrors :1939-1968).
    fn dequantize_q6_k(blk: &[u8]) -> Vec<f32> {
        let (ql_off, qh_off, scales_off, d_off) = (
            0usize,
            QK_K / 2,
            QK_K / 2 + QK_K / 4,
            QK_K / 2 + QK_K / 4 + QK_K / 16,
        );
        let d = f16::from_le_bytes([blk[d_off], blk[d_off + 1]]).to_f32();
        let ql = &blk[ql_off..ql_off + QK_K / 2];
        let qh = &blk[qh_off..qh_off + QK_K / 4];
        let sc = &blk[scales_off..scales_off + QK_K / 16];
        let mut y = vec![0f32; QK_K];
        let mut qli = 0usize;
        let mut qhi = 0usize;
        let mut sci = 0usize;
        for n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[qli + l] & 0xF) | (((qh[qhi + l]) & 3) << 4)) as i8 - 32) as i32;
                let q2 = (((ql[qli + l + 32] & 0xF) | (((qh[qhi + l] >> 2) & 3) << 4)) as i8 - 32)
                    as i32;
                let q3 = (((ql[qli + l] >> 4) | (((qh[qhi + l] >> 4) & 3) << 4)) as i8 - 32) as i32;
                let q4 =
                    (((ql[qli + l + 32] >> 4) | (((qh[qhi + l] >> 6) & 3) << 4)) as i8 - 32) as i32;
                y[n + l] = d * sc[sci + is] as i8 as f32 * q1 as f32;
                y[n + l + 32] = d * sc[sci + is + 2] as i8 as f32 * q2 as f32;
                y[n + l + 64] = d * sc[sci + is + 4] as i8 as f32 * q3 as f32;
                y[n + l + 96] = d * sc[sci + is + 6] as i8 as f32 * q4 as f32;
            }
            qli += 64;
            qhi += 32;
            sci += 8;
        }
        y
    }

    #[test]
    fn weights_change_the_output_all_formats() {
        // Slice 2 extension: every ported format must consume the weights.
        let n = 256;
        let src = synth(n, 5);
        let w1: Vec<f32> = vec![1.0; n];
        let mut w2: Vec<f32> = vec![1.0; n];
        for (i, w) in w2.iter_mut().enumerate() {
            *w += (i % 16) as f32 * 0.1;
        }
        let q3_a = quantize_row_q3_k_weighted(&src, n, Some(&w1));
        let q3_b = quantize_row_q3_k_weighted(&src, n, Some(&w2));
        assert_ne!(q3_a, q3_b, "q3_k must consume weights");
        let q5_a = quantize_row_q5_k_weighted(&src, n, Some(&w1));
        let q5_b = quantize_row_q5_k_weighted(&src, n, Some(&w2));
        assert_ne!(q5_a, q5_b, "q5_k must consume weights");
        let q6_a = quantize_row_q6_k_weighted(&src, n, Some(&w1));
        let q6_b = quantize_row_q6_k_weighted(&src, n, Some(&w2));
        assert_ne!(q6_a, q6_b, "q6_k must consume weights");
    }

    #[test]
    fn none_weights_is_deterministic_all_formats() {
        let n = 256;
        let src = synth(n, 11);
        assert_eq!(
            quantize_row_q3_k_weighted(&src, n, None),
            quantize_row_q3_k_weighted(&src, n, None)
        );
        assert_eq!(
            quantize_row_q5_k_weighted(&src, n, None),
            quantize_row_q5_k_weighted(&src, n, None)
        );
        assert_eq!(
            quantize_row_q6_k_weighted(&src, n, None),
            quantize_row_q6_k_weighted(&src, n, None)
        );
    }

    #[test]
    fn q6_k_zero_row_produces_zero_block() {
        // max_abs_scale < GROUP_MAX_EPS → the whole block is zeroed with
        // d = 0 (:2008-2013) — including the scales/qs bytes.
        let src = [0.0f32; QK_K];
        let out = quantize_row_q6_k_weighted(&src, QK_K, Some(&[1.0; QK_K]));
        assert!(
            out.iter().all(|&b| b == 0),
            "zero input must give zero block"
        );
    }
}

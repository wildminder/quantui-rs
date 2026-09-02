//! Weighted IQ-quantizers — ports of llama.cpp's `quantize_row_iq*_impl`
//! functions (docs/ref/llama.cpp, MIT; see `ggml_iq_grid` for the lattice
//! infrastructure and `gguf_quants` for the shared K-quant helpers).
//!
//! Scope (Unsloth plan Phase 4.3, IQ slice A): iq2_xxs (:3294) and
//! iq2_xs (:3472) — the 2-bit lattice family sharing `kMaxQ = 3` and
//! `make_qp_quants` group scales.
//!
//! Both impls share the same skeleton:
//! - per 256-block: sigma2 = sumx2/QK_K (NOTE: no factor 2 here, unlike
//!   the K-quants!), group weights `qw * sqrt(sigma2 + x^2)`, waux =
//!   sqrt(weight);
//! - per 8-lane group: sign-normalize to positive xval with parity fixup
//!   (odd nflip → flip the min weight*x^2 element, :3340-3361);
//! - group scale via `make_qp_quants` (xxs) or direct `max/(2*kMaxQ-1)`
//!   initial (xs), then the ±6 (xxs) / ±9 (xs) `is` search: nearest_int
//!   the lanes, look up kmap, off-grid → best neighbour, keep the best
//!   weighted LS scale;
//! - a second pass for xs: re-fit off-grid groups with the final scale;
//! - pack: grid indices + signs into q2 u16s, 6-bit scales nibbles,
//!   super-block d = max_scale/31.
//!
//! C float associations are preserved exactly (see the K-quant module's
//! comments — `w*x*q` means `((w*x)*q)` etc.).

use half::f16;

use crate::gguf_iq_grid::{lattice, IqLattice};
use crate::gguf_quants::{make_qp_quants, nearest_int};

const QK_K: usize = 256;
/// `GROUP_MAX_EPS` (ggml-quants.c:613).
const GROUP_MAX_EPS: f32 = 1e-15f32;

/// Bytes per block_iq2_xxs: d: f16 + qs: [u16; 32] = 66.
pub const IQ2_XXS_BLOCK_BYTES: usize = 2 + 2 * (QK_K / 8);
/// Bytes per block_iq2_xs: d: f16 + qs: [u16; 32] + scales: [u8; 8] = 74.
pub const IQ2_XS_BLOCK_BYTES: usize = 2 + 2 * (QK_K / 8) + QK_K / 32;

/// Port of `iq2_find_best_neighbour` (:3270-3292) — pick the neighbour
/// list entry minimizing the weighted squared distance `scale*q - x`.
/// Returns the grid index and writes the lane L values (both packed
/// grid bytes and the caller's L slice).
fn iq2_find_best_neighbour(
    neighbours: &[u16],
    grid: &[u64],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) -> usize {
    let num_neighbors = neighbours[0] as usize;
    debug_assert!(num_neighbors > 0);
    let mut best_d2 = f32::INFINITY;
    let mut grid_index: isize = -1;
    for j in 1..=num_neighbors {
        let pg = grid[neighbours[j] as usize].to_le_bytes();
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = pg[i] as i8 as f32;
            let diff = scale * q - xval[i];
            // C association (:3282-3283): (weight*diff)*diff.
            d2 += (weight[i] * diff) * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = neighbours[j] as isize;
        }
    }
    debug_assert!(grid_index >= 0);
    let pg = grid[grid_index as usize].to_le_bytes();
    for i in 0..8 {
        l[i] = ((pg[i] as i8) - 1) / 2;
    }
    grid_index as usize
}

/// Sign-normalize one 8-lane group into xval with the parity fixup and
/// sign byte (xxs/xs shared code, :3343-3360 / :3524-3541).
/// Returns (signs, nflip) and writes xval[8k..8k+8].
#[inline]
fn sign_normalize_group(xb: &[f32], weight: &[f32], xval: &mut [f32], k: usize) -> (u8, usize) {
    let mut nflip = 0usize;
    let mut s: u8 = 0;
    for i in 0..8 {
        if xb[8 * k + i] >= 0.0 {
            xval[8 * k + i] = xb[8 * k + i];
        } else {
            xval[8 * k + i] = -xb[8 * k + i];
            nflip += 1;
            s |= 1 << i;
        }
    }
    if nflip % 2 == 1 {
        // Flip the element with the smallest weight*x^2 (:3350-3358).
        let mut imin = 0usize;
        let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
        for i in 1..8 {
            let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
            if ax < min {
                min = ax;
                imin = i;
            }
        }
        xval[8 * k + imin] = -xval[8 * k + imin];
        s ^= 1 << imin;
    }
    (s & 127, nflip)
}

/// Port of `quantize_row_iq2_xxs_impl` (:3294-3470).
pub fn quantize_row_iq2_xxs_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: &[f32],
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let lat = lattice(IqLattice::Iq2Xxs);
    let nbl = n_per_row / QK_K;
    let mut out = vec![0u8; nbl * IQ2_XXS_BLOCK_BYTES];

    const K_MAX_Q: i32 = 3;
    let (qs_off, d_off) = (2usize, 0usize);

    let mut scales = vec![0f32; QK_K / 32];
    let mut weight = vec![0f32; 32];
    let mut xval = vec![0f32; 32];
    let mut l = vec![0i8; 32];
    let mut laux = vec![0i8; 32];
    let mut waux = vec![0f32; 32];
    let mut block_signs = [0u8; 4];

    for ibl in 0..nbl {
        let xb_block = &x[ibl * QK_K..(ibl + 1) * QK_K];
        let ob = &mut out[ibl * IQ2_XXS_BLOCK_BYTES..(ibl + 1) * IQ2_XXS_BLOCK_BYTES];
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
        // q2 mirrors the C uint32_t[2*(QK_K/32)] = u32[16].
        let mut q2: [u32; 16] = [0; 16];
        let mut max_scale = 0.0f32;

        let mut sumx2 = 0.0f32;
        for &v in xb_block {
            sumx2 += v * v;
        }
        let sigma2 = sumx2 / QK_K as f32;

        for ib in 0..QK_K / 32 {
            let xb = &xb_block[32 * ib..32 * ib + 32];
            let qw = &quant_weights[QK_K * ibl + 32 * ib..QK_K * ibl + 32 * ib + 32];
            for i in 0..32 {
                // (:3338) — no factor 2 on sigma2 for the iq2 family!
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
            }
            for i in 0..32 {
                waux[i] = weight[i].sqrt();
            }
            for k in 0..4 {
                let (s, _) = sign_normalize_group(xb, &weight, &mut xval, k);
                block_signs[k] = s;
            }
            let mut max = xval[0];
            for &v in &xval[1..32] {
                max = max.max(v);
            }
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                l.fill(0);
                continue;
            }
            // (:3369) — L is u8-viewed in C; kMaxQ+1 = 4 levels.
            let mut l_u8 = [0u8; 32];
            let mut scale = make_qp_quants(32, K_MAX_Q + 1, &xval, &mut l_u8, &weight);
            for i in 0..32 {
                l[i] = l_u8[i] as i8;
            }
            let eff_max = scale * K_MAX_Q as f32;
            if eff_max <= 0.0 {
                scales[ib] = 0.0;
                l.fill(0);
                continue;
            }
            let mut best = 0.0f32;
            // NOTE: unlike qp/qx searches, the iq2_xxs loop does NOT skip
            // is=0 (:3377 has no continue) — all 13 offsets run.
            for is in -6i32..=6 {
                let id = (2.0 * K_MAX_Q as f32 - 1.0 + 0.1 * is as f32) / eff_max;
                let this_scale = 1.0 / id;
                for k in 0..4 {
                    for i in 0..8 {
                        let li = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] = li.clamp(0, K_MAX_Q - 1) as i8;
                    }
                    let u = lat.pack_index(&laux[8 * k..8 * k + 8]);
                    let grid_index = lat.kmap[u as usize];
                    if grid_index < 0 {
                        let neighbours = lat.neighbours_of(grid_index);
                        iq2_find_best_neighbour(
                            neighbours,
                            &lat.grid,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            this_scale,
                            &mut laux[8 * k..8 * k + 8],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    // (:3397-3398) — ((w*x)*q), ((w*q)*q).
                    sumqx += (w * xval[i]) * q;
                    sumq2 += (w * q) * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                }
            }
            if scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..4 {
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let li = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        let li = li.clamp(0, K_MAX_Q - 1);
                        u |= (li as u16) << (2 * i);
                        l[8 * k + i] = li as i8;
                    }
                    let grid_index = lat.kmap[u as usize];
                    if grid_index < 0 {
                        let neighbours = lat.neighbours_of(grid_index);
                        iq2_find_best_neighbour(
                            neighbours,
                            &lat.grid,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            scale,
                            &mut l[8 * k..8 * k + 8],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l[i] as f32 + 1.0;
                    sumqx += (w * xval[i]) * q;
                    sumq2 += (w * q) * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                // (:3431-3436) — defensive sign flip, upstream keeps it.
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }
            for k in 0..4 {
                let u = lat.pack_index(&l[8 * k..8 * k + 8]);
                let grid_index = lat.kmap[u as usize];
                // (:3442-3446) — after the final pass the point MUST be
                // on the lattice (upstream GGML_ABORTs otherwise).
                assert!(
                    grid_index >= 0,
                    "iq2_xxs: point {u} not on grid after quantize"
                );
                // (:3447-3448) — u32 packing: index in low 8k bits,
                // signs in bit 7k.
                q2[2 * ib] |= (grid_index as u32) << (8 * k);
                q2[2 * ib + 1] |= (block_signs[k] as u32) << (7 * k);
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            // qs already zeroed via out init (:3455-3457).
            continue;
        }
        let d = max_scale / 31.0;
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        let id = 1.0 / d;
        for ib in 0..QK_K / 32 {
            let mut li = nearest_int(0.5 * (id * scales[ib] - 1.0));
            li = li.clamp(0, 15);
            q2[2 * ib + 1] |= (li as u32) << 28;
        }
        // Layout: d: f16 | qs: 16 × u32 (LE) — q2 packs 4 grid indices
        // (8k bits) + 4 signs (7k bits) + the scale nibble (bit 28) per
        // u32 PAIR; memcpy moves all 64 bytes (:3468).
        for (i, &v) in q2.iter().enumerate() {
            ob[qs_off + 4 * i..qs_off + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Port of `quantize_row_iq2_xs_impl` (:3472-3651).
pub fn quantize_row_iq2_xs_weighted(x: &[f32], n_per_row: usize, quant_weights: &[f32]) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let lat = lattice(IqLattice::Iq2Xs);
    let nbl = n_per_row / QK_K;
    let mut out = vec![0u8; nbl * IQ2_XS_BLOCK_BYTES];

    const K_MAX_Q: i32 = 3;
    let (qs_off, scales_off, d_off) = (2usize, 2 + 2 * (QK_K / 8), 0usize);

    let mut scales = vec![0f32; QK_K / 16];
    let mut weight = vec![0f32; 16];
    let mut xval = vec![0f32; 16];
    let mut l = vec![0i8; 16];
    let mut laux = vec![0i8; 16];
    let mut waux = vec![0f32; 16];
    let mut is_on_grid_aux = [false; 2];
    let mut block_signs = [0u8; 2];

    for ibl in 0..nbl {
        let xb_block = &x[ibl * QK_K..(ibl + 1) * QK_K];
        let ob = &mut out[ibl * IQ2_XS_BLOCK_BYTES..(ibl + 1) * IQ2_XS_BLOCK_BYTES];
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let mut q2: [u16; 2 * (QK_K / 16)] = [0; 32];
        ob[scales_off..scales_off + QK_K / 32].fill(0);
        let mut max_scale = 0.0f32;

        let mut sumx2 = 0.0f32;
        for &v in xb_block {
            sumx2 += v * v;
        }
        let sigma2 = sumx2 / QK_K as f32;

        for ib in 0..QK_K / 16 {
            let xb = &xb_block[16 * ib..16 * ib + 16];
            let qw = &quant_weights[QK_K * ibl + 16 * ib..QK_K * ibl + 16 * ib + 16];
            for i in 0..16 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
            }
            for i in 0..16 {
                waux[i] = weight[i].sqrt();
            }
            for k in 0..2 {
                let (s, _) = sign_normalize_group(xb, &weight, &mut xval, k);
                block_signs[k] = s;
            }
            let mut max = xval[0];
            for &v in &xval[1..16] {
                max = max.max(v);
            }
            l.fill(0);
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2.0 * K_MAX_Q as f32 - 1.0);
            // Per-group state (C resets these per ib iteration too).
            let mut is_on_grid = [true; 2];
            for is in -9i32..=9 {
                let id = (2.0 * K_MAX_Q as f32 - 1.0 + 0.1 * is as f32) / max;
                let this_scale = 1.0 / id;
                for k in 0..2 {
                    for i in 0..8 {
                        let li = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] = li.clamp(0, K_MAX_Q - 1) as i8;
                    }
                    let u = lat.pack_index(&laux[8 * k..8 * k + 8]);
                    let grid_index = lat.kmap[u as usize];
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let neighbours = lat.neighbours_of(grid_index);
                        iq2_find_best_neighbour(
                            neighbours,
                            &lat.grid,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            this_scale,
                            &mut laux[8 * k..8 * k + 8],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += (w * xval[i]) * q;
                    sumq2 += (w * q) * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid.copy_from_slice(&is_on_grid_aux);
                }
            }
            let mut n_not_ongrid = 0usize;
            for k in 0..2 {
                if !is_on_grid[k] {
                    n_not_ongrid += 1;
                }
            }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..2 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let li = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        let li = li.clamp(0, K_MAX_Q - 1);
                        u |= (li as u16) << (2 * i);
                        l[8 * k + i] = li as i8;
                    }
                    let grid_index = lat.kmap[u as usize];
                    if grid_index < 0 {
                        let neighbours = lat.neighbours_of(grid_index);
                        iq2_find_best_neighbour(
                            neighbours,
                            &lat.grid,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            scale,
                            &mut l[8 * k..8 * k + 8],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = 2.0 * l[i] as f32 + 1.0;
                    sumqx += (w * xval[i]) * q;
                    sumq2 += (w * q) * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..2 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }
            for k in 0..2 {
                let u = lat.pack_index(&l[8 * k..8 * k + 8]);
                let grid_index = lat.kmap[u as usize];
                assert!(
                    grid_index >= 0,
                    "iq2_xs: point {u} not on grid after quantize"
                );
                // (:3626) — u16: grid index (9 bits) | signs << 9.
                q2[2 * ib + k] = (grid_index as u16) | ((block_signs[k] as u16) << 9);
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        let id = 1.0 / d;
        for ib in 0..QK_K / 16 {
            let mut li = nearest_int(0.5 * (id * scales[ib] - 1.0));
            li = li.clamp(0, 15);
            if ib % 2 == 0 {
                ob[scales_off + ib / 2] = li as u8;
            } else {
                ob[scales_off + ib / 2] |= (li as u8) << 4;
            }
        }
        // Layout: d: f16 | qs: 32 × u16 (LE) | scales: [u8; 8].
        for (i, &v) in q2.iter().enumerate() {
            ob[qs_off + 2 * i..qs_off + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
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
    fn iq2_xxs_shape_and_weights_matter() {
        let n = 512;
        let src = synth(n, 42);
        let w1: Vec<f32> = synth(n, 7).iter().map(|v| 1.0 + v.abs()).collect();
        let mut w2 = w1.clone();
        for (i, w) in w2.iter_mut().enumerate() {
            *w += (i % 8) as f32 * 0.1;
        }
        let a = quantize_row_iq2_xxs_weighted(&src, n, &w1);
        let b = quantize_row_iq2_xxs_weighted(&src, n, &w2);
        assert_eq!(a.len(), 2 * IQ2_XXS_BLOCK_BYTES);
        assert_ne!(a, b, "weights must be consumed");
        // d is finite (nonzero block).
        let d = f16::from_le_bytes([a[0], a[1]]).to_f32();
        assert!(d.is_finite());
    }

    #[test]
    fn iq2_xs_shape_and_weights_matter() {
        let n = 512;
        let src = synth(n, 42);
        let w1: Vec<f32> = synth(n, 7).iter().map(|v| 1.0 + v.abs()).collect();
        let mut w2 = w1.clone();
        for (i, w) in w2.iter_mut().enumerate() {
            *w += (i % 16) as f32 * 0.1;
        }
        let a = quantize_row_iq2_xs_weighted(&src, n, &w1);
        let b = quantize_row_iq2_xs_weighted(&src, n, &w2);
        assert_eq!(a.len(), 2 * IQ2_XS_BLOCK_BYTES);
        assert_ne!(a, b, "weights must be consumed");
    }
}

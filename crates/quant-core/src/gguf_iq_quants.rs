//! Weighted IQ-quantizers — ports of llama.cpp's `quantize_row_iq*_impl`
//! functions (docs/ref/llama.cpp, MIT; see `ggml_iq_grid` for the lattice
//! infrastructure and `gguf_quants` for the shared K-quant helpers).
//!
//! Scope (Unsloth plan Phase 4.3):
//! - IQ slice A: iq2_xxs (:3294), iq2_xs (:3472) — 8-lane 2-bit lattices.
//! - IQ slice B: iq2_s (:5142), iq3_xxs (:3938), iq3_s (:4169).
//!
//! Family conventions that DIFFER between impls (each is faithful to
//! its upstream function — do NOT "unify" them):
//! - sigma2: iq2_xxs/iq2_xs use `sumx2/QK_K` (NO ×2); iq2_s/iq3_xxs/
//!   iq3_s use `2*sumx2/QK_K` (the K-quant convention).
//! - sign bytes: iq2_xxs/iq2_xs/iq3_xxs apply the odd-parity fixup and
//!   mask `& 127`; iq2_s/iq3_s keep the RAW sign byte (all 8 bits, no
//!   parity fixup).
//! - d fudge factors: iq2_xxs ×1.0125, iq2_s ×0.9875, iq3_xxs ×1.0125,
//!   iq3_s ×1.033 — applied to the f16 super-block scale.
//! - search steps: iq2_xxs ±6×0.1, iq2_xs/iq2_s ±9×0.1, iq3_xxs
//!   ±15×0.2, iq3_s ±9×0.2 (iq3 lanes are 4×3-bit with kMaxQ=8).
//!
//! C float associations are preserved exactly (see the K-quant module's
//! comments — `w*x*q` means `((w*x)*q)` etc.).

use half::f16;

use crate::gguf_iq_grid::{lattice, IqLattice};
use crate::gguf_quants::{make_qp_quants, nearest_int};

const QK_K: usize = 256;
/// `GROUP_MAX_EPS` (ggml-quants.c:613).
const GROUP_MAX_EPS: f32 = 1e-15f32;
/// `GROUP_MAX_EPS_IQ3_XXS` (:21) — the iq3 lattice families gate on a
/// much larger epsilon.
const GROUP_MAX_EPS_IQ3_XXS: f32 = 1e-8f32;
/// `GROUP_MAX_EPS_IQ2_S` (:22).
const GROUP_MAX_EPS_IQ2_S: f32 = 1e-8f32;

/// Bytes per block_iq2_xxs: d: f16 + qs: [u16; 32] = 66.
pub const IQ2_XXS_BLOCK_BYTES: usize = 2 + 2 * (QK_K / 8);
/// Bytes per block_iq2_xs: d: f16 + qs: [u16; 32] + scales: [u8; 8] = 74.
pub const IQ2_XS_BLOCK_BYTES: usize = 2 + 2 * (QK_K / 8) + QK_K / 32;
/// Bytes per block_iq2_s: d: f16 + qs: [u8; 64] + qh: [u8; 8] +
/// scales: [u8; 8] = 82 (ggml-common.h:396-402).
pub const IQ2_S_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 32;
/// Bytes per block_iq3_xxs: d: f16 + qs: [u8; 96] = 98 (:407-411).
pub const IQ3_XXS_BLOCK_BYTES: usize = 2 + 3 * (QK_K / 8);
/// Bytes per block_iq3_s: d: f16 + qs: [u8; 64] + qh: [u8; 8] +
/// signs: [u8; 32] + scales: [u8; 4] = 110 (:415-422).
pub const IQ3_S_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64;

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

// ─── IQ2_S (:5142-5309) ───────────────────────────────────────────────

/// Port of `quantize_row_iq2_s_impl`. Distinct from iq2_xs: sigma2 ×2,
/// RAW 8-bit signs (no parity fixup), `&127`-free sign flip, d ×0.9875,
/// and the qs/qh/signs byte packing (:5285-5288).
pub fn quantize_row_iq2_s_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let lat = lattice(IqLattice::Iq2S);
    let nbl = n_per_row / QK_K;
    let mut out = vec![0u8; nbl * IQ2_S_BLOCK_BYTES];

    const K_MAX_Q: i32 = 3;
    // Layout: d: f16 | qs: [u8; 64] | qh: [u8; 8] | scales: [u8; 8].
    // (ggml-common.h puts qs[64] then qh[8] then scales[8].)
    let (d_off, qs_off, qh_off, scales_off) = (0usize, 2, 2 + QK_K / 4, 2 + QK_K / 4 + QK_K / 32);

    let mut scales = vec![0f32; QK_K / 16];
    let mut weight = vec![0f32; 16];
    let mut xval = vec![0f32; 16];
    let mut l = vec![0i8; 16];
    let mut laux = vec![0i8; 16];
    let mut waux = vec![0f32; 16];
    let mut block_signs = [0u8; 2];

    for ibl in 0..nbl {
        let xb_block = &x[ibl * QK_K..(ibl + 1) * QK_K];
        let ob = &mut out[ibl * IQ2_S_BLOCK_BYTES..(ibl + 1) * IQ2_S_BLOCK_BYTES];
        ob.fill(0);
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let mut max_scale = 0.0f32;

        let mut sumx2 = 0.0f32;
        for &v in xb_block {
            sumx2 += v * v;
        }
        // (:5181) — WITH the factor 2 (unlike xxs/xs!).
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for ib in 0..QK_K / 16 {
            let xb = &xb_block[16 * ib..16 * ib + 16];
            let qw = quant_weights.map(|w| &w[QK_K * ibl + 16 * ib..QK_K * ibl + 16 * ib + 16]);
            for i in 0..16 {
                weight[i] = match qw {
                    Some(q) => q[i] * (sigma2 + xb[i] * xb[i]).sqrt(),
                    // (:5189) — unweighted formula differs from iq3's x^2!
                    None => 0.25 * sigma2 + xb[i] * xb[i],
                };
            }
            for i in 0..16 {
                waux[i] = weight[i].sqrt();
            }
            // (:5192-5201) — NO parity fixup: raw sign byte, all 8 bits.
            for k in 0..2 {
                let mut s: u8 = 0;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        s |= 1 << i;
                    }
                }
                block_signs[k] = s;
            }
            let mut max = xval[0];
            for &v in &xval[1..16] {
                max = max.max(v);
            }
            l.fill(0);
            if max < GROUP_MAX_EPS_IQ2_S {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2.0 * K_MAX_Q as f32 - 1.0);
            let mut is_on_grid = [true; 2];
            let mut is_on_grid_aux = [false; 2];
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
                    is_on_grid = is_on_grid_aux;
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
                // (:5273) — ~block_signs[k] WITHOUT the &127 mask.
                for k in 0..2 {
                    block_signs[k] = !block_signs[k];
                }
            }
            for k in 0..2 {
                let u = lat.pack_index(&l[8 * k..8 * k + 8]);
                let grid_index = lat.kmap[u as usize];
                assert!(
                    grid_index >= 0,
                    "iq2_s: point {u} not on grid after quantize"
                );
                // (:5285-5288) — qs: index low 8 bits at [2ib+k];
                // qh: bit 8+ packed 2 per byte; signs at qs[QK_K/8 + i8].
                let i8_idx = 2 * ib + k;
                ob[qs_off + i8_idx] = (grid_index as u16 & 255) as u8;
                ob[qh_off + i8_idx / 4] |= (((grid_index as u16) >> 8) as u8) << (2 * (i8_idx % 4));
                ob[qs_off + QK_K / 8 + i8_idx] = block_signs[k];
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        // (:5300) — iq2_s fudge factor ×0.9875.
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(d * 0.9875).to_le_bytes());
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
    }
    out
}

// ─── IQ3_XXS (:3938-4150) ──────────────────────────────────────────────

/// Port of `quantize_row_iq3_xxs_impl` (grid 256). 4 lanes × 3 bits,
/// kMaxQ = 8, ±15×0.2 search, sign-parity fixup `& 127`, d ×1.0125.
/// Accepts `None` weights because upstream's `*_ref` calls this same
/// impl with NULL (:4163-4166).
pub fn quantize_row_iq3_xxs_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let lat = lattice(IqLattice::Iq3Xxs);
    let nbl = n_per_row / QK_K;
    let mut out = vec![0u8; nbl * IQ3_XXS_BLOCK_BYTES];

    const K_MAX_Q: i32 = 8;
    // Layout: d: f16 | qs: [u8; 96] — grid index bytes 8 per group.
    let (d_off, qs_off) = (0usize, 2);

    let mut scales = vec![0f32; QK_K / 32];
    let mut weight = vec![0f32; 32];
    let mut xval = vec![0f32; 32];
    let mut l = vec![0i8; 32];
    let mut laux = vec![0i8; 32];
    let mut waux = vec![0f32; 32];
    let mut is_on_grid_aux = [false; 8];
    let mut block_signs = [0u8; 4];

    for ibl in 0..nbl {
        let xb_block = &x[ibl * QK_K..(ibl + 1) * QK_K];
        let ob = &mut out[ibl * IQ3_XXS_BLOCK_BYTES..(ibl + 1) * IQ3_XXS_BLOCK_BYTES];
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
        ob[qs_off..qs_off + 3 * (QK_K / 8)].fill(0);
        let mut max_scale = 0.0f32;

        let mut sumx2 = 0.0f32;
        for &v in xb_block {
            sumx2 += v * v;
        }
        // (:3996) — WITH the factor 2.
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for ib in 0..QK_K / 32 {
            let xb = &xb_block[32 * ib..32 * ib + 32];
            let qw = quant_weights.map(|w| &w[QK_K * ibl + 32 * ib..QK_K * ibl + 32 * ib + 32]);
            for i in 0..32 {
                weight[i] = match qw {
                    Some(q) => q[i] * (sigma2 + xb[i] * xb[i]).sqrt(),
                    None => xb[i] * xb[i],
                };
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
            l.fill(0);
            if max < GROUP_MAX_EPS_IQ3_XXS {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2.0 * K_MAX_Q as f32 - 1.0);
            // Per-group state (C resets it per ib iteration too).
            let mut is_on_grid = [true; 8];
            for is in -15i32..=15 {
                let id = (2.0 * K_MAX_Q as f32 - 1.0 + 0.2 * is as f32) / max;
                let this_scale = 1.0 / id;
                for k in 0..8 {
                    for i in 0..4 {
                        let li = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        laux[4 * k + i] = li.clamp(0, K_MAX_Q - 1) as i8;
                    }
                    let u = lat.pack_index(&laux[4 * k..4 * k + 4]);
                    let grid_index = lat.kmap[u as usize];
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let neighbours = lat.neighbours_of(grid_index);
                        iq3_find_best_neighbour(
                            neighbours,
                            &lat.grid,
                            &xval[4 * k..4 * k + 4],
                            &waux[4 * k..4 * k + 4],
                            this_scale,
                            &mut laux[4 * k..4 * k + 4],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += (w * xval[i]) * q;
                    sumq2 += (w * q) * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid = is_on_grid_aux;
                }
            }
            let mut n_not_ongrid = 0usize;
            for k in 0..8 {
                if !is_on_grid[k] {
                    n_not_ongrid += 1;
                }
            }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..8 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let li = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        let li = li.clamp(0, K_MAX_Q - 1);
                        u |= (li as u16) << (3 * i);
                        l[4 * k + i] = li as i8;
                    }
                    let grid_index = lat.kmap[u as usize];
                    let grid_index = if grid_index < 0 {
                        let neighbours = lat.neighbours_of(grid_index);
                        iq3_find_best_neighbour(
                            neighbours,
                            &lat.grid,
                            &xval[4 * k..4 * k + 4],
                            &waux[4 * k..4 * k + 4],
                            scale,
                            &mut l[4 * k..4 * k + 4],
                        )
                    } else {
                        grid_index as usize
                    };
                    // (:4087-4088) — refit writes the grid lanes too.
                    let pg = lat.grid[grid_index].to_le_bytes();
                    for i in 0..4 {
                        l[4 * k + i] = ((pg[i] as i8) - 1) / 2;
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
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }
            for k in 0..8 {
                let u = lat.pack_index(&l[4 * k..4 * k + 4]);
                let grid_index = lat.kmap[u as usize];
                assert!(
                    grid_index >= 0,
                    "iq3_xxs: point {u} not on grid after quantize"
                );
                // (:4115-4116) — grid 256: full index fits one byte.
                ob[qs_off + 8 * ib + k] = grid_index as u8;
            }
            // (:4123) — signs packed 4×7-bit into a u32. Buffer layout
            // (:3982-3984): q3 = 64 index bytes (QK_K/4) THEN 8 sign u32s
            // (QK_K/32) — scales_and_signs aliases q3 + QK_K/4, and the
            // scale nibbles OR into the same u32s at :4142.
            let signs_off = qs_off + QK_K / 4;
            let v: u32 = block_signs[0] as u32
                | ((block_signs[1] as u32) << 7)
                | ((block_signs[2] as u32) << 14)
                | ((block_signs[3] as u32) << 21);
            ob[signs_off + 4 * ib..signs_off + 4 * ib + 4].copy_from_slice(&v.to_le_bytes());
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        // (:4137) — fudge ×1.0125.
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(d * 1.0125).to_le_bytes());
        let id = 1.0 / d;
        for ib in 0..QK_K / 32 {
            let mut li = nearest_int(0.5 * (id * scales[ib] - 1.0));
            li = li.clamp(0, 15);
            // (:4142) — scale nibble ORs into the same u32 as the signs.
            let signs_off = qs_off + QK_K / 4;
            let cur = u32::from_le_bytes(
                ob[signs_off + 4 * ib..signs_off + 4 * ib + 4]
                    .try_into()
                    .unwrap(),
            );
            let v = cur | ((li as u32) << 28);
            ob[signs_off + 4 * ib..signs_off + 4 * ib + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Port of `iq3_find_best_neighbour` (:3914-3936) — the 4-lane variant.
fn iq3_find_best_neighbour(
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
        for i in 0..4 {
            let q = pg[i] as i8 as f32;
            let diff = scale * q - xval[i];
            // (:3922-3923) — (weight*diff)*diff.
            d2 += (weight[i] * diff) * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = neighbours[j] as isize;
        }
    }
    debug_assert!(grid_index >= 0);
    let pg = grid[grid_index as usize].to_le_bytes();
    for i in 0..4 {
        l[i] = ((pg[i] as i8) - 1) / 2;
    }
    grid_index as usize
}

// ─── IQ3_S (:4169-4350) ────────────────────────────────────────────────

/// Port of `quantize_row_iq3_s_impl` (block_size 32, grid 512).
/// Distinctive points vs iq3_xxs:
/// - RAW 8-bit sign bytes, NO parity fixup (:4227-4236);
/// - zero gate is `!max` — NOT an epsilon (:4240);
/// - `is_on_grid` starts FALSE (:4246) and the refit loop does NOT skip
///   on-grid sub-groups (the `continue` is commented out upstream :4283)
///   — every sub-group is refit and L overwritten from the grid row
///   (:4295-4296);
/// - d fudge ×1.033 (:4339); scales packed as nibble pairs (:4346);
/// - signs stored as their own byte array (:4327), qh one bit per
///   sub-group (:4324).
pub fn quantize_row_iq3_s_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let lat = lattice(IqLattice::Iq3S);
    let nbl = n_per_row / QK_K;
    let mut out = vec![0u8; nbl * IQ3_S_BLOCK_BYTES];

    const K_MAX_Q: i32 = 8;
    const BLOCK_SIZE: usize = 32; // IQ3S_BLOCK_SIZE (:4352)
    const BS4: usize = BLOCK_SIZE / 4; // 8
    const BS8: usize = BLOCK_SIZE / 8; // 4

    // Layout: d: f16 | qs: [u8; 64] | qh: [u8; 8] | signs: [u8; 32] |
    // scales: [u8; 4].
    let (d_off, qs_off, qh_off, signs_off, scales_off) = (
        0usize,
        2,
        2 + QK_K / 4,
        2 + QK_K / 4 + QK_K / 32,
        2 + QK_K / 4 + QK_K / 32 + QK_K / 8,
    );

    let mut scales = vec![0f32; QK_K / BLOCK_SIZE];
    let mut weight = vec![0f32; BLOCK_SIZE];
    let mut xval = vec![0f32; BLOCK_SIZE];
    let mut l = vec![0i8; BLOCK_SIZE];
    let mut laux = vec![0i8; BLOCK_SIZE];
    let mut waux = vec![0f32; BLOCK_SIZE];
    let mut is_on_grid_aux = [false; BS4];
    let mut block_signs = [0u8; BS8];

    for ibl in 0..nbl {
        let xb_block = &x[ibl * QK_K..(ibl + 1) * QK_K];
        let ob = &mut out[ibl * IQ3_S_BLOCK_BYTES..(ibl + 1) * IQ3_S_BLOCK_BYTES];
        ob.fill(0);
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let mut max_scale = 0.0f32;

        let mut sumx2 = 0.0f32;
        for &v in xb_block {
            sumx2 += v * v;
        }
        // (:4216) — WITH the factor 2.
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for ib in 0..QK_K / BLOCK_SIZE {
            let xb = &xb_block[BLOCK_SIZE * ib..BLOCK_SIZE * ib + BLOCK_SIZE];
            let qw = quant_weights.map(|w| {
                &w[QK_K * ibl + BLOCK_SIZE * ib..QK_K * ibl + BLOCK_SIZE * ib + BLOCK_SIZE]
            });
            for i in 0..BLOCK_SIZE {
                weight[i] = match qw {
                    Some(q) => q[i] * (sigma2 + xb[i] * xb[i]).sqrt(),
                    None => xb[i] * xb[i],
                };
            }
            for i in 0..BLOCK_SIZE {
                waux[i] = weight[i].sqrt();
            }
            // (:4227-4236) — NO parity fixup; raw 8-bit sign bytes.
            for k in 0..BS8 {
                let mut s: u8 = 0;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        s |= 1 << i;
                    }
                }
                block_signs[k] = s;
            }
            let mut max = xval[0];
            for &v in &xval[1..BLOCK_SIZE] {
                max = max.max(v);
            }
            l.fill(0);
            if max == 0.0 {
                // (:4240-4243) — plain zero test, no epsilon.
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2.0 * K_MAX_Q as f32 - 1.0);
            // Per-group state (C resets it per ib iteration too).
            let mut is_on_grid = [false; BS4];
            for is in -9i32..=9 {
                let id = (2.0 * K_MAX_Q as f32 - 1.0 + 0.2 * is as f32) / max;
                let this_scale = 1.0 / id;
                for k in 0..BS4 {
                    for i in 0..4 {
                        let li = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        laux[4 * k + i] = li.clamp(0, K_MAX_Q - 1) as i8;
                    }
                    let u = lat.pack_index(&laux[4 * k..4 * k + 4]);
                    let grid_index = lat.kmap[u as usize];
                    is_on_grid_aux[k] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let neighbours = lat.neighbours_of(grid_index);
                        iq3_find_best_neighbour(
                            neighbours,
                            &lat.grid,
                            &xval[4 * k..4 * k + 4],
                            &waux[4 * k..4 * k + 4],
                            this_scale,
                            &mut laux[4 * k..4 * k + 4],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..BLOCK_SIZE {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += (w * xval[i]) * q;
                    sumq2 += (w * q) * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid = is_on_grid_aux;
                }
            }
            let mut n_not_ongrid = 0usize;
            for k in 0..BS4 {
                if !is_on_grid[k] {
                    n_not_ongrid += 1;
                }
            }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..BS4 {
                    // (:4283) — the `if (is_on_grid[k]) continue;` is
                    // COMMENTED OUT upstream: every sub-group is refit.
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let li = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        let li = li.clamp(0, K_MAX_Q - 1);
                        u |= (li as u16) << (3 * i);
                    }
                    let grid_index = lat.kmap[u as usize];
                    let grid_index = if grid_index < 0 {
                        let neighbours = lat.neighbours_of(grid_index);
                        iq3_find_best_neighbour(
                            neighbours,
                            &lat.grid,
                            &xval[4 * k..4 * k + 4],
                            &waux[4 * k..4 * k + 4],
                            scale,
                            &mut l[4 * k..4 * k + 4],
                        )
                    } else {
                        grid_index as usize
                    };
                    // (:4295-4296) — L overwritten from the grid row.
                    let pg = lat.grid[grid_index].to_le_bytes();
                    for i in 0..4 {
                        l[4 * k + i] = ((pg[i] as i8) - 1) / 2;
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..BLOCK_SIZE {
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
                // (:4311) — ~block_signs[k], no mask.
                for k in 0..BS8 {
                    block_signs[k] = !block_signs[k];
                }
            }
            // Pack: qs low bytes + qh high bits (:4323-4324), signs
            // (:4327). qs/qh/signs pointers advance per group (:4326-4328).
            for k in 0..BS4 {
                let u = lat.pack_index(&l[4 * k..4 * k + 4]);
                let grid_index = lat.kmap[u as usize];
                assert!(
                    grid_index >= 0,
                    "iq3_s: point {u} not on grid after quantize"
                );
                let gsub = ib * BS4 + k;
                ob[qs_off + gsub] = (grid_index as u16 & 255) as u8;
                ob[qh_off + gsub / 8] |= (((grid_index as u16) >> 8) as u8) << (gsub % 8);
            }
            for k in 0..BS8 {
                ob[signs_off + ib * BS8 + k] = block_signs[k];
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        // (:4339) — fudge ×1.033.
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(d * 1.033).to_le_bytes());
        let id = 1.0 / d;
        // (:4341-4346) — nibble pairs.
        let mut ib = 0usize;
        while ib < QK_K / BLOCK_SIZE {
            let mut l1 = nearest_int(0.5 * (id * scales[ib] - 1.0));
            l1 = l1.clamp(0, 15);
            let mut l2 = nearest_int(0.5 * (id * scales[ib + 1] - 1.0));
            l2 = l2.clamp(0, 15);
            ob[scales_off + ib / 2] = (l1 | (l2 << 4)) as u8;
            ib += 2;
        }
    }
    out
}

// ─── IQ1 family helpers ───────────────────────────────────────────────

/// `GROUP_MAX_EPS_IQ1_S` (:24).
const GROUP_MAX_EPS_IQ1_S: f32 = 1e-12f32;
/// `GROUP_MAX_EPS_IQ1_M` (:23).
const GROUP_MAX_EPS_IQ1_M: f32 = 1e-7f32;
/// `IQ1S_DELTA` / `IQ1M_DELTA` (ggml-common.h:1132-1133) = 0.125.
const IQ1_DELTA: f32 = 0.125f32;
/// `IQ1S_BLOCK_SIZE` (:4506).
const IQ1S_BLOCK_SIZE: usize = 32;
/// `IQ1M_BLOCK_SIZE` (:4507).
const IQ1M_BLOCK_SIZE: usize = 16;

/// Port of `iq1_find_best_neighbour2` (:4443-4498) — like the iq2
/// variant but the distance is computed through the 3-value grid table
/// `xg` (x_p or x_m), and if no neighbour list entry exists the search
/// falls back to the ENTIRE grid (:4463-4478).
#[allow(clippy::too_many_arguments)] // mirrors the C signature verbatim
fn iq1_find_best_neighbour2(
    neighbours: &[u16],
    grid: &[u64],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    xg: &[f32; 3],
    l: &mut [i8],
    ngrid: usize,
) -> usize {
    let num_neighbors = neighbours[0] as usize;
    debug_assert!(num_neighbors > 0);
    let mut best_score = f32::INFINITY;
    let mut grid_index: isize = -1;
    for j in 1..=num_neighbors {
        let pg = grid[neighbours[j] as usize].to_le_bytes();
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = xg[((pg[i] as i8 - 1) / 2) as usize];
            let w = weight[i];
            let diff = scale * q - xval[i];
            // (:4456) — (w*diff)*diff.
            d2 += (w * diff) * diff;
        }
        if d2 < best_score {
            best_score = d2;
            grid_index = neighbours[j] as isize;
        }
    }
    if grid_index < 0 {
        // Full-grid fallback (:4464-4478).
        for i in 0..ngrid {
            let gi = grid[i].to_le_bytes();
            let mut d2 = 0.0f32;
            for j in 0..8 {
                let w = weight[j];
                let q = xg[((gi[j] as i8 - 1) / 2) as usize];
                let diff = scale * q - xval[i];
                d2 += (w * diff) * diff;
            }
            if d2 < best_score {
                best_score = d2;
                grid_index = i as isize;
            }
        }
    }
    assert!(grid_index >= 0, "iq1: no grid point found");
    let pg = grid[grid_index as usize].to_le_bytes();
    for i in 0..8 {
        l[i] = ((pg[i] as i8) - 1) / 2;
    }
    grid_index as usize
}

// ─── IQ1_S (:4508-4670) ────────────────────────────────────────────────

/// Bytes per block_iq1_s: d: f16 + qs: [u8; 32] + qh: [u16; 8] = 50
/// (ggml-common.h:425-430: sizeof == 2 + QK_K/8 + QK_K/16).
pub const IQ1_S_BLOCK_BYTES: usize = 2 + QK_K / 8 + 2 * (QK_K / 32);

/// Port of `quantize_row_iq1_s_impl` (:4508-4670). 1.5625 bpw: ternary
/// (-1, 0, 1) ± delta with an exhaustive 2-boundary split search over
/// the sorted block (:4567-4603), then the 2048-point lattice snap.
pub fn quantize_row_iq1_s_weighted(x: &[f32], n_per_row: usize, quant_weights: &[f32]) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let lat = lattice(IqLattice::Iq1);
    let nbl = n_per_row / QK_K;
    let mut out = vec![0u8; nbl * IQ1_S_BLOCK_BYTES];

    let bs = IQ1S_BLOCK_SIZE;
    // (:4536-4537) — the ±delta quant value tables.
    let x_p = [-1.0 + IQ1_DELTA, IQ1_DELTA, 1.0 + IQ1_DELTA];
    let x_m = [-1.0 - IQ1_DELTA, -IQ1_DELTA, 1.0 - IQ1_DELTA];

    // Layout: d: f16 | qs: [u8; 32] | qh: [u16; 8] (LE).
    let (d_off, qs_off, qh_off) = (0usize, 2, 2 + QK_K / 8);

    let mut scales = vec![0f32; QK_K / bs];
    let mut weight = vec![0f32; bs];
    let mut l = vec![1i8; bs];
    let mut index = vec![0u16; bs / 8];
    let mut shifts = vec![0i8; QK_K / bs];

    for ibl in 0..nbl {
        let xb_block = &x[ibl * QK_K..(ibl + 1) * QK_K];
        let ob = &mut out[ibl * IQ1_S_BLOCK_BYTES..(ibl + 1) * IQ1_S_BLOCK_BYTES];
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
        ob[qs_off..qs_off + QK_K / 8].fill(0);
        for h in 0..QK_K / bs {
            ob[qh_off + 2 * h..qh_off + 2 * h + 2].fill(0);
        }
        let mut max_scale = 0.0f32;

        let mut sumx2 = 0.0f32;
        for &v in xb_block {
            sumx2 += v * v;
        }
        // (:4553) — WITH the factor 2.
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for ib in 0..QK_K / bs {
            let xb = &xb_block[bs * ib..bs * ib + bs];
            let qw = &quant_weights[QK_K * ibl + bs * ib..QK_K * ibl + bs * ib + bs];
            for i in 0..bs {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
            }
            let mut max = xb[0].abs();
            for &v in &xb[1..bs] {
                max = max.max(v.abs());
            }
            if max < GROUP_MAX_EPS_IQ1_S {
                scales[ib] = 0.0;
                shifts[ib] = 1;
                l.fill(1);
                continue;
            }
            // Sort (value, index) pairs ascending by value — C qsorts a
            // float array with the index packed after each float
            // (:4573-4577). Rust: sort indices by value with a stable
            // sort (qsort is unstable BUT the comparator only looks at
            // the value; ties keep qsort's arbitrary order. Ties in
            // uniform-noise fixtures are measure-zero; parity confirmed
            // against goldens).
            let mut idx: Vec<usize> = (0..bs).collect();
            idx.sort_by(|&a, &b| {
                xb[a]
                    .partial_cmp(&xb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Prefix sums over the sorted order (:4578-4585).
            let mut sumx = vec![0f32; bs + 1];
            let mut sumw = vec![0f32; bs + 1];
            for j in 0..bs {
                let i = idx[j];
                sumx[j + 1] = sumx[j] + (weight[i] * xb[i]);
                sumw[j + 1] = sumw[j] + weight[i];
            }
            // Exhaustive boundary search (:4586-4603).
            let mut best_score = f32::MIN;
            let mut scale = max;
            let mut besti1: isize = -1;
            let mut besti2: isize = -1;
            let mut best_shift = 0i8;
            for i1 in 0..=bs {
                for i2 in i1..=bs {
                    let mut sumqx = (sumx[i1] - sumx[0]) * x_p[0]
                        + (sumx[i2] - sumx[i1]) * x_p[1]
                        + (sumx[bs] - sumx[i2]) * x_p[2];
                    let mut sumq2 = (sumw[i1] - sumw[0]) * (x_p[0] * x_p[0])
                        + (sumw[i2] - sumw[i1]) * (x_p[1] * x_p[1])
                        + (sumw[bs] - sumw[i2]) * (x_p[2] * x_p[2]);
                    if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        besti1 = i1 as isize;
                        besti2 = i2 as isize;
                        best_shift = 1;
                    }
                    sumqx = (sumx[i1] - sumx[0]) * x_m[0]
                        + (sumx[i2] - sumx[i1]) * x_m[1]
                        + (sumx[bs] - sumx[i2]) * x_m[2];
                    sumq2 = (sumw[i1] - sumw[0]) * (x_m[0] * x_m[0])
                        + (sumw[i2] - sumw[i1]) * (x_m[1] * x_m[1])
                        + (sumw[bs] - sumw[i2]) * (x_m[2] * x_m[2]);
                    if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        besti1 = i1 as isize;
                        besti2 = i2 as isize;
                        best_shift = -1;
                    }
                }
            }
            if besti1 < 0 || besti2 < 0 || best_shift == 0 {
                scales[ib] = 0.0;
                shifts[ib] = 1;
                l.fill(1);
                continue;
            }
            let (bi1, bi2) = (besti1 as usize, besti2 as usize);
            for j in 0..bi1 {
                l[idx[j]] = 0;
            }
            for j in bi1..bi2 {
                l[idx[j]] = 1;
            }
            for j in bi2..bs {
                l[idx[j]] = 2;
            }
            if scale < 0.0 {
                // (:4613-4616) — flip L and the shift.
                for j in 0..bs {
                    l[j] = 2 - l[j];
                }
                scale = -scale;
                best_shift = -best_shift;
            }
            let mut all_on_grid = true;
            let xx: &[f32; 3] = if best_shift == 1 { &x_p } else { &x_m };
            for k in 0..bs / 8 {
                let u = lat.pack_index(&l[8 * k..8 * k + 8]);
                let grid_index = lat.kmap[u as usize];
                let grid_index = if grid_index < 0 {
                    all_on_grid = false;
                    let neighbours = lat.neighbours_of(grid_index);
                    let gi = iq1_find_best_neighbour2(
                        neighbours,
                        &lat.grid,
                        &xb[8 * k..8 * k + 8],
                        &weight[8 * k..8 * k + 8],
                        scale,
                        xx,
                        &mut l[8 * k..8 * k + 8],
                        lat.grid.len(),
                    );
                    debug_assert!(gi as isize >= 0);
                    gi
                } else {
                    grid_index as usize
                };
                index[k] = grid_index as u16;
            }
            if !all_on_grid {
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for k in 0..bs / 8 {
                    let pg = lat.grid[index[k] as usize].to_le_bytes();
                    for j in 0..8 {
                        let w = weight[8 * k + j];
                        let q = xx[((pg[j] as i8 - 1) / 2) as usize];
                        // (:4638-4639) — ((w*q)*x), ((w*q)*q).
                        sumqx += (w * q) * xb[8 * k + j];
                        sumq2 += (w * q) * q;
                    }
                }
                if sumqx > 0.0 && sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            // (:4644-4649) — qs low bytes; qh u16 packs the high bits.
            let mut h: u16 = 0;
            for k in 0..bs / 8 {
                ob[qs_off + (bs / 8) * ib + k] = (index[k] & 255) as u8;
                h |= (index[k] >> 8) << (3 * k);
            }
            ob[qh_off + 2 * ib..qh_off + 2 * ib + 2].copy_from_slice(&h.to_le_bytes());
            scales[ib] = scale;
            shifts[ib] = best_shift;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 15.0;
        // (:4661) — the 1.125 fudge ("Don't ask me why it is needed.").
        ob[d_off..d_off + 2].copy_from_slice(&f16::from_f32(d * 1.125).to_le_bytes());
        let id = 1.0 / d;
        for ib in 0..QK_K / bs {
            let mut li = nearest_int(0.5 * (id * scales[ib] - 1.0));
            li = li.clamp(0, 7);
            if shifts[ib] == -1 {
                li |= 8;
            }
            // (:4667) — OR into the qh u16 at bit 12.
            let cur = u16::from_le_bytes([ob[qh_off + 2 * ib], ob[qh_off + 2 * ib + 1]]);
            let v = cur | ((li as u16) << 12);
            ob[qh_off + 2 * ib..qh_off + 2 * ib + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

// ─── IQ1_M (:4692-4944) ────────────────────────────────────────────────

/// Bytes per block_iq1_m: qs: [u8; 32] + qh: [u8; 16] + scales: [u8; 8]
/// = 56 (ggml-common.h:433-438; NO d field — the scale hides in
/// scales nibbles, :4938-4942).
pub const IQ1_M_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 16 + QK_K / 32;

/// Port of `quantize_row_iq1_m_impl` (:4692-4944). 1.75 bpw. The
/// super-block scale is NOT a d field — it is scattered across the
/// 4 scales u16s' top nibbles (:4939-4942). Block size 16, 4 quadrant
/// sign combinations searched per split (:4770-4849).
pub fn quantize_row_iq1_m_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let lat = lattice(IqLattice::Iq1);
    let nbl = n_per_row / QK_K;
    let mut out = vec![0u8; nbl * IQ1_M_BLOCK_BYTES];

    let bs = IQ1M_BLOCK_SIZE;
    let x_p = [-1.0 + IQ1_DELTA, IQ1_DELTA, 1.0 + IQ1_DELTA];
    let x_m = [-1.0 - IQ1_DELTA, -IQ1_DELTA, 1.0 - IQ1_DELTA];
    // (:4720) — quadrant masks into qh.
    let masks: [u8; 4] = [0x00, 0x80, 0x08, 0x88];

    // Layout: qs: [u8; 32] | qh: [u8; 16] | scales: [u8; 8].
    let (qs_off, qh_off, scales_off) = (0usize, QK_K / 8, QK_K / 8 + QK_K / 16);

    let mut scales = vec![0f32; QK_K / bs];
    let mut weight = vec![0f32; bs];
    let mut l = vec![1i8; bs];
    let mut index = vec![0u16; bs / 8];
    let mut shifts = vec![0i8; QK_K / bs];

    for ibl in 0..nbl {
        let xb_block = &x[ibl * QK_K..(ibl + 1) * QK_K];
        let ob = &mut out[ibl * IQ1_M_BLOCK_BYTES..(ibl + 1) * IQ1_M_BLOCK_BYTES];
        ob[qs_off..qs_off + QK_K / 8].fill(0);
        ob[qh_off..qh_off + QK_K / 16].fill(0);
        ob[scales_off..scales_off + QK_K / 32].fill(0);
        let mut max_scale = 0.0f32;

        let mut sumx2 = 0.0f32;
        for &v in xb_block {
            sumx2 += v * v;
        }
        // (:4739) — WITH the factor 2.
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for ib in 0..QK_K / bs {
            let xb = &xb_block[bs * ib..bs * ib + bs];
            let qw = quant_weights.map(|w| &w[QK_K * ibl + bs * ib..QK_K * ibl + bs * ib + bs]);
            for i in 0..bs {
                weight[i] = match qw {
                    Some(q) => q[i] * (sigma2 + xb[i] * xb[i]).sqrt(),
                    None => xb[i] * xb[i],
                };
            }
            let mut max = xb[0].abs();
            for &v in &xb[1..bs] {
                max = max.max(v.abs());
            }
            if max < GROUP_MAX_EPS_IQ1_M {
                scales[ib] = 0.0;
                shifts[ib] = 0;
                l.fill(1);
                continue;
            }
            // Sorted boundary search — 4 sign-quadrant combinations
            // (:4763-4851). Quadrant k: 0 (+,+) 1 (+,-) 2 (-,+) 3 (-,-)
            // where the FIRST half (i < block_size/2) picks x_p vs x_m
            // by parity of k's high bit and the SECOND half by the low
            // bit — see the four sumqx[k] accumulations.
            let mut idx: Vec<usize> = (0..bs).collect();
            idx.sort_by(|&a, &b| {
                xb[a]
                    .partial_cmp(&xb[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut best_score = f32::MIN;
            let mut scale = max;
            let mut besti1: isize = -1;
            let mut besti2: isize = -1;
            let mut best_k: isize = -1;
            for i1 in 0..=bs {
                for i2 in i1..=bs {
                    let mut sumqx = [0f32; 4];
                    let mut sumq2 = [0f32; 4];
                    // Segment [0, i1) — L value 0.
                    for &i in &idx[..i1] {
                        let a = if i < bs / 2 { 0 } else { 1 };
                        sumqx[0] += (weight[i] * x_p[0]) * xb[i];
                        sumqx[1] += (weight[i] * (if a == 0 { x_p[0] } else { x_m[0] })) * xb[i];
                        sumqx[2] += (weight[i] * (if a == 0 { x_m[0] } else { x_p[0] })) * xb[i];
                        sumqx[3] += (weight[i] * x_m[0]) * xb[i];
                        sumq2[0] += (weight[i] * x_p[0]) * x_p[0];
                        sumq2[1] += (weight[i] * (if a == 0 { x_p[0] } else { x_m[0] }))
                            * if a == 0 { x_p[0] } else { x_m[0] };
                        sumq2[2] += (weight[i] * (if a == 0 { x_m[0] } else { x_p[0] }))
                            * if a == 0 { x_m[0] } else { x_p[0] };
                        sumq2[3] += (weight[i] * x_m[0]) * x_m[0];
                    }
                    // Segment [i1, i2) — L value 1.
                    for &i in &idx[i1..i2] {
                        let a = if i < bs / 2 { 0 } else { 1 };
                        sumqx[0] += (weight[i] * x_p[1]) * xb[i];
                        sumqx[1] += (weight[i] * (if a == 0 { x_p[1] } else { x_m[1] })) * xb[i];
                        sumqx[2] += (weight[i] * (if a == 0 { x_m[1] } else { x_p[1] })) * xb[i];
                        sumqx[3] += (weight[i] * x_m[1]) * xb[i];
                        sumq2[0] += (weight[i] * x_p[1]) * x_p[1];
                        sumq2[1] += (weight[i] * (if a == 0 { x_p[1] } else { x_m[1] }))
                            * if a == 0 { x_p[1] } else { x_m[1] };
                        sumq2[2] += (weight[i] * (if a == 0 { x_m[1] } else { x_p[1] }))
                            * if a == 0 { x_m[1] } else { x_p[1] };
                        sumq2[3] += (weight[i] * x_m[1]) * x_m[1];
                    }
                    // Segment [i2, bs) — L value 2.
                    for &i in &idx[i2..bs] {
                        let a = if i < bs / 2 { 0 } else { 1 };
                        sumqx[0] += (weight[i] * x_p[2]) * xb[i];
                        sumqx[1] += (weight[i] * (if a == 0 { x_p[2] } else { x_m[2] })) * xb[i];
                        sumqx[2] += (weight[i] * (if a == 0 { x_m[2] } else { x_p[2] })) * xb[i];
                        sumqx[3] += (weight[i] * x_m[2]) * xb[i];
                        sumq2[0] += (weight[i] * x_p[2]) * x_p[2];
                        sumq2[1] += (weight[i] * (if a == 0 { x_p[2] } else { x_m[2] }))
                            * if a == 0 { x_p[2] } else { x_m[2] };
                        sumq2[2] += (weight[i] * (if a == 0 { x_m[2] } else { x_p[2] }))
                            * if a == 0 { x_m[2] } else { x_p[2] };
                        sumq2[3] += (weight[i] * x_m[2]) * x_m[2];
                    }
                    for k in 0..4 {
                        if sumq2[k] > 0.0 && sumqx[k] * sumqx[k] > best_score * sumq2[k] {
                            scale = sumqx[k] / sumq2[k];
                            best_score = scale * sumqx[k];
                            besti1 = i1 as isize;
                            besti2 = i2 as isize;
                            best_k = k as isize;
                        }
                    }
                }
            }
            if besti1 < 0 || besti2 < 0 || best_k < 0 {
                scales[ib] = 0.0;
                shifts[ib] = 0;
                l.fill(1);
                continue;
            }
            let (bi1, bi2) = (besti1 as usize, besti2 as usize);
            for j in 0..bi1 {
                l[idx[j]] = 0;
            }
            for j in bi1..bi2 {
                l[idx[j]] = 1;
            }
            for j in bi2..bs {
                l[idx[j]] = 2;
            }
            if scale < 0.0 {
                for j in 0..bs {
                    l[j] = 2 - l[j];
                }
                scale = -scale;
                // (:4864) — quadrant flip 0<->3, 1<->2.
                best_k = match best_k {
                    0 => 3,
                    1 => 2,
                    2 => 1,
                    _ => 0,
                };
            }
            let mut all_on_grid = true;
            for k in 0..bs / 8 {
                // (:4868-4869) — per-half-block sign table by quadrant.
                let xx: &[f32; 3] = if k == 0 {
                    if best_k < 2 {
                        &x_p
                    } else {
                        &x_m
                    }
                } else if best_k % 2 == 0 {
                    &x_p
                } else {
                    &x_m
                };
                let u = lat.pack_index(&l[8 * k..8 * k + 8]);
                let grid_index = lat.kmap[u as usize];
                let grid_index = if grid_index < 0 {
                    all_on_grid = false;
                    let neighbours = lat.neighbours_of(grid_index);
                    let gi = iq1_find_best_neighbour2(
                        neighbours,
                        &lat.grid,
                        &xb[8 * k..8 * k + 8],
                        &weight[8 * k..8 * k + 8],
                        scale,
                        xx,
                        &mut l[8 * k..8 * k + 8],
                        lat.grid.len(),
                    );
                    debug_assert!(gi as isize >= 0);
                    gi
                } else {
                    grid_index as usize
                };
                index[k] = grid_index as u16;
            }
            if !all_on_grid {
                let mut sumqx_f = 0.0f32;
                let mut sumq2_f = 0.0f32;
                for k in 0..bs / 8 {
                    let xx: &[f32; 3] = if k == 0 {
                        if best_k < 2 {
                            &x_p
                        } else {
                            &x_m
                        }
                    } else if best_k % 2 == 0 {
                        &x_p
                    } else {
                        &x_m
                    };
                    let pg = lat.grid[index[k] as usize].to_le_bytes();
                    for j in 0..8 {
                        let w = weight[8 * k + j];
                        let q = xx[((pg[j] as i8 - 1) / 2) as usize];
                        sumqx_f += (w * q) * xb[8 * k + j];
                        sumq2_f += (w * q) * q;
                    }
                }
                if sumqx_f > 0.0 && sumq2_f > 0.0 {
                    scale = sumqx_f / sumq2_f;
                }
            }
            // (:4896-4898) — two grid indices per 16-block.
            ob[qs_off + 2 * ib] = (index[0] & 255) as u8;
            ob[qs_off + 2 * ib + 1] = (index[1] & 255) as u8;
            ob[qh_off + ib] = ((index[0] >> 8) | ((index[1] >> 8) << 4)) as u8;
            scales[ib] = scale;
            shifts[ib] = best_k as i8;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            continue;
        }

        // (:4909-4937) — second pass: re-fit d from the ACTUAL grid
        // quantas (with (2l+1) factor), then scatter d into scales.
        let mut sc = [0u16; 4]; // alias of y[ibl].scales
        let d = max_scale / 15.0;
        let id = 1.0 / d;
        let mut sumqx_f = 0.0f32;
        let mut sumq2_f = 0.0f32;
        for ib in 0..QK_K / bs {
            let mut li = nearest_int(0.5 * (id * scales[ib] - 1.0));
            li = li.clamp(0, 7);
            sc[ib / 4] |= (li as u16) << (3 * (ib % 4));
            ob[qh_off + ib] |= masks[shifts[ib] as usize];
            let xb = &xb_block[bs * ib..bs * ib + bs];
            let qw = quant_weights.map(|w| &w[QK_K * ibl + bs * ib..QK_K * ibl + bs * ib + bs]);
            for i in 0..bs {
                weight[i] = match qw {
                    Some(q) => q[i] * (sigma2 + xb[i] * xb[i]).sqrt(),
                    None => xb[i] * xb[i],
                };
            }
            for k in 0..bs / 8 {
                let xx: &[f32; 3] = if k == 0 {
                    if shifts[ib] < 2 {
                        &x_p
                    } else {
                        &x_m
                    }
                } else if shifts[ib] % 2 == 0 {
                    &x_p
                } else {
                    &x_m
                };
                // (:4928) — reconstruct the grid index from qs/qh. C
                // promotes qh to int before the shift & 0x700 mask.
                let idx_rec = ob[qs_off + 2 * ib + k] as usize
                    + (((ob[qh_off + ib] as u32) << (8 - 4 * k) & 0x700) as usize);
                let pg = lat.grid[idx_rec].to_le_bytes();
                for j in 0..8 {
                    let w = weight[8 * k + j];
                    let q = xx[((pg[j] as i8 - 1) / 2) as usize] * (2 * li + 1) as f32;
                    sumqx_f += (w * q) * xb[8 * k + j];
                    sumq2_f += (w * q) * q;
                }
            }
        }
        let d = if sumq2_f > 0.0 { sumqx_f / sumq2_f } else { d };
        // (:4938-4942) — f16 d * 1.1125, scattered 4 bits at a time
        // into the top nibbles of the 4 scales u16s.
        let s_f16 = f16::from_f32(d * 1.1125);
        let s_u16 = s_f16.to_bits();
        sc[0] |= (s_u16 & 0x000f) << 12;
        sc[1] |= (s_u16 & 0x00f0) << 8;
        sc[2] |= (s_u16 & 0x0f00) << 4;
        // `<< 0` kept for visual symmetry with the C nibble-unpack pattern.
        #[allow(clippy::identity_op)]
        {
            sc[3] |= (s_u16 & 0xf000) << 0;
        }
        for (i, &v) in sc.iter().enumerate() {
            ob[scales_off + 2 * i..scales_off + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

// ─── IQ4 family (:4966-5133) ────────────────────────────────────────────

/// Bytes per block_iq4_nl: d: f16 + qs: [u8; 16] = 18 (QK4_NL = 32,
/// ggml-common.h:448-452).
pub const IQ4_NL_BLOCK_BYTES: usize = 2 + 32 / 2;
/// Bytes per block_iq4_xs: d: f16 + scales_h: u16 + scales_l: [u8; 4] +
/// qs: [u8; 128] = 136 (ggml-common.h:454-460).
pub const IQ4_XS_BLOCK_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2;

/// Port of `best_index_int8` (:28-37) — binary search into a sorted
/// i8 value table.
fn best_index_int8(n: usize, val: &[i8], x: f32) -> usize {
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[n - 1] as f32 {
        return n - 1;
    }
    let mut ml = 0usize;
    let mut mu = n - 1;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if (x - val[mu - 1] as f32) < (val[mu] as f32 - x) {
        mu - 1
    } else {
        mu
    }
}

/// Port of `quantize_row_iq4_nl_impl` (:4966-5075) — the shared engine
/// for iq4_nl (super_block 32, single block) and iq4_xs (super_block
/// QK_K=256, 8 sub-blocks with the 6-bit packed scales).
///
/// Returns the final d (for the caller's f16 store) and the quints; the
/// output slice `q4` must be `super_block_size/2` bytes (plus scales
/// packing done by the CALLER because the layouts differ).
///
/// Actually — to stay byte-faithful we mirror the C signature: the
/// caller passes dh/scales_h/scales_l views and we write everything,
/// exactly like upstream. ntry: 7 (weighted), -1 (ref).
#[allow(clippy::too_many_arguments)]
fn quantize_row_iq4_nl_impl(
    super_block_size: usize,
    block_size: usize,
    x: &[f32],                   // ONE super-block
    dh: &mut [u8],               // 2 bytes f16
    q4: &mut [u8],               // super_block_size/2
    scales_h: Option<&mut [u8]>, // 2*(nb+7)/8 bytes, iq4_xs only
    scales_l: &mut [u8],         // nb/2, iq4_xs only (unused for nl)
    values: &[i8; 16],           // kvalues_iq4nl
    quant_weights: Option<&[f32]>,
    ntry: i32,
    l_buf: &mut [u8],       // super_block_size
    weight_buf: &mut [f32], // block_size
    scales: &mut [f32],     // super_block_size/block_size
) {
    let nb = super_block_size / block_size;
    let mut sigma2 = 0.0f32;
    for &v in x {
        sigma2 += v * v;
    }
    // (:4975) — 2/sbs factor.
    sigma2 *= 2.0 / super_block_size as f32;

    q4.fill(0);
    dh.copy_from_slice(&f16::from_f32(0.0).to_le_bytes());

    let mut max_scale = 0.0f32;
    let mut amax_scale = 0.0f32;
    for ib in 0..nb {
        let xb = &x[ib * block_size..ib * block_size + block_size];
        let lb = &mut l_buf[ib * block_size..ib * block_size + block_size];
        let qw = quant_weights.map(|w| &w[ib * block_size..ib * block_size + block_size]);
        for j in 0..block_size {
            weight_buf[j] = match qw {
                Some(q) => q[j] * (sigma2 + xb[j] * xb[j]).sqrt(),
                None => xb[j] * xb[j],
            };
        }
        // (:4990-4996) — signed max at the largest |x|.
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        // (:5001) — ntry > 0 starts from the NEGATIVE ratio (weighted
        // path), the ref from the positive.
        let mut d = if ntry > 0 {
            -max / values[0] as f32
        } else {
            max / values[0] as f32
        };
        let mut id = 1.0 / d;
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for j in 0..block_size {
            let al = id * xb[j];
            let l = best_index_int8(16, values, al);
            lb[j] = l as u8;
            let q = values[l] as f32;
            let w = weight_buf[j];
            // (:5010-5011) — ((w*q)*x), ((w*q)*q).
            sumqx += (w * q) * xb[j];
            sumq2 += (w * q) * q;
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;
        for itry in -ntry..=ntry {
            // (:5016) — the itry=0 pass is INCLUDED (no skip).
            id = (itry as f32 + values[0] as f32) / max;
            let mut sumqx2 = 0.0f32;
            let mut sumq22 = 0.0f32;
            for j in 0..block_size {
                let al = id * xb[j];
                let l = best_index_int8(16, values, al);
                let q = values[l] as f32;
                let w = weight_buf[j];
                sumqx2 += (w * q) * xb[j];
                sumq22 += (w * q) * q;
            }
            if sumq22 > 0.0 && sumqx2 * sumqx2 > best * sumq22 {
                d = sumqx2 / sumq22;
                best = d * sumqx2;
            }
        }
        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    if nb > 1 {
        // iq4_xs path (:5037-5059): 6-bit packed scales + requantize.
        // scales_h is a uint16_t[] in C — view our byte slice as u16s
        // (LE) so `sh_u16[ib/8] |= l_h << 2*(ib%8)` matches upstream.
        let sh = scales_h.unwrap();
        sh.fill(0);
        let n_h = (nb + 7) / 8;
        let sh_u16: Vec<u16> = vec![0; n_h];
        // (:5040) — NEGATIVE super-block scale.
        let d = -max_scale / 32.0;
        dh.copy_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut sh_u16 = sh_u16;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for ib in 0..nb {
            let mut li = nearest_int(id * scales[ib]);
            li = li.clamp(-32, 31);
            let dl = d * li as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let lb = &mut l_buf[ib * block_size..ib * block_size + block_size];
            let xb = &x[ib * block_size..ib * block_size + block_size];
            for j in 0..block_size {
                lb[j] = best_index_int8(16, values, idl * xb[j]) as u8;
            }
            let li = li + 32;
            let l_l = (li & 0xf) as u8;
            let l_h = (li >> 4) as u8;
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            sh_u16[ib / 8] |= (l_h as u16) << (2 * (ib % 8));
        }
        // Store the u16 view back into the byte slice (LE).
        for (i, &v) in sh_u16.iter().enumerate() {
            sh[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
        }
    } else {
        // iq4_nl path (:5060-5068).
        dh.copy_from_slice(&f16::from_f32(scales[0]).to_le_bytes());
        if ntry > 0 {
            let id = if scales[0] != 0.0 {
                1.0 / scales[0]
            } else {
                0.0
            };
            for j in 0..super_block_size {
                l_buf[j] = best_index_int8(16, values, id * x[j]) as u8;
            }
        }
    }

    // (:5070-5074) — final nibble packing.
    for i in 0..super_block_size / 32 {
        for j in 0..16 {
            q4[16 * i + j] = l_buf[32 * i + j] | (l_buf[32 * i + 16 + j] << 4);
        }
    }
}

/// Port of `quantize_iq4_nl` (:5077-5097) — 32-element blocks, no
/// super-block scales. NOTE: unlike the other IQ formats this one is
/// NOT QK_K-locked (n_per_row % 32 == 0 suffices).
pub fn quantize_row_iq4_nl_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % 32 == 0);
    let nblock = n_per_row / 32;
    let mut out = vec![0u8; nblock * IQ4_NL_BLOCK_BYTES];
    let values = crate::gguf_iq_grid::KVALUES_IQ4NL;

    let mut l_buf = [0u8; 32];
    let mut weight_buf = [0f32; 32];
    let mut scales = [0f32; 1];
    let mut unused_h = [0u8; 2];
    let mut unused_l = [0u8; 0];
    let mut q4 = [0u8; 16];

    for ibl in 0..nblock {
        let xb = &x[ibl * 32..ibl * 32 + 32];
        let qw = quant_weights.map(|w| &w[ibl * 32..ibl * 32 + 32]);
        let mut dh = [0u8; 2];
        quantize_row_iq4_nl_impl(
            32,
            32,
            xb,
            &mut dh,
            &mut q4,
            Some(&mut unused_h),
            &mut unused_l,
            &values,
            qw,
            7,
            &mut l_buf,
            &mut weight_buf,
            &mut scales,
        );
        let ob = &mut out[ibl * IQ4_NL_BLOCK_BYTES..(ibl + 1) * IQ4_NL_BLOCK_BYTES];
        ob[..2].copy_from_slice(&dh);
        ob[2..].copy_from_slice(&q4);
    }
    out
}

/// Port of `quantize_iq4_xs` (:5115-5133) — QK_K super-blocks of 8
/// 32-sub-blocks; per-block 6-bit scales (4 low in scales_l nibbles,
/// 2 high in scales_h), NEGATIVE d = -max_scale/32 (:5040).
pub fn quantize_row_iq4_xs_weighted(
    x: &[f32],
    n_per_row: usize,
    quant_weights: Option<&[f32]>,
) -> Vec<u8> {
    assert!(n_per_row % QK_K == 0);
    let nblock = n_per_row / QK_K;
    let mut out = vec![0u8; nblock * IQ4_XS_BLOCK_BYTES];
    let values = crate::gguf_iq_grid::KVALUES_IQ4NL;

    // Layout: d: f16 | scales_h: u16 | scales_l: [u8; 4] | qs: [u8; 128].
    let (d_off, sh_off, sl_off, qs_off) = (0usize, 2, 4, 4 + QK_K / 64);

    let mut l_buf = [0u8; QK_K];
    let mut weight_buf = [0f32; 32];
    let mut scales = [0f32; QK_K / 32];
    let mut unused_h = [0u8; 0];
    let mut q4 = [0u8; QK_K / 2];

    for ibl in 0..nblock {
        let xb = &x[ibl * QK_K..ibl * QK_K + QK_K];
        let qw = quant_weights.map(|w| &w[ibl * QK_K..ibl * QK_K + QK_K]);
        let mut dh = [0u8; 2];
        let mut scales_h = [0u8; 2];
        let mut scales_l = [0u8; QK_K / 64];
        // NOTE: the C impl's super_block branch expects
        // scales_h sized (nb+7)/8 * 2 = 2 bytes and writes
        // scales_h[ib/8] — a u16 viewed as bytes. We pass a 2-byte
        // array; the u16 view matches the block layout.
        let _ = &mut unused_h;
        quantize_row_iq4_nl_impl(
            QK_K,
            32,
            xb,
            &mut dh,
            &mut q4,
            Some(&mut scales_h),
            &mut scales_l,
            &values,
            qw,
            7,
            &mut l_buf,
            &mut weight_buf,
            &mut scales,
        );
        let ob = &mut out[ibl * IQ4_XS_BLOCK_BYTES..(ibl + 1) * IQ4_XS_BLOCK_BYTES];
        ob[d_off..d_off + 2].copy_from_slice(&dh);
        ob[sh_off..sh_off + 2].copy_from_slice(&scales_h);
        ob[sl_off..sl_off + QK_K / 64].copy_from_slice(&scales_l);
        ob[qs_off..qs_off + QK_K / 2].copy_from_slice(&q4);
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

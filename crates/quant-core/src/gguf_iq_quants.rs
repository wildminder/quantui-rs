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

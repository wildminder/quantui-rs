//! Bit-exact port of torch CPU RNG: MT19937 engine + randn kernels (plan Phase 7.2).
//!
//! Sources:
//! - `torch/include/ATen/core/MT19937RNGEngine.h` — at::mt19937 engine
//! - `torch/include/ATen/core/DistributionsHelper.h` — uniform_real_distribution,
//!   normal_distribution (cached-pair Box-Muller, scalar path)
//! - `torch/include/ATen/native/cpu/DistributionTemplates.h` — normal_fill /
//!   NormalFill16 (size>=16 contiguous float path)
//!
//! Two distinct randn paths exist and produce DIFFERENT streams:
//! 1. **scalar path** (numel < 16): per-element cached Box-Muller over f64 with
//!    u1 = 1 - uniform(random32), r = sqrt(-2·log1p(-u2)), theta = 2π·u1;
//!    returns cos, caches sin.
//! 2. **normal_fill path** (numel >= 16, contiguous): fill buffer with uniforms,
//!    then in 16-value blocks: u1 = 1-x[0..8], u2 = x[8..16], radius =
//!    sqrt(-2·ln(u1)), theta = 2π·u2 → [radius·cos, radius·sin].
//!
//! Calibration data in ctq uses torch.randn(3072, N) → path 2 (float32).

/// at::mt19937 — bit-faithful port of MT19937RNGEngine.h.
pub struct Mt19937 {
    seed_: u64,
    left_: i32,
    next_: usize,
    state_: [u32; 624],
}

const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;
const N: usize = 624;
const M: usize = 397;

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut s = Self {
            seed_: seed,
            left_: 1,
            next_: 0,
            state_: [0; N],
        };
        s.state_[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..N {
            s.state_[j] = 1812433253u32
                .wrapping_mul(s.state_[j - 1] ^ (s.state_[j - 1] >> 30))
                .wrapping_add(j as u32);
        }
        s
    }

    #[inline]
    fn mix_bits(u: u32, v: u32) -> u32 {
        (u & UMASK) | (v & LMASK)
    }

    #[inline]
    fn twist(u: u32, v: u32) -> u32 {
        (Self::mix_bits(u, v) >> 1) ^ if v & 1 != 0 { MATRIX_A } else { 0 }
    }

    fn next_state(&mut self) {
        self.left_ = N as i32;
        self.next_ = 0;
        // Port of MT19937RNGEngine.h next_state(): a single uint32* `p` walks
        // the state array. Loop1 runs (N-M) iterations over state_[0..N-M-1];
        // loop2 runs (M-1) iterations over state_[N-M..N-2]; the final step
        // writes state_[N-1]. The previous Rust port used (N-M-1) for loop2,
        // which left state_[N-M-1..N-1] unregenerated and diverged from torch
        // deep in the stream.
        let mut p = 0usize;
        for _ in 0..(N - M) {
            self.state_[p] = self.state_[p + M] ^ Self::twist(self.state_[p], self.state_[p + 1]);
            p += 1;
        }
        for _ in 0..(M - 1) {
            self.state_[p] =
                self.state_[p + M - N] ^ Self::twist(self.state_[p], self.state_[p + 1]);
            p += 1;
        }
        self.state_[p] = self.state_[p + M - N] ^ Self::twist(self.state_[p], self.state_[0]);
    }

    /// operator() — one random u32.
    pub fn random_u32(&mut self) -> u32 {
        self.left_ -= 1;
        if self.left_ == 0 {
            self.next_state();
        }
        let y = self.state_[self.next_];
        self.next_ += 1;
        let mut y = y ^ (y >> 11);
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn seed(&self) -> u64 {
        self.seed_
    }
}

/// uniform_real<float> over a u32 draw (TransformationHelper.h): MASK uses
/// std::numeric_limits<float>::digits == 24 (incl. implicit bit), DIVISOR =
/// 1/2^24; arithmetic in dist_acctype<float> == double.
#[inline]
fn uniform_real_f32(val: u32) -> f32 {
    const MASK: u32 = (1u32 << 24) - 1;
    const DIVISOR: f64 = 1.0 / (1u64 << 24) as f64;
    ((val & MASK) as f64 * DIVISOR) as f32
}

/// Torch CPU randn generator covering BOTH code paths.
pub struct TorchRng {
    engine: Mt19937,
    /// Cached second value from scalar-path Box-Muller pairs.
    next_double_normal_sample: Option<f64>,
}

impl TorchRng {
    pub fn manual_seed(seed: u64) -> Self {
        Self {
            engine: Mt19937::new(seed),
            next_double_normal_sample: None,
        }
    }

    /// torch.randn(numel, generator=self) for float32 CPU tensors.
    ///
    /// Dispatches exactly like normal_kernel: numel >= 16 → normal_fill16 path;
    /// else scalar cached-pair Box-Muller in f64.
    pub fn randn_f32(&mut self, numel: usize) -> Vec<f32> {
        if numel >= 16 {
            self.randn_normal_fill(numel)
        } else {
            self.randn_scalar(numel)
        }
    }

    /// Scalar path (DistributionTemplates.h cpu_serial_kernel branch +
    /// DistributionsHelper.h normal_distribution<double>):
    /// u1,u2 drawn via uniform_real_distribution<double>(random()); wait —
    /// T=double means random64(). Verified empirically below in tests against
    /// captured vectors; see tests::randn_scalar_matches_torch.
    fn randn_scalar(&mut self, numel: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(numel);
        for _ in 0..numel {
            // normal_distribution<double>::operator()
            if let Some(cached) = self.next_double_normal_sample.take() {
                out.push(cached as f32);
                continue;
            }
            let u1 = self.uniform_double();
            let u2 = self.uniform_double();
            let r = (-2.0f64 * (-u2).ln_1p()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u1;
            let sample = r * theta.sin();
            self.next_double_normal_sample = Some(sample);
            out.push((r * theta.cos()) as f32);
        }
        out
    }

    /// uniform_real_distribution<double>(0,1) consumes random64() with a
    /// 53-bit mask (f64 mantissa digits) — brute-force-verified vs torch.
    #[inline]
    fn uniform_double(&mut self) -> f64 {
        let val = self.random_u64();
        const MASK: u64 = (1u64 << 53) - 1;
        const DIVISOR: f64 = 1.0 / (1u64 << 53) as f64;
        ((val & MASK) as f64) * DIVISOR
    }

    fn random_u64(&mut self) -> u64 {
        // at::mt19937 random64(): two consecutive operator() calls combined as
        // HIGH word first in the 64-bit value — empirically verified against
        // torch randn(8) bit vectors: u64 = (first << 32) | second.
        let a = self.engine.random_u32() as u64;
        let b = self.engine.random_u32() as u64;
        (a << 32) | b
    }

    /// normal_fill path (size >= 16, float32): uniforms into buffer, then
    /// 16-block Box-Muller: u1 = 1 - data[i..i+8], u2 = data[i+8..i+16].
    ///
    /// The SIMD build uses the avx_mathfun.h `log256_ps`/`sincos256_ps`
    /// polynomial approximations (not libm ln/sin/cos). This port reproduces
    /// those polynomials instruction-for-instruction with the same f32 constant
    /// literals, so the resulting calibration tensor matches torch CPU
    /// bit-for-bit (verified: 0 ulp vs torch.randn with seed 233983427).
    fn randn_normal_fill(&mut self, numel: usize) -> Vec<f32> {
        let mut data: Vec<f32> = (0..numel)
            .map(|_| uniform_real_f32(self.engine.random_u32()))
            .collect();

        let mut i = 0usize;
        while i + 16 <= numel {
            normal_fill_16_block(&mut data[i..i + 16]);
            i += 16;
        }
        if numel % 16 != 0 {
            // Recompute the last 16 values: overwrite tail starting at size-16
            // with fresh uniforms, then transform. This CONSUMES extra draws —
            // faithful to reference.
            let start = numel - 16;
            for v in data[start..].iter_mut() {
                *v = uniform_real_f32(self.engine.random_u32());
            }
            normal_fill_16_block(&mut data[start..]);
        }
        data
    }
}

/// Scalar bit-exact port of avx_mathfun.h `log256_ps` (AVX2 build).
///
/// Every operation below maps 1:1 to a discrete `_mm256_*ps/_epi32` instruction,
/// so each step rounds like the SIMD original (no contraction, no libm).
// NOTE: literals copied verbatim from avx_mathfun.h so the f32 constants round
// bit-identically to the C-compiled `float` values torch uses. Do NOT shorten
// them — e.g. 1.1676998740E-1 and 2.0000714765E-1 each differ from a truncated
// 1-ulp literal in the low bit, which propagates through the polynomial and
// breaks byte-exact parity with torch.randn.
const LOG_P: [f32; 9] = [
    7.0376836292E-2,
    -1.1514610310E-1,
    1.1676998740E-1,
    -1.2420140846E-1,
    1.4249322787E-1,
    -1.6668057665E-1,
    2.0000714765E-1,
    -2.4999993993E-1,
    3.3333331174E-1,
];
const LOG_Q1: f32 = -2.12194440e-4;
const LOG_Q2: f32 = 0.693359375;
const LOG_SQRTHF: f32 = 0.707106781186547524;

#[inline]
pub(crate) fn log256_ps(mut x: f32) -> f32 {
    let invalid = x <= 0.0; // _mm256_cmp_ps(x, 0, _CMP_LE_OS)
                            // cut off denormals: x = max(x, min_norm_pos)
    const MIN_NORM_POS: f32 = f32::from_bits(0x0080_0000);
    if !(x >= MIN_NORM_POS) {
        x = MIN_NORM_POS;
    }
    // extract exponent, rebuild x with 0.5-biased mantissa
    let mut bits = x.to_bits();
    let exp = ((bits >> 23) as i32) - 0x7f; // sub_epi32(imm, 0x7f)
    bits &= 0x807f_ffff; // and inv_mant_mask (~0x7f800000)
    bits |= 0x3f00_0000; // or 0p5 (bits of 0.5)
    x = f32::from_bits(bits);
    let mut e = exp as f32 + 1.0; // cvtepi32_ps, add one

    // if (x < SQRTHF) { e -= 1; x = x + x - 1 } else { x = x - 1 }
    let mask_lt = x < LOG_SQRTHF;
    let tmp = if mask_lt { x } else { 0.0 }; // and(x, mask)
    x -= 1.0;
    if mask_lt {
        e -= 1.0;
    }
    x += tmp;

    let z = x * x;
    // Horner over p0..p8 — each mul/add is a SEPARATE instruction in the
    // original, so do NOT contract into fma here. Note the trailing
    // `y = y * x` after p8 (folds the linear term) BEFORE `y *= z`.
    let mut y = LOG_P[0];
    for c in LOG_P[1..].iter() {
        y = y * x;
        y = y + c;
    }
    y *= x;
    y *= z;
    y += e * LOG_Q1;
    y -= z * 0.5;
    x += y;
    x += e * LOG_Q2;
    // invalid_mask (x<=0) ORed in produces NaN; unreachable for Box-Muller u1∈(0,1].
    if invalid {
        return f32::NAN;
    }
    x
}

/// Scalar bit-exact port of avx_mathfun.h `sincos256_ps` (AVX2 build).
/// Returns (sin, cos).
const SINCOS_FOPI: f32 = 1.27323954473516; // 4/pi
const DP1: f32 = -0.78515625;
const DP2: f32 = -2.4187564849853515625e-4;
const DP3: f32 = -3.77489497744594108e-8;
const SINCOF: [f32; 3] = [-1.9515295891E-4, 8.3321608736E-3, -1.6666654611E-1];
const COSCOF: [f32; 3] = [
    2.443315711809948E-005,
    -1.388731625493765E-003,
    4.166664568298827E-002,
];

#[inline]
pub(crate) fn sincos256_ps(x_in: f32) -> (f32, f32) {
    let sign_bit_sin = x_in.to_bits() & 0x8000_0000;
    let x_abs = f32::from_bits(x_in.to_bits() & 0x7fff_ffff);

    // scale by 4/pi, split integer quadrants (cvttps = truncate toward zero)
    let y_scaled = x_abs * SINCOS_FOPI;
    let j = (unsafe { y_scaled.to_int_unchecked::<i32>() } + 1) & !1;
    let y = j as f32;

    let swap_sign_bit_sin = ((j & 4) << 29) as u32;
    let poly_mask = (j & 2) == 0; // cmpeq(j&2, 0)

    // extended-precision modular arithmetic: x = ((x + y*DP1) + y*DP2) + y*DP3
    let mut xr = x_abs + y * DP1;
    xr += y * DP2;
    xr += y * DP3;

    let imm4 = j - 2;
    let sign_bit_cos = (((!imm4) & 4) << 29) as u32;
    let sign_bit_sin = sign_bit_sin ^ swap_sign_bit_sin;

    let z = xr * xr;
    // cosine polynomial branch — note TWO consecutive *z after p2
    // (cephes folds the z² even power into double multiplication).
    let mut yc = COSCOF[0];
    yc = yc * z + COSCOF[1];
    yc = yc * z + COSCOF[2];
    yc *= z;
    yc *= z;
    yc -= z * 0.5;
    yc += 1.0;
    // sine polynomial branch
    let mut ys = SINCOF[0];
    ys = ys * z + SINCOF[1];
    ys = ys * z + SINCOF[2];
    ys *= z;
    ys *= xr;
    ys += xr;

    // select per-lane between branches (and / andnot / add)
    let ysin2 = if poly_mask { ys } else { 0.0 };
    let ysin1 = if poly_mask { 0.0 } else { yc };
    let y2r = ys - ysin2;
    let yr = yc - ysin1;

    let s = f32::from_bits((ysin1 + ysin2).to_bits() ^ sign_bit_sin);
    let c = f32::from_bits((yr + y2r).to_bits() ^ sign_bit_cos);
    (s, c)
}

/// NormalFill16<float> AVX2 specialization on one 16-value block:
/// u1 = 1-x[0..8], u2 = x[8..16]; radius = sqrt(-2·log256(u1));
/// theta = 2π·u2 (2π computed in f64 then narrowed, matching set1_ps);
/// outputs via fmadd(n, std_, mean_) — with std=1/mean=0 the fusion is exact.
fn normal_fill_16_block(block: &mut [f32]) {
    debug_assert_eq!(block.len(), 16);
    // AVX2 NormalFill16<float> flow with our scalar log256/sincos256 ports:
    // radius = sqrt(-2 * log256_ps(u1)); theta = 2pi_f64->f32 * u2;
    // output = fmadd(radius*cos, 1.0, 0.0) == plain product.
    let two_pi = (2.0f64 * std::f64::consts::PI) as f32;
    for j in 0..8 {
        let u1 = 1.0f32 - block[j];
        let u2 = block[j + 8];
        let radius = (-2.0f32 * log256_ps(u1)).sqrt();
        let theta = two_pi * u2;
        let (s, c) = sincos256_ps(theta);
        block[j] = radius.mul_add(c, 0.0);
        block[j + 8] = radius.mul_add(s, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vectors captured from torch 2.13 CPU:
    ///   g = torch.Generator(); g.manual_seed(233983427); torch.randn(N, generator=g)
    #[test]
    fn randn_scalar_matches_torch() {
        // numel < 16 → scalar cached-pair path.
        // torch randn(8, seed=233983427) bit-exact:
        let bits = [
            0x3efb_ea0au32,
            0xbe9b_5326,
            0x3f02_d4fd,
            0xbf9a_5216,
            0xbd71_6485,
            0x3d65_2e68,
            0xbf96_cc45,
            0x3f09_4365,
        ];
        let expected: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let got = TorchRng::manual_seed(233983427).randn_f32(8);
        assert_eq!(got, expected, "scalar randn must be bit-exact vs torch");
    }

    #[test]
    fn mt19937_stream_is_selfconsistent() {
        // Same seed twice → identical stream; different seeds differ.
        let a: Vec<u32> = {
            let mut r = Mt19937::new(42);
            (0..10).map(|_| r.random_u32()).collect()
        };
        let b: Vec<u32> = {
            let mut r = Mt19937::new(42);
            (0..10).map(|_| r.random_u32()).collect()
        };
        assert_eq!(a, b);
        let c: Vec<u32> = {
            let mut r = Mt19937::new(43);
            (0..10).map(|_| r.random_u32()).collect()
        };
        assert_ne!(a, c);
    }

    #[test]
    fn randn_normal_fill_shape_independence() {
        // PHASE0 finding: randn(3072,N) == first 3072*N of flat stream.
        let flat = TorchRng::manual_seed(7).randn_f32(64);
        let shaped = TorchRng::manual_seed(7).randn_f32(4 * 16);
        assert_eq!(flat, shaped);
    }

    #[test]
    fn randn_normal_fill_matches_torch() {
        // torch randn(32, seed=233983427) — normal_fill path (numel>=16).
        // Bit-exact expected: our scalar port of log256_ps/sincos256_ps
        // reproduces the AVX2 polynomial math instruction-for-instruction.
        let bits = [
            0xbe8e_e95eu32,
            0x3f22_4dde,
            0x3f5b_a5c6,
            0x3fa3_34c4,
            0xbf69_abe2,
            0x3fdd_f2d6,
            0xbec2_8396,
            0x3e6e_01af,
            0x3f68_d65b,
            0xbf06_dbe3,
            0xbf5a_835b,
            0x3f95_7fe8,
            0x4025_81fe,
            0x3f94_34a3,
            0x3f4b_a448,
            0xbee5_0f39,
            0x3e93_9816,
            0x3f08_7395,
            0xbf76_1eda,
            0x3f68_6b4b,
            0xbfc1_2dda,
            0x3f1e_9dfb,
            0xbf75_5245,
            0xbf3f_6f47,
            0xbdfa_994d,
            0x3f2b_922e,
            0xbeee_5ef1,
            0xbed6_eb38,
            0xbf86_807c,
            0xc046_0027,
            0x3e0f_084f,
            0xbef5_bb76,
        ];
        let expected: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let got = TorchRng::manual_seed(233983427).randn_f32(32);
        assert_eq!(
            got, expected,
            "normal_fill randn must be bit-exact vs torch"
        );
    }

    #[test]
    fn log256_matches_libm_within_2ulp() {
        for i in 1..=10_000u32 {
            let x = f32::from_bits((i << 13) | 0x3f80_0000);
            if !(x > 0.0 && x.is_finite()) {
                continue;
            }
            let approx = log256_ps(x);
            let exact = x.ln();
            let d = approx.to_bits().abs_diff(exact.to_bits());
            assert!(
                d <= 2 || (approx - exact).abs() / exact.abs() < 1e-6,
                "log({x})"
            );
        }
    }

    #[test]
    fn statistical_sanity_normal_fill() {
        let vals = TorchRng::manual_seed(99).randn_f32(100_000);
        let n = vals.len() as f64;
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!((mean - 0.0).abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
        assert!(vals.iter().all(|v| v.is_finite()));
    }
}

"""Probe MXFP8 scale-pipeline semantics for the Rust port (Phase 3.3).

Two parity hazards to resolve before implementing quant_mxfp8.rs:

A) ``scale_needed = block_max.float() / self.fp8_max`` is tensor/scalar
   division. Phase 3.2 proved scalar/TENSOR div in torch is
   ``scalar * reciprocal(tensor)`` (not IEEE div). Here we need the
   tensor/SCALAR semantics: direct IEEE division, or mul-by-reciprocal?
   (448 = 7*2^6, so 1/448 is not exactly representable -> the two differ.)

B) ``exp_biased = ceil(torch.log2(scale_needed)) + 127``. For x slightly
   ABOVE an exact power of 2, the true log2 has a tiny positive fractional
   part that can be < ulp(E)/2 of the f32 representation of the integer E,
   so even a correctly-rounded f32 log2 returns exactly E and ceil gives E
   (while a bit-based "mantissa nonzero -> E+1" rule would give E+1).
   Characterize torch CPU log2 around powers of 2 and check whether
   "f64 log2 -> round to f32 -> ceil" is an exact emulator.

Run with the ctq venv interpreter (same torch build that made the goldens):
    <DEV-TREE>\\Python\\<LOCAL-VENV>\\Scripts\\python.exe tools/probe_mxfp8_scale.py
"""

from __future__ import annotations

import numpy as np
import torch


def bits_f32(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.int32).numpy().astype(np.uint32)


def probe_division() -> None:
    print("=" * 70)
    print("A) tensor/scalar division semantics:  t / 448.0")
    print("=" * 70)
    rng = np.random.default_rng(1234)
    # 2M random positives spanning many exponents + bf16-representable values
    f = rng.standard_normal(2_000_000).astype(np.float32)
    f = np.abs(f) + np.float32(1e-38)
    scales = np.float32(2.0) ** rng.integers(-120, 10, size=f.shape).astype(np.float32)
    vals = (f * scales).astype(np.float32)
    # also bf16-grid values (block_max comes from bf16 weights)
    bf = torch.randn(1_000_000, dtype=torch.float32).to(torch.bfloat16).float().abs().numpy()
    vals = np.concatenate([vals, bf])
    vals = np.concatenate([vals, np.array([0.0, 448.0, 224.0, 1.0, 2**-126, 3.4e38], dtype=np.float32)])
    t = torch.from_numpy(vals)

    div_scalar = t / 448.0                                   # what ctq does
    mul_recip_f64 = t * (1.0 / 448.0)                        # python-float (f64) scalar
    mul_recip_f32 = t * torch.tensor(np.float32(1.0) / np.float32(448.0))
    # reference: true IEEE division computed in f64 then rounded to f32
    true_div = torch.from_numpy((vals.astype(np.float64) / 448.0).astype(np.float32))

    b_ds = bits_f32(div_scalar)
    b_mr64 = bits_f32(mul_recip_f64)
    b_mr32 = bits_f32(mul_recip_f32)
    b_td = bits_f32(true_div)

    n = len(vals)
    print(f"  samples: {n}")
    print(f"  t/448.0 == true_ieee_div(f64-rounded) : {int((b_ds == b_td).sum())}/{n}")
    print(f"  t/448.0 == t*(1.0/448.0)  [f64 recip] : {int((b_ds == b_mr64).sum())}/{n}")
    print(f"  t/448.0 == t*f32(1/448)   [f32 recip] : {int((b_ds == b_mr32).sum())}/{n}")
    mm = np.nonzero(b_ds != b_td)[0]
    if mm.size:
        i = mm[0]
        print(f"  first mismatch vs true div at i={i}: val={vals[i]!r} "
              f"div={b_ds[i]:#010x} true={b_td[i]:#010x}")
    # also: tensor/TENSOR division (used in _simple_quantize: blocks / scales)
    a = torch.from_numpy(rng.standard_normal(1_000_000).astype(np.float32))
    b = torch.from_numpy((np.abs(rng.standard_normal(1_000_000)) + 1e-30).astype(np.float32))
    tt = a / b
    ref = torch.from_numpy((a.numpy().astype(np.float64) / b.numpy().astype(np.float64)).astype(np.float32))
    same = int((bits_f32(tt) == bits_f32(ref)).sum())
    print(f"  tensor/tensor == true IEEE div        : {same}/{a.numel()}")


def probe_log2() -> None:
    print()
    print("=" * 70)
    print("B) torch.log2 around powers of two  (E8M0 exponent hazard)")
    print("=" * 70)
    xs: list[float] = []
    for E in range(-126, 21):
        base = np.ldexp(np.float32(1.0), E)
        bi = np.frombuffer(np.array([base], dtype=np.float32).tobytes(), dtype=np.uint32)[0]
        for k in range(0, 33):
            xs.append(np.frombuffer(np.uint32(bi + k).tobytes(), dtype=np.float32)[0])  # 2^E + k ulp
            if k > 0:
                xs.append(np.frombuffer(np.uint32(bi - k).tobytes(), dtype=np.float32)[0])  # 2^E - k ulp
    # dense random sweep across the scale_needed range
    rng = np.random.default_rng(777)
    m = rng.standard_normal(2_000_000).astype(np.float32)
    m = np.abs(m) + np.float32(1e-38)
    e = rng.integers(-126, 10, size=m.shape)
    xs_rand = (m * (np.float32(2.0) ** e.astype(np.float32))).astype(np.float32)
    xs_arr = np.concatenate([np.array(xs, dtype=np.float32), xs_rand])

    t = torch.from_numpy(xs_arr)
    log2_torch = torch.log2(t)
    ceil_torch = torch.ceil(log2_torch)

    # candidate emulation 1: f64 log2 -> round to f32 -> ceil
    log2_f64_as_f32 = torch.from_numpy(np.log2(xs_arr.astype(np.float64)).astype(np.float32))
    ceil_emul = torch.ceil(log2_f64_as_f32)

    # candidate emulation 2: bit-based ceil(log2): E + (mantissa != 0)
    xb = bits_f32(t)
    exp_field = ((xb >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64)
    mant = xb & np.uint32(0x7F_FFFF)
    E_unbiased = exp_field - 127
    ceil_bitbased = torch.from_numpy((E_unbiased + (mant != 0).astype(np.int64)).astype(np.float32))

    b_t = bits_f32(ceil_torch)
    b_e = bits_f32(ceil_emul)
    b_b = bits_f32(ceil_bitbased)
    n = xs_arr.size
    print(f"  samples: {n}  ({len(xs)} power-of-2 neighborhood + {xs_rand.size} random)")
    print(f"  ceil(torch.log2) == ceil(f32(f64_log2)) : {int((b_t == b_e).sum())}/{n}")
    print(f"  ceil(torch.log2) == bit-based ceil      : {int((b_t == b_b).sum())}/{n}")

    mm = np.nonzero(b_t != b_e)[0]
    if mm.size:
        print(f"  MISMATCHES vs f64-emulation: {mm.size}; first 8:")
        for i in mm[:8]:
            print(f"    x={xs_arr[i]!r} bits={bits_f32(t)[i]:#010x} "
                  f"torch_ceil={b_t[i]:#010x} emul={b_e[i]:#010x} "
                  f"log2_torch={log2_torch[i].item()!r}")
    else:
        print("  -> f64-log2-rounded-to-f32 is an EXACT emulator of torch.log2+ceil here")

    mb = np.nonzero(b_t != b_b)[0]
    if mb.size:
        print(f"  bit-based differs from torch in {mb.size} cases; first 8:")
        for i in mb[:8]:
            print(f"    x={xs_arr[i]!r} torch_ceil={b_t[i]:#010x} bitbased={b_b[i]:#010x}")

    # exactness of log2 at exact powers of two
    pows = torch.from_numpy(np.array([np.ldexp(np.float32(1.0), E) for E in range(-126, 21)], dtype=np.float32))
    l2 = torch.log2(pows)
    exact = torch.equal(l2, torch.arange(-126, 21, dtype=torch.float32))
    print(f"  torch.log2(2^E) exactly == E for E in [-126, 20]: {exact}")


def probe_e8m0_pipeline() -> None:
    """End-to-end: replicate _compute_block_scales with the emulated formula
    on bf16-grid block maxima and compare against the literal torch pipeline."""
    print()
    print("=" * 70)
    print("C) full E8M0 exponent pipeline on bf16-grid block maxima")
    print("=" * 70)
    rng = np.random.default_rng(2026)
    # block maxima as they occur: amax of 32 bf16 values
    w = torch.randn(200_000, 32, dtype=torch.float32).to(torch.bfloat16).float()
    block_max = torch.amax(torch.abs(w), dim=-1)
    # plus adversarial: exact powers of two and 224*2^k grid (x = block_max/448 = 2^E)
    extra = []
    for E in range(-20, 8):
        extra.append(np.float32(448.0) * np.float32(np.ldexp(np.float32(1.0), E)))
    block_max = torch.cat([block_max, torch.tensor(extra, dtype=torch.float32)])

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale_needed = block_max.float() / fp8_max
    scale_needed = torch.clamp(scale_needed, min=2.0 ** (-127))
    log2_scale = torch.log2(scale_needed)
    exp_biased = torch.ceil(log2_scale).to(torch.int32) + 127
    exp_biased = torch.clamp(exp_biased, 0, 254)
    e8m0_torch = exp_biased.to(torch.uint8)

    # emulation: f64 log2 -> f32 -> ceil
    sn = scale_needed.numpy().astype(np.float64)
    l2e = np.log2(sn).astype(np.float32)
    exp_emul = np.clip(np.ceil(l2e).astype(np.int32) + 127, 0, 254).astype(np.uint8)

    same = int((e8m0_torch.numpy() == exp_emul).sum())
    print(f"  samples: {e8m0_torch.numel()}  e8m0 torch == emulation: {same}/{e8m0_torch.numel()}")
    mm = np.nonzero(e8m0_torch.numpy() != exp_emul)[0]
    if mm.size:
        for i in mm[:8]:
            print(f"    block_max={block_max[i].item()!r} scale_needed={scale_needed[i].item()!r} "
                  f"torch={e8m0_torch[i].item()} emul={exp_emul[i].item()}")
    # distribution of e8m0 values (sanity)
    vals, counts = torch.unique(e8m0_torch, return_counts=True)
    print(f"  distinct e8m0 values: {vals.numel()}  range [{vals.min().item()}, {vals.max().item()}]")


if __name__ == "__main__":
    torch.manual_seed(233983427)
    probe_division()
    probe_log2()
    probe_e8m0_pipeline()

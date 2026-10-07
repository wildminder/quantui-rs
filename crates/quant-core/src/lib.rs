//! quant-core: core library for the quantui-rs CLI.
//!
//! Module layout mirrors the rewrite plan §3 (local-only plan document).
//! All phases have landed (master plan Phases 0-11 + the all-formats plan
//! Phases A-E), so every module below is fully implemented — nothing is
//! stubbed or gated behind a phase marker any more.

pub mod comfy_loader_contract;
pub mod comfy_schema;
// dtype casting for the `cast` subcommand. NOT a quantizer: no scaling, no
// metadata, and it writes no quantization markers.
pub mod cast;
pub mod convrot;
pub mod dtype;
pub mod manifest;
pub mod quant;
pub mod quant_fp8;
pub mod quant_mxfp8;
pub mod quant_nvfp4;
pub mod stream;
// Enabled early: Phase 1 (safetensors IO) scaffolding created in Phase 0 per spec.
pub mod bias_correction;
// Bit-exactness proofs + (Phase 1 of the SIMD plan) the packed-Err transpose
// and the 8x8 AVX2 microkernel for the bias-correction GEMM. The proofs in
// `bias_gemm::proofs` are the gate for the whole optimization: they prove
// `f32::mul_add` reproduces the f64-emulated FMA bitwise.
pub mod bias_gemm;
pub mod discover;
// IQ-family lattice infrastructure (Unsloth plan Phase 4.3 IQ slice).
// gguf_iq_grid is the runtime port (init/kmap/neighbors);
// gguf_iq_tables_data is generated — see tools/extract_iq_tables.py.
pub mod gguf_convert;
mod gguf_iq_grid;
pub mod gguf_iq_quants;
mod gguf_iq_tables_data;
pub mod gguf_names;
pub mod gguf_quants;
pub mod gguf_recipe;
pub mod gguf_registry;
pub mod gguf_verify;
pub mod imatrix;
// QuaRot Eq. 3 incoherence diagnostic (measurement only — nothing calls it yet).
pub mod incoherence;
// Tier 2: opt-in quality refinements. `Quality::Exact` is the zero value and
// keeps every existing format byte-exact.
pub mod llama_policy;
pub mod quality;
pub mod st_io;
pub mod torch_rng;
pub mod validator;

//! quant-core: core library for the quantui-rs CLI.
//!
//! Module layout mirrors plan §3 (docs/plans/2026-08-25-quantui-rust-rewrite-plan.md).
//! Only modules needed for the current phase are enabled; the rest are stubbed out
//! with TODO(phase-N) markers and will be uncommented as phases land.

pub mod dtype;
// Enabled early: Phase 1 (safetensors IO) scaffolding created in Phase 0 per spec.
pub mod st_io;
// TODO(phase-5) pub mod manifest;
// TODO(phase-2) pub mod quant;
// TODO(phase-3) pub mod quant_fp8;
// TODO(phase-3) pub mod quant_mxfp8;
// TODO(phase-3) pub mod quant_nvfp4;
// TODO(phase-10) pub mod gguf_registry;
// TODO(phase-7) pub mod torch_rng;
// TODO(phase-4) pub mod comfy_schema;
// TODO(phase-8) pub mod validator;
// TODO(phase-6) pub mod discover;

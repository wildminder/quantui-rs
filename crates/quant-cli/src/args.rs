//! clap argument definitions for every `quantui-rs` subcommand (plan Phase 9.1).
//!
//! Keeping all arg structs in one module mirrors the plan's `cli/src/args.rs`
//! layout and makes `--help` surface easy to snapshot-test.

use std::path::PathBuf;

use clap::{Args, ValueEnum};

// --------------------------------------------------------------------------- //
// quantize
// --------------------------------------------------------------------------- //

/// INT8 scaling mode (`tensor | row | block`).
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingModeArg {
    Tensor,
    Row,
    Block,
}

/// Output destination layout for sharded inputs.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputModeArg {
    /// One output `.safetensors` per input shard (default; preserves sharding).
    Sharded,
    /// Merge all shards into ONE output `.safetensors`.
    Single,
}

/// Output dtype used when casting skipped (unquantized) weights.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrigDtypeArg {
    Bfloat16,
    Float16,
}

/// Target quantization format. `int8` is the shipped streaming path;
/// `fp8_e4m3` / `mxfp8` / `nvfp4` are being wired into the streaming
/// orchestrator (plan docs/plans/2026-08-28-all-formats-wiring-plan.md).
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatArg {
    Int8,
    #[value(name = "fp8_e4m3")]
    Fp8E4m3,
    Mxfp8,
    Nvfp4,
}

impl FormatArg {
    /// The CLI value name (used in error messages).
    pub fn as_str(&self) -> &'static str {
        match self {
            FormatArg::Int8 => "int8",
            FormatArg::Fp8E4m3 => "fp8_e4m3",
            FormatArg::Mxfp8 => "mxfp8",
            FormatArg::Nvfp4 => "nvfp4",
        }
    }

    /// Formats with FIXED scaling parameters (block scaling at the format's
    /// own block size). Explicit `--scaling-mode` / `--block-size` for these
    /// is a usage error (exit 2).
    pub fn has_fixed_scaling(&self) -> bool {
        matches!(self, FormatArg::Mxfp8 | FormatArg::Nvfp4)
    }
}

#[derive(Args, Debug)]
pub struct QuantizeArgs {
    /// Input: a single `.safetensors` file OR a sharded HF model folder
    /// (containing `model.safetensors.index.json`).
    pub input: PathBuf,

    /// Output path. A `.safetensors` file for single-file / `--output-mode
    /// single` runs, or a directory for sharded runs. If omitted, a name is
    /// suggested automatically (`<base>-int8-simple-heur.safetensors` etc.).
    pub output: Option<PathBuf>,

    /// Target format (`int8`, `fp8_e4m3`, `mxfp8`, `nvfp4`).
    #[arg(long, value_enum, default_value_t = FormatArg::Int8)]
    pub format: FormatArg,

    /// Scaling mode. INT8/FP8: `tensor | row | block` (default `block`).
    /// MXFP8/NVFP4 have fixed block scaling — passing this flag with those
    /// formats is a usage error.
    #[arg(long, short = 'm', value_enum)]
    pub scaling_mode: Option<ScalingModeArg>,

    /// Block size for `--scaling-mode block` (64, 128 or 256; default 128).
    /// MXFP8/NVFP4 have fixed block sizes (32/16) — passing this flag with
    /// those formats is a usage error.
    #[arg(long, short = 'b')]
    pub block_size: Option<u32>,

    /// Enable the skip-inefficient-layers heuristic (on by default; layers
    /// whose dims are not divisible by the block size are copied unchanged).
    #[arg(long, overrides_with = "no_heur")]
    pub heur: bool,

    /// Disable the skip-inefficient-layers heuristic (quantize every 2D
    /// weight; fails on dims not divisible by the block size).
    #[arg(long, overrides_with = "heur")]
    pub no_heur: bool,

    /// Regex; matching layer names are kept at full precision (not quantized).
    #[arg(long)]
    pub exclude_layers: Option<String>,

    /// Output layout for sharded inputs.
    #[arg(long, value_enum, default_value_t = OutputModeArg::Sharded)]
    pub output_mode: OutputModeArg,

    /// Dtype for casting skipped (unquantized) weights.
    #[arg(long, value_enum, default_value_t = OrigDtypeArg::Bfloat16)]
    pub orig_dtype: OrigDtypeArg,

    /// Calibration seed for bias correction (parity contract).
    #[arg(long, default_value_t = 233983427)]
    pub calib_seed: i64,

    /// Simple (no learned rounding) semantics — always on; accepted for
    /// reference compatibility.
    #[arg(long, default_value_t = true)]
    pub simple: bool,

    /// Disable the progress bar (plain, CI-friendly output).
    #[arg(long)]
    pub no_progress: bool,
}

// --------------------------------------------------------------------------- //
// validate
// --------------------------------------------------------------------------- //

#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// One or more `.safetensors` files or directories containing them
    /// (sharded output folders are expanded to their shard files).
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,

    /// Also run the numeric pass (weight bounds, scale finiteness/positivity,
    /// input_scale == 1.0). Reads tensor payloads, not just headers.
    #[arg(long)]
    pub numeric: bool,
}

// --------------------------------------------------------------------------- //
// info
// --------------------------------------------------------------------------- //

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// The `.safetensors` file to inspect.
    pub input: PathBuf,

    /// Print the raw JSON header instead of the summary table.
    #[arg(long)]
    pub raw: bool,
}

// --------------------------------------------------------------------------- //
// gguf
// --------------------------------------------------------------------------- //

#[derive(Args, Debug)]
pub struct GgufArgs {
    /// Input: a single `.safetensors` file OR a sharded HF model folder
    /// (containing `model.safetensors.index.json`). Optional only with
    /// `--list-methods`.
    pub input: Option<PathBuf>,

    /// Output `.gguf` file path. If omitted, auto-named
    /// `<base>-<method>.gguf` next to the input.
    pub output: Option<PathBuf>,

    /// GGUF quantization method id (e.g. `q4_k_m`, `q8_0`, `f16`). Use
    /// `--list-methods` to see all options.
    #[arg(long, short = 'm', default_value = "q4_k_m")]
    pub method: String,

    /// Override the GGUF architecture string (else detected from config.json).
    #[arg(long)]
    pub arch: Option<String>,

    /// Override the `general.name` metadata (else derived from the input).
    #[arg(long)]
    pub name: Option<String>,

    /// List all supported GGUF methods and exit.
    #[arg(long)]
    pub list_methods: bool,

    /// Disable the progress bar (plain, CI-friendly output).
    #[arg(long)]
    pub no_progress: bool,
}

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
/// `fp8_e4m3` / `mxfp8` / `nvfp4` are wired into the streaming
/// orchestrator (plan docs/plans/2026-08-28-all-formats-wiring-plan.md).
///
/// `int8_convrot` (plan Phase 7.1) is INT8 **row-wise** with a group-wise
/// Hadamard rotation applied to the weight before quantization, at a FIXED
/// group size of 256. `--scaling-mode` / `--block-size` do not apply: the
/// reference only rotates in row mode (`learned_rounding.py:869`), so the
/// preset forces it instead of silently producing an unrotated layer.
/// Layers whose `in_features` is not divisible by 256 stay plain row-wise
/// INT8 (the reference warns and leaves them unrotated).
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatArg {
    Int8,
    #[value(name = "fp8_e4m3")]
    Fp8E4m3,
    Mxfp8,
    Nvfp4,
    #[value(name = "int8_convrot")]
    Int8Convrot,
}

impl FormatArg {
    /// The CLI value name (used in error messages).
    pub fn as_str(&self) -> &'static str {
        match self {
            FormatArg::Int8 => "int8",
            FormatArg::Fp8E4m3 => "fp8_e4m3",
            FormatArg::Mxfp8 => "mxfp8",
            FormatArg::Nvfp4 => "nvfp4",
            FormatArg::Int8Convrot => "int8_convrot",
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

    /// Re-parse the output header(s) after a successful run and fail
    /// loudly (exit 1) if any tensor from the report is missing or
    /// unparseable. Header-only — cheap but not free on 10 GB outputs.
    #[arg(long)]
    pub verify_output: bool,
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

/// Conversion inputs/outputs: everything that defines HOW the GGUF is
/// produced. Split from [`GgufVerificationArgs`] so each group stays small
/// (`Commands::Gguf` was boxed for clippy::large_enum_variant once already).
#[derive(Args, Debug)]
pub struct GgufConversionArgs {
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

    /// Importance matrix file (GGUF or legacy binary) used by the weighted
    /// quantizers. REQUIRED for every `iq*` method — without it those
    /// methods exit 2 (the result would be garbage; same contract as
    /// llama-quantize and Unsloth's `imatrix_file=`).
    #[arg(long, value_name = "PATH")]
    pub imatrix: Option<PathBuf>,

    /// Per-tensor recipe file (Phase 6, the open `UD-*` equivalent):
    /// `regex=qtype` lines, `#` comments, first-match-wins against the
    /// GGUF tensor name, optional trailing bare `qtype` default.
    /// Every `qtype` must be a usable method id.
    #[arg(long, value_name = "PATH")]
    pub tensor_type_file: Option<PathBuf>,

    /// Override the quantization for the token-embedding tensor
    /// (`token_embd.weight`), e.g. `q8_0` or `f16`.
    #[arg(long, value_name = "METHOD")]
    pub token_embedding_type: Option<String>,

    /// Override the quantization for the output tensor (`output.weight`).
    #[arg(long, value_name = "METHOD")]
    pub output_tensor_type: Option<String>,
}

/// Post-conversion verification / extraction: everything that reads or
/// compares GGUF files rather than producing one.
#[derive(Args, Debug)]
pub struct GgufVerificationArgs {
    /// Dump the effective per-tensor scheme assignment as a recipe file
    /// (Phase 6.3, inspection): one `name-exact=qtype` line per tensor,
    /// in conversion order, plus a `# method <id>` header. Feeding the
    /// dump back via `--tensor-type-file` reproduces the same assignment.
    #[arg(long, value_name = "PATH")]
    pub emit_recipe: Option<PathBuf>,

    /// After conversion, compare the output against a REFERENCE GGUF
    /// (e.g. an unsloth / llama-quantize produced file) and print an
    /// equivalence report: per-tensor dtype diff (name-mapped), byte
    /// comparison, dead-block/scale-rule/genuine classification of
    /// payload differences, and a spec-conformance scan of our file.
    /// Exit 3 only if our file has spec violations.
    #[arg(long, value_name = "REFERENCE_GGUF")]
    pub verify_against: Option<PathBuf>,

    /// Extract the per-tensor dtype assignment from an existing GGUF
    /// (unsloth, llama-quantize, or our own output) and apply it: one
    /// exact-name rule per reference tensor, fed through the same
    /// machinery as --tensor-type-file (the row-width demotion guard
    /// still applies on top). Reference tensors missing from the input
    /// are ignored; input tensors missing from the reference keep the
    /// method default. F32 rules are skipped (1-D convention).
    #[arg(long, value_name = "REFERENCE_GGUF")]
    pub recipe_from: Option<PathBuf>,

    /// Override the GGUF architecture string (else detected from config.json).
    #[arg(long)]
    pub arch: Option<String>,

    /// Override the `general.name` metadata (else derived from the input).
    #[arg(long)]
    pub name: Option<String>,

    /// Audit an existing GGUF file instead of converting: dtype census +
    /// spec-conformance scan (gguf.cpp:724 per-row rule). Takes no
    /// INPUT/OUTPUT/--method. Exit 0 clean, 3 violations, 1 unparseable.
    #[arg(long, value_name = "FILE_GGUF", conflicts_with_all = ["tensor_type_file", "imatrix"])]
    pub audit: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct GgufArgs {
    #[command(flatten)]
    pub conversion: GgufConversionArgs,

    #[command(flatten)]
    pub verification: GgufVerificationArgs,

    /// List all supported GGUF methods and exit.
    #[arg(long)]
    pub list_methods: bool,

    /// Disable the progress bar (plain, CI-friendly output).
    #[arg(long)]
    pub no_progress: bool,
}

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
/// orchestrator (the all-formats wiring plan, local-only plan document).
///
/// # `--help` is a user-facing contract: how to read these variants
///
/// Every variant's help text states, explicitly, whether it is **byte-exact**
/// against the Python/torch reference and `llama-quantize`. That is the point
/// of this enum: a user must be able to tell a byte-exact format from a
/// quality-tuned one **without reading the source**.
///
/// Two distinct kinds of non-plain id exist, and they must never be conflated:
///
/// (a) **Rotation presets** (`int8_convrot`, `nvfp4_rot16`) change the emitted
/// bytes, but the rotation is a *parity-exact transform* — a different
/// algorithm, not a lower-fidelity one. Labelled **byte-exact**.
/// (b) **Quality modes** are labelled **NOT byte-exact**. `nvfp4_l2` and
/// `int8_clip09` are the ones reachable from `--format`; the rule is encoded
/// here and in the `parity:` marker line, which is sourced from
/// `quant_core::quality::Quality` rather than from any string in this file.
///
/// ⚠️ **`nvfp4_l2` is NOT byte-exact, on purpose.** It routes
/// `stream.rs`'s NVFP4 site to `quantize_nvfp4_weight_quality` (the anchored
/// alternating L2 scale search, measured at 17.7% aggregate relative-L2
/// improvement) instead of the frozen `quantize_nvfp4_weight`. It is gated on
/// `config.quality`, which is `Exact` for every other variant, so no existing
/// format moves a single byte. Its `Quality::id()` is non-`None`, so it also
/// gets a distinct `config_hash` — which is what stops a quality run from
/// resuming into a byte-exact partial file, since
/// `StreamState::load_manifest` trusts that hash alone.
///
/// `Quality::Mxfp8E8m0Compensated` is deliberately **not** reachable: it is
/// measured at exactly 0.00% improvement (a proven no-op), so wiring it would
/// offer a "quality" mode that changes nothing but the filename.
///
/// `int8_convrot` (plan Phase 7.1) is INT8 **row-wise** with a group-wise
/// Hadamard rotation applied to the weight before quantization, at a FIXED
/// group size of 256. `--scaling-mode` / `--block-size` do not apply: the
/// reference only rotates in row mode (`learned_rounding.py:869`), so the
/// preset forces it instead of silently producing an unrotated layer.
/// Layers whose `in_features` is not divisible by 256 stay plain row-wise
/// INT8 (the reference warns and leaves them unrotated).
///
/// # `nvfp4_rot16` — UNVERIFIED END TO END, DO NOT SHIP BLIND
///
/// `nvfp4_rot16` applies the same group-wise Hadamard rotation to NVFP4
/// weights at group size 16 -- chosen because the rotation group size should
/// EQUAL the quantizer block size (DuQuant++, arXiv:2604.17789; The Great
/// Inversion, arXiv:2608.25188) and NVFP4's block size is 16. The rotation
/// is a parity-exact transform, so this step changes no pinned digest; the
/// preset is distinguished by `convrot` / `convrot_group_size` in the
/// `config_hash`, never by a `Quality` variant.
///
/// **The rotation is applied OFFLINE to the weights only.** For the result to
/// be numerically usable, the consumer must apply the INVERSE rotation
/// (`x @ H`, blockwise) ONLINE at inference. Whether ComfyUI does this is
/// **UNKNOWN and untestable from this repository** -- no test here can retire
/// that risk, because the failure is not in the artifact we write but in
/// whether a downstream runtime honours it. If the consumer does NOT apply
/// the inverse, every rotated layer is garbage: a valid-looking file with
/// silently wrong numerics. Treat end-to-end correctness as UNVERIFIED until
/// someone confirms the consumer path out of band.
///
/// A second, narrower gap: the family-B `comfy_quant` blob schema has no
/// `convrot` / `convrot_groupsize` keys (only the INT8 family-A blob does), so
/// the emitted metadata does not record that this tensor was rotated. A
/// consumer therefore has no in-band signal to key off. See
/// `stream.rs` (`Format::Nvfp4` arm) for the rotation site.
///
/// Like `nvfp4`, this preset has FIXED scaling (block scaling at 16), so
/// `--scaling-mode` / `--block-size` remain usage errors.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatArg {
    /// `int8` — byte-exact vs the Python/torch reference.
    Int8,
    /// `fp8_e4m3` — byte-exact vs the Python/torch reference. Scaling mode and
    /// block size are selectable.
    #[value(name = "fp8_e4m3")]
    Fp8E4m3,
    /// `mxfp8` — byte-exact vs the Python/torch reference. Fixed block scaling
    /// at 32, so `--scaling-mode` / `--block-size` are usage errors.
    Mxfp8,
    /// `nvfp4` — byte-exact vs the Python/torch reference. Fixed block scaling
    /// at 16, so `--scaling-mode` / `--block-size` are usage errors.
    Nvfp4,
    /// `int8_convrot` — INT8 row-wise + group-wise Hadamard rotation at group
    /// size 256. **Byte-exact**, but a different algorithm from plain `int8`.
    /// The rotation is a parity-exact transform, not a quality trade, so this
    /// is NOT a quality-tuned format.
    #[value(name = "int8_convrot")]
    Int8Convrot,
    /// `nvfp4_rot16` — NVFP4 + group-wise Hadamard rotation at group size 16
    /// (== NVFP4's own block size). **Byte-exact**, for the same reason as
    /// `int8_convrot`: the rotation is parity-exact, so this is NOT a
    /// quality-tuned format. See the end-to-end warning above before using it.
    #[value(name = "nvfp4_rot16")]
    Nvfp4Rot16,
    /// `nvfp4_l2` — NVFP4 with the anchored alternating L2 scale search over
    /// the per-tensor / block scale pair. **NOT byte-exact**: this is the one
    /// `--format` value that selects a non-`Exact`
    /// [`quant_core::quality::Quality`], and it exists to buy reconstruction
    /// accuracy (measured 17.7% aggregate / 21-26% typical relative-L2
    /// improvement) at the cost of the parity guarantee.
    ///
    /// Everything else about it is inherited from `nvfp4`: same E2M1 codes,
    /// same fixed block scaling at 16, so `--scaling-mode` / `--block-size`
    /// remain usage errors. Only the *choice of scale grid point* differs.
    ///
    /// Reachability is gated on `config.quality` in `stream.rs`; because
    /// `Quality::Exact` is every other variant's quality, the `nvfp4` path
    /// there is byte-for-byte unchanged.
    #[value(name = "nvfp4_l2")]
    Nvfp4L2,
    /// `int8_clip09` — INT8 with absmax clipping: the quant scale is derived
    /// from `0.9 * row_max` instead of `row_max`, so the top 10% of the range
    /// saturates and everything below it gets a finer step. **NOT byte-exact**
    /// ([`quant_core::quality::Quality::Int8Clip09`]) — it deliberately
    /// abandons the reference's absmax scale.
    ///
    /// Unlike plain `int8`, this preset takes `--scaling-mode` /
    /// `--block-size` in all three modes (tensor / row / block): the clip is
    /// applied to whichever amax the chosen mode already computes, so there is
    /// no reason to constrain it.
    ///
    /// ⚠️ **MEASURED, AND THE MEASUREMENT IS NEGATIVE.** This preset is
    /// retained as a **documented negative result**, not as a recommended
    /// mode. Clipping's published benefit is a perplexity gain; this crate
    /// cannot compute perplexity, and clipping is usually said to help
    /// ACTIVATION outliers — which this weight-only path never quantizes.
    ///
    /// On the crate's own metric (weight-space `rel_l2` through the shipped
    /// dequantizer) it is not merely unimproved but **catastrophically worse**:
    /// 31x-14000x the error of plain INT8 depending on the distribution, 0 of 4
    /// improved. The reason is structural rather than a tuning failure — INT8's
    /// 127 levels already leave almost no rounding error for a finer step to
    /// recover, while the saturated tail's error is unbounded. The ratio is
    /// fixed at 0.9 and is NOT tunable: sweeping it until the benchmark passed
    /// would be fitting a threshold to the fixture rather than measuring the
    /// idea. See `tests/int8_clip09_negative_result.rs` and
    /// `benches/quality_error.rs`.
    #[value(name = "int8_clip09")]
    Int8Clip09,
}

impl FormatArg {
    /// Every variant, in declaration order.
    ///
    /// Read by the `#[cfg(test)]` modules in this crate (a BIN-only crate
    /// cannot export it to `tests/`) to assert pairwise-distinct `format_id`s,
    /// distinct auto-names, and a correct parity label for every variant — so a
    /// new variant cannot be added without also getting all three. `#[cfg_attr]`
    /// rather than a bare `#[allow]` because the constant IS used, just only
    /// under test; `dead_code` cannot see across the cfg boundary.
    #[cfg_attr(not(test), allow(dead_code))]
    pub const ALL: [FormatArg; 8] = [
        FormatArg::Int8,
        FormatArg::Fp8E4m3,
        FormatArg::Mxfp8,
        FormatArg::Nvfp4,
        FormatArg::Int8Convrot,
        FormatArg::Nvfp4Rot16,
        FormatArg::Nvfp4L2,
        FormatArg::Int8Clip09,
    ];

    /// The `Quality` refinement this `--format` value selects.
    ///
    /// This is the **single source of truth** for the `parity:` marker line:
    /// `quantize::parity_line` reads it rather than matching on [`FormatArg`]
    /// a second time, so a new preset cannot print an `exact` label it does
    /// not deserve.
    ///
    /// Six of the eight variants are [`Quality::Exact`]. The two rotation
    /// presets are byte-exact *despite* changing the emitted bytes, because a
    /// Hadamard rotation is an orthogonal transform: it changes which numbers
    /// land in which bucket, not the arithmetic. That distinction is exactly
    /// what a `Quality`-keyed accessor preserves and a "does the id look
    /// fancy" heuristic would not — the rotation presets are precisely the
    /// case that would be mislabelled `quality-tuned` by such a heuristic.
    ///
    /// Only `Nvfp4L2` and `Int8Clip09` are non-`Exact`. They are deliberately
    /// routed here and not from a per-format branch inside `stream.rs`,
    /// because the same table also feeds `config_hash` — so a quality run
    /// cannot share a resume guard with a parity run.
    pub fn quality(&self) -> quant_core::quality::Quality {
        use quant_core::quality::Quality;
        match self {
            FormatArg::Int8
            | FormatArg::Fp8E4m3
            | FormatArg::Mxfp8
            | FormatArg::Nvfp4
            | FormatArg::Int8Convrot
            | FormatArg::Nvfp4Rot16 => Quality::Exact,
            FormatArg::Nvfp4L2 => Quality::Nvfp4L2ScaleSearch,
            FormatArg::Int8Clip09 => Quality::Int8Clip09,
        }
    }

    /// The CLI value name (used in error messages).
    pub fn as_str(&self) -> &'static str {
        match self {
            FormatArg::Int8 => "int8",
            FormatArg::Fp8E4m3 => "fp8_e4m3",
            FormatArg::Mxfp8 => "mxfp8",
            FormatArg::Nvfp4 => "nvfp4",
            FormatArg::Int8Convrot => "int8_convrot",
            FormatArg::Nvfp4Rot16 => "nvfp4_rot16",
            FormatArg::Nvfp4L2 => "nvfp4_l2",
            FormatArg::Int8Clip09 => "int8_clip09",
        }
    }

    /// Formats with FIXED scaling parameters (block scaling at the format's
    /// own block size). Explicit `--scaling-mode` / `--block-size` for these
    /// is a usage error (exit 2).
    ///
    /// `Nvfp4Rot16` inherits NVFP4's fixed block scaling (it is NVFP4 plus a
    /// rotation, not a different scaling scheme), and so does `Nvfp4L2` (it
    /// is NVFP4 plus a scale *search*, not a different scaling scheme — the
    /// search only picks which E4M3 grid point the same block scale lands on).
    pub fn has_fixed_scaling(&self) -> bool {
        matches!(
            self,
            FormatArg::Mxfp8 | FormatArg::Nvfp4 | FormatArg::Nvfp4Rot16 | FormatArg::Nvfp4L2
        )
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

    /// Target format (`int8`, `fp8_e4m3`, `mxfp8`, `nvfp4`, `int8_convrot`,
    /// `nvfp4_rot16`, `nvfp4_l2`).
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

    /// Check the file against **ComfyUI's own loader contract** rather than the
    /// reference encoder's stricter one.
    ///
    /// The default pass validates what `convert_to_quant` produces; this
    /// validates what `comfy/ops.py` consumes. The two differ, so a file can
    /// pass one and fail the other. Specifically this rejects a `format` string
    /// that is not a key in ComfyUI's `QUANT_ALGOS` (which the loader indexes
    /// with no fallback, so it raises `KeyError`), and any format whose
    /// required sibling scale tensors are missing.
    ///
    /// A pass here means the file is **contract-conformant**, not proven
    /// loadable: only a real ComfyUI load settles whether the runtime applies
    /// a convrot inverse rotation.
    #[arg(long)]
    pub comfy: bool,
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
// cast
// --------------------------------------------------------------------------- //

/// Target float dtype for the `cast` subcommand.
///
/// Deliberately narrower than [`ScalingModeArg`]'s "everything" feel: `cast`
/// only ever produces one of these three, and mapping them to
/// [`quant_core::dtype::DType`] is the whole job.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastToArg {
    /// bfloat16 — 16-bit, 8 exponent bits. The VibeVoice target.
    Bf16,
    /// float16 — 16-bit, 5 exponent bits (max finite 65504).
    F16,
    /// float32 — 32-bit.
    F32,
}

#[derive(Args, Debug)]
pub struct CastArgs {
    /// Input: a single `.safetensors` file OR a sharded HF model folder
    /// (containing `model.safetensors.index.json`). A folder is merged into a
    /// single output file.
    pub input: PathBuf,

    /// Output `.safetensors` path. Defaults to `<base>-<tag>.safetensors`
    /// beside the input, where `<tag>` is the target dtype.
    pub output: Option<PathBuf>,

    /// Target float dtype for every float tensor in the model.
    #[arg(long = "to", value_enum)]
    pub to: CastToArg,

    /// Disable the progress bar (plain, CI-friendly output).
    #[arg(long)]
    pub no_progress: bool,
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

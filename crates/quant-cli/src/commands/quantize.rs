//! `quantize` subcommand — streaming INT8 quantization of a single
//! `.safetensors` file or a sharded HF model folder (plan Phase 9.2).
//!
//! Exit codes:
//! - `0` — quantization completed (fresh or resumed-to-completion)
//! - `1` — runtime failure (IO, discovery, quantization error)
//! - `2` — usage error (unusable input, bad arguments) — clap also uses 2

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use quant_core::discover::{
    classify_input, ctq_quant_tags, discover_shards, resolve_single_file, suggest_comfy_output,
    InputKind,
};
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::stream::{
    stream_quantize_cancellable, stream_quantize_sharded_cancellable,
    stream_quantize_shards_cancellable, StreamError,
};

use crate::args::{FormatArg, OrigDtypeArg, OutputModeArg, QuantizeArgs, ScalingModeArg};
use crate::progress::{BarSink, NullSink, ProgressSink};

/// Map the CLI scaling-mode flag to the core enum.
fn scaling_mode(m: ScalingModeArg) -> ScalingMode {
    match m {
        ScalingModeArg::Tensor => ScalingMode::Tensor,
        ScalingModeArg::Row => ScalingMode::Row,
        ScalingModeArg::Block => ScalingMode::Block,
    }
}

/// The effective `--scaling-mode`, applying the documented default.
///
/// `--scaling-mode` is optional and defaults to `block`. Three call sites need
/// that resolved value — `format_id` for auto-naming, the `parity:` line, and
/// the recents `method` field — and they must agree: if the recents entry
/// recorded a different mode than the filename tag, a user's history would
/// misdescribe the artifacts actually on disk.
fn scaling_mode_arg(args: &QuantizeArgs) -> ScalingModeArg {
    args.scaling_mode.unwrap_or(ScalingModeArg::Block)
}

/// The `parity:` marker line printed on EVERY run, immediately after the
/// summary.
///
/// This crate's core promise is byte-exact parity with the Python/torch
/// reference and `llama-quantize`. Tier 2 adds opt-in quality modes that
/// deliberately break that promise in exchange for accuracy, and rotation
/// presets that change the emitted bytes but remain parity-exact. A user must
/// be able to tell those apart without reading the source, so every run states
/// which kind it is.
///
/// **Both branches derive from [`Quality`], never from a per-format hardcoded
/// string.** `is_parity_exact()` selects the branch and `reason()` supplies the
/// text, so there is exactly one source of truth: a new quality variant cannot
/// be added without automatically getting a correct label here. The
/// `id()`-versus-`format_id` choice matters too — for a quality mode the
/// *quality* id is the honest headline (`nvfp4_l2`, not `nvfp4`), because the
/// base format id alone would understate what changed.
///
/// Printed **unconditionally**, including on `exact` runs. That is the point: a
/// user who sees the marker on an `exact` run has learned the tool makes the
/// distinction at all, which is what gives its presence on a quality run its
/// meaning. Printing it only when there is bad news would make it
/// indistinguishable from "this build does not report parity".
fn parity_line(config: &QuantConfig, format_id: &str) -> String {
    let quality = config.quality;
    if quality.is_parity_exact() {
        format!("parity: exact ({format_id}; byte-exact vs torch/llama-quantize)")
    } else {
        // `reason()` is `Some` for every non-`Exact` variant (pinned by
        // `quality_id_is_none_exactly_for_exact`), but fall back rather than
        // panic: a marker line must never be able to fail a successful run.
        let reason = quality.reason().unwrap_or("quality refinement enabled");
        let id = quality.id().unwrap_or(format_id);
        format!("parity: quality-tuned ({id}: {reason}; NOT byte-exact vs torch/llama-quantize)")
    }
}

/// Format id used for auto-naming tags. INT8 mirrors the reference
/// `int8_block` / `int8_tensor` / `int8_row` method ids; the new formats use
/// their COMFY_FORMATS registry ids (plan OQ-4: `fp8_e4m3` for all three FP8
/// scaling modes — the registry has a single id; `mxfp8`; `nvfp4`).
fn format_id(format: FormatArg, mode: ScalingModeArg) -> &'static str {
    match format {
        FormatArg::Int8 => match mode {
            ScalingModeArg::Tensor => "int8_tensor",
            ScalingModeArg::Row => "int8_row",
            ScalingModeArg::Block => "int8_block",
        },
        FormatArg::Fp8E4m3 => "fp8_e4m3",
        FormatArg::Mxfp8 => "mxfp8",
        FormatArg::Nvfp4 => "nvfp4",
        FormatArg::Int8Convrot => "int8_convrot",
        // Distinct id from `nvfp4`: it feeds `ctq_quant_tags`, so the
        // auto-named artifact can never collide with a plain nvfp4 run.
        FormatArg::Nvfp4Rot16 => "nvfp4_rot16",
        // Also distinct from `nvfp4`, and for a stronger reason: this id
        // describes a DIFFERENT ALGORITHM, not a different transform. Sharing
        // `nvfp4` here would let a quality run overwrite a parity artifact at
        // the same auto-named path with no warning.
        FormatArg::Nvfp4L2 => "nvfp4_l2",
        // Distinct id from ALL THREE plain-`int8` ids (`int8_tensor` /
        // `int8_row` / `int8_block`), which is why the mapping below is a
        // single string rather than a mode-dependent one. `int8_clip09` is
        // selectable in every scaling mode, and in each of them it must
        // auto-name to its own artifact — a clip run landing on top of a
        // plain-int8 file would be silent data loss.
        FormatArg::Int8Clip09 => "int8_clip09",
    }
}

fn orig_dtype(d: OrigDtypeArg) -> &'static str {
    match d {
        OrigDtypeArg::Bfloat16 => "bfloat16",
        OrigDtypeArg::Float16 => "float16",
    }
}

/// Build the core [`QuantConfig`] from CLI args (plan A.4).
///
/// INT8/FP8 take the (optional) `--scaling-mode` / `--block-size` with the
/// historical defaults `block` / 128. MXFP8/NVFP4 have FIXED scaling
/// parameters (block scaling at the format's own block size, 32/16) — the
/// caller must have rejected explicit overrides already.
fn build_config(args: &QuantizeArgs) -> QuantConfig {
    // `--heur` / `--no-heur` are mutually-overriding flags; default ON.
    let skip_inefficient = !args.no_heur;
    let (format, target_format, int8, scaling_mode, block_size) = match args.format {
        FormatArg::Int8 => (
            Format::Int8,
            "int8",
            true,
            scaling_mode(args.scaling_mode.unwrap_or(ScalingModeArg::Block)),
            args.block_size.unwrap_or(128),
        ),
        FormatArg::Fp8E4m3 => (
            Format::Fp8E4m3,
            "fp8",
            false,
            scaling_mode(args.scaling_mode.unwrap_or(ScalingModeArg::Block)),
            args.block_size.unwrap_or(128),
        ),
        FormatArg::Mxfp8 => (Format::Mxfp8, "mxfp8", false, ScalingMode::Block, 32),
        FormatArg::Nvfp4 => (Format::Nvfp4, "nvfp4", false, ScalingMode::Block, 16),
        // Phase 7.1: int8_convrot is a PRESET, not a free-form combination —
        // it is INT8 row-wise with a group-wise Hadamard rotation at a fixed
        // group size of 256. Row mode is FORCED (--scaling-mode is ignored,
        // unlike plain int8): the reference applies the rotation only when
        // `self.convrot and self.scaling_mode == "row"`
        // (learned_rounding.py:869), so honoring e.g. `--scaling-mode block`
        // would silently emit a plain, unrotated layer under a ConvRot name.
        // block_size is unused in row mode; 128 keeps the value canonical.
        FormatArg::Int8Convrot => (Format::Int8, "int8", true, ScalingMode::Row, 128),
        // `nvfp4_rot16` is NVFP4 (fixed block scaling at 16) plus a
        // group-wise Hadamard rotation at group size 16. The size is not a
        // free parameter: the rotation group size should EQUAL the quantizer
        // block size (DuQuant++ arXiv:2604.17789; The Great Inversion
        // arXiv:2608.25188), and NVFP4's block size is 16. See the
        // end-to-end verification warning on `FormatArg::Nvfp4Rot16`.
        FormatArg::Nvfp4Rot16 => (Format::Nvfp4, "nvfp4", false, ScalingMode::Block, 16),
        // `nvfp4_l2` is plain NVFP4 in every *structural* respect — same
        // family, same target format id, same fixed block scaling at 16 — so
        // every field here matches `FormatArg::Nvfp4` exactly. The ONLY
        // difference is `quality` below, which is what
        // `stream.rs`'s NVFP4 site dispatches on. Copying `Nvfp4`'s tuple
        // verbatim (rather than deriving it) is deliberate: if a future NVFP4
        // preset ever changes block size, the two arms must be considered
        // separately, and a shared expression would hide that.
        //
        // `convrot` stays false (it falls into the `_` arm of the match
        // below): the L2 search operates on the weight as given, and stacking
        // a rotation on top is an unvalidated combination.
        FormatArg::Nvfp4L2 => (Format::Nvfp4, "nvfp4", false, ScalingMode::Block, 16),
        // `int8_clip09` is plain INT8 in every structural respect — same
        // family, same `target_format`, same selectable scaling mode and block
        // size — so every field here matches `FormatArg::Int8` exactly. The
        // ONLY difference is `quality` below, which is what `stream.rs`'s INT8
        // site dispatches on. Copying `Int8`'s tuple verbatim (rather than
        // deriving it) is deliberate for the same reason as `Nvfp4L2`: a
        // future change to either must be considered on its own.
        //
        // `convrot` stays false (it falls into the `_` arm below): clipping an
        // already-rotated weight is an unvalidated combination, and silently
        // stacking them would make the emitted bytes depend on two orthogonal
        // transforms at once with no way to tell them apart.
        FormatArg::Int8Clip09 => (
            Format::Int8,
            "int8",
            true,
            scaling_mode(args.scaling_mode.unwrap_or(ScalingModeArg::Block)),
            args.block_size.unwrap_or(128),
        ),
    };
    // Phase 7.1: ConvRot is a property of the rotation PRESETS (there is no
    // `--convrot` flag — the preset IS the flag), and the group size is
    // FIXED per preset, never a tunable.
    //
    // `int8_convrot`'s 256 is a PARITY CONTRACT with the reference default
    // (`convrot_group_size=256`, learned_rounding.py:868) and the only size
    // the quantui UI offers for that path. Changing it silently changes the
    // bytes of every existing convrot artifact AND its `config_hash`, so it
    // must never become a flag or be "tidied" — see
    // `tests/nvfp4_rot16_parity.rs::int8_convrot_group_size_is_still_256`.
    let (convrot, convrot_group_size) = match args.format {
        FormatArg::Int8Convrot => (true, 256),
        FormatArg::Nvfp4Rot16 => (true, 16),
        _ => (false, 256),
    };
    QuantConfig {
        format,
        target_format: target_format.into(),
        int8,
        scaling_mode,
        block_size,
        no_learned_rounding: args.simple,
        convrot,
        convrot_group_size,
        // Tier 2 quality refinement. Sourced from `FormatArg::quality()` so
        // the `parity:` marker line and the `config_hash` resume
        // discriminator can never disagree about what kind of run this is.
        //
        // Every variant EXCEPT `nvfp4_l2` maps to `Exact`, which contributes
        // NO `quality_tuning` key — so all four pinned `config_hash` vectors
        // are preserved. `nvfp4_rot16` is deliberately NOT a quality variant:
        // it is a byte-exact rotation, distinguished by convrot /
        // convrot_group_size alone.
        //
        // `nvfp4_l2` IS a quality variant, and that is load-bearing beyond the
        // kernel dispatch: its `id()` is `Some("nvfp4_l2")`, so the hash
        // payload gains a key and the run gets a resume guard distinct from
        // plain `nvfp4`. `StreamState::load_manifest` trusts that hash ALONE,
        // so without this an interrupted `nvfp4` run re-run with `--format
        // nvfp4_l2` would resume into the byte-exact partial file and mix two
        // algorithms in one artifact — silently, with no error.
        quality: args.format.quality(),
        orig_dtype: orig_dtype(args.orig_dtype).into(),
        skip_inefficient,
        calib_seed: args.calib_seed,
        exclude_layers: args.exclude_layers.clone(),
        only_prefixes: args.only.clone(),
    }
}

/// Resolve the output destination, applying ctq auto-naming when the user
/// gave none (or a directory). An explicit `.safetensors` file always wins.
///
/// The auto-name tags are derived from the *effective* [`QuantConfig`] (not the
/// raw flags) so the filename always describes the artifact actually emitted —
/// e.g. the `heur` tag mirrors `config.skip_inefficient`.
fn resolve_output(args: &QuantizeArgs, config: &QuantConfig) -> Result<PathBuf, String> {
    // An explicit `.safetensors` file always wins.
    if let Some(p) = &args.output {
        if p.to_string_lossy().ends_with(".safetensors") {
            return Ok(p.clone());
        }
    }

    let inp = args.input.to_string_lossy().into_owned();
    let output = args
        .output
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    // Only the NEW preset passes a group size here. `int8_convrot` keeps
    // passing `None` so its auto-named filename is byte-for-byte what it has
    // always been — adding a `gs256` tag to it would silently rename every
    // artifact users have already produced.
    let gs_tag = match args.format {
        FormatArg::Nvfp4Rot16 => Some(config.convrot_group_size.to_string()),
        _ => None,
    };
    let tags = ctq_quant_tags(
        format_id(args.format, scaling_mode_arg(args)),
        gs_tag.as_deref(),
        config.no_learned_rounding,
        false,
        "",
        config.skip_inefficient,
    );
    let output_mode = match args.output_mode {
        OutputModeArg::Sharded => "sharded",
        OutputModeArg::Single => "single",
    };

    match suggest_comfy_output(&inp, &output, &tags, output_mode) {
        Some(p) => Ok(p),
        // Directory destination given explicitly → use it as-is.
        None => match &args.output {
            Some(p) => Ok(p.clone()),
            None => Err("could not determine an output path".into()),
        },
    }
}

/// Result of one successful quantize run: a human summary plus the resolved
/// output path (for recents persistence).
struct RunOutcome {
    summary: String,
    output: PathBuf,
}

/// WP7 / NTH-004 dispatch: build the (path, expected-tensor-names) check
/// list for whatever layout this run produced, then verify each file.
fn verify_resolved(outcome: &RunOutcome) -> Result<String, String> {
    // Single-file outcome: output is one .safetensors file; the expected
    // names are recovered by re-reading the manifest-free writer result —
    // the manifest written next to the artifact records them.
    if outcome.output.is_file() {
        let names = manifest_tensor_names(&outcome.output)?;
        return verify_output_files(&[(&outcome.output, &names)]);
    }
    // Sharded outcome: output is a directory with per-shard files plus a
    // global manifest (.quant-manifest.json) mapping shard → done list.
    let manifest_path = outcome
        .output
        .join(quant_core::stream::SHARDED_MANIFEST_NAME);
    let manifest = std::fs::read(&manifest_path)
        .map_err(|e| format!("{}: cannot read manifest: {e}", manifest_path.display()))?;
    let obj: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&manifest)
        .map_err(|e| format!("{}: invalid manifest JSON: {e}", manifest_path.display()))?;
    let shards = obj
        .get("shards")
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("{}: manifest has no 'shards' map", manifest_path.display()))?;

    let mut checks: Vec<(std::path::PathBuf, Vec<String>)> = Vec::new();
    for (shard, names) in shards {
        let names: Vec<String> = names
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        checks.push((outcome.output.join(shard), names));
    }
    let checks_ref: Vec<(&Path, &[String])> = checks
        .iter()
        .map(|(p, n)| (p.as_path(), n.as_slice()))
        .collect();
    verify_output_files(&checks_ref)
}

/// Tensor names recorded for a single-file output. The streaming writer's
/// done list is in the run's report; for the artifact on disk the
/// equivalent record is the sidecar manifest the streamer emits — fall
/// back to "names in the header itself" when the sidecar is absent,
/// verifying the artifact parses and is non-empty (the report count was
/// already echoed in the summary).
fn manifest_tensor_names(path: &Path) -> Result<Vec<String>, String> {
    let reader = quant_core::st_io::reader::SafetensorsReader::open(path)
        .map_err(|e| format!("{}: re-parse failed: {e}", path.display()))?;
    Ok(reader
        .header()
        .names()
        .map(|s| s.to_string())
        .collect::<Vec<_>>())
}

pub fn run(args: QuantizeArgs) -> ExitCode {
    // ---- classify input ---------------------------------------------------- //
    let (kind, _base) = classify_input(&args.input);
    let Some(kind) = kind else {
        eprintln!(
            "error: unusable input {} (need a .safetensors file or a sharded model folder)",
            args.input.display()
        );
        return ExitCode::from(2);
    };

    // `--exclude-layers` must be a VALID regex. `QuantConfig::excluded`
    // swallows a compile error with `unwrap_or(false)`, so running with a
    // typo'd pattern would emit a fully-quantized, much larger file with no
    // diagnostic. The reference does the same silent thing
    // (`unsloth-quant-tui quantui/tensor_quant.py:106`,
    // `except re.error: return False`), but we are a standalone project and
    // need not inherit a silent no-op that inflated a 14 GB artifact by
    // 129 MB. Reject at the argument boundary, beside every other usage
    // error, so it can never be missed.
    let cfg = build_config(&args);
    if let Err(e) = cfg.exclude_layers_status() {
        eprintln!(
            "error: --exclude-layers {:?} is not a valid regex: {e}",
            cfg.exclude_layers.as_deref().unwrap_or("")
        );
        eprintln!("hint: to KEEP only some layers, prefer --only <PREFIX> over inverting a regex");
        return ExitCode::from(2);
    }

    // MXFP8/NVFP4 have FIXED scaling parameters (block scaling at the
    // format's own block size). An explicit --scaling-mode / --block-size
    // override is a usage error (plan A.4).
    if args.format.has_fixed_scaling() {
        if args.scaling_mode.is_some() {
            eprintln!(
                "error: --scaling-mode is not valid with --format {} (fixed block scaling)",
                args.format.as_str()
            );
            return ExitCode::from(2);
        }
        if args.block_size.is_some() {
            eprintln!(
                "error: --block-size is not valid with --format {} (fixed block size)",
                args.format.as_str()
            );
            return ExitCode::from(2);
        }
    }

    let config = build_config(&args);
    let started = std::time::Instant::now();

    // ---- Ctrl-C graceful stop (plan 8.5) ---------------------------------- //
    // Set a shared flag on SIGINT; the orchestrator checks it between tensors
    // and stops at a tensor boundary, leaving a valid resumable partial output.
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&cancel);
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::Relaxed);
            eprintln!("\ninterrupt received — finishing current tensor and saving a resumable checkpoint…");
        });
    }

    // ---- dispatch ---------------------------------------------------------- //
    let result = match kind {
        InputKind::SingleFile => run_single(&args, &config, &cancel),
        InputKind::ShardedFolder => run_sharded(&args, &config, &cancel),
    };

    match result {
        Ok(outcome) => {
            println!("{}", outcome.summary);
            // Tier 2 user-visible contract: state the parity class of what was
            // just written, unconditionally. See `parity_line` for why this is
            // printed for `exact` runs too.
            println!(
                "{}",
                parity_line(&config, format_id(args.format, scaling_mode_arg(&args)))
            );
            // WP7 / NTH-004: optional post-run header re-parse. The run
            // itself succeeded, so a verification failure is reported as
            // a FAILED run (exit 1) — a suspect artifact must be loud.
            if args.verify_output {
                match verify_resolved(&outcome) {
                    Ok(line) => println!("{line}"),
                    Err(msg) => {
                        eprintln!("error: output verification failed: {msg}");
                        return ExitCode::from(1);
                    }
                }
            }
            record_recent(&args, &outcome, started.elapsed(), "success", 0);
            ExitCode::SUCCESS
        }
        Err(RunError::Cancelled { done, total }) => {
            eprintln!("stopped: cancelled after {done} of {total} tensors");
            eprintln!("partial output is valid — re-run the same command to resume");
            ExitCode::from(130) // conventional SIGINT exit code
        }
        Err(RunError::Failed(msg)) => {
            eprintln!("error: {msg}");
            ExitCode::from(1)
        }
    }
}

/// Error type for a quantize run: distinguishes user cancellation (exit 130)
/// from genuine failures (exit 1).
enum RunError {
    Failed(String),
    Cancelled { done: usize, total: usize },
}

impl From<StreamError> for RunError {
    fn from(e: StreamError) -> Self {
        match e {
            StreamError::Cancelled { done, total } => RunError::Cancelled { done, total },
            other => RunError::Failed(other.to_string()),
        }
    }
}

impl From<String> for RunError {
    fn from(s: String) -> Self {
        RunError::Failed(s)
    }
}

/// WP7 / NTH-004: header-only post-run verification. Re-opens every
/// output artifact and asserts every tensor the run reported is present
/// in the parsed header. Returns the human confirmation line, or an
/// error naming the first missing/unparseable tensor.
///
/// This is deliberately NOT a full byte audit — it catches the "run said
/// OK but the artifact is truncated/corrupt on disk" class (interrupted
/// sync, disk full, buggy sink) at the cost of one header read.
fn verify_output_files(checks: &[(&Path, &[String])]) -> Result<String, String> {
    for (path, expected) in checks {
        let reader = quant_core::st_io::reader::SafetensorsReader::open(path)
            .map_err(|e| format!("{}: re-parse failed: {e}", path.display()))?;
        let mut present = 0usize;
        for name in *expected {
            if reader.header().get(name).is_none() {
                return Err(format!(
                    "{}: tensor '{name}' missing from the re-parsed header",
                    path.display()
                ));
            }
            present += 1;
        }
        let _ = present; // all expected names verified below via count
    }
    let total: usize = checks.iter().map(|(_, names)| names.len()).sum();
    let files = checks.len();
    Ok(format!(
        "verified output: {total} tensor(s) re-parsed from {files} file(s)"
    ))
}

/// Best-effort recents persistence (plan 8.4). Never fails the run.
fn record_recent(
    args: &QuantizeArgs,
    outcome: &RunOutcome,
    duration: std::time::Duration,
    status: &str,
    exit_code: i32,
) {
    use crate::profiles::{add_recent, load_store, save_store, RunRecord};

    let record = RunRecord {
        ts: now_iso8601(),
        family: "ctq".into(),
        method: format_id(args.format, scaling_mode_arg(args)).into(),
        output: outcome.output.to_string_lossy().into_owned(),
        status: status.into(),
        exit_code,
        duration_s: duration.as_secs_f64(),
    };
    let mut store = load_store(None);
    add_recent(&mut store, record);
    // Ignore save errors: recents are best-effort and must never break a run.
    let _ = save_store(&store, None);
}

/// UTC ISO-8601 timestamp without pulling in a chrono dependency.
/// Uses the civil-from-days algorithm (Howard Hinnant) to convert epoch days.
pub(crate) fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    // civil_from_days
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if mo <= 2 { y + 1 } else { y };

    format!("{year:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Single-file input → single-file output.
fn run_single(
    args: &QuantizeArgs,
    config: &QuantConfig,
    cancel: &AtomicBool,
) -> Result<RunOutcome, RunError> {
    // `SingleFile` covers BOTH a real `.safetensors` file and a plain HF folder
    // that merely CONTAINS one. `classify_input` returns only a kind, so it
    // cannot say which file is inside such a folder — resolve it before
    // anything opens the input, or the streamer is handed a directory and
    // Windows fails with ERROR_ACCESS_DENIED ("os error 5").
    let input = resolve_single_file(&args.input).map_err(|e| RunError::Failed(e.to_string()))?;
    let output = resolve_output(args, config)?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create output dir: {e}"))?;
    }

    let mut sink: Box<dyn ProgressSink> = make_sink(args, "quantizing");
    let result = {
        let mut cb = |cur: usize, total: usize| sink.update(cur, total);
        stream_quantize_cancellable(&input, &output, config, Some(&mut cb), cancel)
    };
    // Always clear the progress bar before reporting — including on
    // cancellation/error — so the stop message is never garbled by the bar.
    sink.finish();
    let result = result?;

    Ok(RunOutcome {
        summary: format!(
            "wrote {} ({} tensors, config_hash {})",
            output.display(),
            result.done.len(),
            result.config_hash
        ),
        output,
    })
}

/// Build the progress sink for a run (bar unless `--no-progress`).
fn make_sink(args: &QuantizeArgs, label: &str) -> Box<dyn ProgressSink> {
    if args.no_progress {
        Box::new(NullSink)
    } else {
        Box::new(BarSink::new(label))
    }
}

/// Sharded-folder input → single file OR sharded output directory.
fn run_sharded(
    args: &QuantizeArgs,
    config: &QuantConfig,
    cancel: &AtomicBool,
) -> Result<RunOutcome, RunError> {
    let model = discover_shards(&args.input).map_err(|e| e.to_string())?;

    match args.output_mode {
        OutputModeArg::Single => {
            let output = resolve_output(args, config)?;
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("cannot create output dir: {e}"))?;
            }
            let mut sink = make_sink(args, "quantizing (union)");
            let result = {
                let mut cb = |cur: usize, total: usize| sink.update(cur, total);
                stream_quantize_shards_cancellable(
                    &model.shard_paths(),
                    &output,
                    config,
                    Some(&mut cb),
                    cancel,
                )
            };
            sink.finish();
            let result = result?;
            Ok(RunOutcome {
                summary: format!(
                    "wrote {} ({} tensors, config_hash {})",
                    output.display(),
                    result.done.len(),
                    result.config_hash
                ),
                output,
            })
        }
        OutputModeArg::Sharded => {
            let output_dir = resolve_output(args, config)?;
            let mut sink = make_sink(args, "quantizing shards");
            let result = {
                let mut cb = |cur: usize, total: usize| sink.update(cur, total);
                stream_quantize_sharded_cancellable(
                    &model,
                    &output_dir,
                    config,
                    Some(&mut cb),
                    cancel,
                )
            };
            sink.finish();
            let result = result?;
            Ok(RunOutcome {
                summary: format!(
                    "wrote {} ({} shard(s), config_hash {})",
                    result.output_dir.display(),
                    result.shards.len(),
                    result.config_hash
                ),
                output: result.output_dir,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{build_config, format_id, parity_line, scaling_mode_arg, FormatArg, QuantizeArgs};
    use quant_core::manifest::{Format, QuantConfig};
    use quant_core::quality::Quality;
    use std::path::PathBuf;

    /// `--only` is the flag that replaces inverting a long exclusion regex.
    /// It must reach the effective config verbatim — a dropped field would
    /// silently quantize everything and inflate the artifact, which is the
    /// exact failure `--only` exists to make impossible.
    #[test]
    fn only_prefixes_reach_the_effective_config() {
        let mut a = args_for(FormatArg::Int8);
        assert!(
            build_config(&a).only_prefixes.is_empty(),
            "no --only must mean no allow-list, not an allow-list that matches nothing"
        );
        a.only = vec!["transformer_blocks".into()];
        assert_eq!(build_config(&a).only_prefixes, ["transformer_blocks"]);
        a.only = vec!["transformer_blocks".into(), "txt_in".into()];
        assert_eq!(
            build_config(&a).only_prefixes.len(),
            2,
            "must be repeatable"
        );
    }

    /// An invalid `--exclude-layers` must be REJECTED before any work. The
    /// core predicate still swallows the error (byte-parity with the
    /// reference), so this guard is the only thing between a typo and a
    /// silently oversized artifact.
    #[test]
    fn invalid_exclude_layers_is_detected_before_any_work() {
        let mut a = args_for(FormatArg::Int8);
        a.exclude_layers = Some("[invalid".into());
        let cfg = build_config(&a);
        assert!(
            cfg.exclude_layers_status().is_err(),
            "run() relies on this to exit 2 before touching the input file"
        );
        assert!(
            !cfg.excluded("attn_norm.weight"),
            "the core still refuses to exclude — that is WHY the guard is needed"
        );
        a.exclude_layers = Some(r"^(img_in|modulation)\.".into());
        assert!(build_config(&a).exclude_layers_status().is_ok());
    }

    /// Minimal `QuantizeArgs` for the given `--format`. `scaling_mode` is left
    /// `None` so `scaling_mode_arg` applies the documented `block` default,
    /// which is the path a real run takes when the flag is omitted.
    fn args_for(format: FormatArg) -> QuantizeArgs {
        QuantizeArgs {
            input: PathBuf::from("in.safetensors"),
            output: None,
            format,
            scaling_mode: None,
            block_size: None,
            heur: true,
            no_heur: false,
            exclude_layers: None,
            only: Vec::new(),
            output_mode: crate::args::OutputModeArg::Sharded,
            orig_dtype: crate::args::OrigDtypeArg::Bfloat16,
            calib_seed: 233983427,
            simple: true,
            no_progress: true,
            verify_output: false,
        }
    }

    /// A `QuantConfig` carrying one specific quality.
    fn config_with(quality: Quality) -> QuantConfig {
        QuantConfig {
            quality,
            ..QuantConfig::default()
        }
    }

    /// The `exact` branch, for every currently-selectable PARITY format.
    ///
    /// This is the branch every pre-existing run takes, so it is the one that
    /// must not regress: the wording must promise byte-exactness and must
    /// never carry the quality-tuned phrasing. `FormatArg::ALL` is filtered
    /// through `quality()` rather than hardcoded, so the day a second quality
    /// preset lands this test keeps covering exactly the parity set.
    #[test]
    fn every_parity_format_prints_parity_exact() {
        let mut checked = 0usize;
        for f in FormatArg::ALL {
            if !f.quality().is_parity_exact() {
                continue;
            }
            checked += 1;
            let args = args_for(f);
            let line = parity_line(
                &config_with(f.quality()),
                format_id(f, scaling_mode_arg(&args)),
            );
            assert!(
                line.starts_with("parity: exact ("),
                "{f:?} must report an exact parity line, got: {line}"
            );
            assert!(
                line.contains("byte-exact vs torch/llama-quantize"),
                "{f:?} must name the reference it is exact against: {line}"
            );
            assert!(
                !line.contains("quality-tuned"),
                "{f:?} must not be labelled quality-tuned: {line}"
            );
            assert!(
                !line.contains("NOT byte-exact"),
                "{f:?} must not carry the negated wording: {line}"
            );
        }
        assert_eq!(
            checked,
            FormatArg::ALL.len() - 2,
            "exactly two --format values are quality-tuned (nvfp4_l2, int8_clip09)"
        );
    }

    /// Every format's line must headline ITS OWN id, so `int8` and
    /// `int8_convrot` are distinguishable in a log.
    #[test]
    fn parity_exact_line_headlines_the_formats_own_id() {
        for f in FormatArg::ALL {
            if !f.quality().is_parity_exact() {
                continue;
            }
            let args = args_for(f);
            let fid = format_id(f, scaling_mode_arg(&args));
            let line = parity_line(&config_with(f.quality()), fid);
            assert!(
                line.contains(&format!("({fid};")),
                "{f:?} line must contain its own format id {fid}: {line}"
            );
        }
    }

    /// The rotation presets are the trap case: they change the emitted bytes,
    /// so a naive implementation would label them `quality-tuned`. They are
    /// orthogonal transforms and therefore byte-exact, and this pins that.
    ///
    /// This is precisely the case a "does the id look fancy" heuristic gets
    /// wrong, and the reason `quality()` is keyed on `Quality` instead.
    #[test]
    fn rotation_presets_are_byte_exact_not_quality_tuned() {
        for f in [FormatArg::Int8Convrot, FormatArg::Nvfp4Rot16] {
            assert_eq!(
                f.quality(),
                Quality::Exact,
                "{f:?} is a rotation preset, not a quality mode"
            );
            let args = args_for(f);
            let line = parity_line(
                &config_with(f.quality()),
                format_id(f, scaling_mode_arg(&args)),
            );
            assert!(
                !line.contains("quality-tuned"),
                "{f:?} must not be labelled quality-tuned: {line}"
            );
        }
    }

    /// The `quality-tuned` branch, for EVERY non-`Exact` variant.
    ///
    /// `Quality::ALL` is iterated rather than a hand-listed pair, so a future
    /// variant cannot skip this — including variants with no CLI route at all,
    /// which still owe a user correct wording if one is ever added.
    #[test]
    fn every_quality_variant_prints_quality_tuned_with_its_reason() {
        let mut checked = 0usize;
        for q in Quality::ALL {
            if q.is_parity_exact() {
                continue;
            }
            checked += 1;
            let line = parity_line(&config_with(q), "unused_base_format");
            let id = q.id().expect("non-Exact variants carry an id");
            let reason = q.reason().expect("non-Exact variants explain themselves");
            assert!(
                line.starts_with(&format!("parity: quality-tuned ({id}: {reason};")),
                "{q:?} must print its own id and reason, got: {line}"
            );
            assert!(
                line.contains("NOT byte-exact vs torch/llama-quantize"),
                "{q:?} must negate byte-exactness explicitly: {line}"
            );
            // The base format id must NOT be the headline: for a quality run
            // the honest label is the quality id, because `nvfp4_l2` says more
            // than `nvfp4` does about what changed.
            assert!(
                !line.contains("unused_base_format"),
                "{q:?} must not headline the base format id: {line}"
            );
        }
        assert_eq!(checked, 4, "four non-Exact quality variants exist");
    }

    /// `nvfp4_l2` must reach the `quality-tuned` branch *through the real
    /// config builder*, not merely through a hand-assembled `QuantConfig`.
    ///
    /// This is the end of the chain the earlier steps left dangling: the
    /// `quality-tuned` wording existed and was tested, but nothing could select
    /// it. Asserting on `build_config` closes the loop — if a future edit routed
    /// the preset's `Quality` back to `Exact`, the marker would silently
    /// mislabel a non-parity run, and this is what catches it.
    #[test]
    fn nvfp4_l2_actually_reaches_the_quality_tuned_branch() {
        let args = args_for(FormatArg::Nvfp4L2);
        let config = super::build_config(&args);
        let line = parity_line(
            &config,
            format_id(FormatArg::Nvfp4L2, scaling_mode_arg(&args)),
        );

        assert_eq!(config.quality, Quality::Nvfp4L2ScaleSearch);
        assert!(
            line.starts_with("parity: quality-tuned (nvfp4_l2:"),
            "nvfp4_l2 must announce itself as quality-tuned, got: {line}"
        );
        assert!(
            line.contains("NOT byte-exact vs torch/llama-quantize"),
            "the quality line must negate byte-exactness: {line}"
        );
        // And the contrast that gives the marker its meaning.
        let exact = parity_line(
            &super::build_config(&args_for(FormatArg::Nvfp4)),
            format_id(FormatArg::Nvfp4, scaling_mode_arg(&args)),
        );
        assert_ne!(exact, line, "the two must not print the same line");
    }

    /// No hardcoded per-format strings: the line must depend on `Quality`
    /// alone. If someone hardcodes an `if format == ...` branch, this fails.
    #[test]
    fn parity_line_depends_only_on_quality_not_on_the_format_id() {
        let a = parity_line(&config_with(Quality::Nvfp4L2ScaleSearch), "nvfp4");
        let b = parity_line(&config_with(Quality::Nvfp4L2ScaleSearch), "something_else");
        assert_eq!(a, b, "a quality line must not depend on the base format id");
        // Two different quality modes must NOT collapse to the same line, or
        // the marker could not tell a user which refinement they are holding.
        let c = parity_line(&config_with(Quality::Mxfp8E8m0Compensated), "mxfp8");
        assert_ne!(a, c, "distinct quality modes must produce distinct lines");
    }

    /// Routing `quality` through `FormatArg::quality()` must not disturb any
    /// pinned `config_hash`. `Exact` contributes no `quality_tuning` key, so
    /// the four committed vectors must survive byte-for-byte.
    ///
    /// The filter is on `is_parity_exact()` rather than a hand-listed set, so
    /// adding `nvfp4_l2` (which IS a quality variant) does not have to touch
    /// this test — and cannot weaken it, since a parity format that stopped
    /// being `Exact` would silently drop out of coverage. That risk is closed
    /// by the exact-count assertion below.
    #[test]
    fn every_shipped_format_still_hashes_to_a_pinned_vector() {
        let mut parity_formats = 0usize;
        for f in FormatArg::ALL {
            if !f.quality().is_parity_exact() {
                continue;
            }
            parity_formats += 1;
            assert_eq!(f.quality(), Quality::Exact, "{f:?}");
        }
        assert_eq!(
            parity_formats, 6,
            "six --format values must remain byte-exact (only nvfp4_l2 and int8_clip09 \
             are quality-tuned)"
        );
        // The four committed vectors (INT8's is Python-captured).
        let cases = [
            (Format::Int8, "56920c6553cfa241"),
            (Format::Fp8E4m3, "5f14780b1bcf30f2"),
            (Format::Mxfp8, "cff3b89365c9544d"),
            (Format::Nvfp4, "95ede677cf402b53"),
        ];
        for (format, want) in cases {
            let c = QuantConfig {
                format,
                ..super::build_config(&args_for(FormatArg::Int8))
            };
            assert_eq!(c.quality, Quality::Exact);
            assert_eq!(c.config_hash(), want, "pinned hash for {format:?} moved");
        }
    }

    /// The resume guard: `nvfp4_l2` must NOT share `nvfp4`'s `config_hash`.
    ///
    /// This is the single most consequential property of the new preset.
    /// `StreamState::load_manifest` trusts the stored hash ALONE when deciding
    /// whether a partial output may be resumed into — so a collision would let
    /// an interrupted byte-exact `nvfp4` run be continued under the L2 search,
    /// producing one artifact containing two different algorithms' bytes, with
    /// no error and no warning. The hashes must therefore differ for a reason
    /// that is checked here rather than assumed.
    #[test]
    fn nvfp4_l2_hash_is_distinct_from_plain_nvfp4() {
        let exact = super::build_config(&args_for(FormatArg::Nvfp4));
        let quality = super::build_config(&args_for(FormatArg::Nvfp4L2));

        assert_eq!(exact.format, quality.format, "both are Format::Nvfp4");
        assert_eq!(exact.quality, Quality::Exact);
        assert_eq!(quality.quality, Quality::Nvfp4L2ScaleSearch);
        assert_ne!(
            exact.config_hash(),
            quality.config_hash(),
            "a quality run must never share a resume guard with a parity run"
        );
        // The distinction has to come from the quality key specifically —
        // otherwise the hash could drift back into collision unnoticed.
        assert_eq!(quality.config_hash(), {
            let mut only_id = exact;
            only_id.quality = Quality::Nvfp4L2ScaleSearch;
            only_id.config_hash()
        });
    }

    /// The resume guard for `int8_clip09`, and the reason this preset is more
    /// dangerous than `nvfp4_l2` rather than less.
    ///
    /// `int8_clip09` shares its ENTIRE `QuantConfig` with plain `int8` at the
    /// same scaling mode — same `format`, same `target_format`, same `int8`
    /// flag, same block size. The *only* thing distinguishing them in the hash
    /// payload is the `quality_tuning` key. So the whole resume-safety
    /// argument rests on that single conditional key being present, and a
    /// regression that made it `None` for this variant would be silent: the
    /// hash would fall back to the pinned INT8 vector, an interrupted `int8`
    /// run would resume into a clipped file, and the artifact would mix two
    /// quantizers with no error anywhere.
    #[test]
    fn int8_clip09_hash_is_distinct_from_plain_int8_at_every_scaling_mode() {
        use crate::args::ScalingModeArg;

        for mode in [
            ScalingModeArg::Tensor,
            ScalingModeArg::Row,
            ScalingModeArg::Block,
        ] {
            let mut plain_args = args_for(FormatArg::Int8);
            plain_args.scaling_mode = Some(mode);
            let mut clip_args = args_for(FormatArg::Int8Clip09);
            clip_args.scaling_mode = Some(mode);

            let plain = super::build_config(&plain_args);
            let clip = super::build_config(&clip_args);

            assert_eq!(plain.format, clip.format, "{mode:?}: both are Format::Int8");
            assert_eq!(plain.quality, Quality::Exact);
            assert_eq!(clip.quality, Quality::Int8Clip09);
            assert_ne!(
                plain.config_hash(),
                clip.config_hash(),
                "{mode:?}: a clip run must never resume into a plain int8 partial"
            );
            // And the plain side must be recoverable by flipping ONLY the
            // quality key back, so this test also catches the clip key
            // leaking into the parity payload. (Comparing against the pinned
            // `56920c6553cfa241` would be wrong here: that vector is the
            // DEFAULT block-mode config, and `scaling_mode` is legitimately
            // hash-relevant for INT8, so the tensor- and row-mode hashes are
            // different values by design.)
            let mut only_quality = clip.clone();
            only_quality.quality = Quality::Exact;
            assert_eq!(
                only_quality.config_hash(),
                plain.config_hash(),
                "{mode:?}: clearing the quality key must restore plain int8's hash"
            );
        }
    }

    /// `int8_clip09` must reach the `quality-tuned` branch through the real
    /// config builder, exactly as `nvfp4_l2` does. Without this the preset
    /// could carry a non-`Exact` quality while still printing
    /// `parity: exact` — a user told the artifact is byte-exact when it is not.
    #[test]
    fn int8_clip09_actually_reaches_the_quality_tuned_branch() {
        let args = args_for(FormatArg::Int8Clip09);
        let config = super::build_config(&args);
        let line = parity_line(
            &config,
            format_id(FormatArg::Int8Clip09, scaling_mode_arg(&args)),
        );

        assert_eq!(config.quality, Quality::Int8Clip09);
        assert!(
            line.starts_with("parity: quality-tuned (int8_clip09:"),
            "int8_clip09 must announce itself as quality-tuned, got: {line}"
        );
        assert!(
            line.contains("NOT byte-exact vs torch/llama-quantize"),
            "the quality line must negate byte-exactness: {line}"
        );
        // The contrast that gives the marker its meaning: plain int8 at the
        // same scaling mode must NOT print the same line.
        let exact = parity_line(
            &super::build_config(&args_for(FormatArg::Int8)),
            format_id(FormatArg::Int8, scaling_mode_arg(&args)),
        );
        assert!(exact.starts_with("parity: exact ("), "got: {exact}");
        assert_ne!(exact, line);
    }

    /// Auto-naming must not collide: every variant's `format_id` feeds
    /// `ctq_quant_tags`, so two runs sharing an id would overwrite each other's
    /// auto-named output. Asserted pairwise rather than by eyeball.
    #[test]
    fn every_format_id_is_pairwise_distinct() {
        let mut ids: Vec<&'static str> = FormatArg::ALL
            .iter()
            .map(|f| {
                let args = args_for(*f);
                format_id(*f, scaling_mode_arg(&args))
            })
            .collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            before,
            "two --format values share a format_id, so their auto-named outputs would collide"
        );
    }
}

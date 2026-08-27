//! `quantize` subcommand — streaming INT8 quantization of a single
//! `.safetensors` file or a sharded HF model folder (plan Phase 9.2).
//!
//! Exit codes:
//! - `0` — quantization completed (fresh or resumed-to-completion)
//! - `1` — runtime failure (IO, discovery, quantization error)
//! - `2` — usage error (unusable input, bad arguments) — clap also uses 2

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use quant_core::discover::{
    classify_input, ctq_quant_tags, discover_shards, suggest_comfy_output, InputKind,
};
use quant_core::manifest::{QuantConfig, ScalingMode};
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

/// Format id used for auto-naming tags (mirrors reference `int8_block` /
/// `int8_tensor` / `int8_row` method ids).
fn format_id(m: ScalingModeArg) -> &'static str {
    match m {
        ScalingModeArg::Tensor => "int8_tensor",
        ScalingModeArg::Row => "int8_row",
        ScalingModeArg::Block => "int8_block",
    }
}

fn orig_dtype(d: OrigDtypeArg) -> &'static str {
    match d {
        OrigDtypeArg::Bfloat16 => "bfloat16",
        OrigDtypeArg::Float16 => "float16",
    }
}

/// Build the core [`QuantConfig`] from CLI args.
fn build_config(args: &QuantizeArgs) -> QuantConfig {
    // `--heur` / `--no-heur` are mutually-overriding flags; default ON.
    let skip_inefficient = !args.no_heur;
    QuantConfig {
        target_format: "int8".into(),
        int8: true,
        scaling_mode: scaling_mode(args.scaling_mode),
        block_size: args.block_size,
        no_learned_rounding: args.simple,
        convrot: false,
        convrot_group_size: 256,
        orig_dtype: orig_dtype(args.orig_dtype).into(),
        skip_inefficient,
        calib_seed: args.calib_seed,
        exclude_layers: args.exclude_layers.clone(),
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
    let tags = ctq_quant_tags(
        format_id(args.scaling_mode),
        None,
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

    if args.format != FormatArg::Int8 {
        eprintln!("error: only `int8` is supported by the streaming quantizer");
        return ExitCode::from(2);
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
        method: format_id(args.scaling_mode).into(),
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
    let output = resolve_output(args, config)?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create output dir: {e}"))?;
    }

    let mut sink: Box<dyn ProgressSink> = make_sink(args, "quantizing");
    let result = {
        let mut cb = |cur: usize, total: usize| sink.update(cur, total);
        stream_quantize_cancellable(&args.input, &output, config, Some(&mut cb), cancel)?
    };
    sink.finish();

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
                )?
            };
            sink.finish();
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
                )?
            };
            sink.finish();
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

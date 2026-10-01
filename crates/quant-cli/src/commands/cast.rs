//! `cast` subcommand — dtype-cast a model into a single-file
//! bf16/fp16/fp32 `.safetensors`.
//!
//! # Why this exists separately from `quantize`
//!
//! `quantize`'s output contract REQUIRES ComfyUI quantization metadata: the
//! `.comfy_quant` blob, `weight_scale` tensors, and
//! `__metadata__._quantization_metadata`. A plain bf16 file must **not** carry
//! those — a file that advertises itself as ComfyUI-quantized while containing
//! nothing quantized loads as a broken model. So this command has its own
//! contract, and the single most important invariant it maintains is:
//!
//! > The output contains **no quantization metadata whatsoever**.
//!
//! The only metadata written is `{"format": "pt"}`, which is what
//! `safetensors.torch.save_file` writes and what the ComfyUI/transformers
//! loaders expect. It is a format tag, not a quantization marker.
//!
//! # Reporting
//!
//! This command prints a `cast:` summary and **never** a `parity:` line. In
//! this repo `parity:` is a load-bearing marker with a fixed meaning
//! (`Quality::is_parity_exact()` on `quantize` runs — either `exact` or
//! `quality-tuned`). A cast is neither: f32→bf16 is not reversible, so
//! claiming byte-exact parity would be false. Emitting `parity:` here would
//! dilute the marker that makes it meaningful everywhere else.
//!
//! # Exit codes
//!
//! | code | meaning |
//! |---|---|
//! | 0 | success |
//! | 1 | data error — bf16/f16 overflow refusal, f64 source, unreadable input |
//! | 2 | usage error — input is neither a file nor a sharded folder |
//! | 130 | SIGINT |

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use quant_core::cast::{cast_tensor, output_dtype_for, CastOutcome};
use quant_core::discover::{self, InputKind};
use quant_core::dtype::DType;
use quant_core::st_io::reader::SafetensorsReader;
use quant_core::st_io::writer::IncrementalWriter;
use serde_json::{Map, Value};

use crate::args::{CastArgs, CastToArg};
use crate::progress::{BarSink, NullSink, ProgressSink};

/// Bytes to reserve for the header slot up front.
///
/// The reference model is 1204 tensors; each entry costs roughly 100 bytes of
/// JSON, so ~256 KiB covers it with room to spare. The writer grows the slot
/// automatically if this is too small (`writer.rs:211`), so this is purely an
/// optimization that avoids a full-file rewrite mid-run.
const HEADER_SLOT_HINT: usize = 1 << 18;

/// Map the `--to` flag onto a [`DType`].
fn target_dtype(to: CastToArg) -> DType {
    match to {
        CastToArg::Bf16 => DType::Bf16,
        CastToArg::F16 => DType::F16,
        CastToArg::F32 => DType::F32,
    }
}

/// Short tag used in the default output filename.
fn target_tag(to: CastToArg) -> &'static str {
    match to {
        CastToArg::Bf16 => "bf16",
        CastToArg::F16 => "f16",
        CastToArg::F32 => "f32",
    }
}

/// Derive the output path: `<base>-<tag>.safetensors` beside the input.
///
/// Mirrors the `suggest_comfy_output` convention (`discover.rs:401`). The base
/// comes from `classify_input`, which already handles both the single-file
/// stem and the sharded-folder name.
fn derive_output(input: &Path, to: CastToArg) -> Result<PathBuf, String> {
    let base = discover::classify_input(input).1.ok_or_else(|| {
        format!(
            "{} is neither a .safetensors file nor a sharded model folder",
            input.display()
        )
    })?;
    // "Beside the input" means: next to the FILE, or alongside the FOLDER —
    // never inside it. Writing into the source folder would mutate the very
    // directory this command promises to only read, and would put the merged
    // output where a re-run of the source discovery could trip over it.
    let dir = if input.is_dir() {
        input
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    } else {
        input
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    };
    Ok(dir.join(format!("{base}-{}.safetensors", target_tag(to))))
}

/// The only metadata this command is allowed to write.
///
/// Deliberately a fixed, minimal map. Adding a quantization key here would make
/// every cast output claim to be a quantized model.
fn output_metadata() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("format".into(), Value::String("pt".into()));
    m
}

/// Print the `cast:` summary (never a `parity:` line).
///
/// BYTE-STABLE BY CONTRACT: nothing derived from wall-clock time, the
/// filesystem, or iteration order of a hash map may appear here, so two runs of
/// the same input print identical bytes. `quantize` and `gguf` both keep their
/// printed summaries stable for the same reason — elapsed time is recorded into
/// the recent-runs history instead, and the live progress bar shows
/// `{elapsed_precise}` while the run is in flight, which is where the user is
/// actually looking. A varying field here would make `cast` the only command
/// whose summary cannot be grepped or diffed across runs.
fn report(total: usize, verbatim: usize, converted: usize, src_desc: &str, dst: DType) {
    if converted == 0 {
        // Name the dtypes here too. This is the same-dtype branch, and for the
        // all-bf16 target model it is the COMMON case — so it is exactly the
        // run that must not leave the reader guessing what was copied.
        println!("cast: {total} tensors, lossless ({src_desc} -> {dst}, verbatim copy)");
    } else if verbatim == 0 {
        println!(
            "cast: {total} tensors, {converted} converted \
             ({src_desc} -> {dst}, round-to-nearest-even)"
        );
    } else {
        println!(
            "cast: {total} tensors, {verbatim} verbatim, {converted} converted \
             ({src_desc} -> {dst}, round-to-nearest-even)"
        );
    }
}

/// Temporary sibling path that the real output is renamed from.
///
/// `IncrementalWriter::open_new_with` calls `File::create` immediately, so the
/// destination path exists from the first moment. If the cast then fails
/// (e.g. the f16 overflow refusal), the writer is dropped WITHOUT `finalize()`
/// and a half-built file survives at the destination: a `.safetensors` whose
/// header parses as valid JSON but which contains zero tensors. To a loader
/// that is worse than no file at all.
///
/// So the writer is pointed at a temporary path and renamed into place only
/// after `finalize()` succeeds. Same directory as the destination so the
/// rename is atomic (a cross-device rename would not be, and would silently
/// degrade to copy+delete).
fn temp_path_for(out: &Path) -> PathBuf {
    let stem = out
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out.safetensors".to_string());
    let dir = out.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!(".{stem}.cast-tmp"))
}

/// Why a run failed, which decides the exit code.
///
/// The spec fixes the mapping: a USAGE error (bad path, not a safetensors file
/// or sharded folder) exits 2, while a DATA error (the model cannot be
/// represented — f16 overflow, f64 source) exits 1. Collapsing both to 1 would
/// make a typo in a path look like a corrupt model.
struct CliError {
    message: String,
    usage: bool,
}

impl CliError {
    fn usage(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            usage: true,
        }
    }

    fn data(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            usage: false,
        }
    }
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        // A bare string from the write path is a DATA failure (overflow
        // refusal, IO error on a well-formed model).
        Self::data(message)
    }
}

pub fn run(args: CastArgs) -> ExitCode {
    match run_inner(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {}", e.message);
            ExitCode::from(if e.usage { 2 } else { 1 })
        }
    }
}

fn run_inner(args: CastArgs) -> Result<ExitCode, CliError> {
    let target = target_dtype(args.to);

    // --- Input classification (reuses discover; never reimplemented) --------
    let (kind, base) = discover::classify_input(&args.input);
    let kind = kind.ok_or_else(|| {
        CliError::usage(format!(
            "{} is neither a .safetensors file nor a sharded model folder",
            args.input.display()
        ))
    })?;
    let _ = base; // only needed for the default output name, computed below

    let out_path = match &args.output {
        Some(p) => p.clone(),
        None => derive_output(&args.input, args.to).map_err(CliError::usage)?,
    };

    // --- Build the tensor-name -> shard map --------------------------------
    // For a single file this is one shard holding everything. For a sharded
    // folder we use the union header so names map to the right shard, and the
    // ORDER is the union's first-appearance order (stable across runs, which
    // is what makes the output byte-deterministic).
    let shard_paths: Vec<PathBuf> = match kind {
        // `SingleFile` may name a FOLDER that merely *contains* one
        // `.safetensors` (plain HF layout), so resolve to the real file before
        // anything opens it — handing the folder to `SafetensorsReader` is what
        // produced Windows ERROR_ACCESS_DENIED ("os error 5").
        InputKind::SingleFile => {
            let file = discover::resolve_single_file(&args.input)
                .map_err(|e| CliError::usage(e.to_string()))?;
            vec![file]
        }
        InputKind::ShardedFolder => {
            let model = discover::discover_shards(&args.input).map_err(|e| e.to_string())?;
            model.shard_paths()
        }
    };
    if shard_paths.is_empty() {
        return Err(CliError::data(format!(
            "no .safetensors shards found in {}",
            args.input.display()
        )));
    }

    let union = discover::resolve_union(&shard_paths).map_err(|e| e.to_string())?;
    let name_to_shard: HashMap<&str, usize> = union
        .name_to_shard
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();

    // --- Write (to a temp path, renamed into place only on success) --------
    //
    // The closure owns every fallible step after the temp file is created, so
    // ANY error (overflow refusal, IO failure, missing tensor) removes the
    // temp file and leaves no output at all. On success the temp file is
    // renamed over the destination, which is atomic within a directory.
    let tmp_path = temp_path_for(&out_path);

    // Progress sink, built exactly as `quantize`/`gguf` build theirs. The
    // total is known UP FRONT from the union header, so the bar is accurate
    // from the first frame — unlike those two, which discover it as they go.
    let mut sink: Box<dyn ProgressSink> = if args.no_progress {
        Box::new(NullSink)
    } else {
        Box::new(BarSink::new(&format!("cast -> {}", target_tag(args.to))))
    };

    let write_result = (|| -> Result<(usize, usize, usize, Vec<DType>), String> {
        let mut writer =
            IncrementalWriter::open_new_with(&tmp_path, HEADER_SLOT_HINT, Some(output_metadata()))
                .map_err(|e| format!("creating {}: {e}", tmp_path.display()))?;

        let mut total = 0usize;
        let mut verbatim = 0usize;
        let mut converted = 0usize;
        // Distinct FLOAT source dtypes actually seen, in first-appearance
        // order. A Vec rather than a BTreeSet because `DType` derives
        // PartialEq but not Ord, and `dtype.rs` is not a file this change may
        // touch. First-appearance order is also the more useful report order.
        let mut src_dtypes: Vec<DType> = Vec::new();

        // One reader per shard, held open. `tensor_bytes` yields BORROWED
        // slices into each mmap, so a tensor's payload is converted straight
        // into the output buffer and never all held in memory at once. Peak RSS
        // is therefore bounded by the largest single tensor (445 MiB for the
        // target model), not by the 5.04 GiB model size.
        let mut readers: Vec<SafetensorsReader> = Vec::with_capacity(shard_paths.len());
        for sp in &shard_paths {
            readers.push(
                SafetensorsReader::open(sp)
                    .map_err(|e| format!("opening {}: {e}", sp.display()))?,
            );
        }

        // Announce the total before the first tensor, so the bar renders as
        // `0/N` rather than an indeterminate bar.
        let expected = union.entries.len();
        sink.update(0, expected);

        for (name, info) in &union.entries {
            let src = name_to_shard
                .get(name.as_str())
                .copied()
                .ok_or_else(|| format!("internal: no shard for tensor {name:?}"))?;
            let reader = &readers[src];

            let payload = reader
                .tensor_bytes(name)
                .map_err(|e| format!("reading {name}: {e}"))?;

            let is_float = matches!(info.dtype, DType::F32 | DType::F16 | DType::Bf16);
            if is_float && !src_dtypes.contains(&info.dtype) {
                src_dtypes.push(info.dtype);
            }

            let (bytes, outcome) = cast_tensor(name, payload, info.dtype, target, &info.shape)
                .map_err(|e| e.to_string())?;

            // Non-float tensors keep their ORIGINAL header spelling (the numpy
            // "U16"-for-bf16 quirk means the source string is not always
            // derivable from the enum). Float tensors are written under the
            // target's canonical spelling.
            let (out_dtype, raw_override) = if is_float {
                (output_dtype_for(info.dtype, target), None)
            } else {
                (info.dtype, Some(info.dtype_raw.as_str()))
            };

            writer
                .add_tensor(name, out_dtype, raw_override, &info.shape, &bytes)
                .map_err(|e| format!("writing {name}: {e}"))?;

            total += 1;
            match outcome {
                CastOutcome::Verbatim => verbatim += 1,
                CastOutcome::Converted => converted += 1,
            }
            // Advance after the tensor is durably written, so the position can
            // never claim progress the file does not have.
            sink.update(total, expected);
        }

        writer
            .finalize()
            .map_err(|e| format!("finalizing {}: {e}", tmp_path.display()))?;
        Ok((total, verbatim, converted, src_dtypes))
    })();

    // Always clear the progress bar before reporting anything — including on
    // the overflow / f64 / IO error paths — so the message is never garbled by
    // a live bar. Mirrors `quantize.rs::run_single`.
    sink.finish();

    let (total, verbatim, converted, src_dtypes) = write_result.map_err(|e| {
        // The destination was never touched — the half-built file is still at
        // the temp path. Remove it so a failure cannot leave something that
        // looks like a valid `.safetensors` next to the user's model.
        // Best-effort: if removal itself fails there is nothing better to do,
        // and the *destination* path remains absent, which is the property
        // that actually matters.
        let _ = std::fs::remove_file(&tmp_path);
        CliError::data(e)
    })?;

    // Publish atomically. Same directory as the destination, so this is a
    // rename and not a cross-device copy.
    if out_path != tmp_path {
        std::fs::rename(&tmp_path, &out_path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_path);
            format!(
                "publishing {} -> {}: {e}",
                tmp_path.display(),
                out_path.display()
            )
        })?;
    }

    // --- Report ------------------------------------------------------------
    // Name the source dtype(s) we actually saw. A bf16 -> f32 run printing
    // "(float -> F32)" tells the user nothing about where the data came from;
    // "BF16 -> F32" does. A model with mixed float sources is reported as
    // such rather than guessed at.
    let src_desc = match src_dtypes.as_slice() {
        [] => "non-float".to_string(),
        [only] => only.to_string(),
        many => {
            let joined: Vec<String> = many.iter().map(|d| d.to_string()).collect();
            format!("mixed({})", joined.join("+"))
        }
    };
    report(total, verbatim, converted, &src_desc, target);
    Ok(ExitCode::SUCCESS)
}

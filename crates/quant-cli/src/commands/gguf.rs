//! `gguf` subcommand — HF safetensors → GGUF conversion (plan Phase 10.2).
//!
//! Exit codes:
//! - `0` — conversion completed (or `--list-methods` printed)
//! - `1` — runtime failure (IO, discovery, encoding error)
//! - `2` — usage error (missing input, unknown method, Dynamic 2.0 method)

use std::path::PathBuf;
use std::process::ExitCode;

use quant_core::discover::classify_input;
use quant_core::gguf_convert::{convert_hf_to_gguf, GgufConvertConfig, GgufError};
use quant_core::gguf_registry;

use crate::args::GgufArgs;
use crate::progress::{BarSink, NullSink, ProgressSink};

/// Print the method table (mirrors the reference `list_line` format:
/// `id           bpw  [DYNAMIC 2.0] description`), plus the Unsloth-plan
/// capability markers: `[IMATRIX]` for methods official Unsloth gates
/// behind `imatrix_file=`, and `(unsupported natively)` for rejected ids.
fn print_methods() {
    println!("GGUF quantization methods (quantui-rs native):");
    println!();
    for e in gguf_registry::METHODS {
        let m = &e.method;
        let badge = if m.dynamic_v2 {
            "[DYNAMIC 2.0] "
        } else if m.requires_imatrix {
            "[IMATRIX] "
        } else {
            ""
        };
        let bpw = match m.approx_bpw {
            Some(b) => format!("{b:>4}bpw "),
            None => "      ".to_string(),
        };
        let supported = if m.dynamic_v2 {
            " (unsupported natively)"
        } else {
            ""
        };
        println!("{:<12} {bpw} {badge}{}{supported}", m.id, m.description);
    }
}

/// Auto-name the output: `<base>-<method>.gguf` next to the input (or in the
/// current directory when the input has no parent).
fn auto_output(input: &std::path::Path, method: &str) -> PathBuf {
    let (_, base) = classify_input(input);
    let base = base.unwrap_or_else(|| "model".to_string());
    let file = format!("{base}-{method}.gguf");
    match input.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(file),
        _ => PathBuf::from(file),
    }
}

pub fn run(args: GgufArgs) -> ExitCode {
    if args.list_methods {
        print_methods();
        return ExitCode::SUCCESS;
    }

    let Some(input) = &args.input else {
        eprintln!("error: <INPUT> is required (a .safetensors file or sharded model folder)");
        eprintln!("hint: use --list-methods to see available GGUF methods");
        return ExitCode::from(2);
    };

    // Validate the method early so a typo fails with exit 2 (usage), not 1.
    // Phase 2.1: every rejection names the specific cause and suggests only
    // methods that can actually run (usable_ids).
    let entry = match gguf_registry::get_method(&args.method) {
        None => {
            eprintln!(
                "error: unknown GGUF method '{}'. Supported: {}",
                args.method,
                gguf_registry::usable_ids().join(", ")
            );
            return ExitCode::from(2);
        }
        Some(e) if e.method.dynamic_v2 => {
            eprintln!(
                "error: '{}' is an Unsloth Dynamic 2.0 per-layer variant and is not supported natively.",
                args.method
            );
            eprintln!("The proprietary per-layer bit-width heuristic is a documented out-of-scope boundary.");
            eprintln!("Supported: {}", gguf_registry::usable_ids().join(", "));
            return ExitCode::from(2);
        }
        Some(e)
            if matches!(
                e.method.support,
                gguf_registry::BackendSupport::NoEncoder(_)
            ) =>
        {
            if let Some(reason) = gguf_registry::rejection_reason(&args.method) {
                eprintln!("error: method '{}' cannot run: {}", args.method, reason);
            }
            eprintln!("Supported: {}", gguf_registry::usable_ids().join(", "));
            return ExitCode::from(2);
        }
        Some(e) => e,
    };

    // Phase 4.2: Unsloth-style imatrix gate (save.py:2162 — every iq* id is
    // in IMATRIX_QUANTS). An iq* method without --imatrix would produce
    // garbage (llama-quantize refuses the same way: "this quantization
    // requires an imatrix!", llama-quant.cpp:1084-1090), so reject it up
    // front with exit 2 naming the method and the flag.
    let imatrix = if entry.method.requires_imatrix {
        let Some(path) = &args.imatrix else {
            eprintln!(
                "error: GGUF method '{}' requires an importance matrix (--imatrix <PATH>).",
                args.method
            );
            eprintln!("Without it the quantized weights would be garbage; llama-quantize refuses the same conversion.");
            eprintln!("hint: pass --imatrix <imatrix_file> (GGUF or legacy binary format).");
            return ExitCode::from(2);
        };
        match load_imatrix(path) {
            Ok(im) => Some(im),
            Err(code) => return code,
        }
    } else if let Some(path) = &args.imatrix {
        // Not required for this method, but the user pointed at a file —
        // load it anyway (llama-quantize accepts --imatrix with any ftype;
        // K-quants simply consume the weights per tensor). A bad file is
        // still a hard error: the user asked for it explicitly.
        match load_imatrix(path) {
            Ok(im) => Some(im),
            Err(code) => return code,
        }
    } else {
        None
    };

    // Phase 6: per-tensor recipe + the two category overrides. Every qtype
    // must be a USABLE method id — validated here (exit 2) so a typo never
    // surfaces mid-conversion.
    let recipe = match &args.tensor_type_file {
        Some(path) => match quant_core::gguf_recipe::TensorRecipe::load(path) {
            Ok(r) => {
                eprintln!(
                    "recipe: {} rule(s){} loaded from {}",
                    r.rules.len(),
                    if r.default.is_some() {
                        " + default"
                    } else {
                        ""
                    },
                    path.display()
                );
                Some(r)
            }
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    for (flag, id) in [
        (
            "--token-embedding-type",
            args.token_embedding_type.as_deref(),
        ),
        ("--output-tensor-type", args.output_tensor_type.as_deref()),
    ] {
        if let Some(id) = id {
            match gguf_registry::get_method(id) {
                Some(e)
                    if !e.method.dynamic_v2
                        && !matches!(
                            e.method.support,
                            gguf_registry::BackendSupport::NoEncoder(_)
                        ) => {}
                _ => {
                    eprintln!(
                        "error: {flag} '{}' is not a usable GGUF method. Supported: {}",
                        id,
                        gguf_registry::usable_ids().join(", ")
                    );
                    return ExitCode::from(2);
                }
            }
        }
    }

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| auto_output(input, &args.method));

    let cfg = GgufConvertConfig {
        method_id: args.method.clone(),
        arch: args.arch.clone(),
        name: args.name.clone(),
        imatrix,
        recipe,
        token_embedding_type: args.token_embedding_type.clone(),
        output_tensor_type: args.output_tensor_type.clone(),
    };

    let mut sink: Box<dyn ProgressSink> = if args.no_progress {
        Box::new(NullSink)
    } else {
        Box::new(BarSink::new(&format!("gguf {}", args.method)))
    };

    let started = std::time::Instant::now();
    let result = {
        let mut cb = |cur: usize, total: usize| sink.update(cur, total);
        convert_hf_to_gguf(input, &output, &cfg, Some(&mut cb))
    };
    sink.finish();

    match result {
        Ok(report) => {
            // Phase 6.3: --emit-recipe dumps the effective assignment as a
            // recipe file. `^name$` anchored patterns reproduce the exact
            // per-tensor map when fed back through --tensor-type-file.
            if let Some(path) = &args.emit_recipe {
                let mut text = format!(
                    "# effective per-tensor recipe for method '{}'\n\
                     # generated by quantui-rs --emit-recipe\n",
                    args.method
                );
                for (name, scheme) in &report.effective_schemes {
                    let id = scheme_id_for_dump(*scheme);
                    text.push_str(&format!("^{name}$={id}\n"));
                }
                match std::fs::write(path, text) {
                    Ok(()) => {
                        println!("recipe: effective assignment written to {}", path.display())
                    }
                    Err(e) => {
                        eprintln!("error: cannot write --emit-recipe {}: {e}", path.display());
                        return ExitCode::from(1);
                    }
                }
            }

            // Phase 2.3: a silent F16 degrade is a bug, not a feature. The
            // per-tensor warnings already went to stderr during conversion;
            // restate the summary here so it is impossible to miss, without
            // failing the run (resume semantics — see plan §3-H).
            if report.fallback_f16 > 0 {
                eprintln!(
                    "warning: {} of {} tensors were NOT quantized with '{}' and were stored as F16: [{}]",
                    report.fallback_f16,
                    report.tensors,
                    args.method,
                    report.fallback_tensors.join(", ")
                );
            }
            // Per-row block-size demotion (port of llama-quantize
            // `tensor_type_fallback`). The per-tensor warnings already went
            // to stderr; restate the summary so a model full of odd-shaped
            // conv kernels is impossible to miss. Distinct wording from the
            // block above — these tensors ARE quantized (or legally F16),
            // they just did not get the requested block size.
            if report.row_fallback > 0 {
                eprintln!(
                    "warning: {} of {} tensors had a row width (ne[0]) incompatible with \
                     the requested block size and were demoted: [{}]",
                    report.row_fallback,
                    report.tensors,
                    report.row_fallback_tensors.join(", ")
                );
            }
            println!(
                "wrote {} ({} tensors: {} quantized, {} kept F32, {} F16-fallback; arch {}; {:.1} MB in {:.2}s)",
                report.output.display(),
                report.tensors,
                report.quantized,
                report.kept_f32,
                report.fallback_f16,
                report.arch,
                report.output_bytes as f64 / (1024.0 * 1024.0),
                started.elapsed().as_secs_f64()
            );
            record_recent(&args, &report.output, started.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            // Method/input classification errors are usage errors (exit 2);
            // everything else is a runtime failure (exit 1).
            let code = match &e {
                GgufError::UnknownMethod(..)
                | GgufError::DynamicMethod(_)
                | GgufError::BadInput(_)
                | GgufError::NoEncoder(..) => 2,
                _ => 1,
            };
            eprintln!("error: {e}");
            ExitCode::from(code)
        }
    }
}

/// Map an effective scheme back to a canonical usable method id for the
/// `--emit-recipe` dump (first usable method in registry order whose
/// default equals the scheme — deterministic).
fn scheme_id_for_dump(scheme: quant_core::gguf_registry::GgufScheme) -> String {
    for e in gguf_registry::METHODS {
        if !e.method.dynamic_v2
            && !matches!(
                e.method.support,
                gguf_registry::BackendSupport::NoEncoder(_)
            )
            && e.policy.default == scheme
        {
            return e.method.id.to_string();
        }
    }
    format!("{scheme:?}")
}

/// Load an imatrix file, or fail with exit 2 and the loader's specific
/// cause. Phase 4.2 contract: a bad --imatrix is a usage error (the user
/// handed us an unusable file), never a mid-conversion runtime crash.
fn load_imatrix(path: &std::path::Path) -> Result<quant_core::imatrix::Imatrix, ExitCode> {
    match quant_core::imatrix::Imatrix::load(path) {
        Ok(im) => {
            eprintln!(
                "imatrix: loaded {} importance matrix entries from {} ({} chunks, dataset{})",
                im.len(),
                path.display(),
                im.chunk_count,
                im.datasets
                    .first()
                    .map(|d| format!(" '{d}'"))
                    .unwrap_or_default()
            );
            Ok(im)
        }
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("hint: the file must be a GGUF imatrix or a legacy binary imatrix.");
            Err(ExitCode::from(2))
        }
    }
}

/// Best-effort recents persistence (same store as `quantize`). Never fails.
fn record_recent(args: &GgufArgs, output: &std::path::Path, duration: std::time::Duration) {
    use crate::profiles::{add_recent, load_store, save_store, RunRecord};

    let record = RunRecord {
        ts: super::quantize::now_iso8601(),
        family: "gguf".into(),
        method: args.method.clone(),
        output: output.to_string_lossy().into_owned(),
        status: "success".into(),
        exit_code: 0,
        duration_s: duration.as_secs_f64(),
    };
    let mut store = load_store(None);
    add_recent(&mut store, record);
    let _ = save_store(&store, None);
}

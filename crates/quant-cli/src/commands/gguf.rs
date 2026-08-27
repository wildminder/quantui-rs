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
/// `id           bpw  [DYNAMIC 2.0] description`).
fn print_methods() {
    println!("GGUF quantization methods (quantui-rs native):");
    println!();
    for e in gguf_registry::METHODS {
        let m = &e.method;
        let badge = if m.dynamic_v2 { "[DYNAMIC 2.0] " } else { "" };
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
    match gguf_registry::get_method(&args.method) {
        None => {
            eprintln!(
                "error: unknown GGUF method '{}'. Supported: {}",
                args.method,
                gguf_registry::supported_ids().join(", ")
            );
            return ExitCode::from(2);
        }
        Some(e) if e.method.dynamic_v2 => {
            eprintln!(
                "error: '{}' is an Unsloth Dynamic 2.0 per-layer variant and is not supported natively.",
                args.method
            );
            eprintln!("The proprietary per-layer bit-width heuristic is a documented out-of-scope boundary.");
            eprintln!("Supported: {}", gguf_registry::supported_ids().join(", "));
            return ExitCode::from(2);
        }
        Some(_) => {}
    }

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| auto_output(input, &args.method));

    let cfg = GgufConvertConfig {
        method_id: args.method.clone(),
        arch: args.arch.clone(),
        name: args.name.clone(),
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
                | GgufError::BadInput(_) => 2,
                _ => 1,
            };
            eprintln!("error: {e}");
            ExitCode::from(code)
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

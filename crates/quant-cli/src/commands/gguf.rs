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

use crate::args::{GgufArgs, GgufConversionArgs, GgufVerificationArgs};
use crate::progress::{BarSink, NullSink, ProgressSink};

/// Flat view of the destructured [GgufArgs] groups: one field per flag with
/// the original names, so the body of [run] keeps its `args.field` shape
/// unchanged after the IMP-002 regrouping.
struct FlatGgufArgs {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    method: String,
    imatrix: Option<PathBuf>,
    tensor_type_file: Option<PathBuf>,
    token_embedding_type: Option<String>,
    output_tensor_type: Option<String>,
    emit_recipe: Option<PathBuf>,
    verify_against: Option<PathBuf>,
    recipe_from: Option<PathBuf>,
    arch: Option<String>,
    name: Option<String>,
    audit: Option<PathBuf>,
    list_methods: bool,
    no_progress: bool,
}

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
    // IMP-002: destructure the two arg groups once, then rebuild a flat
    // view so every existing `args.field` path below stays unchanged.
    // Note on `--method` + `--audit`: `--method` carries a clap
    // default_value, so it cannot conflict-declare against `--audit`
    // (defaults don't conflict). A user passing `--audit x.gguf --method
    // q8_0` is harmless — the method is simply ignored — so only
    // INPUT/OUTPUT are hard conflicts (checked in the audit branch).
    let GgufArgs {
        conversion:
            GgufConversionArgs {
                input,
                output,
                method,
                imatrix,
                tensor_type_file,
                token_embedding_type,
                output_tensor_type,
            },
        verification:
            GgufVerificationArgs {
                emit_recipe,
                verify_against,
                recipe_from,
                arch,
                name,
                audit,
            },
        list_methods,
        no_progress,
    } = args;
    // Flat view over the destructured groups — the rest of this function
    // keeps using `args.method`-style paths (now local shadow bindings in
    // an `args`-shaped struct).
    let args = FlatGgufArgs {
        input,
        output,
        method,
        imatrix,
        tensor_type_file,
        token_embedding_type,
        output_tensor_type,
        emit_recipe,
        verify_against,
        recipe_from,
        arch,
        name,
        audit,
        list_methods,
        no_progress,
    };

    if args.list_methods {
        print_methods();
        return ExitCode::SUCCESS;
    }

    // [IMP-003] --audit: standalone spec-conformance + dtype census of any
    // GGUF file. Skips conversion entirely — INPUT/OUTPUT are meaningless
    // here and rejected (exit 2); `--method` is simply ignored (it has a
    // clap default, so it cannot be a hard conflict). Exit 0 clean /
    // 3 violations / 1 unparseable.
    if let Some(audit_path) = &args.audit {
        if args.input.is_some() || args.output.is_some() {
            eprintln!("error: --audit audits an existing GGUF file and takes no INPUT or OUTPUT");
            eprintln!("usage: quantui-rs gguf --audit <file.gguf>");
            return ExitCode::from(2);
        }
        return match quant_core::gguf_verify::audit_gguf(audit_path) {
            Ok(a) => {
                println!(
                    "audit: {} ({} tensors)",
                    audit_path.display(),
                    a.tensor_count
                );
                let hist: Vec<String> = a
                    .dtype_histogram
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect();
                println!("  dtype histogram: {}", hist.join(", "));
                if a.violations.is_empty() {
                    println!("  spec-conformance: OK (0 violations)");
                    ExitCode::SUCCESS
                } else {
                    println!(
                        "  spec-conformance: {} VIOLATIONS (quantized tensor with ne[0] % blck_size != 0, gguf.cpp:724):",
                        a.violations.len()
                    );
                    for v in a.violations.iter().take(25) {
                        println!(
                            "      {}  {}  ne0={}  blck={}",
                            v.name, v.dtype, v.ne0, v.blck
                        );
                    }
                    if a.violations.len() > 25 {
                        println!("      ... and {} more", a.violations.len() - 25);
                    }
                    ExitCode::from(3)
                }
            }
            Err(e) => {
                eprintln!("error: audit failed: {e}");
                ExitCode::from(1)
            }
        };
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
    //
    // Task #8: --recipe-from extracts the recipe from a reference GGUF
    // instead of reading a file; it feeds the SAME TensorRecipe machinery
    // (row-width demotion still applies on top of the extracted rules).
    let recipe = if let Some(ref_path) = &args.recipe_from {
        match quant_core::gguf_recipe::recipe_from_gguf(ref_path) {
            Ok(r) => {
                eprintln!(
                    "recipe-from: {} rule(s) extracted from {} (F32 skipped; exact-name rules)",
                    r.rules.len(),
                    ref_path.display()
                );
                Some(r)
            }
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        match &args.tensor_type_file {
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
        }
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
        // None → QUANTUI_RS_GGUF_JOBS / available parallelism (IMP-004).
        jobs: None,
    };

    let mut sink: Box<dyn ProgressSink> = if args.no_progress {
        Box::new(NullSink)
    } else {
        Box::new(BarSink::new(&format!("gguf {}", args.method)))
    };

    let started = std::time::Instant::now();
    // Warning-spam fix: both callbacks share ONE sink — progress keeps the
    // bar moving, warnings render ABOVE the bar (via the same indicatif
    // instance) instead of re-breaking it. RefCell split-borrow: core
    // invokes the callbacks strictly sequentially (never reentrant), so
    // the runtime borrow can never conflict.
    let sink_cell = std::cell::RefCell::new(&mut sink);
    let result = {
        let mut cb = |cur: usize, total: usize| {
            sink_cell.borrow_mut().update(cur, total);
        };
        let mut wb = |line: &str| {
            sink_cell.borrow_mut().warn(line);
        };
        convert_hf_to_gguf(input, &output, &cfg, Some(&mut cb), Some(&mut wb))
    };
    sink.finish();

    match result {
        Ok(report) => {
            // Phase 6.3: --emit-recipe dumps the effective assignment as a
            // recipe file. `^name$` anchored patterns reproduce the exact
            // per-tensor map when fed back through --tensor-type-file.
            if let Some(path) = &args.emit_recipe {
                let mut text = format!(
                    "# quantui-rs recipe format {}\n\
                     # effective per-tensor recipe for method '{}'\n\
                     # generated by quantui-rs --emit-recipe\n",
                    quant_core::gguf_recipe::RECIPE_FORMAT_VERSION,
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
            // per-tensor warnings were rendered above the progress bar
            // during conversion (first-of-kind on a live bar, all of them
            // when piped); this summary restates the full list so it is
            // impossible to miss, without failing the run (resume
            // semantics — see plan §3-H).
            // Task #7: --verify-against — oracle equivalence report vs a
            // reference GGUF. Report tool: exit 0 regardless of diffs,
            // UNLESS our own file has spec violations (then exit 3).
            if let Some(ref_path) = &args.verify_against {
                match quant_core::gguf_verify::verify_against(&report.output, ref_path) {
                    Ok(vr) => {
                        let (exact, equiv, divergent) = vr.summary();
                        println!();
                        println!(
                            "verify-against: {} vs {}",
                            report.output.display(),
                            ref_path.display()
                        );
                        println!(
                            "  byte-exact: {exact} | numerically-equivalent: {equiv} | divergent: {divergent}"
                        );
                        for d in &vr.diffs {
                            println!(
                                "  ! {:<18} blocks {}/{} differ, max|Δ|={:.3e} [{}]",
                                format!("{} ({})", d.name, kind_label(d.kind)),
                                d.diff_blocks,
                                d.total_blocks,
                                d.max_abs_err,
                                kind_label(d.kind)
                            );
                        }
                        for m in &vr.dtype_mismatch {
                            println!("  ~ dtype mismatch: {m}");
                        }
                        for m in &vr.dims_mismatch {
                            println!("  ~ dims mismatch: {m}");
                        }
                        if !vr.ours_only.is_empty() {
                            println!(
                                "  ? ours-only (unmatched after name mapping): {}",
                                vr.ours_only.len()
                            );
                            for u in vr.ours_only.iter().take(10) {
                                println!("      {u}");
                            }
                        }
                        if !vr.ref_only.is_empty() {
                            println!("  ? reference-only: {}", vr.ref_only.len());
                            for u in vr.ref_only.iter().take(10) {
                                println!("      {u}");
                            }
                        }
                        if vr.skipped_f32 > 0 {
                            println!("  (skipped {} F32 tensors)", vr.skipped_f32);
                        }
                        if vr.spec_violations.is_empty() {
                            println!("  spec-conformance: OK (0 violations)");
                        } else {
                            println!(
                                "  spec-conformance: {} VIOLATIONS (quantized tensor with ne[0] % blck_size != 0):",
                                vr.spec_violations.len()
                            );
                            for v in vr.spec_violations.iter().take(25) {
                                println!("      {v}");
                            }
                        }
                        if !vr.spec_violations.is_empty() {
                            return ExitCode::from(3);
                        }
                    }
                    Err(e) => {
                        eprintln!("error: verify-against failed: {e}");
                        return ExitCode::from(1);
                    }
                }
            }

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
            // `tensor_type_fallback`). The per-tensor warnings were
            // rendered above the progress bar during conversion; this
            // summary restates the full list so a model full of odd-shaped
            // conv kernels is impossible to miss. Distinct wording from
            // the block above — these tensors ARE quantized (or legally
            // F16), they just did not get the requested block size.
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
            record_recent(&args.method, &report.output, started.elapsed());
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

/// Short human label for a verify-against diff classification.
fn kind_label(k: quant_core::gguf_verify::DiffKind) -> &'static str {
    match k {
        quant_core::gguf_verify::DiffKind::DeadBlockCosmetic => "dead-block cosmetic",
        quant_core::gguf_verify::DiffKind::ScaleRuleDiff => "scale-rule diff",
        quant_core::gguf_verify::DiffKind::GenuineDivergence => "GENUINE DIVERGENCE",
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
fn record_recent(method: &str, output: &std::path::Path, duration: std::time::Duration) {
    use crate::profiles::{add_recent, load_store, save_store, RunRecord};

    let record = RunRecord {
        ts: super::quantize::now_iso8601(),
        family: "gguf".into(),
        method: method.to_string(),
        output: output.to_string_lossy().into_owned(),
        status: "success".into(),
        exit_code: 0,
        duration_s: duration.as_secs_f64(),
    };
    let mut store = load_store(None);
    add_recent(&mut store, record);
    let _ = save_store(&store, None);
}

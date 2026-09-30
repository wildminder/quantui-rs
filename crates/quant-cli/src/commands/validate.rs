//! `validate` subcommand — structural (+ optional numeric) validation of
//! quantized safetensors outputs.
//!
//! Exit codes (Phase 8.3 contract):
//! - `0` — every validated file passed (`report.ok == true`; warnings allowed)
//! - `1` — at least one file failed validation
//! - `2` — usage error (bad arguments, unreadable path) — clap also uses 2

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use quant_core::comfy_loader_contract::{check_comfy_loadable, is_comfy_loadable, ContractIssue};
use quant_core::validator::{format_report, validate_comfy_quant};

use crate::args::ValidateArgs;

/// Collect the `.safetensors` files to validate for one CLI path argument.
fn expand_path(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| format!("cannot read directory {}: {e}", path.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!(
                "no .safetensors files found in directory {}",
                path.display()
            ));
        }
        return Ok(files);
    }
    Err(format!("path does not exist: {}", path.display()))
}

pub fn run(args: ValidateArgs) -> ExitCode {
    let mut targets: Vec<PathBuf> = Vec::new();
    for p in &args.paths {
        match expand_path(p) {
            Ok(files) => targets.extend(files),
            Err(msg) => {
                eprintln!("error: {msg}");
                return ExitCode::from(2);
            }
        }
    }

    let mut any_failed = false;
    for (i, target) in targets.iter().enumerate() {
        if i > 0 {
            println!();
        }
        if args.comfy {
            // The CONSUMER's contract, not the reference encoder's. Kept as a
            // separate branch rather than folded into `validate_comfy_quant`
            // because being stricter than ComfyUI rejects files it loads fine,
            // and being looser passes files it cannot load at all.
            let issues = check_comfy_loadable(target);
            println!("{}", format_comfy_report(target, &issues));
            if !is_comfy_loadable(&issues) {
                any_failed = true;
            }
            continue;
        }
        let report = validate_comfy_quant(target, args.numeric);
        println!("{}", format_report(&report));
        if !report.ok {
            any_failed = true;
        }
    }

    if any_failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// Render a ComfyUI-contract report.
///
/// Kept byte-stable (no timings, no paths beyond the target) so it is diffable
/// across runs, matching the existing report's discipline.
fn format_comfy_report(path: &Path, issues: &[ContractIssue]) -> String {
    use quant_core::comfy_loader_contract::ContractIssue::*;

    let mut out = format!("ComfyUI loader contract: {}\n", path.display());
    for issue in issues {
        match issue {
            Fatal(m) => out.push_str(&format!("  ERROR   {m}\n")),
            Advisory(m) => out.push_str(&format!("  note    {m}\n")),
        }
    }
    if issues.is_empty() {
        out.push_str("  OK      no quantized layers found; nothing for ComfyUI to resolve\n");
    } else if is_comfy_loadable(issues) {
        out.push_str("  OK      contract-conformant (NOT proof of load — only a real ComfyUI load settles that)\n");
    } else {
        out.push_str("  FAIL    ComfyUI would raise on this file\n");
    }
    out
}

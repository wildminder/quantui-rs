//! `validate` subcommand — structural (+ optional numeric) validation of
//! quantized safetensors outputs.
//!
//! Exit codes (Phase 8.3 contract):
//! - `0` — every validated file passed (`report.ok == true`; warnings allowed)
//! - `1` — at least one file failed validation
//! - `2` — usage error (bad arguments, unreadable path) — clap also uses 2

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Args;
use quant_core::validator::{format_report, validate_comfy_quant};

#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// One or more `.safetensors` files or directories containing them
    /// (sharded output folders are expanded to their shard files).
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    /// Also run the numeric pass (weight bounds, scale finiteness/positivity,
    /// input_scale == 1.0). Reads tensor payloads, not just headers.
    #[arg(long)]
    numeric: bool,
}

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

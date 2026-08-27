//! `info` subcommand — safetensors header inspection without loading tensor
//! payloads (plan Phase 9.3).
//!
//! Prints a per-tensor table (name, dtype, shape, size) plus comfy_quant
//! format detection per quantized layer. `--raw` dumps the JSON header.
//!
//! Exit codes: 0 ok, 1 unreadable/invalid file, 2 usage.

use std::process::ExitCode;

use quant_core::comfy_schema::{layer_prefix, parse_blob};
use quant_core::st_io::reader::SafetensorsReader;

use crate::args::InfoArgs;

/// Human-friendly byte size.
fn fmt_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[i])
    }
}

fn fmt_shape(shape: &[u64]) -> String {
    if shape.is_empty() {
        return "[]".to_string();
    }
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", dims.join(", "))
}

pub fn run(args: InfoArgs) -> ExitCode {
    let reader = match SafetensorsReader::open(&args.input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}: {e}", args.input.display());
            return ExitCode::from(1);
        }
    };
    let header = reader.header();

    if args.raw {
        // Raw JSON header body (as stored, minus the 8-byte length prefix).
        println!("{}", String::from_utf8_lossy(&header.serialize_json()));
        return ExitCode::SUCCESS;
    }

    // ---- summary line ------------------------------------------------------ //
    let total_bytes: u64 = header
        .iter()
        .map(|(_, info)| info.data_offsets.1 - info.data_offsets.0)
        .sum();
    println!("file    : {}", args.input.display());
    println!(
        "tensors : {} ({} of payload)",
        header.len(),
        fmt_bytes(total_bytes)
    );
    if let Some(meta) = header.metadata() {
        println!(
            "metadata: {}",
            serde_json::to_string(meta).unwrap_or_default()
        );
    }

    // ---- comfy_quant format detection ------------------------------------- //
    // Collect layer prefix → format string from the `.comfy_quant` markers.
    let mut formats: Vec<(String, String)> = Vec::new();
    for (name, info) in header.iter() {
        if !name.ends_with(".comfy_quant") {
            continue;
        }
        let Some(prefix) = layer_prefix(name) else {
            continue;
        };
        let blob = match reader.tensor_bytes(name) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let fmt = match parse_blob(blob) {
            Ok(cfg) => cfg.format.as_str().to_string(),
            Err(e) => format!("<unparseable: {e}>"),
        };
        let _ = info;
        formats.push((prefix.to_string(), fmt));
    }

    // ---- per-tensor table -------------------------------------------------- //
    println!();
    println!(
        "{:<60} {:<10} {:<20} {:>10}",
        "name", "dtype", "shape", "size"
    );
    for (name, info) in header.iter() {
        let size = info.data_offsets.1 - info.data_offsets.0;
        println!(
            "{:<60} {:<10} {:<20} {:>10}",
            truncate(name, 60),
            info.dtype_raw,
            fmt_shape(&info.shape),
            fmt_bytes(size)
        );
    }

    // ---- quantized layers -------------------------------------------------- //
    if !formats.is_empty() {
        println!();
        println!("quantized layers ({}):", formats.len());
        for (prefix, fmt) in &formats {
            println!("  {prefix}: {fmt}");
        }
    } else {
        println!();
        println!("quantized layers: none (no .comfy_quant markers)");
    }

    ExitCode::SUCCESS
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("…{}", &s[s.len() - (max - 1)..])
    }
}

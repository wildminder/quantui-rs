//! quantui-rs: standalone CLI for quantizing safetensors models.
//!
//! Subcommands: `quantize` (Phase 9.2), `validate` (Phase 8.3), `info`
//! (Phase 9.3). Arg definitions live in `args.rs` (Phase 9.1).

mod args;
mod commands;
mod profiles;
mod progress;

use clap::{Parser, Subcommand};

use args::{InfoArgs, QuantizeArgs, ValidateArgs};

/// Standalone, dependency-free-at-runtime CLI for ComfyUI/GGUF model quantization.
#[derive(Parser, Debug)]
#[command(
    name = "quantui-rs",
    version = env!("CARGO_PKG_VERSION"),
    about = "Quantize safetensors models (INT8/FP8/MXFP8/NVFP4) and convert HF -> GGUF"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Quantize a safetensors model (single file or sharded folder).
    Quantize(QuantizeArgs),
    /// Structural + numeric validation of a quantized output.
    Validate(ValidateArgs),
    /// Inspect a safetensors header without loading tensors.
    Info(InfoArgs),
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Commands::Quantize(args) => commands::quantize::run(args),
        Commands::Validate(args) => commands::validate::run(args),
        Commands::Info(args) => commands::info::run(args),
    }
}

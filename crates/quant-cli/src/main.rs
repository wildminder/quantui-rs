//! quantui-rs: standalone CLI for quantizing safetensors models.
//!
//! Phase 0 stub: subcommand surface exists, implementations land in later phases.

use clap::{Parser, Subcommand};

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
    Quantize,
    /// Structural + numeric validation of a quantized output.
    Validate,
    /// Inspect a safetensors header without loading tensors.
    Info,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Quantize => {
            println!("quantize: not implemented");
        }
        Commands::Validate => {
            println!("validate: not implemented");
        }
        Commands::Info => {
            println!("info: not implemented");
        }
    }
}

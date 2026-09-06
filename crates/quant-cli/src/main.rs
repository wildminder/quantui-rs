//! quantui-rs: standalone CLI for quantizing safetensors models.
//!
//! Subcommands: `quantize` (Phase 9.2), `validate` (Phase 8.3), `info`
//! (Phase 9.3). Arg definitions live in `args.rs` (Phase 9.1).

mod args;
mod commands;
mod profiles;
mod progress;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, shells::Shell as CompletableShell};

use args::{GgufArgs, InfoArgs, QuantizeArgs, ValidateArgs};

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
    /// Convert a HF safetensors model to GGUF (single file or sharded folder).
    // Boxed: GgufArgs has grown (Phase 6 recipes + verify-against) and a
    // bare variant dominated the enum (clippy::large_enum_variant).
    Gguf(Box<GgufArgs>),
    /// Structural + numeric validation of a quantized output.
    Validate(ValidateArgs),
    /// Inspect a safetensors header without loading tensors.
    Info(InfoArgs),
    /// Emit a shell completion script (hidden from --help; documented in
    /// the README's "Shell completions" section). WP8 / NTH-005.
    #[command(hide = true)]
    Completions {
        /// Target shell.
        #[arg(value_enum)]
        shell: CompletableShell,
    },
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Commands::Quantize(args) => commands::quantize::run(args),
        Commands::Gguf(args) => commands::gguf::run(*args),
        Commands::Validate(args) => commands::validate::run(args),
        Commands::Info(args) => commands::info::run(args),
        Commands::Completions { shell } => {
            // `generate` writes the script to stdout; a broken pipe
            // (quantui-rs completions bash | head) must not panic.
            let mut cmd = Cli::command();
            let mut out = std::io::stdout();
            generate(shell, &mut cmd, "quantui-rs", &mut out);
            std::process::ExitCode::SUCCESS
        }
    }
}

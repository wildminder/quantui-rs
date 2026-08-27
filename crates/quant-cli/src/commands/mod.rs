//! CLI subcommand implementations.
//!
//! Each subcommand lives in its own module and exposes a `run` function that
//! returns a [`std::process::ExitCode`]. `main.rs` only parses args and dispatches.

pub mod validate;

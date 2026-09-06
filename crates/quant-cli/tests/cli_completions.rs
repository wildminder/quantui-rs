//! WP8 / NTH-005 — `completions <SHELL>` subcommand.
//!
//! One parametrized pass over the five supported shells: exit 0, output
//! non-empty and naming the binary; plus the two shell-specific anchors
//! that prove the right generator ran (bash's function prefix,
//! powershell's Register-ArgumentCompleter), and the exit-2 contract for
//! an unknown shell.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_quantui-rs"))
}

/// (shell, marker the generated script must contain).
const CASES: &[(&str, &str)] = &[
    ("bash", "_quantui__rs()"),
    ("zsh", "#compdef quantui-rs"),
    ("fish", "__fish_quantui_rs"),
    ("powershell", "Register-ArgumentCompleter"),
    ("elvish", "quantui-rs"),
];

#[test]
fn completions_each_shell_exits_0_with_script() {
    for (shell, marker) in CASES {
        let o = bin().args(["completions", shell]).output().unwrap();
        assert_eq!(
            o.status.code(),
            Some(0),
            "{shell}: completion generation failed: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        let stdout = String::from_utf8_lossy(&o.stdout);
        assert!(
            !stdout.trim().is_empty(),
            "{shell}: completion script is empty"
        );
        assert!(
            stdout.contains("quantui-rs"),
            "{shell}: script must name the binary"
        );
        assert!(
            stdout.contains(marker),
            "{shell}: expected '{marker}' in the generated script (got {} bytes)",
            stdout.len()
        );
    }
}

#[test]
fn completions_hidden_from_help() {
    // The subcommand is intentionally undocumented in --help (plan 8.2):
    // keep the top-level surface clean.
    let o = bin().arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(
        !stdout.contains("completions"),
        "completions must stay hidden from --help: {stdout}"
    );
}

#[test]
fn completions_unknown_shell_exit_2() {
    let o = bin().args(["completions", "tcsh"]).output().unwrap();
    assert_eq!(
        o.status.code(),
        Some(2),
        "invalid shell is a usage error: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        stderr.contains("invalid value") || stderr.contains("possible values"),
        "error must name the bad value and the valid ones: {stderr}"
    );
}

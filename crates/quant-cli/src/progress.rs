//! Progress reporting for `quantize` runs (plan Phase 9.4).
//!
//! Wraps `indicatif` behind a tiny trait so the orchestrator's
//! `(cur, total)` callback stays decoupled from rendering, and so tests can
//! drive progress-state transitions without a terminal.
//!
//! `warn()` routes user-facing warning lines through the SAME sink as the
//! bar (warning-spam fix): a live bar renders the line ABOVE itself via
//! indicatif's `println`, so the bar stays the last line instead of being
//! re-broken by every per-tensor warning. `NullSink` (and the hidden-bar
//! fallback) print plain `eprintln!` — byte-identical to the pre-fix
//! direct-print behaviour that piped/CI runs and the pinned stderr tests
//! rely on.

use std::collections::HashMap;

use indicatif::{ProgressBar, ProgressStyle};

/// Anything that can consume `(cur, total)` progress updates and warning
/// lines.
pub trait ProgressSink {
    fn update(&mut self, cur: usize, total: usize);
    fn warn(&mut self, line: &str);
    fn finish(&mut self);
}

/// Collapse "kind" for a warning line: the recurring per-tensor warnings
/// all repeat the same reason with only the tensor name varying, so on a
/// live bar the first line per kind is shown and the rest are counted and
/// summarized in ONE line at `finish()` — the affected tensors are already
/// listed in the run summary the CLI prints anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum WarningKind {
    /// "warning: <tensor> - ncols N not divisible by M ... falling back"
    RowFallback,
    /// "warning: tensor '<name>' fell back to F16: ..."
    FellBackF16,
    /// Anything else (imatrix notes, future kinds) — never collapsed.
    Other,
}

impl WarningKind {
    fn classify(line: &str) -> Self {
        if line.contains("not divisible by") {
            WarningKind::RowFallback
        } else if line.contains("fell back to F16") {
            WarningKind::FellBackF16
        } else {
            WarningKind::Other
        }
    }
}

/// Terminal progress bar (indicatif). Renders to stderr so stdout stays clean
/// for machine-readable output. indicatif auto-suppresses drawing on non-TTY
/// targets, so this is safe in CI/piped contexts too.
pub struct BarSink {
    bar: ProgressBar,
    last_total: usize,
    /// Repeat-collapser state (warning-spam fix): first line per kind is
    /// printed above the bar; duplicates only bump a counter, summarized
    /// by ONE `note:` line at `finish()`.
    suppressed: HashMap<WarningKind, u32>,
}

impl BarSink {
    pub fn new(label: &str) -> Self {
        let bar = ProgressBar::new(0);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        bar.set_message(label.to_string());
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        Self {
            bar,
            last_total: 0,
            suppressed: HashMap::new(),
        }
    }
}

impl ProgressSink for BarSink {
    fn update(&mut self, cur: usize, total: usize) {
        if total != self.last_total {
            self.bar.set_length(total as u64);
            self.last_total = total;
        }
        self.bar.set_position(cur as u64);
    }

    fn warn(&mut self, line: &str) {
        // Hidden bar (piped stderr / CI): indicatif's println would DROP
        // the line entirely, so fall back to a plain eprintln — output
        // stays byte-identical to the pre-fix behaviour, no collapse.
        if self.bar.is_hidden() {
            eprintln!("{line}");
            return;
        }
        let kind = WarningKind::classify(line);
        if kind == WarningKind::Other {
            self.bar.println(line);
            return;
        }
        let seen = self.suppressed.entry(kind).or_insert(0);
        if *seen == 0 {
            self.bar.println(line);
        }
        *seen += 1;
    }

    fn finish(&mut self) {
        // Repeat summary FIRST, while the bar can still render above-bar
        // lines: one note per kind that had duplicates, so the user knows
        // warnings were folded rather than lost. Then clear the bar.
        let kinds = [
            (WarningKind::RowFallback, "row-fallback"),
            (WarningKind::FellBackF16, "F16-fallback"),
        ];
        for (kind, label) in kinds {
            if let Some(&n) = self.suppressed.get(&kind) {
                if n > 1 {
                    self.bar.println(format!(
                        "note: {n} tensors emitted the {label} warning; \
                         the full tensor list is in the run summary"
                    ));
                }
            }
        }
        self.bar.finish_and_clear();
    }
}

/// No-op sink used for `--no-progress` and non-TTY runs. `warn` still
/// prints: warnings are part of the output contract, only the bar is
/// suppressed — byte-identical to the pre-fix direct `eprintln!`.
pub struct NullSink;

impl ProgressSink for NullSink {
    fn update(&mut self, _cur: usize, _total: usize) {}
    fn warn(&mut self, line: &str) {
        eprintln!("{line}");
    }
    fn finish(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collects events so tests can assert progress-state transitions.
    struct RecordingSink {
        events: Vec<(usize, usize)>,
        warnings: Vec<String>,
        finished: bool,
    }

    impl ProgressSink for RecordingSink {
        fn update(&mut self, cur: usize, total: usize) {
            self.events.push((cur, total));
        }
        fn warn(&mut self, line: &str) {
            self.warnings.push(line.to_string());
        }
        fn finish(&mut self) {
            self.finished = true;
        }
    }

    #[test]
    fn sink_receives_monotonic_updates() {
        let mut sink = RecordingSink {
            events: Vec::new(),
            warnings: Vec::new(),
            finished: false,
        };
        // Simulate the orchestrator's (cur, total) stream.
        for cur in 1..=5 {
            sink.update(cur, 5);
        }
        sink.finish();

        assert_eq!(sink.events.len(), 5);
        assert!(sink.events.windows(2).all(|w| w[1].0 == w[0].0 + 1));
        assert_eq!(sink.events.last(), Some(&(5, 5)));
        assert!(sink.finished);
    }

    #[test]
    fn recording_sink_collects_warnings() {
        let mut sink = RecordingSink {
            events: Vec::new(),
            warnings: Vec::new(),
            finished: false,
        };
        sink.update(1, 4);
        sink.warn("warning: first");
        sink.update(2, 4);
        sink.warn("warning: second");
        sink.finish();

        assert_eq!(sink.warnings, vec!["warning: first", "warning: second"]);
        assert!(sink.finished);
    }

    #[test]
    fn null_sink_is_noop() {
        let mut sink = NullSink;
        sink.update(1, 10);
        sink.finish();
        // No panic, no state — the contract is simply "does nothing"
        // for progress. `warn` deliberately still prints (asserted by
        // the CLI-level stderr tests, which run piped = NullSink path).
    }

    // ── warning-kind classification (warning-spam fix) ─────────────

    #[test]
    fn warning_kind_classification() {
        assert_eq!(
            WarningKind::classify(
                "warning: model.layers.0.x.weight - ncols      7 not divisible by  32 (required for type    Q8_0) -> falling back to    F16"
            ),
            WarningKind::RowFallback
        );
        assert_eq!(
            WarningKind::classify(
                "warning: tensor 'blk.0.attn_q.weight' fell back to F16: method 'q8_0' scheme Q8_0 cannot describe a row of 7 elements; output is valid GGUF but this tensor is NOT q8_0-quantized"
            ),
            WarningKind::FellBackF16
        );
        // Notes and unknown future lines are never collapsed.
        assert_eq!(
            WarningKind::classify("note: did not find weights for 'x'"),
            WarningKind::Other
        );
        assert_eq!(
            WarningKind::classify("warning: something entirely new"),
            WarningKind::Other
        );
    }

    /// The collapse DECISION table, independent of a live terminal:
    /// first-of-kind shown, duplicates suppressed-and-counted, unknown
    /// kinds always shown. This pins the logic `BarSink::warn` applies
    /// on a visible bar (the piped fallback prints everything, pinned
    /// separately by the CLI stderr tests).
    #[test]
    fn bar_sink_collapse_decision_table() {
        // (line, kind) pairs a visible BarSink would see from a
        // VibeVoice-style run: resolve-warn(i), encode-warn(i), plus a
        // one-off imatrix note.
        let rows: Vec<(&str, WarningKind)> = vec![
            ("row-fallback warn tensor A", WarningKind::RowFallback),
            ("f16-fallback warn tensor A", WarningKind::FellBackF16),
            ("row-fallback warn tensor B", WarningKind::RowFallback),
            ("note: imatrix something", WarningKind::Other),
            ("row-fallback warn tensor C", WarningKind::RowFallback),
            ("f16-fallback warn tensor B", WarningKind::FellBackF16),
        ];

        // Drive the same decision BarSink::warn makes per line.
        let mut suppressed: HashMap<WarningKind, u32> = HashMap::new();
        let mut shown: Vec<&str> = Vec::new();
        for (line, kind) in rows {
            if kind == WarningKind::Other {
                shown.push(line);
                continue;
            }
            let seen = suppressed.entry(kind).or_insert(0);
            if *seen == 0 {
                shown.push(line);
            }
            *seen += 1;
        }

        // First of each collapsible kind + every Other line; counts fold
        // the rest.
        assert_eq!(
            shown,
            vec![
                "row-fallback warn tensor A",
                "f16-fallback warn tensor A",
                "note: imatrix something",
            ]
        );
        assert_eq!(suppressed.get(&WarningKind::RowFallback), Some(&3));
        assert_eq!(suppressed.get(&WarningKind::FellBackF16), Some(&2));
        assert!(!suppressed.contains_key(&WarningKind::Other));
    }
}

//! Progress reporting for `quantize` runs (plan Phase 9.4).
//!
//! Wraps `indicatif` behind a tiny trait so the orchestrator's
//! `(cur, total)` callback stays decoupled from rendering, and so tests can
//! drive progress-state transitions without a terminal.

use indicatif::{ProgressBar, ProgressStyle};

/// Anything that can consume `(cur, total)` progress updates.
pub trait ProgressSink {
    fn update(&mut self, cur: usize, total: usize);
    fn finish(&mut self);
}

/// Terminal progress bar (indicatif). Renders to stderr so stdout stays clean
/// for machine-readable output. indicatif auto-suppresses drawing on non-TTY
/// targets, so this is safe in CI/piped contexts too.
pub struct BarSink {
    bar: ProgressBar,
    last_total: usize,
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
        Self { bar, last_total: 0 }
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

    fn finish(&mut self) {
        self.bar.finish_and_clear();
    }
}

/// No-op sink used for `--no-progress` and non-TTY runs.
pub struct NullSink;

impl ProgressSink for NullSink {
    fn update(&mut self, _cur: usize, _total: usize) {}
    fn finish(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collects events so tests can assert progress-state transitions.
    struct RecordingSink {
        events: Vec<(usize, usize)>,
        finished: bool,
    }

    impl ProgressSink for RecordingSink {
        fn update(&mut self, cur: usize, total: usize) {
            self.events.push((cur, total));
        }
        fn finish(&mut self) {
            self.finished = true;
        }
    }

    #[test]
    fn sink_receives_monotonic_updates() {
        let mut sink = RecordingSink {
            events: Vec::new(),
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
    fn null_sink_is_noop() {
        let mut sink = NullSink;
        sink.update(1, 10);
        sink.finish();
        // No panic, no state — the contract is simply "does nothing".
    }
}

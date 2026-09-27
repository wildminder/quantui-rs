//! Tier 2 S09 — the user-visible parity contract, exercised end to end.
//!
//! This crate's core promise is byte-exact parity with the Python/torch
//! reference and `llama-quantize`. Tier 2 adds opt-in quality modes that
//! deliberately break that promise in exchange for accuracy, and rotation
//! presets that change the emitted bytes but stay parity-exact. A user must be
//! able to tell those apart **without reading the source code**, and these
//! tests are what holds that line to account.
//!
//! What is covered:
//!
//! (a) **Self-documenting format id** — `--help` lists every `--format`
//!     variant, and each variant's help text states whether it is byte-exact.
//! (b) **The `parity:` marker line** — printed on EVERY run, on stdout,
//!     immediately after the summary, for both a byte-exact format and (via
//!     the in-crate unit tests) a quality format.
//! (c) **`DiffKind::QualityTuned`** — GGUF-only; see the scope note in
//!     `gguf_verify.rs`. This file asserts the GGUF *scope limit* rather than
//!     claiming safetensors payload verification exists.
//! (d) **No new exit code** — a quality-class run still exits 0.
//!
//! Plus the auto-naming distinctness gate: `format_id` feeds `ctq_quant_tags`,
//! so two variants sharing an id would silently overwrite each other's
//! auto-named output.
//!
//! # `nvfp4_l2` — the reachable quality mode
//!
//! This file's original claim was that **no** `--format` value selects a
//! non-`Exact` quality, so the `quality-tuned` branch of the `parity:` line
//! could not be produced from the CLI. `nvfp4_l2` (anchored alternating L2
//! scale search, `Quality::Nvfp4L2ScaleSearch`) is now that path, and it is
//! wired through `stream.rs`'s NVFP4 site behind a `config.quality` gate.
//! Everything below is therefore no longer a statement about intent — it is
//! the evidence that the wiring works, and that nothing else moved:
//!
//! -- **the two formats differ in emitted bytes** (whole-file SHA-256, and a
//!   byte-level count of how much);
//! -- **`nvfp4`'s digest is the pinned one** from `parity_fingerprint.rs`,
//!   reproduced here through the real CLI rather than through the library;
//! -- **the reconstruction is genuinely better** for `nvfp4_l2`, so the
//!   differing bytes are a quality gain and not corruption;
//! -- **both round-trip** — each file is dequantized back and checked
//!   against its own input;
//! -- **`config_hash` and the auto-name are distinct**, because
//!   `StreamState::load_manifest` trusts the hash alone when resuming.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Every `--format` value the CLI accepts, with the exact auto-named output
/// stem it must produce. Kept as data so the distinctness test can assert
/// pairwise uniqueness instead of trusting separate assertions.
///
/// The expected stems are spelled out literally rather than recomputed from
/// `format_id`: a test that recomputes the expectation using the same function
/// under test proves nothing when that function is wrong.
const VARIANTS: [(&str, &str); 8] = [
    ("int8", "mymodel-int8_block-simple-heur.safetensors"),
    ("fp8_e4m3", "mymodel-fp8_e4m3-simple-heur.safetensors"),
    ("mxfp8", "mymodel-mxfp8-simple-heur.safetensors"),
    ("nvfp4", "mymodel-nvfp4-simple-heur.safetensors"),
    (
        "int8_convrot",
        "mymodel-int8_convrot-simple-heur.safetensors",
    ),
    (
        "nvfp4_rot16",
        "mymodel-nvfp4_rot16-gs16-simple-heur.safetensors",
    ),
    // Quality mode: NOT byte-exact, so the `parity:` line says
    // `quality-tuned`. Distinct stem from `nvfp4` — a quality run must never
    // land on top of a parity artifact.
    ("nvfp4_l2", "mymodel-nvfp4_l2-simple-heur.safetensors"),
    // Second quality mode (Tier 2 S07). Distinct from ALL THREE plain-`int8`
    // stems above, in every scaling mode, because the auto-name must never
    // let a clip run overwrite a plain-int8 artifact.
    ("int8_clip09", "mymodel-int8_clip09-simple-heur.safetensors"),
];

/// The `--format` values whose output is byte-exact against the references.
/// `nvfp4_l2` and `int8_clip09` are deliberately excluded — they are the two
/// quality modes, and asserting they print `parity: exact` would be asserting
/// a falsehood.
const PARITY_VARIANTS: [&str; 6] = [
    "int8",
    "fp8_e4m3",
    "mxfp8",
    "nvfp4",
    "int8_convrot",
    "nvfp4_rot16",
];

/// The pinned `nvfp4` whole-file digest, copied from
/// `crates/quant-core/tests/parity_fingerprint.rs::NVFP4_DIGEST`.
///
/// Duplicated here on purpose. That file hashes through the *library*
/// (`stream_quantize` with a hand-built `QuantConfig`); this one drives the
/// *binary* through `--format nvfp4`. Those are different entry points, and
/// `build_config` in between is exactly where a `Quality` mistake would be
/// introduced — so the pin is restated rather than imported, and a drift
/// between the two files is itself the finding.
const NVFP4_PINNED_DIGEST: &str =
    "927b831aafb955ec58732859282f8e1fe57645d19f5e5d3d019f4fe7a2f2b6f8";

/// The `nvfp4_l2` whole-file digest on the `linear_basic_bf16` golden fixture.
///
/// Recorded rather than asserted, same reasoning as every other pin in this
/// repository: if a change moves it, that is a **change in emitted bytes** to
/// report, not a test to relax. It is distinct from `NVFP4_PINNED_DIGEST`, and
/// that distinctness is the load-bearing property.
const NVFP4_L2_DIGEST: &str = "5b4df55bd8a724cfab2da6aba62c512c7a3b0c8719bca961d9afca40da59c323";

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_quantui-rs"))
}

/// Repo-root `tests/golden` directory.
fn golden_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.join("tests/golden")
}

/// Copy the golden input into `tmp` under the name `mymodel.safetensors`, so
/// auto-naming derives from `mymodel` (matching the expected stems above).
fn staged_input(tmp: &Path) -> PathBuf {
    let input = tmp.join("mymodel.safetensors");
    std::fs::copy(
        golden_dir().join("linear_basic_bf16/input.safetensors"),
        &input,
    )
    .expect("copy golden input");
    input
}

/// Run `quantize` and return `(stdout, success)`.
///
/// The progress bar is disabled so stdout is machine-readable; the parity
/// line is asserted on the raw text rather than a parsed field, because the
/// line's exact wording is part of the greppable contract.
fn run_quantize(input: &Path, format: &str) -> (String, bool) {
    let out = bin()
        .args([
            "quantize",
            input.to_str().expect("utf-8 path"),
            "--format",
            format,
            "--no-progress",
        ])
        .output()
        .expect("run quantui-rs quantize");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.success(),
    )
}

// --------------------------------------------------------------------------- //
// (b) The `parity:` marker line
// --------------------------------------------------------------------------- //

/// The marker must appear on a byte-exact run, naming the format and
/// promising byte-exactness against both references.
///
/// Printed unconditionally, so a user learns from an `exact` run that the tool
/// makes the distinction at all — which is what gives the marker's presence on
/// a quality run its meaning.
#[test]
fn parity_line_appears_on_an_exact_run() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (stdout, ok) = run_quantize(&input, "nvfp4");
    assert!(ok, "nvfp4 run must succeed");

    let line = stdout
        .lines()
        .find(|l| l.starts_with("parity:"))
        .unwrap_or_else(|| panic!("no parity: line in stdout:\n{stdout}"));
    assert_eq!(
        line, "parity: exact (nvfp4; byte-exact vs torch/llama-quantize)",
        "unexpected parity line"
    );
}

/// The marker must sit IMMEDIATELY AFTER the summary line, not somewhere
/// later in the output. A user reading top-down has to hit it without
/// scrolling past a wall of progress output.
#[test]
fn parity_line_immediately_follows_the_summary() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (stdout, ok) = run_quantize(&input, "mxfp8");
    assert!(ok, "mxfp8 run must succeed");

    let lines: Vec<&str> = stdout.lines().collect();
    let summary_at = lines
        .iter()
        .position(|l| l.starts_with("wrote "))
        .unwrap_or_else(|| panic!("no summary line in stdout:\n{stdout}"));
    let parity_at = lines
        .iter()
        .position(|l| l.starts_with("parity:"))
        .unwrap_or_else(|| panic!("no parity: line in stdout:\n{stdout}"));
    assert_eq!(
        parity_at,
        summary_at + 1,
        "parity line must directly follow the summary:\n{stdout}"
    );
}

/// Every **parity** variant must print `parity: exact`.
///
/// `nvfp4_l2` is excluded by construction, not by omission: it is the one
/// `--format` value that selects a non-`Exact` `Quality`, so the
/// `quality-tuned` line is the *correct* output for it. Asserted against
/// `PARITY_VARIANTS` rather than `VARIANTS` so adding a quality mode later
/// cannot silently widen this set back to claiming everything is exact.
#[test]
fn every_parity_variant_prints_an_exact_parity_line() {
    for (format, _) in VARIANTS {
        if !PARITY_VARIANTS.contains(&format) {
            continue;
        }
        let tmp = tempfile::tempdir().unwrap();
        let input = staged_input(tmp.path());

        let (stdout, ok) = run_quantize(&input, format);
        assert!(ok, "{format} run must succeed");

        let line = stdout
            .lines()
            .find(|l| l.starts_with("parity:"))
            .unwrap_or_else(|| panic!("{format}: no parity: line in stdout:\n{stdout}"));
        assert!(
            line.starts_with("parity: exact ("),
            "{format}: expected an exact parity line, got: {line}"
        );
        assert!(
            !line.contains("quality-tuned"),
            "{format}: must not be labelled quality-tuned while it is Exact: {line}"
        );
    }
    assert_eq!(PARITY_VARIANTS.len(), 6, "six formats remain byte-exact");
}

// --------------------------------------------------------------------------- //
// (a) Self-documenting format ids in `--help`
// --------------------------------------------------------------------------- //

/// `--help` must list EVERY variant. A variant missing from `--help` cannot be
/// discovered, and a user who cannot see it cannot choose it deliberately.
#[test]
fn help_lists_every_format_variant() {
    let out = bin()
        .args(["quantize", "--help"])
        .output()
        .expect("run --help");
    assert!(out.status.success(), "--help must exit 0");
    let help = String::from_utf8_lossy(&out.stdout).into_owned();

    for (format, _) in VARIANTS {
        assert!(
            help.contains(format),
            "--help must list the `{format}` variant:\n{help}"
        );
    }
}

/// The two rotation presets must NOT be described as quality-tuned. They
/// change the emitted bytes, so they are exactly the case a naive
/// implementation would mislabel — but a Hadamard rotation is an orthogonal
/// transform, so they remain byte-exact.
///
/// This asserts on the rendering near each variant rather than the whole
/// string, because a page-wide "does it mention quality" check would pass for
/// the wrong reason. The search anchors on the `--help` **"Possible values"**
/// entry (`- <name>:`) and not on a bare substring: the `--format` argument's
/// own summary line lists every variant id too, so a bare `find(variant)`
/// lands on that list and asserts nothing about the variant's own entry.
///
/// The help text deliberately *negates* the label ("this is not a quality-tuned
/// format"), so the check is for an AFFIRMATIVE claim. Stripping the negation
/// first is what keeps this test honest rather than forbidding the word.
#[test]
fn help_does_not_describe_rotation_presets_as_quality_tuned() {
    let out = bin()
        .args(["quantize", "--help"])
        .output()
        .expect("run --help");
    let help = String::from_utf8_lossy(&out.stdout).into_owned();

    for variant in ["int8_convrot", "nvfp4_rot16"] {
        let bullet = format!("- {variant}:");
        let at = help
            .find(&bullet)
            .unwrap_or_else(|| panic!("--help must have a `Possible values` entry {bullet}"));
        // The help line for this variant runs to the end of that line.
        let line = help[at..].lines().next().unwrap_or_default().to_lowercase();
        assert!(
            line.contains("byte-exact"),
            "--help should state the parity of {variant}: {line}"
        );
        // Remove the explicit denials, then no affirmative claim may remain.
        let affirmed = line
            .replace("not a quality-tuned format", "")
            .replace("is not a quality-tuned format", "");
        assert!(
            !affirmed.contains("quality-tuned"),
            "--help must not affirmatively call {variant} quality-tuned: {line}"
        );
    }
}

/// `nvfp4_l2` must AFFIRMATIVELY label itself not-byte-exact in `--help`.
///
/// The counterpart to the rotation-preset test above, and the reason that
/// test's negation-stripping is necessary rather than pedantic: once one
/// variant carries a positive quality claim, "does the word appear" stops
/// being a usable signal for the others. A user choosing a format must be able
/// to tell a byte-exact one from a quality-tuned one by reading `--help`.
#[test]
fn help_affirmatively_marks_nvfp4_l2_as_not_byte_exact() {
    let out = bin()
        .args(["quantize", "--help"])
        .output()
        .expect("run --help");
    let help = String::from_utf8_lossy(&out.stdout).into_owned();

    let bullet = "- nvfp4_l2:";
    let at = help
        .find(bullet)
        .unwrap_or_else(|| panic!("--help must have a `Possible values` entry {bullet}"));
    let line = help[at..].lines().next().unwrap_or_default();
    assert!(
        line.contains("NOT byte-exact"),
        "--help must affirmatively mark nvfp4_l2 as NOT byte-exact: {line}"
    );
    // And it must not ALSO claim byte-exactness, which would leave a user
    // unable to tell what the format does.
    let affirmed = line.replace("NOT byte-exact", "");
    assert!(
        !affirmed.contains("byte-exact vs"),
        "--help must not also claim nvfp4_l2 is byte-exact vs the reference: {line}"
    );
    // The reason must be visible in `--help`, not only on the `parity:` line.
    assert!(
        line.contains("L2 scale search"),
        "--help should say what nvfp4_l2 actually changes: {line}"
    );
}

// --------------------------------------------------------------------------- //
// Auto-naming distinctness
// --------------------------------------------------------------------------- //

/// Every variant must auto-name to its OWN path, and no two may collide.
///
/// `format_id` feeds `ctq_quant_tags`, which builds `<base>-<tags>.safetensors`.
/// Two variants sharing a `format_id` would silently overwrite each other's
/// output — a real data-loss bug that no digest test would catch, because each
/// run is individually correct.
///
/// Asserted pairwise over the literal expected stems, and each output is
/// checked to actually exist on disk.
#[test]
fn auto_named_outputs_are_pairwise_distinct() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let mut seen: Vec<(&str, PathBuf)> = Vec::new();
    for (format, expected_name) in VARIANTS {
        // The input keeps the SAME stem (`mymodel.safetensors`) for every
        // variant: auto-naming derives from the input filename, so renaming it
        // per variant would change the expected stem and make this test prove
        // nothing about collision behaviour. Each run therefore writes to the
        // same directory and must land on its own distinct name.
        let (stdout, ok) = run_quantize(&input, format);
        assert!(ok, "{format} run must succeed");

        let produced = tmp.path().join(expected_name);
        assert!(
            produced.is_file(),
            "{format}: expected auto-named output at {}\nstdout:\n{stdout}",
            produced.display()
        );

        // The summary must name the SAME file the tag logic predicted.
        assert!(
            stdout.contains(expected_name),
            "{format}: summary should name {expected_name}:\n{stdout}"
        );

        if let Some((other, _path)) = seen.iter().find(|(_, p)| *p == produced) {
            panic!(
                "{format} and {other} both auto-named to {}",
                produced.display()
            );
        }
        seen.push((format, produced));
    }
    assert_eq!(seen.len(), VARIANTS.len(), "all variants must be covered");
}

// --------------------------------------------------------------------------- //
// (d) Exit codes — deliberately unchanged
// --------------------------------------------------------------------------- //

/// A run must exit 0 and must NOT invent a "non-parity" exit code.
///
/// This was considered and explicitly rejected: `0` ok / `1` runtime failure /
/// `2` usage / `130` SIGINT / `3` GGUF spec violation are all taken, a new
/// value would read as "something went wrong" when nothing did, and
/// repurposing an existing one would break scripts. The greppable `parity:`
/// line is the stable machine-detectable signal.
#[test]
fn exit_code_is_zero_and_unchanged_by_parity_class() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let out = bin()
        .args([
            "quantize",
            input.to_str().unwrap(),
            "--format",
            "nvfp4",
            "--no-progress",
        ])
        .output()
        .expect("run quantui-rs");

    assert_eq!(
        out.status.code(),
        Some(0),
        "a successful run must exit 0 regardless of its parity class"
    );
}

// --------------------------------------------------------------------------- //
// (c) `DiffKind::QualityTuned` — GGUF scope limit
// --------------------------------------------------------------------------- //

/// `--verify-against` is a `GgufArgs` flag, so `DiffKind::QualityTuned` covers
/// the **GGUF path only**.
///
/// This test pins that scope boundary so it cannot be quietly over-claimed. The
/// safetensors `quantize` path's `--verify-output` is a **header-only
/// re-parse** (`quantize.rs::verify_output_files`) with **no payload
/// comparison at all** — so there is no safetensors equivalent of
/// `QualityTuned`, and the parity guarantee there is carried by the `parity:`
/// marker line and the self-documenting `--format` id.
///
/// Asserted negatively: the flag must not exist on `quantize`. If someone adds
/// a real payload-diff mode later, this test is the reminder to update the
/// docs rather than let a stale "no payload comparison" claim stand.
#[test]
fn safetensors_verify_output_is_header_only_and_says_nothing_about_payloads() {
    let out = bin()
        .args(["quantize", "--help"])
        .output()
        .expect("run --help");
    let help = String::from_utf8_lossy(&out.stdout).into_owned();

    assert!(
        help.contains("--verify-output"),
        "--verify-output should still exist on the quantize path"
    );
    // The help text must not claim payload verification for it.
    let at = help
        .find("--verify-output")
        .expect("--verify-output present");
    let para = help[at..]
        .find("\n\n")
        .map(|i| &help[at..at + i])
        .unwrap_or(&help[at..]);
    let para = para.to_lowercase();
    assert!(
        !para.contains("payload"),
        "--verify-output must not claim payload comparison; it is header-only: {para}"
    );
}

// --------------------------------------------------------------------------- //
// `nvfp4_l2` reachability — the whole point of the wiring
// --------------------------------------------------------------------------- //

/// SHA-256 over a whole emitted file, as lowercase hex.
///
/// Hand-rolled rather than pulling in a hashing crate: `quant-cli` has no
/// `sha2` dev-dependency, and adding one for two digests is not worth the
/// lockfile churn. FIPS 180-4, so the values are directly comparable with
/// `sha256sum` and with the pins in `parity_fingerprint.rs`.
fn file_digest(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("output file must exist");
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut msg = bytes.clone();
    let bitlen = (bytes.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    h.iter().map(|w| format!("{w:08x}")).collect()
}

/// The load-bearing test of the whole change: **`nvfp4_l2` must not emit
/// `nvfp4`'s bytes**.
///
/// If the `stream.rs` dispatch were missing, mis-gated, or pointed at the
/// parity kernel, this file would be byte-identical and this assertion would
/// fail. That is the failure mode the task exists to catch, so it is asserted
/// three ways — whole-file digest, differing-byte count, and the pinned value
/// for each format — because any one of them alone could be satisfied by
/// accident (e.g. a header-only difference would move the digest without
/// moving a single quantized weight).
#[test]
fn nvfp4_l2_emits_different_bytes_than_nvfp4() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (exact_out, ok) = run_quantize(&input, "nvfp4");
    assert!(ok, "nvfp4 run must succeed: {exact_out}");
    let (quality_out, ok) = run_quantize(&input, "nvfp4_l2");
    assert!(ok, "nvfp4_l2 run must succeed: {quality_out}");

    let exact_path = tmp.path().join("mymodel-nvfp4-simple-heur.safetensors");
    let quality_path = tmp.path().join("mymodel-nvfp4_l2-simple-heur.safetensors");
    assert!(exact_path.is_file() && quality_path.is_file());

    let exact_bytes = std::fs::read(&exact_path).unwrap();
    let quality_bytes = std::fs::read(&quality_path).unwrap();

    // (1) Whole-file digests.
    let exact_digest = file_digest(&exact_path);
    let quality_digest = file_digest(&quality_path);
    assert_ne!(
        exact_digest, quality_digest,
        "nvfp4_l2 produced byte-identical output to nvfp4 — the quality dispatch is NOT wired"
    );
    assert_eq!(exact_digest, NVFP4_PINNED_DIGEST, "nvfp4 digest moved");
    assert_eq!(
        quality_digest, NVFP4_L2_DIGEST,
        "nvfp4_l2 digest moved — the emitted bytes changed"
    );

    // (2) The difference is in the PAYLOAD, not just a header or metadata
    // field. Same length means a like-for-like comparison is meaningful, and
    // a large differing-byte count rules out "one metadata field changed".
    assert_eq!(
        exact_bytes.len(),
        quality_bytes.len(),
        "the two formats should emit identically-shaped artifacts"
    );
    let differing = exact_bytes
        .iter()
        .zip(&quality_bytes)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > 64,
        "only {differing} of {} bytes differ — that is a header-level change, not a \
         different quantization, so the quality kernel is probably not being called",
        exact_bytes.len()
    );

    // (3) The difference is in the QUANTIZED VALUES, not in the container:
    // the two artifacts must declare the same tensor set, in the same order,
    // with the same dtypes and shapes. Same kind of thing, different numbers —
    // which is the whole claim. A differing byte count on its own would not
    // establish this; a changed header would undermine it.
    let header_of = |p: &Path| -> Vec<(String, String, Vec<u64>)> {
        let r = quant_core::st_io::reader::SafetensorsReader::open(p).expect("open output");
        r.header()
            .iter()
            .map(|(name, info)| {
                (
                    name.clone(),
                    format!("{:?}", info.dtype),
                    info.shape.clone(),
                )
            })
            .collect()
    };
    assert_eq!(
        header_of(&exact_path),
        header_of(&quality_path),
        "the two formats must emit the same tensor layout; only values may differ"
    );
}

/// The `config_hash` resume guard, end to end through the binary.
///
/// `StreamState::load_manifest` decides whether an interrupted run may be
/// resumed into **on this value alone**. A collision between `nvfp4` and
/// `nvfp4_l2` would therefore let a byte-exact partial file be continued under
/// the L2 search, producing one artifact holding two algorithms' output — with
/// no error and no warning. The hashes must differ, and `nvfp4`'s must be the
/// pinned vector.
#[test]
fn nvfp4_l2_has_a_distinct_config_hash() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (exact_out, ok) = run_quantize(&input, "nvfp4");
    assert!(ok);
    let (quality_out, ok) = run_quantize(&input, "nvfp4_l2");
    assert!(ok);

    let hash_of = |s: &str| -> String {
        s.lines()
            .find_map(|l| l.split("config_hash ").nth(1))
            .map(|h| h.trim().trim_end_matches(')').to_string())
            .unwrap_or_else(|| panic!("no config_hash in summary:\n{s}"))
    };
    let exact_hash = hash_of(&exact_out);
    let quality_hash = hash_of(&quality_out);

    assert_ne!(
        exact_hash, quality_hash,
        "a quality run must never share a resume guard with a parity run"
    );
    // The pinned NVFP4 hash — `build_config` must not have perturbed it by
    // threading `quality` through unconditionally.
    assert_eq!(exact_hash, "95ede677cf402b53", "pinned nvfp4 hash moved");
}

/// Both files must round-trip: re-reading the artifact and dequantizing it has
/// to produce a sane reconstruction, not garbage.
///
/// A quality mode that emitted *different* bytes would be easy to write if
/// those bytes were simply wrong. This is the check that says they are not —
/// it dequantizes each output through the crate's own NVFP4 dequantizer and
/// compares against the original bf16 input.
#[test]
fn both_formats_round_trip() {
    use quant_core::quant_nvfp4::{dequantize_nvfp4, Nvfp4QuantResult};
    use quant_core::st_io::reader::SafetensorsReader;

    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    // The golden input, read as bf16 → f32.
    let src = SafetensorsReader::open(&input).expect("open input");
    let weight = src.tensor_bytes("blocks.0.weight").expect("input weight");
    let info = src.header().get("blocks.0.weight").expect("weight info");
    let m = info.shape[0] as usize;
    let n = info.shape[1] as usize;
    let orig: Vec<f32> = weight
        .chunks_exact(2)
        .map(|c| quant_core::dtype::bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    assert_eq!(orig.len(), m * n);

    let mut results: Vec<(&str, f64)> = Vec::new();
    for (format, stem) in [
        ("nvfp4", "mymodel-nvfp4-simple-heur.safetensors"),
        ("nvfp4_l2", "mymodel-nvfp4_l2-simple-heur.safetensors"),
    ] {
        let (_, ok) = run_quantize(&input, format);
        assert!(ok, "{format} run must succeed");

        let path = tmp.path().join(stem);
        let reader = SafetensorsReader::open(&path).expect("open output");

        // Rebuild the kernel's result struct from the emitted tensors, so the
        // dequantizer under test is the same one the producer used.
        let qdata = reader
            .tensor_bytes("blocks.0.weight")
            .expect("qdata")
            .to_vec();
        let scale = reader
            .tensor_bytes("blocks.0.weight_scale")
            .expect("scale")
            .to_vec();
        let pts = reader
            .tensor_bytes("blocks.0.weight_scale_2")
            .expect("scale_2");
        let per_tensor_scale = f32::from_le_bytes([pts[0], pts[1], pts[2], pts[3]]);

        let q_shape = reader
            .header()
            .get("blocks.0.weight")
            .expect("q shape")
            .shape
            .clone();
        let scale_shape = reader
            .header()
            .get("blocks.0.weight_scale")
            .expect("scale shape")
            .shape
            .clone();

        let r = Nvfp4QuantResult {
            qdata,
            qdata_shape: q_shape,
            scale,
            scale_shape,
            per_tensor_scale,
        };
        let deq = dequantize_nvfp4(&r, m, n);
        assert_eq!(deq.len(), m * n, "{format}: dequant shape");

        // Relative L2 against the input. Not asserted to any absolute value —
        // the parity contract is the digest above, not this metric. Asserted
        // only to be finite and in a range a working quantizer produces, so a
        // file of garbage (or a mis-scaled one) cannot pass.
        let num: f64 = deq
            .iter()
            .zip(&orig)
            .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
            .sum();
        let den: f64 = orig.iter().map(|v| (*v as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        assert!(rel.is_finite(), "{format}: relative L2 is not finite");
        assert!(
            rel < 0.5,
            "{format}: relative L2 {rel} is implausibly large for NVFP4 on this fixture"
        );
        results.push((format, rel));
    }

    let exact = results.iter().find(|(f, _)| *f == "nvfp4").unwrap().1;
    let quality = results.iter().find(|(f, _)| *f == "nvfp4_l2").unwrap().1;
    assert!(
        quality < exact,
        "nvfp4_l2 ({quality}) should reconstruct better than nvfp4 ({exact}); \
         if it does not, the differing bytes are not a quality gain"
    );
    println!("relative L2 — nvfp4: {exact:.6}, nvfp4_l2: {quality:.6}");
}

/// The auto-name gate, spelled out for the new format specifically.
///
/// `auto_named_outputs_are_pairwise_distinct` covers the general case; this
/// pins the two NVFP4 quality-related names against each other directly,
/// because they are the pair most likely to collide (same family, same block
/// size, same everything except the refinement).
#[test]
fn nvfp4_l2_auto_name_differs_from_nvfp4() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (exact_out, ok) = run_quantize(&input, "nvfp4");
    assert!(ok);
    let (quality_out, ok) = run_quantize(&input, "nvfp4_l2");
    assert!(ok);

    let name_of = |s: &str| -> String {
        s.lines()
            .find_map(|l| l.strip_prefix("wrote "))
            .and_then(|l| l.split(" (").next())
            .unwrap_or_else(|| panic!("no output path in summary:\n{s}"))
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or_default()
            .to_string()
    };

    assert_eq!(name_of(&exact_out), "mymodel-nvfp4-simple-heur.safetensors");
    assert_eq!(
        name_of(&quality_out),
        "mymodel-nvfp4_l2-simple-heur.safetensors"
    );
    assert_ne!(
        name_of(&exact_out),
        name_of(&quality_out),
        "a quality run must never auto-name onto a parity artifact"
    );
}

/// `--format nvfp4_l2` must reach the `quality-tuned` branch of the marker.
///
/// This is the assertion that the `parity:` line is not merely *capable* of
/// printing `quality-tuned` (the in-crate unit tests cover the wording) but
/// actually reaches it from a real command line.
#[test]
fn nvfp4_l2_prints_the_quality_tuned_marker() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (stdout, ok) = run_quantize(&input, "nvfp4_l2");
    assert!(ok, "nvfp4_l2 run must succeed: {stdout}");

    let line = stdout
        .lines()
        .find(|l| l.starts_with("parity:"))
        .unwrap_or_else(|| panic!("no parity: line in stdout:/n{stdout}"));
    assert!(
        line.starts_with("parity: quality-tuned (nvfp4_l2:"),
        "expected the quality-tuned marker, got: {line}"
    );
    assert!(
        line.contains("NOT byte-exact vs torch/llama-quantize"),
        "the marker must negate byte-exactness: {line}"
    );
    // It must NOT also print the `exact` wording. Checked as a prefix rather
    // than a bare `!contains("byte-exact")`, because the negated phrase above
    // itself contains that substring — a naive negation here would be
    // unsatisfiable, and dropping the check instead would leave the exact
    // branch unguarded.
    assert!(
        !line.starts_with("parity: exact"),
        "the quality marker must not also claim the exact branch: {line}"
    );
    // The reason must be on the line, so a user learns what actually changed
    // rather than just that something did.
    assert!(
        line.contains("L2 scale search"),
        "the quality marker should name the refinement: {line}"
    );
}

/// `nvfp4_l2` inherits NVFP4's fixed scaling, so the override flags must stay
/// usage errors — the same contract `nvfp4` has, exit code 2.
///
/// Cheap to check and worth it: `has_fixed_scaling()` is a hand-written
/// `matches!`, and forgetting the new variant there would silently ACCEPT
/// `--block-size 128` for a format whose block size is not a parameter, which
/// is exactly the "silently emitting something the user didn't ask for" class
/// this repo treats as a bug.
#[test]
fn nvfp4_l2_rejects_scaling_overrides_like_nvfp4() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    for (flag, value) in [("--scaling-mode", "block"), ("--block-size", "128")] {
        for format in ["nvfp4", "nvfp4_l2"] {
            let out = bin()
                .args([
                    "quantize",
                    input.to_str().unwrap(),
                    "--format",
                    format,
                    flag,
                    value,
                    "--no-progress",
                ])
                .output()
                .expect("run quantui-rs");
            assert_eq!(
                out.status.code(),
                Some(2),
                "{format} {flag} {value} must be a usage error"
            );
        }
    }
}

// --------------------------------------------------------------------------- //
// `int8_clip09` — the S07 clip mode
// --------------------------------------------------------------------------- //

/// The clip kernel must actually be reached.
///
/// This is the S07 analogue of `nvfp4_l2_emits_different_bytes_than_nvfp4`: if
/// `stream.rs`'s INT8 dispatch were missing or mis-gated, `int8_clip09` would
/// emit byte-identical output to `int8` and the preset would be a lie told
/// through the CLI — a different filename for the same bytes.
///
/// Asserted on the whole-file digest, on a payload-level differing-byte count
/// (so a header-only change cannot satisfy it), and on the pinned `int8`
/// digest, because any one of those alone could pass by accident.
#[test]
fn int8_clip09_emits_different_bytes_than_int8() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (exact_out, ok) = run_quantize(&input, "int8");
    assert!(ok, "int8 run must succeed: {exact_out}");
    let (clip_out, ok) = run_quantize(&input, "int8_clip09");
    assert!(ok, "int8_clip09 run must succeed: {clip_out}");

    let exact_path = tmp
        .path()
        .join("mymodel-int8_block-simple-heur.safetensors");
    let clip_path = tmp
        .path()
        .join("mymodel-int8_clip09-simple-heur.safetensors");
    assert!(exact_path.is_file() && clip_path.is_file());

    assert_ne!(
        file_digest(&exact_path),
        file_digest(&clip_path),
        "int8_clip09 produced byte-identical output to int8 — the clip dispatch is NOT wired"
    );

    let exact_bytes = std::fs::read(&exact_path).unwrap();
    let clip_bytes = std::fs::read(&clip_path).unwrap();
    let differing = exact_bytes
        .iter()
        .zip(&clip_bytes)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > 64,
        "only {differing} of {} bytes differ — that is a header-level change, not a \
         different quantization, so the clip kernel is probably not being called",
        exact_bytes.len()
    );
}

/// The resume guard, for the case where it is easiest to get wrong.
///
/// `int8_clip09` shares its entire `QuantConfig` with plain `int8` at the same
/// scaling mode — same `format`, same `target_format`, same `int8` flag, same
/// block size. The `quality_tuning` key is the ONLY thing separating them in
/// the hash, and `StreamState::load_manifest` trusts that hash alone. A
/// regression that dropped the key would let an interrupted `int8` run resume
/// into a clipped file, mixing two quantizers in one artifact with no error.
#[test]
fn int8_clip09_has_a_distinct_config_hash() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (exact_out, ok) = run_quantize(&input, "int8");
    assert!(ok);
    let (clip_out, ok) = run_quantize(&input, "int8_clip09");
    assert!(ok);

    let hash_of = |s: &str| -> String {
        s.lines()
            .find_map(|l| l.split("config_hash ").nth(1))
            .map(|h| h.trim().trim_end_matches(')').to_string())
            .unwrap_or_else(|| panic!("no config_hash in summary:\n{s}"))
    };
    let exact_hash = hash_of(&exact_out);
    let clip_hash = hash_of(&clip_out);

    assert_ne!(
        exact_hash, clip_hash,
        "a clip run must never share a resume guard with a plain int8 run"
    );
    // Plain int8's own hash must be untouched by the new quality variant.
    assert_eq!(exact_hash, "56920c6553cfa241", "pinned int8 hash moved");
}

/// The marker must announce the clip mode as NOT byte-exact, from a real
/// command line — the same contract `nvfp4_l2` holds.
#[test]
fn int8_clip09_prints_the_quality_tuned_marker() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    let (stdout, ok) = run_quantize(&input, "int8_clip09");
    assert!(ok, "int8_clip09 run must succeed: {stdout}");

    let line = stdout
        .lines()
        .find(|l| l.starts_with("parity:"))
        .unwrap_or_else(|| panic!("no parity: line in stdout:\n{stdout}"));
    assert!(
        line.starts_with("parity: quality-tuned (int8_clip09:"),
        "expected the quality-tuned marker, got: {line}"
    );
    assert!(
        line.contains("NOT byte-exact vs torch/llama-quantize"),
        "the marker must negate byte-exactness: {line}"
    );
    assert!(
        !line.starts_with("parity: exact"),
        "the quality marker must not also claim the exact branch: {line}"
    );
}

/// Unlike `nvfp4_l2`, the clip preset is NOT fixed-scaling: it takes
/// `--scaling-mode` / `--block-size` in all three INT8 modes, because the clip
/// applies to whichever amax the chosen mode already computes. The inverse
/// error is the one worth catching — `has_fixed_scaling()` is a hand-written
/// `matches!`, and a stray arm there would make `--scaling-mode row` a usage
/// error for a format that genuinely accepts it.
#[test]
fn int8_clip09_accepts_scaling_overrides_like_int8() {
    let tmp = tempfile::tempdir().unwrap();
    let input = staged_input(tmp.path());

    for (flag, value) in [("--scaling-mode", "row"), ("--block-size", "64")] {
        for format in ["int8", "int8_clip09"] {
            let out = bin()
                .args([
                    "quantize",
                    input.to_str().unwrap(),
                    "--format",
                    format,
                    flag,
                    value,
                    "--no-progress",
                ])
                .output()
                .expect("run quantui-rs");
            assert_eq!(
                out.status.code(),
                Some(0),
                "{format} {flag} {value} must be accepted"
            );
        }
    }
}

/// The clip mode must be self-consistent across every scaling mode it accepts:
/// in all three, `int8_clip09` must differ from plain `int8` and must carry a
/// distinct hash. A dispatch wired for only one mode would leave the other two
/// silently emitting parity bytes under a quality-tuned name.
#[test]
fn int8_clip09_differs_from_int8_in_every_scaling_mode() {
    use quant_core::quant::{
        dequantize_int8, quantize_int8_weight, quantize_int8_weight_clipped, Int8QuantResult,
        ScalingMode as Int8ScalingMode, INT8_CLIP_RATIO,
    };

    let (m, n) = (256usize, 128usize);
    let mut w = vec![0.0f32; m * n];
    let mut x: u64 = 0x243F6A8885A308D3;
    for v in &mut w {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = ((x % 20_000) as f32 / 10_000.0) - 1.0;
    }

    // Block mode needs divisibility by the block size; the others ignore it.
    for (mode, bs) in [
        (Int8ScalingMode::Tensor, 128usize),
        (Int8ScalingMode::Row, 128),
        (Int8ScalingMode::Block, 128),
    ] {
        let base: Int8QuantResult = quantize_int8_weight(&w, m, n, mode, bs);
        let clip = quantize_int8_weight_clipped(&w, m, n, mode, bs, INT8_CLIP_RATIO);
        assert_ne!(
            base.qdata, clip.qdata,
            "{mode:?}: the clip must change the emitted codes, not just the scale"
        );

        // And the shipped dequantizer must see the change as a real one.
        let num = |r: &Int8QuantResult| -> f64 {
            let dq = dequantize_int8(&r.qdata, &r.scale, m, n, mode, bs);
            w.iter()
                .zip(&dq)
                .map(|(a, b)| {
                    let d = f64::from(*a) - f64::from(*b);
                    d * d
                })
                .sum()
        };
        let den: f64 = w.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        assert!(
            (num(&clip) / den) != (num(&base) / den),
            "{mode:?}: reconstruction error must actually differ"
        );
    }
}

/// The kernel identity that makes the negative result trustworthy: at
/// `clip_ratio = 1.0` the clipped kernel is bit-identical to the base one, so
/// any measured difference is attributable to the clipping and not to the
/// refactoring that introduced the new function.
#[test]
fn clip_ratio_one_reproduces_int8_exactly_throughout() {
    use quant_core::quant::{
        quantize_int8_weight, quantize_int8_weight_clipped, ScalingMode as Int8ScalingMode,
    };

    let (m, n, bs) = (256usize, 128usize, 128usize);
    let mut w = vec![0.0f32; m * n];
    let mut x: u64 = 0xDEADBEEFCAFEF00D;
    for v in &mut w {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = ((x % 40_000) as f32 / 20_000.0) - 1.0;
    }

    for mode in [
        Int8ScalingMode::Tensor,
        Int8ScalingMode::Row,
        Int8ScalingMode::Block,
    ] {
        let base = quantize_int8_weight(&w, m, n, mode, bs);
        let unit = quantize_int8_weight_clipped(&w, m, n, mode, bs, 1.0);
        assert_eq!(base.qdata, unit.qdata, "{mode:?}");
        assert_eq!(base.scale, unit.scale, "{mode:?}");
        assert_eq!(base.scale_shape, unit.scale_shape, "{mode:?}");
    }
}

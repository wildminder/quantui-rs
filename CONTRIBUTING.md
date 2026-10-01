# Contributing to quantui-rs

## Development setup

```sh
git clone https://github.com/wildminder/quantui-rs.git
cd quantui-rs
cargo build --release
```

The workspace requires Rust 1.89 or newer. `cargo` installs the pinned
`rlx-gguf 0.2.14` dependency automatically.

## Required checks

Run the same gates as CI before opening a pull request:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --release --locked
```

The GGUF end-to-end sweep is also expected for changes that affect conversion
or method resolution:

```sh
bash tools/sweep_gguf_e2e.sh   # expected: passed: 49  failed: 0
```

## Byte-parity rule

The ComfyUI formats and GGUF encoders are verified against committed golden
fixtures and reference implementations. Changes to kernels, tensor ordering,
scale formulas, header layout, or GGUF scheme selection must preserve
byte-exact parity. Do not reformat, simplify, or "clean up" verbatim-port code
merely to satisfy a linter; the workspace lint allow-list exists for this
reason.

When a change intentionally alters output bytes, include the regenerated
fixture, the tool used to generate it, and the reference command in the pull
request.

## Pull requests

- Keep changes focused and describe the observable behavior change.
- Add or update tests for every behavioral change.
- Update `README.md` when CLI flags, output formats, or supported methods
  change.
- Update `THIRD_PARTY_NOTICES.md` when dependency or provenance information
  changes.

## Cutting a release

Releases are tag-driven. `.github/workflows/release.yml` runs on any `v*` tag
push, builds the binaries, and attaches them to the GitHub Release.

```sh
# 1. Bump the version in TWO places — the workspace root Cargo.toml and
#    crates/quant-cli/Cargo.toml (its path dependency on quant-core). Both
#    must agree or cargo refuses the workspace. Then commit that on its own.
git commit -am "release: v0.3.1"

# 2. Tag that commit and push the tag.
git tag v0.3.1
git push origin main --follow-tags
```

`crates/quant-core/Cargo.toml` inherits `version` from `[workspace.package]`, so
it needs no edit. Confirm with `grep -rn '^\s*version' Cargo.toml crates/*/Cargo.toml`.

The version must be identical in `Cargo.toml` and the tag. `verify-tag` fails
the run before building anything if they disagree, because the PE version
resource is stamped from `CARGO_PKG_VERSION` and a mismatch would ship a binary
whose Explorer details and `--version` both report the wrong number.

What gets attached:

| Platform | Runner | Target |
|----------|--------|--------|
| Linux x86_64 | `ubuntu-latest` | `x86_64-unknown-linux-musl` |
| Linux ARM64 | `ubuntu-24.04-arm` | `aarch64-unknown-linux-musl` |
| Windows x86_64 | `windows-latest` | `x86_64-pc-windows-msvc` |
| macOS Apple Silicon | `macos-latest` | `aarch64-apple-darwin` |
| macOS Intel | `macos-15-intel` | `x86_64-apple-darwin` |

Plus one `SHA256SUMS.txt` covering all five, so a user can verify a download
with `sha256sum -c SHA256SUMS.txt`.

Notes on the choices:

- **Linux binaries are static** (musl). A glibc-linked build from ubuntu-24.04
  carries a glibc 2.39 floor and refuses to start on older distributions, which
  is the most common "the download is broken" report for a Rust CLI.
- **Each target builds on a runner of its own architecture**, so nothing is
  cross-compiled. That keeps a linker out of the picture and lets every artifact
  be smoke-tested by running it.
- **`macos-latest` is ARM64.** The Intel build needs the explicit
  `macos-15-intel` label or the job ships a second ARM binary under an x86_64
  file name.
- **Archives are reproducible.** mtime, uid, gid, and user name are all pinned
  to zero, so the same commit produces the same SHA256 on every run.

Release notes come from a committed `RELEASE_NOTES.md` at the repo root when
one exists, otherwise GitHub generates them from the commit list. Re-running the
workflow uploads to the existing Release instead of failing.

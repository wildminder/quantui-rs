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

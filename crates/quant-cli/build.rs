//! Embeds `assets/icon.ico` into the `quantui-rs.exe` resource section.
//!
//! The icon ships INSIDE the binary, not beside it: a CLI has no install
//! directory or launcher to hang an `.ico` on, so an exe without an embedded
//! icon shows the generic blank-page glyph in Explorer, the taskbar, and the
//! Alt-Tab switcher. `build.rs` is the only place that can do it, because the
//! icon must be in the PE resource section before the linker runs.
//!
//! Windows-only by construction: the whole body is gated on
//! `CARGO_CFG_TARGET_OS == "windows"`, i.e. the TARGET, not the host. That
//! matters — a build script runs on the host, so a `#[cfg(windows)]` gate would
//! fire even for `cargo build --target x86_64-unknown-linux-gnu` from Windows
//! and try to embed a PE resource into a Linux binary. With the target gate the
//! Linux and macOS CI legs stay byte-identical to before: no cross-compilation
//! of resources, no `.res` file, no behaviour change at all.
//!
//! Version metadata is written alongside the icon so Explorer's Details tab
//! reports the real version from `Cargo.toml` instead of `0.0.0.0`. Both are
//! read from the manifest so they can never drift from the crate version.
//!
//! Regenerating the icon: `python tools/gen_icon.py` (see that file — the SVG
//! is the source, this only consumes the `.ico`).

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Gate on the TARGET, not the host.
    //
    // A build script is compiled for, and runs on, the HOST — so `#[cfg(windows)]`
    // is true whenever you build on Windows, INCLUDING `cargo build
    // --target x86_64-unknown-linux-gnu` from Windows. That silently tries to
    // embed a resource into a Linux binary and dies with "program not found"
    // (it goes looking for an `rc.exe`). Cargo exposes the intended target as
    // CARGO_CFG_TARGET_OS precisely so this decision can be made correctly.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        // Nothing to embed: PE resources are a Windows-only concept, and the
        // Linux/macOS CI legs must stay byte-identical to before. `icon.svg` /
        // `icon.ico` remain a repo asset for docs and for Windows releases.
        return;
    }

    // The icon lives at the workspace root, not next to this crate: it is a
    // project-wide asset (the README shows the same file).
    //
    // Deliberately NOT `canonicalize()`d. On Windows that returns a
    // `\\?\` VERBATIM path, and `rc.exe` cannot open one — it fails with a
    // bare "path not found", which reads like a missing SDK rather than a
    // path-syntax problem. Building the relative path by hand also keeps the
    // `rerun-if-changed` path readable in build output.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let icon = manifest.join("../../assets/icon.ico");
    if !icon.is_file() {
        panic!(
            "assets/icon.ico must exist and be readable at {}\nrun: python tools/gen_icon.py",
            icon.display()
        );
    }

    println!("cargo:rerun-if-changed={}", icon.display());

    let version = std::env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION");

    {
        let mut res = winresource::WindowsResource::new();

        // Locate rc.exe ourselves and hand winresource the EXACT binary via its
        // `RC_PATH` override.
        //
        // Why not just `set_toolkit_path`: winresource appends `bin\<arch>\rc.exe`
        // to whatever root it is given, but on a real Windows 11 + VS 2022 host
        // the layout is `<root>\bin\<version>\<arch>\rc.exe` — the version
        // segment is in the way, so no single root satisfies it. Its own
        // discovery shells out to `reg.exe`, which a build script should not
        // depend on and which fails on a locked-down host, leaving a bare
        // relative `bin\x64\rc.exe` and a misleading os error 3.
        //
        // Setting RC_PATH bypasses both problems: it is an absolute, verified
        // path to the one file we actually need.
        let rc_exe = find_rc_exe();
        std::env::set_var("RC_PATH", &rc_exe);
        res.set_toolkit_path(&sdk_include_root().to_string_lossy());

        res.set_icon(icon.to_str().expect("icon path is valid UTF-8"));
        // Explorer Details tab. `CompanyName` is deliberately absent: this is
        // an individual open-source project and inventing a vendor would be a
        // false claim in the file properties.
        res.set(
            "FileDescription",
            "quantui-rs — safetensors to ComfyUI/GGUF quantizer",
        );
        res.set("ProductName", "quantui-rs");
        res.set("InternalName", "quantui-rs");
        res.set("OriginalFilename", "quantui-rs.exe");
        res.set("LegalCopyright", "MIT licensed");

        // The four 16-bit words are MAJOR/MINOR/PATCH/RELEASE. `0.3.0` maps to
        // 0.3.0.0, which is what Explorer shows. Parsed (not string-concatenated)
        // so a two-part version like `1.0` cannot silently shift the fields.
        let mut words = [0u64; 4];
        for (i, part) in version.split('.').take(4).enumerate() {
            words[i] = part.parse::<u64>().unwrap_or(0);
        }
        use winresource::VersionInfo;
        res.set_version_info(
            VersionInfo::FILEVERSION,
            (words[0] << 48) | (words[1] << 32) | (words[2] << 16) | words[3],
        );
        res.set_version_info(
            VersionInfo::PRODUCTVERSION,
            (words[0] << 48) | (words[1] << 32) | (words[2] << 16) | words[3],
        );
        // 1 = VFT_APP, an .exe. Without this Explorer shows the field blank.
        res.set_version_info(VersionInfo::FILETYPE, 1);
        // VOS_NT_WINDOWS32
        res.set_version_info(VersionInfo::FILEOS, 0x0004_0004);

        res.compile()
            .expect("failed to embed the .ico into the exe resource section");
    }
}

/// Absolute path to a `rc.exe` that actually exists.
///
/// Pure filesystem probing under the two well-known Windows Kit roots — no
/// subprocess and no registry read. winresource's own discovery shells out to
/// `reg.exe`; that is a subprocess a build script should not depend on, and on
/// a locked-down host it fails, leaving a bare relative `bin\x64\rc.exe` and a
/// misleading "path not found" (os error 3) that reads like a missing SDK.
///
/// Compiled unconditionally, NOT under `#[cfg(windows)]`. These two helpers
/// describe the *build machine's* filesystem, not the compilation target, so
/// gating them on the host would break `cargo build --target
/// x86_64-pc-windows-msvc` from Linux — the exact case the main gate above
/// exists to support. They are pure `std::fs` probing and compile anywhere;
/// `dead_code` only fires on the non-Windows-target path that never calls them.
///
/// Layout as actually observed on Windows 11 + VS 2022 (the version segment
/// sits between `bin` and the arch, which is why a single "toolkit root" cannot
/// express it and why we hand over an absolute path instead):
///
///     <root>\bin\<version>\<arch>\rc.exe
///
/// The older `<root>\<version>\bin\<arch>\rc.exe` and a versionless
/// `<root>\bin\<arch>\rc.exe` are also accepted. Newest version wins,
/// compared NUMERICALLY: lexicographic order gets it backwards
/// ("10.0.9999.0" sorts above "10.0.26100.0" but is older).
#[allow(dead_code)]
fn find_rc_exe() -> std::path::PathBuf {
    use std::path::{Path, PathBuf};

    let arch_dir = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "x86") {
        "x86"
    } else {
        "x64"
    };
    let roots = [
        r"C://Program Files (x86)//Windows Kits//10",
        r"C://Program Files//Windows Kits//10",
    ];

    /// `10.0.26100.0` -> 26100, for a numeric (not lexicographic) compare.
    fn kit_version(name: &std::ffi::OsStr) -> Option<u32> {
        name.to_str()?
            .strip_prefix("10.0.")?
            .split('.')
            .next()?
            .parse::<u32>()
            .ok()
    }

    let mut best: Option<(u32, PathBuf)> = None;
    let mut take = |v: u32, p: PathBuf| {
        if p.is_file() && best.as_ref().is_none_or(|(bv, _)| v > *bv) {
            best = Some((v, p));
        }
    };

    for root in roots {
        // Layout 1: <root>/bin/<version>/<arch>/rc.exe
        if let Ok(entries) = std::fs::read_dir(Path::new(root).join("bin")) {
            for e in entries.flatten() {
                let rc = e.path().join(arch_dir).join("rc.exe");
                take(kit_version(&e.file_name()).unwrap_or(0), rc);
            }
        }
        // Layout 2: <root>/<version>/bin/<arch>/rc.exe
        if let Ok(entries) = std::fs::read_dir(root) {
            for e in entries.flatten() {
                let rc = e.path().join("bin").join(arch_dir).join("rc.exe");
                take(kit_version(&e.file_name()).unwrap_or(0), rc);
            }
        }
    }

    best.map(|(_, p)| p).unwrap_or_else(|| {
        // Nothing found. Fall back to the conventional location so the error
        // names a concrete missing path instead of "not found".
        Path::new(roots[0])
            .join("bin")
            .join(arch_dir)
            .join("rc.exe")
    })
}

/// Root that `winresource` will derive its include dirs from when we let it
/// add the toolkit includes. It walks up from `RC_PATH`, so pointing it at the
/// kit root (rather than at rc.exe itself) keeps that logic working.
///
/// Compiled unconditionally for the same reason as `find_rc_exe` — it inspects
/// the build machine, not the target.
#[allow(dead_code)]
fn sdk_include_root() -> std::path::PathBuf {
    use std::path::Path;
    for root in [
        r"C://Program Files (x86)//Windows Kits//10",
        r"C://Program Files//Windows Kits//10",
    ] {
        if Path::new(root).join("Include").is_dir() {
            return Path::new(root).to_path_buf();
        }
    }
    Path::new(r"C://Program Files (x86)//Windows Kits//10").to_path_buf()
}

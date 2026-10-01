//! Regression tests for FOLDER input to the single-file code paths.
//!
//! # The bug this pins
//!
//! `quantize`, `cast` and `info` used to open the input path directly. But
//! `discover::classify_input` returns `InputKind::SingleFile` for TWO different
//! shapes — a path that is itself a `.safetensors` file, and a plain
//! HuggingFace folder that merely *contains* exactly one `.safetensors` — while
//! only reporting a `(kind, base_name)` pair, never WHICH file is inside. So a
//! folder input reached `File::open` as a directory, and on Windows that is
//! `ERROR_ACCESS_DENIED`:
//!
//! ```text
//! error: io error on C:\_Models\VibeVoice-Realtime-0.5B: Отказано в доступе. (os error 5)
//! ```
//!
//! # Why these are CLI tests and not more unit tests
//!
//! `discover::resolve_single_file` already has unit tests. Those prove the
//! *resolver* is right; they cannot prove any *caller* actually calls it. That
//! gap — unit correct, caller untouched — is exactly the trap this repo has
//! fallen into repeatedly, so the assertions below drive the real binary and
//! would all still pass if a call site silently reverted to opening the input
//! path itself.
//!
//! Fixtures are built in-process into `tempfile` dirs (the pattern used by
//! `cli_cast.rs`), so nothing here depends on a committed binary blob.

use std::path::{Path, PathBuf};
use std::process::Command;

// --------------------------------------------------------------------------- //
// Fixture builder (mirrors cli_cast.rs)
// --------------------------------------------------------------------------- //

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_quantui-rs"))
}

fn align_header_to_8(header: &[u8]) -> Vec<u8> {
    let rem = header.len() % 8;
    if rem == 0 {
        return header.to_vec();
    }
    let mut out = header.to_vec();
    out.extend(std::iter::repeat_n(b' ', 8 - rem));
    out
}

/// One tensor to build: `(name, dtype, shape, payload_bytes)`.
///
/// Aliased because the sharded-folder fixture nests this inside a slice of
/// slices, and the spelled-out 4-tuple trips `clippy::type_complexity`.
type TensorSpec<'a> = (&'a str, &'a str, Vec<u64>, Vec<u8>);

/// Build a `.safetensors` file from `(name, dtype, shape, payload)` tuples.
fn build_st(tensors: &[TensorSpec]) -> Vec<u8> {
    let mut obj = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len() as u64;
        data.extend_from_slice(bytes);
        let end = data.len() as u64;
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::from(*dtype));
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(shape.iter().map(|&d| serde_json::Value::from(d)).collect()),
        );
        entry.insert("data_offsets".into(), serde_json::json!([start, end]));
        obj.insert((*name).to_string(), serde_json::Value::Object(entry));
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(obj)).unwrap();
    let header = align_header_to_8(&header);
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out
}

fn bf16_bytes(words: &[u16]) -> Vec<u8> {
    let mut v = Vec::new();
    for &w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

/// A small all-bf16 model: the shape of the real VibeVoice target, in miniature.
///
/// Both 2-D weights are 128x128 so the default int8/block-128 path divides
/// evenly and the run is a genuine `parity: exact` rather than a
/// skip-everything heuristic run.
fn bf16_model_bytes() -> Vec<u8> {
    build_st(&[
        (
            "embed_tokens.weight",
            "BF16",
            vec![128, 128],
            bf16_bytes(&[0x3F80; 128 * 128]),
        ),
        (
            "layers.0.self_attn.q_proj.weight",
            "BF16",
            vec![128, 128],
            bf16_bytes(&[0x4000; 128 * 128]),
        ),
        (
            "layers.0.mlp.down_proj.weight",
            "BF16",
            vec![128, 128],
            bf16_bytes(&[0x3F00; 128 * 128]),
        ),
    ])
}

fn read_header(path: &Path) -> serde_json::Value {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    serde_json::from_slice(&raw[8..8 + n]).expect("header must be valid JSON")
}

/// A plain HuggingFace folder: exactly one `.safetensors` plus real sidecars.
///
/// This is the shape that triggered the reported crash — note there is NO
/// `model.safetensors.index.json`.
fn write_hf_folder(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let file = dir.join("model.safetensors");
    std::fs::write(&file, bf16_model_bytes()).unwrap();
    std::fs::write(dir.join("config.json"), br#"{"model_type":"vibe"}"#).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    file
}

/// A sharded folder: `model.safetensors.index.json` + the shards it maps.
/// Exercises the `run_sharded` / `discover_shards` path, which must not regress.
fn write_sharded_folder(dir: &Path, shards: &[(&str, &[TensorSpec])]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut weight_map = serde_json::Map::new();
    for (shard_name, tensors) in shards {
        for (tname, ..) in *tensors {
            weight_map.insert(
                (*tname).to_string(),
                serde_json::Value::String((*shard_name).into()),
            );
        }
        std::fs::write(dir.join(shard_name), build_st(tensors)).unwrap();
    }
    let index = serde_json::json!({
        "metadata": {"total_size": 12345},
        "weight_map": weight_map,
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("config.json"), br#"{"model_type":"vibe"}"#).unwrap();
}

/// Assert a run succeeded, printing the child's stderr on failure.
///
/// The stderr is included because the bug being pinned is an *error message*
/// regression: without it a failure here just says "exit 101" and you have to
/// go re-run the binary by hand to learn you got `os error 5` again.
fn assert_ok(res: &std::process::Output, what: &str) {
    assert_eq!(
        res.status.code(),
        Some(0),
        "{what} must succeed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr),
    );
}

/// The reported bug's signature: opening the FOLDER instead of the file inside
/// it. Any occurrence in a success-path run means the resolver was bypassed.
fn assert_no_access_denied(res: &std::process::Output, what: &str) {
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr)
    );
    for needle in ["os error 5", "Access is denied", "Отказано в доступе"] {
        assert!(
            !combined.contains(needle),
            "{what} must not hit the directory-open failure ({needle}).\n\
             That means the folder was opened instead of the .safetensors inside it.\n\
             output: {combined}"
        );
    }
}

// --------------------------------------------------------------------------- //
// The regression: a FOLDER with exactly one .safetensors must work
// --------------------------------------------------------------------------- //

/// `quantize <folder>` must succeed — this is the exact reported failure.
///
/// Asserts the output is real (3 tensors land in it), not merely that the
/// process exited 0: a resolver that returned the folder would fail at open
/// time, but one that returned some *other* file would exit 0 with wrong
/// content, so the tensor count is what gives the assertion teeth.
#[test]
fn quantize_accepts_a_folder_holding_one_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("VibeVoice-Realtime-0.5B");
    write_hf_folder(&model);
    let out = tmp.path().join("out.safetensors");

    let res = bin()
        .arg("quantize")
        .arg(&model)
        .arg(&out)
        .arg("--no-progress")
        .output()
        .unwrap();
    assert_ok(&res, "quantize on a single-safetensors FOLDER");
    assert_no_access_denied(&res, "quantize on a folder");

    assert!(out.exists(), "quantize must write {}", out.display());
    let header = read_header(&out);
    // Assert on the WEIGHTS being present by name, never on a total count:
    // quantization legitimately ADDS companion tensors alongside each weight
    // (`<w>_scale`, `<w>.comfy_quant`, `<w>.input_scale`), so a 3-weight input
    // does NOT produce a 3-entry header. Naming the weights is also the stronger
    // check — it proves the resolved file's contents came through.
    for name in [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ] {
        assert!(
            header.get(name).is_some(),
            "{name} must be present in the output; got {:?}",
            header.as_object().unwrap().keys().collect::<Vec<_>>()
        );
    }
}

/// FOLDER input must produce BYTE-IDENTICAL output to FILE input.
///
/// This is the strongest form of the assertion. "It exited 0" only proves the
/// folder was opened successfully; comparing the two shapes proves the resolver
/// picked the *right* file and that quantize saw exactly the same model either
/// way. quantize is byte-deterministic (pinned by the golden suite), so any
/// difference here is a real defect.
#[test]
fn folder_and_file_input_produce_identical_quantize_output() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes = bf16_model_bytes();

    // Same bytes, two shapes: a bare file, and a folder containing it.
    let as_file = tmp.path().join("as_file.safetensors");
    std::fs::write(&as_file, &bytes).unwrap();
    let folder = tmp.path().join("as_folder");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join("pytorch_model.safetensors"), &bytes).unwrap();

    let out_file = tmp.path().join("from_file.safetensors");
    let out_folder = tmp.path().join("from_folder.safetensors");

    for (input, out) in [(&as_file, &out_file), (&folder, &out_folder)] {
        let res = bin()
            .arg("quantize")
            .arg(input)
            .arg(out)
            .arg("--no-progress")
            .output()
            .unwrap();
        assert_ok(&res, &format!("quantize {}", input.display()));
        assert_no_access_denied(&res, &format!("quantize {}", input.display()));
    }

    assert_eq!(
        std::fs::read(&out_file).unwrap(),
        std::fs::read(&out_folder).unwrap(),
        "a folder input must quantize byte-identically to the file inside it"
    );
}

/// `cast <folder>` must succeed. Same bug, same shape, second call site.
#[test]
fn cast_accepts_a_folder_holding_one_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("VibeVoice-Realtime-0.5B");
    write_hf_folder(&model);
    let out = tmp.path().join("out.safetensors");

    let res = bin()
        .arg("cast")
        .arg(&model)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_ok(&res, "cast on a single-safetensors FOLDER");
    assert_no_access_denied(&res, "cast on a folder");

    assert!(out.exists());
    assert_eq!(
        read_header(&out).as_object().unwrap().len(),
        4,
        "3 tensors + __metadata__"
    );
}

/// `info <folder>` must succeed AND name the file it actually read.
///
/// The second half matters: `info` prints a `file :` line describing what it
/// opened. If that line still echoed the folder while the table described the
/// resolved file, the report would be self-contradictory — so the fix routes
/// the printed path through the resolver too, and this pins it.
#[test]
fn info_accepts_a_folder_and_names_the_resolved_file() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("VibeVoice-Realtime-0.5B");
    let inner = write_hf_folder(&model);

    let res = bin().arg("info").arg(&model).output().unwrap();
    assert_ok(&res, "info on a single-safetensors FOLDER");
    assert_no_access_denied(&res, "info on a folder");

    let stdout = String::from_utf8_lossy(&res.stdout);
    assert!(
        stdout.contains(&inner.display().to_string()),
        "info must report the RESOLVED file it read ({}).\nstdout:\n{stdout}",
        inner.display()
    );
    assert!(
        stdout.contains("tensors : 3"),
        "info must summarise the resolved model's tensors.\nstdout:\n{stdout}"
    );
}

/// `info --raw` on a folder emits the resolved file's JSON header.
#[test]
fn info_raw_accepts_a_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("hf");
    write_hf_folder(&model);

    let res = bin().arg("info").arg(&model).arg("--raw").output().unwrap();
    assert_ok(&res, "info --raw on a folder");
    let stdout = String::from_utf8_lossy(&res.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("raw header must be JSON");
    assert!(
        v.get("embed_tokens.weight").is_some(),
        "raw header must be the resolved file's, got {v}"
    );
}

/// `gguf` was the CONTROL that already worked — it resolved the folder inline.
///
/// Now that it calls the shared resolver, prove the behaviour is unchanged, so
/// the refactor did not trade one bug for another. Uses the golden llama-shaped
/// fixture rather than a hand-rolled one so arch detection has a real
/// `config.json` to read.
#[test]
fn gguf_still_accepts_a_folder_holding_one_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m");
    write_hf_folder(&model);
    // gguf needs a real arch to map names; supply the llama config the golden
    // gguf fixture uses.
    std::fs::write(
        model.join("config.json"),
        br#"{"architectures":["LlamaForCausalLM"],"model_type":"llama",
              "hidden_size":128,"num_hidden_layers":1,"vocab_size":128}"#,
    )
    .unwrap();
    let out = tmp.path().join("m-q8_0.gguf");

    let res = bin()
        .args([
            "gguf",
            model.to_str().unwrap(),
            out.to_str().unwrap(),
            "--method",
            "q8_0",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert_ok(&res, "gguf on a single-safetensors FOLDER");
    assert_no_access_denied(&res, "gguf on a folder");
    assert!(out.exists(), "gguf must write {}", out.display());
}

// --------------------------------------------------------------------------- //
// No regression on the two shapes that already worked
// --------------------------------------------------------------------------- //

/// The plain single-FILE path must be untouched by the fix.
///
/// `resolve_single_file` returns `p.to_path_buf()` for a real file, so this is
/// the branch most at risk of being "simplified" into something that re-scans.
#[test]
fn quantize_still_accepts_a_plain_single_file() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("model.safetensors");
    std::fs::write(&src, bf16_model_bytes()).unwrap();
    let out = tmp.path().join("out.safetensors");

    let res = bin()
        .arg("quantize")
        .arg(&src)
        .arg(&out)
        .arg("--no-progress")
        .output()
        .unwrap();
    assert_ok(&res, "quantize on a plain single FILE");
    let header = read_header(&out);
    for name in [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ] {
        assert!(
            header.get(name).is_some(),
            "{name} must survive the plain-file path"
        );
    }
}

/// The SHARDED-folder path (`run_sharded` + `discover_shards` + the index)
/// must be unaffected. This is the path the resolver does NOT touch, so it is
/// the natural place a refactor could break something.
#[test]
fn quantize_still_merges_a_sharded_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sharded");
    write_sharded_folder(
        &model,
        &[
            (
                "model-00001-of-00002.safetensors",
                &[
                    (
                        "embed_tokens.weight",
                        "BF16",
                        vec![128, 128],
                        bf16_bytes(&[0x3F80; 128 * 128]),
                    ),
                    (
                        "layers.0.q.weight",
                        "BF16",
                        vec![128, 128],
                        bf16_bytes(&[0x4000; 128 * 128]),
                    ),
                ],
            ),
            (
                "model-00002-of-00002.safetensors",
                &[(
                    "layers.1.q.weight",
                    "BF16",
                    vec![128, 128],
                    bf16_bytes(&[0x3F00; 128 * 128]),
                )],
            ),
        ],
    );
    let out = tmp.path().join("merged.safetensors");

    let res = bin()
        .arg("quantize")
        .arg(&model)
        .arg(&out)
        // A SHARDED input defaults to writing a sharded output (a DIRECTORY
        // plus an index), which is not what this assertion wants to read. Ask
        // for the single merged file explicitly — the sharded-output mode is
        // covered by cli_phase9.rs, so nothing is lost here.
        .arg("--output-mode")
        .arg("single")
        .arg("--no-progress")
        .output()
        .unwrap();
    assert_ok(&res, "quantize on a SHARDED folder");
    assert!(
        out.is_file(),
        "expected one merged file at {}, got a directory?",
        out.display()
    );

    let header = read_header(&out);
    for name in [
        "embed_tokens.weight",
        "layers.0.q.weight",
        "layers.1.q.weight",
    ] {
        assert!(
            header.get(name).is_some(),
            "{name} must survive the sharded merge; got {:?}",
            header.as_object().unwrap().keys().collect::<Vec<_>>()
        );
    }
}

/// `cast` on a sharded folder must still merge every shard.
#[test]
fn cast_still_merges_a_sharded_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sharded");
    write_sharded_folder(
        &model,
        &[
            (
                "model-00001-of-00002.safetensors",
                &[
                    (
                        "embed_tokens.weight",
                        "BF16",
                        vec![4, 2],
                        bf16_bytes(&[0x3F80; 8]),
                    ),
                    (
                        "layers.0.q.weight",
                        "BF16",
                        vec![2, 2],
                        bf16_bytes(&[0x4000; 4]),
                    ),
                ],
            ),
            (
                "model-00002-of-00002.safetensors",
                &[(
                    "layers.1.q.weight",
                    "BF16",
                    vec![2, 2],
                    bf16_bytes(&[0x3F00; 4]),
                )],
            ),
        ],
    );
    let out = tmp.path().join("merged.safetensors");

    let res = bin()
        .arg("cast")
        .arg(&model)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_ok(&res, "cast on a SHARDED folder");

    let header = read_header(&out);
    for name in [
        "embed_tokens.weight",
        "layers.0.q.weight",
        "layers.1.q.weight",
    ] {
        assert!(header.get(name).is_some(), "{name} must be merged in");
    }
}

// --------------------------------------------------------------------------- //
// Error paths: clean errors, never panics
// --------------------------------------------------------------------------- //

/// A folder with ZERO `.safetensors` is a clean USAGE error (exit 2), not a
/// panic.
///
/// Exit 2 is the right code: `classify_input` returns `(None, None)` because
/// `sts.len() == 0`, so this is rejected before the resolver runs — the input
/// is not a usable model, which is a usage problem. A Rust panic would exit
/// 101, so the code assertion is what rules a panic out.
#[test]
fn folder_with_zero_safetensors_is_a_clean_usage_error() {
    let tmp = tempfile::tempdir().unwrap();
    let empty = tmp.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    std::fs::write(empty.join("config.json"), b"{}").unwrap();

    for cmd in ["quantize", "cast", "gguf"] {
        let res = bin()
            .arg(cmd)
            .arg(&empty)
            .arg(tmp.path().join("out.bin"))
            .output()
            .unwrap();
        assert_eq!(
            res.status.code(),
            Some(2),
            "{cmd} on a folder with 0 .safetensors must be a clean exit 2 (a panic \
             would be 101).\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&res.stdout),
            String::from_utf8_lossy(&res.stderr),
        );
    }
}

/// A folder with TWO `.safetensors` and no index is also a clean usage error.
///
/// Such a folder is only meaningful as a *sharded* input, and sharded needs an
/// index — so without one there is no unambiguous single file to pick. Picking
/// arbitrarily would silently quantize the wrong half of the model.
#[test]
fn folder_with_two_safetensors_and_no_index_is_a_clean_usage_error() {
    let tmp = tempfile::tempdir().unwrap();
    let two = tmp.path().join("two");
    std::fs::create_dir_all(&two).unwrap();
    std::fs::write(two.join("a.safetensors"), bf16_model_bytes()).unwrap();
    std::fs::write(two.join("b.safetensors"), bf16_model_bytes()).unwrap();

    for cmd in ["quantize", "cast", "gguf"] {
        let res = bin()
            .arg(cmd)
            .arg(&two)
            .arg(tmp.path().join("out.bin"))
            .output()
            .unwrap();
        assert_eq!(
            res.status.code(),
            Some(2),
            "{cmd} on a 2-file folder must be a clean exit 2, not an arbitrary pick \
             and not a panic (101).\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&res.stdout),
            String::from_utf8_lossy(&res.stderr),
        );
    }
}

/// `info` on a folder with no `.safetensors` exits 1, not 2 and not a panic.
///
/// `info` deliberately does NOT gate on `classify_input` — it calls the
/// resolver directly — so it reaches the resolver's own `NoSafetensorsInFolder`
/// error, which the command reports as a DATA error (exit 1) like its other
/// open failures. Pinning the code here documents that the two commands really
/// do take different paths, rather than leaving it accidental.
#[test]
fn info_on_a_folder_with_zero_safetensors_exits_one_cleanly() {
    let tmp = tempfile::tempdir().unwrap();
    let empty = tmp.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    std::fs::write(empty.join("config.json"), b"{}").unwrap();

    let res = bin().arg("info").arg(&empty).output().unwrap();
    assert_eq!(
        res.status.code(),
        Some(1),
        "info on an empty folder must exit 1 cleanly.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr),
    );
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(
        stderr.contains("no .safetensors"),
        "the error must say what is wrong, got: {stderr}"
    );
}

/// `info` on a 2-file folder reports the ambiguity instead of picking one.
#[test]
fn info_on_a_folder_with_two_safetensors_exits_one_cleanly() {
    let tmp = tempfile::tempdir().unwrap();
    let two = tmp.path().join("two");
    std::fs::create_dir_all(&two).unwrap();
    std::fs::write(two.join("a.safetensors"), bf16_model_bytes()).unwrap();
    std::fs::write(two.join("b.safetensors"), bf16_model_bytes()).unwrap();

    let res = bin().arg("info").arg(&two).output().unwrap();
    assert_eq!(
        res.status.code(),
        Some(1),
        "info on a 2-file folder must exit 1 cleanly, not guess.\nstderr: {}",
        String::from_utf8_lossy(&res.stderr),
    );
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(
        stderr.contains("expected exactly 1"),
        "the error must name the ambiguity, got: {stderr}"
    );
}

/// A DIRECTORY named `*.safetensors` beside a real one must not confuse the
/// resolver.
///
/// `read_dir` yields both; only the real file is a candidate. Counting the
/// directory would either fail to open it (the original bug, one level down) or
/// make a valid folder look ambiguous.
#[test]
fn a_directory_named_like_a_shard_does_not_break_folder_resolution() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("hf");
    let inner = write_hf_folder(&model);
    std::fs::create_dir_all(model.join("weights.safetensors")).unwrap();
    let out = tmp.path().join("out.safetensors");

    let res = bin()
        .arg("quantize")
        .arg(&model)
        .arg(&out)
        .arg("--no-progress")
        .output()
        .unwrap();
    assert_ok(&res, "quantize where a DIRECTORY is named *.safetensors");
    assert_no_access_denied(&res, "quantize with a decoy directory");
    assert_eq!(
        std::fs::read(&inner).unwrap(),
        bf16_model_bytes(),
        "the decoy directory must not disturb the real file"
    );
    assert!(
        read_header(&out).get("embed_tokens.weight").is_some(),
        "the REAL file inside must be the one that was quantized, not the decoy"
    );
}

/// Resolving a folder must not modify it — the input is read-only.
///
/// The resolver does a directory scan, which is a read; this guards against a
/// future "helpful" fix that writes (e.g. materialises a symlink or an index)
/// into the user's model directory.
#[test]
fn quantize_on_a_folder_leaves_the_source_untouched() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("hf");
    write_hf_folder(&model);
    let out = tmp.path().join("out.safetensors");

    let snapshot = |dir: &Path| -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| {
                (
                    e.file_name().to_string_lossy().into_owned(),
                    std::fs::read(e.path()).unwrap(),
                )
            })
            .collect();
        v.sort();
        v
    };
    let before = snapshot(&model);

    let res = bin()
        .arg("quantize")
        .arg(&model)
        .arg(&out)
        .arg("--no-progress")
        .output()
        .unwrap();
    assert_ok(&res, "quantize on a folder");

    assert_eq!(
        before,
        snapshot(&model),
        "the source folder's file set and bytes must be unchanged"
    );
}

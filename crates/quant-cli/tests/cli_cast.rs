//! `quantui-rs cast` CLI integration tests.
//!
//! These drive the real binary, so they cover the parts a unit test cannot:
//! argument parsing, exit codes, the temp-file/rename publish discipline, the
//! `cast:` report line, and the promise that the SOURCE is never modified.
//!
//! Exit-code contract exercised here: 0 ok, 1 data error, 2 usage error.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_quantui-rs"))
}

// --------------------------------------------------------------------------- //
// Fixture builder (hand-rolled; mirrors the one in cli_validate.rs)
// --------------------------------------------------------------------------- //

fn align_header_to_8(header: &[u8]) -> Vec<u8> {
    let rem = header.len() % 8;
    if rem == 0 {
        return header.to_vec();
    }
    let mut out = header.to_vec();
    out.extend(std::iter::repeat_n(b' ', 8 - rem));
    out
}

/// Build a `.safetensors` file from `(name, dtype, shape, payload)` tuples.
fn build_st(tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
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
            serde_json::Value::Array(
                shape
                    .iter()
                    .map(|&d| serde_json::Value::from(d))
                    .collect(),
            ),
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

fn f32_bytes(words: &[u32]) -> Vec<u8> {
    let mut v = Vec::new();
    for &w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

/// A small all-bf16 model: the shape of the real VibeVoice target, in miniature.
fn write_bf16_model(path: &Path) {
    let bytes = build_st(&[
        ("embed_tokens.weight", "BF16", vec![8, 4], bf16_bytes(&[0x3F80; 32])),
        ("layers.0.self_attn.q_proj.weight", "BF16", vec![4, 4], bf16_bytes(&[0x4000; 16])),
        ("layers.0.mlp.down_proj.weight", "BF16", vec![4, 4], bf16_bytes(&[0x3F00; 16])),
    ]);
    std::fs::write(path, bytes).unwrap();
}

/// A sharded folder: `model.safetensors.index.json` + two shard files.
///
/// `extra_index` is merged into the weight map so a test can point a tensor at
/// a shard that does not exist (to prove discovery is really consulted).
fn write_sharded_folder(dir: &Path, shards: &[(&str, &[(&str, &str, Vec<u64>, Vec<u8>)])]) {
    let mut weight_map = serde_json::Map::new();
    for (shard_name, tensors) in shards {
        for (tname, _, _, _) in *tensors {
            weight_map.insert((*tname).to_string(), serde_json::Value::String((*shard_name).into()));
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
    // A sidecar file that must survive untouched.
    std::fs::write(dir.join("config.json"), br#"{"model_type":"vibe"}"#).unwrap();
}

/// Read a safetensors header as JSON.
fn read_header(path: &Path) -> serde_json::Value {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    serde_json::from_slice(&raw[8..8 + n]).expect("header must be valid JSON")
}

/// Read one tensor's payload bytes from a safetensors file.
fn read_tensor(path: &Path, name: &str) -> Vec<u8> {
    let raw = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let e = &header[name];
    let [s, e] = e["data_offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>()[..]
    else {
        panic!("bad data_offsets for {name}")
    };
    let base = 8 + n;
    raw[base + s..base + e].to_vec()
}

fn tmp_dir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

// --------------------------------------------------------------------------- //
// Happy paths
// --------------------------------------------------------------------------- //

/// A sharded folder is MERGED into one output file with every tensor present.
///
/// This is the real target shape: VibeVoice is 3 shards / 1204 tensors, and the
/// whole point is one loadable file.
#[test]
fn cast_merges_sharded_folder_to_single_file() {
    let tmp = tmp_dir();
    let model = tmp.path().join("VibeVoice-1.5B");
    std::fs::create_dir(&model).unwrap();
    write_sharded_folder(
        &model,
        &[
            (
                "model-00001-of-00002.safetensors",
                &[
                    ("embed_tokens.weight", "BF16", vec![4, 2], bf16_bytes(&[0x3F80; 8])),
                    ("layers.0.q.weight", "BF16", vec![2, 2], bf16_bytes(&[0x4000; 4])),
                ],
            ),
            (
                "model-00002-of-00002.safetensors",
                &[("layers.1.q.weight", "BF16", vec![2, 2], bf16_bytes(&[0x3F00; 4]))],
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
    assert_eq!(
        res.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&res.stderr)
    );

    // One file, every tensor from BOTH shards.
    let header = read_header(&out);
    for name in ["embed_tokens.weight", "layers.0.q.weight", "layers.1.q.weight"] {
        assert!(
            header.get(name).is_some(),
            "{name} must be present in the merged output"
        );
    }
    // Shapes preserved.
    assert_eq!(header["embed_tokens.weight"]["shape"], serde_json::json!([4, 2]));
    assert_eq!(header["layers.1.q.weight"]["shape"], serde_json::json!([2, 2]));

    // A same-dtype bf16->bf16 merge is a VERBATIM copy: payloads identical.
    assert_eq!(
        read_tensor(&out, "layers.0.q.weight"),
        read_tensor(&model.join("model-00001-of-00002.safetensors"), "layers.0.q.weight"),
        "same-dtype merge must copy payload bytes verbatim"
    );
    assert_eq!(
        read_tensor(&out, "layers.1.q.weight"),
        read_tensor(&model.join("model-00002-of-00002.safetensors"), "layers.1.q.weight"),
        "payload from the second shard must survive verbatim"
    );
}

/// A single-file input casts and keeps its tensors.
#[test]
fn cast_single_file_input_roundtrips() {
    let tmp = tmp_dir();
    let src = tmp.path().join("model.safetensors");
    write_bf16_model(&src);
    let out = tmp.path().join("out.safetensors");

    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&res.stderr));

    let header = read_header(&out);
    assert_eq!(header.as_object().unwrap().len(), 4, "3 tensors + __metadata__");
    assert_eq!(
        read_tensor(&out, "layers.0.mlp.down_proj.weight"),
        read_tensor(&src, "layers.0.mlp.down_proj.weight"),
        "bf16 -> bf16 must be byte-identical"
    );
}

/// Two runs of the same input produce byte-identical output.
///
/// Byte-determinism matters because the output is a model file: a run that
/// varies run-to-run cannot be checksummed or cached.
#[test]
fn cast_is_deterministic_across_two_runs() {
    let tmp = tmp_dir();
    let src = tmp.path().join("model.safetensors");
    write_bf16_model(&src);
    let a = tmp.path().join("a.safetensors");
    let b = tmp.path().join("b.safetensors");

    for out in [&a, &b] {
        let res = bin()
            .arg("cast")
            .arg(&src)
            .arg(out)
            .arg("--to")
            .arg("f32")
            .output()
            .unwrap();
        assert_eq!(res.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&res.stderr));
    }
    assert_eq!(
        std::fs::read(&a).unwrap(),
        std::fs::read(&b).unwrap(),
        "two identical runs must produce byte-identical files"
    );
}

/// The output header carries the dtype that was actually requested.
#[test]
fn cast_output_dtype_matches_request() {
    let tmp = tmp_dir();
    let src = tmp.path().join("model.safetensors");
    std::fs::write(
        &src,
        build_st(&[("w", "F32", vec![4], f32_bytes(&[0x3F800000, 0x40000000, 0x40400000, 0x40800000]))]),
    )
    .unwrap();

    for (flag, want) in [("bf16", "BF16"), ("f16", "F16"), ("f32", "F32")] {
        let out = tmp.path().join(format!("out-{flag}.safetensors"));
        let res = bin()
            .arg("cast")
            .arg(&src)
            .arg(&out)
            .arg("--to")
            .arg(flag)
            .output()
            .unwrap();
        assert_eq!(res.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&res.stderr));
        let header = read_header(&out);
        assert_eq!(
            header["w"]["dtype"],
            serde_json::Value::String(want.into()),
            "--to {flag} must write dtype {want}"
        );
        assert_eq!(header["w"]["shape"], serde_json::json!([4]), "shape preserved");
    }
}

/// Ambiguous and non-float dtypes keep their ORIGINAL header spelling.
///
/// `U16` is the interesting case and is asserted as PRESERVED, not normalised.
/// `DType::from_header_str` maps `"U16" -> DType::U16`, a DISTINCT variant from
/// `DType::Bf16`, deliberately: a `"U16"` header is ambiguous. `safetensors.numpy`
/// writes bf16 tensors under `U16` because numpy has no native bf16, but nothing
/// in the header distinguishes that from a genuine `uint16` tensor. Guessing
/// bf16 would silently corrupt a real uint16 weight — the same class of damage
/// as emitting `Inf` where a number was expected, which this whole feature
/// refuses to do on principle. A cast must not reinterpret an ambiguous dtype,
/// so preservation is the contract.
#[test]
fn cast_preserves_ambiguous_and_non_float_header_spelling() {
    let tmp = tmp_dir();
    let src = tmp.path().join("model.safetensors");
    std::fs::write(
        &src,
        build_st(&[
            ("w.u16", "U16", vec![2], bf16_bytes(&[0x3F80, 0x4000])),
            ("w.i64", "I64", vec![2], vec![0u8; 16]),
            ("w.bool", "BOOL", vec![2], vec![1u8, 0u8]),
        ]),
    )
    .unwrap();
    let out = tmp.path().join("out.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&res.stderr));
    let header = read_header(&out);
    // The ambiguous U16 tensor is passed through untouched, spelling intact.
    assert_eq!(
        header["w.u16"]["dtype"],
        serde_json::Value::String("U16".into()),
        "an ambiguous U16 header must NOT be reinterpreted as BF16"
    );
    // Its payload must survive byte-for-byte too.
    assert_eq!(
        read_tensor(&out, "w.u16"),
        bf16_bytes(&[0x3F80, 0x4000]),
        "a passed-through U16 tensor must keep its exact bytes"
    );
    // The other non-float tensors keep theirs.
    assert_eq!(header["w.i64"]["dtype"], serde_json::Value::String("I64".into()));
    assert_eq!(header["w.bool"]["dtype"], serde_json::Value::String("BOOL".into()));
}

// --------------------------------------------------------------------------- //
// The highest-severity invariant
// --------------------------------------------------------------------------- //

/// The output must carry NO quantization metadata, only `{"format":"pt"}`.
///
/// A bf16 file advertising itself as ComfyUI-quantized loads as a BROKEN model:
/// the loader expects scales that are not there. This is the single most
/// important invariant of the command.
#[test]
fn cast_writes_no_quantization_metadata() {
    let tmp = tmp_dir();
    let src = tmp.path().join("model.safetensors");
    write_bf16_model(&src);
    let out = tmp.path().join("out.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&res.stderr));

    let header = read_header(&out);
    let meta = header["__metadata__"].as_object().expect("__metadata__ present");
    assert_eq!(
        meta.len(),
        1,
        "metadata must hold exactly {{format: pt}}, got {:?}",
        meta.keys().collect::<Vec<_>>()
    );
    assert_eq!(meta["format"], serde_json::Value::String("pt".into()));

    // No tensor name and no metadata key may mention quantization.
    let raw = String::from_utf8_lossy(&read_header_bytes(&out)).into_owned();
    for forbidden in [".comfy_quant", "weight_scale", "_quantization_metadata", "comfy_quant"] {
        assert!(
            !raw.contains(forbidden),
            "output must not mention {forbidden:?}: {raw}"
        );
    }
    // And no tensor may be named like a scale.
    for key in header.as_object().unwrap().keys() {
        assert!(
            !key.contains("scale") && !key.contains("quant"),
            "unexpected scale/quant tensor {key:?}"
        );
    }
}

fn read_header_bytes(path: &Path) -> Vec<u8> {
    let raw = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    raw[8..8 + n].to_vec()
}

/// The `cast:` summary must not be a `parity:` line.
///
/// `parity:` is a load-bearing marker in this repo: it means
/// `Quality::is_parity_exact()` on a `quantize` run, i.e. either `exact` or
/// `quality-tuned`. A cast is NEITHER — f32→bf16 is not reversible, so claiming
/// byte-exact parity would be false. Printing `parity:` here would dilute a
/// marker that is meaningful everywhere else.
#[test]
fn cast_emits_cast_summary_not_parity_line() {
    let tmp = tmp_dir();
    let src = tmp.path().join("model.safetensors");
    write_bf16_model(&src);
    let out = tmp.path().join("out.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(0));

    let stdout = String::from_utf8_lossy(&res.stdout);
    assert!(
        !stdout.contains("parity:"),
        "cast must NOT print a parity: line, got: {stdout}"
    );
    assert!(
        stdout.contains("cast:"),
        "cast must print a cast: summary, got: {stdout}"
    );
    // A pure same-dtype run is reported as lossless, and — like every other
    // branch — names the dtypes it dealt with. This is the common run for the
    // all-bf16 target model, so it is the one that must not be vague.
    assert!(
        stdout.contains("lossless"),
        "a same-dtype run must report the verbatim path, got: {stdout}"
    );
    assert!(
        stdout.contains("BF16 -> BF16"),
        "every report branch must name the source and target dtypes, got: {stdout}"
    );
    assert!(
        stdout.contains("verbatim copy"),
        "the lossless branch must still say the copy was verbatim, got: {stdout}"
    );
}

/// A genuine f32 -> bf16 run reports a conversion, and names both dtypes.
#[test]
fn cast_reports_conversion_with_real_dtype_names() {
    let tmp = tmp_dir();
    let src = tmp.path().join("m.safetensors");
    std::fs::write(
        &src,
        build_st(&[("w", "F32", vec![2], f32_bytes(&[0x3F800000, 0x40000000]))]),
    )
    .unwrap();
    let out = tmp.path().join("o.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&res.stdout);
    assert!(stdout.contains("cast:"), "{stdout}");
    assert!(stdout.contains("F32 -> BF16"), "must name both dtypes: {stdout}");
    assert!(stdout.contains("converted"), "must say it converted: {stdout}");
    assert!(!stdout.contains("parity:"), "{stdout}");
}

// --------------------------------------------------------------------------- //
// Overflow refusal — and the publish discipline
// --------------------------------------------------------------------------- //

/// bf16 -> f16 above the f16 range must exit 1 and name the tensor.
///
/// The fixture is built so the guard is genuinely REACHED: the same file cast
/// to `--to f32` succeeds and yields real converted tensors, which proves the
/// file is well-formed and the failure is the overflow and nothing else. (A
/// broken fixture would exit 1 for the wrong reason and this test would prove
/// nothing.)
#[test]
fn cast_refuses_bf16_to_f16_overflow_naming_the_tensor() {
    let tmp = tmp_dir();
    let src = tmp.path().join("big.safetensors");
    // 0x7E1C as bf16 == 5.18e37, far above the f16 max of 65504.
    std::fs::write(
        &src,
        build_st(&[
            ("a.small", "BF16", vec![2], bf16_bytes(&[0x3F80, 0x4000])),
            ("a.huge", "BF16", vec![2], bf16_bytes(&[0x7E1C, 0x3F80])),
        ]),
    )
    .unwrap();

    // Control: the fixture is valid — casting it to f32 (a widening, always
    // legal) succeeds. So an exit 1 below is the overflow guard, not a
    // malformed input.
    let ok_out = tmp.path().join("ok.safetensors");
    let ok = bin()
        .arg("cast")
        .arg(&src)
        .arg(&ok_out)
        .arg("--to")
        .arg("f32")
        .output()
        .unwrap();
    assert_eq!(
        ok.status.code(),
        Some(0),
        "fixture must be castable to f32, else the overflow test is vacuous: {}",
        String::from_utf8_lossy(&ok.stderr)
    );
    assert!(read_header(&ok_out).get("a.huge").is_some());

    // Now the real refusal.
    let bad_out = tmp.path().join("bad.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&bad_out)
        .arg("--to")
        .arg("f16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(1), "overflow is a DATA error, exit 1");
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(
        stderr.contains("a.huge"),
        "the message must name the offending tensor, got: {stderr}"
    );
    assert!(
        stderr.contains("element"),
        "the message must name the element index, got: {stderr}"
    );
    assert!(
        stderr.contains("Inf"),
        "the message must explain what it avoided, got: {stderr}"
    );
}

/// A refused cast must leave NO output file at all.
///
/// This is the regression test for the half-written-file defect:
/// `IncrementalWriter::open_new_with` creates the destination immediately, so a
/// failure used to leave a `.safetensors` whose header parsed as valid JSON and
/// contained ZERO tensors — a file a loader would happily open and then find
/// empty.
///
/// The command now writes to a temp path and renames on success, so a refusal
/// must leave the destination non-existent.
#[test]
fn refused_cast_leaves_no_output_file() {
    let tmp = tmp_dir();
    let src = tmp.path().join("big.safetensors");
    std::fs::write(
        &src,
        build_st(&[("a.huge", "BF16", vec![2], bf16_bytes(&[0x7E1C, 0x3F80]))]),
    )
    .unwrap();

    // Explicit output path.
    let explicit = tmp.path().join("explicit.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&explicit)
        .arg("--to")
        .arg("f16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(1), "must refuse");
    assert!(
        !explicit.exists(),
        "a refused cast must NOT leave an output file at {}",
        explicit.display()
    );

    // Default output path (derived beside the input) — same guarantee.
    let derived = tmp.path().join("big-f16.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg("--to")
        .arg("f16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(1), "must refuse");
    assert!(
        !derived.exists(),
        "a refused cast must not create the DERIVED output path either"
    );

    // And no temp file may be left lying around either.
    let leftovers: Vec<String> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("cast-tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temp files must be cleaned up on failure, found: {leftovers:?}"
    );
}

/// A successful cast must not leave a temp file behind either.
#[test]
fn successful_cast_leaves_no_temp_file() {
    let tmp = tmp_dir();
    let src = tmp.path().join("m.safetensors");
    write_bf16_model(&src);
    let out = tmp.path().join("out.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(0));
    let leftovers: Vec<String> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("cast-tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temp files left after success: {leftovers:?}");
    assert!(out.exists(), "the real output must exist after success");
}

// --------------------------------------------------------------------------- //
// Usage errors and the read-only guarantee
// --------------------------------------------------------------------------- //

/// An unknown `--to` value is a USAGE error: exit 2.
#[test]
fn cast_rejects_unknown_to_value() {
    let tmp = tmp_dir();
    let src = tmp.path().join("m.safetensors");
    write_bf16_model(&src);
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(tmp.path().join("o.safetensors"))
        .arg("--to")
        .arg("float64")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(2), "unknown --to is a usage error");
}

/// A missing input path is a USAGE error: exit 2, not 1.
#[test]
fn cast_rejects_missing_input() {
    let tmp = tmp_dir();
    let res = bin()
        .arg("cast")
        .arg(tmp.path().join("nope.safetensors"))
        .arg(tmp.path().join("o.safetensors"))
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(2), "missing input is a usage error");
}

/// A path that is neither a `.safetensors` nor a sharded folder is exit 2.
#[test]
fn cast_rejects_non_safetensors_input() {
    let tmp = tmp_dir();
    let junk = tmp.path().join("notes.txt");
    std::fs::write(&junk, b"hello").unwrap();
    let res = bin()
        .arg("cast")
        .arg(&junk)
        .arg(tmp.path().join("o.safetensors"))
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(2), "non-safetensors input is a usage error");
}

/// An `f64` source is a DATA error (exit 1), not a usage error.
#[test]
fn cast_rejects_f64_source_with_exit_1() {
    let tmp = tmp_dir();
    let src = tmp.path().join("f64.safetensors");
    let mut payload = Vec::new();
    for v in [1.0f64, 2.0, 3.0, 4.0] {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&src, build_st(&[("w", "F64", vec![4], payload)])).unwrap();
    let out = tmp.path().join("o.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&src)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(1), "f64 source is a DATA error");
    assert!(!out.exists(), "a refused f64 cast must leave no output");
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(stderr.contains("double rounding"), "must explain why: {stderr}");
}

/// `cast` only READS its input: shards, index, and sidecars must be untouched.
#[test]
fn cast_leaves_source_folder_untouched() {
    let tmp = tmp_dir();
    let model = tmp.path().join("M");
    std::fs::create_dir(&model).unwrap();
    write_sharded_folder(
        &model,
        &[(
            "model-00001-of-00001.safetensors",
            &[("w", "BF16", vec![2, 2], bf16_bytes(&[0x3F80; 4]))],
        )],
    );

    // Snapshot every file: name -> (size, content).
    let before: Vec<(String, u64, Vec<u8>)> = std::fs::read_dir(&model)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| {
            let p = e.path();
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                p.metadata().unwrap().len(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    assert!(before.iter().any(|(n, _, _)| n == "model.safetensors.index.json"));
    assert!(before.iter().any(|(n, _, _)| n == "config.json"));

    let out = tmp.path().join("out.safetensors");
    let res = bin()
        .arg("cast")
        .arg(&model)
        .arg(&out)
        .arg("--to")
        .arg("bf16")
        .output()
        .unwrap();
    assert_eq!(res.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&res.stderr));

    // Same set of files, same sizes, same bytes.
    let after: Vec<(String, u64, Vec<u8>)> = std::fs::read_dir(&model)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| {
            let p = e.path();
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                p.metadata().unwrap().len(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        before.iter().map(|(n, s, _)| (n.clone(), *s)).collect::<Vec<_>>(),
        after.iter().map(|(n, s, _)| (n.clone(), *s)).collect::<Vec<_>>(),
        "the source folder's file set and sizes must be unchanged"
    );
    for (name, _, want) in &before {
        let got = after.iter().find(|(n, _, _)| n == name).unwrap();
        assert_eq!(&got.2, want, "{name} must be byte-identical afterwards");
    }
}

/// The default output name follows `<base>-<tag>.safetensors` beside the input.
#[test]
fn cast_derives_default_output_name() {
    let tmp = tmp_dir();
    let model = tmp.path().join("VibeVoice-1.5B");
    std::fs::create_dir(&model).unwrap();
    write_sharded_folder(
        &model,
        &[(
            "model-00001-of-00001.safetensors",
            &[("w", "BF16", vec![2], bf16_bytes(&[0x3F80; 2]))],
        )],
    );
    let res = bin().arg("cast").arg(&model).arg("--to").arg("bf16").output().unwrap();
    assert_eq!(res.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&res.stderr));
    let expected = tmp.path().join("VibeVoice-1.5B-bf16.safetensors");
    assert!(expected.exists(), "expected {}", expected.display());
}

//! T2 — `validate --comfy` encodes the CONSUMER's contract, not the encoder's.
//!
//! # The asymmetry this file exists to pin
//!
//! `validate_comfy_quant` checks the *reference encoder's* notion of a
//! well-formed blob (it requires `orig_dtype`, among others). ComfyUI's loader
//! is a looser and DIFFERENT contract. Being stricter than the consumer is a
//! bug in both directions:
//!
//! * it rejects files ComfyUI loads perfectly well, and
//! * it passes files ComfyUI cannot load at all.
//!
//! The first version of `comfy_loader_contract` made exactly that first
//! mistake — it used `comfy_schema::parse_blob`, which requires `orig_dtype`,
//! and consequently reported ComfyUI's OWN reference file as unloadable. The
//! test `comfy_reference_own_output_is_contract_clean` below is the regression
//! guard for that mistake, and it is the most important assertion in the file:
//! **the contract checker must pass ComfyUI's own output.** A checker that
//! rejects the reference is worse than no checker.
//!
//! Fixtures are hand-built with `serde_json` rather than the reference
//! encoders, because these cases (an unknown format string, a missing scale, a
//! non-integer group size) are exactly the states the encoders cannot produce —
//! that is the whole point of testing a *contract* rather than an encoder.

mod common;

use common::write_input;
use quant_core::comfy_loader_contract::{check_comfy_loadable, is_comfy_loadable, ContractIssue};
use quant_core::dtype::DType;
use quant_core::st_io::IncrementalWriter;
use tempfile::TempDir;

/// Build a single-layer safetensors file from raw parts so tests can express
/// states the real encoder would never emit.
fn write_raw(
    path: &std::path::Path,
    weight_shape: Vec<u64>,
    weight_dtype: DType,
    weight_bytes: Vec<u8>,
    blob: Option<&str>,
    scale: Option<(DType, Vec<u64>, Vec<u8>)>,
) {
    let mut w = IncrementalWriter::open_new(path).expect("open");
    if let Some(b) = blob {
        // The blob is a U8 tensor holding one byte per JSON character, so its
        // shape is the byte LENGTH (U8 itemsize == 1).
        let bytes = b.as_bytes().to_vec();
        w.add_tensor(
            "lin.comfy_quant",
            DType::U8,
            None,
            &[bytes.len() as u64],
            &bytes,
        )
        .expect("blob");
    }
    w.add_tensor(
        "lin.weight",
        weight_dtype,
        None,
        &weight_shape,
        &weight_bytes,
    )
    .expect("weight");
    if let Some((dt, shp, bytes)) = scale {
        w.add_tensor("lin.weight_scale", dt, None, &shp, &bytes)
            .expect("scale");
    }
    w.finalize().expect("finalize");
}

/// `count` F32 values, all equal to `v`. Shape and byte length must agree or
/// the safetensors writer rejects the file (it validates
/// `prod(shape) * itemsize == len`).
fn f32_repeated(v: f32, count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count * 4);
    for _ in 0..count {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn blob_of(json: &str) -> Vec<u8> {
    json.as_bytes().to_vec()
}

/// Every fatal message, joined — for substring assertions.
fn fatals(path: &std::path::Path) -> String {
    check_comfy_loadable(path)
        .iter()
        .filter_map(|i| match i {
            ContractIssue::Fatal(m) => Some(m.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// --------------------------------------------------------------------------
// The load-bearing regression guard
// --------------------------------------------------------------------------

/// ComfyUI's own blob for an int8_tensorwise+convrot layer is
/// `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`
/// — note NO `orig_dtype`. All 229 blobs in the real reference file are
/// exactly this shape.
///
/// A checker built on `comfy_schema::parse_blob` rejects this, because
/// `parse_blob` implements the reference ENCODER's contract where
/// `orig_dtype` is mandatory. The first draft of this module did exactly that
/// and reported the real ComfyUI file as unloadable. This test is why that bug
/// is now impossible to reintroduce silently.
#[test]
fn comfy_reference_own_blob_shape_passes() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("ref.safetensors");
    write_raw(
        &p,
        vec![64, 2048],
        DType::I8,
        vec![0u8; 64 * 2048],
        Some(r#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#),
        Some((DType::F32, vec![64, 1], f32_repeated(0.5, 64))),
    );
    let issues = check_comfy_loadable(&p);
    assert!(
        is_comfy_loadable(&issues),
        "ComfyUI's OWN blob shape must be contract-clean. fatals:\n{}",
        fatals(&p)
    );
}

/// A file with NO `__metadata__` at all is fine. `comfy/utils.py:156` returns
/// `header.get("__metadata__", {})` and `:130` skips the key, so metadata is
/// optional — and `yue2_format` (present on every ComfyUI-produced file) is
/// never read by any ComfyUI code. Pinned so nobody "fixes" this by demanding
/// metadata.
#[test]
fn absent_metadata_is_not_a_violation() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("nometa.safetensors");
    write_raw(
        &p,
        vec![128, 256],
        DType::I8,
        vec![0u8; 128 * 256],
        Some(r#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#),
        Some((DType::F32, vec![128, 1], f32_repeated(0.25, 128))),
    );
    // write_input/IncrementalWriter emits no __metadata__ for this file, so
    // the mere absence is the test.
    let meta = common::read_metadata(&p);
    assert!(meta.is_none(), "fixture must carry no __metadata__");
    let issues = check_comfy_loadable(&p);
    assert!(
        is_comfy_loadable(&issues),
        "a file without __metadata__ must still be contract-clean. fatals:\n{}",
        fatals(&p)
    );
}

// --------------------------------------------------------------------------
// Check 2 — the format string must be a QUANT_ALGOS key
// --------------------------------------------------------------------------

/// `int8_blockwise` is one of OUR format strings (`comfy_schema::INT8_BLOCKWISE`)
/// but is NOT a key in ComfyUI's `QUANT_ALGOS`. The load path does
/// `QUANT_ALGOS[module.quant_format]` with no `.get()` and no normalization
/// (comfy/ops.py:1206), so such a file raises `KeyError` and simply will not
/// open. This is a real, currently-unflagged incompatibility.
#[test]
fn format_string_outside_comfy_algos_is_fatal() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("blockwise.safetensors");
    write_raw(
        &p,
        vec![128, 256],
        DType::I8,
        vec![0u8; 128 * 256],
        Some(r#"{"format": "int8_blockwise", "orig_dtype": "torch.bfloat16", "group_size": 128}"#),
        Some((DType::F32, vec![1, 1], f32_repeated(0.5, 1))),
    );
    let f = fatals(&p);
    // Assert on the SPECIFIC message, not merely on the substring
    // "QUANT_ALGOS": the payload-dtype diagnostic also mentions QUANT_ALGOS
    // (it cites `QUANT_ALGOS[format]['storage_t']`), so a loose match would let
    // this test pass for entirely the wrong reason — which it did, until a
    // mutation that disabled the membership check stayed green.
    assert!(
        f.contains("is not a key in ComfyUI's QUANT_ALGOS"),
        "an unknown-to-ComfyUI format string must be reported as a missing \
         QUANT_ALGOS key specifically; got:\n{f}"
    );
    assert!(
        !f.contains("cannot back"),
        "the failure must be the format-string check, not the dtype check; got:\n{f}"
    );
    assert!(!is_comfy_loadable(&check_comfy_loadable(&p)));
}

/// A missing `format` key is `None` -> ComfyUI's explicit ValueError
/// (ops.py:1203-1204).
#[test]
fn missing_format_key_is_fatal() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("noformat.safetensors");
    write_raw(
        &p,
        vec![64, 256],
        DType::I8,
        vec![0u8; 64 * 256],
        Some(r#"{"convrot": true, "convrot_groupsize": 256}"#),
        Some((DType::F32, vec![64, 1], f32_repeated(0.5, 64))),
    );
    let f = fatals(&p);
    assert!(
        f.contains("no string \"format\""),
        "a blob without `format` must be fatal; got:\n{f}"
    );
}

// --------------------------------------------------------------------------
// Check 3 — required sibling scales
// --------------------------------------------------------------------------

/// `int8_tensorwise` pops `weight_scale` unconditionally
/// (comfy/ops.py:1224-1228) and raises when it is absent.
#[test]
fn missing_weight_scale_is_fatal() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("noscale.safetensors");
    write_raw(
        &p,
        vec![64, 256],
        DType::I8,
        vec![0u8; 64 * 256],
        Some(r#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#),
        None, // no scale
    );
    let f = fatals(&p);
    assert!(
        f.contains("lin.weight_scale"),
        "int8_tensorwise without weight_scale must be fatal; got:\n{f}"
    );
}

/// NVFP4 needs BOTH `weight_scale` and `weight_scale_2`
/// (comfy/ops.py:1218-1223). Supplying only one must be fatal.
#[test]
fn nvfp4_missing_second_scale_is_fatal() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("nvfp4.safetensors");
    write_raw(
        &p,
        vec![64, 256],
        DType::U8,
        vec![0u8; 64 * 256],
        Some(r#"{"format": "nvfp4", "group_size": 16}"#),
        Some((DType::U8, vec![64, 16], vec![0u8; 64 * 16])),
    );
    let f = fatals(&p);
    assert!(
        f.contains("weight_scale_2"),
        "nvfp4 without weight_scale_2 must be fatal; got:\n{f}"
    );
}

// --------------------------------------------------------------------------
// Check 4 — convrot_groupsize must be an integer
// --------------------------------------------------------------------------

/// ComfyUI calls `int(...)` on this value (comfy/ops.py:1234-1236).
#[test]
fn non_integer_convrot_groupsize_is_fatal() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("badgs.safetensors");
    write_raw(
        &p,
        vec![64, 256],
        DType::I8,
        vec![0u8; 64 * 256],
        Some(r#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": "256"}"#),
        Some((DType::F32, vec![64, 1], f32_repeated(0.5, 64))),
    );
    let f = fatals(&p);
    assert!(
        f.contains("convrot_groupsize"),
        "a string convrot_groupsize must be fatal; got:\n{f}"
    );
}

// --------------------------------------------------------------------------
// Check 5 — payload dtype must match storage_t
// --------------------------------------------------------------------------

/// `int8_tensorwise` has `storage_t = torch.int8` (comfy/quant_ops.py:238-243).
/// A float payload cannot back it.
#[test]
fn wrong_payload_dtype_is_fatal() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("wrongdtype.safetensors");
    write_raw(
        &p,
        vec![64, 256],
        DType::F32, // should be I8
        vec![0u8; 64 * 256 * 4],
        Some(r#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#),
        Some((DType::F32, vec![64, 1], f32_repeated(0.5, 64))),
    );
    let f = fatals(&p);
    assert!(
        f.contains("cannot back"),
        "a float payload for int8_tensorwise must be fatal; got:\n{f}"
    );
}

// --------------------------------------------------------------------------
// Check 6 — a blob with no weight tensor
// --------------------------------------------------------------------------

#[test]
fn blob_without_weight_is_fatal() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("noweight.safetensors");
    // Write only the blob; no .weight.
    let mut w = IncrementalWriter::open_new(&p).expect("open");
    let b = blob_of(r#"{"format": "int8_tensorwise"}"#);
    w.add_tensor("orphan.comfy_quant", DType::U8, None, &[b.len() as u64], &b)
        .expect("blob");
    w.finalize().expect("finalize");

    let f = fatals(&p);
    assert!(
        f.contains("no .weight tensor"),
        "a .comfy_quant with no .weight must be fatal; got:\n{f}"
    );
}

// --------------------------------------------------------------------------
// Non-quantized files are not this check's business
// --------------------------------------------------------------------------

/// A plain unquantized model has no blobs and no contract obligations. It must
/// NOT be reported as a violation — otherwise the flag would be useless on a
/// mixed file.
#[test]
fn plain_file_is_not_a_violation() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("plain.safetensors");
    write_input(&p, &[("w.weight", DType::F32, vec![4, 4], vec![0u8; 64])]);
    assert!(is_comfy_loadable(&check_comfy_loadable(&p)));
}

// --------------------------------------------------------------------------
// Determinism
// --------------------------------------------------------------------------

/// The report is a pure function of the bytes, so repeated runs must agree
/// exactly — otherwise it is useless in a diff.
#[test]
fn report_is_deterministic() {
    let dir = TempDir::new().unwrap();
    let p = dir.path().join("det.safetensors");
    write_raw(
        &p,
        vec![64, 256],
        DType::I8,
        vec![0u8; 64 * 256],
        Some(r#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#),
        Some((DType::F32, vec![64, 1], f32_repeated(0.5, 64))),
    );
    let a = check_comfy_loadable(&p);
    let b = check_comfy_loadable(&p);
    assert_eq!(a, b, "contract report must be byte-stable across runs");
}

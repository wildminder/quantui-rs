//! T1 — the ConvRot efficiency skip is a COLUMN-axis predicate.
//!
//! # The bug this pins
//!
//! `is_quantizable` used one predicate for every format:
//! `should_skip_shape(shape, heur_block_size(config))`, which demands **both**
//! dimensions be `>= block_size` and divisible by it. That is a **block-mode**
//! requirement. ConvRot is row mode — one scale per row, reduced over `n`
//! elements, no block grid — so constraining the row count is simply wrong.
//!
//! Measured on YuE2-3B against the ComfyUI reference (229 quantized tensors):
//! the both-dims predicate skipped `llm2vae.weight [64, 2048]`, the only tensor
//! in the model with `m < 128`, and one the reference DOES quantize. The
//! column-only predicate misses 0 of 229 and still skips `vae2llm [2048, 64]`
//! (its `n = 64`) exactly as the reference does.
//!
//! # What must NOT change
//!
//! `vae2llm [2048, 64]` stays skipped — that asymmetry is required, not an
//! oversight. And golden `[256, 128]` stays *quantized*: a
//! `n % convrot_group_size` predicate would also match the reference but would
//! newly skip it, moving a pinned parity digest and dropping coverage of every
//! weight with `in_features < 256`. It is quantized but NOT rotated, which is
//! the documented `convrot_applied=False` fallback.
//!
//! `should_skip_shape` itself is deliberately untouched: it is a verbatim port
//! of `stream_quant.py::_should_skip_shape` whose both-dims semantics are
//! pinned by `tests/phase2_quant_parity.rs`.

mod common;

use common::{bf16_weight_bytes, load, write_input};
use quant_core::dtype::DType;
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::stream::stream_quantize;
use tempfile::TempDir;

/// ConvRot config: the `int8_convrot` preset — INT8, row mode, gs 256.
/// Mirrors `commands/quantize.rs::build_config` for `FormatArg::Int8Convrot`.
fn convrot_cfg() -> QuantConfig {
    QuantConfig {
        format: Format::Int8,
        target_format: "int8".into(),
        int8: true,
        scaling_mode: ScalingMode::Row,
        block_size: 128,
        convrot: true,
        convrot_group_size: 256,
        skip_inefficient: true,
        ..QuantConfig::default()
    }
}

/// Plain INT8 (no rotation), same shape/skip settings — the control arm.
fn plain_int8_cfg() -> QuantConfig {
    QuantConfig {
        convrot: false,
        ..convrot_cfg()
    }
}

/// Quantize one synthetic `[m, n]` weight and report whether it was emitted as
/// a quantized layer (i.e. got a `.comfy_quant` blob) or left in the source
/// dtype.
///
/// NOTE on naming: a source tensor `w.weight` produces the blob `w.comfy_quant`
/// — the layer prefix strips the `.weight` suffix (`stream.rs:1488`), which is
/// why the returned prefixes carry no `.weight`.
fn quantized_tensors(m: usize, n: usize, cfg: &QuantConfig) -> Vec<String> {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("in.safetensors");
    write_input(
        &input,
        &[(
            "w.weight",
            DType::Bf16,
            vec![m as u64, n as u64],
            bf16_weight_bytes(m, n),
        )],
    );
    let out = dir.path().join("out.safetensors");
    stream_quantize(&input, &out, cfg).expect("quantize must succeed");

    let reader = load(&out);
    reader
        .header()
        .names()
        .filter(|k| k.ends_with(".comfy_quant"))
        .map(|k| k.trim_end_matches(".comfy_quant").to_string())
        .collect()
}

fn is_quantized(m: usize, n: usize, cfg: &QuantConfig) -> bool {
    !quantized_tensors(m, n, cfg).is_empty()
}

// --------------------------------------------------------------------------
// The fix
// --------------------------------------------------------------------------

/// THE REGRESSION THIS FIXES. `llm2vae [64, 2048]` has fewer rows than
/// `block_size` (64 < 128) but 2048 % 256 == 0, so the rotation is well
/// defined. The reference quantizes it; before the fix we silently skipped it.
#[test]
fn convrot_quantizes_wide_tensor_with_few_rows() {
    assert!(
        is_quantized(64, 2048, &convrot_cfg()),
        "[64, 2048] must be quantized under convrot — its n=2048 is divisible by \
         the group size 256, and the reference quantizes this exact tensor \
         (llm2vae). A both-dims skip predicate drops it."
    );
}

/// The required asymmetry: `vae2llm [2048, 64]` stays UNquantized because its
/// column count (64) is below the block size. The reference leaves it BF16 too.
#[test]
fn convrot_still_skips_tall_tensor_with_narrow_columns() {
    assert!(
        !is_quantized(2048, 64, &convrot_cfg()),
        "[2048, 64] must stay skipped — n=64 < 128. The ComfyUI reference leaves \
         vae2llm in BF16, and quantizing it would be a divergence."
    );
}

/// REGRESSION GUARD for the pinned parity digest. Golden fixture
/// `linear_basic_bf16` contains `blocks.0.weight [256, 128]`; it is quantized
/// today and must stay so. A `n % convrot_group_size` predicate would skip it
/// and move `INT8_CONVROT_DIGEST` in tests/parity_fingerprint.rs.
#[test]
fn convrot_keeps_golden_256x128_quantized() {
    assert!(
        is_quantized(256, 128, &convrot_cfg()),
        "[256, 128] must stay quantized — it is a pinned-digest regression guard. \
         Its n=128 is not divisible by the convrot group size 256, so it is \
         quantized PLAIN (no convrot keys in its blob), which is the documented \
         convrot_applied=False fallback, not a regression."
    );
}

/// The other golden tensor stays skipped, unchanged.
#[test]
fn convrot_keeps_golden_128x64_skipped() {
    assert!(
        !is_quantized(128, 64, &convrot_cfg()),
        "[128, 64] must stay skipped — n=64 < 128."
    );
}

// --------------------------------------------------------------------------
// The plain-INT8 control arm — nothing outside convrot may move
// --------------------------------------------------------------------------

/// Plain INT8 must be entirely unaffected by the convrot branch. `[64, 2048]`
/// is skipped here exactly as before.
#[test]
fn plain_int8_is_unaffected_and_still_skips_short_row_tensor() {
    assert!(
        !is_quantized(64, 2048, &plain_int8_cfg()),
        "plain INT8 (block semantics) must still skip [64, 2048] — the convrot \
         column-axis rule must not leak into the non-convrot path."
    );
}

/// Plain INT8 row mode also keeps both-dims semantics: `[256, 128]` passes
/// (both dims >= 128 and divisible) and is quantized, as it always was.
#[test]
fn plain_int8_row_mode_keeps_both_dims_semantics() {
    assert!(
        is_quantized(256, 128, &plain_int8_cfg()),
        "plain INT8 row mode must still quantize [256, 128] via the unchanged \
         both-dims predicate."
    );
}

/// Plain INT8 row mode still rejects a row-divisibility failure, which the
/// convrot rule deliberately ignores. `[130, 128]` has m % 128 != 0.
#[test]
fn plain_int8_row_mode_still_rejects_bad_row_divisibility() {
    assert!(
        !is_quantized(130, 128, &plain_int8_cfg()),
        "plain INT8 must still skip [130, 128] — m % 128 != 0. If this starts \
         being quantized, the convrot branch has leaked."
    );
}

/// `--no-heur` turns the efficiency skip into a hard error for INT8 rather
/// than a silent skip (stream.rs: the `NotDivisible` guard fires when
/// `!config.skip_inefficient` and a dim is indivisible). That is the documented
/// contract, and this test pins it so the flag cannot be quietly redefined.
#[test]
fn no_heur_errors_on_indivisible_int8_shape() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("in.safetensors");
    write_input(
        &input,
        &[(
            "w.weight",
            DType::Bf16,
            vec![64, 2048],
            bf16_weight_bytes(64, 2048),
        )],
    );
    let out = dir.path().join("out.safetensors");
    let cfg = QuantConfig {
        skip_inefficient: false,
        ..convrot_cfg()
    };
    let err = stream_quantize(&input, &out, &cfg).expect_err("must reject");
    assert!(
        format!("{err:?}").contains("NotDivisible"),
        "expected a NotDivisible error for [64, 2048] with --no-heur, got: {err:?}"
    );
}

// --------------------------------------------------------------------------
// End-to-end: the emitted blob SET, not just a count
// --------------------------------------------------------------------------

/// Quantize a 3-tensor model and assert the exact set of quantized layers.
/// Asserting the key SET (not a count) is what makes this discriminating: a
/// wrong predicate could coincidentally produce the same NUMBER of blobs while
/// quantizing the wrong tensors.
#[test]
fn e2e_blob_set_matches_expected_two_of_three() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("in.safetensors");
    write_input(
        &input,
        &[
            (
                "llm2vae.weight",
                DType::Bf16,
                vec![64, 2048],
                bf16_weight_bytes(64, 2048),
            ),
            (
                "vae2llm.weight",
                DType::Bf16,
                vec![2048, 64],
                bf16_weight_bytes(2048, 64),
            ),
            (
                "mid.weight",
                DType::Bf16,
                vec![256, 128],
                bf16_weight_bytes(256, 128),
            ),
        ],
    );
    let out = dir.path().join("out.safetensors");
    stream_quantize(&input, &out, &convrot_cfg()).expect("quantize");

    let reader = load(&out);
    let mut got: Vec<String> = reader
        .header()
        .names()
        .filter(|k| k.ends_with(".comfy_quant"))
        .map(|k| k.trim_end_matches(".comfy_quant").to_string())
        .collect();
    got.sort();

    // llm2vae: recovered by the fix. mid: kept (plain, n%256 != 0).
    // vae2llm: stays BF16, no blob — matching the reference exactly.
    // Prefixes carry no `.weight`: the blob key strips it (stream.rs:1488).
    assert_eq!(
        got,
        vec!["llm2vae".to_string(), "mid".to_string()],
        "expected exactly llm2vae + mid quantized; vae2llm [2048,64] must stay BF16"
    );

    // vae2llm must be present but NOT quantized, i.e. no .weight_scale either.
    assert!(
        !reader.header().names().any(|k| k == "vae2llm.weight_scale"),
        "vae2llm must not receive a weight_scale while unquantized"
    );
    // llm2vae IS quantized, so it must carry a scale.
    assert!(
        reader.header().names().any(|k| k == "llm2vae.weight_scale"),
        "a quantized llm2vae must carry weight_scale"
    );
}

/// The rotation decision itself must not change: a tensor whose `n` is not
/// divisible by the group size is quantized but carries NO convrot keys, and
/// one whose `n` IS divisible does carry them. This guards against a "fix"
/// that starts rotating everything.
#[test]
fn rotation_keys_track_group_size_divisibility() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("in.safetensors");
    write_input(
        &input,
        &[
            (
                "rot.weight",
                DType::Bf16,
                vec![64, 512],
                bf16_weight_bytes(64, 512),
            ),
            (
                "plain.weight",
                DType::Bf16,
                vec![256, 128],
                bf16_weight_bytes(256, 128),
            ),
        ],
    );
    let out = dir.path().join("out.safetensors");
    stream_quantize(&input, &out, &convrot_cfg()).expect("quantize");

    let reader = load(&out);
    let names: Vec<String> = reader.header().names().cloned().collect();
    let blob = |key: &str| -> Option<String> {
        let k = format!("{key}.comfy_quant");
        if !names.contains(&k) {
            return None;
        }
        let raw = reader.tensor_bytes(&k).ok()?;
        Some(String::from_utf8_lossy(raw).into_owned())
    };

    let rot = blob("rot").expect("rot must be quantized");
    assert!(
        rot.contains("\"convrot\":true") || rot.contains("\"convrot\": true"),
        "rot has n=512 (divisible by 256) so it MUST be rotated; blob was: {rot}"
    );

    let plain = blob("plain").expect("plain must be quantized");
    assert!(
        !plain.contains("\"convrot\":true") && !plain.contains("\"convrot\": true"),
        "plain has n=128 (NOT divisible by 256) so it must stay PLAIN \
         with no convrot key; blob was: {plain}"
    );
}

//! S05 — `nvfp4_rot16`: NVFP4 with a group-wise Hadamard rotation at
//! group size 16 (== NVFP4's own block size).
//!
//! # What this step is (and is not)
//!
//! This preset is **byte-exact**: a Hadamard rotation is a parity-exact
//! transform of the weights, not a quality knob. It therefore introduces NO
//! [`Quality`] variant and moves NONE of the five pinned digests in
//! `parity_fingerprint.rs`. Its distinctness comes solely from
//! `convrot` / `convrot_group_size` in the `config_hash`.
//!
//! # THE LOAD-BEARING CAVEAT (read before shipping)
//!
//! The rotation is applied OFFLINE to the weights. For the artifact to be
//! numerically usable, the CONSUMER must apply the INVERSE rotation online.
//! Whether ComfyUI does so is **UNKNOWN and cannot be tested here** — the
//! failure is not in the file we write but in a downstream runtime. See the
//! `FormatArg::Nvfp4Rot16` doc comment.
//!
//! The second caveat is testable, and [`rot16_emits_no_convrot_metadata`]
//! documents it: the family-B `comfy_quant` blob schema has no convrot keys,
//! so nothing in the emitted artifact records that the weights are rotated.
//!
//! # Two traps this file is written around
//!
//! -- `[1, 2, ..., 16]` is an EIGENVECTOR of the H16 built from the
//!   Theorem-3.3 H4, so it rotates to itself. Using it to assert "the
//!   rotation changes something" fails against a correct implementation. See
//!   [`hadamard_16_is_orthogonal`].
//! -- The NVFP4 skip heuristic rejects any layer whose rows OR cols are
//!   `< 16` or not divisible by 16. A test fixture must be at least
//!   `[16, 16]`-shaped or the layer is COPIED unquantized and the test
//!   silently proves nothing. And with the heuristic ON, `in_features = 30`
//!   is skipped before the rotation rule is ever reached — the
//!   rotation-divisibility rule is only observable with it OFF. See
//!   [`rot16_leaves_indivisible_layers_unrotated`].

use quant_core::convrot::{build_hadamard, rotate_weight};
use quant_core::discover::{ctq_output_stem, ctq_quant_tags};
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::quality::Quality;

use tempfile::TempDir;

/// The `nvfp4_rot16` config as the CLI builds it: NVFP4, block scaling at 16,
/// rotation at group size 16. `Quality::Exact` — this step adds no quality
/// knob, and `Exact` contributes no `quality_tuning` hash key.
fn rot16_config() -> QuantConfig {
    QuantConfig {
        format: Format::Nvfp4,
        target_format: "nvfp4".into(),
        int8: false,
        scaling_mode: ScalingMode::Block,
        block_size: 16,
        convrot: true,
        convrot_group_size: 16,
        quality: Quality::Exact,
        ..QuantConfig::default()
    }
}

/// Plain NVFP4, no rotation.
fn plain_nvfp4_config() -> QuantConfig {
    QuantConfig {
        format: Format::Nvfp4,
        target_format: "nvfp4".into(),
        int8: false,
        scaling_mode: ScalingMode::Block,
        block_size: 16,
        ..QuantConfig::default()
    }
}

/// Write a deterministic, asymmetric, non-uniform `[m, n]` F32 weight.
///
/// Non-uniform on purpose: a symmetric or constant weight could hide a
/// missing rotation behind a coincidence. Named `w.weight` so the streaming
/// quantizer treats it as quantizable.
fn write_weight(path: &std::path::Path, m: usize, n: usize) {
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    let data: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32))
        .collect();
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor =
        TensorView::new(Dtype::F32, vec![m, n], &bytes).expect("test weight must be constructible");
    let tensors: std::collections::BTreeMap<String, TensorView> =
        [("w.weight".to_string(), tensor)].into_iter().collect();
    let out = safetensors::serialize(tensors, None).expect("serialize");
    std::fs::write(path, out).expect("write");
}

// ---------------------------------------------------------------------------
// 1. The most important test in this step: int8_convrot is UNTOUCHED
// ---------------------------------------------------------------------------

/// `int8_convrot` must still build group size 256, with an unchanged hash.
///
/// This is the tripwire against someone "tidying" 256 away or turning it into
/// a flag. 256 is a parity contract with the reference default
/// (`learned_rounding.py:868`), not a tunable.
#[test]
fn int8_convrot_group_size_is_still_256() {
    // Mirrors the CLI's `build_config` mapping for `FormatArg::Int8Convrot`.
    let cfg = QuantConfig {
        format: Format::Int8,
        target_format: "int8".into(),
        int8: true,
        scaling_mode: ScalingMode::Row,
        block_size: 128,
        convrot: true,
        convrot_group_size: 256,
        ..QuantConfig::default()
    };
    assert_eq!(
        cfg.convrot_group_size, 256,
        "int8_convrot's group size is a parity contract with the reference \
         default (learned_rounding.py:868) — it must never become a tunable"
    );

    // 256 must also still be reachable through the `Quality::Exact` default
    // that the CLI now sets explicitly.
    assert_eq!(Quality::Exact, Quality::default());
    assert_eq!(
        Quality::Exact.id(),
        None,
        "Exact adds no quality_tuning key"
    );

    // The strongest form of the tripwire: the WHOLE config hash is unchanged,
    // so a resumed `int8_convrot` run still recognises its own checkpoints.
    // `Quality::Exact` contributes no `quality_tuning` key, which is why
    // adding it in `build_config` did not move this value.
    assert_eq!(
        cfg.config_hash(),
        "e21b57672411c7c2",
        "int8_convrot's config_hash moved — this is a parity event, not a refactor"
    );
}

// ---------------------------------------------------------------------------
// 2. Resume safety: rot16 must not share a hash with plain nvfp4
// ---------------------------------------------------------------------------

/// `config_hash` is the ONLY resume guard — `StreamState::load_manifest`
/// trusts it alone. If `nvfp4_rot16` hashed the same as plain `nvfp4`, a
/// resumed run would splice rotated and unrotated weights into one file.
///
/// Plain NVFP4 is pinned at `95ede677cf402b53` by
/// `manifest::tests::config_hash_nvfp4_vector`.
#[test]
fn rot16_hash_differs_from_nvfp4_hash() {
    let plain = plain_nvfp4_config();
    let rot = rot16_config();

    // The reference pin for plain nvfp4 — if this moves, something is very
    // wrong upstream of this step.
    assert_eq!(
        plain.config_hash(),
        "95ede677cf402b53",
        "plain nvfp4's pinned hash moved — this step must not affect it"
    );

    assert_ne!(
        rot.config_hash(),
        plain.config_hash(),
        "rot16 MUST NOT share a hash with plain nvfp4: config_hash is the only \
         resume guard, so a collision would let a resumed run mix rotated and \
         unrotated weights in one artifact"
    );

    // PINNED: the value a real `--format nvfp4_rot16` run reports. Computed
    // from the exact config `build_config` produces for `FormatArg::Nvfp4Rot16`.
    // If this moves, every existing rot16 artifact becomes unresumable.
    assert_eq!(
        rot.config_hash(),
        "3de1874bf6bf2e01",
        "the nvfp4_rot16 config_hash moved — resumed runs would no longer \
         recognise their own checkpoints"
    );

    // Distinctness must come from convrot alone — drop the rotation fields
    // and the two configs become the same request.
    let mut stripped = rot.clone();
    stripped.convrot = false;
    stripped.convrot_group_size = 256;
    assert_eq!(
        stripped.config_hash(),
        plain.config_hash(),
        "the ONLY difference between the two configs must be the rotation"
    );

    // Also distinct from int8_convrot (different format AND group size).
    let int8_convrot = QuantConfig {
        format: Format::Int8,
        target_format: "int8".into(),
        int8: true,
        scaling_mode: ScalingMode::Row,
        block_size: 128,
        convrot: true,
        convrot_group_size: 256,
        ..QuantConfig::default()
    };
    assert_ne!(rot.config_hash(), int8_convrot.config_hash());
}

// ---------------------------------------------------------------------------
// 3. Auto-naming: distinct tags and a distinct output path
// ---------------------------------------------------------------------------

/// The new preset must auto-name to a path distinct from plain `nvfp4`, so
/// two runs cannot overwrite each other's output.
#[test]
fn resolve_output_path_differs_from_plain_nvfp4() {
    // `resolve_output` passes a `gs` tag only for the new preset.
    let rot_tags = ctq_quant_tags("nvfp4_rot16", Some("16"), true, false, "", true);
    let plain_tags = ctq_quant_tags("nvfp4", None, true, false, "", true);

    assert!(
        rot_tags.iter().any(|t| t == "gs16"),
        "the new preset must carry a gs16 tag, got {rot_tags:?}"
    );
    assert!(
        !plain_tags.iter().any(|t| t.starts_with("gs")),
        "plain nvfp4 has no rotation and must not gain a gs tag, got {plain_tags:?}"
    );

    let base = "model";
    let rot_path = ctq_output_stem(base, &rot_tags);
    let plain_path = ctq_output_stem(base, &plain_tags);
    assert_ne!(
        rot_path, plain_path,
        "auto-named outputs must differ or a rot16 run can clobber a plain nvfp4 run"
    );
    assert!(rot_path.contains("nvfp4_rot16"), "got {rot_path}");
    assert!(rot_path.contains("gs16"), "got {rot_path}");

    // int8_convrot keeps passing `None` — its filename must be UNCHANGED by
    // this step (adding a gs256 tag would silently rename existing artifacts).
    let convrot_tags = ctq_quant_tags("int8_convrot", None, true, false, "", true);
    assert_eq!(
        ctq_output_stem(base, &convrot_tags),
        "model-int8_convrot-simple-heur",
        "int8_convrot's auto-naming must be untouched by this step"
    );
}

// ---------------------------------------------------------------------------
// 4. Hadamard at gs=16 is orthogonal (no new rotation code needed)
// ---------------------------------------------------------------------------

/// 16 is on the power-of-4 ladder, which is why this preset needs NO new
/// rotation numerics. If someone "simplifies" the ladder check in
/// `build_hadamard`, this fails.
#[test]
fn hadamard_16_is_orthogonal() {
    let gs = 16usize;
    let h = build_hadamard(16).expect("16 is on the power-of-4 ladder (4^2)");

    assert_eq!(h.len(), gs * gs, "H must be {gs}x{gs}");

    // H @ H^T == I, i.e. dot(row_i, row_j) == delta_ij.
    for i in 0..gs {
        for j in 0..gs {
            let dot: f32 = (0..gs).map(|k| h[i * gs + k] * h[j * gs + k]).sum();
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expect).abs() < 1e-5,
                "H not orthogonal at ({i},{j}): {dot} != {expect}"
            );
        }
    }

    // The rotation must actually mix the values: a rotation that returned the
    // input unchanged (or a no-op) would pass an "is it orthogonal" check
    // while doing nothing.
    //
    // NOTE: do NOT use `w = [1, 2, ..., 16]` here. That vector is a genuine
    // EIGENVECTOR of the H16 built from the Theorem-3.3 H4 (verified against
    // numpy: `w @ H.T == w`), so it rotates to itself and the assertion below
    // would fail for a perfectly correct implementation. An impulse vector is
    // not an eigenvector and genuinely mixes.
    let mut w = vec![0.0f32; gs];
    w[0] = 1.0;
    let rotated = rotate_weight(&w, &h, 1, gs, 16).expect("16 divides 16");
    assert_ne!(rotated, w, "rotation must mix the weight values");
    // Every entry is now ±1/4 — the signature of a genuine Hadamard mix.
    assert!(
        rotated
            .iter()
            .all(|v| (*v - 0.25).abs() < 1e-6 || (*v + 0.25).abs() < 1e-6),
        "an impulse must spread to ±1/sqrt(16) = ±0.25 across the group, got {rotated:?}"
    );
}

// ---------------------------------------------------------------------------
// 5. Indivisible layers stay unrotated (and say so)
// ---------------------------------------------------------------------------

/// A layer whose `in_features` is NOT divisible by the group size must be
/// quantized UNROTATED, matching the per-tensor rule in `stream.rs`.
///
/// Two subtleties this test has to respect:
///
/// -- `skip_inefficient` MUST be off. With the heuristic ON (the CLI
///    default), NVFP4's block size is 16, so `should_skip_shape` already
///    rejects `in_features = 30` (`30 % 16 != 0`) and the layer is COPIED
///    unquantized before the rotation rule is ever reached. That would make
///    this test pass for the wrong reason.
/// -- The heuristic off means MXFP8/NVFP4 pad internally and never error
///    (`stream.rs`: only INT8 raises `NotDivisible`), so `in_features = 30`
///    is quantized — just not rotated.
#[test]
fn rot16_leaves_indivisible_layers_unrotated() {
    let tmp = TempDir::new().unwrap();
    // rows = 32 (>= 16), in_features = 30: NOT divisible by the group size 16.
    let input = tmp.path().join("in.safetensors");
    write_weight(&input, 32, 30);

    let mut rot = rot16_config();
    rot.skip_inefficient = false;
    let mut plain = plain_nvfp4_config();
    plain.skip_inefficient = false;

    let rot_out = tmp.path().join("rot.safetensors");
    let plain_out = tmp.path().join("plain.safetensors");
    quant_core::stream::stream_quantize(&input, &rot_out, &rot)
        .expect("rot16 must quantize an indivisible layer (unrotated)");
    quant_core::stream::stream_quantize(&input, &plain_out, &plain)
        .expect("plain nvfp4 must quantize the same layer");

    assert_eq!(
        std::fs::read(&rot_out).unwrap(),
        std::fs::read(&plain_out).unwrap(),
        "in_features 30 is not divisible by the group size 16, so the layer \
         must stay UNROTATED — its bytes must equal the plain-nvfp4 bytes"
    );
}

/// The converse, and the test that actually proves the rotation is wired in:
/// a divisible layer MUST be rotated, i.e. its bytes must DIFFER from plain
/// nvfp4. Without this, the test above would also pass on a build that never
/// rotates anything (a silent no-op "rotated" format).
#[test]
fn rot16_actually_rotates_divisible_layers() {
    let tmp = TempDir::new().unwrap();
    // rows = 32, in_features = 32: divisible by 16 → must be rotated.
    let input = tmp.path().join("in.safetensors");
    write_weight(&input, 32, 32);

    let mut rot = rot16_config();
    rot.skip_inefficient = false;
    let mut plain = plain_nvfp4_config();
    plain.skip_inefficient = false;

    let rot_out = tmp.path().join("rot.safetensors");
    let plain_out = tmp.path().join("plain.safetensors");
    quant_core::stream::stream_quantize(&input, &rot_out, &rot)
        .expect("rot16 must quantize a divisible layer");
    quant_core::stream::stream_quantize(&input, &plain_out, &plain)
        .expect("plain nvfp4 must quantize the same layer");

    assert_ne!(
        std::fs::read(&rot_out).unwrap(),
        std::fs::read(&plain_out).unwrap(),
        "a divisible layer MUST be rotated — bytes identical to plain nvfp4 \
         mean the rotation is not actually applied"
    );
}

// ---------------------------------------------------------------------------
// 6. Documented gap: no in-band convrot metadata for family B
// ---------------------------------------------------------------------------

/// The family-B `comfy_quant` blob schema has no `convrot` /
/// `convrot_groupsize` keys, so the emitted artifact carries NO in-band
/// signal that its weights were rotated.
///
/// This test pins that gap deliberately: it asserts the key is ABSENT, so
/// that anyone adding convrot metadata to the NVFP4 blob is forced to update
/// this test and confront the consumer-compatibility question.
#[test]
fn rot16_emits_no_convrot_metadata() {
    let tmp = TempDir::new().unwrap();
    let input = tmp.path().join("in.safetensors");
    write_weight(&input, 32, 32);
    let out = tmp.path().join("rot.safetensors");

    quant_core::stream::stream_quantize(&input, &out, &rot16_config()).unwrap();

    let reader = quant_core::st_io::reader::SafetensorsReader::open(&out).unwrap();
    let mut saw_blob = false;
    for name in reader.header().names() {
        let name = name.to_string();
        if !name.ends_with(".comfy_quant") {
            continue;
        }
        saw_blob = true;
        // `tensor_bytes` already returns exactly this tensor's payload.
        let raw = reader.tensor_bytes(&name).expect("blob bytes");
        let json = String::from_utf8_lossy(raw);
        assert!(
            !json.contains("convrot"),
            "family-B blob unexpectedly carries convrot metadata: {json}"
        );
    }
    assert!(saw_blob, "expected at least one .comfy_quant blob");
}

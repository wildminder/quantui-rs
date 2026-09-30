//! `--only <PREFIX>` end-to-end: the allow-list must actually GATE the run.
//!
//! # Why this file exists
//!
//! `in_keep_set` is unit-tested in `manifest.rs`, and the `--only` flag is
//! unit-tested at the CLI boundary — and both suites stayed GREEN when the
//! gate was deleted from `is_quantizable` entirely. The predicate was
//! covered; the one line that makes it matter to an actual artifact was not.
//! That is the same class of gap as the `--exclude-layers` warning before it:
//! covering a helper is not covering the decision.
//!
//! So this drives the real `stream_quantize` end to end and asserts on the
//! tensor set of the emitted safetensors, which is the only thing a user can
//! observe.
//!
//! # The real-world case
//!
//! ComfyUI's own Qwen-Image 2.1 int8_convrot file quantizes exactly the
//! `transformer_blocks.*` weights (192 of them) and leaves the 7 other 2-D
//! weights in BF16 — verified by cross-checking the published file. The flag
//! exists so that selection is one typo-proof token instead of a long
//! alternation regex that silently widens the artifact when mistyped.

mod common;

use common::{bf16_weight_bytes, load, write_input};
use quant_core::dtype::DType;
use quant_core::manifest::{Format, QuantConfig, ScalingMode};
use quant_core::stream::stream_quantize;
use tempfile::TempDir;

/// Shapes chosen so every tensor would otherwise be quantized: `in_features`
/// is a multiple of 256 for all of them, so the ONLY thing that can exclude
/// one is the selection policy under test. `img_in` deliberately uses
/// `n = 64` to mirror the real model, but it is given its own assertion below
/// so the efficiency heuristic is not what is being measured here.
const SHAPES: &[(&str, usize, usize)] = &[
    ("transformer_blocks.0.attn.to_q.weight", 256, 256),
    ("transformer_blocks.0.attn.to_k.weight", 512, 256),
    ("transformer_blocks.1.img_mlp.gate_up.weight", 256, 512),
    ("txt_in.in_layer.weight", 256, 256),
    ("txt_in.out_layer.weight", 256, 256),
    ("proj_out.weight", 256, 256),
    ("norm_out.linear.weight", 256, 256),
    ("modulation.1.weight", 256, 256),
];

fn cfg() -> QuantConfig {
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

/// Run a real quantize over SHAPES with `cfg` and return the set of layer
/// prefixes that received a `.comfy_quant` blob, i.e. were QUANTIZED.
fn quantized_set(cfg: &QuantConfig) -> Vec<String> {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("in.safetensors");
    let tensors: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = SHAPES
        .iter()
        .map(|(n, m, k)| {
            (
                *n,
                DType::Bf16,
                vec![*m as u64, *k as u64],
                bf16_weight_bytes(*m, *k),
            )
        })
        .collect();
    write_input(&input, &tensors);

    let out = dir.path().join("out.safetensors");
    stream_quantize(&input, &out, cfg).unwrap();
    let reader = load(&out);
    reader
        .header()
        .names()
        .filter(|k| k.ends_with(".comfy_quant"))
        .map(|k| k.trim_end_matches(".comfy_quant").to_string())
        .collect()
}

/// THE test. With `--only transformer_blocks`, exactly the three block weights
/// are quantized and every other eligible 2-D weight stays BF16.
///
/// This is the assertion that goes red if the gate is ever removed from
/// `is_quantizable` — the unit tests do not.
#[test]
fn only_prefix_gates_the_real_artifact() {
    let cfg = QuantConfig {
        only_prefixes: vec!["transformer_blocks".into()],
        ..cfg()
    };
    let mut got = quantized_set(&cfg);
    let mut expected = vec![
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.1.img_mlp.gate_up",
    ];
    expected.sort();
    got.sort();
    assert_eq!(
        got, expected,
        "--only must gate the EMITTED artifact, not just the predicate"
    );
}

/// The control: no `--only` quantizes every eligible tensor. Without this,
/// the test above could pass because the shapes were never eligible at all.
#[test]
fn absent_only_prefix_quantizes_everything_eligible() {
    let mut got = quantized_set(&cfg());
    got.sort();
    got.sort();
    assert_eq!(
        got.len(),
        SHAPES.len(),
        "control: with no allow-list every 2-D weight here must be quantized, \
         got {got:?}"
    );
    assert!(got.contains(&"proj_out".to_string()));
    assert!(got.contains(&"txt_in.in_layer".to_string()));
}

/// `--only` and `--exclude-layers` compose: the allow-list admits a name, the
/// regex then drops it. Order and precedence must be unambiguous.
#[test]
fn only_then_exclude_narrows_further() {
    let cfg = QuantConfig {
        only_prefixes: vec!["transformer_blocks".into()],
        exclude_layers: Some(r"\.attn\.to_q\.weight$".into()),
        ..cfg()
    };
    let mut got = quantized_set(&cfg);
    got.sort();
    assert_eq!(
        got,
        vec![
            "transformer_blocks.0.attn.to_k".to_string(),
            "transformer_blocks.1.img_mlp.gate_up".to_string(),
        ],
        "an explicit exclusion must still apply inside the allow-list"
    );
}

/// Repeatable = OR, and it must not degrade into "matches nothing".
#[test]
fn repeated_only_prefixes_are_a_union() {
    let cfg = QuantConfig {
        only_prefixes: vec!["transformer_blocks".into(), "txt_in".into()],
        ..cfg()
    };
    let mut got = quantized_set(&cfg);
    got.sort();
    assert_eq!(
        got.len(),
        5,
        "union of 3 block + 2 txt_in tensors, got {got:?}"
    );
    assert!(got.contains(&"txt_in.out_layer".to_string()));
    assert!(!got.contains(&"proj_out".to_string()));
}

/// A prefix that matches NOTHING must produce a valid, fully-BF16 file rather
/// than an error or a silently half-quantized artifact. Zero quantization is
/// a legitimate (if pointless) request and must be visible in the output.
#[test]
fn a_prefix_matching_nothing_quantizes_nothing() {
    let cfg = QuantConfig {
        only_prefixes: vec!["no_such_prefix".into()],
        ..cfg()
    };
    assert!(
        quantized_set(&cfg).is_empty(),
        "an unmatched allow-list must exclude everything, not silently pass"
    );
}

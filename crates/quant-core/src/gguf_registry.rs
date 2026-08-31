//! GGUF method registry (Phase 10.3).
//!
//! Data-driven port of the reference `quantui/quant_methods.py` GGUF
//! `METHODS` list — the single source of truth for which GGUF quantization
//! methods the CLI offers — plus the per-tensor scheme policies that the
//! composite methods (`q4_k_m`, `q5_k_m`, `q3_k_*`, `q2_k`) encode in
//! llama.cpp's `llama-quantize`.
//!
//! Scope boundary (plan §1 / §7): the Unsloth Dynamic 2.0 per-layer
//! selective variants (`q4_k_xl` / `q3_k_xl` / `q2_k_xl`, and the
//! `dynamic_v2`-tagged `q4_nl` entry) are NOT supported natively — the
//! proprietary per-layer bit-width heuristic is a permanent Python-only
//! boundary. They are kept in the registry data (verbatim ids/labels/bpw/
//! descriptions, `dynamic_v2 = true`) so the CLI can list them and reject
//! them with a clear message instead of a silent typo failure.
//!
//! Note: `q4_1` and `q5_1` are NOT in that rejected set. The ported
//! reference `quant_methods.py` mislabelled them "Dynamic 2.0 format", but
//! official Unsloth's `ALLOWED_QUANTS` (save.py:163-170) and llama.cpp both
//! treat them as plain legacy types with ftypes and encoders; they were
//! reclassified in the 2026-08-31 Unsloth coverage plan (Phase 1).
//!
//! The registry is pure data + lookup: no IO, no rlx-gguf types leak into
//! the public surface (callers map [`GgufScheme`] → `GgmlType` at the
//! conversion boundary), so the rest of quant-core stays independent of
//! the GGUF dependency.

/// One GGUF quantization method (port of `quant_methods.QuantMethod`,
/// GGUF family only).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GgufMethod {
    /// Method id, exactly the reference id (`"q4_k_m"`, `"f16"`, ...).
    pub id: &'static str,
    /// Dropdown label, verbatim from the reference.
    pub label: &'static str,
    /// Unsloth Dynamic 2.0 per-layer selective variant (unsupported
    /// natively; listed for completeness + friendly rejection).
    pub dynamic_v2: bool,
    /// Approximate bits-per-weight (reference `approx_bpw`, verbatim).
    pub approx_bpw: Option<f64>,
    /// Description, verbatim from the reference.
    pub description: &'static str,
    /// Backend encoder capability (Unsloth plan Phase 0.2).
    pub support: BackendSupport,
    /// Whether official Unsloth gates this method behind `imatrix_file=`
    /// (save.py:2162: every `iq*` id in `IMATRIX_QUANTS`).
    pub requires_imatrix: bool,
}

/// What the pinned backend (`rlx-gguf` 0.2.14) can actually do with a
/// method. A method is only listed as usable when this is `Encodable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSupport {
    /// Every scheme the policy can select has an rlx-gguf encoder.
    Encodable,
    /// Kept in the registry for honest listing/rejection, but the backend
    /// has no encoder for at least one scheme the policy would select.
    /// Carries the reason for the CLI's error message.
    NoEncoder(&'static str),
}

/// Storage scheme for one tensor.
///
/// Deliberately a local enum (not `rlx_gguf::GgmlType`) so the registry
/// has no dependency on the GGUF crate; the conversion command maps this
/// 1:1 onto `GgmlType` variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufScheme {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq4Nl,
    // ── added by the Unsloth coverage plan Phase 3 ──
    Iq1S,
    Iq1M,
    Iq2S,
    Iq3S,
    Iq4Xs,
    Tq1_0,
    Tq2_0,
    Q1_0,
    Q2_0,
}

/// Per-tensor quantization policy of a method.
///
/// `Default` covers the plain methods (`q4_k_s` → Q4K everywhere, `f16` →
/// F16 everywhere, ...). The composite `_M`/`_L`/`_XS` methods carry
/// explicit rules ported from llama.cpp's `llama-quantize`
/// (`new_quantize` / `llama_model_quantize`): a list of (substring match
/// on the GGUF tensor name, scheme) overrides evaluated first-match-wins,
/// plus the shared conventions:
///
/// - 1-D tensors (norms, biases) stay F32,
/// - `token_embd.weight` / `output.weight` get `embd_scheme` (F16 for the
///   K-quant composites, else the default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MethodPolicy {
    /// Scheme for quantizable 2-D tensors with no rule match.
    pub default: GgufScheme,
    /// Scheme for `token_embd.weight` and `output.weight`.
    pub embd_scheme: GgufScheme,
    /// First-match-wins (GGUF-name substring, scheme) overrides.
    pub rules: &'static [(&'static str, GgufScheme)],
}

/// Registry entry: method metadata + its per-tensor policy.
#[derive(Debug, Clone, Copy)]
pub struct RegistryEntry {
    pub method: GgufMethod,
    pub policy: MethodPolicy,
}

// ─── policy fragments (llama.cpp `llama-quantize` ports) ────────────

const Q6K: GgufScheme = GgufScheme::Q6K;
const Q5K: GgufScheme = GgufScheme::Q5K;
const Q4K: GgufScheme = GgufScheme::Q4K;
const Q3K: GgufScheme = GgufScheme::Q3K;
const Q2K: GgufScheme = GgufScheme::Q2K;
const F16: GgufScheme = GgufScheme::F16;
const F32: GgufScheme = GgufScheme::F32;

/// Q5_K_M: attn_v + ffn_down get Q6_K, rest Q5_K.
static RULES_Q5_K_M: &[(&str, GgufScheme)] = &[("attn_v", Q6K), ("ffn_down", Q6K)];

/// Q4_K_M: attn_v + ffn_down get Q6_K, rest Q4_K.
static RULES_Q4_K_M: &[(&str, GgufScheme)] = &[("attn_v", Q6K), ("ffn_down", Q6K)];

/// Q3_K_M: attn_output Q5_K; attn_q/k/v Q4_K; half of ffn_down Q6_K;
/// rest Q3_K. (The "half" selection in llama-quantize is by layer index
/// parity; we apply Q6_K to every ffn_down — the documented simplification
/// keeps the policy data-driven without a layer-count dependency.)
static RULES_Q3_K_M: &[(&str, GgufScheme)] = &[
    ("attn_output", Q5K),
    ("attn_q", Q4K),
    ("attn_k", Q4K),
    ("attn_v", Q4K),
    ("ffn_down", Q6K),
];

/// Q3_K_L: attn_output Q5_K, rest Q3_K.
static RULES_Q3_K_L: &[(&str, GgufScheme)] = &[("attn_output", Q5K)];

/// Q3_K_S / Q3_K_XS / Q2_K: attn_v Q4_K, rest the base scheme.
static RULES_ATTN_V_Q4K: &[(&str, GgufScheme)] = &[("attn_v", Q4K)];

// ─── the registry (verbatim reference order) ────────────────────────

/// All GGUF methods, in the exact reference `METHODS` order. The three
/// UD-* Dynamic 2.0 entries are included with `dynamic_v2 = true` and an
/// empty policy (they are rejected before the policy is ever consulted).
pub static METHODS: &[RegistryEntry] = &[
    entry(
        "f16",
        "F16 (16-bit, lossless)",
        false,
        Some(16.0),
        "Full 16-bit. Largest, lossless. Best as an intermediate before manual quant.",
        MethodPolicy {
            default: F16,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q8_0",
        "Q8_0 (8-bit)",
        false,
        Some(8.6),
        "8-bit. Near-lossless, high memory use. Fast conversion.",
        MethodPolicy {
            default: GgufScheme::Q8_0,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q6_k",
        "Q6_K (6-bit)",
        false,
        Some(6.6),
        "6-bit K-quant. Very good quality, fairly large.",
        MethodPolicy {
            default: Q6K,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q5_k_m",
        "Q5_K_M (5-bit, recommended)",
        false,
        Some(5.5),
        "Recommended 5-bit. Near-lossless quality with good size.",
        MethodPolicy {
            default: Q5K,
            embd_scheme: F16,
            rules: RULES_Q5_K_M,
        },
    ),
    entry(
        "q5_k_s",
        "Q5_K_S (5-bit small)",
        false,
        Some(5.5),
        "5-bit small. Uses Q5_K for all tensors.",
        MethodPolicy {
            default: Q5K,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q5_0",
        "Q5_0",
        false,
        Some(5.5),
        "Higher accuracy, slower inference.",
        MethodPolicy {
            default: GgufScheme::Q5_0,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    // NOTE: the ported reference `docs/ref/quantui/quant_methods.py:131`
    // labels q5_1 a "Dynamic 2.0 format", which is why this entry used to be
    // rejected. That is wrong, and per the project's source-of-truth order
    // (official Unsloth > llama.cpp > the quantui reference) it does not
    // govern: `ref/unsloth/unsloth/save.py:170` lists q5_1 in ALLOWED_QUANTS
    // as a plain legacy type, and llama.cpp has LLAMA_FTYPE_MOSTLY_Q5_1
    // (include/llama.h:126) with an encoder in the ggml_quantize_chunk
    // dispatch (ggml/src/ggml.c:7999). Byte-parity goldens for it already
    // pass. Do not "restore" dynamic_v2 here without re-checking Unsloth.
    entry(
        "q5_1",
        "Q5_1",
        false,
        Some(5.5),
        "Even higher accuracy, resource usage and slower inference.",
        MethodPolicy {
            default: GgufScheme::Q5_1,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q4_k_m",
        "Q4_K_M (4-bit, recommended)",
        false,
        Some(4.85),
        "Recommended 4-bit. Good balance of size and quality.",
        MethodPolicy {
            default: Q4K,
            embd_scheme: F16,
            rules: RULES_Q4_K_M,
        },
    ),
    entry(
        "q4_k_s",
        "Q4_K_S (4-bit small)",
        false,
        Some(4.5),
        "4-bit small. Q4_K for all tensors.",
        MethodPolicy {
            default: Q4K,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q4_0",
        "Q4_0",
        false,
        Some(4.55),
        "Original 4-bit method.",
        MethodPolicy {
            default: GgufScheme::Q4_0,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    // See the q5_1 note: the ported reference (`quant_methods.py:124`) is
    // wrong here too. Official Unsloth `ALLOWED_QUANTS`
    // (`ref/unsloth/unsloth/save.py:164`) lists q4_1 as a plain legacy type,
    // and llama.cpp has LLAMA_FTYPE_MOSTLY_Q4_1 (include/llama.h:120) with an
    // encoder in the ggml_quantize_chunk dispatch (ggml/src/ggml.c:7998).
    entry(
        "q4_1",
        "Q4_1",
        false,
        Some(4.8),
        "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
        MethodPolicy {
            default: GgufScheme::Q4_1,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q4_nl",
        "Q4_NL (Dynamic 2.0 format)",
        true,
        Some(4.5),
        "New Dynamic 2.0 efficiency format for Apple Silicon / ARM.",
        MethodPolicy {
            default: GgufScheme::Iq4Nl,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q3_k_m",
        "Q3_K_M (3-bit)",
        false,
        Some(3.9),
        "3-bit. Q4_K for key tensors.",
        MethodPolicy {
            default: Q3K,
            embd_scheme: F16,
            rules: RULES_Q3_K_M,
        },
    ),
    entry(
        "q3_k_l",
        "Q3_K_L (3-bit large)",
        false,
        Some(4.0),
        "3-bit large.",
        MethodPolicy {
            default: Q3K,
            embd_scheme: F16,
            rules: RULES_Q3_K_L,
        },
    ),
    entry(
        "q3_k_s",
        "Q3_K_S (3-bit small)",
        false,
        Some(3.5),
        "3-bit small. Q3_K for all tensors.",
        MethodPolicy {
            default: Q3K,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q3_k_xs",
        "Q3_K_XS (3-bit XS)",
        false,
        Some(3.3),
        "3-bit extra-small.",
        MethodPolicy {
            default: Q3K,
            embd_scheme: F16,
            rules: RULES_ATTN_V_Q4K,
        },
    ),
    entry(
        "q2_k",
        "Q2_K (2-bit)",
        false,
        Some(2.96),
        "2-bit. Q4_K for key tensors.",
        MethodPolicy {
            default: Q2K,
            embd_scheme: F16,
            rules: RULES_ATTN_V_Q4K,
        },
    ),
    // The four iq* entries: all are in official Unsloth IMATRIX_QUANTS
    // (save.py:176-188), so requires_imatrix = true — the metadata records
    // the Unsloth contract. The CLI gate is deliberately NOT enforced yet:
    // per plan §6-Q5 it ships together with the weighted quantizers
    // (Phase 4.3/4.2), otherwise every iq* method becomes unusable today.
    entry_full(
        "iq4_nl",
        "IQ4_NL (imatrix)",
        false,
        Some(4.5),
        "Importance-matrix 4-bit (needs an imatrix file).",
        MethodPolicy {
            default: GgufScheme::Iq4Nl,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    entry_full(
        "iq3_xxs",
        "IQ3_XXS (imatrix)",
        false,
        Some(3.06),
        "Importance quant, very small.",
        MethodPolicy {
            default: GgufScheme::Iq3Xxs,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    entry_full(
        "iq2_xxs",
        "IQ2_XXS (imatrix)",
        false,
        Some(2.06),
        "Importance quant, tiny.",
        MethodPolicy {
            default: GgufScheme::Iq2Xxs,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    entry_full(
        "iq2_xs",
        "IQ2_XS (imatrix)",
        false,
        Some(2.31),
        "Importance quant.",
        MethodPolicy {
            default: GgufScheme::Iq2Xs,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    // ══ Unsloth coverage plan Phase 3 additions (2026-08-31) ══════════
    // The original 20 entries above stay in the exact reference order.
    // New methods follow, each with the Unsloth/llama.cpp citation.
    //
    // 3.1 — f32 / bf16: Unsloth ALLOWED_QUANTS (save.py:152,153). f32 is a
    // passthrough (llama.cpp dispatch ggml.c:8021, memcpy); bf16 keeps the
    // source dtype when the model is already bf16.
    entry(
        "f32",
        "F32 (32-bit, lossless)",
        false,
        Some(32.6),
        "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
        MethodPolicy {
            default: F32,
            embd_scheme: F32,
            rules: &[],
        },
    ),
    entry(
        "bf16",
        "BF16 (bfloat16, lossless)",
        false,
        Some(16.6),
        "Bfloat16 - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
        MethodPolicy {
            default: GgufScheme::Bf16,
            embd_scheme: GgufScheme::Bf16,
            rules: &[],
        },
    ),
    // 3.3 — iq1_s / iq1_m / iq2_s / iq3_s / iq4_xs: Unsloth
    // IMATRIX_QUANTS (save.py:176-188). All require an imatrix; the CLI
    // gate ships with Phase 4.2/4.3 per plan Q5, until then the metadata
    // marks them and the encoder runs with uniform weights (documented).
    entry_full(
        "iq1_s",
        "IQ1_S (imatrix)",
        false,
        Some(1.56),
        "1.56 bpw. Smallest, lowest quality. Needs an imatrix.",
        MethodPolicy {
            default: GgufScheme::Iq1S,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    entry_full(
        "iq1_m",
        "IQ1_M (imatrix)",
        false,
        Some(1.75),
        "1.75 bpw. Very small. Needs an imatrix.",
        MethodPolicy {
            default: GgufScheme::Iq1M,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    entry_full(
        "iq2_s",
        "IQ2_S (imatrix)",
        false,
        Some(2.5),
        "2.5 bpw. Needs an imatrix.",
        MethodPolicy {
            default: GgufScheme::Iq2S,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    entry_full(
        "iq3_s",
        "IQ3_S (imatrix)",
        false,
        Some(3.44),
        "3.44 bpw. Needs an imatrix.",
        MethodPolicy {
            default: GgufScheme::Iq3S,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    entry_full(
        "iq4_xs",
        "IQ4_XS (imatrix)",
        false,
        Some(4.25),
        "4.25 bpw. Benefits from an imatrix.",
        MethodPolicy {
            default: GgufScheme::Iq4Xs,
            embd_scheme: F16,
            rules: &[],
        },
        BackendSupport::Encodable,
        true,
    ),
    // 3.7 / 3.8 — llama.cpp-only types (decision Q4: plain list entries,
    // no gating flag). LLAMA_FTYPE_MOSTLY_TQ1_0/TQ2_0 (llama.h:153-154),
    // Q1_0/Q2_0 (llama.h:157-158); encoders verified in rlx-gguf dispatch.
    entry(
        "tq1_0",
        "TQ1_0 (ternary 1-bit)",
        false,
        Some(1.69),
        "Ternary quantization for BitNet models (1.58-bit).",
        MethodPolicy {
            default: GgufScheme::Tq1_0,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "tq2_0",
        "TQ2_0 (ternary 2-bit)",
        false,
        Some(2.06),
        "Ternary 2-bit for BitNet b1.58 models.",
        MethodPolicy {
            default: GgufScheme::Tq2_0,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q1_0",
        "Q1_0 (1-bit)",
        false,
        Some(1.7),
        "1-bit quantization (llama.cpp only, not in Unsloth's list).",
        MethodPolicy {
            default: GgufScheme::Q1_0,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q2_0",
        "Q2_0 (2-bit)",
        false,
        Some(2.7),
        "2-bit quantization (llama.cpp only, not in Unsloth's list).",
        MethodPolicy {
            default: GgufScheme::Q2_0,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    // ── Unsloth Dynamic 2.0 per-layer selective (unsupported natively) ──
    entry(
        "q4_k_xl",
        "UD-Q4_K_XL (Dynamic 2.0)",
        true,
        Some(4.5),
        "Dynamic 2.0 per-layer selective. Best quality at ~Q4 size. Output: UD-Q4_K_XL.",
        MethodPolicy {
            default: Q4K,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q3_k_xl",
        "UD-Q3_K_XL (Dynamic 2.0)",
        true,
        Some(3.5),
        "Dynamic 2.0 per-layer selective, smaller. Output: UD-Q3_K_XL.",
        MethodPolicy {
            default: Q3K,
            embd_scheme: F16,
            rules: &[],
        },
    ),
    entry(
        "q2_k_xl",
        "UD-Q2_K_XL (Dynamic 2.0)",
        true,
        Some(2.7),
        "Dynamic 2.0 per-layer selective, smallest. Output: UD-Q2_K_XL.",
        MethodPolicy {
            default: Q2K,
            embd_scheme: F16,
            rules: &[],
        },
    ),
];

const fn entry(
    id: &'static str,
    label: &'static str,
    dynamic_v2: bool,
    approx_bpw: Option<f64>,
    description: &'static str,
    policy: MethodPolicy,
) -> RegistryEntry {
    entry_full(
        id,
        label,
        dynamic_v2,
        approx_bpw,
        description,
        policy,
        BackendSupport::Encodable,
        false,
    )
}

/// Full constructor with explicit capability metadata (Phase 0.2).
#[allow(clippy::too_many_arguments)] // data-driven table constructor
const fn entry_full(
    id: &'static str,
    label: &'static str,
    dynamic_v2: bool,
    approx_bpw: Option<f64>,
    description: &'static str,
    policy: MethodPolicy,
    support: BackendSupport,
    requires_imatrix: bool,
) -> RegistryEntry {
    RegistryEntry {
        method: GgufMethod {
            id,
            label,
            dynamic_v2,
            approx_bpw,
            description,
            support,
            requires_imatrix,
        },
        policy,
    }
}

/// Look up a method by id (case-sensitive, exactly the reference ids).
pub fn get_method(id: &str) -> Option<&'static RegistryEntry> {
    METHODS.iter().find(|e| e.method.id == id)
}

/// All natively supported (non-Dynamic-2.0) method ids, in registry order.
pub fn supported_ids() -> Vec<&'static str> {
    METHODS
        .iter()
        .filter(|e| !e.method.dynamic_v2)
        .map(|e| e.method.id)
        .collect()
}

/// Ids a conversion can actually run with today: non-dynamic AND every
/// scheme the policy selects has a backend encoder. This — not
/// [`supported_ids`] — is what `--list-methods` advertises as runnable
/// and what the unknown-method error should suggest once the first
/// `NoEncoder` entry lands (Unsloth plan Phase 2.1).
pub fn usable_ids() -> Vec<&'static str> {
    METHODS
        .iter()
        .filter(|e| !e.method.dynamic_v2 && e.method.support == BackendSupport::Encodable)
        .map(|e| e.method.id)
        .collect()
}

/// Whether a conversion attempt with `id` would fail before any tensor is
/// encoded, and why (dynamic_v2 proprietary, or missing backend encoder).
pub fn rejection_reason(id: &str) -> Option<&'static str> {
    let e = get_method(id)?;
    if e.method.dynamic_v2 {
        return Some("Unsloth Dynamic 2.0 per-layer variant (proprietary)");
    }
    match e.method.support {
        BackendSupport::NoEncoder(reason) => Some(reason),
        BackendSupport::Encodable => None,
    }
}

/// All Dynamic 2.0 ids (listed by the CLI, rejected at conversion time).
pub fn dynamic_ids() -> Vec<&'static str> {
    METHODS
        .iter()
        .filter(|e| e.method.dynamic_v2)
        .map(|e| e.method.id)
        .collect()
}

/// Resolve the storage scheme for one tensor under `method`'s policy.
///
/// `gguf_name` is the tensor's GGUF-side name (e.g. `blk.0.attn_v.weight`)
/// — the policy rules match llama.cpp naming, so the HF→GGUF name mapping
/// must happen before this call. `ndim` is the tensor's rank.
///
/// Conventions (llama.cpp `llama_model_quantize`):
/// - 1-D tensors (norms, biases, positions) stay F32 — quantizing them
///   buys nothing and costs accuracy;
/// - `token_embd.weight` / `output.weight` use the method's `embd_scheme`;
/// - everything else: first matching rule wins, else the default scheme.
pub fn scheme_for(method: &RegistryEntry, gguf_name: &str, ndim: usize) -> GgufScheme {
    if ndim < 2 {
        return F32;
    }
    if gguf_name == "token_embd.weight" || gguf_name == "output.weight" {
        return method.policy.embd_scheme;
    }
    for (needle, scheme) in method.policy.rules {
        if gguf_name.contains(needle) {
            return *scheme;
        }
    }
    method.policy.default
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_matches_reference_method_list() {
        // Verbatim port of docs/ref/quantui/quant_methods.py METHODS
        // (ids, labels, dynamic flags, bpw, descriptions, order).
        let expected: &[(&str, &str, bool, Option<f64>, &str)] = &[
            (
                "f16",
                "F16 (16-bit, lossless)",
                false,
                Some(16.0),
                "Full 16-bit. Largest, lossless. Best as an intermediate before manual quant.",
            ),
            (
                "q8_0",
                "Q8_0 (8-bit)",
                false,
                Some(8.6),
                "8-bit. Near-lossless, high memory use. Fast conversion.",
            ),
            (
                "q6_k",
                "Q6_K (6-bit)",
                false,
                Some(6.6),
                "6-bit K-quant. Very good quality, fairly large.",
            ),
            (
                "q5_k_m",
                "Q5_K_M (5-bit, recommended)",
                false,
                Some(5.5),
                "Recommended 5-bit. Near-lossless quality with good size.",
            ),
            (
                "q5_k_s",
                "Q5_K_S (5-bit small)",
                false,
                Some(5.5),
                "5-bit small. Uses Q5_K for all tensors.",
            ),
            (
                "q5_0",
                "Q5_0",
                false,
                Some(5.5),
                "Higher accuracy, slower inference.",
            ),
            (
                "q5_1",
                "Q5_1",
                false,
                Some(5.5),
                "Even higher accuracy, resource usage and slower inference.",
            ),
            (
                "q4_k_m",
                "Q4_K_M (4-bit, recommended)",
                false,
                Some(4.85),
                "Recommended 4-bit. Good balance of size and quality.",
            ),
            (
                "q4_k_s",
                "Q4_K_S (4-bit small)",
                false,
                Some(4.5),
                "4-bit small. Q4_K for all tensors.",
            ),
            ("q4_0", "Q4_0", false, Some(4.55), "Original 4-bit method."),
            (
                "q4_1",
                "Q4_1",
                false,
                Some(4.8),
                "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
            ),
            (
                "q4_nl",
                "Q4_NL (Dynamic 2.0 format)",
                true,
                Some(4.5),
                "New Dynamic 2.0 efficiency format for Apple Silicon / ARM.",
            ),
            (
                "q3_k_m",
                "Q3_K_M (3-bit)",
                false,
                Some(3.9),
                "3-bit. Q4_K for key tensors.",
            ),
            (
                "q3_k_l",
                "Q3_K_L (3-bit large)",
                false,
                Some(4.0),
                "3-bit large.",
            ),
            (
                "q3_k_s",
                "Q3_K_S (3-bit small)",
                false,
                Some(3.5),
                "3-bit small. Q3_K for all tensors.",
            ),
            (
                "q3_k_xs",
                "Q3_K_XS (3-bit XS)",
                false,
                Some(3.3),
                "3-bit extra-small.",
            ),
            (
                "q2_k",
                "Q2_K (2-bit)",
                false,
                Some(2.96),
                "2-bit. Q4_K for key tensors.",
            ),
            (
                "iq4_nl",
                "IQ4_NL (imatrix)",
                false,
                Some(4.5),
                "Importance-matrix 4-bit (needs an imatrix file).",
            ),
            (
                "iq3_xxs",
                "IQ3_XXS (imatrix)",
                false,
                Some(3.06),
                "Importance quant, very small.",
            ),
            (
                "iq2_xxs",
                "IQ2_XXS (imatrix)",
                false,
                Some(2.06),
                "Importance quant, tiny.",
            ),
            (
                "iq2_xs",
                "IQ2_XS (imatrix)",
                false,
                Some(2.31),
                "Importance quant.",
            ),
            // ── Phase 3 additions (Unsloth coverage plan) ──
            (
                "f32",
                "F32 (32-bit, lossless)",
                false,
                Some(32.6),
                "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
            ),
            (
                "bf16",
                "BF16 (bfloat16, lossless)",
                false,
                Some(16.6),
                "Bfloat16 - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
            ),
            (
                "iq1_s",
                "IQ1_S (imatrix)",
                false,
                Some(1.56),
                "1.56 bpw. Smallest, lowest quality. Needs an imatrix.",
            ),
            (
                "iq1_m",
                "IQ1_M (imatrix)",
                false,
                Some(1.75),
                "1.75 bpw. Very small. Needs an imatrix.",
            ),
            (
                "iq2_s",
                "IQ2_S (imatrix)",
                false,
                Some(2.5),
                "2.5 bpw. Needs an imatrix.",
            ),
            (
                "iq3_s",
                "IQ3_S (imatrix)",
                false,
                Some(3.44),
                "3.44 bpw. Needs an imatrix.",
            ),
            (
                "iq4_xs",
                "IQ4_XS (imatrix)",
                false,
                Some(4.25),
                "4.25 bpw. Benefits from an imatrix.",
            ),
            (
                "tq1_0",
                "TQ1_0 (ternary 1-bit)",
                false,
                Some(1.69),
                "Ternary quantization for BitNet models (1.58-bit).",
            ),
            (
                "tq2_0",
                "TQ2_0 (ternary 2-bit)",
                false,
                Some(2.06),
                "Ternary 2-bit for BitNet b1.58 models.",
            ),
            (
                "q1_0",
                "Q1_0 (1-bit)",
                false,
                Some(1.7),
                "1-bit quantization (llama.cpp only, not in Unsloth's list).",
            ),
            (
                "q2_0",
                "Q2_0 (2-bit)",
                false,
                Some(2.7),
                "2-bit quantization (llama.cpp only, not in Unsloth's list).",
            ),
            (
                "q4_k_xl",
                "UD-Q4_K_XL (Dynamic 2.0)",
                true,
                Some(4.5),
                "Dynamic 2.0 per-layer selective. Best quality at ~Q4 size. Output: UD-Q4_K_XL.",
            ),
            (
                "q3_k_xl",
                "UD-Q3_K_XL (Dynamic 2.0)",
                true,
                Some(3.5),
                "Dynamic 2.0 per-layer selective, smaller. Output: UD-Q3_K_XL.",
            ),
            (
                "q2_k_xl",
                "UD-Q2_K_XL (Dynamic 2.0)",
                true,
                Some(2.7),
                "Dynamic 2.0 per-layer selective, smallest. Output: UD-Q2_K_XL.",
            ),
        ];
        assert_eq!(METHODS.len(), expected.len());
        for (entry, (id, label, dyn2, bpw, desc)) in METHODS.iter().zip(expected) {
            assert_eq!(entry.method.id, *id, "id mismatch at {id}");
            assert_eq!(entry.method.label, *label, "label mismatch at {id}");
            assert_eq!(
                entry.method.dynamic_v2, *dyn2,
                "dynamic flag mismatch at {id}"
            );
            assert_eq!(entry.method.approx_bpw, *bpw, "bpw mismatch at {id}");
            assert_eq!(
                entry.method.description, *desc,
                "description mismatch at {id}"
            );
        }
    }

    #[test]
    fn every_non_dynamic_method_maps_to_an_encoder_scheme() {
        // Plan 10.3 verification: every non-dynamic ref method maps to a
        // scheme that rlx-gguf 0.2.14 has an encoder for (verified against
        // quantize.rs dispatch: F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/
        // Q2K..Q8K/IQ4NL/IQ4XS/IQ2XXS/IQ2XS/IQ2S/IQ3XXS/IQ3S/IQ1S/IQ1M).
        for e in METHODS.iter().filter(|e| !e.method.dynamic_v2) {
            let schemes: Vec<GgufScheme> = std::iter::once(e.policy.default)
                .chain(std::iter::once(e.policy.embd_scheme))
                .chain(e.policy.rules.iter().map(|(_, s)| *s))
                .collect();
            for s in schemes {
                assert!(
                    matches!(
                        s,
                        GgufScheme::F32
                            | GgufScheme::F16
                            | GgufScheme::Bf16
                            | GgufScheme::Q8_0
                            | GgufScheme::Q4_0
                            | GgufScheme::Q4_1
                            | GgufScheme::Q5_0
                            | GgufScheme::Q5_1
                            | GgufScheme::Q2K
                            | GgufScheme::Q3K
                            | GgufScheme::Q4K
                            | GgufScheme::Q5K
                            | GgufScheme::Q6K
                            | GgufScheme::Q8K
                            | GgufScheme::Iq2Xxs
                            | GgufScheme::Iq2Xs
                            | GgufScheme::Iq3Xxs
                            | GgufScheme::Iq4Nl
                            | GgufScheme::Iq1S
                            | GgufScheme::Iq1M
                            | GgufScheme::Iq2S
                            | GgufScheme::Iq3S
                            | GgufScheme::Iq4Xs
                            | GgufScheme::Tq1_0
                            | GgufScheme::Tq2_0
                            | GgufScheme::Q1_0
                            | GgufScheme::Q2_0
                    ),
                    "method {} uses scheme {:?} without an rlx-gguf encoder",
                    e.method.id,
                    s
                );
            }
        }
    }

    #[test]
    fn dynamic_variants_are_exactly_the_reference_dynamic_set() {
        // q4_1 and q5_1 were removed from this set: official Unsloth lists
        // both as plain legacy types in ALLOWED_QUANTS (save.py:163,170) and
        // llama.cpp gives them ftypes + encoders. The ported quantui reference
        // that tagged them Dynamic 2.0 is wrong (see the entries' notes).
        assert_eq!(
            dynamic_ids(),
            vec!["q4_nl", "q4_k_xl", "q3_k_xl", "q2_k_xl"]
        );
        assert_eq!(supported_ids().len(), METHODS.len() - 4);
    }

    #[test]
    fn q4_1_and_q5_1_are_usable() {
        // Phase 1 lock: these two must stay resolvable, non-dynamic, and
        // must resolve to their own encoders. If this test fails because
        // someone "restored" the dynamic_v2 tag, re-read the Unsloth
        // ALLOWED_QUANTS table before changing anything else.
        for id in ["q4_1", "q5_1"] {
            let e = get_method(id).unwrap_or_else(|| panic!("{id} missing"));
            assert!(!e.method.dynamic_v2, "{id} must not be dynamic_v2");
            assert!(supported_ids().contains(&id), "{id} must be usable");
        }
        assert_eq!(get_method("q4_1").unwrap().policy.default, GgufScheme::Q4_1);
        assert_eq!(get_method("q5_1").unwrap().policy.default, GgufScheme::Q5_1);
    }

    #[test]
    fn every_method_declares_support() {
        // Phase 0.2: every entry carries explicit capability metadata, and
        // today's NoEncoder set is EMPTY — the 2026-08-31 plan removed the
        // last wrongly-assumed members (iq2_m/iq3_m are policy variants,
        // not missing encoders; they don't exist in this registry yet).
        // When Phase 3 adds iq2_m/iq3_m as usable entries, this assertion
        // stays empty-set; if someone adds a NoEncoder method, they must
        // update this test deliberately.
        let no_encoder: Vec<&str> = METHODS
            .iter()
            .filter(|e| matches!(e.method.support, BackendSupport::NoEncoder(_)))
            .map(|e| e.method.id)
            .collect();
        assert_eq!(no_encoder, Vec::<&str>::new());
        for e in METHODS {
            assert!(
                matches!(
                    e.method.support,
                    BackendSupport::Encodable | BackendSupport::NoEncoder(_)
                ),
                "{}: support must be declared",
                e.method.id
            );
        }
    }

    #[test]
    fn usable_ids_equals_supported_ids_while_all_encodable() {
        // Today every non-dynamic entry is encodable, so the two lists agree.
        // The moment a NoEncoder entry lands (plan Phase 2.1), this test
        // must be replaced by an explicit delta assertion.
        assert_eq!(usable_ids(), supported_ids());
        assert_eq!(usable_ids().len(), METHODS.len() - 4);
    }

    #[test]
    fn rejection_reason_matches_dynamic_and_encoder_gaps() {
        assert_eq!(
            rejection_reason("q4_k_xl"),
            Some("Unsloth Dynamic 2.0 per-layer variant (proprietary)")
        );
        assert_eq!(
            rejection_reason("q4_nl"),
            Some("Unsloth Dynamic 2.0 per-layer variant (proprietary)")
        );
        // q4_1/q5_1: reclassified in Phase 1 — must NOT be rejected.
        assert_eq!(rejection_reason("q4_1"), None);
        assert_eq!(rejection_reason("q5_1"), None);
        assert_eq!(rejection_reason("q4_k_m"), None);
        assert_eq!(rejection_reason("bogus"), None); // unknown → unknown-method path
    }

    #[test]
    fn requires_imatrix_matches_iq_prefix() {
        // Unsloth's gate (save.py:2162) is exactly "id in IMATRIX_QUANTS",
        // and every IMATRIX_QUANTS key starts with "iq". Phase 4.2 will
        // enforce the CLI gate; this test pins the metadata contract.
        for e in METHODS {
            assert_eq!(
                e.method.requires_imatrix,
                e.method.id.starts_with("iq"),
                "{}: requires_imatrix must match the iq* prefix rule",
                e.method.id
            );
        }
    }

    #[test]
    fn get_method_lookup() {
        assert_eq!(
            get_method("q4_k_m").unwrap().method.label,
            "Q4_K_M (4-bit, recommended)"
        );
        assert!(get_method("nope").is_none());
        assert!(get_method("Q4_K_M").is_none()); // case-sensitive, like the ref dict
    }

    #[test]
    fn scheme_for_composite_policies() {
        let q4km = get_method("q4_k_m").unwrap();
        // 1-D tensors stay F32 regardless of rules.
        assert_eq!(scheme_for(q4km, "blk.0.attn_norm.weight", 1), F32);
        assert_eq!(scheme_for(q4km, "blk.0.ffn_norm.bias", 1), F32);
        // Embeddings/output use the embd scheme.
        assert_eq!(scheme_for(q4km, "token_embd.weight", 2), F16);
        assert_eq!(scheme_for(q4km, "output.weight", 2), F16);
        // Rule matches.
        assert_eq!(scheme_for(q4km, "blk.3.attn_v.weight", 2), Q6K);
        assert_eq!(scheme_for(q4km, "blk.3.ffn_down.weight", 2), Q6K);
        // Default.
        assert_eq!(scheme_for(q4km, "blk.3.attn_q.weight", 2), Q4K);
        assert_eq!(scheme_for(q4km, "blk.3.ffn_up.weight", 2), Q4K);

        let q3km = get_method("q3_k_m").unwrap();
        assert_eq!(scheme_for(q3km, "blk.1.attn_output.weight", 2), Q5K);
        assert_eq!(scheme_for(q3km, "blk.1.attn_q.weight", 2), Q4K);
        assert_eq!(scheme_for(q3km, "blk.1.ffn_down.weight", 2), Q6K);
        assert_eq!(scheme_for(q3km, "blk.1.ffn_gate.weight", 2), Q3K);

        let q2k = get_method("q2_k").unwrap();
        assert_eq!(scheme_for(q2k, "blk.9.attn_v.weight", 2), Q4K);
        assert_eq!(scheme_for(q2k, "blk.9.attn_k.weight", 2), Q2K);
    }

    #[test]
    fn scheme_for_plain_methods() {
        let f16 = get_method("f16").unwrap();
        assert_eq!(scheme_for(f16, "blk.0.attn_q.weight", 2), F16);
        assert_eq!(scheme_for(f16, "token_embd.weight", 2), F16);
        assert_eq!(scheme_for(f16, "blk.0.attn_norm.weight", 1), F32);

        let q8 = get_method("q8_0").unwrap();
        assert_eq!(scheme_for(q8, "blk.0.ffn_down.weight", 2), GgufScheme::Q8_0);

        let q5ks = get_method("q5_k_s").unwrap();
        assert_eq!(scheme_for(q5ks, "blk.0.attn_v.weight", 2), Q5K); // no rules in _S
    }
}

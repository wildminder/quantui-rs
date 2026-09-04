# quantui-rs

Standalone, single-binary Rust CLI that quantizes Hugging Face `safetensors`
models to **ComfyUI-compatible INT8 / FP8 / MXFP8 / NVFP4** and converts HF
models to **GGUF** (34 usable llama.cpp-compatible methods) — with a hard
contract: **outputs are byte-exact against the Python/torch and llama.cpp
references** on all supported paths.

One static binary. No Python, no torch, no runtime dependencies.

```sh
# Quantize a model to INT8 for ComfyUI
quantui-rs quantize mymodel.safetensors

# Convert a HF model to a GGUF (llama.cpp / unsloth ecosystem)
quantui-rs gguf mymodel.safetensors -m q8_0

# Quantize "aligned with" an existing reference GGUF (e.g. unsloth output)
quantui-rs gguf model.safetensors out.gguf -m q8_0 \
    --recipe-from unsloth-Q8_0.gguf --verify-against unsloth-Q8_0.gguf
```

## Table of contents

1. [Building](#building)
2. [Command overview](#command-overview)
3. [Quantizing to INT8/FP8/MXFP8/NVFP4 (`quantize`)](#quantize)
4. [Converting to GGUF (`gguf`)](#gguf)
5. [Which GGUF method should I use?](#method-selection)
6. [Per-tensor recipes — the open `UD-*`](#recipes)
7. [`--verify-against` — oracle equivalence report](#verify-against)
8. [`--recipe-from` — quantize aligned with a reference](#recipe-from)
9. [Universal model support (multimodal / wrapped checkpoints)](#universal-models)
10. [Validating (`validate`) and inspecting (`info`)](#validate-info)
11. [Exit codes](#exit-codes)
12. [Worked examples: real models](#worked-examples)
13. [Parity contract & known boundaries](#parity)
14. [Performance](#performance)
15. [Repository layout & development](#development)

---

## Building

Requires Rust (stable, ≥ 1.89):

```sh
cargo build --release
# binary: target/release/quantui-rs(.exe)  (~2.6 MB, LTO + stripped)
```

## Command overview

| Command | Purpose |
|---|---|
| `quantize` | safetensors → ComfyUI quantized safetensors (INT8 plain/Hadamard, FP8 E4M3, MXFP8, NVFP4) |
| `gguf` | safetensors → GGUF (34 usable llama.cpp methods, recipes, oracle verification) |
| `validate` | Structural + numeric validation of a quantized output |
| `info` | Inspect a safetensors header without loading tensors |

Both conversion commands accept a single `.safetensors` file **or** a sharded
HF model folder (containing `model.safetensors.index.json`).

---

<a name="quantize"></a>
## Quantizing to INT8/FP8/MXFP8/NVFP4 (`quantize`)

### Quick start

```sh
# 1. Quantize (defaults: INT8, block scaling, block size 128, heuristics on).
#    Output is auto-named next to the input.
quantui-rs quantize mymodel.safetensors
# -> wrote mymodel-int8_block-simple-heur.safetensors (N tensors, config_hash ...)

# 2. Verify the result.
quantui-rs validate mymodel-int8_block-simple-heur.safetensors --numeric
# -> PASS

# 3. Inspect what was produced.
quantui-rs info mymodel-int8_block-simple-heur.safetensors
```

The output is a standard `.safetensors` file that ComfyUI's quantized-model
loaders consume directly: every quantized layer carries a `.comfy_quant`
config blob describing its layout.

### Which format to pick

| `--format` | Bits/weight | Use when |
|---|---|---|
| `int8` (default) | ~8 | The well-trodden path; block/row/tensor scaling; best tooling support |
| `int8_convrot` | ~8 | Hadamard-rotated INT8 (reference `convrot` preset; better accuracy at the same size) |
| `fp8_e4m3` | 8 | Your loader supports FP8; block/row/tensor scaling available |
| `mxfp8` | 8 | MXFP8 with fixed 32-element blocks (swizzled scales); AVOID_KEY_NAMES exclusions apply |
| `nvfp4` | 4 | Maximum compression; NVFP4 with 16-element microblocks + per-tensor scale; AVOID_KEY_NAMES apply |

### Full parameter reference

```
quantui-rs quantize [OPTIONS] <INPUT> [OUTPUT]

  <INPUT>                 .safetensors file OR sharded HF model folder
  [OUTPUT]                .safetensors file (single/merged) or directory
                          (sharded); omitted = auto-named

      --format <FORMAT>   int8 | int8_convrot | fp8_e4m3 | mxfp8 |
                          nvfp4                      [default: int8]
  -m, --scaling-mode <M>  tensor | row | block            [default: block]
                          (INT8/FP8 only; MXFP8/NVFP4 are fixed-block —
                           passing it with those exits 2;
                           int8_convrot forces row, ignoring the flag)
  -b, --block-size <BS>   64 | 128 | 256                  [default: 128]
                          (INT8/FP8 only; MXFP8=32, NVFP4=16 are fixed —
                           passing it with those exits 2)
      --heur / --no-heur  Skip-inefficient-layers heuristic [default: on]
      --exclude-layers <RE>  Regex; matching layers stay full precision
      --output-mode <M>   sharded | single                [default: sharded]
      --orig-dtype <D>    bfloat16 | float16 (skipped-weight cast)
                          [default: bfloat16]
      --calib-seed <N>    Bias-correction seed [default: 233983427]
      --simple            Accepted for reference compatibility (always on)
      --no-progress       Plain, CI-friendly output (no progress bar)
```

**Parameter guidance:**

- **`--scaling-mode block` (default) with `--block-size 128`** is the
  reference default and the best accuracy/size trade-off. Use `row` when a
  layer's dims don't divide the block size and you don't want it skipped;
  `tensor` only when you want maximum compression and can afford the
  accuracy loss.
- **`--heur` (default on)** keeps small/oddly-shaped layers in BF16 instead
  of producing garbage — leave it on unless you have a specific reason.
- **`--exclude-layers`** takes a regex matched against tensor names, e.g.
  `--exclude-layers "norm|embedding"` keeps all norms and embeddings at full
  precision.
- **`--calib-seed`** is pinned to the reference value `233983427`. This
  pinning is what makes runs reproducible and byte-identical to the Python
  reference. Don't change it unless you intentionally want a different (but
  still valid) bias correction.

### What gets quantized

| Tensor | Treatment |
|---|---|
| 2D `*.weight` whose dims are divisible by the block size | **Quantized** |
| 2D `*.weight` not divisible (with default `--heur`) | Copied, cast to `--orig-dtype` (BF16) |
| 2D `*.weight` matching `--exclude-layers` regex | Copied, cast to `--orig-dtype` |
| Everything else (biases, norms, embeddings, …) | Copied unchanged |

Each quantized layer `X` produces four tensors: `X.weight` (quantized),
`X.weight_scale` (dequant scales), `X.comfy_quant` (JSON layout blob for
ComfyUI), and `X.input_scale` (scalar 1.0, INT8 only).

### Resume & Ctrl-C

After **every tensor** the writer flushes the header and a checkpoint file
`<output>.quant-manifest.json`:

- **Kill the process or press Ctrl-C at any time** → exit code 130, and the
  partial output remains valid and loadable.
- **Re-run the same command** → already-done tensors are skipped; the final
  file is byte-identical to an uninterrupted run.
- **Change the config** → the `config_hash` no longer matches and the run
  restarts cleanly (prevents corrupt mixed-config files).

### Sharded HF models

```sh
# One output shard per input shard (sharding preserved) — default
quantui-rs quantize ./my-hf-model ./my-hf-model-int8

# Merge all shards into ONE output file
quantui-rs quantize ./my-hf-model --output-mode single
```

### INT8 details (scaling modes, convrot)

`-m/--scaling-mode` controls scale granularity:

| Mode | Scale shape (256×128 layer, bs=128) | Notes |
|---|---|---|
| `block` (default) | `[2, 1]` — one scale per bs×bs tile | Best trade-off; needs divisible dims |
| `row` | `[256, 1]` — one scale per output row | No divisibility requirement |
| `tensor` | `[]` scalar | Coarsest |

`--format int8_convrot` is a preset: **row-wise INT8 + group-wise Hadamard
rotation** (`convrot_group_size=256`). Layers with `in_features % 256 == 0`
are rotated (blob carries `{"convrot": true, "convrot_groupsize": 256,
"per_row": true}`); others fall back to plain row INT8. Byte-exact vs the
reference's batch path (8 golden cases at group sizes 256 and 64).

### FP8 / MXFP8 / NVFP4 differences

```sh
quantui-rs quantize mymodel.safetensors --format fp8_e4m3
quantui-rs quantize mymodel.safetensors --format mxfp8
quantui-rs quantize mymodel.safetensors --format nvfp4
```

| | `int8` | `fp8_e4m3` | `mxfp8` | `nvfp4` |
|---|---|---|---|---|
| `X.weight` dtype | `I8` | `F8_E4M3` | `F8_E4M3` | `U8` (2×E2M1 packed/byte) |
| `X.weight_scale` | `F32` | `F32` | `U8` (e8m0) | `F8_E4M3` |
| Scale layout | per mode | per mode | swizzled `[256,4]` | swizzled `[256,8]` |
| `X.weight_scale_2` | — | — | — | `F32` scalar |
| Scaling modes | tensor/row/block | tensor/row/block | fixed `block` | fixed `block` |
| Block size | 64/128/256 | 64/128/256 | fixed 32 | fixed 16 |
| AVOID_KEY_NAMES exclusions | no | no | **yes** | **yes** |
| `__metadata__` on output | no | no | **yes** | **yes** |

MXFP8/NVFP4 scales use ctq's *swizzled* (blocked) layout; NVFP4 additionally
carries a per-tensor `weight_scale_2`. MXFP8/NVFP4 outputs carry a
`__metadata__._quantization_metadata` JSON listing every quantized layer.
Resume, Ctrl-C, sharded input, `validate`, and `info` work identically for
all formats.

---

<a name="gguf"></a>
## Converting to GGUF (`gguf`)

### Quick start

```sh
quantui-rs gguf mymodel.safetensors                 # q4_k_m (default)
quantui-rs gguf ./my-hf-model -m q8_0 out.gguf      # sharded folder input
quantui-rs gguf mymodel.safetensors -m iq4_xs --imatrix imatrix.dat
quantui-rs gguf --list-methods                      # show all methods
```

Output is auto-named `<base>-<method>.gguf` next to the input when `[OUTPUT]`
is omitted. **Both paths are positional** — there are no `--input`/`--output`
flags.

### Full parameter reference

```
quantui-rs gguf [OPTIONS] [INPUT] [OUTPUT]

  [INPUT]    single .safetensors file OR sharded HF model folder
  [OUTPUT]   .gguf file (omitted = auto-named <base>-<method>.gguf)

  -m, --method <METHOD>          GGUF method id [default: q4_k_m]
      --imatrix <PATH>           Importance matrix (GGUF or legacy binary).
                                 REQUIRED for every iq* method (exit 2 without)
      --tensor-type-file <PATH>  Per-tensor recipe file (see Recipes)
      --token-embedding-type <METHOD>   Override token_embd.weight quant
      --output-tensor-type <METHOD>     Override output.weight quant
      --emit-recipe <PATH>       Dump the effective per-tensor assignment
      --verify-against <REF.GGUF>  Oracle equivalence report after conversion
      --recipe-from <REF.GGUF>   Extract + apply a reference's dtype assignment
      --arch <ARCH>              Override the GGUF arch string
                                 (else detected from config.json)
      --name <NAME>              Override general.name metadata
      --list-methods             List all methods and exit
      --no-progress              Plain, CI-friendly output
```

### What the converter does

- **Architecture metadata**: detected from `config.json`
  (`model_type`/`architectures`) and written as GGUF
  `general.architecture` + `<arch>.*` numeric metadata. Known arch classes
  include llama/mistral/mixtral, qwen2/3, gemma, phi, gpt2, bloom, falcon,
  stablelm and **lfm2** (`LFM2ForCausalLM` / `Lfm2ForCausalLM` /
  `LFM2VLForConditionalGeneration` → `lfm2`). Override with `--arch` (needed
  when converting a bare safetensors file with no `config.json` next to it).
- **Tensor naming**: HF → llama.cpp GGUF names for the dense llama family
  (`model.layers.0.self_attn.q_proj.weight` → `blk.0.attn_q.weight`), plus
  generic nested-prefix handling and LFM2/LFM2.5 cores — see
  [Universal model support](#universal-models). Names with no known mapping
  pass through **unchanged** — nothing is ever guessed.
- **Per-tensor policy**: composite methods apply llama-quantize rules
  (e.g. `q4_k_m` uses Q6_K for key/output tensors, Q4_K elsewhere); 1-D
  tensors → F32 by the shared convention.
- **Spec conformance**: tensors whose row width (`ne[0]`) isn't divisible by
  the chosen scheme's block size are demoted exactly like llama-quantize's
  `tensor_type_fallback` (IQ\*→IQ4_NL, Q2_K/Q3_K/TQ\*→Q4_0, Q4_K→Q5_0,
  Q5_K→Q5_1, Q6_K→Q8_0, else F16) — loudly, with a per-tensor warning and a
  summary. The output is always a spec-conformant GGUF.

### Method capability matrix

All 38 registry methods (Unsloth's list plus llama.cpp-only `q1_0`/`q2_0`):

| Method | Class | Parity |
|---|---|---|
| `f16`, `bf16`, `f32` | usable, no imatrix | byte-exact vs gguf-py writer; `f32` lossless passthrough |
| `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1` | usable, no imatrix | byte-exact vs gguf-py writer (49/49 golden cases) |
| `q2_k`, `q2_k_l`, `q3_k_*` (`m/l/s/xs`), `q4_k_*` (`m/s`), `q5_k_*` (`m/s`), `q6_k` | usable, no imatrix | byte-exact vs llama-quantize (weighted, `--imatrix` goldens); no-imatrix path locked by equivalence test |
| `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq2_m`, `iq3_xxs`, `iq3_s`, `iq3_m`, `iq4_nl`, `iq4_xs` | usable, **requires `--imatrix`** | byte-exact vs llama-quantize (weighted, `--imatrix` goldens) |
| `tq1_0`, `tq2_0`, `q1_0`, `q2_0` | usable, no imatrix | structural (encode + bounded round-trip) |
| `q4_nl`, `q4_k_xl`, `q3_k_xl`, `q2_k_xl` | **rejected** (exit 2) | Unsloth Dynamic 2.0 — proprietary heuristic |

**The imatrix caveat**: every `iq*` method requires `--imatrix <PATH>`.
Without it, quantization produces garbage — the same conversion
llama-quantize refuses. K-quants consume one when supplied (optional for
them). Files load in GGUF or llama-quantize's legacy binary format.

---

<a name="method-selection"></a>
## Which GGUF method should I use?

| Goal | Method | Notes |
|---|---|---|
| Best overall size/quality (4-bit) | `q4_k_m` | The default; near-lossless key tensors |
| Near-lossless | `q8_0` or `q6_k` | `q8_0` is the safest; `q6_k` smaller |
| Smallest usable | `q4_k_s` → `q3_k_m` → `q2_k_l` | Test quality as you go down |
| Extreme compression (needs imatrix) | `iq4_xs` → `iq3_m` → `iq2_m` | imatrix quality matters a lot here |
| BitNet-class ternary models | `tq1_0` / `tq2_0` | For models trained ternary |
| Debugging / reference | `f16`, `bf16`, `f32` | Lossless; use as intermediates |

To generate an imatrix, use llama.cpp's `llama-imatrix` (or unsloth's
pipeline) on calibration text; both the GGUF and legacy binary imatrix
formats load.

---

<a name="recipes"></a>
## Per-tensor recipes — the open `UD-*`

Unsloth's Dynamic 2.0 presets (`UD-Q4_K_XL` etc.) are a proprietary
per-layer bit-width heuristic and are rejected with exit 2. `--tensor-type-file`
is the open equivalent: any per-tensor assignment expressible as a
first-match-wins regex list, with your choices explicit and inspectable.

### Recipe file format (`--tensor-type-file <PATH>`)

```text
# lines starting with '#' are comments; blank lines are ignored

# regex=qtype — first matching rule wins (llama-quant.cpp:716-726 semantics)
blk\.[0-9]+\.ffn_down\.weight=q6_k
attn_v\.weight=q5_k_s

# a trailing BARE qtype is the default for tensors no rule matches
q4_k_m
```

- The regex uses **search** (unanchored substring) semantics against the
  **GGUF-side** tensor name (`blk.0.attn_q.weight` — after HF→GGUF mapping).
- Every `qtype` must be a usable method id; a typo exits 2 with the line
  number.
- `--token-embedding-type` / `--output-tensor-type` sit **before** the
  recipe (llama-quantize's early-return overrides), except that a recipe
  rule explicitly naming `per_layer_token_embd` still wins for it.
- Recipes only apply when the method's default type is quantized — an
  `f16`/`f32`/`bf16` method ignores the recipe entirely (upstream parity).
- The row-width demotion guard still applies **on top** of any recipe rule.

### Dumping the effective assignment

```sh
quantui-rs gguf model.safetensors out.gguf -m q4_k_m \
    --tensor-type-file my.recipe --emit-recipe effective.recipe
```

`effective.recipe` carries one `^name$=qtype` line per tensor in conversion
order; feeding it back via `--tensor-type-file` reproduces the assignment
exactly.

---

<a name="verify-against"></a>
## `--verify-against` — oracle equivalence report

Compare your freshly converted GGUF against a reference GGUF (unsloth
output, llama-quantize output, or a previous run of your own):

```sh
quantui-rs gguf model.safetensors out.gguf -m q8_0 \
    --verify-against unsloth-Q8_0.gguf
```

The report (stdout):

```
verify-against: out.gguf vs unsloth-Q8_0.gguf
  byte-exact: 140 | numerically-equivalent: 26 | divergent: 0
  ! blk.1.ffn_gate.weight (dead-block cosmetic)  blocks 128/688128 differ, max|Δ|=0.0e0 [DeadBlockCosmetic]
  ~ dtype mismatch: token_embd.weight: ours=F16 ref=Q8_0
  ? ours-only (unmatched after name mapping): 441
  ? reference-only: 0
  (skipped 99 F32 tensors)
  spec-conformance: OK (0 violations)
```

**Diff classification** — what each line means:

| Label | Meaning | Action |
|---|---|---|
| `DeadBlockCosmetic` | Every differing block has scale **0 on both sides** (denormal dead channels). The int codes differ, but reconstruction is bit-identical (`q·0 = 0`). | None — cosmetic |
| `ScaleRuleDiff` | Blocks differ in scale (e.g. unsloth's MSE-tuned Q4_0 scale vs llama.cpp's `max/-8`). Bounded reconstruction difference. | None — different but valid rules |
| `GenuineDivergence` | Same scale, differing codes — the quantizers disagree on the math. | **Bug signal** — please report |
| `dtype mismatch` | The two files assigned different types to a tensor. | Informational (often a deliberate convention) |
| `ours-only` / `reference-only` | Tensors with no counterpart after name mapping (e.g. vision-tower tensors the reference keeps in a separate mmproj file). | Informational |

The report also runs a **spec-conformance scan of YOUR file** (the
`gguf.cpp:724` rule: every quantized tensor must have `ne[0] % block_size
== 0`).

**Exit codes**: `0` normal report (diffs allowed), `3` your file has spec
violations, `1` the reference can't be parsed.

---

<a name="recipe-from"></a>
## `--recipe-from` — quantize aligned with a reference

Extract the per-tensor dtype assignment from any existing GGUF and apply it
to your conversion:

```sh
quantui-rs gguf model.safetensors out.gguf -m q8_0 \
    --recipe-from unsloth-Q8_0.gguf \
    --verify-against unsloth-Q8_0.gguf
```

- One **exact-name** rule (`^name$=qtype`) per reference tensor; dotted
  names are regex-escaped, so a rule can never fuzzy-match a neighbor.
- Reference tensors absent from your input are ignored; your input tensors
  absent from the reference keep the method default.
- F32 entries are skipped (1-D norms/biases go to F32 by the shared
  convention anyway).
- The extracted rules feed the same machinery as `--tensor-type-file`, so
  the row-width demotion guard still applies, and `--emit-recipe` still
  dumps the effective assignment.

This is the open implementation of the "quantize aligned with unsloth"
workflow: the reference's *recipe* is honored, while the quantization math
remains llama.cpp-compatible (a single ground truth — never a second one).
Verified on LFM2.5-VL-3B: with `--recipe-from` + `--verify-against`, every
shared quantized tensor is byte-exact or dead-block-cosmetic, 0 divergent,
0 spec violations.

---

<a name="universal-models"></a>
## Universal model support (multimodal / wrapped checkpoints)

The naming layer is **generic**, not per-model: it handles the two layouts
multimodal checkpoints actually use, verified against real models
(VibeVoice-1.5B, LFM2.5-VL-3B).

**1. Nested language-model wrappers.** VibeVoice and LFM2-VL nest the dense
LM one level below the top-level module. Both wrappers are stripped
automatically:

| HF name (input) | GGUF name (output) |
|---|---|
| `model.language_model.layers.3.self_attn.q_proj.weight` | `blk.3.attn_q.weight` |
| `model.model.layers.1.self_attn.o_proj.weight` | `blk.1.attn_output.weight` |
| `model.language_model.embed_tokens.weight` | `token_embd.weight` |
| `model.language_model.norm.weight` | `output_norm.weight` |

**2. LFM2 / LFM2.5 layer cores** (verified against the llama.cpp reference
`tensor_mapping.py` and real unsloth LFM2 GGUFs):

| HF name | GGUF name |
|---|---|
| `…layers.0.conv.conv.weight` | `blk.0.shortconv.conv.weight` |
| `…layers.0.conv.in_proj.weight` | `blk.0.shortconv.in_proj.weight` |
| `…layers.0.conv.out_proj.weight` | `blk.0.shortconv.out_proj.weight` |
| `…layers.0.feed_forward.w1.weight` | `blk.0.ffn_gate.weight` |
| `…layers.0.feed_forward.w2.weight` | `blk.0.ffn_down.weight` |
| `…layers.0.feed_forward.w3.weight` | `blk.0.ffn_up.weight` |
| `…layers.0.operator_norm.weight` | `blk.0.attn_norm.weight` |
| `…layers.0.ffn_norm.weight` | `blk.0.ffn_norm.weight` |
| `…self_attn.out_proj.weight` (LFM2.5 attention layers) | `blk.N.attn_output.weight` |
| `…self_attn.q_layernorm.weight` | `blk.N.attn_q_norm.weight` |
| `…self_attn.k_layernorm.weight` | `blk.N.attn_k_norm.weight` |
| `model.embedding_norm.weight` | `token_embd_norm.weight` |

**3. Everything else passes through unchanged** — vision towers
(`model.vision_tower.*`), projectors (`model.multi_modal_projector.*`),
conv tokenizer heads (`model.acoustic_tokenizer.*`, …). Unmapped names are
never guessed.

**4. Arch detection**: `LFM2ForCausalLM`, `Lfm2ForCausalLM`,
`LFM2VLForConditionalGeneration` → GGUF arch `lfm2` (llama.cpp
`LLM_ARCH_LFM2`), via `architectures` or `model_type` in `config.json`.
Converting a bare `.safetensors` without a neighboring `config.json`?
Pass `--arch lfm2` explicitly.

> **Note**: multimodal checkpoints quantize correctly, but the vision tower
> stays in the main GGUF with its original names. For a llama.cpp-loadable
> split you'd extract the mmproj separately (as unsloth does) — the
> converter's contract is spec-conformant GGUF output, not mmproj splitting.

---

<a name="validate-info"></a>
## `validate` and `info`

### validate — check a quantized output

```sh
quantui-rs validate out.safetensors            # structural checks (headers only)
quantui-rs validate out.safetensors --numeric  # + read tensor payloads
quantui-rs validate ./sharded-output-dir/      # directories expand to shards
```

Covers all 7 ctq formats (INT8 tensor/row/block, FP8, MXFP8, NVFP4):
per-layer layout, scale shapes, blob field types, orphan detection. `--numeric`
additionally checks INT8 weight bounds (±127, no −128), scale
finiteness/positivity, E4M3/e8m0 NaN handling, and `input_scale == 1.0`.

```
file: output.safetensors (0 GB)
quantized matrices      : 1
  GS128                : 1
full-precision weights  : 2
quantized share         : 79.50%
formats found           : int8_blockwise

PASS
```

### info — inspect a header

```sh
quantui-rs info model.safetensors         # per-tensor table + format detection
quantui-rs info model.safetensors --raw   # raw JSON header
```

Parses only the header (no tensor payloads).

---

<a name="exit-codes"></a>
## Exit codes (all commands)

| Code | Meaning |
|---|---|
| 0 | Success (validate: all checks passed; gguf: report printed, diffs allowed) |
| 1 | Runtime failure (I/O error, invalid file, validate found issues, unparseable `--verify-against` reference) |
| 2 | Usage error (bad arguments, unknown GGUF method, missing input, missing `--imatrix`, bad recipe) |
| 3 | **gguf only**: `--verify-against` found spec violations in OUR output |
| 130 | Cancelled by Ctrl-C (partial output is resumable) |

---

<a name="worked-examples"></a>
## Worked examples: real models

### LFM2.5-VL-3B → Q8_0, aligned with unsloth

```sh
# The source is a bare multimodal safetensors (no config.json in the folder):
# pass the arch explicitly. Recipe from unsloth's Q8_0, verified against it.
quantui-rs gguf LFM2.5-VL-3B-original.safetensors LFM2.5-VL-3B-Q8_0-ours.gguf \
    -m q8_0 --arch lfm2 \
    --recipe-from LFM2.5-VL-3B-Q8_0-unsloth.gguf \
    --verify-against LFM2.5-VL-3B-Q8_0-unsloth.gguf
```

Result measured on the real model: 167 rules extracted; 141 byte-exact
shared tensors, 26 dead-block-cosmetic diffs (max\|Δ\| = 0.0 — reconstruction
identical), 0 divergent, 0 spec violations. The 441 `ours-only` tensors are
exactly the vision tower + projector, which unsloth keeps in a separate
`mmproj-BF16.gguf`.

The odd-shaped LFM2 shortconv kernels (`ne[0] = 3`) are demoted to F16
loudly — llama-quantize's own behavior for rows no quantized block can
describe.

### VibeVoice-1.5B → Q8_0

```sh
quantui-rs gguf VibeVoice-1.5B-bf16.safetensors VibeVoice-1.5B-q8_0.gguf -m q8_0
```

The LM backbone (`model.language_model.*`) maps to `blk.*` /
`token_embd.weight` / `output_norm.weight`; the acoustic/semantic tokenizer
heads and prediction head pass through under their original names. The 102
conv kernels (row widths 4/7/8/10/16) are demoted to F16 with loud warnings
— the output is always spec-conformant.

---

<a name="parity"></a>
## Parity contract & known boundaries

**Byte-exact (golden-verified):**

- INT8 streaming quantization: tensor/row/block scaling, skip heuristics,
  skipped-weight dtype casting, bias correction (bit-exact torch MT19937 +
  randn + oneDNN-style sgemm + cascade-sum ports), comfy_quant blobs, header
  8-byte alignment, manifest checkpoints — single-file and sharded, both
  output modes. Verified **whole-file** byte-for-byte.
- FP8 (tensor/row/block), MXFP8 and NVFP4 streaming — verified **per-tensor**
  (payload bytes + `dtype`/`shape` + metadata string). Whole-file equality
  is structurally impossible here: ctq writes tensors in its own processing
  order with sorted header keys, the streaming orchestrator writes in input
  order with a 64 KiB header slot. Per-tensor parity is the same guarantee
  at the level that matters.
- Calibration draw order (INT8: file order; ctq formats: alphabetical —
  ctq builds its key list from `safe_open.keys()`, which is sorted) — locked
  in both directions by `tests/golden/sharded_unsorted`.
- GGUF legacy encoders F16/BF16/F32/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1 — byte-identical
  to gguf-py (49/49 golden cases).
- GGUF weighted K-quant and IQ\* row quantizers (Q4_K/Q2_K/Q3_K/Q5_K/Q6_K,
  iq1_s/m, iq2_xxs/xs/s/m, iq3_xxs/s/m, iq4_nl/xs) — byte-identical to the
  real `llama-quantize --imatrix` on shared fixtures, at both the unit and
  the full-driver e2e tier (14/14 each).
- INT8 row scaling reproduces torch's exact float semantics — including
  `127.0/row_max` computed as `127.0 * reciprocal(row_max)` (torch's
  double-rounding quirk) — locked by the `int8_convrot` goldens.

**Correctness ladder for GGUF output** (what "correct" means here):

1. **Spec-conformant** — always; every output passes the per-row
   `ne[0] % block_size` rule, enforced at type-selection time exactly like
   llama-quantize.
2. **Equivalent** — on demand via `--verify-against`: byte-exact or
   numerically-identical-after-dequant vs any reference.
3. **Byte-exact** — vs real `llama-quantize` goldens (the sole ground
   truth). Unsloth is an oracle input (recipes, imatrix), never a byte-parity
   target.

**Documented boundaries (out of scope for v1):**

- GGUF ternary (`tq1_0`/`tq2_0`) and 1/2-bit (`q1_0`/`q2_0`) encoders are
  structurally verified but have no external byte-parity reference.
- GGUF K-quant byte-parity goldens are generated **with** an imatrix; the
  no-imatrix path is locked by a `None ≡ uniform-1.0` equivalence test.
- Learned-rounding optimizers (AdamW/RAdam/Prodigy) remain Python-only.
- Unsloth Dynamic 2.0 per-layer mixing is proprietary and not replicated
  (`--tensor-type-file` / `--recipe-from` are the open equivalents).
- W4A4/W4A8 layouts are deferred to v2. mmproj extraction (separating the
  vision tower into its own GGUF) is not performed — multimodal tensors stay
  in the main file under their original names.

---

<a name="performance"></a>
## Performance

Measured on a 24-core Windows box, 1.004 GiB fixture (32×4096×4096 bf16),
vs the Python reference streaming path:

| Metric | Rust | Python ref | Ratio |
|---|---|---|---|
| End-to-end 1 GiB streaming | 4.10 s (250 MiB/s) | 4.80 s (214 MiB/s) | 1.17× |
| CPU-bound quant kernel (4096×4096) | 68.6 ms (933 MiB/s) | 127.5 ms (245 MiB/s) | 1.86× |

The byte-exact contract rules out numeric shortcuts that would widen the
gap — parity was prioritized over raw speed. Reproduce with
`cargo bench -p quant-core` and `tools/bench_python_ref.py`.

---

<a name="development"></a>
## Repository layout & development

```
crates/quant-core/     library: safetensors IO, INT8/FP8/MXFP8/NVFP4 kernels,
                       streaming orchestrator, bias correction, torch-RNG port,
                       comfy_quant schema, validator, GGUF registry + converter,
                       gguf_verify (oracle report), gguf_recipe (per-tensor recipes)
crates/quant-cli/      binary `quantui-rs`: clap CLI, progress, profiles
tests/golden/          Python/torch-generated golden fixtures (byte-parity refs)
tools/                 golden + benchmark generators, GGUF diagnostics (Python, ctq venv)
docs/plans/            design plans + execution log (local-only, not git-tracked)
```

```sh
cargo test --workspace                 # 478 tests incl. golden byte-parity
cargo clippy --workspace --all-targets # clean with -D warnings
cargo fmt --check
cargo bench -p quant-core              # throughput benchmarks (needs fixture)
```

CI (`.github/workflows/ci.yml`) runs fmt + clippy + full test suite + release
build on Windows and Linux. Golden fixtures are committed and marked binary so
byte-compare tests are valid on every OS.

### GGUF tooling scripts (`tools/`)

| Script | Purpose |
|---|---|
| `cmp_ours_vs_unsloth.py` | Byte-compare our GGUF vs unsloth's (the logic now productized as `--verify-against`) |
| `diag_vibevoice_q8.py` | Spec-violation scanner (`ne[0] % blck` audit; now built into `--verify-against`) |
| `probe_block_stats.py` | Per-block Q8_0 scale/code diff analysis |
| `inspect_lfm.py` | Inventory a model's safetensors + GGUFs |
| `gen_golden_gguf.py` / `gen_golden_llamacpp_weighted.py` | Golden fixture generation |
| `sweep_gguf_e2e.sh` | Convert + validate a fixture with EVERY usable GGUF method (49 checks) |

## License

MIT

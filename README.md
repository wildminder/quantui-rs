<a id="readme-top"></a>

# ⟪ quantui-rs ⟫

**A single-binary Rust CLI that quantizes, converts, and casts Hugging Face
`safetensors` models** — to ComfyUI-compatible INT8/FP8/MXFP8/NVFP4, to GGUF for
the llama.cpp ecosystem, or to a single-file bf16/fp16/fp32 `.safetensors`.

One static binary. No Python, no torch, no runtime dependencies.

**The contract:** outputs are **byte-exact against the Python/torch and
llama.cpp references** on all default paths. The handful of opt-in formats that
trade that guarantee for accuracy say so on every run
([`parity:` marker](#conformance)).

[![Rust][rust-shield]][rust-url]
[![License MIT][mit-shield]][license-url]
[![Platform][platform-shield]][building-url]
[![Parity][parity-shield]][conformance]
[![Stars][stars-shield]][stars-url]
[![Last commit][commit-shield]][commit-url]

```sh
# Quantize a model to INT8 for ComfyUI
quantui-rs quantize mymodel.safetensors

# Convert a HF model to a GGUF (llama.cpp / unsloth ecosystem)
quantui-rs gguf mymodel.safetensors -m q8_0

# Merge a sharded HF model into ONE single-file bf16 .safetensors
quantui-rs cast ./my-hf-model --to bf16

# Quantize "aligned with" an existing reference GGUF (e.g. unsloth output)
quantui-rs gguf model.safetensors out.gguf -m q8_0 \
    --recipe-from unsloth-Q8_0.gguf --verify-against unsloth-Q8_0.gguf
```

<details>
<summary><b>Table of Contents</b></summary>

- [Building](#building)
- [Command overview](#command-overview)
- [Quantizing to INT8/FP8/MXFP8/NVFP4 (`quantize`)](#quantize)
- [Format &amp; parameter matrix](#format-matrix)
- [Quality modes — `nvfp4_l2` and `nvfp4_rot16`](#quality-modes)
- [Format conformance &amp; the `parity:` marker](#conformance)
- [Converting to GGUF (`gguf`)](#gguf)
- [Which GGUF method should I use?](#method-selection)
- [Per-tensor recipes — the open `UD-*`](#recipes)
- [`--verify-against` — oracle equivalence report](#verify-against)
- [`--recipe-from` — quantize aligned with a reference](#recipe-from)
- [Universal model support (multimodal / wrapped checkpoints)](#universal-models)
- [Validating (`validate`) and inspecting (`info`)](#validate-info)
- [Casting to a single-file bf16/fp16 model (`cast`)](#cast)
- [Exit codes](#exit-codes)
- [Shell completions](#completions)
- [Worked examples: real models](#worked-examples)
- [Parity contract &amp; known boundaries](#parity)
- [Performance](#performance)
- [Development, benchmarks &amp; tooling](#development) — see [DEVELOPMENT.md](DEVELOPMENT.md)
- [License](#license)

</details>

<p id="building" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Building

Requires Rust (stable, ≥ 1.89):

```sh
cargo build --release
# binary: target/release/quantui-rs(.exe)  (~2.6 MB, LTO + stripped)
```

<p id="command-overview" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Command overview

| Command | Purpose |
|---|---|
| `quantize` | safetensors → ComfyUI quantized safetensors (INT8 plain/Hadamard, FP8 E4M3, MXFP8, NVFP4) |
| `gguf` | safetensors → GGUF (34 usable llama.cpp methods, recipes, oracle verification) |
| `cast` | safetensors (single **or** sharded) → one single-file bf16/fp16/fp32 `.safetensors`. **No quantization** |
| `validate` | Structural + numeric validation of a quantized output |
| `info` | Inspect a safetensors header without loading tensors |

`quantize`, `gguf` and `cast` accept a single `.safetensors` file **or** a
sharded HF model folder (containing `model.safetensors.index.json`).

<p id="quantize" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Quantizing to INT8/FP8/MXFP8/NVFP4 (`quantize`)

### ▸ Quick start

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

### ▸ Which format to pick

| `--format` | Bits/weight | Use when |
|---|---|---|
| `int8` (default) | ~8 | The well-trodden path; block/row/tensor scaling; best tooling support |
| `int8_convrot` | ~8 | Hadamard-rotated INT8 (reference `convrot` preset; better accuracy at the same size) |
| `fp8_e4m3` | 8 | Your loader supports FP8; block/row/tensor scaling available |
| `mxfp8` | 8 | MXFP8 with fixed 32-element blocks (swizzled scales); AVOID_KEY_NAMES exclusions apply |
| `nvfp4` | 4 | Maximum compression; NVFP4 with 16-element microblocks + per-tensor scale; AVOID_KEY_NAMES apply |
| `nvfp4_l2` | 4 | **Quality mode.** Same 4-bit size, ~17.7% lower reconstruction error — but **NOT byte-exact**. See [Quality modes](#quality-modes) |
| `nvfp4_rot16` | 4 | NVFP4 + Hadamard rotation at group size 16. **Byte-exact**, but see the inverse-rotation caveat before using it |
| `int8_clip09` | ~8 | ⚠️ **A measured negative result, kept only so the result stays reproducible.** It regresses 31×–1.4e4× in weight-space L2. Do not use it — see [Tried, measured, rejected](DEVELOPMENT.md#rejected) |

Everything down to `nvfp4` is byte-exact against the Python/torch reference.
The last three are opt-in and differ in kind: `nvfp4_l2` trades the byte-exactness
guarantee for accuracy, `nvfp4_rot16` keeps the guarantee (a rotation is a
parity-exact transform) but carries an end-to-end caveat, and `int8_clip09` is
an opt-in that is **known to be worse**. Every run states which kind it was on
the [`parity:` line](#conformance).

> [!WARNING]
> **`int8_clip09` is known to be worse, not better.** It regresses 31×–1.4e4×
> in weight-space L2. It is kept only so the measured negative result stays
> reproducible — do not use it in production. Details in
> [Tried, measured, rejected](DEVELOPMENT.md#rejected).

<a id="format-matrix"></a>

### ▸ Format & parameter matrix

Every combination the CLI accepts, and exactly what it emits. `bpw` is the
**measured** on-disk cost including scales, computed for a 4096×4096 layer —
scales are real bytes, not free, and at 4-bit they are a third of the file.

| `--format` | Scaling mode | Block size | `X.weight` | `X.weight_scale` | Extra tensors | bpw | Parity |
|---|---|---|---|---|---|---|---|
| `int8` | `block` | 64 | `I8` | `F32` `[r/64, c/64]` | `input_scale` | 8.008 | `exact` |
| `int8` | `block` | **128** (default) | `I8` | `F32` `[r/128, c/128]` | `input_scale` | 8.002 | `exact` |
| `int8` | `block` | 256 | `I8` | `F32` `[r/256, c/256]` | `input_scale` | 8.000 | `exact` |
| `int8` | `row` | — | `I8` | `F32` `[r, 1]` | `input_scale` | 8.008 | `exact` |
| `int8` | `tensor` | — | `I8` | `F32` scalar | `input_scale` | 8.000 | `exact` |
| `int8_convrot` | `row` (forced) | 256 (group) | `I8` | `F32` `[r, 1]` | `input_scale` | 8.008 | `exact` |
| `int8_clip09` | `row` | — | `I8` | `F32` `[r, 1]` | `input_scale` | 8.008 | ⚠️ measured-worse |
| `fp8_e4m3` | `block` | 64 | `F8_E4M3` | `F32` `[r/64, c/64]` | — | 8.008 | `exact` |
| `fp8_e4m3` | `block` | **128** (default) | `F8_E4M3` | `F32` `[r/128, c/128]` | — | 8.002 | `exact` |
| `fp8_e4m3` | `block` | 256 | `F8_E4M3` | `F32` `[r/256, c/256]` | — | 8.000 | `exact` |
| `fp8_e4m3` | `row` | — | `F8_E4M3` | `F32` `[r, 1]` | — | 8.008 | `exact` |
| `fp8_e4m3` | `tensor` | — | `F8_E4M3` | `F32` scalar | — | 8.000 | `exact` |
| `mxfp8` | `block` (fixed) | 32 (fixed) | `F8_E4M3` | `U8` e8m0, swizzled `[256,4]` | — | **8.250** | `exact` |
| `nvfp4` | `block` (fixed) | 16 (fixed) | `U8` (2×E2M1/byte) | `F8_E4M3`, swizzled `[256,8]` | `weight_scale_2` `F32` | **4.500** | `exact` |
| `nvfp4_l2` | `block` (fixed) | 16 (fixed) | `U8` (2×E2M1/byte) | `F8_E4M3`, swizzled `[256,8]` | `weight_scale_2` `F32` | **4.500** | `quality-tuned` |
| `nvfp4_rot16` | `block` (fixed) | 16 (fixed) + rotation 16 | `U8` (2×E2M1/byte) | `F8_E4M3`, swizzled `[256,8]` | `weight_scale_2` `F32` | **4.500** | `exact` |

**Reading the numbers.** INT8 and FP8 land at ~8.00 bpw — the F32 scales are
noise at that width. The 4-bit formats are where scales stop being noise:
MXFP8's 1-byte-per-32 E8M0 scale adds **0.25 bpw** (→ 8.25), and NVFP4's
1-byte-per-16 E4M3 scale adds **0.5 bpw** (→ 4.5). A "4-bit" NVFP4 file is
really 4.5.

**Fixed-block formats reject the block flags.** Passing `-m` or `-b` with
`mxfp8`/`nvfp4` **exits 2** — those block sizes are part of the format
definition, not a tuning knob. `int8_convrot` forces `row` and ignores `-m`.

**Which tensors are affected.** Only 2D `*.weight` whose dims satisfy the block
size are quantized; everything else is copied at `--orig-dtype`. `mxfp8` and
`nvfp4` additionally apply `AVOID_KEY_NAMES` exclusions, so a nominally 4-bit
model is partly full-precision in practice — run `info` on the output to see
the real quantized share.

### ▸ Full parameter reference

```
quantui-rs quantize [OPTIONS] <INPUT> [OUTPUT]

  <INPUT>                 .safetensors file OR sharded HF model folder
  [OUTPUT]                .safetensors file (single/merged) or directory
                          (sharded); omitted = auto-named

      --format <FORMAT>   int8 | int8_convrot | fp8_e4m3 | mxfp8 |
                          nvfp4 | nvfp4_l2 | nvfp4_rot16 | int8_clip09
                                             [default: int8]
                          (nvfp4_l2 and int8_clip09 are NOT byte-exact —
                           deliberate quality trades; nvfp4_rot16 IS
                           byte-exact. See "Quality modes".)
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
      --verify-output     Re-parse the output header(s) after the run;
                          exit 1 if any reported tensor is missing
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
- **`--verify-output`** is a cheap post-run integrity check: the output
  header(s) are re-parsed and every reported tensor must be present, or
  the run exits 1 (`error: output verification failed: …`). Off by
  default — on a 10 GB output the header read is cheap but not free.

### ▸ What gets quantized

| Tensor | Treatment |
|---|---|
| 2D `*.weight` whose dims are divisible by the block size | **Quantized** |
| 2D `*.weight` not divisible (with default `--heur`) | Copied, cast to `--orig-dtype` (BF16) |
| 2D `*.weight` matching `--exclude-layers` regex | Copied, cast to `--orig-dtype` |
| Everything else (biases, norms, embeddings, …) | Copied unchanged |

Each quantized layer `X` produces four tensors: `X.weight` (quantized),
`X.weight_scale` (dequant scales), `X.comfy_quant` (JSON layout blob for
ComfyUI), and `X.input_scale` (scalar 1.0, INT8 only).

### ▸ Sharded HF models

```sh
# One output shard per input shard (sharding preserved) — default
quantui-rs quantize ./my-hf-model ./my-hf-model-int8

# Merge all shards into ONE output file
quantui-rs quantize ./my-hf-model --output-mode single
```

### ▸ INT8 details (scaling modes, convrot)

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

### ▸ FP8 / MXFP8 / NVFP4 differences

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

([back to top](#readme-top))

<p id="quality-modes" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Quality modes — `nvfp4_l2` and `nvfp4_rot16`

The byte-exactness contract is the crate's defining property, so nothing below
is ever reached by default: these are opt-in `--format` values, and every other
format is byte-for-byte unchanged. They are documented here because a user
should be able to tell a byte-exact format from a quality-tuned one **without
reading the source** — and because one of them carries a caveat that no test in
this repository can retire.

| | `nvfp4_l2` | `nvfp4_rot16` |
|---|---|---|
| What it does | NVFP4 with an **anchored alternating L2 scale search** over the per-tensor / per-block scale pair | NVFP4 + **Hadamard (fast Walsh–Hadamard) rotation** at group size **16** |
| `parity:` | `quality-tuned` — **NOT byte-exact** | `exact` — **byte-exact** |
| Why | The reference picks each scale pair by pure absmax, which is exact but leaves accuracy on the table; this searches instead | A rotation is a parity-exact transform, not an approximation |
| Measured | **17.7% lower relative L2** in aggregate over 4 distributions × 4 shapes | No accuracy cost; identical bytes to the reference |
| Inherited | Everything else from `nvfp4`: same E2M1 codes, fixed block scaling at 16, so `--scaling-mode` / `--block-size` remain usage errors. Only the *choice of scale grid point* differs. | Group size 16 == NVFP4's own block size. `int8_convrot` is unchanged at its original group size of 256 |

```sh
quantui-rs quantize mymodel.safetensors --format nvfp4_l2
# parity: quality-tuned (nvfp4_l2: NVFP4 anchored alternating L2 scale search; NOT byte-exact vs torch/llama-quantize)

quantui-rs quantize mymodel.safetensors --format nvfp4_rot16
# parity: exact (nvfp4_rot16; byte-exact vs torch/llama-quantize)
```

**The 17.7% aggregate is not a guarantee.** The gain is **21–26%** on uniform and
Gaussian data, **6–11%** on heavy-tailed, and **1.5–31%** on spiky inputs — the
search has less to work with when a few elements dominate the absmax.

**Why the search must be _anchored_.** The reconstruction is
`X̂ = s_T · s_G · Q(X/(s_T·s_G))`, so rescaling `s_T → k·s_T, s_G → s_G/k` leaves
the product invariant; and because the E2M1 grid is `{2^j, 1.5·2^j}`, the 4-bit
codes are *also* unchanged whenever `k` is a power of two. The objective is
therefore **exactly flat along powers of two** — a naive search is not merely
unconstrained but genuinely ambiguous, and drifts to an arbitrary,
eventually-unrepresentable scale. The implementation searches a bounded ±4 E4M3
code window around the absmax anchor (±0.5 octave); a test pins that it cannot
drift. Formulation follows arXiv:2509.23202.

**Why rotation at group size 16.** Aligning the rotation block with the
microscaling group eliminates cross-block variance and halves the online
rotation cost (**DuQuant++**, arXiv:2604.17789), reached independently from a
coding-theory angle by **The Great Inversion** (arXiv:2608.25188). The Hadamard
construction follows **ConvRot** (arXiv:2512.03673), whose Theorem 3.3 proves
all Kronecker powers `H_{4^k}` are *regular* (row/column sums ±√n) — which
avoids the degenerate all-ones column a naive Sylvester construction gives.

> [!CAUTION]
> **The rotation is applied offline, so the consuming runtime must apply the
> _inverse_ rotation online at inference.** Whether ComfyUI does this **cannot be
> verified from inside this repository** — the failure, if any, is not in the
> artifact written here but in whether a downstream runtime honours it. If the
> consumer does *not* apply the inverse, every rotated layer is garbage: a
> valid-looking file with silently wrong numerics. Treat end-to-end correctness
> as **unverified** until the consumer path is confirmed out of band.
>
> A narrower gap: the NVFP4-family `comfy_quant` blob has no `convrot` /
> `convrot_groupsize` keys (only the INT8 family-A blob does), so the emitted
> metadata does not record that a tensor was rotated — a consumer has no
> in-band signal to key off.

([back to top](#readme-top))

<p id="conformance" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Format conformance & the `parity:` marker

### ▸ The `parity:` line

Every `quantize` run prints exactly one `parity:` line, on **every** run:

```
parity: exact (nvfp4; byte-exact vs torch/llama-quantize)
parity: quality-tuned (nvfp4_l2: NVFP4 anchored alternating L2 scale search; NOT byte-exact vs torch/llama-quantize)
```

It prints unconditionally, *including on exact runs*. That is the point: a user
who has seen the marker on an exact run has learned the tool makes the
distinction at all, which is what gives its presence on a quality run its
meaning. Printing it only when there is bad news would make it
indistinguishable from "this build does not report parity".

The line is derived from a single source of truth in the core
(`Quality::is_parity_exact()` / `Quality::reason()`), not from per-format
string matching — so a new format cannot be added without automatically getting
a correct label.

**Exit codes are deliberately unchanged.** `0` still means success, *including*
for a quality-tuned run. A distinct exit code would read as failure and break
every existing script; the greppable `parity:` line is the machine-detectable
signal instead:

```sh
quantui-rs quantize m.safetensors --format nvfp4_l2 2>&1 | grep -q '^parity: quality-tuned' \
  && echo "NOT byte-exact — do not compare against reference bytes"
```

### ▸ Conformance vectors

Bit-exact conformance vectors are adopted from **Golden Ruler**
(arXiv:2606.09686v3, upstream `gHashTag/t27`), vendored as test fixtures and
asserted in **both the encode and the decode direction**. Conformance is
asserted on the **integer bit pattern**, never on decoded-value closeness —
the source paper's stated criterion.

<p id="gguf" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Converting to GGUF (`gguf`)

### ▸ Quick start

```sh
quantui-rs gguf mymodel.safetensors                 # q4_k_m (default)
quantui-rs gguf ./my-hf-model -m q8_0 out.gguf      # sharded folder input
quantui-rs gguf mymodel.safetensors -m iq4_xs --imatrix imatrix.dat
quantui-rs gguf --list-methods                      # show all methods
```

Output is auto-named `<base>-<method>.gguf` next to the input when `[OUTPUT]`
is omitted. **Both paths are positional** — there are no `--input`/`--output`
flags.

### ▸ Full parameter reference

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
      --audit <FILE.GGUF>        Audit an existing GGUF (no conversion):
                                 dtype census + spec-conformance scan;
                                 exit 0 clean / 3 violations / 1 unparseable
      --arch <ARCH>              Override the GGUF arch string
                                 (else detected from config.json)
      --name <NAME>              Override general.name metadata
      --list-methods             List all methods and exit
      --no-progress              Plain, CI-friendly output
```

### ▸ Auditing an existing GGUF (`--audit`)

Check any GGUF — a downloaded unsloth file, a llama-quantize output, or your
own — without converting anything:

```sh
quantui-rs gguf --audit model-Q8_0.gguf
# audit: model-Q8_0.gguf (266 tensors)
#   dtype histogram: F16=2, F32=99, Q8_0=165
#   spec-conformance: OK (0 violations)
```

The audit runs the same per-row spec check the converter enforces
(`ne[0] % block_size == 0` for every quantized tensor — the `gguf.cpp:724`
rule that makes files loadable by llama.cpp) and prints a dtype census.
Exit code `3` flags a spec-violating file, so it scripts cleanly as a gate:

```sh
quantui-rs gguf --audit "$f" || echo "$f is spec-violating"
```

### ▸ What the converter does

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
- **Warning rendering**: per-tensor warnings are rendered ABOVE the live
  progress bar (via indicatif), so the bar stays the last line instead of
  being re-broken by each warning. On a terminal the first warning of each
  recurring kind is shown and the repeats are folded into one `note:`
  line at the end (the full tensor list is in the run summary); piped or
  `--no-progress` output prints every warning line, byte-identical to
  historical output. The warning stream (report list + callback) is
  identical for every `QUANTUI_RS_GGUF_JOBS` setting.
- **Parallel encoding**: tensor *payloads* are quantized in parallel on a
  rayon pool, in chunks of 8 tensors or 512 MiB of raw input (whichever
  comes first). Everything order-sensitive — name mapping, scheme
  resolution (the llama-quantize policy counters), writing, warnings,
  report accounting and progress — stays sequential, so the output is
  **byte-identical** regardless of thread count.

### ▸ `QUANTUI_RS_GGUF_JOBS` — encode thread count

```sh
QUANTUI_RS_GGUF_JOBS=1 quantui-rs gguf model.safetensors -m q8_0   # sequential
QUANTUI_RS_GGUF_JOBS=8 quantui-rs gguf model.safetensors -m q8_0   # 8 threads
```

Unset = one thread per logical core (`available_parallelism`). `1` forces
the fully sequential pipeline (chunk size 1 on a single-thread pool) —
there is only one code path, `jobs` merely parameterises it, so this is
also the reference mode if you ever want to A/B the output:

```sh
QUANTUI_RS_GGUF_JOBS=1 quantui-rs gguf m.safetensors seq.gguf -m q8_0
QUANTUI_RS_GGUF_JOBS=4 quantui-rs gguf m.safetensors par.gguf -m q8_0
fc /b seq.gguf par.gguf        # Windows (`cmp` on Linux) — identical
```

Parallelism does **not** change the file: both runs above produce the same
bytes, the same stderr and the same report. It only changes how fast they
are produced.

### ▸ Method capability matrix

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

([back to top](#readme-top))

<p id="method-selection" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Which GGUF method should I use?

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

<p id="recipes" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Per-tensor recipes — the open `UD-*`

Unsloth's Dynamic 2.0 presets (`UD-Q4_K_XL` etc.) are a proprietary
per-layer bit-width heuristic and are rejected with exit 2. `--tensor-type-file`
is the open equivalent: any per-tensor assignment expressible as a
first-match-wins regex list, with your choices explicit and inspectable.

### ▸ Recipe file format (`--tensor-type-file <PATH>`)

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

### ▸ Dumping the effective assignment

```sh
quantui-rs gguf model.safetensors out.gguf -m q4_k_m \
    --tensor-type-file my.recipe --emit-recipe effective.recipe
```

`effective.recipe` carries one `^name$=qtype` line per tensor in conversion
order; feeding it back via `--tensor-type-file` reproduces the assignment
exactly.

<p id="verify-against" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ `--verify-against` — oracle equivalence report

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

([back to top](#readme-top))

<p id="recipe-from" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ `--recipe-from` — quantize aligned with a reference

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

<p id="universal-models" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Universal model support (multimodal / wrapped checkpoints)

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

([back to top](#readme-top))

<p id="validate-info" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ `validate` and `info`

### ▸ validate — check a quantized output

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

### ▸ info — inspect a header

```sh
quantui-rs info model.safetensors         # per-tensor table + format detection
quantui-rs info model.safetensors --raw   # raw JSON header
```

Parses only the header (no tensor payloads).

<p id="cast" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Casting to a single-file bf16/fp16 model (`cast`)

`gguf -m bf16` already converts losslessly **to GGUF**. `cast` is the same
operation for the other direction: producing a **single-file
`.safetensors`** in bf16, fp16 or fp32 — the format ComfyUI and transformers
actually load.

```sh
# Merge a 3-shard HF model into ONE bf16 .safetensors (auto-named)
quantui-rs cast ./VibeVoice-1.5B --to bf16
# -> wrote VibeVoice-1.5B-bf16.safetensors (1204 tensors)

# Explicit destination, and a real narrowing to fp16
quantui-rs cast ./my-model merged-f16.safetensors --to f16

# Widen back up
quantui-rs cast merged-f16.safetensors --to f32
```

```
quantui-rs cast [OPTIONS] <INPUT> [OUTPUT]

  <INPUT>                 .safetensors file OR sharded HF model folder
  [OUTPUT]                .safetensors path; omitted = auto-named
                          <base>-<tag>.safetensors beside the input
      --to <TO>           bf16 | f16 | f32            [required]
      --no-progress       Plain, CI-friendly output (no progress bar)
```

A live progress bar is shown while casting, exactly as for `quantize` and
`gguf` — position, elapsed time, and the target dtype. The bar renders on
**stderr**; the `cast:` summary goes to **stdout**, so piping stdout stays
machine-readable. indicatif auto-suppresses the bar when stderr is not a
terminal, and `--no-progress` disables it unconditionally.

`cast` knows the total tensor count up front (it comes from the merged header),
so the bar is accurate from the first tick — `quantize` and `gguf` only learn
their total as they go. The position advances only after a tensor is durably
written, so it never claims progress the file does not have. The bar is
cleared before anything is printed, including on the overflow and I/O error
paths, so a message is never garbled by a live bar.

There is nothing else to tune: the target dtype is the only decision, and
every other parameter is fixed by the format.

### ▸ This is a cast, not a quantization

The output carries **no quantization metadata at all** — no `.comfy_quant`
blob, no `weight_scale` tensors, no `__metadata__._quantization_metadata`. The
only metadata written is `{"format": "pt"}`, the same tag
`safetensors.torch.save_file` writes. That is deliberate: a bf16 file
advertising itself as ComfyUI-quantized while containing nothing quantized
loads as a **broken** model, which is why this is a separate command instead
of a `--format` value on `quantize`.

For the same reason `cast` prints a `cast:` summary, **never** a `parity:`
line — that marker means `exact` vs `quality-tuned` for `quantize` runs, and a
cast is neither. f32→bf16 is not reversible, so claiming byte-exact parity
would be false.

### ▸ Conversion rules

| Source → target | Behaviour |
|---|---|
| Same dtype (bf16→bf16) | **Verbatim byte copy.** No decode, no re-encode |
| Wider float (f32→bf16/f16) | Round-to-nearest-even narrowing |
| Narrower float (bf16/f32→f16) | RNE narrowing, **refused on overflow** (below) |
| Widening (bf16/f16→f32) | Exact |
| Non-float (`I64`, `U8`, `BOOL`, …) | Passed through **untouched**, original header spelling preserved |
| `F64` source | **Refused** — `f64→f32→bf16` is a double rounding whose result depends on the intermediate step |

**Same-dtype is a true byte copy, not a round-trip.** A bf16→bf16 pass through
`f32` is numerically lossless, but it can quiet a signalling-NaN payload and
alter bits nobody asked to change. The fast path copies the payload untouched.

**Overflow is an error, never a saturation.** bf16's range far exceeds f16's,
so a bf16 value above f16's limit has no f16 representation — and
`half::f16::from_f32` would return `Inf` for it. A file full of `Inf` where
numbers were expected is silent data corruption that only shows up as garbage
activations at inference, so `cast` refuses, **names the tensor and element**,
and exits 1:

```
error: tensor "encoder.layers.0.mlp.gate_proj.weight": element 4112 is
9.9e29, which overflows f16 (max finite 65504) and would be written as Inf.
Refusing to emit Inf where a number was expected.
```

Subnormal collapse to `±0.0` is *not* treated as an error — that is ordinary,
expected f16 narrowing. An `Inf` **input** is also not an error: f16 has a real
infinity, so `Inf → Inf` is exact.

### ▸ Safety

The output is written to a temporary path and renamed into place only after
the final header flush succeeds. A refused or interrupted run therefore leaves
**no** output file, and never disturbs a pre-existing file at the destination.
The source folder — including `model.safetensors.index.json` — is only ever
read, never modified.

### ▸ Determinism

Two runs over the same input produce byte-identical output: tensors are
emitted in the union header's first-appearance order, and nothing in the path
is time- or hash-dependent.

The `cast:` summary line is **byte-stable too** — it carries no timestamp, no
duration and no path-dependent text, so it can be grepped and diffed like the
`quantize` and `gguf` summaries. Elapsed time appears in the progress bar
during the run, not in the final line.

([back to top](#readme-top))

<p id="exit-codes" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Exit codes (all commands)

| Code | Meaning |
|---|---|
| 0 | Success (validate: all checks passed; gguf: report printed, diffs allowed) |
| 1 | Runtime failure (I/O error, invalid file, validate found issues, unparseable `--verify-against` reference, **cast: f16 overflow or `f64` source**) |
| 2 | Usage error (bad arguments, unknown GGUF method, missing input, missing `--imatrix`, bad recipe, **cast: unusable input**) |
| 3 | **gguf only**: `--verify-against` found spec violations in OUR output |
| 130 | Cancelled by Ctrl-C (partial output is resumable) |

**A `cast` refusal exits 1, not 2.** The command line was well-formed — the
model simply cannot be represented at the requested dtype — so it is a data
error, not a usage error.

**Exit codes are unaffected by the opt-in quality formats.** A `nvfp4_l2` run
exits `0` on success like any other — a non-zero code would read as failure and
break existing scripts. Scripts that need to know whether a run was byte-exact
should grep the [`parity:` line](#conformance) instead.

<p id="completions" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Shell completions

The `completions` subcommand emits a completion script for your shell
(hidden from `--help` to keep the surface clean):

```sh
# bash: one-shot, current session
source <(quantui-rs completions bash)

# bash: persistent
quantui-rs completions bash > ~/.local/share/bash-completion/completions/quantui-rs

# zsh
quantui-rs completions zsh > "${fpath[1]}/_quantui-rs"

# fish
quantui-rs completions fish > ~/.config/fish/completions/quantui-rs.fish

# PowerShell
quantui-rs completions powershell >> $PROFILE

# elvish
quantui-rs completions elvish >> ~/.elvish/rc.elv
```

Supported shells: `bash`, `zsh`, `fish`, `powershell`, `elvish`. After
reloading your shell, `quantui-rs <TAB>` completes subcommands and flags.

<p id="worked-examples" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Worked examples: real models

### ▸ LFM2.5-VL-3B → Q8_0, aligned with unsloth

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

### ▸ VibeVoice-1.5B → Q8_0

```sh
quantui-rs gguf VibeVoice-1.5B-bf16.safetensors VibeVoice-1.5B-q8_0.gguf -m q8_0
```

The LM backbone (`model.language_model.*`) maps to `blk.*` /
`token_embd.weight` / `output_norm.weight`; the acoustic/semantic tokenizer
heads and prediction head pass through under their original names. The 102
conv kernels (row widths 4/7/8/10/16) are demoted to F16 with loud warnings
— the output is always spec-conformant.

### ▸ VibeVoice-1.5B → single-file bf16 (`cast`)

The shipped repo is 3 shards; ComfyUI and transformers want one file:

```sh
quantui-rs cast ./VibeVoice-1.5B --to bf16
# cast: 1204 tensors, lossless (BF16 -> BF16, verbatim copy)
```

All 1204 tensors were **already** BF16, so this is a pure shard merge — the
payload is copied byte-for-byte and nothing is re-encoded. Output:
`VibeVoice-1.5B-bf16.safetensors` (5.04 GiB), verified byte-identical to the
concatenated shards and loadable by the upstream `safetensors` library under
torch. The 3 source shards and the index file are left untouched.

Note the difference from the GGUF example above: that one *converts* to a
different container, this one only *merges* — hence `lossless (verbatim copy)`
rather than a `parity:` line.

([back to top](#readme-top))

<p id="parity" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Parity contract & known boundaries

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
- Bit-exact format conformance vectors from **Golden Ruler**
  (arXiv:2606.09686v3, upstream `gHashTag/t27`), vendored as fixtures and
  asserted on integer bit patterns in both the encode and the decode
  direction — see [Format conformance](#conformance).

**Byte-exactness holds on every default path.** The only exceptions are the
deliberate opt-ins, and every run declares which kind it was on the
[`parity:` line](#conformance):

- `nvfp4_l2` — **NOT byte-exact.** Trades the guarantee for ~17.7% lower
  reconstruction error. See [Quality modes](#quality-modes).
- `int8_clip09` — **NOT byte-exact**, and measured *worse*. Kept reachable only
  so the negative result stays reproducible. See
  [Tried, measured, rejected](DEVELOPMENT.md#rejected).
- `nvfp4_rot16` / `int8_convrot` — **byte-exact** (a rotation is a parity-exact
  transform), but `nvfp4_rot16` carries an unverified end-to-end caveat: the
  consumer must apply the inverse rotation online. See
  [Quality modes](#quality-modes).

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
- **The output is a weight-tensor container, not a standalone
  inference-loadable model.** No `tokenizer.ggml.*` keys are written, so
  `llama.cpp` cannot load these files directly — the consuming runtime
  (e.g. a ComfyUI loader) supplies the vocabulary. This matches the
  reference implementation's native writer; the Unsloth path gets tokenizer
  metadata only because it delegates to llama.cpp's own converter.
- `general.file_type` and `general.quantization_version` are not written.
  Downstream tooling falls back to inferring the quantization from tensor
  shapes.

([back to top](#readme-top))

<p id="performance" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>
## ❯ Performance

Measured on a 24-core Windows box, 1.004 GiB fixture (32×4096×4096 bf16),
vs the Python reference streaming path:

| Metric | Rust | Python ref | Ratio |
|---|---|---|---|
| End-to-end 1 GiB streaming | 4.10 s (250 MiB/s) | 4.80 s (214 MiB/s) | 1.17× |
| CPU-bound quant kernel (4096×4096) | 68.6 ms (933 MiB/s) | 127.5 ms (245 MiB/s) | 1.86× |

The byte-exact contract rules out numeric shortcuts that would widen the
gap — parity was prioritized over raw speed. Reproduce with
`cargo bench -p quant-core` and `tools/bench_python_ref.py`.

<p id="development" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Development, benchmarks & tooling

Repository layout, the test/bench commands, the nightly benchmark report, the
GGUF tooling scripts, and the measured-negative results that were
deliberately not recommended — all moved to
**[DEVELOPMENT.md](DEVELOPMENT.md)**.

<p id="license" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ License

MIT — see [LICENSE](LICENSE). Third-party notices:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Contributions:
[CONTRIBUTING.md](CONTRIBUTING.md). Security reports:
[SECURITY.md](SECURITY.md).

<!-- HEADER BADGES -->

[rust-shield]: https://img.shields.io/badge/Rust-1.89%2B-000000?style=for-the-badge&logo=rust&logoColor=white
[rust-url]: https://www.rust-lang.org
[mit-shield]: https://img.shields.io/badge/License-MIT-8957e5?style=for-the-badge&logo=opensourceinitiative&logoColor=white
[license-url]: LICENSE
[platform-shield]: https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-000000?style=for-the-badge&logo=github&logoColor=white
[building-url]: https://github.com/wildminder/quantui-rs#building
[parity-shield]: https://img.shields.io/badge/parity-exact%20on%20default%20paths-8957e5?style=for-the-badge&logo=shield&logoColor=white
[conformance]: #conformance
[stars-shield]: https://img.shields.io/github/stars/wildminder/quantui-rs?style=for-the-badge&logo=github&logoColor=white
[stars-url]: https://github.com/wildminder/quantui-rs/stargazers
[commit-shield]: https://img.shields.io/github/last-commit/wildminder/quantui-rs?style=for-the-badge&logo=github&logoColor=white
[commit-url]: https://github.com/wildminder/quantui-rs/commits/main

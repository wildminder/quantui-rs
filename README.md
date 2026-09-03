# quantui-rs

Standalone, single-binary Rust CLI that reimplements the Python streaming model
quantizer (`quantui`, backed by `convert_to_quant` + `comfy_kitchen`) — with a
hard contract: **outputs are byte-exact against the Python/torch reference** on
all supported paths.

It quantizes Hugging Face `safetensors` models (single file or sharded folder)
to ComfyUI-compatible **INT8 (plain or Hadamard-rotated), FP8 E4M3, MXFP8 and
NVFP4** outputs, converts HF models to GGUF (34 usable quantization methods,
including all K-quants and IQ\* via `--imatrix`), validates quantized files,
and inspects safetensors headers — with resumable streaming, progress bars,
and graceful Ctrl-C stops.

## Highlights

- **Five target formats**: `--format int8` (default), `int8_convrot`
  (row-wise INT8 + group-wise Hadamard rotation), `fp8_e4m3`, `mxfp8`,
  `nvfp4` — all reachable from the CLI, all verified against Python/torch
  goldens.
- **Byte-exact parity** with the Python reference: streaming quantization
  (including bias correction via a bit-exact port of torch's MT19937 RNG),
  comfy_quant blob encoding, header layout, manifest checkpoints — verified by
  golden byte-compare tests.
- **Resumable streaming**: quantizes tensor-by-tensor, flushing a header +
  manifest checkpoint after every tensor. Kill it at any point (or press
  Ctrl-C) and re-run to resume; partial outputs are always valid and loadable.
- **Sharded HF models**: reads `model.safetensors.index.json` folders, supports
  `--output-mode sharded` (one output per input shard) or `single` (merge).
- **GGUF conversion**: native HF → GGUF with llama.cpp-compatible naming,
  per-method tensor policies — 34 usable methods including every K-quant,
  IQ\* (via `--imatrix`), ternary and 1/2-bit scheme, all byte-identical to
  the corresponding reference encoders; plus per-tensor recipe overrides
  (`--tensor-type-file`, the open equivalent of Unsloth's proprietary
  `UD-*` presets).
- **Zero runtime dependencies**: one static binary, no Python, no torch.

## Building

Requires Rust (stable, ≥ 1.89):

```sh
cargo build --release
# binary: target/release/quantui-rs(.exe)  (~2.6 MB, LTO + stripped)
```

## Quick start: quantize a model to INT8

```sh
# 1. Quantize (defaults: block scaling, block size 128, heuristics on).
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
config blob describing its INT8 layout.

## INT8 quantization in detail

### What gets quantized

The streaming quantizer walks every tensor in the input (in file order) and
applies these rules — identical to the reference:

| Tensor | Treatment |
|---|---|
| 2D `*.weight` whose dims are divisible by the block size | **Quantized to INT8** |
| 2D `*.weight` not divisible by the block size (with default `--heur`) | Copied, cast to `--orig-dtype` (bfloat16) |
| 2D `*.weight` matching `--exclude-layers` regex | Copied, cast to `--orig-dtype` |
| Everything else (biases, norms, embeddings, …) | Copied unchanged |

`--no-heur` disables the divisibility heuristic and quantizes *every* 2D
weight (the run then fails on dims not divisible by the block size).

### Output structure

Each quantized layer `X` produces four tensors:

| Tensor | Dtype | Meaning |
|---|---|---|
| `X.weight` | `I8` | Quantized weights |
| `X.weight_scale` | `F32` | Dequant scales (shape depends on scaling mode) |
| `X.comfy_quant` | `U8` | JSON blob describing the INT8 layout for ComfyUI |
| `X.input_scale` | `F32` | Scalar `1.0` (parity with the reference) |

Real example — `quantui-rs info` on a quantized file with one quantized layer
(`blocks.0`, 256×128, block mode bs=128) and one skipped layer (`blocks.1`,
128×64 — not divisible by 128, kept as BF16):

```
name                          dtype   shape        size
blocks.0.bias                 F32     [256]        1.00 KB
blocks.1.bias                 F32     [128]        512 B
norm.weight                   F32     [256]        1.00 KB
blocks.0.weight               I8      [256, 128]   32.00 KB
blocks.0.weight_scale         F32     [2, 1]       8 B
blocks.0.comfy_quant          U8      [79]         79 B
blocks.0.input_scale          F32     []           4 B
blocks.1.weight               BF16    [128, 64]    16.00 KB

quantized layers (1):
  blocks.0: int8_blockwise
```

### Scaling modes (`-m/--scaling-mode`)

| Mode | Scale granularity | `weight_scale` shape | Notes |
|---|---|---|---|
| `block` (default) | One scale per `bs×bs` tile | `[m/bs, n/bs]` | Best accuracy/size trade-off; requires divisible dims |
| `row` | One scale per output row | `[m, 1]` | No divisibility requirement |
| `tensor` | One scale for the whole tensor | `[]` (scalar) | Coarsest granularity; single scale is squeezed to scalar |

(A scale tensor with exactly one element is squeezed to a scalar `[]` on disk,
mirroring the reference's `normalize_tensorwise_scales` behavior.)

Block size (`-b/--block-size`): `64`, `128` (default), or `256` — only used
with `block` mode.

### Auto-naming

Omit the output path and a ctq-compatible name is generated from the
*effective* config: `<base>-<format>-simple-heur.safetensors`, where
`<format>` is `int8_block` / `int8_row` / `int8_tensor` (INT8 encodes the
scaling mode) or `fp8_e4m3` / `mxfp8` / `nvfp4` (these use a single id for
all modes, matching the reference registry). Examples:

```
mymodel-int8_block-simple-heur.safetensors
mymodel-mxfp8-simple-heur.safetensors
```

An explicit `.safetensors` output path always wins.

### Resume & Ctrl-C

After **every tensor** the writer flushes the safetensors header and a
checkpoint file `<output>.quant-manifest.json`:

```json
{"version":1,"config_hash":"56920c6553cfa241","order":["..."],"done":["..."]}
```

- **Kill the process or press Ctrl-C at any time** → exit code 130, and the
  partial output remains valid and loadable.
- **Re-run the same command** → already-done tensors are skipped, the run
  continues where it stopped, and the final file is byte-identical to an
  uninterrupted run.
- **Change the config** (e.g. different block size) → the `config_hash` no
  longer matches and the run restarts cleanly from scratch (this prevents
  corrupt mixed-config files).

### Sharded HF models

Point `quantize` at a folder containing `model.safetensors.index.json`:

```sh
# Default: one output shard per input shard (sharding preserved)
quantui-rs quantize ./my-hf-model ./my-hf-model-int8

# Merge all shards into ONE output file
quantui-rs quantize ./my-hf-model --output-mode single
```

- **`--output-mode sharded`** (default): writes one quantized `.safetensors`
  per input shard, plus a rewritten `model.safetensors.index.json`, copied
  sidecar files, and a global `.quant-manifest.json` in the output directory.
- **`--output-mode single`**: streams the union of all shards into one output
  file (replaces the old merge-then-quantize temp-file path).

Tensors are processed in first-appearance (union) order; calibration and bias
correction behave exactly like the reference in both modes.

### Bias correction & the calibration seed

Quantized layers undergo bias correction against simulated calibration data,
exactly mirroring `convert_to_quant`. The RNG stream is a bit-exact port of
torch's CPU MT19937 + `randn`, pinned to `--calib-seed 233983427` by default —
this pinning is what makes streaming runs reproducible and byte-identical to
the reference. Don't change the seed unless you intentionally want a different
(but still valid) correction.

## `int8_convrot`: INT8 with Hadamard rotation

```sh
quantui-rs quantize mymodel.safetensors --format int8_convrot
```

A preset, not a free-form combination: **INT8 row-wise** quantization with a
group-wise Hadamard rotation (`convrot_group_size=256`, the reference default)
applied to each 2-D weight before quantization, followed by rotation-aware
bias correction (`Y_ref = X_rot @ W_rotᵀ` vs `Y_qnt = X_rot @ W_dqᵀ`). The
CLI forces row mode — the reference only rotates in row mode, so accepting
`--scaling-mode block` here would silently produce an unrotated layer.

Per tensor, exactly like the reference:

- `in_features % 256 == 0` → rotated; the `.comfy_quant` blob carries
  `{"convrot": true, "convrot_groupsize": 256, "per_row": true}`.
- Not divisible (or 1-D/4-D shapes) → plain row-wise INT8, blob keeps
  `per_row: true` and drops only the `convrot` keys.
- Skip heuristics apply (`--heur`, default on): layers whose dims are < 128
  or not divisible by 128 are cast to bf16, unquantized.

Byte-exact vs `convert_to_quant`'s batch path — 8 golden cases at group sizes
256 and 64 lock the rotation gate, both bias-correction paths, the blob
strings, and per-tensor payload bytes. Verified per-tensor (the ctq batch
driver writes tensors in its own processing order with sorted keys, so
whole-file equality does not apply).

## Beyond INT8: FP8 E4M3 / MXFP8 / NVFP4

```sh
quantui-rs quantize mymodel.safetensors --format fp8_e4m3
quantui-rs quantize mymodel.safetensors --format mxfp8
quantui-rs quantize mymodel.safetensors --format nvfp4
```

These mirror `convert_to_quant`'s dedicated format paths. Everything else —
resume, Ctrl-C, sharded input, both output modes, `validate`, `info` — works
identically to INT8.

### Per-format behaviour

Given a 256×128 layer, the streaming output looks like this:

| | `int8` | `fp8_e4m3` | `mxfp8` | `nvfp4` |
|---|---|---|---|---|
| `X.weight` dtype | `I8` `[256,128]` | `F8_E4M3` `[256,128]` | `F8_E4M3` `[256,128]` | `U8` `[256,64]` (2×E2M1 packed/byte) |
| `X.weight_scale` | `F32` | `F32` | `U8` (e8m0) | `F8_E4M3` |
| Scale layout | per mode | per mode | swizzled (blocked) `[256,4]` | swizzled (blocked) `[256,8]` |
| `X.weight_scale_2` | — | — | — | `F32` scalar (per-tensor) |
| `X.input_scale` | `F32` 1.0 | — | — | — |
| Scaling modes | `tensor`/`row`/`block` | `tensor`/`row`/`block` | fixed `block` | fixed `block` |
| Block size | 64/128/256 | 64/128/256 | fixed 32 | fixed 16 |
| Dims not divisible by block | skipped (with `--heur`) | block→row fallback | padded internally | padded internally |
| `AVOID_KEY_NAMES` exclusions (`norm`, `bias`, `lm_head`, …) | no | no | **yes** | **yes** |
| `__metadata__` on output | no | no | **yes** | **yes** |

Detected per-layer format strings (`quantui-rs info`): `int8_blockwise`,
`float8_e4m3fn` / `float8_e4m3fn_rowwise` / `float8_e4m3fn_blockwise`,
`mxfp8`, `nvfp4`.

MXFP8/NVFP4 scales are written in ctq's *swizzled* (blocked) layout; NVFP4
additionally carries a per-tensor `weight_scale_2`. `validate` understands
all of these.

### `__metadata__._quantization_metadata`

MXFP8/NVFP4 outputs carry a file-level `__metadata__` entry — a JSON string
listing every quantized layer, byte-identical to ctq's:

```json
{"format_version": "1.0", "layers": {"blocks.0": {"format": "mxfp8",
 "group_size": 32, "orig_dtype": "torch.bfloat16", "orig_shape": [256, 128]}}}
```

Only **quantized** layers appear; outer and layer keys are sorted, and
`orig_dtype` uses ctq's `torch.*` spelling. INT8 and FP8 outputs never carry
`__metadata__` (matching ctq, where `save_quant_metadata` defaults off on
those paths), and neither does a MXFP8/NVFP4 run in which every layer was
skipped.

### Parity contract for the non-INT8 formats

INT8 is verified **whole-file** byte-for-byte. The newer formats are verified
**per-tensor**: every tensor's payload bytes plus its `(dtype, shape)`, and the
exact `__metadata__` string. Whole-file equality is structurally impossible
here, because ctq writes tensors in its own processing order with sorted
header keys and minimal padding, while the streaming orchestrator writes in
input order with a 64 KiB header slot. Per-tensor parity is the same guarantee
at the level that matters — no numeric or layout detail is left unchecked.

## `quantize` CLI reference

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

## GGUF conversion in detail

```sh
quantui-rs gguf mymodel.safetensors                 # q4_k_m (default)
quantui-rs gguf ./my-hf-model -m q8_0 out.gguf      # sharded folder input
quantui-rs gguf mymodel.safetensors -m iq4_xs --imatrix imatrix.dat
quantui-rs gguf mymodel.safetensors --tensor-type-file recipe.txt
quantui-rs gguf --list-methods                      # show all methods
```

- **Methods**: 34 usable — every non-`UD-*` method in official Unsloth's
  list (see the matrix below). Output is auto-named `<base>-<method>.gguf`.
- **Per-tensor policy**: composite methods apply llama-quantize rules
  (e.g. `q4_k_m` uses Q6_K for key/output tensors, Q4_K elsewhere; 1-D
  tensors → F32, embeddings → F16 where the method prescribes it).
- **Architecture metadata** is detected from `config.json`
  (`model_type`/`architectures`) and written as GGUF `general.architecture` +
  `<arch>.*` metadata; override with `--arch`/`--name`.
- **Tensor naming** follows llama.cpp conventions for the dense
  llama/qwen/mistral/gemma family (`model.layers.0.self_attn.q_proj.weight` →
  `blk.0.attn_q.weight`), with HF→GGUF dimension reversal.
- **Per-tensor recipes** (`--tensor-type-file`): `regex=qtype` lines,
  first-match-wins against the GGUF tensor name, optional trailing bare
  `qtype` as default. `--token-embedding-type` / `--output-tensor-type`
  override those two categories. `--emit-recipe <path>` dumps the effective
  per-tensor assignment; feeding the dump back reproduces it exactly.

### Method capability matrix

All 38 methods in the registry (Unsloth's list plus llama.cpp-only
`q1_0`/`q2_0`), with their capability class and parity tier:

| Method | Class | Parity |
|---|---|---|
| `f16`, `bf16`, `f32` | usable, no imatrix | byte-exact vs gguf-py writer (`f16`/`bf16`); `f32` is a lossless passthrough |
| `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1` | usable, no imatrix | byte-exact vs gguf-py writer (49/49 golden cases) |
| `q2_k`, `q2_k_l`, `q3_k_*` (`m/l/s/xs`), `q4_k_*` (`m/s`), `q5_k_*` (`m/s`), `q6_k` | usable, no imatrix | byte-exact vs llama-quantize (weighted, `--imatrix` goldens); no-imatrix path locked by a `None ≡ uniform-1.0` equivalence test + the structural sweep |
| `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq2_m`, `iq3_xxs`, `iq3_s`, `iq3_m`, `iq4_nl`, `iq4_xs` | usable, **requires `--imatrix`** | byte-exact vs llama-quantize (weighted, `--imatrix` goldens) |
| `tq1_0`, `tq2_0` | usable, no imatrix | structural (encode + bounded round-trip) |
| `q1_0`, `q2_0` | usable, no imatrix | structural (encode + bounded round-trip) |
| `q4_nl`, `q4_k_xl`, `q3_k_xl`, `q2_k_xl` | **rejected** (exit 2) | Unsloth Dynamic 2.0 — proprietary per-layer bit-width heuristic |

**The imatrix caveat**: every `iq*` method requires an importance matrix
(`--imatrix <PATH>`). Without it, quantization produces garbage — the same
conversion llama-quantize refuses. The matrix is consumed per-tensor by the
weighted quantizers; `K-quants` also consume one when supplied (optional
for them). Files load in GGUF or llama-quantize's legacy binary format.

**The `UD-*` boundary**: Unsloth's Dynamic 2.0 presets (`UD-Q4_K_XL` etc.)
are a proprietary per-layer bit-width heuristic. They are rejected with exit
2 and an explanation. `--tensor-type-file` is the open equivalent: any
per-tensor recipe expressible as a first-match-wins regex list reproduces
what a `UD-*` preset does, with your own choices explicit and inspectable.

## `validate` — check a quantized output

```sh
quantui-rs validate out.safetensors            # structural checks (headers only)
quantui-rs validate out.safetensors --numeric  # + read tensor payloads
quantui-rs validate ./sharded-output-dir/      # directories expand to shards
```

Covers all 7 ctq formats (INT8 tensor/row/block, FP8, MXFP8, NVFP4): per-layer
layout (`weight` + `weight_scale` + `.comfy_quant` + `input_scale`), scale
shapes, blob field types, orphan detection, and a summary:

```
file: output.safetensors (0 GB)
quantized matrices      : 1
  GS128                : 1
full-precision weights  : 2
quantized share         : 79.50%
formats found           : int8_blockwise

PASS
```

`--numeric` additionally checks INT8 weight bounds (±127, no −128), scale
finiteness/positivity, E4M3/e8m0 NaN handling, and `input_scale == 1.0`.
Exit codes: 0 ok / 1 any-fail / 2 usage.

## `info` — inspect a header

```sh
quantui-rs info model.safetensors         # per-tensor table + format detection
quantui-rs info model.safetensors --raw   # raw JSON header
```

Parses only the header (no tensor payloads): per-tensor name/dtype/shape/size
table plus detected comfy_quant formats per layer.

## Exit codes (all commands)

| Code | Meaning |
|---|---|
| 0 | Success (validate: all checks passed) |
| 1 | Runtime failure (I/O error, invalid file, validate found issues) |
| 2 | Usage error (bad arguments, unknown GGUF method, missing input) |
| 130 | Cancelled by Ctrl-C (partial output is resumable) |

## Parity contract & known boundaries

**Byte-exact (golden-verified):**

- INT8 streaming quantization: tensor/row/block scaling, skip heuristics,
  skipped-weight dtype casting, bias correction (bit-exact torch MT19937 +
  randn + oneDNN-style sgemm + cascade-sum ports), comfy_quant blobs, header
  8-byte alignment, manifest checkpoints — single-file and sharded, both
  output modes. Verified **whole-file** byte-for-byte.
- FP8 (tensor/row/block), MXFP8 and NVFP4 streaming: kernels, per-format
  dequant for bias correction, swizzled scales, NVFP4 `weight_scale_2`,
  `__metadata__._quantization_metadata`, resume + Ctrl-C, sharded + union
  modes — verified **per-tensor** (payload bytes + `dtype`/`shape` + metadata
  string); see the contract above for why whole-file equality doesn't apply.
- Calibration draw order, which differs between INT8 (file order, via the
  reference streamer) and the ctq formats (alphabetical, since ctq builds its
  key list from `safetensors.safe_open.keys()`, which is always sorted) —
  locked in both directions by `tests/golden/sharded_unsorted`.
- GGUF legacy encoders F16/BF16/F32/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1 — byte-identical to
  gguf-py (49/49 golden cases).
- GGUF weighted K-quant and IQ\* row quantizers (Q4_K/Q2_K/Q3_K/Q5_K/Q6_K,
  iq1_s/m, iq2_xxs/xs/s/m, iq3_xxs/s/m, iq4_nl/xs) — byte-identical to the
  real `llama-quantize --imatrix` on shared fixtures, at both the unit and
  the full-driver e2e tier (14/14 each).
- INT8 row scaling reproduces torch's exact float semantics — including
  `127.0/row_max` computed as `127.0 * reciprocal(row_max)` (torch's
  `div_true_kernel` double-rounds scalar/tensor division) — locked by the
  `int8_convrot` goldens, including both ConvRot bias-correction paths.

**Documented boundaries (out of scope for v1):**

- GGUF ternary (`tq1_0`/`tq2_0`) and 1/2-bit (`q1_0`/`q2_0`) encoders are
  structurally verified (encode + bounded round-trip) but have no external
  byte-parity reference (not in gguf-py; llama.cpp only).
- GGUF K-quant byte-parity goldens are generated **with** an imatrix; the
  no-imatrix path is locked by a `None ≡ uniform-1.0` equivalence test
  (llama.cpp's `if (!quant_weights)` branch), not by an external golden.
- Learned-rounding optimizers (AdamW/RAdam/Prodigy) remain Python-only.
- Unsloth Dynamic 2.0 per-layer mixing is proprietary and not replicated
  (`--tensor-type-file` is the open equivalent).
- W4A4/W4A8 layouts are deferred to v2.

## Performance

Measured on a 24-core Windows box, 1.004 GiB fixture (32×4096×4096 bf16),
vs the Python reference streaming path:

| Metric | Rust | Python ref | Ratio |
|---|---|---|---|
| End-to-end 1 GiB streaming | 4.10 s (250 MiB/s) | 4.80 s (214 MiB/s) | 1.17× |
| CPU-bound quant kernel (4096×4096) | 68.6 ms (933 MiB/s) | 127.5 ms (245 MiB/s) | 1.86× |

The Python reference kernel is already torch/ctq C++, and end-to-end runs are
dominated by disk I/O + per-tensor checkpointing; the byte-exact contract rules
out the numeric shortcuts that would widen the gap, so parity was prioritized
over raw speed. Reproduce with `cargo bench -p quant-core` and
`tools/bench_python_ref.py` (fixture: `tools/gen_bench_fixture.py`).

## Repository layout

```
crates/quant-core/     library: safetensors IO, INT8/FP8/MXFP8/NVFP4 kernels,
                       streaming orchestrator, bias correction, torch-RNG port,
                       comfy_quant schema, validator, GGUF registry + converter
crates/quant-cli/      binary `quantui-rs`: clap CLI, progress, profiles
tests/golden/          Python/torch-generated golden fixtures (byte-parity refs)
tools/                 golden + benchmark generators (Python, ctq venv)
docs/plans/            design plan + execution log
```

## Development

```sh
cargo test --workspace                 # 452 tests incl. golden byte-parity
cargo clippy --workspace --all-targets # clean with -D warnings
cargo fmt --check
cargo bench -p quant-core              # throughput benchmarks (needs fixture)
```

CI (`.github/workflows/ci.yml`) runs fmt + clippy + full test suite + release
build on Windows and Linux. Golden fixtures are committed and marked binary so
byte-compare tests are valid on every OS.

## License

MIT

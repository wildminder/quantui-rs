# quantui-rs

Standalone, single-binary Rust CLI that reimplements the Python streaming model
quantizer (`quantui`, backed by `convert_to_quant` + `comfy_kitchen`) — with a
hard contract: **outputs are byte-exact against the Python/torch reference** on
all supported paths.

It quantizes Hugging Face `safetensors` models (single file or sharded folder)
to ComfyUI-compatible INT8 outputs, converts HF models to GGUF, validates
quantized files, and inspects safetensors headers — with resumable streaming,
progress bars, and graceful Ctrl-C stops.

## Highlights

- **Byte-exact parity** with the Python reference: INT8 streaming quantization
  (including bias correction via a bit-exact port of torch's MT19937 RNG),
  comfy_quant blob encoding, header layout, manifest checkpoints — verified by
  golden byte-compare tests.
- **Resumable streaming**: quantizes tensor-by-tensor, flushing a header +
  manifest checkpoint after every tensor. Kill it at any point (or press
  Ctrl-C) and re-run to resume; partial outputs are always valid and loadable.
- **Sharded HF models**: reads `model.safetensors.index.json` folders, supports
  `--output-mode sharded` (one output per input shard) or `single` (merge).
- **GGUF conversion**: native HF → GGUF with llama.cpp-compatible naming,
  per-method tensor policies (`q4_k_m`, `q8_0`, …), and byte-identical legacy
  quant encoders (verified against gguf-py).
- **Zero runtime dependencies**: one static binary, no Python, no torch.

## Building

Requires Rust (stable, ≥ 1.89):

```sh
cargo build --release
# binary: target/release/quantui-rs(.exe)  (~2.5 MB, LTO + stripped)
```

## Usage

### `quantize` — safetensors → ComfyUI INT8

```sh
# Single file (auto-named output: mymodel-int8_block-simple-heur.safetensors)
quantui-rs quantize mymodel.safetensors

# Explicit output, block scaling with 128 blocks (the defaults)
quantui-rs quantize mymodel.safetensors out.safetensors -m block -b 128

# Sharded HF folder, merged into one output file
quantui-rs quantize ./my-hf-model --output-mode single

# Keep some layers at full precision
quantui-rs quantize mymodel.safetensors --exclude-layers "attn_norm|text_embed"
```

Key options: `-m/--scaling-mode tensor|row|block` (default `block`),
`-b/--block-size 64|128|256` (default `128`), `--no-heur` (quantize every 2D
weight instead of skipping non-divisible ones), `--orig-dtype bfloat16|float16`
(cast target for skipped weights), `--calib-seed` (pinned parity seed,
default `233983427`), `--no-progress`.

Interrupted runs resume automatically: re-run the same command and completed
tensors are skipped. A changed config forces a clean restart (config-hash guard).

### `gguf` — HF safetensors → GGUF

```sh
quantui-rs gguf mymodel.safetensors                 # q4_k_m (default)
quantui-rs gguf ./my-hf-model -m q8_0 out.gguf      # sharded folder input
quantui-rs gguf --list-methods                      # show all methods
```

Supports the dense llama/qwen/mistral/gemma tensor-name family; architecture
metadata is detected from `config.json` (override with `--arch`/`--name`).
Unsloth Dynamic 2.0 variants (`q5_1`, `q4_1`, `q4_nl`, `*_k_xl`) are listed but
rejected — they require proprietary per-layer heuristics.

### `validate` — check a quantized output

```sh
quantui-rs validate out.safetensors            # structural checks
quantui-rs validate out.safetensors --numeric  # + payload checks
quantui-rs validate ./sharded-output-dir/      # expands to shards
```

Covers all 7 ctq formats (INT8 tensor/row/block, FP8, MXFP8, NVFP4): layer
layout, scale shapes, comfy_quant blobs, orphans; `--numeric` adds weight
bounds, scale finiteness/positivity, and `input_scale == 1.0`. Exit codes:
0 ok / 1 any-fail / 2 usage.

### `info` — inspect a header

```sh
quantui-rs info model.safetensors         # per-tensor table + format detection
quantui-rs info model.safetensors --raw   # raw JSON header
```

## Parity contract & known boundaries

**Byte-exact (golden-verified):**

- INT8 streaming quantization: tensor/row/block scaling, skip heuristics,
  skipped-weight dtype casting, bias correction (bit-exact torch MT19937 +
  randn + oneDNN-style sgemm + cascade-sum ports), comfy_quant blobs, header
  8-byte alignment, manifest checkpoints — single-file and sharded, both
  output modes.
- FP8 / MXFP8 / NVFP4 kernels (standalone, golden-verified).
- GGUF legacy encoders F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1 — byte-identical to
  gguf-py (49/49 golden cases).

**Documented boundaries (out of scope for v1):**

- GGUF K-quants/IQ\* produce valid files but are **not** byte-identical to
  llama.cpp (simpler min/max search than upstream `make_qx_quants`).
- Learned-rounding optimizers (AdamW/RAdam/Prodigy) remain Python-only.
- Unsloth Dynamic 2.0 per-layer mixing is proprietary and not replicated.
- ConvRot int8 and W4A4/W4A8 layouts are deferred to v2.

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
cargo test --workspace                 # 210 tests incl. golden byte-parity
cargo clippy --workspace --all-targets # clean with -D warnings
cargo fmt --check
cargo bench -p quant-core              # throughput benchmarks (needs fixture)
```

CI (`.github/workflows/ci.yml`) runs fmt + clippy + full test suite + release
build on Windows and Linux. Golden fixtures are committed and marked binary so
byte-compare tests are valid on every OS.

## License

MIT

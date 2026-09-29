# quantui-rs — development & appendix

The developer companion to the [README](README.md): repository layout, how to
run the suite and benchmarks, the GGUF tooling scripts, and the measured-negative
results that did **not** make it into a recommended format.

Everything here is reference material for working *on* the tool. For what the tool
does and how to run it, start at the [README](README.md).

## Contents

- [Tried, measured, rejected](#rejected)
- [Repository layout & development](#development)
  - [Benchmarks & the nightly report](#benchmarks)
  - [GGUF tooling scripts (`tools/`)](#tooling)

---

<a name="rejected"></a>
## ❯ Tried, measured, rejected

Three techniques from the literature were implemented, measured on this crate's
own metric, and deliberately **not** recommended. Recording them is a result,
not a failure — a reader deciding whether to try one of these deserves to know
it was already tried here:

| Technique | Source | Measured outcome |
|---|---|---|
| MXFP8 E8M0 `4/3` scale compensation | arXiv:2509.23202 | **Bit-exact no-op.** `rel_l2` ratio `1.000000` — 0.00% change |
| MXAttention `Qmax = 7.25` | arXiv:2607.24377 | **Inert.** No output byte changes |
| INT8 absmax clip ratio 0.9 | QuaRot, arXiv:2404.00456 | **31×–1.4e4× worse** in weight-space L2; 0 of 4 distributions improved |

The MXFP8 case is the sharpest: E4M3 halves exactly, so doubling the scale and
halving every code reconstructs the identical `f32`, and this crate's scale
rounds *up*, so it never clips in the first place — leaving nothing for the
compensation to relieve. The paper's variant rounds down and does clip, so the
premise holds there and simply does not apply to this kernel.

`Qmax` is a clamp bound, and the encoder already saturates at 6.0 (the format's
maximum), so no bound ≥ 6.0 can change an emitted byte. Clipping trades bounded
rounding error for *unbounded* saturation error, and INT8's 127 levels leave
only ~1.5e-5 of rounding error to recover; the published gain is a perplexity
result on activations, and this crate quantizes weights and does not measure
perplexity.

**The transferable lesson:** the one technique that worked runs a real *search*
and lets the data pick a point; the three that failed each nudged a hardcoded
constant, which has no feedback signal. Each paper also assumed a kernel
differing from this one in exactly one decisive respect — `floor` vs `ceil`
scale rounding, a clamp below vs at saturation, activations vs weights — and
that difference is what decides the outcome.

> **Note**: the INT8 clip ratio 0.9 result is also reachable from the CLI as the
> opt-in `--format int8_clip09`, kept only so the negative stays reproducible.
> See [Which format to pick](README.md#quantize) in the README. It is **not**
> byte-exact and it is measured *worse* — do not use it.

---

<a name="development"></a>
## ❯ Repository layout & development

```
crates/quant-core/     library: safetensors IO, INT8/FP8/MXFP8/NVFP4 kernels,
                       streaming orchestrator, bias correction, torch-RNG port,
                       comfy_quant schema, validator, GGUF registry + converter,
                       gguf_verify (oracle report), gguf_recipe (per-tensor recipes)
crates/quant-cli/      binary `quantui-rs`: clap CLI, progress, profiles
tests/golden/          Python/torch-generated golden fixtures (byte-parity refs)
tools/                 golden + benchmark generators, GGUF diagnostics
                       (Python; needs torch + the reference quantizer)
design docs           design plans + execution log (kept outside the published repo)
```

```sh
cargo test --workspace                 # 696 tests incl. golden byte-parity
cargo clippy --workspace --all-targets # clean with -D warnings
cargo fmt --check
cargo bench -p quant-core              # throughput benchmarks (needs fixture)
```

CI (`.github/workflows/ci.yml`) runs fmt + clippy + full test suite + release
build on Windows, Linux and macOS. Golden fixtures are committed and marked
binary so byte-compare tests are valid on every OS.

<a name="benchmarks"></a>
### ▸ Benchmarks & the nightly report

- `cargo bench -p quant-core --bench stream_throughput` — GB/s on a ~1 GB
  generated fixture (`python tools/gen_bench_fixture.py` first).
- `cargo bench -p quant-core --bench stream_small` — the same full
  `stream_quantize` path on the **committed** 8 MB fixture
  (`tests/bench/bench_small.safetensors`, reproducible via
  `tools/gen_bench_small.py`, SHA-256 self-checked).

A nightly workflow (`.github/workflows/bench.yml`) runs `stream_small` on
ubuntu, saves a criterion baseline and uploads the full report to the
`criterion-nightly` workflow artifact (30-day retention). It is an
**awareness signal, not a merge gate** — shared-runner timing is noisy. To
read the artifact: download it, open `report/index.html`, and compare
`change` columns against the previous night's artifact. Local A/B:

```sh
cargo bench -p quant-core --bench stream_small -- --save-baseline mine
# ... change code ...
cargo bench -p quant-core --bench stream_small -- --baseline mine
```

<a name="tooling"></a>
### ▸ GGUF tooling scripts (`tools/`)

| Script | Purpose |
|---|---|
| `cmp_ours_vs_unsloth.py` | Byte-compare our GGUF vs unsloth's (the logic now productized as `--verify-against`) |
| `diag_vibevoice_q8.py` | Spec-violation scanner (`ne[0] % blck` audit; now built into `--verify-against`) |
| `probe_block_stats.py` | Per-block Q8_0 scale/code diff analysis |
| `inspect_lfm.py` | Inventory a model's safetensors + GGUFs |
| `gen_golden_gguf.py` / `gen_golden_llamacpp_weighted.py` | Golden fixture generation |
| `sweep_gguf_e2e.sh` | Convert + validate a fixture with EVERY usable GGUF method (49 checks) |

---

## ❯ License

MIT — see [LICENSE](LICENSE) at the repository root.

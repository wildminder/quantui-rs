# Third-party notices

`quantui-rs` is distributed under the [MIT License](LICENSE). This file records
third-party components used as runtime or development dependencies and upstream
projects referenced by the byte-parity implementation.

## Rust dependencies

The runtime dependency set is `rlx-gguf`, `serde`, `serde_json`, `half`,
`regex`, `rayon`, `memmap2`, `thiserror`, `sha2`, `clap`, `clap_complete`,
`indicatif`, and `ctrlc`. Development-only dependencies are `safetensors`,
`tempfile`, `criterion`, and `proptest`.

Each dependency remains under its own license as declared in its crate
manifest. In particular, `rlx-gguf` is pinned to `0.2.14` and is licensed
`MIT OR Apache-2.0`; versions `0.2.13` and earlier were GPL-3.0 and are not
used.

## Referenced and ported upstream projects

### llama.cpp / ggml — MIT

Ported components include GGML quantization kernels, IQ lattice tables, the
per-tensor scheme-selection policy engine, imatrix handling, and the per-row
block-size fallback logic.

> Copyright (c) 2023-2026 The ggml authors  
> https://github.com/ggml-org/llama.cpp

Line-level citations to upstream source files are preserved in the Rust source
comments.

### Comfy-Org/comfy-kitchen — Apache-2.0

The MXFP8 and NVFP4 eager quantization pipelines were used as numerical
references for the corresponding Rust kernels.

> Copyright (c) 2025 Comfy Org. All rights reserved.  
> https://github.com/Comfy-Org/comfy-kitchen

Verified against the vendored checkout: `LICENSE` is the Apache License 2.0
with the `Copyright (c) 2025 Comfy Org. All rights reserved.` appendix.

The vendored reference is version 0.2.35; the byte-parity goldens were
generated against 0.2.31. The numerical paths used here are unchanged between
those releases — verified by diff: `float_utils.py` is byte-identical, and the
bodies of `quantize_mxfp8` and `quantize_nvfp4` in
`backends/eager/quantization.py` hash identically across the two.

### ComfyUI-ZImage-Triton — MIT

The group-wise Hadamard rotation used by the `int8_convrot` preset descends from
ComfyUI-ZImage-Triton's `src/zimage_triton/quantization/hadamard.py`.

> Copyright (c) 2025 sewonkim  
> https://github.com/newgrit1004/ComfyUI-ZImage-Triton

Verified against the vendored checkout: `LICENSE` is the MIT License,
`Copyright (c) 2025 sewonkim`.

Note that the rotation actually implemented in this project is
`convert_to_quant`'s later revision of that file (see the notice below), not
ZImage's own current code: the ZImage revision builds a plain Sylvester matrix
via `scipy.linalg.hadamard`, whereas ctq added the Theorem-3.3 base H4
Kronecker construction to avoid Sylvester's all-ones column. The method
originates in QuaRot (2024) / ConvRot (2025).

### PyTorch — BSD-style license

The CPU MT19937 engine, normal-distribution kernels, and reduction behavior
were ported to reproduce the reference stream and bias-correction numerics.

> Copyright (c) 2016- Facebook, Inc. (Adam Paszke)  
> Copyright (c) 2014- Facebook, Inc. (Soumith Chintala)  
> Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)  
> Copyright (c) 2012-2014 Deepmind Technologies (Koray Kavukcuoglu)  
> Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)  
> Copyright (c) 2011-2013 NYU (Clement Farabet)  
> https://github.com/pytorch/pytorch

Redistributions must retain the upstream copyright notice, conditions, and
disclaimer.

### oneDNN — Apache-2.0

The K-chunked SGEMM accumulation behavior reproduced by the bias-correction
path is documented in the Rust source comments.

> Copyright 2016-2025 Intel Corporation  
> Copyright 2018 YANDEX LLC  
> Copyright 2019-2025 FUJITSU LIMITED  
> Copyright 2020-2025 Arm Ltd. and affiliates  
> Copyright 2020-2025 Codeplay Software Limited  
> https://github.com/oneapi-src/oneDNN

### convert_to_quant — MIT

The INT8, FP8 E4M3, MXFP8, and NVFP4 simple-path implementations, the
`comfy_quant` blob schema, and the calibration streaming behavior were derived
from:

> https://github.com/silveroxides/convert_to_quant
> Copyright (c) silveroxides and contributors
> Licensed under the MIT License

The MIT grant is declared in three independent places in the distributed
package: the `License:` field and the `License :: OSI Approved :: MIT
License` classifier in the wheel METADATA, and the license badge in the
project README. Note that no standalone `LICENSE` file is shipped inside the
installed package, so the grant rests on that metadata.

What was reimplemented in Rust is the *arithmetic* these modules specify —
symmetric INT8 scale formulas, the FP8 E4M3 cast and its stored reciprocal
scale, the MXFP8 e8m0 block-scale ladder, the NVFP4 E2M1 block layout and
nibble packing, the `.comfy_quant` JSON key ordering, and the calibration
draw order. No source text was copied: the Rust is an independent
implementation in idiomatic Rust (rayon parallel iterators, no transliterated
Python control flow), and upstream symbol names appear only in provenance
comments, never in code identifiers.

The affected Rust modules are listed by their module-level provenance
comments: `quant.rs`, `quant_fp8.rs`, `quant_mxfp8.rs`, `quant_nvfp4.rs`,
`comfy_schema.rs`, `convrot.rs`, `stream.rs`, and `bias_correction.rs`.
Note that `convrot.rs` is ultimately derived from ComfyUI-ZImage-Triton
(MIT) rather than from convert_to_quant — see its own notice above.

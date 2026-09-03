"""Phase 8.3 e2e sweep fixture generator (Unsloth coverage plan).

Builds, under tools/sweep_e2e/:
  model/            tiny HF model folder (model.safetensors + config.json)
                    — same layout as crates/quant-cli/tests/cli_gguf.rs
                    write_tiny_model(): 32x64 embed, 32x32 q_proj, 32x64
                    down_proj, 32 norm, 64x32 lm_head, all BF16 LCG-synth.
  imatrix.dat      legacy llama-quantize imatrix: one entry per 2-D tensor
                    (values = abs of the source floats, the natural
                    importance proxy; ncall=1 so sums pass through).

The model dims are deliberately small but 32/64-divisible so the llama
policy engine's categories all engage (attn_q/attn_v/attn_output,
ffn_gate/ffn_down/ffn_up, token_embd, output) with n_gqa=1.

Deterministic: the LCG matches cli_gguf.rs synth() exactly, so the fixture
is reproducible from both sides.
"""

import json
import struct
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "sweep_e2e"
H = 256  # hidden — must be a multiple of QK_K (256): the IQ quantizers
         # assert n_per_row % 256 == 0.
V = 512  # vocab


def synth(n: int, seed: int) -> list[float]:
    """Exact port of cli_gguf.rs synth(): 64-bit LCG -> [-1, 1)."""
    s = seed
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append(((s >> 33) / 0xFFFFFFFF) * 2.0 - 1.0)
    return out


def bf16_bytes(vals: list[float]) -> bytes:
    out = bytearray()
    for v in vals:
        # round-to-nearest f32 -> bf16 (top 16 bits of the u32 when the
        # low 16 rounds up, exactly half::bf16::from_f32)
        b = struct.pack("<f", v)
        u = struct.unpack("<I", b)[0]
        if u & 0x8000:
            u += 0x10000  # carry into the top half
        out += struct.pack("<H", u >> 16)
    return bytes(out)


def f32_bytes(vals: list[float]) -> bytes:
    return b"".join(struct.pack("<f", v) for v in vals)


TENSORS: list[tuple[str, str, list[int], bytes]] = []


def main() -> None:
    model = OUT / "model"
    model.mkdir(parents=True, exist_ok=True)

    tensors = [
        ("model.embed_tokens.weight", [V, H], bf16_bytes(synth(V * H, 1))),
        ("model.layers.0.self_attn.q_proj.weight", [H, H], bf16_bytes(synth(H * H, 2))),
        ("model.layers.0.self_attn.k_proj.weight", [H, H], bf16_bytes(synth(H * H, 3))),
        ("model.layers.0.self_attn.v_proj.weight", [H, H], bf16_bytes(synth(H * H, 4))),
        ("model.layers.0.self_attn.o_proj.weight", [H, H], bf16_bytes(synth(H * H, 5))),
        ("model.layers.0.mlp.gate_proj.weight", [H, H], bf16_bytes(synth(H * H, 6))),
        ("model.layers.0.mlp.up_proj.weight", [H, H], bf16_bytes(synth(H * H, 7))),
        ("model.layers.0.mlp.down_proj.weight", [H, H], bf16_bytes(synth(H * H, 8))),
        ("model.norm.weight", [H], bf16_bytes(synth(H, 11))),
        ("lm_head.weight", [V, H], bf16_bytes(synth(V * H, 12))),
    ]

    data = bytearray()
    header: dict[str, dict] = {}
    for name, shape, blob in tensors:
        start = len(data)
        data += blob
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [start, len(data)],
        }
    hjson = json.dumps(header, separators=(",", ":"))
    pad = (8 - len(hjson) % 8) % 8
    hjson += " " * pad
    (model / "model.safetensors").write_bytes(
        struct.pack("<Q", len(hjson)) + hjson.encode() + bytes(data)
    )

    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": H,
                "num_hidden_layers": 1,
                "vocab_size": V,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
            },
            indent=2,
        )
    )

    # Legacy imatrix: one entry per 2-D tensor, GGUF tensor names.
    # llama-quantize contract: the weight vector has ONE value per INPUT
    # COLUMN (n_per_row = ne[0] = in_features), not per element. All
    # tensors here share in_features = H = 32. Column importance proxy =
    # mean |value| down each column of the source matrix (ncall=1).
    gguf_names = [
        ("token_embd.weight", synth(V * H, 1)),
        ("blk.0.attn_q.weight", synth(H * H, 2)),
        ("blk.0.attn_k.weight", synth(H * H, 3)),
        ("blk.0.attn_v.weight", synth(H * H, 4)),
        ("blk.0.attn_output.weight", synth(H * H, 5)),
        ("blk.0.ffn_gate.weight", synth(H * H, 6)),
        ("blk.0.ffn_up.weight", synth(H * H, 7)),
        ("blk.0.ffn_down.weight", synth(H * H, 8)),
        ("output.weight", synth(V * H, 12)),
    ]
    out = bytearray()
    out += struct.pack("<i", len(gguf_names))  # n_entries
    for name, vals in gguf_names:
        # mean |v| per column (column j = vals[j::H]) — a natural
        # importance proxy that varies per column.
        col_importance = [
            sum(abs(vals[r * H + j]) for r in range(len(vals) // H)) / (len(vals) // H)
            for j in range(H)
        ]
        nb = name.encode()
        out += struct.pack("<i", len(nb))
        out += nb
        out += struct.pack("<i", 1)  # ncall
        out += struct.pack("<i", len(col_importance))
        for v in col_importance:
            out += struct.pack("<f", v)
    (OUT / "imatrix.dat").write_bytes(bytes(out))

    print(f"wrote {model} ({len(tensors)} tensors)")
    print(f"wrote {OUT / 'imatrix.dat'} ({len(gguf_names)} entries)")


if __name__ == "__main__":
    sys.exit(main())

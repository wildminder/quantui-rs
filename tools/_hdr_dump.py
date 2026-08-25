import struct
for p in ("tests/golden/linear_basic_bf16/input.safetensors", "tests/golden/conv_net/output.safetensors"):
    fh = open(p, "rb")
    n = struct.unpack("<Q", fh.read(8))[0]
    raw = fh.read(n)
    fh.close()
    print("==", p, "slot:", n)
    print(repr(raw[:220]))
    print("tail:", repr(raw[-40:]))

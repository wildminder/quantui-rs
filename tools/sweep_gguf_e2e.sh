#!/usr/bin/env bash
# Phase 8.3 e2e sweep (Unsloth coverage plan): convert the tiny HF model
# with EVERY usable GGUF method and validate each result.
#
# Usage: tools/sweep_gguf_e2e.sh   (from the workspace root; set BIN to
# override the binary path)
#
# iq* methods run twice: once expecting exit 2 WITHOUT --imatrix (the
# Unsloth gate), once with it. Everything else runs once, no imatrix.
# Exit 0 iff every method behaved as specified and every output is a
# loadable GGUF (magic check) with plausible size.

set -u
cd "$(dirname "$0")/.." || exit 1

# Portable default: the release binary is `.exe` on Windows and extensionless
# elsewhere. Callers can still override with BIN=... (see CONTRIBUTING.md).
if [ -z "${BIN:-}" ]; then
  if [ -x "./target/release/quantui-rs.exe" ]; then
    BIN="./target/release/quantui-rs.exe"
  else
    BIN="./target/release/quantui-rs"
  fi
fi
MODEL="tools/sweep_e2e/model"
IMATRIX="tools/sweep_e2e/imatrix.dat"
OUT="tools/sweep_e2e/out"
GGUF_PY="tools/sweep_e2e/gguf_py_check.py"

rm -rf "$OUT"; mkdir -p "$OUT"

NO_IMATRIX=(
  f16 bf16 f32 q8_0 q6_k q5_k_m q5_k_s q5_0 q5_1 q4_k_m q4_k_s q4_0 q4_1
  q3_k_m q3_k_l q3_k_s q3_k_xs q2_k q2_k_l tq1_0 tq2_0 q1_0 q2_0
)
WITH_IMATRIX=(
  iq4_nl iq4_xs iq3_xxs iq3_s iq3_m iq2_xxs iq2_xs iq2_s iq2_m iq1_s iq1_m
)

fail=0
pass=0

convert() {  # convert <method> [extra args...]
  local m="$1"; shift
  local out="$OUT/$m.gguf"
  if ! "$BIN" gguf "$MODEL" -m "$m" --imatrix "$IMATRIX" --no-progress "$out" "$@" >"$OUT/$m.log" 2>&1; then
    echo "FAIL convert $m (exit $?)"; sed -n '1,5p' "$OUT/$m.log"; fail=$((fail+1)); return 1
  fi
  # magic + plausible size
  if ! head -c4 "$out" | grep -q "GGUF"; then
    echo "FAIL $m: bad magic"; fail=$((fail+1)); return 1
  fi
  if [ ! -s "$out" ]; then
    echo "FAIL $m: empty output"; fail=$((fail+1)); return 1
  fi
  pass=$((pass+1)); return 0
}

# 1) plain methods (no imatrix passed at all)
for m in "${NO_IMATRIX[@]}"; do
  if ! "$BIN" gguf "$MODEL" -m "$m" --no-progress "$OUT/$m.gguf" >"$OUT/$m.log" 2>&1; then
    echo "FAIL convert $m"; sed -n '1,5p' "$OUT/$m.log"; fail=$((fail+1)); continue
  fi
  if ! head -c4 "$OUT/$m.gguf" | grep -q "GGUF"; then
    echo "FAIL $m: bad magic"; fail=$((fail+1)); continue
  fi
  pass=$((pass+1))
done

# 2) iq* WITHOUT --imatrix must exit 2 (Unsloth gate)
for m in "${WITH_IMATRIX[@]}"; do
  "$BIN" gguf "$MODEL" -m "$m" --no-progress "$OUT/$m.gate.gguf" >"$OUT/$m.gate.log" 2>&1
  code=$?
  if [ "$code" -ne 2 ]; then
    echo "FAIL gate $m: expected exit 2 without --imatrix, got $code"; fail=$((fail+1)); continue
  fi
  if [ -e "$OUT/$m.gate.gguf" ]; then
    echo "FAIL gate $m: output file exists despite exit 2"; fail=$((fail+1)); continue
  fi
  pass=$((pass+1))
done

# 3) iq* WITH --imatrix must convert
for m in "${WITH_IMATRIX[@]}"; do
  convert "$m" || true
done

# 4) rejected UD-*/Dynamic 2.0 must exit 2
for m in q4_nl q4_k_xl q3_k_xl q2_k_xl; do
  "$BIN" gguf "$MODEL" -m "$m" --no-progress "$OUT/$m.rej.gguf" >"$OUT/$m.rej.log" 2>&1
  code=$?
  if [ "$code" -ne 2 ]; then
    echo "FAIL reject $m: expected exit 2, got $code"; fail=$((fail+1)); continue
  fi
  pass=$((pass+1))
done

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ] || exit 1

# 5) summary table: every produced .gguf + size
echo
echo "produced outputs:"
python - "$OUT" <<'EOF'
import sys, pathlib
out = pathlib.Path(sys.argv[1])
for f in sorted(out.glob("*.gguf")):
    if ".gate." in f.name or ".rej." in f.name:
        continue
    print(f"  {f.name:<20} {f.stat().st_size:>10,} bytes")
EOF

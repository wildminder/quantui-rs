#!/bin/bash
# Build llama-quantize from the vendored upstream llama.cpp tree with MSVC,
# driven from Git Bash without vcvars (cmd.exe is blocked in this env).
#
# The MSVC / Windows SDK locations are machine-specific. Either run from a
# Visual Studio prompt (vcvarsall sets INCLUDE/LIB), or export these before
# invoking the script:
#   MSVC_BIN   path to the MSVC x64 bin dir (hosted cl.exe)
#   MSVC_INC   path to the MSVC include dir
#   MSVC_LIB   path to the MSVC x64 lib dir
#   SDK_INC_UM / SDK_INC_UCRT / SDK_INC_SHARED  Windows Kit include dirs
#   SDK_LIB_UCRT / SDK_LIB_UM                   Windows Kit lib dirs (x64)
#   NINJA      ninja executable to drive the build (default: PATH lookup)
#   CMAKE      cmake executable to configure the build (default: PATH lookup)
#   SRC        llama.cpp checkout to build (required, e.g. your local
#              llama.cpp source; the tree is not vendored in this repo)
#   BUILD      build dir (default: $TMPDIR/llamacpp-build)
# Usage: bash build_llamacpp.sh [target]
set -e

SRC="${SRC:?set SRC to your llama.cpp checkout}"
BUILD="${BUILD:-${TMPDIR:-/tmp}/llamacpp-build}"
NINJA="${NINJA:-ninja}"
CMAKE="${CMAKE:-cmake}"

MSVC_BIN="${MSVC_BIN:?set MSVC_BIN or run from a VS prompt}"
MSVC_INC="${MSVC_INC:?set MSVC_INC or run from a VS prompt}"
SDK_INC_UM="${SDK_INC_UM:?set SDK_INC_UM or run from a VS prompt}"
SDK_INC_UCRT="${SDK_INC_UCRT:?set SDK_INC_UCRT or run from a VS prompt}"
SDK_INC_SHARED="${SDK_INC_SHARED:?set SDK_INC_SHARED or run from a VS prompt}"
MSVC_LIB="${MSVC_LIB:?set MSVC_LIB or run from a VS prompt}"
SDK_LIB_UCRT="${SDK_LIB_UCRT:?set SDK_LIB_UCRT or run from a VS prompt}"
SDK_LIB_UM="${SDK_LIB_UM:?set SDK_LIB_UM or run from a VS prompt}"

# cl.exe wants Windows-style (backslash) include/lib path lists. Accept
# either /c/... or C:/... style inputs and normalize to C:\...\;C:\...
wpath() { sed -e 's|^/\([a-zA-Z]\)/|\1:/|' -e 's|/|\\|g' <<<"$1"; }
export INCLUDE="$(wpath "$MSVC_INC");$(wpath "$SDK_INC_UCRT");$(wpath "$SDK_INC_UM");$(wpath "$SDK_INC_SHARED")"
export LIB="$(wpath "$MSVC_LIB");$(wpath "$SDK_LIB_UCRT");$(wpath "$SDK_LIB_UM")"
export PATH="$MSVC_BIN:$PATH"
export CC=cl
export CXX=cl
export CMAKE_GENERATOR=Ninja

mkdir -p "$BUILD"
cd "$BUILD"

if [ ! -f build.ninja ]; then
  "$CMAKE" "$SRC" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA" \
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_SSE42=OFF -DGGML_SSE3=OFF
fi

"$NINJA" "${1:-llama-quantize}"
echo "BUILD_OK: $(ls -la bin/llama-quantize.exe 2>/dev/null || echo 'target not found')"

#!/bin/bash
# Build llama-quantize from the vendored docs/ref/llama.cpp with MSVC,
# driven from Git Bash without vcvars (cmd.exe is blocked in this env).
# Usage: bash build_llamacpp.sh [target]
set -e

MSVC_BIN="<LOCAL-TOOLS>/Dev/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
MSVC_INC="<LOCAL-TOOLS>\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\include"
SDK_INC_UM="C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\um"
SDK_INC_UCRT="C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\ucrt"
SDK_INC_SHARED="C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\shared"
MSVC_LIB="<LOCAL-TOOLS>\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\lib\\x64"
SDK_LIB_UCRT="C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.22621.0\\ucrt\\x64"
SDK_LIB_UM="C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.22621.0\\um\\x64"

export INCLUDE="$MSVC_INC;$SDK_INC_UCRT;$SDK_INC_UM;$SDK_INC_SHARED"
export LIB="$MSVC_LIB;$SDK_LIB_UCRT;$SDK_LIB_UM"
export PATH="/c/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64:$MSVC_BIN:$PATH"
export CC=cl
export CXX=cl
export CMAKE_GENERATOR=Ninja

SRC="<REPO-DIR>/docs/ref/llama.cpp"
BUILD="<LOCAL-BUILD>"
CMAKE="<LOCAL-TOOLS>/Dev/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
NINJA="${NINJA:-ninja}"
NINJA_W="${NINJA_W:-ninja}"

mkdir -p "$BUILD"
cd "$BUILD"

if [ ! -f build.ninja ]; then
  "$CMAKE" "$SRC" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA_W" \
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_SSE42=OFF -DGGML_SSE3=OFF
fi

"$NINJA" "${1:-llama-quantize}"
echo "BUILD_OK: $(ls -la bin/llama-quantize.exe 2>/dev/null || echo 'target not found')"

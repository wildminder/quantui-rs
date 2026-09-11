/* Debug harness 2: call the BUILT ggml.dll's ggml_quantize_chunk with
 * our fixture row + weights, byte-compare against the verbatim-C harness.
 * Build (Git Bash + MSVC env, from tools/):
 *   cl /nologo /O2 qp_dll_dbg.c /link ..<llamacpp-build>/bin/ggml.lib
 * Simpler: load ggml.dll at runtime via LoadLibraryA to avoid lib paths.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

typedef int64_t i64;
typedef uint32_t u32;

/* ggml_quantize_chunk(enum ggml_type, const float*, void*, i64 start,
 *                     i64 nrows, i64 n_per_row, const float * imatrix);
 * GGML_TYPE_Q4_K = 12.
 */
typedef i64 (*quantize_chunk_fn)(u32, const void *, void *, i64, i64, i64, const void *);

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <ggml.dll> <src.f32.bin> <weights.f32.bin>\n", argv[0]);
        return 1;
    }
    HMODULE g = LoadLibraryA(argv[1]);
    if (!g) { fprintf(stderr, "LoadLibrary failed: %lu\n", GetLastError()); return 1; }
    quantize_chunk_fn qc = (quantize_chunk_fn) GetProcAddress(g, "ggml_quantize_chunk");
    if (!qc) { fprintf(stderr, "no ggml_quantize_chunk: %lu\n", GetLastError()); return 1; }

    FILE * fs = fopen(argv[2], "rb"), * fw = fopen(argv[3], "rb");
    if (!fs || !fw) { fprintf(stderr, "open failed\n"); return 1; }
    static float x[512], w[256];
    fread(x, 4, 512, fs);
    fread(w, 4, 256, fw);
    fclose(fs); fclose(fw);

    /* GGML_TYPE_Q4_K = 12, GGML_TYPE_Q2_K = 10. One row at a time,
     * n_per_row 256, weights = the per-tensor vector (llama-quant.cpp
     * passes the same base per row). */
    static uint8_t out[2][144];
    for (int row = 0; row < 2; ++row) {
        i64 n = qc(12 /* Q4_K */, x + row*256, out[row], 0, 1, 256, w);
        printf("row %d: ggml_quantize_chunk -> %lld bytes\n", row, (long long)n);
        for (int i = 0; i < 16; ++i) printf("%02x ", out[row][i]);
        printf("\n");
    }
    return 0;
}

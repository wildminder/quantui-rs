/* Debug harness: run llama.cpp's make_qkx3_quants + make_qp_quants
 * verbatim (copied from upstream llama.cpp ggml-quants.c) on the same
 * fixture data as the Rust parity test, and print the intermediate
 * scales/mins so the Rust port can be diffed against it.
 *
 * Build (Git Bash, MSVC env set):
 *   cl /O2 /Fe:qp_dbg.exe qp_quants_dbg.c
 * Usage: qp_dbg.exe <src.f32.bin> <weights.f32.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#define GGML_RESTRICT __restrict
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

typedef uint8_t  u8;
typedef int8_t   i8;
typedef uint16_t u16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef float    f32;

#define QK_K 256
#define GROUP_MAX_EPS 1e-15f

static inline int nearest_int(f32 fval) {
    assert(fabsf(fval) <= 4194303.f);
    f32 val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

/* ===== verbatim from ggml-quants.c (only the two helpers) ===== */

static float make_qkx3_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0]*x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max <= min) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff*diff;
        float w = weights ? weights[i] : x[i]*x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff*diff;
                float w = weights ? weights[i] : x[i]*x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float make_qp_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L, const float * GGML_RESTRICT quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    if (max < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale*L[i];
        float w = quant_weights[i];
        best_mse += w*diff*diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]);
            l = MIN(nmax, l);
            float diff = x[i] - scale_is*l;
            float w = quant_weights[i];
            mse += w*diff*diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

/* ===== harness: replicate the Q4_K impl's first-pass for one row ===== */
#define restrict_var
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s src.bin weights.bin\n", argv[0]); return 1; }
    FILE * fs = fopen(argv[1], "rb"), * fw = fopen(argv[2], "rb");
    if (!fs || !fw) { fprintf(stderr, "open failed\n"); return 1; }
    static float x[QK_K], qw[QK_K];
    fread(x, 4, QK_K, fs);
    fread(qw, 4, QK_K, fw);
    fclose(fs); fclose(fw);

    uint8_t L[QK_K], Laux[32], Ls[QK_K/32], Lm[QK_K/32];
    float weights[32], sw[QK_K/32], mins[QK_K/32], scales[QK_K/32];

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; ++l) sum_x2 += x[l]*x[l];
    float sigma2 = 2*sum_x2/QK_K;

    for (int j = 0; j < QK_K/32; ++j) {
        const float * q = qw + 32*j;
        for (int l = 0; l < 32; ++l) weights[l] = q[l] * sqrtf(sigma2 + x[32*j+l]*x[32*j+l]);
        float sumw = 0;
        for (int l = 0; l < 32; ++l) sumw += weights[l];
        sw[j] = sumw;
        scales[j] = make_qkx3_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        printf("group %d: scale=%.8g min=%.8g\n", j, scales[j], mins[j]);
    }
    float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
    float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);
    printf("d_block=%.8g m_block=%.8g\n", d_block, m_block);
    printf("Ls:"); for (int j = 0; j < 8; ++j) printf(" %d", Ls[j]);
    printf("\nLm:"); for (int j = 0; j < 8; ++j) printf(" %d", Lm[j]);
    printf("\n");
    return 0;
}

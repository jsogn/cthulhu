/* DCT 重量化原生加速：8x8 正交 DCT-II 前向 + 量化 + 逆变换单次融合。
 *
 * 语义对齐 transform/extra_attacks.py::_requant_plane：
 *   - float32 平面，宽高必须为 8 的整数倍（调用方保证，否则走 numpy）；
 *   - 量化取整用 rintf（IEEE 半偶舍入），与 np.round 一致；
 *   - 不裁剪输出（clip 由上层 dct_requant 负责）。
 *
 * 构建：scripts/build_native_dct.sh（macOS clang）。
 */
#include <stddef.h>
#include <math.h>

static float M[8][8];
static int initialized = 0;

static void init_coeffs(void) {
    for (int k = 0; k < 8; k++) {
        for (int n = 0; n < 8; n++) {
            float v = cosf(3.14159265358979f / 8.0f * (n + 0.5f) * (float)k);
            v *= 0.5f;                 /* sqrt(2/8) */
            if (k == 0) v *= 0.70710678118655f;  /* 1/sqrt(2) */
            M[k][n] = v;
        }
    }
    initialized = 1;
}

void dct_requant_plane(const float* in, float* out, int H, int W, float step) {
    if (!initialized) init_coeffs();
    const int bh = H / 8, bw = W / 8;
    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            float tmp[8][8];
            for (int i = 0; i < 8; i++)
                for (int l = 0; l < 8; l++) {
                    float s = 0.0f;
                    for (int k = 0; k < 8; k++)
                        s += M[i][k] * in[(by * 8 + k) * W + (bx * 8 + l)];
                    tmp[i][l] = s;
                }
            float c[8][8];
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) {
                    float s = 0.0f;
                    for (int l = 0; l < 8; l++)
                        s += tmp[i][l] * M[j][l];
                    c[i][j] = rintf(s / step) * step;
                }
            float t[8][8];
            for (int i = 0; i < 8; i++)
                for (int l = 0; l < 8; l++) {
                    float s = 0.0f;
                    for (int k = 0; k < 8; k++)
                        s += M[k][i] * c[k][l];
                    t[i][l] = s;
                }
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) {
                    float s = 0.0f;
                    for (int l = 0; l < 8; l++)
                        s += t[i][l] * M[l][j];
                    out[(by * 8 + i) * W + (bx * 8 + j)] = s;
                }
        }
    }
}

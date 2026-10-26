#include "support.h"
#ifdef HOST_TEST
static float oracle_exp(float x) { return (float)exp((double)x); }
static float oracle_sqrt(float x) { return (float)sqrt((double)x); }
static float oracle_tanh(float x) { return (float)tanh((double)x); }
#else
static float oracle_exp(float x) { return expf(x); }
static float oracle_sqrt(float x) { return sqrtf(x); }
static float oracle_tanh(float x) { return tanhf(x); }
#endif
static void compare(float actual, float expected, float atol, float rtol,
                    int *errors, float *maximum) {
  if (actual == expected)
    return;
  if (!__builtin_isfinite(actual) || !__builtin_isfinite(expected)) {
    ++*errors;
    *maximum = __builtin_inff();
    return;
  }
  if (!check_close(actual, expected, atol, rtol))
    ++*errors;
  float d = actual - expected;
  if (d < 0)
    d = -d;
  if (d > *maximum)
    *maximum = d;
}
extern void _mlir_ciface_kernel_patch_embed_196x768(MemRef4 *, MemRef4 *,
                                                    MemRef1 *, MemRef4 *);
int launch(void) {
  const int H = 224, W = 224, C = 3, P = 16, O = 768, G = 14;
  size_t xbytes = (size_t)C * H * W * sizeof(float),
         wbytes = (size_t)O * C * P * P * sizeof(float);
  size_t woff = (xbytes + 63) & ~(size_t)63;
  size_t boff = (woff + wbytes + 63) & ~(size_t)63;
  size_t ooff = (boff + O * sizeof(float) + 63) & ~(size_t)63;
  float *x = (float *)workspace(0), *w = (float *)workspace(woff);
  float *bias = (float *)workspace(boff);
  float *out = (float *)workspace(ooff);
  for (int i = 0; i < C * H * W; ++i)
    x[i] = (float)((i * 5) % 31 - 15) * 0.00390625f;
  for (int i = 0; i < O * C * P * P; ++i)
    w[i] = (float)((i * 3) % 17 - 8) * 0.001953125f;
  for (int i = 0; i < O; ++i)
    bias[i] = (float)(i % 13 - 6) * 0.00390625f;
  MemRef4 X = make_4(x, 1, C, H, W);
  MemRef4 Weight = make_4(w, O, C, P, P);
  MemRef1 Bias = make_1(bias, O);
  MemRef4 Out = make_4(out, 1, O, G, G);
  _mlir_ciface_kernel_patch_embed_196x768(&X, &Weight, &Bias, &Out);
  int errors = 0;
  float maximum = 0.0f;
  for (int p = 0; p < G * G; ++p) {
    int py = p / G, px = p % G;
    for (int o = 0; o < O; ++o) {
      float want = bias[o];
      for (int c = 0; c < C; ++c)
        for (int dy = 0; dy < P; ++dy)
          for (int dx = 0; dx < P; ++dx)
            want += x[c * H * W + (py * P + dy) * W + px * P + dx] *
                    w[o * C * P * P + c * P * P + dy * P + dx];
      compare(out[o * G * G + p], want, 3e-3f, 3e-3f, &errors, &maximum);
    }
  }
  return print_check("patch_embed_196x768", errors, maximum);
}

#include "support.h"
#ifdef HOST_TEST
static float oracle_exp(float x) { return (float)exp((double)x); }
static float oracle_sqrt(float x) { return (float)sqrt((double)x); }
static float oracle_erf(float x) { return (float)erf((double)x); }
#else
static float oracle_exp(float x) { return expf(x); }
static float oracle_sqrt(float x) { return sqrtf(x); }
static float oracle_erf(float x) { return erff(x); }
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
extern void _mlir_ciface_kernel_attention_qk_12x128x128x64(MemRef3 *, MemRef3 *, MemRef3 *);
int launch(void) {
  const int H = 12, M = 128, N = 128, K = 64;
  size_t off = 0;
  float *a = (float *)workspace(off);
  off = vit_align64(off + (size_t)H * M * K * 4u);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)H * K * N * 4u);
  float *c = (float *)workspace(off);
  for (int i = 0; i < H * M * K; ++i)
    a[i] = (i % 29 - 14) * 0.0078125f;
  for (int i = 0; i < H * K * N; ++i)
    b[i] = (i % 23 - 11) * 0.01171875f;
  MemRef3 A = make_3(a, H, M, K), B = make_3(b, H, K, N), C = make_3(c, H, M, N);
  _mlir_ciface_kernel_attention_qk_12x128x128x64(&A, &B, &C);
  int errors = 0;
  float maximum = 0;
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        float want = 0;
        for (int q = 0; q < K; ++q)
          want += a[h * M * K + i * K + q] * b[h * K * N + q * N + j];
        want *= 0.125f;
        compare(c[h * M * N + i * N + j], want, 5e-3f, 5e-3f, &errors, &maximum);
      }
  return print_check("attention_qk_12x128x128x64", errors, maximum);
}

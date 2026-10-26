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
extern void _mlir_ciface_kernel_attention_pv_12x197x64x197(MemRef3 *, MemRef3 *,
                                                           MemRef3 *);
int launch(void) {
  const int H = 12, M = 197, N = 64, K = 197;
  float *a = (float *)workspace(0),
        *b = (float *)workspace(((size_t)H * M * K * 4 + 63) & ~(size_t)63),
        *c = (float *)workspace(
            ((((size_t)H * M * K + (size_t)H * K * N) * 4 + 63) & ~(size_t)63));
  for (int i = 0; i < H * M * K; ++i)
    a[i] = (i % 29 - 14) * 0.0078125f;
  for (int i = 0; i < H * K * N; ++i)
    b[i] = (i % 23 - 11) * 0.01171875f;
  MemRef3 A = make_3(a, H, M, K), B = make_3(b, H, K, N),
          C = make_3(c, H, M, N);
  _mlir_ciface_kernel_attention_pv_12x197x64x197(&A, &B, &C);
  int errors = 0;
  float maximum = 0;
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        float want = 0;
        for (int q = 0; q < K; ++q)
          want += a[h * M * K + i * K + q] * b[h * K * N + q * N + j];
        compare(c[h * M * N + i * N + j], want, 5e-3f, 5e-3f, &errors,
                &maximum);
      }
  return print_check("attention_pv_12x197x64x197", errors, maximum);
}

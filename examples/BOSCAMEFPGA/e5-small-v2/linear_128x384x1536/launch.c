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
extern void _mlir_ciface_kernel_linear_128x384x1536(MemRef2 *, MemRef2 *, MemRef2 *);
int launch(void) {
  const int M = 128, N = 384, K = 1536;
  size_t off = 0;
  float *a = (float *)workspace(off);
  off = vit_align64(off + (size_t)M * K * 4u);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)N * K * 4u);
  float *c = (float *)workspace(off);
  for (int i = 0; i < M * K; ++i)
    a[i] = (float)((i * 13) % 29 - 14) * 0.0078125f;
  for (int i = 0; i < N * K; ++i)
    b[i] = (float)((i * 7) % 23 - 11) * 0.01171875f;
  for (int i = 0; i < M * N; ++i)
    c[i] = 0.0f;
  MemRef2 A = make_2(a, M, K), B = make_2(b, N, K), C = make_2(c, M, N);
  _mlir_ciface_kernel_linear_128x384x1536(&A, &B, &C);
  int errors = 0;
  float maximum = 0.0f;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float want = 0.0f;
      for (int q = 0; q < K; ++q)
        want += a[i * K + q] * b[j * K + q];
      compare(c[i * N + j], want, 3e-3f, 3e-3f, &errors, &maximum);
    }
  return print_check("linear_128x384x1536", errors, maximum);
}

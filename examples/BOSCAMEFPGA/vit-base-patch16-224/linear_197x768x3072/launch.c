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
extern void _mlir_ciface_kernel_linear_197x768x3072(MemRef2 *, MemRef2 *,
                                                    MemRef2 *);
int launch(void) {
  float *a = (float *)workspace(0),
        *b = (float *)workspace(
            (((size_t)197 * 3072 * sizeof(float) + 63) & ~(size_t)63));
  size_t co = ((size_t)197 * 3072 + (size_t)768 * 3072) * sizeof(float) + 128;
  float *c = (float *)workspace(co & ~(size_t)63);
  for (int i = 0; i < 197 * 3072; ++i)
    a[i] = (float)((i * 13) % 29 - 14) * 0.0078125f;
  for (int i = 0; i < 768 * 3072; ++i)
    b[i] = (float)((i * 7) % 23 - 11) * 0.01171875f;
  for (int i = 0; i < 197 * 768; ++i)
    c[i] = 0.0f;
  MemRef2 A = make_2(a, 197, 3072), B = make_2(b, 768, 3072),
          C = make_2(c, 197, 768);
  _mlir_ciface_kernel_linear_197x768x3072(&A, &B, &C);
  int errors = 0;
  float maximum = 0.0f;
  for (int i = 0; i < 197; ++i)
    for (int j = 0; j < 768; ++j) {
      float want = 0.0f;
      for (int q = 0; q < 3072; ++q)
        want += a[i * 3072 + q] * b[j * 3072 + q];
      compare(c[i * 768 + j], want, 3e-3f, 3e-3f, &errors, &maximum);
    }
  return print_check("linear_197x768x3072", errors, maximum);
}

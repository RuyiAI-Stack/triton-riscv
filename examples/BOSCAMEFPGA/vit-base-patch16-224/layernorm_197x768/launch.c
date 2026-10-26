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
extern void _mlir_ciface_kernel_layernorm_197x768(MemRef2 *, MemRef1 *,
                                                  MemRef1 *, MemRef2 *);
int launch(void) {
  float *x = (float *)workspace(0),
        *w = (float *)workspace(((size_t)197 * 768 * 4 + 63) & ~(size_t)63);
  float *b =
      (float *)workspace(((((size_t)197 * 768 + 768) * 4 + 63) & ~(size_t)63));
  float *o = (float *)workspace(
      ((((size_t)197 * 768 + 2 * 768) * 4 + 63) & ~(size_t)63));
  for (int i = 0; i < 197 * 768; ++i)
    x[i] = (float)((i * 5) % 37 - 18) * 0.015625f;
  for (int i = 0; i < 768; ++i) {
    w[i] = 1.0f + (i % 11) * 0.01f;
    b[i] = (i % 17 - 8) * 0.00390625f;
  }
  MemRef2 X = make_2(x, 197, 768), O = make_2(o, 197, 768);
  MemRef1 W = make_1(w, 768), B = make_1(b, 768);
  _mlir_ciface_kernel_layernorm_197x768(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int r = 0; r < 197; ++r) {
    float mean = 0, var = 0;
    for (int j = 0; j < 768; ++j)
      mean += x[r * 768 + j];
    mean /= 768;
    for (int j = 0; j < 768; ++j) {
      float d = x[r * 768 + j] - mean;
      var += d * d;
    }
    var /= 768;
    for (int j = 0; j < 768; ++j) {
      float want =
          (x[r * 768 + j] - mean) / oracle_sqrt(var + 1e-12f) * w[j] + b[j];
      compare(o[r * 768 + j], want, 4e-4f, 4e-4f, &errors, &maximum);
    }
  }
  return print_check("layernorm_197x768", errors, maximum);
}

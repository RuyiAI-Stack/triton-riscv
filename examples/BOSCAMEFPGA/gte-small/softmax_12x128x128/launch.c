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
extern void _mlir_ciface_kernel_softmax_12x128x128(MemRef3 *, MemRef3 *);
int launch(void) {
  const int H = 12, L = 128, R = H * L, COUNT = R * L;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)COUNT * 4u);
  float *o = (float *)workspace(off);
  for (int i = 0; i < COUNT; ++i)
    x[i] = (i % 31 - 15) * 0.03125f;
  MemRef3 X = make_3(x, H, L, L), O = make_3(o, H, L, L);
  _mlir_ciface_kernel_softmax_12x128x128(&X, &O);
  int errors = 0;
  float maximum = 0;
  for (int r = 0; r < R; ++r) {
    float max = -__builtin_inff(), sum = 0;
    for (int j = 0; j < L; ++j)
      if (x[r * L + j] > max)
        max = x[r * L + j];
    for (int j = 0; j < L; ++j)
      sum += oracle_exp(x[r * L + j] - max);
    for (int j = 0; j < L; ++j)
      compare(o[r * L + j], oracle_exp(x[r * L + j] - max) / sum, 4e-4f, 4e-4f,
              &errors, &maximum);
  }
  return print_check("softmax_12x128x128", errors, maximum);
}

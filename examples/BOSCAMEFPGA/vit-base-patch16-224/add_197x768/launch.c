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
extern void _mlir_ciface_kernel_add_197x768(MemRef2 *, MemRef2 *, MemRef2 *);
int launch(void) {
  float *x = (float *)workspace(0),
        *y = (float *)workspace(((size_t)151296 * 4 + 63) & ~(size_t)63),
        *o = (float *)workspace(((size_t)151296 * 8 + 127) & ~(size_t)63);
  for (int i = 0; i < 151296; ++i) {
    x[i] = (i % 31) * 0.0078125f;
    y[i] = (i % 17 - 8) * 0.015625f;
    o[i] = 0;
  }
  MemRef2 X = make_2(x, 197, 768), Y = make_2(y, 197, 768),
          O = make_2(o, 197, 768);
  _mlir_ciface_kernel_add_197x768(&X, &Y, &O);
  int errors = 0;
  float maximum = 0;
  for (int i = 0; i < 151296; ++i)
    compare(o[i], x[i] + y[i], 1e-6f, 1e-6f, &errors, &maximum);
  return print_check("add_197x768", errors, maximum);
}

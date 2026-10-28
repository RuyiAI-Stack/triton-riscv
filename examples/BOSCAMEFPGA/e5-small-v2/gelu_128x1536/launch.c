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
extern void _mlir_ciface_kernel_gelu_128x1536(MemRef2 *, MemRef2 *);
int launch(void) {
  const int COUNT = 196608;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)COUNT * 4u);
  float *o = (float *)workspace(off);
  for (int i = 0; i < COUNT; ++i)
    x[i] = (float)((i * 7) % 41 - 20) * 0.03125f;
  MemRef2 X = make_2(x, 128, 1536), O = make_2(o, 128, 1536);
  _mlir_ciface_kernel_gelu_128x1536(&X, &O);
  int errors = 0;
  float maximum = 0;
  for (int i = 0; i < COUNT; ++i) {
    float v = x[i], want = 0.5f * v * (1.0f + oracle_erf(v * 0.7071067811865475f));
    compare(o[i], want, 3e-4f, 3e-4f, &errors, &maximum);
  }
  return print_check("gelu_128x1536", errors, maximum);
}

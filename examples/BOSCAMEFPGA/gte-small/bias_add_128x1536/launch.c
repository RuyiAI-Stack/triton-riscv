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
extern void _mlir_ciface_kernel_bias_add_128x1536(MemRef2 *, MemRef1 *, MemRef2 *);
int launch(void) {
  const int ROWS = 128, WIDTH = 1536, COUNT = ROWS * WIDTH;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)COUNT * 4u);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)WIDTH * 4u);
  float *o = (float *)workspace(off);
  for (int i = 0; i < COUNT; ++i) {
    x[i] = (i % 31 - 15) * 0.0078125f;
    o[i] = 0;
  }
  for (int i = 0; i < WIDTH; ++i)
    b[i] = (i % 17 - 8) * 0.015625f;
  MemRef2 X = make_2(x, ROWS, WIDTH), O = make_2(o, ROWS, WIDTH);
  MemRef1 B = make_1(b, WIDTH);
  _mlir_ciface_kernel_bias_add_128x1536(&X, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int i = 0; i < COUNT; ++i)
    compare(o[i], x[i] + b[i % WIDTH], 1e-6f, 1e-6f, &errors, &maximum);
  return print_check("bias_add_128x1536", errors, maximum);
}

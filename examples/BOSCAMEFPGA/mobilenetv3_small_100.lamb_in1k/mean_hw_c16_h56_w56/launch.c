#include "support.h"
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

extern void _mlir_ciface_kernel_mean_hw_c16_h56_w56(MemRef1 *, MemRef1 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)50176 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)16 * 4u);

  for (int i = 0; i < 50176; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 16; ++i)
    o[i] = 0.0f;

  MemRef1 X = make_1(x, 50176), O = make_1(o, 16);
  _mlir_ciface_kernel_mean_hw_c16_h56_w56(&X, &O);
  int errors = 0;
  float maximum = 0;
  for (int c = 0; c < 16; ++c) {
    double sum = 0;
    for (int s = 0; s < 3136; ++s) sum += x[c * 3136 + s];
    compare(o[c], (float)(sum / 3136), 1e-3f, 1e-3f, &errors, &maximum);
  }
  return print_check("mean_hw_c16_h56_w56", errors, maximum);
}

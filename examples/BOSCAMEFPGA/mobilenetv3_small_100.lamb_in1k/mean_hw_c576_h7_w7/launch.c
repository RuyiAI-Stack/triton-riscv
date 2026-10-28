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

extern void _mlir_ciface_kernel_mean_hw_c576_h7_w7(MemRef1 *, MemRef1 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)28224 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)576 * 4u);

  for (int i = 0; i < 28224; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 576; ++i)
    o[i] = 0.0f;

  MemRef1 X = make_1(x, 28224), O = make_1(o, 576);
  _mlir_ciface_kernel_mean_hw_c576_h7_w7(&X, &O);
  int errors = 0;
  float maximum = 0;
  for (int c = 0; c < 576; ++c) {
    double sum = 0;
    for (int s = 0; s < 49; ++s) sum += x[c * 49 + s];
    compare(o[c], (float)(sum / 49), 1e-3f, 1e-3f, &errors, &maximum);
  }
  return print_check("mean_hw_c576_h7_w7", errors, maximum);
}

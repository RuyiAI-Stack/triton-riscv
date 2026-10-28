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

extern void _mlir_ciface_kernel_mean_hw_c240_h14_w14(MemRef1 *, MemRef1 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)47040 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)240 * 4u);

  for (int i = 0; i < 47040; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 240; ++i)
    o[i] = 0.0f;

  MemRef1 X = make_1(x, 47040), O = make_1(o, 240);
  _mlir_ciface_kernel_mean_hw_c240_h14_w14(&X, &O);
  int errors = 0;
  float maximum = 0;
  for (int c = 0; c < 240; ++c) {
    double sum = 0;
    for (int s = 0; s < 196; ++s) sum += x[c * 196 + s];
    compare(o[c], (float)(sum / 196), 1e-3f, 1e-3f, &errors, &maximum);
  }
  return print_check("mean_hw_c240_h14_w14", errors, maximum);
}

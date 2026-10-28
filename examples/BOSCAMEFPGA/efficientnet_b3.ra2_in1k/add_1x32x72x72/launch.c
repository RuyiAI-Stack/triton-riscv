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
extern void _mlir_ciface_kernel_add_1x32x72x72(MemRef4 *, MemRef4 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)165888 * 4u);
  float *y = (float *)workspace(off);
  off = vit_align64(off + (size_t)165888 * 4u);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)165888 * 4u);
  for (int i = 0; i < 165888; ++i) {
    x[i] = (i % 31) * 0.0078125f;
    y[i] = (i % 17 - 8) * 0.015625f;
    o[i] = 0;
  }
  MemRef4 X = make_4(x, 1, 32, 72, 72), Y = make_4(y, 1, 32, 72, 72), O = make_4(o, 1, 32, 72, 72);
  _mlir_ciface_kernel_add_1x32x72x72(&X, &Y, &O);
  for (int i = 0; i < 165888; ++i)
    compare(o[i], x[i] + y[i], 1e-6f, 1e-6f, &errors, &maximum);
  return print_check("add_1x32x72x72", errors, maximum);
}

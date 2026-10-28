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

extern void _mlir_ciface_kernel_add_count9408(MemRef1 *, MemRef1 *, MemRef1 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)9408 * 4u);

  float *y = (float *)workspace(off);
  off = vit_align64(off + (size_t)9408 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)9408 * 4u);

  for (int i = 0; i < 9408; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 9408; ++i)
    y[i] = (float)((i * 13) % 23 - 11) * 0.03125f;

  for (int i = 0; i < 9408; ++i)
    o[i] = 0.0f;

  MemRef1 X = make_1(x, 9408), Y = make_1(y, 9408), O = make_1(o, 9408);
  _mlir_ciface_kernel_add_count9408(&X, &Y, &O);
  int errors = 0;
  float maximum = 0;
  for (int i = 0; i < 9408; ++i)
    compare(o[i], x[i] + y[i], 0, 0, &errors, &maximum);
  return print_check("add_count9408", errors, maximum);
}

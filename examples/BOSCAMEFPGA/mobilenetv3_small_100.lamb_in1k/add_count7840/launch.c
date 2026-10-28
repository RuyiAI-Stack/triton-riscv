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

extern void _mlir_ciface_kernel_add_count7840(MemRef1 *, MemRef1 *, MemRef1 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)7840 * 4u);

  float *y = (float *)workspace(off);
  off = vit_align64(off + (size_t)7840 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)7840 * 4u);

  for (int i = 0; i < 7840; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 7840; ++i)
    y[i] = (float)((i * 13) % 23 - 11) * 0.03125f;

  for (int i = 0; i < 7840; ++i)
    o[i] = 0.0f;

  MemRef1 X = make_1(x, 7840), Y = make_1(y, 7840), O = make_1(o, 7840);
  _mlir_ciface_kernel_add_count7840(&X, &Y, &O);
  int errors = 0;
  float maximum = 0;
  for (int i = 0; i < 7840; ++i)
    compare(o[i], x[i] + y[i], 0, 0, &errors, &maximum);
  return print_check("add_count7840", errors, maximum);
}

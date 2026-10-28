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

extern void _mlir_ciface_kernel_relu_count24(MemRef1 *, MemRef1 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)24 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)24 * 4u);

  for (int i = 0; i < 24; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 24; ++i)
    o[i] = 0.0f;

  MemRef1 X = make_1(x, 24), O = make_1(o, 24);
  _mlir_ciface_kernel_relu_count24(&X, &O);
  int errors = 0;
  float maximum = 0;
  for (int i = 0; i < 24; ++i)
    compare(o[i], x[i] > 0.0f ? x[i] : 0.0f, 0, 0, &errors, &maximum);
  return print_check("relu_count24", errors, maximum);
}

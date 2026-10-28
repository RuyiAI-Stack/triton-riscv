#include "support.h"
#include <math.h>
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
extern void _mlir_ciface_kernel_silu_1x6x1x1(MemRef4 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)6 * 4u);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)6 * 4u);
  for (int i = 0; i < 6; ++i) {
    x[i] = (i % 31) * 0.0078125f;
    o[i] = 0;
  }
  MemRef4 X = make_4(x, 1, 6, 1, 1), O = make_4(o, 1, 6, 1, 1);
  _mlir_ciface_kernel_silu_1x6x1x1(&X, &O);
  for (int i = 0; i < 6; ++i) {
    float expected = x[i] / (1.0f + expf(-x[i]));
    compare(o[i], expected, 2e-5f, 2e-5f, &errors, &maximum);
  }
  return print_check("silu_1x6x1x1", errors, maximum);
}

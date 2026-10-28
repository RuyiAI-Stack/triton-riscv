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

extern void _mlir_ciface_kernel_se_mul_c576_h7_w7(MemRef1 *, MemRef1 *, MemRef1 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)28224 * 4u);

  float *scale = (float *)workspace(off);
  off = vit_align64(off + (size_t)576 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)28224 * 4u);

  for (int i = 0; i < 28224; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 576; ++i)
    scale[i] = (float)((i * 3) % 7 - 3) * 0.125f;

  for (int i = 0; i < 28224; ++i)
    o[i] = 0.0f;

  MemRef1 X = make_1(x, 28224), S = make_1(scale, 576), O = make_1(o, 28224);
  _mlir_ciface_kernel_se_mul_c576_h7_w7(&X, &S, &O);
  int errors = 0;
  float maximum = 0;
  for (int c = 0; c < 576; ++c)
    for (int s = 0; s < 49; ++s)
      compare(o[c * 49 + s], x[c * 49 + s] * scale[c], 1e-6f, 1e-6f, &errors, &maximum);
  return print_check("se_mul_c576_h7_w7", errors, maximum);
}

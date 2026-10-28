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

extern void _mlir_ciface_kernel_pwconv_cin288_cout96_h7_w7(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)14112 * 4u);

  float *weight = (float *)workspace(off);
  off = vit_align64(off + (size_t)27648 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)96 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)4704 * 4u);

  for (int i = 0; i < 14112; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 27648; ++i)
    weight[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 96; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 4704; ++i)
    o[i] = 0.0f;

  MemRef4 X = make_4(x, 1, 288, 7, 7);
  MemRef4 W = make_4(weight, 96, 288, 1, 1);
  MemRef1 B = make_1(b, 96);
  MemRef4 O = make_4(o, 1, 96, 7, 7);
  _mlir_ciface_kernel_pwconv_cin288_cout96_h7_w7(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int oc = 0; oc < 96; ++oc)
    for (int s = 0; s < 49; ++s) {
      double acc = b[oc];
      for (int ic = 0; ic < 288; ++ic)
        acc += (double)x[ic * 49 + s] * weight[oc * 288 + ic];
      compare(o[oc * 49 + s], (float)acc, 1e-3f, 1e-3f, &errors, &maximum);
    }
  return print_check("pwconv_cin288_cout96_h7_w7", errors, maximum);
}

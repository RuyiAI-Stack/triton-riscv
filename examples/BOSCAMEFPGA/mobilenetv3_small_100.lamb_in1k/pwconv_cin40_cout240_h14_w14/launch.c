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

extern void _mlir_ciface_kernel_pwconv_cin40_cout240_h14_w14(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)7840 * 4u);

  float *weight = (float *)workspace(off);
  off = vit_align64(off + (size_t)9600 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)240 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)47040 * 4u);

  for (int i = 0; i < 7840; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 9600; ++i)
    weight[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 240; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 47040; ++i)
    o[i] = 0.0f;

  MemRef4 X = make_4(x, 1, 40, 14, 14);
  MemRef4 W = make_4(weight, 240, 40, 1, 1);
  MemRef1 B = make_1(b, 240);
  MemRef4 O = make_4(o, 1, 240, 14, 14);
  _mlir_ciface_kernel_pwconv_cin40_cout240_h14_w14(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int oc = 0; oc < 240; ++oc)
    for (int s = 0; s < 196; ++s) {
      double acc = b[oc];
      for (int ic = 0; ic < 40; ++ic)
        acc += (double)x[ic * 196 + s] * weight[oc * 40 + ic];
      compare(o[oc * 196 + s], (float)acc, 1e-3f, 1e-3f, &errors, &maximum);
    }
  return print_check("pwconv_cin40_cout240_h14_w14", errors, maximum);
}

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

extern void _mlir_ciface_kernel_pwconv_cin576_cout144_h1_w1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)576 * 4u);

  float *weight = (float *)workspace(off);
  off = vit_align64(off + (size_t)82944 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)144 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)144 * 4u);

  for (int i = 0; i < 576; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 82944; ++i)
    weight[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 144; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 144; ++i)
    o[i] = 0.0f;

  MemRef4 X = make_4(x, 1, 576, 1, 1);
  MemRef4 W = make_4(weight, 144, 576, 1, 1);
  MemRef1 B = make_1(b, 144);
  MemRef4 O = make_4(o, 1, 144, 1, 1);
  _mlir_ciface_kernel_pwconv_cin576_cout144_h1_w1(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int oc = 0; oc < 144; ++oc)
    for (int s = 0; s < 1; ++s) {
      double acc = b[oc];
      for (int ic = 0; ic < 576; ++ic)
        acc += (double)x[ic * 1 + s] * weight[oc * 576 + ic];
      compare(o[oc * 1 + s], (float)acc, 1e-3f, 1e-3f, &errors, &maximum);
    }
  return print_check("pwconv_cin576_cout144_h1_w1", errors, maximum);
}

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

extern void _mlir_ciface_kernel_pwconv_cin16_cout72_h56_w56(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)50176 * 4u);

  float *weight = (float *)workspace(off);
  off = vit_align64(off + (size_t)1152 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)72 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)225792 * 4u);

  for (int i = 0; i < 50176; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 1152; ++i)
    weight[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 72; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 225792; ++i)
    o[i] = 0.0f;

  MemRef4 X = make_4(x, 1, 16, 56, 56);
  MemRef4 W = make_4(weight, 72, 16, 1, 1);
  MemRef1 B = make_1(b, 72);
  MemRef4 O = make_4(o, 1, 72, 56, 56);
  _mlir_ciface_kernel_pwconv_cin16_cout72_h56_w56(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int oc = 0; oc < 72; ++oc)
    for (int s = 0; s < 3136; ++s) {
      double acc = b[oc];
      for (int ic = 0; ic < 16; ++ic)
        acc += (double)x[ic * 3136 + s] * weight[oc * 16 + ic];
      compare(o[oc * 3136 + s], (float)acc, 1e-3f, 1e-3f, &errors, &maximum);
    }
  return print_check("pwconv_cin16_cout72_h56_w56", errors, maximum);
}

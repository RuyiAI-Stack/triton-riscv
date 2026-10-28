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

extern void _mlir_ciface_kernel_dwconv_c88_h28_k3_s1_p1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)68992 * 4u);

  float *weight = (float *)workspace(off);
  off = vit_align64(off + (size_t)792 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)88 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)68992 * 4u);

  for (int i = 0; i < 68992; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 792; ++i)
    weight[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 88; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 68992; ++i)
    o[i] = 0.0f;

  MemRef4 X = make_4(x, 1, 88, 28, 28);
  MemRef4 W = make_4(weight, 88, 1, 3, 3);
  MemRef1 B = make_1(b, 88);
  MemRef4 O = make_4(o, 1, 88, 28, 28);
  _mlir_ciface_kernel_dwconv_c88_h28_k3_s1_p1(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int ch = 0; ch < 88; ++ch)
    for (int oy = 0; oy < 28; ++oy)
      for (int ox = 0; ox < 28; ++ox) {
        double acc = b[ch];
        for (int ky = 0; ky < 3; ++ky) {
          int iy = oy * 1 + ky - 1;
          for (int kx = 0; kx < 3; ++kx) {
            int ix = ox * 1 + kx - 1;
            if (iy >= 0 && iy < 28 && ix >= 0 && ix < 28)
              acc += (double)x[(ch * 28 + iy) * 28 + ix] * weight[(ch * 3 + ky) * 3 + kx];
          }
        }
        compare(o[(ch * 28 + oy) * 28 + ox], (float)acc, 1e-3f, 1e-3f, &errors, &maximum);
      }
  return print_check("dwconv_c88_h28_k3_s1_p1", errors, maximum);
}

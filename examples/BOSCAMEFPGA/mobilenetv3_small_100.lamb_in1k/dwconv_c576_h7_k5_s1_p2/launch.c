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

extern void _mlir_ciface_kernel_dwconv_c576_h7_k5_s1_p2(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)28224 * 4u);

  float *weight = (float *)workspace(off);
  off = vit_align64(off + (size_t)14400 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)576 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)28224 * 4u);

  for (int i = 0; i < 28224; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 14400; ++i)
    weight[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 576; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 28224; ++i)
    o[i] = 0.0f;

  MemRef4 X = make_4(x, 1, 576, 7, 7);
  MemRef4 W = make_4(weight, 576, 1, 5, 5);
  MemRef1 B = make_1(b, 576);
  MemRef4 O = make_4(o, 1, 576, 7, 7);
  _mlir_ciface_kernel_dwconv_c576_h7_k5_s1_p2(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int ch = 0; ch < 576; ++ch)
    for (int oy = 0; oy < 7; ++oy)
      for (int ox = 0; ox < 7; ++ox) {
        double acc = b[ch];
        for (int ky = 0; ky < 5; ++ky) {
          int iy = oy * 1 + ky - 2;
          for (int kx = 0; kx < 5; ++kx) {
            int ix = ox * 1 + kx - 2;
            if (iy >= 0 && iy < 7 && ix >= 0 && ix < 7)
              acc += (double)x[(ch * 7 + iy) * 7 + ix] * weight[(ch * 5 + ky) * 5 + kx];
          }
        }
        compare(o[(ch * 7 + oy) * 7 + ox], (float)acc, 1e-3f, 1e-3f, &errors, &maximum);
      }
  return print_check("dwconv_c576_h7_k5_s1_p2", errors, maximum);
}

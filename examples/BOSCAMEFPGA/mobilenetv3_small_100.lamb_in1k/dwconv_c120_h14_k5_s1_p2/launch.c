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

extern void _mlir_ciface_kernel_dwconv_c120_h14_k5_s1_p2(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)23520 * 4u);

  float *weight = (float *)workspace(off);
  off = vit_align64(off + (size_t)3000 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)120 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)23520 * 4u);

  for (int i = 0; i < 23520; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 3000; ++i)
    weight[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 120; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 23520; ++i)
    o[i] = 0.0f;

  MemRef4 X = make_4(x, 1, 120, 14, 14);
  MemRef4 W = make_4(weight, 120, 1, 5, 5);
  MemRef1 B = make_1(b, 120);
  MemRef4 O = make_4(o, 1, 120, 14, 14);
  _mlir_ciface_kernel_dwconv_c120_h14_k5_s1_p2(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int ch = 0; ch < 120; ++ch)
    for (int oy = 0; oy < 14; ++oy)
      for (int ox = 0; ox < 14; ++ox) {
        double acc = b[ch];
        for (int ky = 0; ky < 5; ++ky) {
          int iy = oy * 1 + ky - 2;
          for (int kx = 0; kx < 5; ++kx) {
            int ix = ox * 1 + kx - 2;
            if (iy >= 0 && iy < 14 && ix >= 0 && ix < 14)
              acc += (double)x[(ch * 14 + iy) * 14 + ix] * weight[(ch * 5 + ky) * 5 + kx];
          }
        }
        compare(o[(ch * 14 + oy) * 14 + ox], (float)acc, 1e-3f, 1e-3f, &errors, &maximum);
      }
  return print_check("dwconv_c120_h14_k5_s1_p2", errors, maximum);
}

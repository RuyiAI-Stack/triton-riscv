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

extern void _mlir_ciface_kernel_conv_stem_cin3_cout16_h224_k3_s2_p1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)150528 * 4u);

  float *weight = (float *)workspace(off);
  off = vit_align64(off + (size_t)432 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)16 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)200704 * 4u);

  for (int i = 0; i < 150528; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 432; ++i)
    weight[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 16; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 200704; ++i)
    o[i] = 0.0f;

  MemRef4 X = make_4(x, 1, 3, 224, 224);
  MemRef4 W = make_4(weight, 16, 3, 3, 3);
  MemRef1 B = make_1(b, 16);
  MemRef4 O = make_4(o, 1, 16, 112, 112);
  _mlir_ciface_kernel_conv_stem_cin3_cout16_h224_k3_s2_p1(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int oc = 0; oc < 16; ++oc)
    for (int oy = 0; oy < 112; ++oy)
      for (int ox = 0; ox < 112; ++ox) {
        double acc = b[oc];
        for (int ic = 0; ic < 3; ++ic)
          for (int ky = 0; ky < 3; ++ky) {
            int iy = oy * 2 + ky - 1;
            for (int kx = 0; kx < 3; ++kx) {
              int ix = ox * 2 + kx - 1;
              if (iy >= 0 && iy < 224 && ix >= 0 && ix < 224)
                acc += (double)x[((ic * 224 + iy) * 224) + ix] * weight[(((oc * 3 + ic) * 3 + ky) * 3) + kx];
            }
          }
        compare(o[(oc * 112 + oy) * 112 + ox], (float)acc, 1e-3f, 1e-3f, &errors, &maximum);
      }
  return print_check("conv_stem_cin3_cout16_h224_k3_s2_p1", errors, maximum);
}

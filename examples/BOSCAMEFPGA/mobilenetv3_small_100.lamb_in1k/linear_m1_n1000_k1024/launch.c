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

extern void _mlir_ciface_kernel_linear_m1_n1000_k1024(MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)1024 * 4u);

  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)1024000 * 4u);

  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)1000 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)1000 * 4u);

  for (int i = 0; i < 1024; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 1024000; ++i)
    w[i] = (float)((i * 11) % 19 - 9) * 0.015625f;

  for (int i = 0; i < 1000; ++i)
    b[i] = (float)((i * 5) % 9 - 4) * 0.0078125f;

  for (int i = 0; i < 1000; ++i)
    o[i] = 0.0f;

  MemRef2 X = make_2(x, 1, 1024), W = make_2(w, 1000, 1024);
  MemRef1 B = make_1(b, 1000);
  MemRef2 O = make_2(o, 1, 1000);
  _mlir_ciface_kernel_linear_m1_n1000_k1024(&X, &W, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int row = 0; row < 1; ++row)
    for (int col = 0; col < 1000; ++col) {
      double acc = b[col];
      for (int q = 0; q < 1024; ++q)
        acc += (double)x[row * 1024 + q] * w[col * 1024 + q];
      compare(o[row * 1000 + col], (float)acc, 2e-2f, 2e-2f, &errors, &maximum);
    }
  return print_check("linear_m1_n1000_k1024", errors, maximum);
}

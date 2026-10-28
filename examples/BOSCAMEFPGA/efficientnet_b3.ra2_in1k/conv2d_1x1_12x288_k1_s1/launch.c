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
extern void _mlir_ciface_kernel_conv2d_1x1_12x288_k1_s1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)12 * 4u);
  for (int i = 0; i < 12; ++i) x[i] = 0;
  MemRef4 X = make_4(x, 1, 12, 1, 1);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)3456 * 4u);
  for (int i = 0; i < 3456; ++i) w[i] = 0;
  MemRef4 W = make_4(w, 288, 12, 1, 1);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)288 * 4u);
  for (int i = 0; i < 288; ++i) b[i] = 0;
  MemRef1 B = make_1(b, 288);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)288 * 4u);
  for (int i = 0; i < 288; ++i) o[i] = 0;
  MemRef4 O = make_4(o, 1, 288, 1, 1);
  _mlir_ciface_kernel_conv2d_1x1_12x288_k1_s1(&X, &W, &B, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

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
extern void _mlir_ciface_kernel_dwconv2d_18x18x816_k5_s2(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)264384 * 4u);
  for (int i = 0; i < 264384; ++i) x[i] = 0;
  MemRef4 X = make_4(x, 1, 816, 18, 18);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)20400 * 4u);
  for (int i = 0; i < 20400; ++i) w[i] = 0;
  MemRef4 W = make_4(w, 816, 1, 5, 5);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)816 * 4u);
  for (int i = 0; i < 816; ++i) b[i] = 0;
  MemRef1 B = make_1(b, 816);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)66096 * 4u);
  for (int i = 0; i < 66096; ++i) o[i] = 0;
  MemRef4 O = make_4(o, 1, 816, 9, 9);
  _mlir_ciface_kernel_dwconv2d_18x18x816_k5_s2(&X, &W, &B, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

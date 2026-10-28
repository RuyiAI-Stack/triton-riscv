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
extern void _mlir_ciface_kernel_conv2d_72x72_144x32_k1_s1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)746496 * 4u);
  for (int i = 0; i < 746496; ++i) x[i] = 0;
  MemRef4 X = make_4(x, 1, 144, 72, 72);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)4608 * 4u);
  for (int i = 0; i < 4608; ++i) w[i] = 0;
  MemRef4 W = make_4(w, 32, 144, 1, 1);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)32 * 4u);
  for (int i = 0; i < 32; ++i) b[i] = 0;
  MemRef1 B = make_1(b, 32);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)165888 * 4u);
  for (int i = 0; i < 165888; ++i) o[i] = 0;
  MemRef4 O = make_4(o, 1, 32, 72, 72);
  _mlir_ciface_kernel_conv2d_72x72_144x32_k1_s1(&X, &W, &B, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

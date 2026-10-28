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
extern void _mlir_ciface_kernel_dwconv2d_144x144x144_k3_s2(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)2985984 * 4u);
  for (int i = 0; i < 2985984; ++i) x[i] = 0;
  MemRef4 X = make_4(x, 1, 144, 144, 144);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)1296 * 4u);
  for (int i = 0; i < 1296; ++i) w[i] = 0;
  MemRef4 W = make_4(w, 144, 1, 3, 3);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)144 * 4u);
  for (int i = 0; i < 144; ++i) b[i] = 0;
  MemRef1 B = make_1(b, 144);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)746496 * 4u);
  for (int i = 0; i < 746496; ++i) o[i] = 0;
  MemRef4 O = make_4(o, 1, 144, 72, 72);
  _mlir_ciface_kernel_dwconv2d_144x144x144_k3_s2(&X, &W, &B, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

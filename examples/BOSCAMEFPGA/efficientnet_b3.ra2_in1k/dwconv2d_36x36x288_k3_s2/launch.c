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
extern void _mlir_ciface_kernel_dwconv2d_36x36x288_k3_s2(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)373248 * 4u);
  for (int i = 0; i < 373248; ++i) x[i] = 0;
  MemRef4 X = make_4(x, 1, 288, 36, 36);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)2592 * 4u);
  for (int i = 0; i < 2592; ++i) w[i] = 0;
  MemRef4 W = make_4(w, 288, 1, 3, 3);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)288 * 4u);
  for (int i = 0; i < 288; ++i) b[i] = 0;
  MemRef1 B = make_1(b, 288);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)93312 * 4u);
  for (int i = 0; i < 93312; ++i) o[i] = 0;
  MemRef4 O = make_4(o, 1, 288, 18, 18);
  _mlir_ciface_kernel_dwconv2d_36x36x288_k3_s2(&X, &W, &B, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

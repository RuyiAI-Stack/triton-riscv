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
extern void _mlir_ciface_kernel_dequantize_1x1000(MemRef2 *, MemRef1 *, MemRef1 *, MemRef2 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  int32_t *x = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1000 * 4u);
  for (int i = 0; i < 1000; ++i) x[i] = 0;
  MemRef2 X = make_2(x, 1, 1000);
  float *r = (float *)workspace(off);
  off = vit_align64(off + (size_t)1 * 4u);
  for (int i = 0; i < 1; ++i) r[i] = 0;
  MemRef1 R = make_1(r, 1);
  float *c = (float *)workspace(off);
  off = vit_align64(off + (size_t)1000 * 4u);
  for (int i = 0; i < 1000; ++i) c[i] = 0;
  MemRef1 C = make_1(c, 1000);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)1000 * 4u);
  for (int i = 0; i < 1000; ++i) o[i] = 0;
  MemRef2 O = make_2(o, 1, 1000);
  _mlir_ciface_kernel_dequantize_1x1000(&X, &R, &C, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

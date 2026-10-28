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
extern void _mlir_ciface_kernel_conv2d_9x9_384x1536_k1_s1(MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)31104 * 4u);
  for (int i = 0; i < 31104; ++i) x[i] = 0;
  MemRef4 X = make_4(x, 1, 384, 9, 9);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)589824 * 4u);
  for (int i = 0; i < 589824; ++i) w[i] = 0;
  MemRef4 W = make_4(w, 1536, 384, 1, 1);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)1536 * 4u);
  for (int i = 0; i < 1536; ++i) b[i] = 0;
  MemRef1 B = make_1(b, 1536);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)124416 * 4u);
  for (int i = 0; i < 124416; ++i) o[i] = 0;
  MemRef4 O = make_4(o, 1, 1536, 9, 9);
  _mlir_ciface_kernel_conv2d_9x9_384x1536_k1_s1(&X, &W, &B, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

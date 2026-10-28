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
extern void _mlir_ciface_kernel_linear_1x1000x1536(MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *a = (float *)workspace(off);
  off = vit_align64(off + (size_t)1536 * 4u);
  for (int i = 0; i < 1536; ++i) a[i] = 0;
  MemRef2 A = make_2(a, 1, 1536);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)1536000 * 4u);
  for (int i = 0; i < 1536000; ++i) w[i] = 0;
  MemRef2 W = make_2(w, 1000, 1536);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)1000 * 4u);
  for (int i = 0; i < 1000; ++i) b[i] = 0;
  MemRef1 B = make_1(b, 1000);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)1000 * 4u);
  for (int i = 0; i < 1000; ++i) o[i] = 0;
  MemRef2 O = make_2(o, 1, 1000);
  _mlir_ciface_kernel_linear_1x1000x1536(&A, &W, &B, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

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
extern void _mlir_ciface_kernel_quantize_1x1536(MemRef2 *, MemRef2 *, MemRef1 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)1536 * 4u);
  for (int i = 0; i < 1536; ++i) x[i] = 0;
  MemRef2 X = make_2(x, 1, 1536);
  int8_t *q = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1536 * 1u);
  for (int i = 0; i < 1536; ++i) q[i] = 0;
  MemRef2 Q = make_2(q, 1, 1536);
  float *s = (float *)workspace(off);
  off = vit_align64(off + (size_t)1 * 4u);
  for (int i = 0; i < 1; ++i) s[i] = 0;
  MemRef1 S = make_1(s, 1);
  _mlir_ciface_kernel_quantize_1x1536(&X, &Q, &S);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

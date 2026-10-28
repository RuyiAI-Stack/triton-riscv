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
extern void _mlir_ciface_kernel_gap_72x72x192(MemRef4 *, MemRef2 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)995328 * 4u);
  for (int i = 0; i < 995328; ++i) x[i] = 0;
  MemRef4 X = make_4(x, 1, 192, 72, 72);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)192 * 4u);
  for (int i = 0; i < 192; ++i) o[i] = 0;
  MemRef2 O = make_2(o, 1, 192);
  _mlir_ciface_kernel_gap_72x72x192(&X, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

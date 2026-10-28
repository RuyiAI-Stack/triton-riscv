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

extern void _mlir_ciface_kernel_hardswish_count23520(MemRef1 *, MemRef1 *);
int launch(void) {
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)23520 * 4u);

  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)23520 * 4u);

  for (int i = 0; i < 23520; ++i)
    x[i] = (float)((i * 17) % 29 - 14) * 0.0625f;

  for (int i = 0; i < 23520; ++i)
    o[i] = 0.0f;

  MemRef1 X = make_1(x, 23520), O = make_1(o, 23520);
  _mlir_ciface_kernel_hardswish_count23520(&X, &O);
  int errors = 0;
  float maximum = 0;
  for (int i = 0; i < 23520; ++i)
    compare(o[i], ({ float shifted = x[i] + 3.0f; float lower = shifted > 0.0f ? shifted : 0.0f; float clamped = lower < 6.0f ? lower : 6.0f; (x[i] * clamped) / 6.0f; }), 1e-5f, 1e-5f, &errors, &maximum);
  return print_check("hardswish_count23520", errors, maximum);
}

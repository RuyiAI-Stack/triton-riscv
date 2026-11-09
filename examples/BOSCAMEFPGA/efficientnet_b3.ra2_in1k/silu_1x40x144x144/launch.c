//
// Host / bare-metal harness for FPGA case 'silu_1x40x144x144' (model
// 'efficientnet_b3.ra2_in1k').
//
// Inputs: workspace buffers filled with synthetic operands, then the generated
// _mlir_ciface_* kernel entry.
//
// Outputs: UART or stdout check line from print_check(), plus max error versus
// a CPU oracle.
//
// When used: linked with support.c after AOT compile; run on NH/E6 or as
// HOST_TEST.
//

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
extern void _mlir_ciface_kernel_silu_1x40x144x144(MemRef4 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)829440 * 4u);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)829440 * 4u);
  for (int i = 0; i < 829440; ++i) {
    x[i] = (i % 31) * 0.0078125f;
    o[i] = 0;
  }
  MemRef4 X = make_4(x, 1, 40, 144, 144), O = make_4(o, 1, 40, 144, 144);
  _mlir_ciface_kernel_silu_1x40x144x144(&X, &O);
  for (int i = 0; i < 829440; ++i) {
    float expected = x[i] / (1.0f + expf(-x[i]));
    compare(o[i], expected, 2e-5f, 2e-5f, &errors, &maximum);
  }
  return print_check("silu_1x40x144x144", errors, maximum);
}

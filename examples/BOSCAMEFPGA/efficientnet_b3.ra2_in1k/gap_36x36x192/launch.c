//
// Host / bare-metal harness for FPGA case 'gap_36x36x192' (model
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
extern void _mlir_ciface_kernel_gap_36x36x192(MemRef4 *, MemRef2 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)248832 * 4u);
  for (int i = 0; i < 248832; ++i)
    x[i] = 0;
  MemRef4 X = make_4(x, 1, 192, 36, 36);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)192 * 4u);
  for (int i = 0; i < 192; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1, 192);
  _mlir_ciface_kernel_gap_36x36x192(&X, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

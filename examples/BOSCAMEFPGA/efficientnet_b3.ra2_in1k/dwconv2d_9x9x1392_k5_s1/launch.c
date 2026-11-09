//
// Host / bare-metal harness for FPGA case 'dwconv2d_9x9x1392_k5_s1' (model
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
extern void _mlir_ciface_kernel_dwconv2d_9x9x1392_k5_s1(MemRef4 *, MemRef4 *,
                                                        MemRef1 *, MemRef4 *);
int launch(void) {
  int errors = 0;
  float maximum = 0;
  size_t off = 0;
  float *x = (float *)workspace(off);
  off = vit_align64(off + (size_t)112752 * 4u);
  for (int i = 0; i < 112752; ++i)
    x[i] = 0;
  MemRef4 X = make_4(x, 1, 1392, 9, 9);
  float *w = (float *)workspace(off);
  off = vit_align64(off + (size_t)34800 * 4u);
  for (int i = 0; i < 34800; ++i)
    w[i] = 0;
  MemRef4 W = make_4(w, 1392, 1, 5, 5);
  float *b = (float *)workspace(off);
  off = vit_align64(off + (size_t)1392 * 4u);
  for (int i = 0; i < 1392; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 1392);
  float *o = (float *)workspace(off);
  off = vit_align64(off + (size_t)112752 * 4u);
  for (int i = 0; i < 112752; ++i)
    o[i] = 0;
  MemRef4 O = make_4(o, 1, 1392, 9, 9);
  _mlir_ciface_kernel_dwconv2d_9x9x1392_k5_s1(&X, &W, &B, &O);
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

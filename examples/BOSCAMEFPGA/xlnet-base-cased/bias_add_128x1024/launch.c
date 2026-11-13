//
// Host / bare-metal harness for FPGA case 'bias_add_128x1024' (model
// 'xlnet-base-cased').
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
#ifdef HOST_TEST
#include <math.h>
#endif
static void compare(float a, float b, float at, float rt, int *e, float *m) {
  if (!check_close(a, b, at, rt))
    (*e)++;
  float d = a - b;
  if (d < 0)
    d = -d;
  if (d > *m)
    *m = d;
}
extern void _mlir_ciface_kernel_bias_add_128x1024(MemRef2 *, MemRef1 *,
                                                  MemRef2 *);
int launch(void) {
  float *x = (float *)workspace(0),
        *b = (float *)workspace(((size_t)131072 * 4 + 63) & ~(size_t)63),
        *o = (float *)workspace(((size_t)131072 * 4 + 1024 * 4 + 127) &
                                ~(size_t)63);
  for (int i = 0; i < 131072; ++i) {
    x[i] = (i % 31 - 15) * 0.0078125f;
    o[i] = 0;
  }
  for (int i = 0; i < 1024; ++i)
    b[i] = (i % 17 - 8) * 0.015625f;
  MemRef2 X = make_2(x, 128, 1024), O = make_2(o, 128, 1024);
  MemRef1 B = make_1(b, 1024);
  _mlir_ciface_kernel_bias_add_128x1024(&X, &B, &O);
  int errors = 0;
  float maximum = 0;
  for (int i = 0; i < 131072; ++i)
    compare(o[i], x[i] + b[i % 1024], 1e-6f, 1e-6f, &errors, &maximum);
  return print_check("bias_add_128x1024", errors, maximum);
}

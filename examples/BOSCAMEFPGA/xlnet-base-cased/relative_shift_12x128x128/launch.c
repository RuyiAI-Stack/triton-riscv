//
// Host / bare-metal harness for FPGA case 'relative_shift_12x128x128' (model
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
extern void _mlir_ciface_kernel_relative_shift_12x128x128(MemRef3 *, MemRef3 *);
int launch(void) {
  size_t offset = 0;
  float *X = workspace(offset);
  offset = vit_align64(offset + 393216u * sizeof(*X));
  for (size_t i = 0; i < 393216u; ++i)
    X[i] = (float)((int)(i % 41) - 20) * 0.03125f;
  float *Out = workspace(offset);
  offset = vit_align64(offset + 196608u * sizeof(*Out));
  for (size_t i = 0; i < 196608u; ++i)
    Out[i] = -999.0f;
  MemRef3 mX = make_3(X, 12, 128, 256);
  MemRef3 mOut = make_3(Out, 12, 128, 128);
  _mlir_ciface_kernel_relative_shift_12x128x128(&mX, &mOut);
  unsigned errors = 0;
  float maximum = 0.0f;
  for (int h = 0; h < 12; ++h)
    for (int i = 0; i < 128; ++i)
      for (int j = 0; j < 128; ++j) {
        float want = X[h * 128 * 256 + 128 + i * 255 + j];
        float delta = __builtin_fabsf(Out[(h * 128 + i) * 128 + j] - want);
        errors +=
            !check_close(Out[(h * 128 + i) * 128 + j], want, 1e-06f, 1e-06f);
        if (delta > maximum)
          maximum = delta;
      }
  return print_check("relative_shift_12x128x128", errors, maximum);
}

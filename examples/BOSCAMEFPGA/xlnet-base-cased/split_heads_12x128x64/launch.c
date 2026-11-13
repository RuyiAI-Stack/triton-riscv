//
// Host / bare-metal harness for FPGA case 'split_heads_12x128x64' (model
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
extern void _mlir_ciface_kernel_split_heads_12x128x64(MemRef2 *, MemRef3 *);
int launch(void) {
  size_t offset = 0;
  float *X = workspace(offset);
  offset = vit_align64(offset + 98304u * sizeof(*X));
  for (size_t i = 0; i < 98304u; ++i)
    X[i] = (float)((int)(i % 41) - 20) * 0.03125f;
  float *Out = workspace(offset);
  offset = vit_align64(offset + 98304u * sizeof(*Out));
  for (size_t i = 0; i < 98304u; ++i)
    Out[i] = -999.0f;
  MemRef2 mX = make_2(X, 128, 768);
  MemRef3 mOut = make_3(Out, 12, 128, 64);
  _mlir_ciface_kernel_split_heads_12x128x64(&mX, &mOut);
  unsigned errors = 0;
  float maximum = 0.0f;
  for (int h = 0; h < 12; ++h)
    for (int i = 0; i < 128; ++i)
      for (int j = 0; j < 64; ++j) {
        float want = X[i * 768 + h * 64 + j];
        float delta = __builtin_fabsf(Out[(h * 128 + i) * 64 + j] - want);
        errors +=
            !check_close(Out[(h * 128 + i) * 64 + j], want, 1e-06f, 1e-06f);
        if (delta > maximum)
          maximum = delta;
      }
  return print_check("split_heads_12x128x64", errors, maximum);
}

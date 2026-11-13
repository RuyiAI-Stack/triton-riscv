//
// Host / bare-metal harness for FPGA case 'head_bias_add_12x128x64' (model
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
extern void _mlir_ciface_kernel_head_bias_add_12x128x64(MemRef3 *, MemRef2 *,
                                                        MemRef3 *);
int launch(void) {
  size_t offset = 0;
  float *X = workspace(offset);
  offset = vit_align64(offset + 98304u * sizeof(*X));
  for (size_t i = 0; i < 98304u; ++i)
    X[i] = (float)((int)(i % 41) - 20) * 0.03125f;
  float *Bias = workspace(offset);
  offset = vit_align64(offset + 768u * sizeof(*Bias));
  for (size_t i = 0; i < 768u; ++i)
    Bias[i] = (float)((int)(i % 41) - 20) * 0.03125f;
  float *Out = workspace(offset);
  offset = vit_align64(offset + 98304u * sizeof(*Out));
  for (size_t i = 0; i < 98304u; ++i)
    Out[i] = -999.0f;
  MemRef3 mX = make_3(X, 12, 128, 64);
  MemRef2 mBias = make_2(Bias, 12, 64);
  MemRef3 mOut = make_3(Out, 12, 128, 64);
  _mlir_ciface_kernel_head_bias_add_12x128x64(&mX, &mBias, &mOut);
  unsigned errors = 0;
  float maximum = 0.0f;
  for (int h = 0; h < 12; ++h)
    for (int i = 0; i < 128 * 64; ++i) {
      float want = X[h * 128 * 64 + i] + Bias[h * 64 + i % 64];
      float delta = __builtin_fabsf(Out[h * 128 * 64 + i] - want);
      errors += !check_close(Out[h * 128 * 64 + i], want, 1e-06f, 1e-06f);
      if (delta > maximum)
        maximum = delta;
    }
  return print_check("head_bias_add_12x128x64", errors, maximum);
}

//
// Host / bare-metal harness for FPGA case 'masked_softmax_12x128x128' (model
// 'electra-base-discriminator').
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
extern void _mlir_ciface_kernel_masked_softmax_12x128x128(MemRef3 *, MemRef1 *,
                                                          MemRef3 *);
int launch(void) {
  size_t offset = 0;
  float *X = workspace(offset);
  offset = vit_align64(offset + (size_t)196608 * sizeof(float));
  MemRef3 dX = make_3(X, 12, 128, 128);
  int32_t *Mask = workspace(offset);
  offset = vit_align64(offset + (size_t)128 * sizeof(int32_t));
  MemRef1 dMask = make_1(Mask, 128);
  float *Out = workspace(offset);
  offset = vit_align64(offset + (size_t)196608 * sizeof(float));
  MemRef3 dOut = make_3(Out, 12, 128, 128);
  for (int c = 0; c < 128; c++)
    Mask[c] = (c < 117 && c % 11 != 3);
  for (int i = 0; i < 12 * 128 * 128; i++)
    X[i] = (float)(i % 97 - 48) * 0.0625f;
#if !defined(HOST_TEST)
  ame_fence();
#endif
  uint64_t start = nr_cycles();
  _mlir_ciface_kernel_masked_softmax_12x128x128(&dX, &dMask, &dOut);
  nr_puts("cycles masked_softmax_12x128x128: ");
  nr_hex64(nr_cycles() - start);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int row = 0; row < 12 * 128; row++) {
    float top = -1e30f, total = 0;
    for (int c = 0; c < 128; c++)
      if (Mask[c] && X[row * 128 + c] > top)
        top = X[row * 128 + c];
    for (int c = 0; c < 128; c++)
      if (Mask[c])
        total += expf(X[row * 128 + c] - top);
    for (int c = 0; c < 128; c++) {
      float want = Mask[c] ? expf(X[row * 128 + c] - top) / total : 0.0f;
      errors += !check_close(Out[row * 128 + c], want, 2e-5f, 2e-5f);
    }
  }
  return print_check("masked_softmax_12x128x128", errors, maximum);
}

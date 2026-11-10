//
// Host / bare-metal harness for FPGA case 'pack_heads_12x128x64' (model
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
extern void _mlir_ciface_kernel_pack_heads_12x128x64(MemRef2 *, MemRef3 *);
int launch(void) {
  size_t offset = 0;
  float *X = workspace(offset);
  offset = vit_align64(offset + (size_t)98304 * sizeof(float));
  MemRef2 dX = make_2(X, 128, 768);
  float *Out = workspace(offset);
  offset = vit_align64(offset + (size_t)98304 * sizeof(float));
  MemRef3 dOut = make_3(Out, 12, 128, 64);
  for (int i = 0; i < 128 * 768; i++)
    X[i] = (float)(i % 1021) * 0.00390625f;
#if !defined(HOST_TEST)
  ame_fence();
#endif
  uint64_t start = nr_cycles();
  _mlir_ciface_kernel_pack_heads_12x128x64(&dX, &dOut);
  nr_puts("cycles pack_heads_12x128x64: ");
  nr_hex64(nr_cycles() - start);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int row = 0; row < 128; row++)
    for (int head = 0; head < 12; head++)
      for (int col = 0; col < 64; col++) {
        errors += Out[head * 128 * 64 + row * 64 + col] !=
                  X[row * 12 * 64 + head * 64 + col];
      }
  return print_check("pack_heads_12x128x64", errors, maximum);
}

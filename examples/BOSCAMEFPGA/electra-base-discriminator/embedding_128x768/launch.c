//
// Host / bare-metal harness for FPGA case 'embedding_128x768' (model
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
extern void _mlir_ciface_kernel_embedding_128x768(MemRef1 *, MemRef1 *,
                                                  MemRef2 *, MemRef2 *,
                                                  MemRef2 *, MemRef2 *);
int launch(void) {
  size_t offset = 0;
  int32_t *IDs = workspace(offset);
  offset = vit_align64(offset + (size_t)128 * sizeof(int32_t));
  MemRef1 dIDs = make_1(IDs, 128);
  int32_t *Types = workspace(offset);
  offset = vit_align64(offset + (size_t)128 * sizeof(int32_t));
  MemRef1 dTypes = make_1(Types, 128);
  float *Word = workspace(offset);
  offset = vit_align64(offset + (size_t)23440896 * sizeof(float));
  MemRef2 dWord = make_2(Word, 30522, 768);
  float *Position = workspace(offset);
  offset = vit_align64(offset + (size_t)393216 * sizeof(float));
  MemRef2 dPosition = make_2(Position, 512, 768);
  float *Segment = workspace(offset);
  offset = vit_align64(offset + (size_t)1536 * sizeof(float));
  MemRef2 dSegment = make_2(Segment, 2, 768);
  float *Out = workspace(offset);
  offset = vit_align64(offset + (size_t)98304 * sizeof(float));
  MemRef2 dOut = make_2(Out, 128, 768);
  for (int r = 0; r < 128; r++) {
    IDs[r] = (r * 239) % 30522;
    Types[r] = r % 2;
  }
  IDs[127] = 30521;
  for (size_t i = 0; i < (size_t)30522 * 768; i++)
    Word[i] = (float)((int)(i % 67) - 33) * 0.00390625f;
  for (int i = 0; i < 512 * 768; i++)
    Position[i] = (float)(i % 37) * 0.001953125f;
  for (int i = 0; i < 2 * 768; i++)
    Segment[i] = (float)(i % 19) * 0.015625f;
#if !defined(HOST_TEST)
  ame_fence();
#endif
  uint64_t start = nr_cycles();
  _mlir_ciface_kernel_embedding_128x768(&dIDs, &dTypes, &dWord, &dPosition,
                                        &dSegment, &dOut);
  nr_puts("cycles embedding_128x768: ");
  nr_hex64(nr_cycles() - start);
  nr_puts("\r\n");
  unsigned errors = 0;
  float maximum = 0;
  for (int r = 0; r < 128; r++)
    for (int c = 0; c < 768; c++) {
      float want =
          (Word[(size_t)IDs[r] * 768 + c] + Segment[Types[r] * 768 + c]) +
          Position[r * 768 + c];
      errors += !check_close(Out[r * 768 + c], want, 1e-6f, 1e-6f);
    }
  return print_check("embedding_128x768", errors, maximum);
}

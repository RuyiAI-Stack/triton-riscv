//
// Host / bare-metal harness for FPGA case 'embedding_128x768' (model
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
extern void _mlir_ciface_kernel_embedding_128x768(MemRef1 *, MemRef2 *,
                                                  MemRef2 *);
int launch(void) {
  size_t offset = 0;
  int32_t *Ids = workspace(offset);
  offset = vit_align64(offset + 128u * sizeof(*Ids));
  for (size_t i = 0; i < 128u; ++i)
    Ids[i] = (int32_t)(i % 7);
  float *Table = workspace(offset);
  offset = vit_align64(offset + 24576000u * sizeof(*Table));
  for (size_t i = 0; i < 24576000u; ++i)
    Table[i] = (float)((int)(i % 41) - 20) * 0.03125f;
  float *Out = workspace(offset);
  offset = vit_align64(offset + 98304u * sizeof(*Out));
  for (size_t i = 0; i < 98304u; ++i)
    Out[i] = -999.0f;
  for (int i = 0; i < 128; ++i)
    Ids[i] = (i * 251) % 32000;
  MemRef1 mIds = make_1(Ids, 128);
  MemRef2 mTable = make_2(Table, 32000, 768);
  MemRef2 mOut = make_2(Out, 128, 768);
  _mlir_ciface_kernel_embedding_128x768(&mIds, &mTable, &mOut);
  unsigned errors = 0;
  float maximum = 0.0f;
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 768; ++j) {
      float want = Table[Ids[i] * 768 + j];
      float delta = __builtin_fabsf(Out[i * 768 + j] - want);
      errors += !check_close(Out[i * 768 + j], want, 1e-06f, 1e-06f);
      if (delta > maximum)
        maximum = delta;
    }
  return print_check("embedding_128x768", errors, maximum);
}

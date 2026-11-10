//
// Host / bare-metal harness for FPGA case 'pack_heads_12x197x197' (model
// 'vit-base-patch16-224').
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
#include "tools/launch_checks.h"
extern void _mlir_ciface_kernel_pack_heads_12x197x197(MemRef2 *, MemRef2 *);
int launch(void) {
  const int H = 12, WIDTH = 38809, SRC = 38809, DST = 38816;
  float *x = workspace(64);
  float *y = workspace(vit_align64(64 + (size_t)H * SRC * 4 + 128));
  vit_guard_set(x, (size_t)H * SRC * 4);
  vit_guard_set(y, (size_t)H * DST * 4);
  for (int i = 0; i < H * SRC; ++i)
    x[i] = (float)(i % 997 - 498);
  for (int i = 0; i < H * DST; ++i)
    y[i] = -999.0f;
  MemRef2 X = make_2(x, H, SRC), Y = make_2(y, H, DST);
  _mlir_ciface_kernel_pack_heads_12x197x197(&X, &Y);
  unsigned errors = 0;
  for (int h = 0; h < H; ++h) {
    for (int i = 0; i < WIDTH; ++i)
      errors += !vit_f32_equal_bits(y[(size_t)h * DST + i],
                                    (float)((h * SRC + i) % 997 - 498));
    // Pack padding is a gap, and must never receive an unmasked tail write.
    for (int i = WIDTH; i < DST; ++i)
      errors += !vit_f32_equal_bits(y[(size_t)h * DST + i], -999.0f);
  }
  for (int i = 0; i < H * SRC; ++i)
    errors += !vit_f32_equal_bits(x[i], (float)(i % 997 - 498));
  errors += vit_guard_errors(x, (size_t)H * SRC * 4);
  errors += vit_guard_errors(y, (size_t)H * DST * 4);
  return print_check("pack_heads_12x197x197", errors, 0.0f);
}

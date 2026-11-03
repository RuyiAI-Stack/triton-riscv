#include "support.h"
extern void _mlir_ciface_kernel_conv2d_9x9_1392x232_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [81, 1392] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)81 * 1392 * 1u);
  for (int i = 0; i < 81 * 1392; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 81, 1392);
  /* Weight: i8 [OC, C] = [232, 1392] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)232 * 1392 * 1u);
  for (int i = 0; i < 232 * 1392; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 232, 1392);
  /* Bias: i32 [232] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)232 * 4u);
  for (int i = 0; i < 232; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 232);
  /* Out: i32 [H*W, OC] = [81, 232] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)81 * 232 * 4u);
  for (int i = 0; i < 81 * 232; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 81, 232);
  _mlir_ciface_kernel_conv2d_9x9_1392x232_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 81; ++s) {
    int32_t *row = o + s * 232;
    for (int oc = 0; oc < 232; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

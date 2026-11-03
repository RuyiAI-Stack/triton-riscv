#include "support.h"
extern void _mlir_ciface_kernel_conv2d_1x1_1392x58_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [1, 1392] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 1392 * 1u);
  for (int i = 0; i < 1 * 1392; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 1, 1392);
  /* Weight: i8 [OC, C] = [58, 1392] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)58 * 1392 * 1u);
  for (int i = 0; i < 58 * 1392; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 58, 1392);
  /* Bias: i32 [58] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)58 * 4u);
  for (int i = 0; i < 58; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 58);
  /* Out: i32 [H*W, OC] = [1, 58] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 58 * 4u);
  for (int i = 0; i < 1 * 58; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1, 58);
  _mlir_ciface_kernel_conv2d_1x1_1392x58_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 1; ++s) {
    int32_t *row = o + s * 58;
    for (int oc = 0; oc < 58; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

#include "support.h"
extern void _mlir_ciface_kernel_conv2d_1x1_10x40_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [1, 10] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 10 * 1u);
  for (int i = 0; i < 1 * 10; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 1, 10);
  /* Weight: i8 [OC, C] = [40, 10] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)40 * 10 * 1u);
  for (int i = 0; i < 40 * 10; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 40, 10);
  /* Bias: i32 [40] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)40 * 4u);
  for (int i = 0; i < 40; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 40);
  /* Out: i32 [H*W, OC] = [1, 40] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 40 * 4u);
  for (int i = 0; i < 1 * 40; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1, 40);
  _mlir_ciface_kernel_conv2d_1x1_10x40_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 1; ++s) {
    int32_t *row = o + s * 40;
    for (int oc = 0; oc < 40; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

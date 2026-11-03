#include "support.h"
extern void _mlir_ciface_kernel_conv2d_1x1_576x24_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [1, 576] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 576 * 1u);
  for (int i = 0; i < 1 * 576; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 1, 576);
  /* Weight: i8 [OC, C] = [24, 576] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)24 * 576 * 1u);
  for (int i = 0; i < 24 * 576; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 24, 576);
  /* Bias: i32 [24] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)24 * 4u);
  for (int i = 0; i < 24; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 24);
  /* Out: i32 [H*W, OC] = [1, 24] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 24 * 4u);
  for (int i = 0; i < 1 * 24; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1, 24);
  _mlir_ciface_kernel_conv2d_1x1_576x24_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 1; ++s) {
    int32_t *row = o + s * 24;
    for (int oc = 0; oc < 24; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

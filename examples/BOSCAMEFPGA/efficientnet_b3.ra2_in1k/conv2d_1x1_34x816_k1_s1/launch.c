#include "support.h"
extern void _mlir_ciface_kernel_conv2d_1x1_34x816_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [1, 34] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 34 * 1u);
  for (int i = 0; i < 1 * 34; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 1, 34);
  /* Weight: i8 [OC, C] = [816, 34] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)816 * 34 * 1u);
  for (int i = 0; i < 816 * 34; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 816, 34);
  /* Bias: i32 [816] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)816 * 4u);
  for (int i = 0; i < 816; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 816);
  /* Out: i32 [H*W, OC] = [1, 816] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 816 * 4u);
  for (int i = 0; i < 1 * 816; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1, 816);
  _mlir_ciface_kernel_conv2d_1x1_34x816_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 1; ++s) {
    int32_t *row = o + s * 816;
    for (int oc = 0; oc < 816; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

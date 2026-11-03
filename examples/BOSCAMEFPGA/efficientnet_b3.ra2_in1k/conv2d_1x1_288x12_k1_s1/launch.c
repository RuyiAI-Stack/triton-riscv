#include "support.h"
extern void _mlir_ciface_kernel_conv2d_1x1_288x12_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [1, 288] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 288 * 1u);
  for (int i = 0; i < 1 * 288; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 1, 288);
  /* Weight: i8 [OC, C] = [12, 288] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)12 * 288 * 1u);
  for (int i = 0; i < 12 * 288; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 12, 288);
  /* Bias: i32 [12] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)12 * 4u);
  for (int i = 0; i < 12; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 12);
  /* Out: i32 [H*W, OC] = [1, 12] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 12 * 4u);
  for (int i = 0; i < 1 * 12; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1, 12);
  _mlir_ciface_kernel_conv2d_1x1_288x12_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 1; ++s) {
    int32_t *row = o + s * 12;
    for (int oc = 0; oc < 12; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

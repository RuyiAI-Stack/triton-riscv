#include "support.h"
extern void _mlir_ciface_kernel_conv2d_1x1_6x144_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [1, 6] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 6 * 1u);
  for (int i = 0; i < 1 * 6; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 1, 6);
  /* Weight: i8 [OC, C] = [144, 6] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)144 * 6 * 1u);
  for (int i = 0; i < 144 * 6; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 144, 6);
  /* Bias: i32 [144] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)144 * 4u);
  for (int i = 0; i < 144; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 144);
  /* Out: i32 [H*W, OC] = [1, 144] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1 * 144 * 4u);
  for (int i = 0; i < 1 * 144; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1, 144);
  _mlir_ciface_kernel_conv2d_1x1_6x144_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 1; ++s) {
    int32_t *row = o + s * 144;
    for (int oc = 0; oc < 144; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

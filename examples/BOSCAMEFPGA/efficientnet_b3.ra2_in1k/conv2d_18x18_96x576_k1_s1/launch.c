#include "support.h"
extern void _mlir_ciface_kernel_conv2d_18x18_96x576_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [324, 96] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)324 * 96 * 1u);
  for (int i = 0; i < 324 * 96; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 324, 96);
  /* Weight: i8 [OC, C] = [576, 96] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)576 * 96 * 1u);
  for (int i = 0; i < 576 * 96; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 576, 96);
  /* Bias: i32 [576] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)576 * 4u);
  for (int i = 0; i < 576; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 576);
  /* Out: i32 [H*W, OC] = [324, 576] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)324 * 576 * 4u);
  for (int i = 0; i < 324 * 576; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 324, 576);
  _mlir_ciface_kernel_conv2d_18x18_96x576_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 324; ++s) {
    int32_t *row = o + s * 576;
    for (int oc = 0; oc < 576; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

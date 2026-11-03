#include "support.h"
extern void _mlir_ciface_kernel_conv2d_18x18_576x136_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [324, 576] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)324 * 576 * 1u);
  for (int i = 0; i < 324 * 576; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 324, 576);
  /* Weight: i8 [OC, C] = [136, 576] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)136 * 576 * 1u);
  for (int i = 0; i < 136 * 576; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 136, 576);
  /* Bias: i32 [136] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)136 * 4u);
  for (int i = 0; i < 136; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 136);
  /* Out: i32 [H*W, OC] = [324, 136] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)324 * 136 * 4u);
  for (int i = 0; i < 324 * 136; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 324, 136);
  _mlir_ciface_kernel_conv2d_18x18_576x136_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 324; ++s) {
    int32_t *row = o + s * 136;
    for (int oc = 0; oc < 136; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

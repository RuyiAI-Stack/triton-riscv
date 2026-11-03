#include "support.h"
extern void _mlir_ciface_kernel_conv2d_72x72_192x32_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [5184, 192] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)5184 * 192 * 1u);
  for (int i = 0; i < 5184 * 192; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 5184, 192);
  /* Weight: i8 [OC, C] = [32, 192] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)32 * 192 * 1u);
  for (int i = 0; i < 32 * 192; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 32, 192);
  /* Bias: i32 [32] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)32 * 4u);
  for (int i = 0; i < 32; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 32);
  /* Out: i32 [H*W, OC] = [5184, 32] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)5184 * 32 * 4u);
  for (int i = 0; i < 5184 * 32; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 5184, 32);
  _mlir_ciface_kernel_conv2d_72x72_192x32_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 5184; ++s) {
    int32_t *row = o + s * 32;
    for (int oc = 0; oc < 32; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

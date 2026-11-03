#include "support.h"
extern void _mlir_ciface_kernel_conv2d_36x36_192x48_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [1296, 192] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1296 * 192 * 1u);
  for (int i = 0; i < 1296 * 192; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 1296, 192);
  /* Weight: i8 [OC, C] = [48, 192] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)48 * 192 * 1u);
  for (int i = 0; i < 48 * 192; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 48, 192);
  /* Bias: i32 [48] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)48 * 4u);
  for (int i = 0; i < 48; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 48);
  /* Out: i32 [H*W, OC] = [1296, 48] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1296 * 48 * 4u);
  for (int i = 0; i < 1296 * 48; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1296, 48);
  _mlir_ciface_kernel_conv2d_36x36_192x48_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 1296; ++s) {
    int32_t *row = o + s * 48;
    for (int oc = 0; oc < 48; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

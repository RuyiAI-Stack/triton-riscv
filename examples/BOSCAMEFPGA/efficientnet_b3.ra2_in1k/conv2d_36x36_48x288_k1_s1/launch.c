#include "support.h"
extern void _mlir_ciface_kernel_conv2d_36x36_48x288_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [1296, 48] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1296 * 48 * 1u);
  for (int i = 0; i < 1296 * 48; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 1296, 48);
  /* Weight: i8 [OC, C] = [288, 48] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)288 * 48 * 1u);
  for (int i = 0; i < 288 * 48; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 288, 48);
  /* Bias: i32 [288] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)288 * 4u);
  for (int i = 0; i < 288; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 288);
  /* Out: i32 [H*W, OC] = [1296, 288] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1296 * 288 * 4u);
  for (int i = 0; i < 1296 * 288; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 1296, 288);
  _mlir_ciface_kernel_conv2d_36x36_48x288_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 1296; ++s) {
    int32_t *row = o + s * 288;
    for (int oc = 0; oc < 288; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

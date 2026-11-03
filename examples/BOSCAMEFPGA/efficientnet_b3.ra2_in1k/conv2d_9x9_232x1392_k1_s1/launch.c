#include "support.h"
extern void _mlir_ciface_kernel_conv2d_9x9_232x1392_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [81, 232] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)81 * 232 * 1u);
  for (int i = 0; i < 81 * 232; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 81, 232);
  /* Weight: i8 [OC, C] = [1392, 232] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1392 * 232 * 1u);
  for (int i = 0; i < 1392 * 232; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 1392, 232);
  /* Bias: i32 [1392] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1392 * 4u);
  for (int i = 0; i < 1392; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 1392);
  /* Out: i32 [H*W, OC] = [81, 1392] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)81 * 1392 * 4u);
  for (int i = 0; i < 81 * 1392; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 81, 1392);
  _mlir_ciface_kernel_conv2d_9x9_232x1392_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 81; ++s) {
    int32_t *row = o + s * 1392;
    for (int oc = 0; oc < 1392; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

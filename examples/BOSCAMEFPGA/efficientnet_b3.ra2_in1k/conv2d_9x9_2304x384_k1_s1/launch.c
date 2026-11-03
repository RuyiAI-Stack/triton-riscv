#include "support.h"
extern void _mlir_ciface_kernel_conv2d_9x9_2304x384_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [81, 2304] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)81 * 2304 * 1u);
  for (int i = 0; i < 81 * 2304; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 81, 2304);
  /* Weight: i8 [OC, C] = [384, 2304] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)384 * 2304 * 1u);
  for (int i = 0; i < 384 * 2304; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 384, 2304);
  /* Bias: i32 [384] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)384 * 4u);
  for (int i = 0; i < 384; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 384);
  /* Out: i32 [H*W, OC] = [81, 384] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)81 * 384 * 4u);
  for (int i = 0; i < 81 * 384; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 81, 384);
  _mlir_ciface_kernel_conv2d_9x9_2304x384_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 81; ++s) {
    int32_t *row = o + s * 384;
    for (int oc = 0; oc < 384; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

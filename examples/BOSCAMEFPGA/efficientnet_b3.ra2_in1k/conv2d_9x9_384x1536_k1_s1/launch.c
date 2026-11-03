#include "support.h"
extern void _mlir_ciface_kernel_conv2d_9x9_384x1536_k1_s1(MemRef2 *, MemRef2 *,
                                                           MemRef1 *, MemRef2 *);
int launch(void) {
  size_t off = 0;
  /* Packed X: i8 [H*W, C] = [81, 384] */
  int8_t *x = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)81 * 384 * 1u);
  for (int i = 0; i < 81 * 384; ++i)
    x[i] = 0;
  MemRef2 X = make_2(x, 81, 384);
  /* Weight: i8 [OC, C] = [1536, 384] */
  int8_t *w = (int8_t *)workspace(off);
  off = vit_align64(off + (size_t)1536 * 384 * 1u);
  for (int i = 0; i < 1536 * 384; ++i)
    w[i] = 0;
  MemRef2 W = make_2(w, 1536, 384);
  /* Bias: i32 [1536] */
  int32_t *b = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)1536 * 4u);
  for (int i = 0; i < 1536; ++i)
    b[i] = 0;
  MemRef1 B = make_1(b, 1536);
  /* Out: i32 [H*W, OC] = [81, 1536] */
  int32_t *o = (int32_t *)workspace(off);
  off = vit_align64(off + (size_t)81 * 1536 * 4u);
  for (int i = 0; i < 81 * 1536; ++i)
    o[i] = 0;
  MemRef2 O = make_2(o, 81, 1536);
  _mlir_ciface_kernel_conv2d_9x9_384x1536_k1_s1(&X, &W, &B, &O);
  /* Bias applied outside the AME GEMM tile. */
  for (int s = 0; s < 81; ++s) {
    int32_t *row = o + s * 1536;
    for (int oc = 0; oc < 1536; ++oc)
      row[oc] += b[oc];
  }
  nr_puts("numerical acceptance is simcheck\n");
  return 0;
}

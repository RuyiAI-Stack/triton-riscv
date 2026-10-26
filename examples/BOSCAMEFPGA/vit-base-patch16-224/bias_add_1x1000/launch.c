#include "support.h"
#define R 1
#define W 1000
#define C (R * W)
extern void _mlir_ciface_kernel_bias_add_1x1000(MemRef2 *, MemRef1 *,
                                                MemRef2 *);
int launch(void) {
  size_t bias_offset = vit_align64((size_t)C * sizeof(float));
  size_t output_offset = vit_align64(bias_offset + (size_t)W * sizeof(float));
  float *x = workspace(0), *b = workspace(bias_offset),
        *o = workspace(output_offset);
  for (int i = 0; i < C; ++i)
    x[i] = i * .01f;
  for (int i = 0; i < W; ++i)
    b[i] = i * .002f;
  MemRef2 X = make_2(x, R, W);
  MemRef1 B = make_1(b, W);
  MemRef2 O = make_2(o, R, W);
  _mlir_ciface_kernel_bias_add_1x1000(&X, &B, &O);
  unsigned e = 0;
  for (int i = 0; i < C; ++i) {
    {
      float want = x[i] + b[i % W];
      if (!check_close(o[i], want, 1e-5f, 1e-5f))
        e++;
    }
  }
  return print_check("bias_add_1x1000", e, 0);
}

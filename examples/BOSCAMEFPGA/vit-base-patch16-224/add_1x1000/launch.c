#include "support.h"
#define R 1
#define W 1000
#define C (R * W)
extern void _mlir_ciface_kernel_add_1x1000(MemRef2 *, MemRef2 *, MemRef2 *);
int launch(void) {
  size_t y_offset = vit_align64((size_t)C * sizeof(float));
  size_t output_offset = vit_align64(y_offset + (size_t)C * sizeof(float));
  float *x = workspace(0), *y = workspace(y_offset),
        *o = workspace(output_offset);
  for (int i = 0; i < C; ++i) {
    x[i] = i * .01f;
    y[i] = -i * .002f;
  }
  MemRef2 X = make_2(x, R, W), Y = make_2(y, R, W);
  MemRef2 O = make_2(o, R, W);
  _mlir_ciface_kernel_add_1x1000(&X, &Y, &O);
  unsigned e = 0;
  for (int i = 0; i < C; ++i) {
    {
      float want = x[i] + y[i];
      if (!check_close(o[i], want, 1e-5f, 1e-5f))
        e++;
    }
  }
  return print_check("add_1x1000", e, 0);
}

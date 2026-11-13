//
// Host / bare-metal harness for FPGA case 'quantize_transpose_128x128' (model
// 'electra-base-discriminator').
//
// Inputs: workspace buffers filled with synthetic operands, then the generated
// _mlir_ciface_* kernel entry.
//
// Outputs: UART or stdout check line from print_check(), plus max error versus
// a CPU oracle.
//
// When used: linked with support.c after AOT compile; run on NH/E6 or as
// HOST_TEST.
//

#include "support.h"
#include "tools/attention_reference.h"
#include "tools/launch_checks.h"
#define M 128
#define N 128
extern void _mlir_ciface_kernel_quantize_transpose_128x128(MemRef2 *, MemRef2 *,
                                                           MemRef1 *);
static float input_value(int i, int j) {
  // Include zero columns, both extrema, exact half ties and adjacent quarter
  // values.
  float base = i == 0   ? 127.0f
               : i == 1 ? -127.0f
                        : (float)(i % 29 - 14) * 0.25f;
  return j % 5 == 0 ? 0.0f : base * (j % 2 ? 1.137f : 1.0f);
}
int launch(void) {
  size_t s_offset = vit_align64(128 + (size_t)M * N * 4 + 128);
  size_t q_offset = vit_align64(s_offset + (size_t)N * 4 + 128);
  float *x = workspace(128), *s = workspace(s_offset);
  int8_t *q = workspace(q_offset);
  vit_guard_set(x, (size_t)M * N * 4);
  vit_guard_set(s, (size_t)N * 4);
  vit_guard_set(q, (size_t)M * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      x[(size_t)i * N + j] = input_value(i, j);
  for (int j = 0; j < N; ++j)
    s[j] = -999.0f;
  for (size_t i = 0; i < (size_t)M * N; ++i)
    q[i] = -128;
  MemRef2 X = make_2(x, M, N), Q = make_2(q, N, M);
  MemRef1 S = make_1(s, N);
  _mlir_ciface_kernel_quantize_transpose_128x128(&X, &Q, &S);
  unsigned errors = 0;
  for (int j = 0; j < N; ++j) {
    float maximum = 0.0f;
    for (int i = 0; i < M; ++i) {
      float v = vit_reference_abs(input_value(i, j));
      if (v > maximum)
        maximum = v;
    }
    float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    errors += !vit_f32_equal_bits(s[j], scale);
    for (int i = 0; i < M; ++i)
      errors +=
          q[(size_t)j * M + i] != vit_reference_i8(input_value(i, j), scale);
  }
  errors += vit_guard_errors(x, (size_t)M * N * 4);
  errors += vit_guard_errors(s, (size_t)N * 4);
  errors += vit_guard_errors(q, (size_t)M * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      errors += !vit_f32_equal_bits(x[(size_t)i * N + j], input_value(i, j));
  return print_check("quantize_transpose_128x128", errors, 0.0f);
}

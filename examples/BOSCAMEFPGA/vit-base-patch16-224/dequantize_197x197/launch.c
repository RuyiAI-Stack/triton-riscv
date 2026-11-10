//
// Host / bare-metal harness for FPGA case 'dequantize_197x197' (model
// 'vit-base-patch16-224').
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
#include "tools/launch_checks.h"
#define M 197
#define N 197
extern void _mlir_ciface_kernel_dequantize_197x197(MemRef2 *, MemRef1 *,
                                                   MemRef1 *, MemRef2 *);
int launch(void) {
  size_t row_offset = vit_align64(64 + (size_t)M * N * sizeof(int32_t) + 128);
  size_t column_offset =
      vit_align64(row_offset + (size_t)M * sizeof(float) + 128);
  size_t output_offset =
      vit_align64(column_offset + (size_t)N * sizeof(float) + 128);
  int32_t *x = workspace(64);
  float *r = workspace(row_offset), *s = workspace(column_offset),
        *y = workspace(output_offset);
  vit_guard_set(x, (size_t)M * N * 4);
  vit_guard_set(r, (size_t)M * 4);
  vit_guard_set(s, (size_t)N * 4);
  vit_guard_set(y, (size_t)M * N * 4);
  for (size_t i = 0; i < (size_t)M * N; ++i)
    y[i] = -98765.0f;
  for (int i = 0; i < M; i++)
    r[i] = (i % 17 + 1) * 0.00390625f;
  for (int j = 0; j < N; j++)
    s[j] = (j % 31 + 1) * 0.001953125f;
  for (size_t i = 0; i < (size_t)M * N; i++)
    x[i] = ((int)i % 1023 - 511) * 7919;
  MemRef2 X = make_2(x, M, N), Y = make_2(y, M, N);
  MemRef1 R = make_1(r, M), S = make_1(s, N);
  uint64_t start = nr_cycles();
  _mlir_ciface_kernel_dequantize_197x197(&X, &R, &S, &Y);
  nr_puts("cycles dequantize_197x197: ");
  nr_hex64(nr_cycles() - start);
  nr_puts("\r\n");
  unsigned errors = 0;
  float max_error = 0.0f;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float want = (float)x[(size_t)i * N + j] * r[i] * s[j] * 0.125f,
            dlt = y[(size_t)i * N + j] - want;
      if (dlt < 0)
        dlt = -dlt;
      if (dlt > max_error)
        max_error = dlt;
      if (!vit_f32_equal_bits(y[(size_t)i * N + j], want))
        errors++;
    }
  errors += vit_guard_errors(x, (size_t)M * N * 4);
  errors += vit_guard_errors(r, (size_t)M * 4);
  errors += vit_guard_errors(s, (size_t)N * 4);
  errors += vit_guard_errors(y, (size_t)M * N * 4);
  for (size_t i = 0; i < (size_t)M * N; ++i)
    errors += x[i] != ((int)i % 1023 - 511) * 7919;
  for (int i = 0; i < M; ++i)
    errors += !vit_f32_equal_bits(r[i], (i % 17 + 1) * 0.00390625f);
  for (int j = 0; j < N; ++j)
    errors += !vit_f32_equal_bits(s[j], (j % 31 + 1) * 0.001953125f);
  return print_check("dequantize_197x197", errors, max_error);
}

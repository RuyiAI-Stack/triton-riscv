//
// Host / bare-metal harness for FPGA case 'matmul_i8_128x128x64' (model
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
#include "tools/launch_checks.h"
#define M 128
#define N 128
#define K 64
extern void _mlir_ciface_kernel_matmul_i8_128x128x64(MemRef2 *, MemRef2 *,
                                                     MemRef2 *);
int launch(void) {
  size_t b_offset = vit_align64(64 + (size_t)M * K + 128);
  size_t c_offset = vit_align64(b_offset + (size_t)N * K + 128);
  int8_t *a = workspace(64), *b = workspace(b_offset);
  int32_t *c = workspace(c_offset);
  vit_guard_set(a, (size_t)M * K);
  vit_guard_set(b, (size_t)N * K);
  vit_guard_set(c, (size_t)M * N * 4);
  for (int i = 0; i < M; i++)
    for (int q = 0; q < K; q++)
      a[(size_t)i * K + q] = (int8_t)(((i * 13 + q * 7) % 31) - 15);
  for (int j = 0; j < N; j++)
    for (int q = 0; q < K; q++)
      b[(size_t)j * K + q] = (int8_t)(((j * 11 + q * 5) % 29) - 14);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      c[(size_t)i * N + j] = (i * 3 + j) % 7 - 3;
  MemRef2 A = make_2(a, M, K), B = make_2(b, N, K), C = make_2(c, M, N);
#if !defined(HOST_TEST)
  ame_fence();
#endif
  uint64_t start = nr_cycles();
  _mlir_ciface_kernel_matmul_i8_128x128x64(&A, &B, &C);
  nr_puts("cycles matmul_i8_128x128x64: ");
  nr_hex64(nr_cycles() - start);
  nr_puts("\r\n");
  unsigned errors = 0;
  int32_t max_error = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      int32_t want = 0;
      for (int q = 0; q < K; q++)
        want += (int32_t)a[(size_t)i * K + q] * (int32_t)b[(size_t)j * K + q];
      int32_t got = c[(size_t)i * N + j], dlt = got - want;
      if (dlt < 0)
        dlt = -dlt;
      if (dlt > max_error)
        max_error = dlt;
      if (got != want) {
        if (errors < 4) {
          nr_puts("mismatch ");
          nr_hex32(i);
          nr_puts(" ");
          nr_hex32(j);
          nr_puts("\r\n");
        }
        errors++;
      }
    }
  // VIT_GOLDEN_POSTCHECK_BEGIN
  errors += vit_guard_errors(a, (size_t)M * K);
  errors += vit_guard_errors(b, (size_t)N * K);
  errors += vit_guard_errors(c, (size_t)M * N * 4);
  for (int i = 0; i < M; ++i)
    for (int q = 0; q < K; ++q)
      errors += a[(size_t)i * K + q] != (int8_t)(((i * 13 + q * 7) % 31) - 15);
  for (int j = 0; j < N; ++j)
    for (int q = 0; q < K; ++q)
      errors += b[(size_t)j * K + q] != (int8_t)(((j * 11 + q * 5) % 29) - 14);
  // VIT_GOLDEN_POSTCHECK_END
  return print_check("matmul_i8_128x128x64", errors, (float)max_error);
}

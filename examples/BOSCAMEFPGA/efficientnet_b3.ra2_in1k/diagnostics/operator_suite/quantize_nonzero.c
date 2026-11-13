//
// quantize_nonzero.c for BOSCAME FPGA model 'efficientnet_b3.ra2_in1k'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k.
//

#include "support.h"

extern void _mlir_ciface_kernel_quantize_1x1536(MemRef2 *, MemRef2 *,
                                                MemRef1 *);

static float fixture(int scenario, int index) {
  if (scenario == 0)
    return 0.0f;
  // Every nonzero scenario fixes amax=127 exactly, so scale=1 and the
  // independent oracle can expose signed ties and near-boundary rounding.
  if (index == 1535)
    return -127.0f;
  if (scenario == 1)
    return (float)((index * 73 + 19) % 1001 - 500) * 0.25f;
  int value = index % 251 - 125;
  if (scenario == 2)
    return (float)value + (value < 0 ? -0.5f : 0.5f);
  return (float)value + (index & 1 ? 0.49993896484375f : 0.50006103515625f);
}

int launch(void) {
  float *input = (float *)workspace(0);
  int8_t *output = (int8_t *)workspace(vit_align64(1536u * sizeof(float)));
  float *scale = (float *)workspace(vit_align64(1536u * sizeof(float)) +
                                    vit_align64(1536u));
  MemRef2 X = make_2(input, 1, 1536), Q = make_2(output, 1, 1536);
  MemRef1 S = make_1(scale, 1);
  unsigned errors = 0;
  for (int scenario = 0; scenario < 4; ++scenario) {
    for (int index = 0; index < 1536; ++index) {
      input[index] = fixture(scenario, index);
      output[index] = -128;
    }
    scale[0] = -1.0f;
    _mlir_ciface_kernel_quantize_1x1536(&X, &Q, &S);
    if (scale[0] != 1.0f)
      ++errors;
    for (int index = 0; index < 1536; ++index) {
      float value = fixture(scenario, index);
      int expected = value < 0.0f ? -(int)(-value + 0.5f) : (int)(value + 0.5f);
      if ((int)output[index] != expected)
        ++errors;
    }
    nr_puts("QUANT_NUMERIC scenario=");
    nr_hex32((uint32_t)scenario);
    nr_puts(" outputs=00000600 scale_bits=");
    union {
      float value;
      uint32_t bits;
    } bits = {scale[0]};
    nr_hex32(bits.bits);
    nr_puts(" cumulative_errors=");
    nr_hex32(errors);
    nr_puts("\r\n");
  }
  return print_check("quantize_1x1536_nonzero_four_scenarios", errors,
                     errors ? 1.0f : 0.0f);
}

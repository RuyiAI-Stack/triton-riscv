// Typed entry points generated from the literal operator inventory.
#ifndef ELECTRA_KERNELS_H
#define ELECTRA_KERNELS_H
#include "support.h"
extern void _mlir_ciface_kernel_add_128x768(MemRef2 *, MemRef2 *, MemRef2 *);
extern void _mlir_ciface_kernel_bias_add_128x1(MemRef2 *, MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_bias_add_128x3072(MemRef2 *, MemRef1 *,
                                                  MemRef2 *);
extern void _mlir_ciface_kernel_bias_add_128x768(MemRef2 *, MemRef1 *,
                                                 MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x1(MemRef2 *, MemRef1 *,
                                                 MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x128(MemRef2 *, MemRef1 *,
                                                   MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x3072(MemRef2 *, MemRef1 *,
                                                    MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x64(MemRef2 *, MemRef1 *,
                                                  MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_dequantize_128x768(MemRef2 *, MemRef1 *,
                                                   MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_embedding_128x768(MemRef1 *, MemRef1 *,
                                                  MemRef2 *, MemRef2 *,
                                                  MemRef2 *, MemRef2 *);
extern void _mlir_ciface_kernel_gelu_128x3072(MemRef2 *, MemRef2 *);
extern void _mlir_ciface_kernel_gelu_128x768(MemRef2 *, MemRef2 *);
extern void _mlir_ciface_kernel_layernorm_128x768(MemRef2 *, MemRef1 *,
                                                  MemRef1 *, MemRef2 *);
extern void _mlir_ciface_kernel_masked_softmax_12x128x128(MemRef3 *, MemRef1 *,
                                                          MemRef3 *);
extern void _mlir_ciface_kernel_matmul_i8_128x128x64(MemRef2 *, MemRef2 *,
                                                     MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x1x768(MemRef2 *, MemRef2 *,
                                                    MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x3072x768(MemRef2 *, MemRef2 *,
                                                       MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x64x128(MemRef2 *, MemRef2 *,
                                                     MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x768x3072(MemRef2 *, MemRef2 *,
                                                       MemRef2 *);
extern void _mlir_ciface_kernel_matmul_i8_128x768x768(MemRef2 *, MemRef2 *,
                                                      MemRef2 *);
extern void _mlir_ciface_kernel_pack_heads_12x128x64(MemRef2 *, MemRef3 *);
extern void _mlir_ciface_kernel_quantize_128x128(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
extern void _mlir_ciface_kernel_quantize_128x3072(MemRef2 *, MemRef2 *,
                                                  MemRef1 *);
extern void _mlir_ciface_kernel_quantize_128x64(MemRef2 *, MemRef2 *,
                                                MemRef1 *);
extern void _mlir_ciface_kernel_quantize_128x768(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
extern void _mlir_ciface_kernel_quantize_1x768(MemRef2 *, MemRef2 *, MemRef1 *);
extern void _mlir_ciface_kernel_quantize_3072x768(MemRef2 *, MemRef2 *,
                                                  MemRef1 *);
extern void _mlir_ciface_kernel_quantize_768x3072(MemRef2 *, MemRef2 *,
                                                  MemRef1 *);
extern void _mlir_ciface_kernel_quantize_768x768(MemRef2 *, MemRef2 *,
                                                 MemRef1 *);
extern void _mlir_ciface_kernel_quantize_transpose_128x128(MemRef2 *, MemRef2 *,
                                                           MemRef1 *);
extern void _mlir_ciface_kernel_quantize_transpose_128x64(MemRef2 *, MemRef2 *,
                                                          MemRef1 *);
extern void _mlir_ciface_kernel_unpack_heads_12x128x64(MemRef3 *, MemRef2 *);
#endif

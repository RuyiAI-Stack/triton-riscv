// Whole-model NH/E6 support. Only NH prints UART. No numerical oracle runs
// on the board; all model arithmetic is in the linked real Triton kernels.
#include "../support.h"
#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"

extern int model_launch(void);
extern float model_embedding[384], model_hidden[128 * 384];
#ifndef GTE_MODEL_LAYERS
#define GTE_MODEL_LAYERS 12
#endif
#define GTE_EXPECTED_CALLS (3 + 137 * GTE_MODEL_LAYERS + 2)
#ifdef GTE_MODEL_TRACE_BYTES
unsigned char model_operator_trace[GTE_MODEL_TRACE_BYTES]
    __attribute__((section(".model_exports"), aligned(64), used));
#endif
#ifdef GTE_MODEL_CHECKPOINTS
float model_checkpoints[13][128 * 384]
    __attribute__((section(".model_exports"), aligned(64), used));
static volatile unsigned checkpoints_captured;
void gte_model_checkpoint(unsigned stage, const float *hidden) {
  if (stage >= 13 || stage != checkpoints_captured)
    __builtin_trap();
  nr_copy_bytes(model_checkpoints[stage], hidden, 128u * 384u * 4u);
  nh_e6_cache_writes();
  ++checkpoints_captured;
}
#endif
volatile uint64_t model_calls __attribute__((aligned(64)));
volatile uint64_t model_kernel_cycles;
volatile uint64_t model_cycles;
volatile uint64_t model_active_case;
volatile unsigned model_returned;

uint64_t nr_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}
void *memcpy(void *destination, const void *source, size_t count) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  for (size_t i = 0; i < count; ++i)
    out[i] = in[i];
  return destination;
}
void *memset(void *destination, int value, size_t count) {
  unsigned char *out = destination;
  for (size_t i = 0; i < count; ++i)
    out[i] = (unsigned char)value;
  return destination;
}
// The final hidden state may have been written by DMA. Copy through stable
// stack chunks with the same explicit policies as standalone output capture.
void nr_copy_bytes(void *destination, const void *source, size_t count) {
  volatile unsigned char chunk[256] __attribute__((aligned(64)));
  volatile unsigned char *out = destination;
  const volatile unsigned char *in = source;
  for (size_t off = 0; off < count;) {
    size_t n = count - off;
    if (n > sizeof(chunk))
      n = sizeof(chunk);
    nh_e6_cache_uncached_loads();
    for (size_t i = 0; i < n; ++i)
      chunk[i] = in[off + i];
    __asm__ volatile("fence rw,rw" ::: "memory");
    nh_e6_cache_writes();
    for (size_t i = 0; i < n; ++i)
      out[off + i] = chunk[i];
    __asm__ volatile("fence rw,rw" ::: "memory");
    off += n;
  }
}
void nr_puts(const char *s) { (void)s; }
void nr_hex32(uint32_t v) { (void)v; }
void nr_hex64(uint64_t v) { (void)v; }
float fabsf(float v) { return __builtin_fabsf(v); }
void ame_fence(void) { __asm__ volatile("fence iorw,iorw" ::: "memory"); }

int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  nh_e6_cache_writes();
  uint64_t begin = nr_cycles();
  int result = model_launch();
  nh_e6_cache_writes();
  model_cycles = nr_cycles() - begin;
  model_returned = result == 0;
  __asm__ volatile("fence rw,rw" ::: "memory");
  return result;
}
const char *nh_e6_app_tag(void) { return "gte-small-full-model"; }
void board_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}
static void output(const char *name, const void *pointer, unsigned bytes,
                   const char *shape) {
  print_uart("@@FPGA_OUTPUT {\"version\":1,\"name\":\"");
  print_uart(name);
  print_uart("\",\"address\":\"0x");
  print_uart_hex32((uint32_t)(uintptr_t)pointer);
  print_uart("\",\"nbytes\":");
  print_uart_dec32(bytes);
  print_uart(",\"dtype\":\"<f4\",\"shape\":");
  print_uart(shape);
  print_uart("}\r\n");
}
void nh_e6_app_report(void) {
  __asm__ volatile("fence r,r" ::: "memory");
  nh_e6_print_tag_prefix();
  print_uart("model_calls=");
  print_uart_dec32((uint32_t)model_calls);
  print_uart(" active_case=");
  print_uart_dec32((uint32_t)model_active_case);
  print_uart(" kernel_cycles=0x");
  print_uart_hex64(model_kernel_cycles);
  print_uart(" model_cycles=0x");
  print_uart_hex64(model_cycles);
  print_uart("\r\n");
  uint32_t done = *(volatile uint32_t *)NH_E6_DONE_REG;
  if (done == NH_E6_DONE_TRAP) {
    print_uart(
        "@@FPGA_RESULT {\"version\":1,\"status\":\"trap\",\"mcause\":\"0x");
    print_uart_hex64(e6_diag_mcause);
    print_uart("\",\"mepc\":\"0x");
    print_uart_hex64(e6_diag_mepc);
    print_uart("\",\"mtval\":\"0x");
    print_uart_hex64(e6_diag_mtval);
    print_uart("\"}\r\n");
    return;
  }
  int ok = done == NH_E6_DONE_PASS && model_returned &&
           model_calls == GTE_EXPECTED_CALLS;
#ifdef GTE_MODEL_CHECKPOINTS
  ok = ok && checkpoints_captured == 13;
#endif
  if (ok) {
#ifdef GTE_MODEL_TRACE_BYTES
    output("operator_trace", model_operator_trace, GTE_MODEL_TRACE_BYTES,
           "[" GTE_MODEL_TRACE_SHAPE "]");
#else
    output("sentence_embedding", model_embedding, 384 * 4, "[384]");
    output("last_hidden_state", model_hidden, 128 * 384 * 4, "[128,384]");
#ifdef GTE_MODEL_CHECKPOINTS
    output("layer_checkpoints", model_checkpoints, 13 * 128 * 384 * 4,
           "[13,128,384]");
#endif
#endif
  }
  print_uart(
      ok ? "@@FPGA_RESULT "
           "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
         : "@@FPGA_RESULT "
           "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
}

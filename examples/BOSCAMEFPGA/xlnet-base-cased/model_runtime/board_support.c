// Whole-model execution and snapshot publication only. All checks run locally.
//
#include "bosc_ame_dma.h"
#include "model.h"
#include "model_artifacts.h"
#include "nh_e6_platform.h"
#include "scheduler_output_runtime.h"
#include "uart.h"

volatile uint64_t vit_kernel_case, vit_kernel_calls;
volatile uint64_t vit_trap_sp, vit_trap_case, vit_trap_calls;
static volatile uint64_t model_cycles;
static volatile unsigned model_completed_layers;
extern void vit_e6_trap_entry(void);
extern unsigned char xlnet_workspace[];

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
// Graph scalar copies must read current DMA results and publish CPU stores.
// The temporary is on the E6 stack outside the switchable PMP DDR window.
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
    __asm__ volatile("fence rw, rw" ::: "memory");
    nh_e6_cache_writes();
    for (size_t i = 0; i < n; ++i)
      out[off + i] = chunk[i];
    __asm__ volatile("fence rw, rw" ::: "memory");
    off += n;
  }
}
void xlnet_layer_checkpoint(unsigned layer, const float *hidden) {
#if !XLNET_MODEL_LM_HEAD
  vit_board_capture_output(layer, hidden, 128u * 768u * sizeof(float));
#else
  (void)hidden;
#endif
  model_completed_layers = layer;
  __asm__ volatile("fence rw, rw" ::: "memory");
  nh_e6_progress(0x3000u + layer);
}
int e6_main(void) {
  __asm__ volatile("csrw mtvec, %0" ::"r"(vit_e6_trap_entry) : "memory");
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  nh_e6_cache_writes();
  uint64_t start = nr_cycles();
  int result = xlnet_forward(
      &xlnet_weights, (int32_t *)xlnet_input, (int32_t *)(xlnet_input + 512),
      (int32_t *)(xlnet_input + 1024), xlnet_workspace,
      XLNET_MODEL_SCRATCH_BYTES,
      (float *)(xlnet_workspace + XLNET_MODEL_SCRATCH_BYTES),
#if XLNET_MODEL_LM_HEAD
      (float *)(xlnet_workspace + XLNET_MODEL_SCRATCH_BYTES +
                128u * 768u * sizeof(float)));
  if (!result)
    vit_board_capture_output(0,
                             xlnet_workspace + XLNET_MODEL_SCRATCH_BYTES +
                                 128u * 768u * sizeof(float),
                             128u * 32000u * sizeof(float));
#else
      0);
#endif
  nh_e6_cache_writes();
  model_cycles = nr_cycles() - start;
  __asm__ volatile("fence rw, rw" ::: "memory");
  return result ? result : vit_board_finish_outputs();
}
const char *nh_e6_app_tag(void) { return "xlnet-base-cased-model"; }
void board_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}
void nh_e6_app_report(void) {
  __asm__ volatile("fence r, r" ::: "memory");
  nh_e6_print_tag_prefix();
  print_uart("layers=");
  print_uart_dec32(model_completed_layers);
  print_uart(" model_cycles=0x");
  print_uart_hex64(model_cycles);
  print_uart(" kernel_calls=0x");
  print_uart_hex64(vit_kernel_calls);
  print_uart(" kernel_case=0x");
  print_uart_hex64(vit_kernel_case);
  print_uart(" host_validation=pending\r\n");
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
  } else {
    int ok = done == NH_E6_DONE_PASS && model_completed_layers == 12 &&
             vit_board_outputs_ready();
    if (ok && vit_board_export_outputs())
      ok = 0;
    print_uart(
        ok ? "@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
           : "@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
  }
}
#include "scheduler_output_runtime.inc"

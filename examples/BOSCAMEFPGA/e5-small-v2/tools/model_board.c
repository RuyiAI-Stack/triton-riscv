// Whole-model NH/E6 execution and output declarations; numerical validation
// runs only on downloaded files. Every typed entry binds a real kernel.
#include "model_board.h"
#include "model_case_names.h"
#include "uart.h"

float e5_sentence_output[384]
    __attribute__((section(".model_exports"), aligned(64), used));
float e5_hidden_output[128 * 384]
    __attribute__((section(".model_exports"), aligned(64), used));
float e5_layer_outputs[13 * 128 * 384]
    __attribute__((section(".model_exports"), aligned(64), used));
volatile uint64_t e5_model_cycles, e5_kernel_cycles, e5_kernel_calls;
volatile uint64_t e5_kernel_case, e5_checkpoints, e5_launch_result;
static unsigned nh_checkpoint __attribute__((section(".data.nh"))) = 0;

uint64_t nr_cycles(void) {
  uint64_t n;
  __asm__ volatile("rdcycle %0" : "=r"(n)::"memory");
  return n;
}
void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *d = dst;
  const unsigned char *s = src;
  for (size_t i = 0; i < n; ++i)
    d[i] = s[i];
  return dst;
}
void *memset(void *dst, int value, size_t n) {
  unsigned char *d = dst;
  for (size_t i = 0; i < n; ++i)
    d[i] = (unsigned char)value;
  return dst;
}
void nr_copy_bytes(void *dst, const void *src, size_t n) {
  memcpy(dst, src, n);
}
void nr_puts(const char *s) { (void)s; }
void nr_hex32(uint32_t n) { (void)n; }
void nr_hex64(uint64_t n) { (void)n; }
float fabsf(float x) { return __builtin_fabsf(x); }
void ame_fence(void) { __asm__ volatile("fence iorw, iorw" ::: "memory"); }
uint64_t e5_model_kernel_begin(unsigned case_id) {
  e5_kernel_case = case_id;
  nh_e6_cache_writes();
  return nr_cycles();
}
void e5_model_kernel_end(unsigned case_id, uint64_t begin) {
  (void)case_id;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  uint64_t end = nr_cycles();
  // Match ViT's whole-graph brackets: graph descriptors and control state
  // are CPU-written between calls; each kernel handles its DMA load policy.
  nh_e6_cache_writes();
  e5_kernel_cycles += end - begin;
  ++e5_kernel_calls;
}
void e5_model_checkpoint(unsigned layer, const float *hidden) {
  if (layer >= 13 || e5_checkpoints != layer)
    __builtin_trap();
  nh_e6_cache_writes();
  nr_copy_bytes(e5_layer_outputs + (size_t)layer * 128 * 384, hidden,
                128u * 384u * sizeof(float));
  e5_checkpoints = layer + 1;
  __asm__ volatile("fence rw, rw" ::: "memory");
  nh_e6_progress(0x51000000u + layer);
}
uint32_t e5_model_mmio_read32(uintptr_t address) {
  uint32_t value = *(volatile uint32_t *)address;
  if (address == NH_E6_MAILBOX_REG && value >= 0x51000000u &&
      value <= 0x5100000cu) {
    unsigned completed = value - 0x51000000u + 1;
    if (completed > nh_checkpoint) {
      nh_checkpoint = completed;
      print_uart("[e5-model] layer_checkpoint=");
      print_uart_dec32(completed - 1);
      print_uart("\r\n");
    }
  }
  return value;
}
extern void e5_model_trap_entry(void);
int e6_main(void) {
  __asm__ volatile("csrw mtvec, %0" ::"r"(e5_model_trap_entry) : "memory");
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  nh_e6_cache_writes();
  uint64_t begin = nr_cycles();
  int result = e5_model_launch();
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  e5_model_cycles = nr_cycles() - begin;
  e5_launch_result = (uint64_t)(int64_t)result;
  nh_e6_cache_writes();
  __asm__ volatile("fence rw, rw" ::: "memory");
  return result;
}
const char *nh_e6_app_tag(void) { return "e5-model"; }
void board_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  nh_checkpoint = 0;
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}
static void output(const char *name, const void *pointer, unsigned nbytes,
                   const char *shape) {
  print_uart("@@FPGA_OUTPUT {\"version\":1,\"name\":\"");
  print_uart(name);
  print_uart("\",\"address\":\"0x");
  print_uart_hex32((uint32_t)(uintptr_t)pointer);
  print_uart("\",\"nbytes\":");
  print_uart_dec32(nbytes);
  print_uart(",\"dtype\":\"<f4\",\"shape\":");
  print_uart(shape);
  print_uart("}\r\n");
}
void nh_e6_app_report(void) {
  __asm__ volatile("fence r, r" ::: "memory");
  print_uart("[e5-model] model_cycles=0x");
  print_uart_hex64(e5_model_cycles);
  print_uart(" kernel_cycles_sum=0x");
  print_uart_hex64(e5_kernel_cycles);
  print_uart(" kernel_calls=");
  print_uart_dec32((uint32_t)e5_kernel_calls);
  print_uart(" checkpoints=");
  print_uart_dec32((uint32_t)e5_checkpoints);
  print_uart(" last_case=");
  print_uart_dec32((uint32_t)e5_kernel_case);
  print_uart(" ");
  if (e5_kernel_case < 30)
    print_uart(e5_model_case_names[e5_kernel_case]);
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
  int ok = done == NH_E6_DONE_PASS && e5_launch_result == 0 &&
           e5_kernel_calls == 1649 && e5_checkpoints == 13;
  if (ok) {
    output("sentence_embedding", e5_sentence_output, sizeof(e5_sentence_output),
           "[384]");
    output("last_hidden", e5_hidden_output, sizeof(e5_hidden_output),
           "[128,384]");
    output("layer_hidden", e5_layer_outputs, sizeof(e5_layer_outputs),
           "[13,128,384]");
  }
  print_uart(
      ok ? "@@FPGA_RESULT "
           "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
         : "@@FPGA_RESULT "
           "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
}

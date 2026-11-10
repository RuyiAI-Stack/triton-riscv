// NH owns UART and declarations. E6 executes real typed kernels, never an
// oracle.
#include "board.h"
#include "bosc_ame_dma.h"
#include "model.h"
#include "model_outputs.h"
#include "params.h"
#include "scheduler_output_runtime.h"
#include "uart.h"

extern const unsigned char electra_params[], electra_params_end[];
extern const int32_t electra_input_ids[], electra_token_type_ids[],
    electra_attention_mask[];
static unsigned char scratch[ELECTRA_MODEL_WORKSPACE_BYTES]
    __attribute__((section(".model_workspace"), aligned(64), used));
static float logits[128]
    __attribute__((section(".model_workspace"), aligned(64), used));

typedef struct __attribute__((aligned(64))) {
  volatile uint64_t adapter_cycles, model_cycles, snapshot_cycles,
      elapsed_cycles;
  volatile unsigned started, returned, kind, trace_count;
  volatile int model_status;
} ModelState;
static volatile ModelState state;

uint64_t nr_cycles(void) {
  uint64_t result;
  __asm__ volatile("rdcycle %0" : "=r"(result)::"memory");
  return result;
}
void *memcpy(void *out, const void *in, size_t bytes) {
  unsigned char *a = out;
  const unsigned char *b = in;
  for (size_t i = 0; i < bytes; ++i)
    a[i] = b[i];
  return out;
}
void *memset(void *out, int value, size_t bytes) {
  unsigned char *a = out;
  for (size_t i = 0; i < bytes; ++i)
    a[i] = (unsigned char)value;
  return out;
}
unsigned electra_operator_begin(unsigned kind) {
  nh_e6_cache_writes();
  unsigned slot = state.started++;
  state.kind = kind;
  nh_e6_progress(0x10000u + slot + 1);
  return slot;
}
void electra_operator_end(unsigned slot, uint64_t cycles) {
  nh_e6_cache_writes();
  if (slot != state.returned)
    __builtin_trap();
  state.adapter_cycles += cycles;
  state.returned = slot + 1;
  nh_e6_progress(0x20000u + slot + 1);
}
void electra_model_trace(unsigned stage, const float *data, size_t count) {
  if (stage != state.trace_count || count != (stage == 13 ? 128u : 128u * 768u))
    __builtin_trap();
  uint64_t start = nr_cycles();
  vit_board_capture_output(stage, data, count * sizeof(float));
  state.trace_count++;
  state.snapshot_cycles += nr_cycles() - start;
  nh_e6_progress(0x30000u + stage);
}
int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  nh_e6_cache_writes();
  ElectraWeights weights = {0};
  if (electra_bind_board_weights(&weights, electra_params,
                                 (size_t)(electra_params_end - electra_params)))
    return 1;
  uint64_t begin = nr_cycles();
  int result =
      electra_infer(&weights, electra_input_ids, electra_token_type_ids,
                    electra_attention_mask, logits, scratch, sizeof(scratch));
  uint64_t end = nr_cycles();
  nh_e6_cache_writes();
  state.model_status = result;
  state.elapsed_cycles = end - begin;
  state.model_cycles = end - begin - state.snapshot_cycles;
  if (result || state.returned != 1571u || state.trace_count != 14u)
    return 1;
  return vit_board_finish_outputs();
}
uint32_t electra_board_mmio_read32(uintptr_t address) {
  return *(volatile uint32_t *)address;
}
const char *nh_e6_app_tag(void) { return "electra-full-model"; }
void board_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}
void nh_e6_app_report(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
  print_uart("[electra-full-model] operator_started=");
  print_uart_dec32(state.started);
  print_uart(" operator_returned=");
  print_uart_dec32(state.returned);
  print_uart(" last_kind=");
  print_uart_dec32(state.kind);
  print_uart(" snapshots=");
  print_uart_dec32(state.trace_count);
  print_uart(" adapter_cycles=0x");
  print_uart_hex64(state.adapter_cycles);
  print_uart(" model_cycles=0x");
  print_uart_hex64(state.model_cycles);
  print_uart(" snapshot_cycles=0x");
  print_uart_hex64(state.snapshot_cycles);
  print_uart(" elapsed_cycles=0x");
  print_uart_hex64(state.elapsed_cycles);
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
  int ok = done == NH_E6_DONE_PASS && state.model_status == 0 &&
           state.returned == 1571u && state.trace_count == 14u &&
           vit_board_outputs_ready();
  if (ok && vit_board_export_outputs())
    ok = 0;
  print_uart(
      ok ? "@@FPGA_RESULT "
           "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
         : "@@FPGA_RESULT "
           "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
}
#include "scheduler_output_runtime.inc"

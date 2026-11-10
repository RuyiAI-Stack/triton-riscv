// Only NH owns UART. E6 runs the existing launch initialization/oracle.
#include "board_support.h"
#include "board_case.h"
#include "bosc_ame_dma.h"
#include "uart.h"

static unsigned char arena[VIT_WORKSPACE_BYTES]
    __attribute__((section(".dma_source"), aligned(64), used));

// Each completed invocation occupies a different NH cache line. NH never
// reads an invocation until its MMIO completion marker has been published.
typedef struct __attribute__((aligned(64))) {
  volatile uint64_t cycles;
  unsigned char padding[56];
} CycleSlot;
static CycleSlot cycle_slots[VIT_BOARD_EXPECTED_INVOCATIONS];
// NH can observe invocation completion while E6 is still validating. Keep
// final validation cells separate from every word touched during that phase.
typedef struct __attribute__((aligned(64))) {
  unsigned returned;
  unsigned char padding[60];
} InvocationState;
typedef struct __attribute__((aligned(64))) {
  unsigned validated, checked, errors, maximum_bits;
  unsigned char padding[48];
} ValidationState;
static volatile InvocationState invocation_state;
static volatile ValidationState validation_state;
#define kernel_returned invocation_state.returned
#define validation_executed validation_state.validated
static volatile uint64_t diagnostic_values[8];
static unsigned diagnostic_count;
#if defined(VIT_BOARD_ACCUMULATOR_VALIDATION)
typedef struct __attribute__((aligned(64))) {
  volatile uint64_t output_hash;
  volatile unsigned checked, errors, guard_errors, input_errors, completed;
  unsigned char padding[36];
} ValidationSlot;
static ValidationSlot validation_slots[VIT_BOARD_VALIDATION_ROUNDS];
#endif

// NH owns this cell; .data keeps it out of E6's boot-time BSS clear.
static unsigned nh_reported __attribute__((section(".data.nh_reported"))) = ~0u;

uint64_t nr_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}

void *workspace(size_t offset) {
  if (offset >= sizeof(arena))
    __builtin_trap();
  return arena + offset;
}

// A fresh image has no previous AME operation to resynchronize. Kernel/DMA
// completion is explicit in the generated body; do not issue a DDR 1x1 AME
// transaction merely to prepare the launch's scalar initialization.
void ame_fence(void) { __asm__ volatile("fence iorw, iorw" ::: "memory"); }

void nr_puts(const char *text) { (void)text; }
void nr_hex32(uint32_t value) {
  if (diagnostic_count < 8)
    diagnostic_values[diagnostic_count++] = value;
}
void nr_hex64(uint64_t value) {
  if (diagnostic_count < 8)
    diagnostic_values[diagnostic_count++] = value;
}

void *memcpy(void *destination, const void *source, size_t count) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  for (size_t i = 0; i < count; ++i)
    out[i] = in[i];
  return destination;
}
void nr_copy_bytes(void *destination, const void *source, size_t count) {
  memcpy(destination, source, count);
}
void *memset(void *destination, int value, size_t count) {
  unsigned char *out = destination;
  for (size_t i = 0; i < count; ++i)
    out[i] = (unsigned char)value;
  return destination;
}
float fabsf(float value) { return __builtin_fabsf(value); }

int check_close(float actual, float expected, float atol, float rtol) {
  union {
    float f;
    uint32_t u;
  } a = {actual}, b = {expected};
  if ((a.u & 0x7f800000u) == 0x7f800000u || (b.u & 0x7f800000u) == 0x7f800000u)
    return actual == expected;
  float difference = actual - expected;
  if (difference < 0)
    difference = -difference;
  float magnitude = expected < 0 ? -expected : expected;
  return difference <= atol + rtol * magnitude;
}

int print_check(const char *name, unsigned count, float maximum) {
  (void)name;
  union {
    float f;
    uint32_t u;
  } value = {maximum};
  // DDR result records must use the write-through store policy: the generated
  // wrapper leaves E6 in the post-kernel uncached-load (shared-read) policy,
  // otherwise NH may read stale counters after DONE.
  nh_e6_cache_writes();
  validation_state.checked = VIT_BOARD_EXPECTED_CHECKED;
  validation_state.errors = count;
  validation_state.maximum_bits = value.u;
  validation_executed = 1;
  __asm__ volatile("fence rw, rw" ::: "memory");
  return count != 0;
}

unsigned vit_board_kernel_begin(void) {
  unsigned slot = kernel_returned;
  if (slot >= VIT_BOARD_EXPECTED_INVOCATIONS)
    __builtin_trap();
  nh_e6_progress(0x1000u + slot + 1);
  return slot;
}

void vit_board_kernel_end(unsigned slot, uint64_t cycles) {
  cycle_slots[slot].cycles = cycles;
  kernel_returned = slot + 1;
  __asm__ volatile("fence rw, rw" ::: "memory");
  nh_e6_progress(0x2000u + slot + 1);
}

void vit_board_validation_round(unsigned round, unsigned count,
                                unsigned count_errors, unsigned guard_errors,
                                unsigned input_errors, uint64_t output_hash) {
#if defined(VIT_BOARD_ACCUMULATOR_VALIDATION)
  if (round < VIT_BOARD_VALIDATION_FIRST_ROUND ||
      round - VIT_BOARD_VALIDATION_FIRST_ROUND >= VIT_BOARD_VALIDATION_ROUNDS)
    __builtin_trap();
  ValidationSlot *slot =
      &validation_slots[round - VIT_BOARD_VALIDATION_FIRST_ROUND];
  slot->output_hash = output_hash;
  slot->checked = count;
  slot->errors = count_errors;
  slot->guard_errors = guard_errors;
  slot->input_errors = input_errors;
  __asm__ volatile("fence rw, rw" ::: "memory");
  slot->completed = 1;
#else
  (void)round;
  (void)count;
  (void)count_errors;
  (void)guard_errors;
  (void)input_errors;
  (void)output_hash;
  __builtin_trap();
#endif
}

static void report_cycles(unsigned completed) {
  while (nh_reported < completed) {
    unsigned slot = nh_reported++;
    nh_e6_print_tag_prefix();
    print_uart("invocation=");
    print_uart_dec32(slot);
    print_uart(" kernel_cycles=0x");
    print_uart_hex64(cycle_slots[slot].cycles);
    print_uart("\r\n");
  }
}

uint32_t vit_board_mmio_read32(uintptr_t address) {
  uint32_t value = *(volatile uint32_t *)address;
  if (address == NH_E6_MAILBOX_REG && value > 0x2000u &&
      value <= 0x2000u + VIT_BOARD_EXPECTED_INVOCATIONS) {
    __asm__ volatile("fence r, r" ::: "memory");
    report_cycles(value - 0x2000u);
  }
  return value;
}

int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  nh_e6_cache_writes();
  return launch();
}

const char *nh_e6_app_tag(void) { return VIT_BOARD_CASE; }

void board_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  nh_reported = 0;
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}

void nh_e6_app_report(void) {
  unsigned returned = kernel_returned;
  if (returned <= VIT_BOARD_EXPECTED_INVOCATIONS)
    report_cycles(returned);
#if defined(VIT_BOARD_ACCUMULATOR_VALIDATION)
  for (unsigned round = 0; round < VIT_BOARD_VALIDATION_ROUNDS; ++round) {
    const ValidationSlot *slot = &validation_slots[round];
    if (!slot->completed)
      continue;
    nh_e6_print_tag_prefix();
    print_uart("validation_round=");
    print_uart_dec32(round + VIT_BOARD_VALIDATION_FIRST_ROUND);
    print_uart(" checked=");
    print_uart_dec32(slot->checked);
    print_uart(" errors=");
    print_uart_dec32(slot->errors);
    print_uart(" guard_errors=");
    print_uart_dec32(slot->guard_errors);
    print_uart(" input_errors=");
    print_uart_dec32(slot->input_errors);
    print_uart(" output_fnv64=0x");
    print_uart_hex64(slot->output_hash);
    print_uart("\r\n");
  }
#endif
  nh_e6_print_tag_prefix();
  print_uart("kernel_returned=");
  print_uart_dec32(returned);
  print_uart(" validation_executed=");
  print_uart_dec32(validation_executed);
  print_uart(" checked=");
  print_uart_dec32(validation_state.checked);
  print_uart(" errors=");
  print_uart_dec32(validation_state.errors);
  print_uart(" max_abs_error_f32_bits=0x");
  print_uart_hex32(validation_state.maximum_bits);
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
  } else {
    int ok = done == NH_E6_DONE_PASS && validation_executed &&
             returned == VIT_BOARD_EXPECTED_INVOCATIONS &&
             validation_state.checked == VIT_BOARD_EXPECTED_CHECKED &&
             !validation_state.errors;
    print_uart(
        ok ? "@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
           : "@@FPGA_RESULT "
             "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
  }
}
